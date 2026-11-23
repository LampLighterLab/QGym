"""Train legged-fld's ConvAutoencoder on kine2go retargeted Go2 motions.

    uv run --frozen --extra fld scripts/train_autoencoder.py CLIP.npy [CLIP.npy ...]
        [--val_motions HELD_OUT.npy ...] [--train_fraction 0.8] [--kernel_size 13]
        [--num_steps 5000] [--latent_channels 4] [--seed 0]
        [--wandb_entity E] [--wandb_project P] [--disable_wandb]

Input: kine2go-pipeline .npy files, (T, 61) at 60 Hz (Go2Env.dump_state in
nomagiclab/kine2go-pipeline); only the 12 joint angles are used. Held-out data
is either whole clips (--val_motions) or the end of every clip
(--train_fraction). The normalizer sees training data only, and no window
spans two clips.

W&B entity/project come from user/wandb_config_autoencoder.json (not the
locomotion user/wandb_config.json), overridden by --wandb_entity/--wandb_project;
with neither, W&B is off, with a warning. Keys:
    train/mse, val/mse, val/<clip>  z-scored MSE; val every VAL_EVERY steps
    net/grad_norm, net/param_norm   global L2 norms (gradient before the step)
    parameters/*, gradients/*       wandb.watch histograms
    results/joint_rmse              per-joint eval-mode RMSE [mrad], table + bars
    results/latent_stats            per-latent-channel mean/std
    results/recon_window0, results/latent_window0   window 0, interactive
    results                         the PNG; the .pt is logged as a model artifact
Local outputs: logs/autoencoder/<run_name>.{png,pt}.
"""

import argparse
import os
import time
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from legged_fld.autoencoder import ConvAutoencoder
from legged_fld.data import (
    denormalize,
    fit_normalizer,
    make_window_sequences,
    make_windows,
    normalize,
)
from legged_fld.fld import FourierLatentDynamics
from legged_fld.pae import PeriodicAutoencoder

from gym import GYM_ROOT_DIR
from gym.utils.logging_and_saving import wandb_singleton

OUT_DIR = os.path.join(GYM_ROOT_DIR, "logs", "autoencoder")
WANDB_CONFIG = "wandb_config_autoencoder.json"  # under user/; not the locomotion one

JOINT_NAMES = [
    f"{leg}_{j}" for leg in ("FL", "FR", "RL", "RR") for j in ("hip", "thigh", "calf")
]
# kine2go saves Genesis dof order: cols 6:18 grouped by joint type (4 hips, 4
# thighs, 4 calves; FL, FR, RL, RR within each). Reorder to QGym's leg-major
# order above. Verified against foot motion for thighs/calves.
KINE2GO_JOINT_COLS = [6, 10, 14, 7, 11, 15, 8, 12, 16, 9, 13, 17]
PLOT_JOINTS = ["FL_thigh", "FL_calf", "RR_thigh", "RR_calf"]

DT_S = 1 / 60  # kine2go / AI4Animation capture rate
WINDOW_LENGTH = 61  # odd; 61 samples at 60 Hz = 1.0 s, ~1.6 strides
LR = 1e-3
LOG_EVERY = 250
VAL_EVERY = 50
NUM_HISTOGRAMS = 50  # wandb.watch logs weights/gradients this many times per run


# ------------------------------------------------------------------ data


def load_kine2go_joints(path):
    """(T, 12) joint angles in rad, QGym leg-major order."""
    state = np.load(path, allow_pickle=False)
    if state.ndim != 2 or state.shape[1] != 61:
        raise ValueError(f"expected a (T, 61) kine2go state array, got {state.shape}")
    return torch.tensor(state[:, KINE2GO_JOINT_COLS], dtype=torch.float32)


def clip_name(path):
    """path/to/dog_walk03_joint_pos.npy -> dog_walk03_joint_pos"""
    return os.path.splitext(os.path.basename(path))[0]


def split_trajectory(trajectory, train_fraction):
    """Split in time: (train, held_out or None). No frame is in both."""
    if train_fraction >= 1.0:
        return trajectory, None
    num_train = round(train_fraction * len(trajectory))
    return trajectory[:num_train], trajectory[num_train:]


def load_split(motions, val_motions, train_fraction, window_length):
    """(list of training trajectories, {clip name: held-out trajectory}).

    With val_motions, whole clips are held out. Otherwise each clip is split in
    time by train_fraction; a tail shorter than one window is not validated.
    """
    if val_motions:
        train = [load_kine2go_joints(p) for p in motions]
        return train, {clip_name(p): load_kine2go_joints(p) for p in val_motions}
    train, val = [], {}
    for path in motions:
        head, tail = split_trajectory(load_kine2go_joints(path), train_fraction)
        train.append(head)
        if tail is not None and len(tail) >= window_length:
            val[clip_name(path)] = tail
        elif tail is not None:
            print(f"WARNING: {clip_name(path)} tail has {len(tail)} frames, fewer")
            print(f"  than one window ({window_length}); not validated")
    return train, val


def windows_from_clips(trajectories, mean, std, window_length):
    """Normalize and window each clip on its own, then stack (no cross-clip windows)."""
    windows = []
    for trajectory in trajectories:
        if len(trajectory) < window_length:
            raise ValueError(
                f"clip has {len(trajectory)} frames, fewer than one window "
                f"({window_length})"
            )
        windows.append(make_windows(normalize(trajectory, mean, std), window_length))
    return torch.cat(windows)

def sequences_from_clips(trajectories, mean, std, window_length, horizon):
    """
    Normalize each clip, cut (B, N+1, M, H) window sequences, stack.

    Like windows_from_clips, no sequences crosses clips: sequence b, step i is
    window b+i of the same clip, so a clip needs H+N frames.
    """
    sequences = []
    for trajectory in trajectories:
        normed = normalize(trajectory, mean, std)
        sequences.append(make_window_sequences(normed, window_length, horizon))
    return torch.cat(sequences)


# ------------------------------------------------------------------ model


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(model, num_channels, latent_channels, kernel_size):
    if model == "ae":
        return ConvAutoencoder(
            num_channels, latent_channels, WINDOW_LENGTH, kernel_size=kernel_size
        )
    cls = FourierLatentDynamics if model == "fld" else PeriodicAutoencoder
    return cls(
        num_channels, latent_channels, WINDOW_LENGTH, DT_S, kernel_size=kernel_size
    )

def forward_with_latent(model, x):
    """
    (x_hat, latent), (B, L, H)
    """
    x_hat, latent = model(x)
    if isinstance(model, PeriodicAutoencoder):
        latent = model.reconstruct_latent(latent)
    return x_hat, latent

def reconstruct(model, x):
    """Eval-mode (x_hat, z), using BatchNorm running stats; back to train mode."""
    model.eval()
    with torch.no_grad():
        x_hat, latent = forward_with_latent(model, x)
    model.train()
    return x_hat, latent


def mse(ae, x):
    """Eval-mode reconstruction MSE (z-scored)."""
    x_hat, _ = reconstruct(ae, x)
    return (x_hat - x).pow(2).mean().item()

def fld_loss(model, seq, decay, chunk):
    """
    Eval mode L_FLD over all of seq, chunk sequences at a time.
    """
    model.eval()
    with torch.no_grad():
        total = sum(model.loss(s, decay) * len(s) for s in seq.split(chunk))
    model.train()
    return (total / len(seq)).item()


def global_norm(tensors):
    """L2 norm of all entries of all tensors, as if concatenated."""
    return torch.sqrt(sum(t.pow(2).sum() for t in tensors)).item()


def joint_rmse_mrad(x, x_hat, std):
    """(12,) RMSE per joint over windows and time, in mrad.

    x, x_hat are z-scored (B, 12, H); the mean cancels in the difference, so
    the physical error is the z-scored error times std.
    """
    return 1e3 * ((x_hat - x) * std[:, None]).pow(2).mean(dim=(0, 2)).sqrt()


def train(ae, x, x_val, num_steps, train_loss=None, val_fld=None):
    """Adam on x with cosine decay; x_val is {clip: windows}, possibly empty.

    train_loss: () -> loss tensor; default the full-batch reconstruction loss
    ae.loss(x), logged as train/mse. FLD passes a minibatch L_FLD sampler,
    logged as train/fld. val_fld: () -> float, the held-out L_FLD (val/fld).

    Returns per-step train losses, (step, pooled held-out MSE), and the last
    step's metrics (held-out values there are the final ones).
    """
    train_key = "train/mse" if train_loss is None else "train/fld"
    if train_loss is None:
        train_loss = lambda: ae.loss(x) # noqa: E731
    x_val_all = torch.cat(list(x_val.values())) if x_val else None
    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)
    # add cosine decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * s / num_steps))
    )
    losses, val_history = [], []
    start = time.perf_counter()
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = train_loss()
        loss.backward()
        metrics = {
            train_key: loss.item(),
            "net/grad_norm": global_norm(p.grad for p in ae.parameters()),
            "net/param_norm": global_norm(p.detach() for p in ae.parameters()),
            "opt/lr": scheduler.get_last_lr()[0]
        }
        optimizer.step()
        scheduler.step()
        losses.append(metrics[train_key])
        # count completed steps, so prints land on 250, 500, ... like train.py;
        # step 1 is an early sanity check, the last step is always included
        done = step + 1
        first_or_last = done == 1 or done == num_steps
        if x_val and (first_or_last or done % VAL_EVERY == 0):
            metrics |= {f"val/{name}": mse(ae, xv) for name, xv in x_val.items()}
            metrics["val/mse"] = mse(ae, x_val_all)
            if val_fld is not None:
                metrics["val/fld"] = val_fld()
            val_history.append((step, metrics["val/mse"]))
        # commit now, so wandb.watch's histograms from the next step's
        # forward/backward land on the next step's row
        wandb.log(metrics, step=step, commit=True)
        if first_or_last or done % LOG_EVERY == 0:
            print_progress(step, num_steps, metrics, time.perf_counter() - start)
    return losses, val_history, metrics


# ------------------------------------------------------------------ terminal

WIDTH, PAD = 80, 35  # same layout as learning/utils/logger/Logger.print_to_terminal


def separator(subtitle="", marker="-"):
    if subtitle:
        subtitle += " "
    dashes = marker * ((WIDTH - len(subtitle)) // 2)
    return f"\n{dashes} {subtitle}{dashes}\n"


def entry(key, value="", fmt=".3e", append=""):
    value = format(value, fmt) if isinstance(value, float) else value
    return f"{key:>{PAD}}: {value} {append}".rstrip() + "\n"


def hms(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def print_progress(step, num_steps, metrics, elapsed_s):
    """train.py-style block. Held-out values are from this step (LOG_EVERY is a
    multiple of VAL_EVERY, and the last step always validates)."""
    done = step + 1
    s = separator(f"Step {done}/{num_steps}", "#")
    s += separator("Reconstruction (z-scored MSE)")
    if "train/mse" in metrics:
        s += entry("train", metrics["train/mse"])
    if "val/mse" in metrics:
        s += entry("held-out (pooled)", metrics["val/mse"])
        for key, value in metrics.items():
            if key.startswith("val/") and key not in ("val/mse", "val/fld"):
                s += entry(f"held-out {key.removeprefix('val/')}", value)
    if "train/fld" in metrics:
        s += separator("FLD loss (eq. 5: MSE summed over steps)")
        s += entry("train (minibatch)", metrics["train/fld"])
        if "val/fld" in metrics:
            s += entry("held-out (pooled)", metrics["val/fld"])
    s += separator("Network")
    s += entry("grad norm", metrics["net/grad_norm"])
    s += entry("param norm", metrics["net/param_norm"])
    s += entry("learning rate", metrics["opt/lr"])
    s += separator("Timings")
    s += entry("steps/s", done / elapsed_s, ".1f")
    s += entry("total time", elapsed_s, ".2f", "s")
    s += entry("ETA", hms(elapsed_s / done * (num_steps - done)))
    print(s, flush=True)


def print_wandb_status(run, helper):
    """What W&B is actually doing for this run, so a sweep log shows it up front."""
    s = separator("W&B", "#")
    mode = "disabled" if run.disabled else ("offline" if run.offline else "online")
    s += entry("mode", mode)
    if run.disabled:
        s += entry("note", "nothing is logged or uploaded")
    else:
        s += entry("entity / project", f"{run.entity} / {run.project}")
        s += entry("run name / id", f"{run.name} / {run.id}")
        s += entry("url", run.url or "(none; offline, `wandb sync` later)")
        s += entry("local dir", run.dir)
        s += entry("config file", os.path.join(GYM_ROOT_DIR, "user", WANDB_CONFIG))
        if os.environ.get("WANDB_MODE"):
            s += entry("WANDB_MODE", os.environ["WANDB_MODE"])
        if helper.get_project_name() != run.project:
            s += entry("WARNING", f"requested project {helper.get_project_name()}")
    print(s, flush=True)


# ------------------------------------------------------------------ logging


def init_wandb(args, run_config, run_name):
    helper = wandb_singleton.WandbSingleton()
    helper.set_wandb_values(args, config_name=WANDB_CONFIG)
    if not helper.is_wandb_enabled():
        config_path = os.path.join(GYM_ROOT_DIR, "user", WANDB_CONFIG)
        reason = (
            "--disable_wandb was passed"
            if args.disable_wandb
            else f"no entity/project: {config_path} not found and "
            "--wandb_entity/--wandb_project not given"
        )
        print(f"WARNING: W&B disabled ({reason}); logging locally only.")
    run = wandb.init(
        project=helper.get_project_name(),
        entity=helper.get_entity_name(),
        dir=os.path.join(GYM_ROOT_DIR, "logs"),
        name=run_name,
        config=run_config,
        mode=None if helper.is_wandb_enabled() else "disabled",
    )
    print_wandb_status(run, helper)
    return run


def plot_results(losses, x, x_hat, z, run_name):
    """Loss curve, window-0 reconstruction and latent; saved as the run's PNG."""
    fig, (ax_loss, ax_recon, ax_latent) = plt.subplots(3, 1, figsize=(8, 10))
    ax_loss.semilogy(losses)
    ax_loss.set(title="training loss (z-scored MSE)", xlabel="step")
    t_s = np.arange(x.shape[-1]) * DT_S
    for name in PLOT_JOINTS:
        c = JOINT_NAMES.index(name)
        (line,) = ax_recon.plot(t_s, x[0, c], label=name)
        ax_recon.plot(t_s, x_hat[0, c], "--", color=line.get_color())
    ax_recon.set_title("window 0: data (solid) vs reconstruction (dashed)")
    ax_recon.legend()
    ax_latent.plot(t_s, z[0].T)
    ax_latent.set(title="latent channels of window 0", xlabel="time in window [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{run_name}.png"))
    return fig


def joint_rmse_charts(rmse):
    """{column: (12,) RMSE in mrad} -> printed table, W&B table, one bar chart each."""
    rows = [
        [joint, *(values[j].item() for values in rmse.values())]
        for j, joint in enumerate(JOINT_NAMES)
    ]
    print("per-joint RMSE [mrad]: " + ", ".join(rmse))
    for row in rows:
        print(f"  {row[0]:9s} " + "  ".join(f"{v:7.2f}" for v in row[1:]))
    table = wandb.Table(columns=["joint", *rmse], data=rows)
    bars = {
        f"results/{c}": wandb.plot.bar(table, "joint", c, title=f"{c} (eval mode)")
        for c in rmse
    }
    return {"results/joint_rmse": table, **bars}


def latent_stats_table(z, run):
    """Mean/std of each latent channel over windows and time (also in the summary)."""
    rows = [[c, z[:, c].mean().item(), z[:, c].std().item()] for c in range(z.shape[1])]
    for c, mean, std in rows:
        run.summary[f"latent/mean_c{c}"] = mean
        run.summary[f"latent/std_c{c}"] = std
    return wandb.Table(columns=["channel", "mean", "std"], data=rows)


def window0_line_plots(x, x_hat, z, mean, std):
    """Interactive versions of the PNG's reconstruction (rad) and latent panels."""
    t_s = (np.arange(x.shape[-1]) * DT_S).tolist()
    q, q_hat = denormalize(x, mean, std)[0], denormalize(x_hat, mean, std)[0]
    ys, keys = [], []
    for name in PLOT_JOINTS:
        c = JOINT_NAMES.index(name)
        ys += [q[c].tolist(), q_hat[c].tolist()]
        keys += [f"{name} data", f"{name} recon"]
    xname = "time in window [s]"
    return {
        "results/recon_window0": wandb.plot.line_series(
            xs=t_s,
            ys=ys,
            keys=keys,
            title="window 0: data vs reconstruction [rad]",
            xname=xname,
        ),
        "results/latent_window0": wandb.plot.line_series(
            xs=t_s,
            ys=z[0].tolist(),
            keys=[f"z{c}" for c in range(z.shape[1])],
            title="latent channels of window 0",
            xname=xname,
        ),
    }


# ------------------------------------------------------------------ main


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvAutoencoder on kine2go")
    parser.add_argument("motions", nargs="+", help="training .npy clips")
    parser.add_argument("--model", choices=["ae", "pae", "fld"], default="ae")
    parser.add_argument(
        "--horizon", type=int, default=60, help="fld: N predicted windows (60 = 1 s)"
    )
    parser.add_argument(
        "--decay", type=float, default=1.0, help="fld: alpha decay rate on loss"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="fld: sequences per step"
    )
    parser.add_argument(
        "--val_motions",
        nargs="*",
        default=[],
        help="held-out .npy clips (whole recordings); excludes --train_fraction",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="without --val_motions: train on this leading fraction of each clip",
    )
    parser.add_argument("--kernel_size", type=int, default=61)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    args = parser.parse_args()
    if args.val_motions and args.train_fraction < 1.0:
        parser.error("use either --val_motions or --train_fraction, not both")
    shared = {clip_name(p) for p in args.motions} & set(
        map(clip_name, args.val_motions)
    )
    if shared:
        parser.error(f"clips in both training and held-out sets: {sorted(shared)}")
    return args


def make_run_name(args):
    """One name for the W&B run, the local .png/.pt and the model artifact."""
    if args.val_motions:
        short = (
            clip_name(p).removeprefix("dog_").removesuffix("_joint_pos")
            for p in args.val_motions
        )
        split = "val-" + "+".join(short)
    else:
        split = f"train{args.train_fraction:g}"
    name = (
        f"conv_{args.model}_L{args.latent_channels}_{split}"
        f"_K{args.kernel_size}_s{args.seed}"
    )
    return name + (f"_N{args.horizon}" if args.model == "fld" else "")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = pick_device()
    run_name = make_run_name(args)

    train_trajs, val_trajs = load_split(
        args.motions, args.val_motions, args.train_fraction, WINDOW_LENGTH
    )
    mean, std = fit_normalizer(torch.cat(train_trajs))  # training data only
    x = windows_from_clips(train_trajs, mean, std, WINDOW_LENGTH).to(device)
    x_val = {
        name: windows_from_clips([t], mean, std, WINDOW_LENGTH).to(device)
        for name, t in val_trajs.items()
    }
    print(f"training on {device}: {len(train_trajs)} clips, windows {tuple(x.shape)}")
    for name, xv in x_val.items():
        print(f"held out {name}: windows {tuple(xv.shape)}")

    train_loss = val_fld = None
    if args.model == "fld":
        seq = sequences_from_clips(
            train_trajs, mean, std, WINDOW_LENGTH, args.horizon
        ).to(device)

        # a held-out clip shorter than H+N has no sequences; it still counts in val/mse
        min_frames = WINDOW_LENGTH + args.horizon
        long_enough = [t for t in val_trajs.values() if len(t) >= min_frames]
        seq_val = (
            sequences_from_clips(
                long_enough, mean, std, WINDOW_LENGTH, args.horizon
            ).to(device)
            if long_enough
            else None
        )
        print(f"fld sequences: train {tuple(seq.shape)}, held out "
              f"{tuple(seq_val.shape) if seq_val is not None else None}")

    run_config = {
        **vars(args),
        "dt_s": DT_S,
        "window_length": WINDOW_LENGTH,
        "lr": LR,
        "num_windows": x.shape[0],
        "device": device,
    }
    run = init_wandb(args, run_config, run_name)
    ae = build_model(args.model, x.shape[1], args.latent_channels, args.kernel_size)
    ae = ae.to(device)
    watch_freq = max(args.num_steps // NUM_HISTOGRAMS, 1)
    wandb.watch(ae, log="all", log_freq=watch_freq)
    if not run.disabled:
        print(f"W&B: watching weights/gradients every {watch_freq} steps")

    if args.model == "fld":
        def train_loss():
            idx = torch.randint(len(seq), (args.batch_size,), device=device)
            return ae.loss(seq[idx], args.decay)

        if seq_val is not None:
            def val_fld():
                return fld_loss(ae, seq_val, args.decay, args.batch_size)

    losses, val_history, last = train(ae, x, x_val, args.num_steps, train_loss, val_fld)
    wandb.unwatch(ae)

    final_mse = mse(ae, x)
    run.summary["final_mse_eval"] = final_mse
    print(f"final train mse (eval mode) {final_mse:.3e}")
    x_hat, z = reconstruct(ae, x)
    x, x_hat, z = x.cpu(), x_hat.cpu(), z.cpu()
    rmse = {"train_rmse_mrad": joint_rmse_mrad(x, x_hat, std)}
    if x_val:
        run.summary["final_val_mse"] = last["val/mse"]
        for name in x_val:
            run.summary[f"final_val_mse/{name}"] = last[f"val/{name}"]
            print(f"final held-out mse {name}: {last[f'val/{name}']:.3e}")
        print(f"final held-out mse (pooled): {last['val/mse']:.3e}")
        x_val_all = torch.cat(list(x_val.values()))
        x_val_hat, _ = reconstruct(ae, x_val_all)
        rmse["held_out_rmse_mrad"] = joint_rmse_mrad(
            x_val_all.cpu(), x_val_hat.cpu(), std
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plot_results(losses, x, x_hat, z, run_name)
    wandb.log(
        {
            "results": wandb.Image(fig),
            "results/latent_stats": latent_stats_table(z, run),
            **joint_rmse_charts(rmse),
            **window0_line_plots(x, x_hat, z, mean, std),
        }
    )

    checkpoint_path = os.path.join(OUT_DIR, f"{run_name}.pt")
    torch.save(
        {
            "model": ae.state_dict(),
            "mean": mean,
            "std": std,
            "args": vars(args),
            "motions": [os.path.basename(p) for p in args.motions],
            "val_motions": [os.path.basename(p) for p in args.val_motions],
            "dt_s": DT_S,
            "window_length": WINDOW_LENGTH,
            "joint_names": JOINT_NAMES,
            "train_losses": losses,
            "val_history": val_history,
        },
        checkpoint_path,
    )
    print(f"saved {checkpoint_path}")
    if not run.disabled:
        artifact = wandb.Artifact(run_name, type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        print(f"W&B: logged model artifact {run_name} ({checkpoint_path})")
    wandb.finish()
    if run.disabled:
        print("W&B: disabled, nothing uploaded")
    else:
        print(f"W&B: finished {run.url or run.dir}")


if __name__ == "__main__":
    main()
