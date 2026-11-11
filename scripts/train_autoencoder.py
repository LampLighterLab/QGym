"""Train legged-fld's ConvAutoencoder on a kine2go retargeted Go2 motion.

    uv run --frozen --extra fld scripts/train_autoencoder.py path/to/dog_walk00.npy
        [--num_steps 5000] [--latent_channels 4]
        [--wandb_entity E] [--wandb_project P] [--disable_wandb]

Input: a kine2go-pipeline .npy, (T, 61) at 60 Hz (see Go2Env.dump_state in
nomagiclab/kine2go-pipeline). Only the 12 joint angles are used.

Logs to W&B like scripts/train.py: entity/project from user/wandb_config.json,
overridden by --wandb_entity/--wandb_project; W&B is off if either is missing.
Always saves the loss curve, a reconstruction/latent plot, and the model to
logs/autoencoder/.

W&B keys:
    train/mse, val/mse          z-scored MSE per step (val every VAL_EVERY)
    net/grad_norm, net/param_norm
                                global L2 norms, gradient taken before the step
    parameters/*, gradients/*   histograms from wandb.watch every log_freq steps
    results/joint_rmse          table + bar charts, per-joint eval-mode RMSE in
                                mrad for training and held-out windows
    results/latent_stats        table, per-latent-channel mean/std over windows
    results/recon_window0, results/latent_window0
                                interactive line plots of window 0 (rad)
    results                     the matplotlib PNG
    artifact "<run_name>"       the saved .pt checkpoint (type "model")
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from legged_fld.autoencoder import ConvAutoencoder
from legged_fld.data import denormalize, fit_normalizer, make_windows, normalize

from gym import GYM_ROOT_DIR
from gym.utils.logging_and_saving import wandb_singleton

OUT_DIR = os.path.join(GYM_ROOT_DIR, "logs", "autoencoder")

FOOT_NAMES = ["FL", "FR", "RL", "RR"]
JOINT_NAMES = [f"{foot}_{j}" for foot in FOOT_NAMES for j in ("hip", "thigh", "calf")]
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


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init_wandb(args, run_config):
    wandb_helper = wandb_singleton.WandbSingleton()
    wandb_helper.set_wandb_values(args)
    motion_name = os.path.splitext(os.path.basename(args.motion))[0]
    wandb.init(
        project=wandb_helper.get_project_name(),
        entity=wandb_helper.get_entity_name(),
        dir=os.path.join(GYM_ROOT_DIR, "logs"),
        name=(
            f"conv_ae_L{args.latent_channels}_train{args.train_fraction:g}"
            f"_{motion_name}"
        ),
        config=run_config,
        mode=None if wandb_helper.is_wandb_enabled() else "disabled",
    )


def load_kine2go_joints(path):
    """(T, 12) joint angles in rad, QGym leg-major order."""
    state = np.load(path, allow_pickle=False)
    if state.ndim != 2 or state.shape[1] != 61:
        raise ValueError(f"expected a (T, 61) kine2go state array, got {state.shape}")
    return torch.tensor(state[:, KINE2GO_JOINT_COLS], dtype=torch.float32)


def split_trajectory(trajectory, train_fraction):
    """Split in time: (train, held_out or None). No frame is in both."""
    if train_fraction >= 1.0:
        return trajectory, None
    num_train = round(train_fraction * len(trajectory))
    return trajectory[:num_train], trajectory[num_train:]


def evaluate(ae, x):
    """Eval-mode MSE (BatchNorm running stats), then back to training mode."""
    ae.eval()
    with torch.no_grad():
        mse = ae.loss(x).item()
    ae.train()
    return mse


def reconstruct(ae, x):
    """Eval-mode (x_hat, z), then back to training mode."""
    ae.eval()
    with torch.no_grad():
        x_hat, z = ae(x)
    ae.train()
    return x_hat, z


def global_norm(tensors):
    """L2 norm of all entries of all tensors, as if concatenated."""
    return torch.sqrt(sum(t.pow(2).sum() for t in tensors)).item()


def joint_rmse_mrad(x, x_hat, std):
    """(12,) RMSE per joint over windows and time, in mrad.

    x, x_hat are z-scored (B, 12, H); the mean cancels in the difference, so
    the physical error is the z-scored error times std.
    """
    err_rad = (x_hat - x) * std[:, None]
    return 1e3 * err_rad.pow(2).mean(dim=(0, 2)).sqrt()


def train(ae, x, num_steps, x_val=None):
    """Full-batch Adam. Return per-step train loss and (step, held-out loss)."""
    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)
    losses, val_history = [], []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = ae.loss(x)
        loss.backward()
        metrics = {
            "train/mse": loss.item(),
            "net/grad_norm": global_norm(p.grad for p in ae.parameters()),
            "net/param_norm": global_norm(p.detach() for p in ae.parameters()),
        }
        optimizer.step()
        losses.append(metrics["train/mse"])
        if x_val is not None and (step % VAL_EVERY == 0 or step == num_steps - 1):
            val_history.append((step, evaluate(ae, x_val)))
            metrics["val/mse"] = val_history[-1][1]
        # commit now, so wandb.watch's histograms from the next step's
        # forward/backward land on the next step's row
        wandb.log(metrics, step=step, commit=True)
        if step % LOG_EVERY == 0 or step == num_steps - 1:
            val = f"  held-out {val_history[-1][1]:.3e}" if val_history else ""
            print(f"step {step:5d}  mse {losses[-1]:.3e}{val}")
    return losses, val_history


def report_convergence(losses):
    """Mean loss over the last 10% of steps vs the 10% before it."""
    n = max(len(losses) // 10, 1)
    last, prev = np.mean(losses[-n:]), np.mean(losses[-2 * n : -n])
    improvement = (prev - last) / prev
    print(f"mean mse, last {n} steps: {last:.3e}  (previous {n}: {prev:.3e})")
    print(f"relative improvement over the last {n} steps: {improvement:.1%}")
    if improvement > 0.05:
        print("still improving by >5% -- try more --num_steps")
    return last, improvement


def plot_results(losses, x, x_hat, z, run_name):
    fig, (ax_loss, ax_recon, ax_latent) = plt.subplots(3, 1, figsize=(8, 10))
    ax_loss.semilogy(losses)
    ax_loss.set_title("training loss (z-scored MSE)")
    ax_loss.set_xlabel("step")

    t_s = np.arange(x.shape[-1]) * DT_S
    for name in PLOT_JOINTS:
        c = JOINT_NAMES.index(name)
        (line,) = ax_recon.plot(t_s, x[0, c], label=name)
        ax_recon.plot(t_s, x_hat[0, c], "--", color=line.get_color())
    ax_recon.set_title("window 0: data (solid) vs reconstruction (dashed)")
    ax_recon.legend()

    ax_latent.plot(t_s, z[0].T)
    ax_latent.set_title("latent channels of window 0")
    ax_latent.set_xlabel("time in window [s]")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{run_name}.png"))
    return fig


def joint_rmse_charts(x, x_hat, x_val, x_val_hat, std):
    """Per-joint RMSE table and one bar chart per column, as a wandb.log dict."""
    columns = ["joint", "train_rmse_mrad"]
    rmse_train = joint_rmse_mrad(x, x_hat, std).tolist()
    if x_val is None:
        rows = [[name, r] for name, r in zip(JOINT_NAMES, rmse_train)]
    else:
        columns.append("held_out_rmse_mrad")
        rmse_val = joint_rmse_mrad(x_val, x_val_hat, std).tolist()
        rows = [list(r) for r in zip(JOINT_NAMES, rmse_train, rmse_val)]
    print("per-joint RMSE [mrad]: " + ", ".join(columns[1:]))
    for row in rows:
        print(f"  {row[0]:9s} " + "  ".join(f"{v:7.2f}" for v in row[1:]))
    table = wandb.Table(columns=columns, data=rows)
    charts = {"results/joint_rmse": table}
    for column in columns[1:]:
        charts[f"results/{column}"] = wandb.plot.bar(
            table, "joint", column, title=f"{column} (eval mode)"
        )
    return charts


def latent_stats_table(z):
    """Mean and std of each latent channel over all windows and time.

    Also written to the run summary as latent/mean_c<i>, latent/std_c<i>.
    """
    rows = []
    for c in range(z.shape[1]):
        mean, std = z[:, c].mean().item(), z[:, c].std().item()
        rows.append([c, mean, std])
        wandb.run.summary[f"latent/mean_c{c}"] = mean
        wandb.run.summary[f"latent/std_c{c}"] = std
    return wandb.Table(columns=["channel", "mean", "std"], data=rows)


def window0_line_plots(x, x_hat, z, mean, std):
    """Interactive versions of the PNG's reconstruction and latent panels."""
    t_s = (np.arange(x.shape[-1]) * DT_S).tolist()
    q = denormalize(x, mean, std)[0]
    q_hat = denormalize(x_hat, mean, std)[0]
    ys, keys = [], []
    for name in PLOT_JOINTS:
        c = JOINT_NAMES.index(name)
        ys += [q[c].tolist(), q_hat[c].tolist()]
        keys += [f"{name} data", f"{name} recon"]
    recon = wandb.plot.line_series(
        xs=t_s,
        ys=ys,
        keys=keys,
        title="window 0: data vs reconstruction [rad]",
        xname="time in window [s]",
    )
    latent = wandb.plot.line_series(
        xs=t_s,
        ys=z[0].tolist(),
        keys=[f"z{c}" for c in range(z.shape[1])],
        title="latent channels of window 0",
        xname="time in window [s]",
    )
    return {"results/recon_window0": recon, "results/latent_window0": latent}


def main():
    parser = argparse.ArgumentParser(description="Train ConvAutoencoder on kine2go")
    parser.add_argument("motion", help="kine2go .npy file, (T, 61) at 60 Hz")
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="train on this leading fraction of the clip, hold out the rest",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()
    run_name = f"conv_ae_L{args.latent_channels}_train{args.train_fraction:g}"

    trajectory = load_kine2go_joints(args.motion)
    train_traj, val_traj = split_trajectory(trajectory, args.train_fraction)
    # normalize with training statistics only, so nothing leaks from held-out
    mean, std = fit_normalizer(train_traj)
    # a recording, not one exact cycle: no wrap-around windows
    x = make_windows(normalize(train_traj, mean, std), WINDOW_LENGTH, circular=False)
    x_val = None
    if val_traj is not None:
        x_val = make_windows(normalize(val_traj, mean, std), WINDOW_LENGTH)
        print(f"held out {len(val_traj)} frames -> windows {tuple(x_val.shape)}")
    print(f"training {len(train_traj)} frames -> windows {tuple(x.shape)}")
    print(f"training on {device}")

    init_wandb(
        args,
        {
            **vars(args),
            "dt_s": DT_S,
            "window_length": WINDOW_LENGTH,
            "lr": LR,
            "num_windows": x.shape[0],
            "device": device,
        },
    )

    x = x.to(device)
    if x_val is not None:
        x_val = x_val.to(device)
    ae = ConvAutoencoder(x.shape[1], args.latent_channels, WINDOW_LENGTH).to(device)
    wandb.watch(ae, log="all", log_freq=max(args.num_steps // NUM_HISTOGRAMS, 1))
    losses, val_history = train(ae, x, args.num_steps, x_val)
    wandb.unwatch(ae)

    final_mse = evaluate(ae, x)
    x_hat, z = reconstruct(ae, x)
    x, x_hat, z = x.cpu(), x_hat.cpu(), z.cpu()
    x_val_hat = None
    if x_val is not None:
        x_val_hat, _ = reconstruct(ae, x_val)
        x_val, x_val_hat = x_val.cpu(), x_val_hat.cpu()
    print(f"final mse (eval mode) {final_mse:.3e}")
    if val_history:
        print(f"final held-out mse   {val_history[-1][1]:.3e}")
        wandb.run.summary["final_val_mse"] = val_history[-1][1]
    last_mse, improvement = report_convergence(losses)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plot_results(losses, x, x_hat, z, run_name)
    wandb.log(
        {
            "results": wandb.Image(fig),
            "results/latent_stats": latent_stats_table(z),
            **joint_rmse_charts(x, x_hat, x_val, x_val_hat, std),
            **window0_line_plots(x, x_hat, z, mean, std),
        }
    )
    wandb.run.summary["final_mse_eval"] = final_mse
    wandb.run.summary["mse_last_10pct"] = last_mse
    wandb.run.summary["improvement_last_10pct"] = improvement

    checkpoint_path = os.path.join(OUT_DIR, f"{run_name}.pt")
    torch.save(
        {
            "model": ae.state_dict(),
            "mean": mean,
            "std": std,
            "args": vars(args),
            "dt_s": DT_S,
            "window_length": WINDOW_LENGTH,
            "joint_names": JOINT_NAMES,
            "train_losses": losses,
            "val_history": val_history,
        },
        checkpoint_path,
    )
    print(f"saved to {OUT_DIR}")
    if not wandb.run.disabled:
        artifact = wandb.Artifact(run_name, type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
