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
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from legged_fld.autoencoder import ConvAutoencoder
from legged_fld.data import fit_normalizer, make_windows, normalize

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
        name=f"conv_ae_L{args.latent_channels}_{motion_name}",
        config=run_config,
        mode=None if wandb_helper.is_wandb_enabled() else "disabled",
    )


def load_kine2go_joints(path):
    """(T, 12) joint angles in rad, QGym leg-major order."""
    state = np.load(path, allow_pickle=False)
    if state.ndim != 2 or state.shape[1] != 61:
        raise ValueError(f"expected a (T, 61) kine2go state array, got {state.shape}")
    return torch.tensor(state[:, KINE2GO_JOINT_COLS], dtype=torch.float32)


def train(ae, x, num_steps):
    """Full-batch Adam; log mse to W&B every step. Return the per-step loss."""
    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)
    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = ae.loss(x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        wandb.log({"train/mse": losses[-1]}, step=step)
        if step % LOG_EVERY == 0 or step == num_steps - 1:
            print(f"step {step:5d}  mse {losses[-1]:.3e}")
    return losses


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


def plot_results(losses, x, x_hat, z):
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
    fig.savefig(os.path.join(OUT_DIR, "conv_autoencoder.png"))
    return fig


def main():
    parser = argparse.ArgumentParser(description="Train ConvAutoencoder on kine2go")
    parser.add_argument("motion", help="kine2go .npy file, (T, 61) at 60 Hz")
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()

    trajectory = load_kine2go_joints(args.motion)
    mean, std = fit_normalizer(trajectory)
    # a recording, not one exact cycle: no wrap-around windows
    x = make_windows(normalize(trajectory, mean, std), WINDOW_LENGTH, circular=False)
    print(f"trajectory {tuple(trajectory.shape)} -> windows {tuple(x.shape)}")
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
    ae = ConvAutoencoder(x.shape[1], args.latent_channels, WINDOW_LENGTH).to(device)
    losses = train(ae, x, args.num_steps)

    ae.eval()
    with torch.no_grad():
        final_mse = ae.loss(x).item()
        x_hat, z = ae(x)
    print(f"final mse (eval mode) {final_mse:.3e}")
    last_mse, improvement = report_convergence(losses)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plot_results(losses, x.cpu(), x_hat.cpu(), z.cpu())
    wandb.log({"results": wandb.Image(fig)})
    wandb.run.summary["final_mse_eval"] = final_mse
    wandb.run.summary["mse_last_10pct"] = last_mse
    wandb.run.summary["improvement_last_10pct"] = improvement

    torch.save(
        {
            "model": ae.state_dict(),
            "mean": mean,
            "std": std,
            "args": vars(args),
            "dt_s": DT_S,
            "window_length": WINDOW_LENGTH,
            "joint_names": JOINT_NAMES,
        },
        os.path.join(OUT_DIR, "conv_autoencoder.pt"),
    )
    print(f"saved to {OUT_DIR}")
    wandb.finish()


if __name__ == "__main__":
    main()
