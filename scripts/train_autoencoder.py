"""Train the baseline Autoencoder on one periodic reference trajectory.

Usage:
    uv run --frozen scripts/train_autoencoder.py [--latent_dim 2] [--window_length 51]
        [--wandb_project P] [--wandb_entity E] [--disable_wandb]

Every circular shift of the trajectory becomes one window, so the windows are
the same motion seen at different phases. Logs to W&B (entity/project from
user/wandb_config.json, as in scripts/train.py) and saves the trained model to
logs/autoencoder/.
"""

import argparse
import os

import numpy as np
import torch
import wandb

from gym import GYM_ROOT_DIR
from gym.utils.logging_and_saving import wandb_singleton
from learning.modules import Autoencoder

TRAJECTORY = os.path.join(
    GYM_ROOT_DIR,
    "resources",
    "robots",
    "mini_cheetah",
    "trajectories",
    "single_leg.csv",
)


def load_trajectory(path):
    """Return a (num_steps, num_channels) tensor and the channel names."""
    with open(path) as f:
        names = [name.strip() for name in f.readline().split(",")]
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return torch.tensor(data, dtype=torch.float32), names


def make_windows(trajectory, window_length):
    """Circular sliding windows: (num_steps, C) -> (num_steps, C, window_length)."""
    wrapped = torch.cat([trajectory, trajectory[: window_length - 1]])
    return wrapped.unfold(0, window_length, 1)


def pca_mse(x, latent_dim):
    """Best MSE any linear autoencoder with this latent_dim can reach."""
    X = x.flatten(start_dim=1)
    X = X - X.mean(dim=0)
    singular_vals = torch.linalg.svdvals(X)
    return (singular_vals[latent_dim:].pow(2).sum() / X.numel()).item()


def train(ae, x, num_steps, lr):
    """Full-batch Adam on ae.loss(x); log {"mse": ...} to W&B every step.

    Return the final MSE as a float.
    """
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = ae.loss(x)
        loss.backward()
        optimizer.step()
        wandb.log({"mse": loss.item()}, step=step)
        if step % (num_steps // 10) == 0:
            print(f"step {step:5d} mse {loss.item():.3e}")
    return ae.loss(x).item()


def log_results(x, x_hat, z, names):
    """Reconstruction of window 0 and the latent of every window, as W&B charts."""
    t = list(range(x.shape[-1]))
    wandb.log(
        {
            "reconstruction/window_0": wandb.plot.line_series(
                xs=t,
                ys=[*x[0].tolist(), *x_hat[0].tolist()],
                keys=[*names, *(f"{name} (recon)" for name in names)],
                title="window 0: data vs reconstruction",
                xname="time step",
            ),
            "latent/vs_phase": wandb.plot.line_series(
                xs=list(range(len(z))),
                ys=z.T.tolist(),
                keys=[f"z[{d}]" for d in range(z.shape[1])],
                title="latent of each window vs window start (phase)",
                xname="window start",
            ),
        }
    )
    if z.shape[1] >= 2:
        table = wandb.Table(
            columns=["phase", "z0", "z1"],
            data=[[i, *z[i, :2].tolist()] for i in range(len(z))],
        )
        wandb.log(
            {
                "latent/loop": wandb.plot.scatter(
                    table, "z0", "z1", title="latent loop z0 vs z1"
                )
            }
        )


def main():
    parser = argparse.ArgumentParser(description="Train the baseline Autoencoder")
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--window_length", type=int, default=51)
    parser.add_argument("--num_steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    wandb_helper = wandb_singleton.WandbSingleton()
    wandb_helper.set_wandb_values(args)
    wandb.init(
        project=wandb_helper.get_project_name(),
        entity=wandb_helper.get_entity_name(),
        dir=os.path.join(GYM_ROOT_DIR, "logs"),
        name=f"autoencoder_latent{args.latent_dim}",
        config=vars(args),
        mode=None if wandb_helper.is_wandb_enabled() else "disabled",
    )

    trajectory, names = load_trajectory(TRAJECTORY)
    mean, std = trajectory.mean(dim=0), trajectory.std(dim=0)
    trajectory = (trajectory - mean) / std  # equal weight per joint in the MSE
    x = make_windows(trajectory, args.window_length)
    print(f"trajectory {tuple(trajectory.shape)} -> windows {tuple(x.shape)}")

    ae = Autoencoder(x.shape[1], args.window_length, args.latent_dim)
    ae_mse = train(ae, x, args.num_steps, args.lr)
    linear_mse = pca_mse(x, args.latent_dim)
    wandb.run.summary["final_mse"] = ae_mse
    wandb.run.summary["pca_limit_mse"] = linear_mse
    print(f"\nautoencoder mse     {ae_mse:.3e}")
    print(f"linear (PCA) limit  {linear_mse:.3e}")
    print("nonlinear beats linear:", ae_mse < linear_mse)

    with torch.no_grad():
        x_hat, z = ae(x)
    log_results(x, x_hat, z, names)

    out_dir = os.path.join(GYM_ROOT_DIR, "logs", "autoencoder")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {"model": ae.state_dict(), "mean": mean, "std": std, "args": vars(args)},
        os.path.join(out_dir, "autoencoder.pt"),
    )
    print(f"saved to {out_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
