"""Evaluate ConvAutoencoder checkpoints from scripts/train_autoencoder.py.

    uv run --frozen --extra fld scripts/evaluate_autoencoder.py \
        --data_dir path/to/clips logs/autoencoder/*.pt

Each checkpoint's train/held-out split is rebuilt from --data_dir (it stores
clip basenames). Held-out MSE is printed next to three reference levels:
  nearest window  copy the closest training window: what pure memorization scores
  PCA             best linear autoencoder with the same latent size (L x H)
  jitter floor    error of a model perfect except for retargeting noise above
                  JITTER_CUTOFF_HZ; training error below it means fitting noise
followed by a table averaged over seeds. PNGs go to logs/autoencoder/plots/:
  <run>_loss_curves.png      train/held-out loss vs the reference levels
  <run>_per_joint_error.png  per-joint RMSE [mrad] vs jitter
  <run>_error_spectrum.png   which frequencies the model reproduces
  <run>_latent_portrait.png  latent over time (one loop per stride?)
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from legged_fld.autoencoder import ConvAutoencoder

from gym import GYM_ROOT_DIR

# reuse the loader and split the models were trained with
from train_autoencoder import load_split, windows_from_clips

PLOTS_DIR = os.path.join(GYM_ROOT_DIR, "logs", "autoencoder", "plots")
JITTER_CUTOFF_HZ = 10.0


# ------------------------------------------------------------------ signals


def high_frequency_rms(signal, dt_s, cutoff_hz):
    """Per-channel RMS of the content above cutoff_hz. signal: (T, M)."""
    spectrum = np.fft.rfft(signal - signal.mean(0), axis=0)
    spectrum[np.fft.rfftfreq(len(signal), dt_s) <= cutoff_hz] = 0
    return np.sqrt((np.fft.irfft(spectrum, n=len(signal), axis=0) ** 2).mean(0))


def power_spectrum(signal, dt_s):
    """Frequencies [Hz] and power summed over channels. signal: (T, M)."""
    power = (np.abs(np.fft.rfft(signal - signal.mean(0), axis=0)) ** 2).sum(1)
    return np.fft.rfftfreq(len(signal), dt_s), power


def center_track(windows):
    """Centre sample of every window -> (B, C): a time series over the clip."""
    return windows[:, :, windows.shape[-1] // 2].numpy()


# ------------------------------------------------------------------ baselines


def mse(a, b):
    return (a - b).pow(2).mean().item()


def pca_mse(x_fit, x_eval, num_components):
    """MSE of the best rank-k linear autoencoder fit on x_fit, applied to x_eval."""
    X_fit, X_eval = x_fit.flatten(start_dim=1), x_eval.flatten(start_dim=1)
    mean = X_fit.mean(dim=0)
    V = torch.linalg.svd(X_fit - mean, full_matrices=False)[2][:num_components].T
    return mse(mean + (X_eval - mean) @ V @ V.T, X_eval)


def nearest_window_mse(x_train, x_eval):
    """Mean over x_eval of the MSE to its closest training window."""
    d = torch.cdist(x_eval.flatten(start_dim=1), x_train.flatten(start_dim=1))
    return (d.min(dim=1).values.pow(2) / x_eval[0].numel()).mean().item()


# ------------------------------------------------------------------ runs


def load_run(path, data_dir):
    """Load a checkpoint and rebuild its train/held-out windows from data_dir."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    H, mean, std, a = ckpt["window_length"], ckpt["mean"], ckpt["std"], ckpt["args"]
    # checkpoints from before multi-clip training only have args["motion"]
    motions = ckpt.get("motions", [os.path.basename(a.get("motion", ""))])
    resolve = lambda names: [os.path.join(data_dir, n) for n in names]  # noqa: E731
    train_trajs, val_trajs = load_split(
        resolve(motions),
        resolve(ckpt.get("val_motions", [])),
        a.get("train_fraction", 1.0),
        H,
    )
    ae = ConvAutoencoder(
        len(ckpt["joint_names"]),
        a["latent_channels"],
        H,
        # checkpoints without kernel_size predate the flag: all trained with K = H
        kernel_size=a.get("kernel_size", H),
    )
    ae.load_state_dict(ckpt["model"])
    ae.eval()

    run = {
        "name": os.path.splitext(os.path.basename(path))[0],
        "ckpt": ckpt,
        "H": H,
        "num_latent": a["latent_channels"],
        "std": std.numpy(),
        # jitter from the first clip only: concatenating clips would add fake
        # high-frequency content at the joins
        "train_traj": train_trajs[0],
        "x": windows_from_clips(train_trajs, mean, std, H),
        "x_val": None,
    }
    if val_trajs:
        run["x_val"] = windows_from_clips(list(val_trajs.values()), mean, std, H)
    with torch.no_grad():
        run["x_hat"], run["z"] = ae(run["x"])
        if run["x_val"] is not None:
            run["x_hat_val"], _ = ae(run["x_val"])
    return run


def jitter_floor_z(run, dt_s):
    """MSE (z-units) of a model that is perfect except for the jitter."""
    jitter = high_frequency_rms(run["train_traj"].numpy(), dt_s, JITTER_CUTOFF_HZ)
    return np.mean((jitter / run["std"]) ** 2), jitter


# ------------------------------------------------------------------ plots


def save(fig, filename, tight=True):
    if tight:
        fig.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_loss_curves(run, levels):
    ckpt = run["ckpt"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(ckpt["train_losses"], lw=0.8, label="train (train-mode BN)")
    if ckpt["val_history"]:
        ax.semilogy(*zip(*ckpt["val_history"], strict=True), lw=1.5, label="held-out")
    styles = [("k", "--"), ("C3", ":"), ("C2", ":")]
    for (label, value), (color, ls) in zip(levels.items(), styles):
        ax.axhline(value, color=color, ls=ls, label=f"{label} ({value:.1e})")
    ax.set(xlabel="step", ylabel="z-scored MSE", title=f"{run['name']}: loss")
    ax.legend()
    save(fig, f"{run['name']}_loss_curves.png")


def plot_per_joint_error(run, jitter):
    def rmse_mrad(x, x_hat):
        return 1e3 * (x_hat - x).pow(2).mean(dim=(0, 2)).sqrt().numpy() * run["std"]

    bars = {"train RMSE": rmse_mrad(run["x"], run["x_hat"])}
    if run["x_val"] is not None:
        bars["held-out RMSE"] = rmse_mrad(run["x_val"], run["x_hat_val"])
    bars[f"jitter (>{JITTER_CUTOFF_HZ:g} Hz)"] = 1e3 * jitter

    joints = run["ckpt"]["joint_names"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    idx, width = np.arange(len(joints)), 0.8 / len(bars)
    for k, (label, values) in enumerate(bars.items()):
        ax.bar(idx + (k - (len(bars) - 1) / 2) * width, values, width, label=label)
    ax.set_xticks(idx, joints, rotation=45)
    ax.set(ylabel="mrad", title=f"{run['name']}: per-joint error vs jitter")
    ax.legend()
    save(fig, f"{run['name']}_per_joint_error.png")


def plot_error_spectrum(run, dt_s):
    fig, ax = plt.subplots(figsize=(9, 5))
    splits = [("train", run["x"], run["x_hat"])]
    if run["x_val"] is not None:
        splits.append(("held-out", run["x_val"], run["x_hat_val"]))
    for label, x, x_hat in splits:
        freqs, data = power_spectrum(center_track(x), dt_s)
        _, err = power_spectrum(center_track(x_hat - x), dt_s)
        if label == "train":
            ax.semilogy(freqs, data / data.max(), color="k", label="data (train)")
        ax.semilogy(freqs, err / data.max(), label=f"error ({label})")
    ax.axvline(JITTER_CUTOFF_HZ, color="k", ls="--", lw=0.8, label="jitter cutoff")
    ax.set(xlabel="frequency [Hz]", ylabel="power / data peak")
    ax.set_title(f"{run['name']}: what the model reproduces (error << data)")
    ax.legend()
    save(fig, f"{run['name']}_error_spectrum.png")


def plot_latent_portrait(run, dt_s):
    """Latent channel pairs at window centres; a clean gait loops once per stride."""
    z = center_track(run["z"])
    if z.shape[1] < 2:
        return
    pairs = [(i, i + 1) for i in range(0, min(z.shape[1], 4) - 1, 2)]
    fig, axes = plt.subplots(
        1, len(pairs), figsize=(5.5 * len(pairs), 5), squeeze=False
    )
    t_s = np.arange(len(z)) * dt_s
    for ax, (i, j) in zip(axes[0], pairs, strict=True):
        ax.plot(z[:, i], z[:, j], color="0.8", lw=0.6)
        points = ax.scatter(z[:, i], z[:, j], c=t_s, s=6, cmap="viridis")
        ax.set(xlabel=f"z{i}", ylabel=f"z{j}")
        fig.colorbar(points, ax=ax, label="time over windows [s]")
    fig.suptitle(f"{run['name']}: latent phase portrait")
    save(fig, f"{run['name']}_latent_portrait.png")


# ------------------------------------------------------------------ main


def print_seed_table(rows):
    """Mean ± std over seeds; runs whose names differ only in _s<seed> are grouped."""
    groups = {}
    for row in rows:
        groups.setdefault(re.sub(r"_s\d+$", "", row["name"]), []).append(row)
    keys = ["train", "held-out", "nearest", "PCA"]
    print(f"\n{'mean ± std over seeds':32s}" + "".join(f"{k:>20s}" for k in keys))
    for name, group in sorted(groups.items()):
        cells = []
        for k in keys:
            v = np.array([r[k] for r in group if r[k] is not None])
            cells.append(f"{v.mean():.3e} ± {v.std():.1e}" if len(v) else "-")
        print(f"{name:32s}" + "".join(f"{c:>20s}" for c in cells) + f"  n={len(group)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ConvAutoencoders")
    parser.add_argument("--data_dir", required=True, help="folder with the .npy clips")
    parser.add_argument("checkpoints", nargs="+", help="one or more .pt files")
    args = parser.parse_args()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    rows = []
    for path in args.checkpoints:
        run = load_run(path, args.data_dir)
        dt_s, latent_size = run["ckpt"]["dt_s"], run["num_latent"] * run["H"]
        jitter_z, jitter = jitter_floor_z(run, dt_s)
        row = {"name": run["name"], "train": mse(run["x_hat"], run["x"])}
        row |= {"held-out": None, "nearest": None, "PCA": None}
        levels = {"jitter floor": jitter_z}
        if run["x_val"] is not None:
            row["held-out"] = mse(run["x_hat_val"], run["x_val"])
            row["nearest"] = nearest_window_mse(run["x"], run["x_val"])
            row["PCA"] = pca_mse(run["x"], run["x_val"], latent_size)
            levels |= {"nearest window": row["nearest"], "PCA": row["PCA"]}
        rows.append(row)

        print(f"\n{run['name']}  (z-scored MSE; jitter floor {jitter_z:.2e})")
        print(
            "  "
            + "  ".join(f"{k} {v:.3e}" for k, v in row.items() if v and k != "name")
        )
        plot_loss_curves(run, levels)
        plot_per_joint_error(run, jitter)
        plot_error_spectrum(run, dt_s)
        plot_latent_portrait(run, dt_s)
    print_seed_table(rows)


if __name__ == "__main__":
    main()
