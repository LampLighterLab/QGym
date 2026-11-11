"""Evaluate ConvAutoencoder checkpoints from scripts/train_autoencoder.py.

    uv run --frozen --extra fld scripts/evaluate_autoencoder.py \
        path/to/dog_walk00.npy logs/autoencoder/*.pt

Writes PNGs to logs/autoencoder/plots/ (gitignored via logs/):
  data_stride_overlay.png       every stride of the clip on a common phase axis
  <run>_loss_curves.png         train/held-out loss vs PCA and jitter levels
  <run>_per_joint_error.png     per-joint RMSE [mrad] vs retargeting jitter
  <run>_error_spectrum.png      which frequencies the model reproduces
  <run>_latent_portrait.png     latent trajectory over the clip (one loop/stride?)
  <run>_error_vs_position.png   reconstruction error across the window
  sweep_latent_size.png         error vs latent size, model vs PCA (>1 checkpoint)

"Jitter" is the clip's content above JITTER_CUTOFF_HZ: the stride lives near
1.6 Hz, so this is retargeting (IK) noise. A model below the jitter level is
reproducing noise, i.e. memorizing.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from legged_fld.autoencoder import ConvAutoencoder
from legged_fld.data import make_windows, normalize

from gym import GYM_ROOT_DIR

# reuse the loader, joint order, and split the models were trained with
from train_autoencoder import load_kine2go_joints, split_trajectory

PLOTS_DIR = os.path.join(GYM_ROOT_DIR, "logs", "autoencoder", "plots")
JITTER_CUTOFF_HZ = 10.0
EDGE_SAMPLES = 15  # ignore window edges (zero padding) in the shift check
STRIDE_JOINTS = ["FL_thigh", "FL_calf", "RR_thigh", "RR_calf"]
STRIDE_SAMPLES = 100  # each stride resampled onto this many phase points


# ---------------------------------------------------------------- signals


def high_frequency_rms(signal, dt_s, cutoff_hz):
    """Per-channel RMS of the content above cutoff_hz. signal: (T, M)."""
    spectrum = np.fft.rfft(signal - signal.mean(0), axis=0)
    freqs = np.fft.rfftfreq(len(signal), dt_s)
    spectrum[freqs <= cutoff_hz] = 0
    return np.sqrt((np.fft.irfft(spectrum, n=len(signal), axis=0) ** 2).mean(0))


def power_spectrum(signal, dt_s):
    """Frequencies [Hz] and power summed over channels. signal: (T, M)."""
    power = (np.abs(np.fft.rfft(signal - signal.mean(0), axis=0)) ** 2).sum(1)
    return np.fft.rfftfreq(len(signal), dt_s), power


def dominant_frequency(signal, dt_s):
    freqs, power = power_spectrum(signal, dt_s)
    return freqs[1:][np.argmax(power[1:])]


def pca_mse(x_fit, x_eval, num_components):
    """MSE of the best rank-k linear autoencoder fit on x_fit, applied to x_eval."""
    X_fit, X_eval = x_fit.flatten(start_dim=1), x_eval.flatten(start_dim=1)
    mean = X_fit.mean(dim=0)
    _, _, Vt = torch.linalg.svd(X_fit - mean, full_matrices=False)
    V = Vt[:num_components].T
    recon = mean + (X_eval - mean) @ V @ V.T
    return (recon - X_eval).pow(2).mean().item()


def center_track(windows):
    """Centre sample of every window -> (B, C): a time series over the clip."""
    return windows[:, :, windows.shape[-1] // 2].numpy()


# ---------------------------------------------------------------- models


def load_run(path, trajectory):
    """Load a checkpoint and rebuild its train/held-out windows."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    H, mean, std = ckpt["window_length"], ckpt["mean"], ckpt["std"]
    train_fraction = ckpt["args"].get("train_fraction", 1.0)
    train_traj, val_traj = split_trajectory(trajectory, train_fraction)

    num_latent = ckpt["args"]["latent_channels"]
    ae = ConvAutoencoder(trajectory.shape[1], num_latent, H)
    ae.load_state_dict(ckpt["model"])
    ae.eval()

    run = {
        "name": os.path.splitext(os.path.basename(path))[0],
        "ckpt": ckpt,
        "ae": ae,
        "H": H,
        "num_latent": num_latent,
        "std": std.numpy(),
        "train_traj": train_traj,
        "x": make_windows(normalize(train_traj, mean, std), H),
        "x_val": None,
    }
    if val_traj is not None and len(val_traj) >= H:
        run["x_val"] = make_windows(normalize(val_traj, mean, std), H)
    with torch.no_grad():
        run["x_hat"], run["z"] = ae(run["x"])
        if run["x_val"] is not None:
            run["x_hat_val"], _ = ae(run["x_val"])
    return run


def mse(a, b):
    return (a - b).pow(2).mean().item()


# ---------------------------------------------------------------- plots


def save(fig, filename, tight=True):
    if tight:
        fig.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_stride_overlay(trajectory, joint_names, dt_s):
    """Cut the clip at each stride and overlay strides on a common phase axis.

    Stride starts are upward mean-crossings of FL_thigh (lightly smoothed).
    The spread across strides is variation no periodic model can explain.
    """
    q = trajectory.numpy()
    ref = q[:, joint_names.index("FL_thigh")]
    ref = np.convolve(ref - ref.mean(), np.ones(5) / 5, mode="same")
    period = 1 / dominant_frequency(q, dt_s) / dt_s
    starts = []
    for k in np.flatnonzero((ref[:-1] < 0) & (ref[1:] >= 0)):
        if not starts or k - starts[-1] > 0.5 * period:
            starts.append(k)
    durations_s = np.diff(starts) * dt_s

    phase = np.linspace(0, 1, STRIDE_SAMPLES)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, name in zip(axes.flat, STRIDE_JOINTS, strict=True):
        c = joint_names.index(name)
        strides = np.array(
            [
                np.interp(phase, np.linspace(0, 1, b - a + 1), q[a : b + 1, c])
                for a, b in zip(starts[:-1], starts[1:], strict=True)
            ]
        )
        for stride in strides:
            ax.plot(phase, stride, color="0.7", lw=0.8)
        ax.plot(phase, strides.mean(0), color="C0", lw=2, label="mean stride")
        spread = 1e3 * strides.std(0).mean()
        ax.set_title(f"{name}: spread across strides {spread:.1f} mrad")
        ax.set_ylabel("rad")
    for ax in axes[1]:
        ax.set_xlabel("stride phase")
    axes[0, 0].legend()
    fig.suptitle(
        f"{len(durations_s)} strides, duration "
        f"{durations_s.mean():.3f} ± {durations_s.std():.3f} s"
    )
    save(fig, "data_stride_overlay.png")


def jitter_floor_z(run, dt_s):
    """MSE (z-units) of a model that is perfect except for the jitter."""
    jitter = high_frequency_rms(run["train_traj"].numpy(), dt_s, JITTER_CUTOFF_HZ)
    return np.mean((jitter / run["std"]) ** 2), jitter


def plot_loss_curves(run, levels):
    ckpt = run["ckpt"]
    if "train_losses" not in ckpt:
        print("  (no loss history in this checkpoint; skipping loss curves)")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(ckpt["train_losses"], lw=0.8, label="train (train-mode BN)")
    if ckpt["val_history"]:
        steps, val = zip(*ckpt["val_history"], strict=True)
        ax.semilogy(steps, val, lw=1.5, label="held-out")
    styles = {
        "jitter floor": ("k", "--"),
        "PCA train": ("C2", ":"),
        "PCA held-out": ("C3", ":"),
    }
    for label, value in levels.items():
        color, ls = styles[label]
        ax.axhline(value, color=color, ls=ls, label=f"{label} ({value:.1e})")
    ax.set_xlabel("step")
    ax.set_ylabel("z-scored MSE")
    ax.set_title(f"{run['name']}: loss vs reference levels")
    ax.legend()
    save(fig, f"{run['name']}_loss_curves.png")


def plot_per_joint_error(run, joint_names, jitter):
    def rmse_rad(x, x_hat):
        return 1e3 * (x_hat - x).pow(2).mean(dim=(0, 2)).sqrt().numpy() * run["std"]

    bars = {"train RMSE": rmse_rad(run["x"], run["x_hat"])}
    if run["x_val"] is not None:
        bars["held-out RMSE"] = rmse_rad(run["x_val"], run["x_hat_val"])
    bars[f"jitter (>{JITTER_CUTOFF_HZ:g} Hz)"] = 1e3 * jitter

    fig, ax = plt.subplots(figsize=(10, 4.5))
    idx, width = np.arange(len(joint_names)), 0.8 / len(bars)
    for k, (label, values) in enumerate(bars.items()):
        ax.bar(idx + (k - (len(bars) - 1) / 2) * width, values, width, label=label)
    ax.set_xticks(idx, joint_names, rotation=45)
    ax.set_ylabel("mrad")
    ax.set_title(f"{run['name']}: per-joint error vs retargeting jitter")
    ax.legend()
    save(fig, f"{run['name']}_per_joint_error.png")
    return bars


def plot_error_spectrum(run, dt_s):
    fig, ax = plt.subplots(figsize=(9, 5))
    freqs, data = power_spectrum(center_track(run["x"]), dt_s)
    ax.semilogy(freqs, data / data.max(), label="data (train)")
    err = center_track(run["x_hat"] - run["x"])
    ax.semilogy(
        *_normalized(power_spectrum(err, dt_s), data.max()), label="error (train)"
    )
    if run["x_val"] is not None:
        val_data = power_spectrum(center_track(run["x_val"]), dt_s)
        err_val = center_track(run["x_hat_val"] - run["x_val"])
        ax.semilogy(
            *_normalized(power_spectrum(err_val, dt_s), val_data[1].max()),
            label="error (held-out)",
        )
    ax.axvline(JITTER_CUTOFF_HZ, color="k", ls="--", lw=0.8, label="jitter cutoff")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("power / data peak")
    ax.set_title(f"{run['name']}: what the model reproduces (error << data) and misses")
    ax.legend()
    save(fig, f"{run['name']}_error_spectrum.png")


def _normalized(spectrum, scale):
    freqs, power = spectrum
    return freqs, power / scale


def plot_latent_portrait(run, dt_s):
    """Latent at window centres over the clip. A clean gait gives one loop per
    stride, with successive loops on top of each other."""
    z = center_track(run["z"])
    t_s = np.arange(len(z)) * dt_s
    L = z.shape[1]
    if L == 1:
        f = dominant_frequency(center_track(run["x"]), dt_s)
        lag = max(round(1 / (4 * f * dt_s)), 1)  # quarter stride
        pairs = [(z[:-lag, 0], z[lag:, 0], "z0(t)", f"z0(t + {lag} samples)")]
        t_s = t_s[:-lag]
    else:
        pairs = [
            (z[:, i], z[:, i + 1], f"z{i}", f"z{i + 1}")
            for i in range(0, min(L, 4) - 1, 2)
        ]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 5))
    for ax, (a, b, xlabel, ylabel) in zip(np.atleast_1d(axes), pairs, strict=True):
        ax.plot(a, b, color="0.8", lw=0.6)
        points = ax.scatter(a, b, c=t_s, s=6, cmap="viridis")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.colorbar(points, ax=axes, label="time in clip [s]")
    fig.suptitle(f"{run['name']}: latent phase portrait")
    save(fig, f"{run['name']}_latent_portrait.png", tight=False)  # colorbar


def plot_error_vs_position(run, dt_s):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    t_s = (np.arange(run["H"]) - run["H"] // 2) * dt_s
    train = (run["x_hat"] - run["x"]).pow(2).mean(dim=(0, 1)).numpy()
    ax.semilogy(t_s, train, label="train")
    if run["x_val"] is not None:
        val = (run["x_hat_val"] - run["x_val"]).pow(2).mean(dim=(0, 1)).numpy()
        ax.semilogy(t_s, val, label="held-out")
    ax.set_xlabel("time relative to window centre [s]")
    ax.set_ylabel("z-scored MSE")
    ax.set_title(f"{run['name']}: error across the window (edges see zero padding)")
    ax.legend()
    save(fig, f"{run['name']}_error_vs_position.png")


def plot_sweep(rows, jitter_z):
    rows = sorted(rows, key=lambda r: r["latent_size"])
    sizes = [r["latent_size"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(sizes, [r["train"] for r in rows], "o-", label="AE train")
    ax.loglog(sizes, [r["pca_train"] for r in rows], "s:", label="PCA train")
    if all(r["val"] is not None for r in rows):
        ax.loglog(sizes, [r["val"] for r in rows], "o-", label="AE held-out")
        ax.loglog(sizes, [r["pca_val"] for r in rows], "s:", label="PCA held-out")
    ax.axhline(jitter_z, color="k", ls="--", lw=0.8, label="jitter floor")
    for r in rows:
        ax.annotate(
            f"L={r['L']}",
            (r["latent_size"], r["train"]),
            textcoords="offset points",
            xytext=(4, -12),
            fontsize=8,
        )
    ax.set_xlabel("latent numbers per window (L x H)")
    ax.set_ylabel("z-scored MSE")
    ax.set_title("does the network matter, or only the bottleneck size?")
    ax.legend()
    save(fig, "sweep_latent_size.png")


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description="Evaluate ConvAutoencoders")
    parser.add_argument("motion", help="kine2go .npy file the models were trained on")
    parser.add_argument("checkpoints", nargs="+", help="one or more .pt files")
    args = parser.parse_args()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    trajectory = load_kine2go_joints(args.motion)
    first = torch.load(args.checkpoints[0], map_location="cpu", weights_only=True)
    dt_s, joint_names = first["dt_s"], first["joint_names"]

    print("data")
    plot_stride_overlay(trajectory, joint_names, dt_s)

    rows = []
    for path in args.checkpoints:
        run = load_run(path, trajectory)
        latent_size = run["num_latent"] * run["H"]
        jitter_z, jitter = jitter_floor_z(run, dt_s)
        row = {
            "L": run["num_latent"],
            "train_fraction": run["ckpt"]["args"].get("train_fraction", 1.0),
            "jitter_z": jitter_z,
            "latent_size": latent_size,
            "train": mse(run["x_hat"], run["x"]),
            "pca_train": pca_mse(run["x"], run["x"], latent_size),
            "val": None,
            "pca_val": None,
        }
        levels = {"jitter floor": jitter_z, "PCA train": row["pca_train"]}
        if run["x_val"] is not None:
            row["val"] = mse(run["x_hat_val"], run["x_val"])
            row["pca_val"] = pca_mse(run["x"], run["x_val"], latent_size)
            levels["PCA held-out"] = row["pca_val"]
        rows.append(row)

        e = EDGE_SAMPLES
        z = run["z"]
        later, earlier = z[1:, :, e : run["H"] - e - 1], z[:-1, :, e + 1 : run["H"] - e]
        shift_err = ((later - earlier).norm() / earlier.norm()).item()
        f_motion = dominant_frequency(center_track(run["x"]), dt_s)
        f_latent = dominant_frequency(center_track(z), dt_s)

        print(f"\n{run['name']}: L={run['num_latent']}, {latent_size} numbers/window")
        val = f"{row['val']:.2e}" if row["val"] is not None else "-"
        pca_val = f"{row['pca_val']:.2e}" if row["pca_val"] is not None else "-"
        print(f"  MSE (z)   train {row['train']:.2e}  held-out {val}")
        print(f"  PCA       train {row['pca_train']:.2e}  held-out {pca_val}")
        print(f"  jitter floor {jitter_z:.2e}")
        print(f"  dominant freq: motion {f_motion:.2f} Hz, latent {f_latent:.2f} Hz")
        print(f"  latent shift inconsistency (interior): {shift_err:.1%}")

        plot_loss_curves(run, levels)
        plot_per_joint_error(run, joint_names, jitter)
        plot_error_spectrum(run, dt_s)
        plot_latent_portrait(run, dt_s)
        plot_error_vs_position(run, dt_s)

    # only compare runs trained on the same split
    fractions = [r["train_fraction"] for r in rows]
    sweep = [
        r for r in rows if r["train_fraction"] == max(fractions, key=fractions.count)
    ]
    if len(sweep) > 1:
        print(f"\nsweep (train_fraction {sweep[0]['train_fraction']:g})")
        plot_sweep(sweep, sweep[0]["jitter_z"])


if __name__ == "__main__":
    main()
