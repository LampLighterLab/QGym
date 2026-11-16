"""Leave-one-clip-out sweep for scripts/train_autoencoder.py.

    uv run --frozen --extra fld scripts/sweep_autoencoder.py
        [--data_dir DIR] [--clips 00 01 02 03 04 06] [--seeds 0 1 2]
        [--kernel_size 13] [--num_steps 5000] [--dry_run] [--no_wandb]
        [-- extra train_autoencoder.py args]

For every clip in --clips and every seed, trains on the other clips and holds
that one out (dog_walk<clip>_joint_pos.npy in --data_dir). Each run is a
separate train_autoencoder.py process, logged to
logs/autoencoder/sweep_<time>/<run>.log. At the end, every checkpoint is
scored against the nearest-window (memorization) and same-size PCA baselines,
and the table is printed and saved as summary.csv next to the logs.

Refuses to start with W&B disabled unless --no_wandb is given, so a long sweep
doesn't silently log nowhere.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from gym import GYM_ROOT_DIR

from evaluate_autoencoder import load_run, mse, nearest_window_mse, pca_mse
from train_autoencoder import OUT_DIR, load_kine2go_joints, make_run_name

DEFAULT_DATA_DIR = os.path.expanduser(
    "~/Documents/fourier-stuff/kine2go-pipeline/motion_retargeting/results"
)
TRAIN_SCRIPT = os.path.join(GYM_ROOT_DIR, "scripts", "train_autoencoder.py")
RULE = "-" * 78


def clip_path(data_dir, clip):
    return os.path.join(data_dir, f"dog_walk{clip}_joint_pos.npy")


def format_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def check_inputs(args):
    """Every clip exists; print their lengths. Exit early on anything missing."""
    print(f"clips in {args.data_dir}")
    for clip in args.clips:
        path = clip_path(args.data_dir, clip)
        if not os.path.exists(path):
            sys.exit(f"missing {path}")
        frames = len(load_kine2go_joints(path))
        print(f"  walk{clip}  {frames:5d} frames  ({frames / 60:4.1f} s)")


def check_wandb(args):
    """Mirror WandbSingleton's rule: on iff entity and project are known."""
    if args.no_wandb:
        print("W&B: off (--no_wandb)")
        return
    config = os.path.join(GYM_ROOT_DIR, "user", "wandb_config.json")
    flags = {"--wandb_entity", "--wandb_project"} <= set(args.train_args)
    if not (os.path.exists(config) or flags):
        sys.exit(
            f"W&B would be disabled: {config} not found and no "
            "--wandb_entity/--wandb_project after `--`.\n"
            "Create the config, or pass --no_wandb to run without W&B."
        )
    print(f"W&B: on ({'flags' if flags else config})")


def run_one(args, held_out, seed, log_dir):
    """Train one fold/seed; return (run name, seconds, whether it succeeded)."""
    val_path = clip_path(args.data_dir, held_out)
    train_paths = [clip_path(args.data_dir, c) for c in args.clips if c != held_out]
    name = make_run_name(
        SimpleNamespace(
            val_motions=[val_path],
            train_fraction=1.0,
            latent_channels=args.latent_channels,
            kernel_size=args.kernel_size,
            seed=seed,
        )
    )
    command = [
        sys.executable,
        TRAIN_SCRIPT,
        *train_paths,
        "--val_motions",
        val_path,
        "--kernel_size",
        str(args.kernel_size),
        "--latent_channels",
        str(args.latent_channels),
        "--num_steps",
        str(args.num_steps),
        "--seed",
        str(seed),
        *(["--disable_wandb"] if args.no_wandb else []),
        *args.train_args,
    ]
    if args.dry_run:
        print("    " + " ".join(command))
        return name, 0.0, False

    log_path = os.path.join(log_dir, f"{name}.log")
    start = time.time()
    with open(log_path, "w") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=GYM_ROOT_DIR,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    seconds = time.time() - start
    if result.returncode != 0:
        with open(log_path) as log:
            tail = log.readlines()[-15:]
        print(f"    FAILED (exit {result.returncode}); last lines of {log_path}:")
        print("".join("      " + line for line in tail), end="")
    return name, seconds, result.returncode == 0


def score(name, data_dir):
    """Model, nearest-window and PCA held-out MSE for one saved checkpoint."""
    run = load_run(os.path.join(OUT_DIR, f"{name}.pt"), data_dir)
    return {
        "model": mse(run["x_hat_val"], run["x_val"]),
        "nearest": nearest_window_mse(run["x"], run["x_val"]),
        "pca": pca_mse(run["x"], run["x_val"], run["num_latent"] * run["H"]),
    }


def summarize(results, args, log_dir):
    """Per held-out clip: mean ± std over seeds, and the model / PCA ratio."""
    print(f"\n{RULE}\nheld-out MSE (z-scored), mean ± std over seeds {args.seeds}")
    print(
        f"{'held out':10s}{'model':>22s}{'nearest window':>17s}{'PCA':>10s}"
        f"{'model/PCA':>12s}"
    )
    rows = []
    for clip in args.clips:
        scores = [s for c, _, s in results if c == clip and s is not None]
        if not scores:
            print(f"walk{clip:6s}{'(no finished runs)':>22s}")
            continue
        model = np.array([s["model"] for s in scores])
        nearest, pca = scores[0]["nearest"], scores[0]["pca"]  # seed-independent
        rows.append((clip, model.mean(), model.std(), nearest, pca, len(scores)))
        print(
            f"walk{clip:6s}{model.mean():>12.4f} ± {model.std():.4f}"
            f"{nearest:>17.4f}{pca:>10.4f}{model.mean() / pca:>12.2f}"
        )
    if rows:
        model, _, nearest, pca = np.array([r[1:5] for r in rows]).mean(axis=0)
        print(RULE)
        print(
            f"{'mean':10s}{model:>12.4f}{'':9s}{nearest:>17.4f}{pca:>10.4f}"
            f"{model / pca:>12.2f}"
        )
    print("model < nearest window: it generalizes beyond copying training windows;")
    print("model < PCA: the network beats the best linear model of the same size.")

    summary_path = os.path.join(log_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["held_out", "model_mean", "model_std", "nearest", "pca", "n"])
        writer.writerows(rows)
    print(f"\nsaved {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Leave-one-clip-out AE sweep")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--clips", nargs="+", default=["00", "01", "02", "03", "04", "06"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--kernel_size", type=int, default=13)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--dry_run", action="store_true", help="print commands only")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="after `--`: passed to train_autoencoder.py",
    )
    args = parser.parse_args()
    args.train_args = [a for a in args.train_args if a != "--"]
    args.data_dir = os.path.abspath(os.path.expanduser(args.data_dir))

    jobs = [(clip, seed) for clip in args.clips for seed in args.seeds]
    print(RULE)
    print(
        f"leave-one-clip-out sweep: {len(args.clips)} clips x {len(args.seeds)} "
        f"seeds = {len(jobs)} runs  (K={args.kernel_size}, L={args.latent_channels},"
        f" {args.num_steps} steps)"
    )
    print(RULE)
    check_inputs(args)
    check_wandb(args)
    log_dir = os.path.join(OUT_DIR, time.strftime("sweep_%Y%m%d_%H%M%S"))
    if not args.dry_run:
        os.makedirs(log_dir)
        print(f"logs: {log_dir}")
    print(RULE)

    results, start = [], time.time()
    for i, (clip, seed) in enumerate(jobs, 1):
        print(f"[{i:2d}/{len(jobs)}] hold out walk{clip}, seed {seed}", flush=True)
        name, seconds, ok = run_one(args, clip, seed, log_dir)
        if args.dry_run:
            continue
        scores = score(name, args.data_dir) if ok else None
        results.append((clip, seed, scores))
        if scores:
            eta = (time.time() - start) / i * (len(jobs) - i)
            print(
                f"    held-out {scores['model']:.4f}  (nearest window "
                f"{scores['nearest']:.4f}, PCA {scores['pca']:.4f})  "
                f"{format_duration(seconds)}, ETA {format_duration(eta)}"
            )
    if not args.dry_run:
        print(f"\nfinished in {format_duration(time.time() - start)}")
        summarize(results, args, log_dir)


if __name__ == "__main__":
    main()
