"""Leave-one-clip-out sweep for scripts/train_autoencoder.py.

    uv run --frozen --extra fld scripts/sweep_autoencoder.py
        [--data_dir DIR] [--clips 00 01 02 03 04 06] [--seeds 0 1 2]
        [--kernel_size 13] [--num_steps 5000] [--dry_run] [--no_wandb]
        [--stream] [-- extra train_autoencoder.py args]

For every clip in --clips and every seed, trains on the other clips and holds
that one out (dog_walk<clip>_joint_pos.npy in --data_dir). Each run is a
separate train_autoencoder.py process, logged to
logs/autoencoder/sweep_<time>/<run>.log. At the end, every checkpoint is
scored against the nearest-window (memorization) and same-size PCA baselines,
and the table is printed and saved as summary.csv next to the logs.

Progress display (sweep_progress.py, display only): on a terminal, one table
of all runs redrawn in place; --stream echoes trainer output live instead;
piped/redirected stdout gets one line per finished run.

Refuses to start with W&B disabled unless --no_wandb is given, so a long sweep
doesn't silently log nowhere.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import namedtuple
from types import SimpleNamespace

import numpy as np

from gym import GYM_ROOT_DIR

from evaluate_autoencoder import (
    bottleneck_size,
    load_run,
    mse,
    nearest_window_mse,
    pca_mse,
)
from sweep_progress import RULE, SweepProgress
from train_autoencoder import OUT_DIR, WANDB_CONFIG, load_kine2go_joints, make_run_name

DEFAULT_DATA_DIR = os.path.expanduser(
    "~/Documents/fourier-stuff/kine2go-pipeline/motion_retargeting/results"
)
TRAIN_SCRIPT = os.path.join(GYM_ROOT_DIR, "scripts", "train_autoencoder.py")

Job = namedtuple("Job", "model latent_channels clip seed")


def main():
    args = parse_args()
    jobs = [
        Job(model, L, clip, seed)
        for seed in args.seeds
        for model in args.models
        for L in args.latent_channels
        for clip in args.clips
    ]

    print(RULE)
    print(
        f"leave-one-clip-out sweep: {len(args.models)} models x "
        f"{len(args.latent_channels)} L x {len(args.clips)} clips x "
        f"{len(args.seeds)} seeds = {len(jobs)} runs\n"
        f"  models {args.models}, L {args.latent_channels}, K={args.kernel_size}, "
        f"{args.num_steps} steps"
    )
    print(RULE)
    check_inputs(args)
    check_wandb(args)
    print(RULE)

    if args.dry_run:
        for i, job in enumerate(jobs, 1):
            name, command = train_command(args, job)
            print(
                f"[{i:3d}/{len(jobs)}] {job.model} L{job.latent_channels}, "
                f"hold out walk{job.clip}, seed {job.seed}"
            )
            print("    " + " ".join(command))
        return

    log_dir = os.path.join(OUT_DIR, time.strftime("sweep_%Y%m%d_%H%M%S"))
    os.makedirs(log_dir)
    print(f"logs: {log_dir}")
    progress = SweepProgress(jobs, args.num_steps, log_dir, mode=display_mode(args))

    results = []
    for i, job in enumerate(jobs):
        name, command = train_command(args, job)
        log_path = os.path.join(log_dir, f"{name}.log")
        progress.start_run(i, log_path)
        ok = run(command, log_path, on_line=progress.on_line)
        scores = score(name, args.data_dir) if ok else None
        results.append((job, scores))
        progress.finish_run(i, scores, log_path)
    progress.close()
    summarize(results, args, log_dir)


# --- one run ------------------------------------------------------------------


def clip_path(data_dir, clip):
    return os.path.join(data_dir, f"dog_walk{clip}_joint_pos.npy")


def train_command(args, job):
    """(run name, train_autoencoder.py command) holding out job.clip."""
    val_path = clip_path(args.data_dir, job.clip)
    train_paths = [clip_path(args.data_dir, c) for c in args.clips if c != job.clip]
    name = make_run_name(
        SimpleNamespace(
            model=job.model,
            val_motions=[val_path],
            train_fraction=1.0,
            latent_channels=job.latent_channels,
            kernel_size=args.kernel_size,
            seed=job.seed,
        )
    )
    command = [
        sys.executable,
        TRAIN_SCRIPT, *train_paths,
        "--val_motions", val_path,
        "--model", job.model,
        "--latent_channels", str(job.latent_channels),
        "--kernel_size", str(args.kernel_size),
        "--num_steps", str(args.num_steps),
        "--seed", str(job.seed),
        *(["--disable_wandb"] if args.no_wandb else []),
        *args.train_args,
    ]
    return name, command


def run(command, log_path, on_line):
    """Run one training process; write its output to log_path and pass each line
    to on_line. Return whether it exited cleanly."""
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=GYM_ROOT_DIR,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        for line in process.stdout:
            log.write(line)
            on_line(line)
        return process.wait() == 0


def score(name, data_dir):
    """Model, nearest-window and PCA held-out MSE for one saved checkpoint."""
    loaded = load_run(os.path.join(OUT_DIR, f"{name}.pt"), data_dir)
    x, x_val = loaded["x"], loaded["x_val"]
    return {
        "model": mse(loaded["x_hat_val"], x_val),
        "nearest": nearest_window_mse(x, x_val),
        "pca": pca_mse(x, x_val, bottleneck_size(loaded)),
    }


# --- results ------------------------------------------------------------------

def summarize(results, args, log_dir):
    """Per (model, L): held-out MSE per clip (mean ± std over seeds), then baselines."""
    configs = [(m, L) for m in args.models for L in args.latent_channels]
    print(f"\n{RULE}\nheld-out MSE (z-scored), mean ± std over seeds {args.seeds}")
    print(f"{'config':9s}" + "".join(f"{'walk' + c:>15s}" for c in args.clips)
          + f"{'mean':>9s}")
    rows = []
    for model, L in configs:
        line, means = f"{f'{model} L{L}':9s}", []
        for clip in args.clips:
            scores = [s for j, s in results
                      if (j.model, j.latent_channels, j.clip) == (model, L, clip) and s]
            if not scores:
                line += f"{'—':>15s}"
                continue
            m = np.array([s["model"] for s in scores])
            means.append(m.mean())
            line += f"{f'{m.mean():.4f}±{m.std():.4f}':>15s}"
            rows.append((model, L, clip, m.mean(), m.std(),
                         scores[0]["nearest"], scores[0]["pca"], len(scores)))
        print(line + (f"{np.mean(means):>9.4f}" if means else ""))

    print(f"\n{'config':9s}{'model':>9s}{'PCA, same size':>16s}{'model/PCA':>11s}"
          f"{'nearest':>9s}   (means over clips)")
    for model, L in configs:
        r = np.array([row[3:7] for row in rows if row[:2] == (model, L)])
        if len(r):
            mean, _, nearest, pca = r.mean(axis=0)
            print(f"{f'{model} L{L}':9s}{mean:>9.4f}{pca:>16.4f}{mean / pca:>11.2f}"
                  f"{nearest:>9.4f}")

    summary_path = os.path.join(log_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "latent_channels", "held_out", "model_mean",
                         "model_std", "nearest", "pca", "n"])
        writer.writerows(rows)
    print(f"\nsaved {summary_path}")


# --- setup --------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Leave-one-clip-out AE sweep")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--models", nargs="+", choices=["ae", "pae"], default=["ae", "pae"]
    )
    parser.add_argument(
        "--clips", nargs="+", default=["00", "01", "02", "03", "04", "06"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--kernel_size", type=int, default=1)
    parser.add_argument("--latent_channels", nargs="+", type=int, default=[4,6,8])
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--dry_run", action="store_true", help="print commands only")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="echo trainer output live (scrolling) instead of the progress table",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="after `--`: passed to train_autoencoder.py",
    )
    args = parser.parse_args()
    args.train_args = [a for a in args.train_args if a != "--"]
    args.data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    return args


def display_mode(args):
    """Progress table on a terminal; plain lines when piped; or --stream."""
    if args.stream:
        return "stream"
    return "table" if sys.stdout.isatty() else "lines"


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
    config = os.path.join(GYM_ROOT_DIR, "user", WANDB_CONFIG)
    flags = {"--wandb_entity", "--wandb_project"} <= set(args.train_args)
    if not (os.path.exists(config) or flags):
        sys.exit(
            f"W&B would be disabled: {config} not found and no "
            "--wandb_entity/--wandb_project after `--`.\n"
            "Create the config, or pass --no_wandb to run without W&B."
        )
    print(f"W&B: on ({'flags' if flags else config})")


if __name__ == "__main__":
    main()
