"""Run a controlled Go2Trot evaluation across labeled policies and checkpoints.

Examples:

    uv run --frozen scripts/eval_go2_policy.py \
        --policy baseline=logs/go2trot/Aug06_09-34-49_ \
        --policy current=logs/go2trot/Aug07_15-49-08_ \
        --iterations 100 250 500

    GO2_EVAL_DIR=logs/go2_evaluation \
        uv run --frozen marimo edit notebooks/go2_policy_evaluation.py

The default ``--backend mjx`` selects Q2's MuJoCo Warp path on ``cuda:0``.
Use ``--backend mujoco`` for MuJoCo CPU. For VSim, launch this script through
``uv run --env-file .env.vsim`` and pass ``--backend vsim``.
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys

from gym.utils.legged_eval_metrics import GO2_COMMAND_CASES


CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")
EVALUATION_BACKENDS = {
    "mjx": ("mujoco", "cuda:0", "mujoco-cuda:0"),
    "mujoco": ("mujoco", "cpu", "mujoco-cpu"),
    "vsim": ("vsim", "cuda:0", "vsim"),
}


def parse_labeled_path(value):
    """Parse LABEL=PATH without imposing filename restrictions on PATH."""
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("policy must be written as LABEL=PATH")
    return label, Path(path)


def _checkpoint_iteration(path):
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"checkpoint must be named model_<iteration>.pt: {path}")
    return int(match.group(1))


def resolve_policy_checkpoints(path, requested_iterations):
    """Resolve a checkpoint file or selected iterations from one run directory."""
    path = path.expanduser().resolve()
    if path.is_file():
        return [(path, _checkpoint_iteration(path))]
    if not path.is_dir():
        raise FileNotFoundError(path)

    available = {}
    for checkpoint in path.glob("model_*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
        if match is not None:
            available[int(match.group(1))] = checkpoint.resolve()
    if not available:
        raise FileNotFoundError(f"no model_*.pt checkpoints directly under {path}")

    selected = []
    for requested in requested_iterations:
        iteration = max(available) if requested == "latest" else int(requested)
        if iteration not in available:
            options = ", ".join(str(value) for value in sorted(available))
            raise FileNotFoundError(
                f"model_{iteration}.pt is absent from {path}; available: {options}"
            )
        if iteration not in {value for _, value in selected}:
            selected.append((available[iteration], iteration))
    return selected


def _filename_label(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not cleaned:
        raise ValueError(f"label {value!r} cannot form a filename")
    return cleaned


def resolve_evaluation_backend(name):
    """Map the evaluator's public backend name to Q2's backend/device pair."""
    try:
        return EVALUATION_BACKENDS[name]
    except KeyError as error:
        options = ", ".join(EVALUATION_BACKENDS)
        raise ValueError(
            f"unknown evaluation backend {name!r}; choose {options}"
        ) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        type=parse_labeled_path,
        metavar="LABEL=PATH",
        help="checkpoint file or run directory; repeat to compare policies",
    )
    parser.add_argument(
        "--iterations",
        nargs="+",
        default=["latest"],
        help="checkpoint iterations selected from every run directory",
    )
    parser.add_argument(
        "--backend",
        choices=list(EVALUATION_BACKENDS),
        default="mjx",
        help="mjx: MuJoCo Warp on cuda:0 (default); mujoco: MuJoCo CPU; "
        "vsim: VSim on cuda:0",
    )
    parser.add_argument("--num_envs", type=int, default=200)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--settling_time", type=float, default=0.5)
    parser.add_argument("--contact_threshold", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reset_mode",
        action="append",
        choices=["reset_to_basic", "reset_to_range"],
        help="repeat to evaluate both; default: reset_to_range",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("logs/go2_evaluation"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.num_envs <= 0 or args.num_envs % len(GO2_COMMAND_CASES):
        raise ValueError(
            f"num_envs must be a positive multiple of {len(GO2_COMMAND_CASES)}"
        )
    reset_modes = args.reset_mode or ["reset_to_range"]
    eval_backend, eval_device, evaluation_label = resolve_evaluation_backend(
        args.backend
    )
    repo_root = Path(__file__).resolve().parents[1]
    evaluator = repo_root / "scripts" / "eval_policy.py"
    output_dir = args.out_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_evaluation_label = _filename_label(evaluation_label)

    for label, policy_path in args.policy:
        checkpoints = resolve_policy_checkpoints(policy_path, args.iterations)
        filename_label = _filename_label(label)
        for checkpoint, iteration in checkpoints:
            for reset_mode in reset_modes:
                mode_label = reset_mode.removeprefix("reset_to_")
                output = output_dir / (
                    f"{filename_label}__iter_{iteration:05d}"
                    f"__{filename_evaluation_label}__{mode_label}__seed_{args.seed}.npz"
                )
                if output.exists() and not args.force:
                    raise FileExistsError(
                        f"{output} exists; pass --force to replace it"
                    )
                command = [
                    sys.executable,
                    str(evaluator),
                    "--task",
                    "go2trot",
                    "--ckpt",
                    str(checkpoint),
                    "--train_label",
                    label,
                    "--eval_backend",
                    eval_backend,
                    "--eval_device",
                    eval_device,
                    "--eval_label",
                    evaluation_label,
                    "--num_envs",
                    str(args.num_envs),
                    "--t_end",
                    str(args.duration),
                    "--seed",
                    str(args.seed),
                    "--reset_mode",
                    reset_mode,
                    "--record_tracking",
                    "--record_policy_io",
                    "--command_profile",
                    "go2",
                    "--settling_time",
                    str(args.settling_time),
                    "--contact_threshold",
                    str(args.contact_threshold),
                    "--out",
                    str(output),
                ]
                print(" ".join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
