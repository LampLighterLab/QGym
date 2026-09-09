#!/usr/bin/env python3
"""Validate one policy across Q2 backends with and without friction DR.

The six evaluation cells run in fresh processes because Warp and VSim choose
their physical-parameter topology during setup.  DR-on uses the same crossed
friction grid for every backend: each of the ten Go2 command cases receives
ten evenly spaced coefficients.  This separates backend differences from RNG
differences.

    uv run --env-file .env.vsim scripts/validate_domain_randomization.py
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "logs/go2trot/Aug11_23-14-45_/model_1000.pt"
DEFAULT_OUTPUT = (
    REPO_ROOT / "logs/domain_randomization_validation/Aug11_23-14-45_model_1000"
)


@dataclass(frozen=True)
class Backend:
    label: str
    backend: str
    device: str


BACKENDS = (
    Backend("mujoco", "mujoco", "cpu"),
    Backend("mjx", "mujoco", "cuda:0"),
    Backend("vsim", "vsim", "cuda:0"),
)
DR_MODES = ("off", "on")
BACKEND_PAIRS = (("mujoco", "mjx"), ("mujoco", "vsim"), ("mjx", "vsim"))

# Deliberate physical/task-level scorecard. Raw artifacts retain every metric.
HEADLINE_METRICS = (
    ("mean_reward", "reward", "reward/step"),
    ("survived", "survival", "ratio"),
    ("ep_len", "duration", "s"),
    ("metric_tracking_vx_rmse", "vx RMSE", "m/s"),
    ("metric_tracking_vy_rmse", "vy RMSE", "m/s"),
    ("metric_tracking_yaw_rmse", "yaw RMSE", "rad/s"),
    ("metric_base_tilt_rms", "base tilt", "deg"),
    ("metric_base_height_std", "height std", "m"),
    ("metric_foot_slip_speed_rms", "foot slip", "m/s"),
    ("metric_grf_balance_cv", "GRF balance CV", "ratio"),
    ("metric_torque_utilization_rms", "torque utilization", "ratio"),
)


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--mujoco-njmax", type=int, default=256)
    parser.add_argument("--friction-range", type=float, nargs=2, default=(0.5, 1.0))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete cell artifacts and rerun only missing cells",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="only regenerate summary.json and report.md from existing artifacts",
    )
    return parser.parse_args(argv)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args):
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def cell_label(backend, dr_mode):
    return f"{backend.label}_dr_{dr_mode}"


def cell_command(args, backend, dr_mode, artifact):
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/eval_policy.py"),
        "--task",
        "go2trot",
        "--ckpt",
        str(args.checkpoint.resolve()),
        "--train_label",
        "Aug11_23-14-45_model_1000",
        "--eval_backend",
        backend.backend,
        "--eval_device",
        backend.device,
        "--eval_label",
        backend.label,
        "--num_envs",
        str(args.num_envs),
        "--t_end",
        str(args.duration),
        "--seed",
        str(args.seed),
        "--reset_mode",
        "reset_to_basic",
        "--command_profile",
        "go2",
        "--record_tracking",
        "--original_cfg",
        "--mujoco_njmax",
        str(args.mujoco_njmax),
        "--contact-friction-dr",
        dr_mode,
        "--out",
        str(artifact),
    ]
    if dr_mode == "on":
        command.extend(
            [
                "--contact_friction_grid",
                str(args.friction_range[0]),
                str(args.friction_range[1]),
            ]
        )
    return command


def _scalar(artifact, key):
    return np.asarray(artifact[key]).item()


def validate_artifact(path, args, backend, dr_mode):
    with np.load(path, allow_pickle=False) as artifact:
        if _scalar(artifact, "task") != "go2trot":
            raise ValueError(f"{path}: wrong task")
        if _scalar(artifact, "eval_label") != backend.label:
            raise ValueError(f"{path}: wrong backend label")
        if int(_scalar(artifact, "num_envs")) != args.num_envs:
            raise ValueError(f"{path}: wrong environment count")
        if float(_scalar(artifact, "duration_s")) != args.duration:
            raise ValueError(f"{path}: wrong duration")
        if int(_scalar(artifact, "seed")) != args.seed:
            raise ValueError(f"{path}: wrong seed")
        if _scalar(artifact, "contact_friction_dr") != dr_mode:
            raise ValueError(f"{path}: wrong DR mode")
        if not bool(_scalar(artifact, "original_cfg")):
            raise ValueError(f"{path}: did not use the saved run configs")
        if int(_scalar(artifact, "checkpoint_iteration")) != 1000:
            raise ValueError(f"{path}: wrong checkpoint iteration")
        if int(_scalar(artifact, "mujoco_njmax")) != args.mujoco_njmax:
            raise ValueError(f"{path}: wrong MuJoCo constraint capacity")

        friction = np.asarray(artifact["contact_friction"])
        if friction.shape != (args.num_envs,):
            raise ValueError(f"{path}: wrong friction shape {friction.shape}")
        if dr_mode == "off":
            np.testing.assert_allclose(friction, 1.0, rtol=0.0, atol=1e-6)
        else:
            command_cases = np.asarray(artifact["command_case"])
            expected = np.linspace(
                args.friction_range[0],
                args.friction_range[1],
                args.num_envs // len(np.unique(command_cases)),
                dtype=np.float32,
            )
            for command_case in np.unique(command_cases):
                np.testing.assert_allclose(
                    friction[command_cases == command_case],
                    expected,
                    rtol=0.0,
                    atol=1e-6,
                )
        for key, _, _ in HEADLINE_METRICS:
            values = np.asarray(artifact[key])
            if values.shape != (args.num_envs,):
                raise ValueError(f"{path}: {key} has shape {values.shape}")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_cells(args):
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for backend in BACKENDS:
        for dr_mode in DR_MODES:
            label = cell_label(backend, dr_mode)
            artifact = args.output / f"{label}.npz"
            log_path = args.output / f"{label}.log"
            command = cell_command(args, backend, dr_mode, artifact)
            record = {
                "label": label,
                "backend": asdict(backend),
                "dr_mode": dr_mode,
                "artifact": str(artifact.resolve()),
                "log": str(log_path.resolve()),
                "command": command,
            }
            if args.resume and artifact.is_file():
                validate_artifact(artifact, args, backend, dr_mode)
                record["status"] = "reused"
                records.append(record)
                print(f"[{label}] reused {artifact}")
                continue

            print(f"\n[{label}] {' '.join(command)}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_file.write(line)
                returncode = process.wait()
            record["returncode"] = returncode
            if returncode == 0:
                if "overflow" in log_path.read_text(encoding="utf-8").lower():
                    raise RuntimeError(f"{label} emitted an overflow; see {log_path}")
                validate_artifact(artifact, args, backend, dr_mode)
                record["status"] = "complete"
            else:
                record["status"] = "failed"
            records.append(record)
            if returncode != 0:
                _write_manifest(args, records)
                raise RuntimeError(f"{label} failed; see {log_path}")
            _write_manifest(args, records)
    return records


def _protocol(args):
    return {
        "task": "go2trot",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_iteration": 1000,
        "saved_config_source": str(
            (args.checkpoint.parent / "files/gym/envs").resolve()
        ),
        "num_envs": args.num_envs,
        "duration_s": args.duration,
        "seed": args.seed,
        "mujoco_njmax": args.mujoco_njmax,
        "reset_mode": "reset_to_basic",
        "command_profile": "go2",
        "dr_off_friction": 1.0,
        "dr_on_friction_grid": list(args.friction_range),
        "dr_on_design": (
            "crossed grid: each command case receives the same ten friction levels"
        ),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "dirty_paths": _git_value("status", "--short").splitlines(),
    }


def _write_manifest(args, records):
    _write_json(
        args.output / "manifest.json",
        {"schema_version": 1, "protocol": _protocol(args), "cells": records},
    )


def finite_values(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def descriptive_stats(values):
    values = finite_values(values)
    if not len(values):
        return {key: None for key in ("count", "mean", "std", "median", "p10", "p90")}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
    }


def wasserstein_distance(first, second):
    """Exact one-dimensional first Wasserstein distance."""
    first = np.sort(finite_values(first))
    second = np.sort(finite_values(second))
    if not len(first) or not len(second):
        return None
    support = np.unique(np.concatenate((first, second)))
    if len(support) == 1:
        return 0.0
    first_cdf = np.searchsorted(first, support[:-1], side="right") / len(first)
    second_cdf = np.searchsorted(second, support[:-1], side="right") / len(second)
    return float(np.sum(np.abs(first_cdf - second_cdf) * np.diff(support)))


def ks_distance(first, second):
    first = np.sort(finite_values(first))
    second = np.sort(finite_values(second))
    if not len(first) or not len(second):
        return None
    support = np.unique(np.concatenate((first, second)))
    first_cdf = np.searchsorted(first, support, side="right") / len(first)
    second_cdf = np.searchsorted(second, support, side="right") / len(second)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def compare_vectors(first, second, metric_key):
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    first_finite = finite_values(first)
    second_finite = finite_values(second)
    first_stats = descriptive_stats(first_finite)
    second_stats = descriptive_stats(second_finite)
    wasserstein = wasserstein_distance(first_finite, second_finite)
    ks = ks_distance(first_finite, second_finite)
    pooled_std = np.sqrt(0.5 * (first_stats["std"] ** 2 + second_stats["std"] ** 2))
    if pooled_std > 1e-12:
        normalized_wasserstein = wasserstein / pooled_std
    elif wasserstein <= 1e-12:
        normalized_wasserstein = 0.0
    else:
        normalized_wasserstein = None

    paired = np.isfinite(first) & np.isfinite(second)
    delta = second[paired] - first[paired]
    paired_mae = float(np.mean(np.abs(delta))) if len(delta) else None
    paired_rmse = float(np.sqrt(np.mean(delta**2))) if len(delta) else None
    mean_delta = float(np.mean(delta)) if len(delta) else None
    marginal_mean_delta = second_stats["mean"] - first_stats["mean"]

    if metric_key == "survived":
        magnitude = abs(second_stats["mean"] - first_stats["mean"])
        thresholds = (0.02, 0.05, 0.10)
    else:
        magnitude = normalized_wasserstein
        thresholds = (0.10, 0.25, 0.50)
    if magnitude is None:
        agreement = "large"
    elif magnitude <= thresholds[0]:
        agreement = "close"
    elif magnitude <= thresholds[1]:
        agreement = "small"
    elif magnitude <= thresholds[2]:
        agreement = "moderate"
    else:
        agreement = "large"

    return {
        "first": first_stats,
        "second": second_stats,
        "marginal_mean_delta_second_minus_first": marginal_mean_delta,
        "mean_delta_second_minus_first": mean_delta,
        "wasserstein": wasserstein,
        "normalized_wasserstein": normalized_wasserstein,
        "ks": ks,
        "paired_count": int(np.count_nonzero(paired)),
        "paired_mae": paired_mae,
        "paired_rmse": paired_rmse,
        "agreement": agreement,
    }


def _artifact_vectors(path):
    with np.load(path, allow_pickle=False) as artifact:
        return {
            key: np.asarray(artifact[key], dtype=np.float64)
            for key, _, _ in HEADLINE_METRICS
        } | {"contact_friction": np.asarray(artifact["contact_friction"])}


def _pair_summary(metric_rows):
    normalized = [
        row["normalized_wasserstein"]
        for row in metric_rows.values()
        if row["normalized_wasserstein"] is not None
    ]
    counts = {name: 0 for name in ("close", "small", "moderate", "large")}
    for row in metric_rows.values():
        counts[row["agreement"]] += 1
    return {
        "median_normalized_wasserstein": (
            float(np.median(normalized)) if normalized else None
        ),
        "max_normalized_wasserstein": max(normalized) if normalized else None,
        "median_ks": float(np.median([row["ks"] for row in metric_rows.values()])),
        "agreement_counts": counts,
    }


def analyze(args):
    artifacts = {}
    vectors = {}
    cells = {}
    for backend in BACKENDS:
        for dr_mode in DR_MODES:
            label = cell_label(backend, dr_mode)
            artifact = args.output / f"{label}.npz"
            validate_artifact(artifact, args, backend, dr_mode)
            artifacts[label] = str(artifact.resolve())
            vectors[label] = _artifact_vectors(artifact)
            cells[label] = {
                key: descriptive_stats(vectors[label][key])
                for key, _, _ in HEADLINE_METRICS
            }

    backend_comparisons = {}
    for dr_mode in DR_MODES:
        backend_comparisons[dr_mode] = {}
        for first_backend, second_backend in BACKEND_PAIRS:
            first = vectors[f"{first_backend}_dr_{dr_mode}"]
            second = vectors[f"{second_backend}_dr_{dr_mode}"]
            metric_rows = {
                key: compare_vectors(first[key], second[key], key)
                for key, _, _ in HEADLINE_METRICS
            }
            backend_comparisons[dr_mode][f"{first_backend}__{second_backend}"] = {
                "first_backend": first_backend,
                "second_backend": second_backend,
                "max_abs_friction_difference": float(
                    np.max(
                        np.abs(first["contact_friction"] - second["contact_friction"])
                    )
                ),
                "metrics": metric_rows,
                "summary": _pair_summary(metric_rows),
            }

    dr_effects = {}
    for backend in BACKENDS:
        off = vectors[f"{backend.label}_dr_off"]
        on = vectors[f"{backend.label}_dr_on"]
        metric_rows = {
            key: compare_vectors(off[key], on[key], key)
            for key, _, _ in HEADLINE_METRICS
        }
        dr_effects[backend.label] = {
            "metrics": metric_rows,
            "summary": _pair_summary(metric_rows),
        }

    summary = {
        "schema_version": 1,
        "protocol": _protocol(args),
        "interpretation": {
            "normalized_wasserstein": (
                "W1 divided by pooled sample standard deviation; undefined when "
                "both distributions are constant but unequal"
            ),
            "ks": "maximum absolute difference between empirical CDFs",
            "agreement_bands": {
                "continuous_normalized_wasserstein": {
                    "close": "<= 0.10",
                    "small": "(0.10, 0.25]",
                    "moderate": "(0.25, 0.50]",
                    "large": "> 0.50 or undefined unequal constants",
                },
                "survival_absolute_rate_difference": {
                    "close": "<= 2 percentage points",
                    "small": "(2, 5] percentage points",
                    "moderate": "(5, 10] percentage points",
                    "large": "> 10 percentage points",
                },
            },
            "scope": (
                "Descriptive single-checkpoint, single-seed validation. Agreement "
                "bands are effect-size labels, not significance tests."
            ),
        },
        "artifacts": artifacts,
        "cell_statistics": cells,
        "backend_comparisons": backend_comparisons,
        "dr_on_minus_off": dr_effects,
    }
    _write_json(args.output / "summary.json", summary)
    (args.output / "report.md").write_text(markdown_report(summary))
    return summary


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _fmt_mean_std(stats):
    return f"{_fmt(stats['mean'])} ± {_fmt(stats['std'])}"


def markdown_report(summary):
    protocol = summary["protocol"]
    lines = [
        "# Domain-randomization effect validation",
        "",
        f"Checkpoint: `{protocol['checkpoint']}` (`model_1000.pt`, SHA-256 "
        f"`{protocol['checkpoint_sha256'][:12]}…`).",
        "",
        f"Protocol: {protocol['num_envs']} robots per cell, "
        f"{protocol['duration_s']:g} s, deterministic basic reset, ten fixed Go2 "
        f"command cases. DR off uses μ=1.0. DR on crosses every command with ten "
        f"evenly spaced μ∈[{protocol['dr_on_friction_grid'][0]:g}, "
        f"{protocol['dr_on_friction_grid'][1]:g}]. MuJoCo constraint capacity is "
        f"set to {protocol['mujoco_njmax']} for this validation.",
        "",
        "The tables are descriptive. `nW1` is the first Wasserstein distance "
        "normalized by pooled standard deviation; `KS` is empirical-CDF distance. "
        "Agreement bands are close ≤0.10, small ≤0.25, moderate ≤0.50, and large "
        ">0.50. Survival instead uses 2/5/10 percentage-point bands.",
        "",
        "## DR-on versus DR-off summary",
        "",
        "| Backend | median nW1 | max nW1 | median KS | survival off→on | "
        "reward off→on | close | small | moderate | large |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {key: (label, unit) for key, label, unit in HEADLINE_METRICS}
    for backend, row in summary["dr_on_minus_off"].items():
        aggregate = row["summary"]
        metrics = row["metrics"]
        counts = aggregate["agreement_counts"]
        lines.append(
            f"| {backend} | "
            f"{_fmt(aggregate['median_normalized_wasserstein'])} | "
            f"{_fmt(aggregate['max_normalized_wasserstein'])} | "
            f"{_fmt(aggregate['median_ks'])} | "
            f"{100 * metrics['survived']['first']['mean']:.0f}%→"
            f"{100 * metrics['survived']['second']['mean']:.0f}% | "
            f"{_fmt(metrics['mean_reward']['first']['mean'])}→"
            f"{_fmt(metrics['mean_reward']['second']['mean'])} | "
            f"{counts['close']} | {counts['small']} | "
            f"{counts['moderate']} | {counts['large']} |"
        )

    lines.extend(
        [
            "",
            "## Detailed DR effect within each backend",
            "",
            "Marginal deltas are DR-on mean minus DR-off mean. They are not "
            "automatically improvements because most error metrics are "
            "lower-is-better.",
            "",
            "| Backend | Metric | off mean ± SD | on mean ± SD | marginal Δ | "
            "W1 | nW1 | KS | paired RMSE | effect |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for backend, result in summary["dr_on_minus_off"].items():
        for key, row in result["metrics"].items():
            label, unit = labels[key]
            lines.append(
                f"| {backend} | {label} ({unit}) | "
                f"{_fmt_mean_std(row['first'])} | "
                f"{_fmt_mean_std(row['second'])} | "
                f"{_fmt(row['marginal_mean_delta_second_minus_first'])} | "
                f"{_fmt(row['wasserstein'])} | "
                f"{_fmt(row['normalized_wasserstein'])} | {_fmt(row['ks'])} | "
                f"{_fmt(row['paired_rmse'])} | {row['agreement']} |"
            )

    lines.extend(
        [
            "",
            "## Supplemental cross-backend agreement",
            "",
            "| DR | Backend pair | median nW1 | max nW1 | median KS | close | "
            "small | moderate | large |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dr_mode, pairs in summary["backend_comparisons"].items():
        for pair, row in pairs.items():
            aggregate = row["summary"]
            counts = aggregate["agreement_counts"]
            lines.append(
                f"| {dr_mode} | {pair.replace('__', ' vs ')} | "
                f"{_fmt(aggregate['median_normalized_wasserstein'])} | "
                f"{_fmt(aggregate['max_normalized_wasserstein'])} | "
                f"{_fmt(aggregate['median_ks'])} | {counts['close']} | "
                f"{counts['small']} | {counts['moderate']} | {counts['large']} |"
            )

    for dr_mode, pairs in summary["backend_comparisons"].items():
        lines.extend(["", f"### Detailed backend comparison — DR {dr_mode}", ""])
        lines.extend(
            [
                "| Pair | Metric | mean ± SD A | mean ± SD B | paired Δ | W1 | "
                "nW1 | KS | paired RMSE | agreement |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for pair, pair_row in pairs.items():
            for key, row in pair_row["metrics"].items():
                label, unit = labels[key]
                lines.append(
                    f"| {pair.replace('__', '→')} | {label} ({unit}) | "
                    f"{_fmt_mean_std(row['first'])} | "
                    f"{_fmt_mean_std(row['second'])} | "
                    f"{_fmt(row['mean_delta_second_minus_first'])} | "
                    f"{_fmt(row['wasserstein'])} | "
                    f"{_fmt(row['normalized_wasserstein'])} | {_fmt(row['ks'])} | "
                    f"{_fmt(row['paired_rmse'])} | {row['agreement']} |"
                )

    lines.extend(
        [
            "",
            "Raw per-environment results and trajectories are retained in the six "
            "NPZ files. This one checkpoint/seed measures the effect of the tested "
            "friction distribution; it is not a policy-quality or "
            "statistical-generalization claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = get_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.num_envs <= 0 or args.num_envs % 10:
        raise ValueError("num-envs must be a positive multiple of ten")
    if args.duration <= 0:
        raise ValueError("duration must be positive")
    if args.mujoco_njmax <= 0:
        raise ValueError("mujoco-njmax must be positive")
    if not 0 <= args.friction_range[0] <= args.friction_range[1]:
        raise ValueError("friction-range must satisfy 0 <= LOW <= HIGH")

    if not args.analyze_only:
        records = run_cells(args)
        _write_manifest(args, records)
    summary = analyze(args)
    print(f"\nwrote {args.output / 'summary.json'}")
    print(f"wrote {args.output / 'report.md'}")
    for backend, row in summary["dr_on_minus_off"].items():
        result = row["summary"]
        counts = result["agreement_counts"]
        print(
            f"{backend:>6} DR off vs on: "
            f"median nW1={_fmt(result['median_normalized_wasserstein'])}, "
            f"max={_fmt(result['max_normalized_wasserstein'])}; "
            f"close/small/moderate/large="
            f"{counts['close']}/{counts['small']}/"
            f"{counts['moderate']}/{counts['large']}"
        )


if __name__ == "__main__":
    main()
