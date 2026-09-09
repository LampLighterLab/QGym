#!/usr/bin/env python3
"""Compare policy-observation artifacts from two controlled evaluations.

Both inputs must be produced by ``scripts/eval_policy.py --record_policy_io``
from the same checkpoint and evaluation protocol. The report uses samples for
which the corresponding robot is still in its first episode in both runs.

Example:

    uv run --frozen python scripts/compare_policy_observations.py \
        --reference logs/observation_transfer_audit/vsim.npz \
        --candidate logs/observation_transfer_audit/mujoco_warp.npz \
        --out logs/observation_transfer_audit/comparison
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gym.utils.policy_io import first_episode_mask


MATCHED_PROTOCOL_KEYS = (
    "task",
    "checkpoint_sha256",
    "checkpoint_iteration",
    "reset_mode",
    "command_profile",
    "num_envs",
    "duration_s",
    "seed",
    "ctrl_hz",
    "domain_randomization",
    "actor_observation_fields",
    "actor_observation_names",
    "actor_observation_scales",
    "critic_observation_fields",
    "critic_observation_names",
    "critic_observation_scales",
    "eval_commands",
    "phase_frequency_hz",
    "contact_friction",
    "stiffness_scale",
    "damping_scale",
    "link_mass_scale",
)


def _json_value(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_matched_protocol(reference, candidate):
    """Fail if two external artifacts do not describe the same experiment."""
    for key in MATCHED_PROTOCOL_KEYS:
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"evaluation artifacts differ in {key!r}")


def joint_first_episode_mask(reference, candidate):
    return first_episode_mask(reference["terminated"]) & first_episode_mask(
        candidate["terminated"]
    )


def _normalized_gap(value, scale):
    if scale > 0.0:
        return value / scale
    if value == 0.0:
        return 0.0
    return float("inf")


def component_statistics(reference, candidate, names, scales, valid):
    """Return per-component paired and marginal distribution diagnostics."""
    quantiles = np.linspace(0.0, 1.0, 101)
    rows = []
    for index, (name, scale) in enumerate(zip(names, scales, strict=True)):
        reference_values = reference[..., index][valid]
        candidate_values = candidate[..., index][valid]
        reference_std = float(np.std(reference_values))
        candidate_std = float(np.std(candidate_values))
        pooled_std = np.sqrt(0.5 * (reference_std**2 + candidate_std**2)).item()
        mean_difference = float(
            np.abs(np.mean(reference_values) - np.mean(candidate_values))
        )
        quantile_difference = float(
            np.mean(
                np.abs(
                    np.quantile(reference_values, quantiles)
                    - np.quantile(candidate_values, quantiles)
                )
            )
        )
        paired_rmse = float(
            np.sqrt(np.mean(np.square(reference_values - candidate_values)))
        )
        rows.append(
            {
                "name": str(name),
                "scale": float(scale),
                "reference_mean": float(np.mean(reference_values)),
                "candidate_mean": float(np.mean(candidate_values)),
                "reference_std": reference_std,
                "candidate_std": candidate_std,
                "mean_gap_pooled_std": _normalized_gap(mean_difference, pooled_std),
                "quantile_gap_pooled_std": _normalized_gap(
                    quantile_difference, pooled_std
                ),
                "paired_rmse": paired_rmse,
                "paired_rmse_task_units": paired_rmse * float(scale),
                "initial_reference_mean": float(np.mean(reference[0, :, index])),
                "initial_candidate_mean": float(np.mean(candidate[0, :, index])),
                "initial_max_abs_difference": float(
                    np.max(np.abs(reference[0, :, index] - candidate[0, :, index]))
                ),
            }
        )
    return rows


def field_statistics(fields, components):
    rows = []
    for field in fields:
        field = str(field)
        members = [
            row
            for row in components
            if row["name"] == field or row["name"].startswith(f"{field}.")
        ]
        rows.append(
            {
                "field": field,
                "components": len(members),
                "max_quantile_gap_pooled_std": max(
                    row["quantile_gap_pooled_std"] for row in members
                ),
                "mean_quantile_gap_pooled_std": float(
                    np.mean([row["quantile_gap_pooled_std"] for row in members])
                ),
                "max_initial_abs_difference": max(
                    row["initial_max_abs_difference"] for row in members
                ),
            }
        )
    return rows


def timewise_rmse(reference, candidate, valid):
    values = np.empty(reference.shape[0], dtype=np.float64)
    for step in range(reference.shape[0]):
        difference = reference[step, valid[step]] - candidate[step, valid[step]]
        values[step] = np.sqrt(np.mean(np.square(difference)))
    return values


def _write_component_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def _plot_field_gaps(path, actor_fields):
    ordered = sorted(
        actor_fields,
        key=lambda row: row["max_quantile_gap_pooled_std"],
    )
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.barh(
        [row["field"] for row in ordered],
        [row["max_quantile_gap_pooled_std"] for row in ordered],
        color="#4472c4",
    )
    axis.set_xlabel("largest component quantile gap / pooled standard deviation")
    axis.set_title("Actor-observation distribution mismatch")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_time_divergence(path, time, rmse, valid_fraction):
    figure, left = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    right = left.twinx()
    left.plot(time, rmse, color="#c44e52", label="actor observation RMSE")
    right.plot(
        time,
        valid_fraction,
        color="#4c9f70",
        linestyle="--",
        label="paired robots still alive",
    )
    left.set_xlabel("time [s]")
    left.set_ylabel("paired RMSE in policy-input space", color="#c44e52")
    right.set_ylabel("paired first-episode fraction", color="#4c9f70")
    left.set_title("Cross-backend policy-input divergence")
    left.grid(alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _markdown_table(rows, columns, limit=None):
    selected = rows if limit is None else rows[:limit]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in selected:
        values = []
        for key, _ in columns:
            value = row[key]
            values.append(f"{value:.4g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path, summary, reference_path, candidate_path):
    sanity = summary["sanity"]
    worst = sorted(
        summary["actor_components"],
        key=lambda row: row["quantile_gap_pooled_std"],
        reverse=True,
    )
    initial = sorted(
        summary["actor_components"],
        key=lambda row: row["initial_max_abs_difference"],
        reverse=True,
    )
    fields = sorted(
        summary["actor_fields"],
        key=lambda row: row["max_quantile_gap_pooled_std"],
        reverse=True,
    )
    report = f"""# Cross-backend policy-observation audit

Reference: `{reference_path}`

Candidate: `{candidate_path}`

## Sanity checks

- Checkpoint and controlled protocol metadata match exactly.
- Both artifacts contain {sanity["steps"]} steps, {sanity["num_envs"]} robots,
  {sanity["actor_components"]} actor inputs, and
  {sanity["critic_components"]} critic inputs.
- Shared first-episode samples: {sanity["joint_first_episode_samples"]} /
  {sanity["total_samples"]} ({100.0 * sanity["joint_first_episode_fraction"]:.1f}%).
- Survival: reference {100.0 * sanity["reference_survival"]:.1f}%, candidate
  {100.0 * sanity["candidate_survival"]:.1f}%.
- Initial actor-input RMSE: {sanity["initial_actor_rmse"]:.6g}; maximum absolute
  component difference: {sanity["initial_actor_max_abs_difference"]:.6g}.
- Initial policy-action RMSE: {sanity["initial_policy_action_rmse"]:.6g}; maximum
  absolute action difference: {sanity["initial_policy_action_max_abs_difference"]:.6g}.
- Nominal robot mass: reference {sanity["reference_robot_mass_kg"]:.6g} kg,
  candidate {sanity["candidate_robot_mass_kg"]:.6g} kg
  ({100.0 * sanity["robot_mass_relative_difference"]:.3f}% difference).

The distribution metric compares 101 equally spaced marginal quantiles and
divides their mean absolute gap by the pooled standard deviation. Zero means
matching marginal distributions; one means a typical quantile moved by one
pooled standard deviation. Samples after either robot's first termination are
excluded.

## Actor fields

{
        _markdown_table(
            fields,
            (
                ("field", "field"),
                ("max_quantile_gap_pooled_std", "max quantile gap / std"),
                ("mean_quantile_gap_pooled_std", "mean quantile gap / std"),
                ("max_initial_abs_difference", "initial max abs diff"),
            ),
        )
    }

## Largest initial mismatches

{
        _markdown_table(
            initial,
            (
                ("name", "component"),
                ("initial_max_abs_difference", "initial max abs diff"),
                ("initial_reference_mean", "initial reference mean"),
                ("initial_candidate_mean", "initial candidate mean"),
            ),
            limit=12,
        )
    }

## Largest rollout distribution mismatches

{
        _markdown_table(
            worst,
            (
                ("name", "component"),
                ("quantile_gap_pooled_std", "quantile gap / std"),
                ("mean_gap_pooled_std", "mean gap / std"),
                ("reference_std", "reference std"),
                ("candidate_std", "candidate std"),
            ),
            limit=20,
        )
    }

## Plots

- `actor_field_distribution_gap.png`: worst marginal mismatch within each
  configured actor field.
- `actor_observation_divergence.png`: paired policy-input RMSE through time,
  alongside the fraction of robot pairs still in their first episode.

Complete actor and critic component statistics are in `comparison.json` and
the two CSV files.
"""
    path.write_text(report)


def compare(reference_path, candidate_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (
        np.load(reference_path, allow_pickle=False) as reference,
        np.load(candidate_path, allow_pickle=False) as candidate,
    ):
        validate_matched_protocol(reference, candidate)
        valid = joint_first_episode_mask(reference, candidate)

        actor_components = component_statistics(
            reference["actor_observations"],
            candidate["actor_observations"],
            reference["actor_observation_names"],
            reference["actor_observation_scales"],
            valid,
        )
        critic_components = component_statistics(
            reference["critic_observations"],
            candidate["critic_observations"],
            reference["critic_observation_names"],
            reference["critic_observation_scales"],
            valid,
        )
        actor_fields = field_statistics(
            reference["actor_observation_fields"], actor_components
        )
        critic_fields = field_statistics(
            reference["critic_observation_fields"], critic_components
        )

        initial_actor_difference = (
            reference["actor_observations"][0] - candidate["actor_observations"][0]
        )
        initial_action_difference = (
            reference["policy_actions"][0] - candidate["policy_actions"][0]
        )
        reference_mass = float(np.mean(reference["robot_mass_kg"]))
        candidate_mass = float(np.mean(candidate["robot_mass_kg"]))
        summary = {
            "reference": str(reference_path.resolve()),
            "candidate": str(candidate_path.resolve()),
            "reference_label": str(reference["eval_label"]),
            "candidate_label": str(candidate["eval_label"]),
            "protocol": {
                key: _json_value(reference[key])
                for key in MATCHED_PROTOCOL_KEYS
                if reference[key].ndim == 0
            },
            "sanity": {
                "steps": int(reference["actor_observations"].shape[0]),
                "num_envs": int(reference["num_envs"]),
                "actor_components": int(reference["actor_observations"].shape[2]),
                "critic_components": int(reference["critic_observations"].shape[2]),
                "joint_first_episode_samples": int(np.count_nonzero(valid)),
                "total_samples": int(valid.size),
                "joint_first_episode_fraction": float(np.mean(valid)),
                "reference_survival": float(np.mean(reference["survived"])),
                "candidate_survival": float(np.mean(candidate["survived"])),
                "initial_actor_rmse": float(
                    np.sqrt(np.mean(np.square(initial_actor_difference)))
                ),
                "initial_actor_max_abs_difference": float(
                    np.max(np.abs(initial_actor_difference))
                ),
                "initial_policy_action_rmse": float(
                    np.sqrt(np.mean(np.square(initial_action_difference)))
                ),
                "initial_policy_action_max_abs_difference": float(
                    np.max(np.abs(initial_action_difference))
                ),
                "reference_robot_mass_kg": reference_mass,
                "candidate_robot_mass_kg": candidate_mass,
                "robot_mass_relative_difference": abs(reference_mass - candidate_mass)
                / reference_mass,
            },
            "actor_fields": actor_fields,
            "critic_fields": critic_fields,
            "actor_components": actor_components,
            "critic_components": critic_components,
        }

        time = np.arange(valid.shape[0]) / float(reference["ctrl_hz"])
        actor_rmse = timewise_rmse(
            reference["actor_observations"],
            candidate["actor_observations"],
            valid,
        )
        _plot_field_gaps(output_dir / "actor_field_distribution_gap.png", actor_fields)
        _plot_time_divergence(
            output_dir / "actor_observation_divergence.png",
            time,
            actor_rmse,
            np.mean(valid, axis=1),
        )

    (output_dir / "comparison.json").write_text(json.dumps(summary, indent=2))
    _write_component_csv(output_dir / "actor_components.csv", actor_components)
    _write_component_csv(output_dir / "critic_components.csv", critic_components)
    write_report(output_dir / "report.md", summary, reference_path, candidate_path)
    return summary


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main():
    args = get_args()
    summary = compare(args.reference, args.candidate, args.out)
    sanity = summary["sanity"]
    print(
        f"initial actor RMSE {sanity['initial_actor_rmse']:.4f}; "
        f"survival {100.0 * sanity['reference_survival']:.1f}% / "
        f"{100.0 * sanity['candidate_survival']:.1f}%"
    )
    print(f"wrote {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
