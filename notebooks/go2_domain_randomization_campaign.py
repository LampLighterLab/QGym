"""Live progress and results for the full Go2Trot DR campaign.

Start or resume the campaign first:

    uv run --env-file .env.vsim \
        scripts/run_full_domain_randomization_campaign.py

Then open this report while it is running:

    uv run marimo edit notebooks/go2_domain_randomization_campaign.py

Set ``Q2_DR_CAMPAIGN_DIR`` to select a specific artifact directory.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    variant_order = ["off", "friction", "pd", "mass", "all"]
    variant_colors = {
        "off": "#5f6368",
        "friction": "#1f77b4",
        "pd": "#ff7f0e",
        "mass": "#2ca02c",
        "all": "#9467bd",
    }
    metric_defaults = [
        "survival",
        "tracking_vx_rmse",
        "base_tilt_rms",
        "foot_slip_speed_rms",
        "torque_utilization_rms",
        "grf_balance_cv",
    ]

    def direction_benefit(value, baseline, direction):
        if direction == "higher":
            return value - baseline
        if direction == "closer_to_one":
            return abs(baseline - 1.0) - abs(value - 1.0)
        return baseline - value

    return (
        Path,
        direction_benefit,
        json,
        metric_defaults,
        mo,
        np,
        os,
        pd,
        plt,
        variant_colors,
        variant_order,
    )


@app.cell
def _(Path, mo, os):
    _requested = os.environ.get("Q2_DR_CAMPAIGN_DIR")
    _paths = sorted(Path("logs").glob("dr_full_*"), reverse=True)
    _default_campaign = _paths[0].name if _paths else None
    if _requested:
        _requested_path = Path(_requested)
        if _requested_path not in _paths:
            _paths.insert(0, _requested_path)
        _default_campaign = _requested_path.name
    campaign_options = {path.name: str(path) for path in _paths}
    mo.stop(
        not campaign_options,
        mo.md(
            "**No full campaign directory exists yet.** Run the command in "
            "the notebook docstring first."
        ),
    )
    campaign_picker = mo.ui.dropdown(
        options=campaign_options,
        value=_default_campaign,
        label="campaign",
    )
    refresh = mo.ui.refresh(
        options=["10s", "30s", "1m"], default_interval="30s", label="refresh"
    )
    mo.hstack([campaign_picker, refresh], justify="start")
    return campaign_picker, refresh


@app.cell
def _(Path, campaign_picker, json, refresh):
    refresh.value
    campaign_root = Path(campaign_picker.value)
    manifest_path = campaign_root / "manifest.json"
    summary_path = campaign_root / "summary.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {"speed": [], "training": [], "evaluation": []}
    )
    return campaign_root, manifest, summary


@app.cell
def _(manifest, summary):
    _summary_section = {
        "speed": "speed",
        "train": "training",
        "eval": "evaluation",
    }
    summary_mismatch = {}
    for _stage, _section in _summary_section.items():
        _manifest_labels = {
            cell["label"]
            for cell in manifest.get("cells", {}).values()
            if cell["stage"] == _stage and cell["status"] == "complete"
        }
        _summary_labels = {
            row["label"] for row in summary.get(_section, []) if "label" in row
        }
        _missing = _manifest_labels - _summary_labels
        _extra = _summary_labels - _manifest_labels
        if _missing or _extra:
            summary_mismatch[_stage] = {
                "missing": len(_missing),
                "extra": len(_extra),
            }
    return (summary_mismatch,)


@app.cell
def _(campaign_root, manifest, mo, summary, summary_mismatch):
    _protocol = manifest.get("protocol", {})
    _git = manifest.get("git", {})
    _dirty = "⚠️ dirty source snapshot" if _git.get("dirty") else "clean commit"
    _statuses = [cell["status"] for cell in manifest.get("cells", {}).values()]
    _stop_reason = manifest.get("stop_reason")
    _rejected = [
        cell["label"]
        for cell in manifest.get("cells", {}).values()
        if cell.get("error") == "completed artifact failed resume validation"
    ]
    if _statuses and all(status == "complete" for status in _statuses):
        _state = "complete"
    elif _stop_reason:
        _state = "stopped / incomplete"
    elif "failed" in _statuses:
        _state = "incomplete"
    elif "running" in _statuses:
        _state = "running"
    elif "complete" in _statuses:
        _state = "partial / idle"
    else:
        _state = "not started"
    _protocol = manifest.get("protocol", {})
    _header = mo.md(
        f"""
        # Go2Trot domain-randomization campaign

        **Campaign:** `{campaign_root}`

        **State:** `{_state}` · {_dirty} · commit
        `{_git.get("commit", "unknown")[:10]}`

        Training bundles: **{", ".join(_protocol.get("bundles", {}))}**.
        Excluded training cells: **{", ".join(_protocol.get("excluded_training", [])) or "none"}**.
        Every policy uses the same rollout and optimizer geometry. Results are
        paired by seed. Seed-level policy uncertainty below is descriptive
        sample standard deviation, not a confidence interval.
        """
    )
    _stop_status = (
        mo.callout(
            mo.md(
                f"**Collection stopped intentionally.** {_stop_reason} "
                "Completed cells remain valid evidence; pending cells were not "
                "used in the result panels."
            ),
            kind="warn",
        )
        if _stop_reason
        else mo.md("")
    )
    _integrity_status = (
        mo.callout(
            mo.md(
                "**Evidence-integrity exclusion:** "
                f"`{', '.join(_rejected)}` failed final artifact validation and "
                "is excluded from summaries and policy comparisons."
            ),
            kind="danger",
        )
        if _rejected
        else mo.md("")
    )
    if summary_mismatch:
        _details = "; ".join(
            f"{stage}: {counts['missing']} missing, {counts['extra']} extra"
            for stage, counts in summary_mismatch.items()
        )
        _summary_status = mo.callout(
            mo.md(
                "**Result summary is not synchronized with the manifest.** "
                "Progress above is current, but result panels below are incomplete "
                f"or inconsistent ({_details}). `summary.json` was last updated "
                f"at `{summary.get('updated_at', 'unknown')}`."
            ),
            kind="warn",
        )
    else:
        _summary_status = mo.md(
            "**Result summary:** synchronized with all completed manifest cells "
            f"(updated `{summary.get('updated_at', 'unknown')}`)."
        )
    mo.vstack([_header, _stop_status, _integrity_status, _summary_status])
    return


@app.cell
def _(campaign_root, json, manifest, mo, np, pd, plt):
    _rows = []
    for _cell in manifest.get("cells", {}).values():
        _rows.append({"stage": _cell["stage"], "status": _cell["status"]})
    progress_frame = pd.DataFrame(_rows)
    if progress_frame.empty:
        progress_figure = mo.md("Campaign plan has not been created.")
    else:
        _statuses = ["complete", "running", "failed", "pending"]
        _colors = ["#2ca02c", "#1f77b4", "#d62728", "#d9d9d9"]
        _counts = (
            progress_frame.groupby(["stage", "status"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=_statuses, fill_value=0)
        )
        _fig, _ax = plt.subplots(figsize=(10, 2.8))
        _left = np.zeros(len(_counts))
        for _status, _color in zip(_statuses, _colors):
            _values = _counts[_status].to_numpy()
            _ax.barh(_counts.index, _values, left=_left, color=_color, label=_status)
            _left += _values
        for _index, _total in enumerate(_left):
            _complete = int(_counts.iloc[_index]["complete"])
            _ax.text(_total + max(_left) * 0.01, _index, f"{_complete}/{int(_total)}")
        _ax.set_xlabel("campaign cells")
        _ax.set_title("Live completion by stage")
        _ax.legend(ncol=4, frameon=False, loc="lower right")
        _fig.tight_layout()
        progress_figure = _fig
    _running_training = []
    for _cell in manifest.get("cells", {}).values():
        if _cell["stage"] != "train" or _cell["status"] != "running":
            continue
        _vitals = sorted(
            (campaign_root / "training_runs" / _cell["label"]).glob("*/vitals.jsonl")
        )
        _latest = 0
        if _vitals:
            _contents = _vitals[-1].read_text(encoding="utf-8")
            _lines = _contents.splitlines()
            if _contents and not _contents.endswith("\n"):
                _lines = _lines[:-1]
            if _lines:
                _latest = int(json.loads(_lines[-1])["iteration"])
        _running_training.append(
            {
                "training cell": _cell["label"],
                "latest iteration": _latest,
                "target": manifest.get("protocol", {}).get("train_iterations"),
            }
        )
    mo.vstack(
        [
            progress_figure,
            (
                pd.DataFrame(_running_training)
                if _running_training
                else mo.md("No training cell is currently running.")
            ),
        ]
    )
    return (progress_frame,)


@app.cell
def _(manifest, mo, pd):
    _protocol = manifest.get("protocol", {})
    _cells = list(manifest.get("cells", {}).values())
    _failed = [
        {
            "stage": cell["stage"],
            "label": cell["label"],
            "error": cell.get("error"),
            "log": cell.get("log"),
        }
        for cell in _cells
        if cell["status"] == "failed"
    ]
    _design = pd.DataFrame(
        [
            ["seeds", _protocol.get("seeds")],
            ["excluded training", _protocol.get("excluded_training", [])],
            [
                "evaluation schedule",
                _protocol.get("evaluation_schedule", "after_all_training"),
            ],
            ["training iterations", _protocol.get("train_iterations")],
            ["checkpoints", _protocol.get("checkpoints")],
            ["environments", _protocol.get("train_num_envs")],
            ["rollout samples", _protocol.get("rollout_size")],
            ["optimizer batch", _protocol.get("optimizer_batch_size")],
            ["gradient steps", _protocol.get("max_gradient_steps")],
            ["evaluation domains", len(_protocol.get("eval_domains", {}))],
        ],
        columns=["protocol field", "value"],
    )
    _domains = pd.DataFrame(
        [
            {
                "domain": name,
                "mode": definition["mode"],
                "friction": definition.get("contact_friction", "—"),
                "stiffness": definition.get("stiffness_scale", "—"),
                "damping": definition.get("damping_scale", "—"),
                "link mass": definition.get("link_mass_scale", "—"),
            }
            for name, definition in _protocol.get("eval_domains", {}).items()
        ]
    )
    mo.vstack(
        [
            mo.md(
                "## Campaign design\n\nIntermediate checkpoints are evaluated on "
                "their native backend under nominal and combined in-range physics. "
                f"The final checkpoint is evaluated in {len(_domains)} domains on "
                f"{len(_protocol.get('backends', []))} backends. "
                "Missing cells are never interpolated."
            ),
            _design,
            mo.accordion(
                {
                    f"Evaluation domains ({len(_domains)})": _domains,
                    f"Failed cells ({len(_failed)})": (
                        pd.DataFrame(_failed)
                        if _failed
                        else mo.md("No failed cells recorded.")
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _(manifest, mo, summary):
    _counts = sorted({row["num_envs"] for row in summary.get("speed", [])})
    speed_env_count = mo.ui.dropdown(
        options=_counts,
        value=_counts[-1] if _counts else None,
        label="speed environment count",
    )
    speed_env_count
    return (speed_env_count,)


@app.cell
def _(
    mo,
    np,
    pd,
    plt,
    speed_env_count,
    summary,
    variant_colors,
    variant_order,
):
    _rows = []
    for _cell in summary.get("speed", []):
        if _cell["num_envs"] != speed_env_count.value:
            continue
        for _profile in _cell["profiles"]:
            _rows.append(
                {
                    "backend": _cell["backend"],
                    "bundle": _cell["bundle"],
                    "profile": _profile["profile"],
                    "throughput": _profile["median_env_control_steps_per_s"],
                    "p10": _profile["p10_env_control_steps_per_s"],
                    "p90": _profile["p90_env_control_steps_per_s"],
                }
            )
    speed_frame = pd.DataFrame(_rows)
    if speed_frame.empty:
        speed_plot = mo.md("Speed results have not arrived for this size yet.")
    else:
        _profiles = ["none", "timeout", "all"]
        _backends = sorted(speed_frame["backend"].unique())
        _fig, _axes = plt.subplots(
            len(_backends),
            len(_profiles),
            figsize=(12, 2.8 * len(_backends)),
            squeeze=False,
        )
        for _row, _backend in enumerate(_backends):
            for _column, _profile in enumerate(_profiles):
                _ax = _axes[_row, _column]
                _subset = speed_frame[
                    (speed_frame["backend"] == _backend)
                    & (speed_frame["profile"] == _profile)
                ]
                _off = _subset.loc[_subset["bundle"] == "off", "throughput"]
                if _off.empty:
                    continue
                _baseline = float(_off.iloc[0])
                for _index, _bundle in enumerate(variant_order):
                    _match = _subset[_subset["bundle"] == _bundle]
                    if _match.empty:
                        continue
                    _value = float(_match.iloc[0]["throughput"]) / _baseline
                    _low = float(_match.iloc[0]["p10"]) / _baseline
                    _high = float(_match.iloc[0]["p90"]) / _baseline
                    _ax.errorbar(
                        _value,
                        _index,
                        xerr=[[max(0.0, _value - _low)], [max(0.0, _high - _value)]],
                        fmt="o",
                        color=variant_colors[_bundle],
                    )
                _ax.axvline(1.0, color="black", linewidth=1, alpha=0.5)
                _ax.set_yticks(range(len(variant_order)), variant_order)
                _ax.set_title(f"{_backend} · {_profile}")
                _ax.set_xlabel("throughput / DR-off")
                _ax.grid(axis="x", alpha=0.2)
        _fig.suptitle(f"Runtime cost at {speed_env_count.value:,} environments", y=1.01)
        _fig.tight_layout()
        speed_plot = _fig
    mo.vstack(
        [
            mo.md(
                "## Runtime cost\n\nA value of 1.0 matches nominal throughput; "
                "lower is slower. `none` isolates steady stepping, `timeout` uses "
                "the production episode-reset rate, and `all` is reset-path stress. "
                "Whiskers are the p10–p90 range of timed repeats."
            ),
            speed_plot,
        ]
    )
    return (speed_frame,)


@app.cell
def _(metric_defaults, mo, summary):
    _backends = sorted({row["backend"] for row in summary.get("training", [])})
    training_backend = mo.ui.dropdown(
        options=_backends,
        value=_backends[0] if _backends else None,
        label="train backend",
    )
    training_metric = mo.ui.dropdown(
        options={
            "reward (diagnostic)": "rewards/total_rewards",
            "episode duration": "episode_time",
            "throughput": "steps_per_s",
        },
        value="episode duration",
        label="learning curve",
    )
    mo.hstack([training_backend, training_metric], justify="start")
    return metric_defaults, training_backend, training_metric


@app.cell
def _(
    mo,
    np,
    plt,
    summary,
    training_backend,
    training_metric,
    variant_colors,
    variant_order,
):
    _records = [
        row
        for row in summary.get("training", [])
        if row["backend"] == training_backend.value
    ]
    if not _records:
        learning_plot = mo.md("No completed training cells for this backend yet.")
    else:
        _fig, _ax = plt.subplots(figsize=(10, 4.5))
        for _bundle in variant_order:
            _bundle_records = [
                record for record in _records if record["bundle"] == _bundle
            ]
            _iterations = sorted(
                {
                    point["iteration"]
                    for record in _bundle_records
                    for point in record["curve"]
                }
            )
            _curves = []
            for _record in _bundle_records:
                _by_iteration = {
                    point["iteration"]: point.get(training_metric.value, np.nan)
                    for point in _record["curve"]
                }
                _values = [
                    _by_iteration.get(iteration, np.nan) for iteration in _iterations
                ]
                _ax.plot(
                    _iterations,
                    _values,
                    color=variant_colors[_bundle],
                    alpha=0.2,
                    linewidth=1,
                )
                _curves.append(_values)
            if not _curves:
                continue
            _array = np.asarray(_curves, dtype=float)
            _x = np.asarray(_iterations)
            _mean = np.nanmean(_array, axis=0)
            _ax.plot(
                _x, _mean, color=variant_colors[_bundle], label=_bundle, linewidth=2
            )
            if len(_array) > 1:
                _counts = np.sum(np.isfinite(_array), axis=0)
                _std = np.full(len(_x), np.nan)
                _enough = _counts > 1
                _std[_enough] = np.nanstd(_array[:, _enough], axis=0, ddof=1)
                _ax.fill_between(
                    _x,
                    _mean - _std,
                    _mean + _std,
                    color=variant_colors[_bundle],
                    alpha=0.12,
                )
        _ax.set_xlabel("training iteration")
        _ax.set_ylabel(training_metric.value)
        _ax.set_title(f"Learning progression on {training_backend.value}")
        _ax.legend(ncol=5, frameon=False)
        _ax.grid(alpha=0.2)
        _fig.tight_layout()
        learning_plot = _fig
    mo.vstack(
        [
            mo.md(
                "## Learning progression\n\nThin lines are individual seeds; bold "
                "lines are seed means and shading is one sample SD. Episode duration "
                "is the most useful early stability signal. Reward is shown only as "
                "the policy's training diagnostic, not as the robustness verdict."
            ),
            learning_plot,
        ]
    )
    return


@app.cell
def _(metric_defaults, mo, summary):
    _evaluations = summary.get("evaluation", [])
    _metrics = sorted(
        set(metric_defaults)
        & {name for row in _evaluations for name in row.get("metrics", {})}
    )
    _train_backends = sorted({row["train_backend"] for row in _evaluations})
    _eval_backends = sorted({row["eval_backend"] for row in _evaluations})
    robustness_metric = mo.ui.dropdown(
        options=_metrics,
        value="survival"
        if "survival" in _metrics
        else (_metrics[0] if _metrics else None),
        label="physical metric",
    )
    robustness_train_backend = mo.ui.dropdown(
        options=_train_backends,
        value=_train_backends[0] if _train_backends else None,
        label="training backend",
    )
    robustness_eval_backend = mo.ui.dropdown(
        options=_eval_backends,
        value=_eval_backends[0] if _eval_backends else None,
        label="evaluation backend",
    )
    mo.hstack(
        [
            robustness_metric,
            robustness_train_backend,
            robustness_eval_backend,
        ],
        justify="start",
    )
    return robustness_eval_backend, robustness_metric, robustness_train_backend


@app.cell
def _(
    mo,
    np,
    plt,
    robustness_metric,
    robustness_train_backend,
    summary,
    variant_colors,
    variant_order,
):
    _metric = robustness_metric.value
    _backend = robustness_train_backend.value
    _records = [
        row
        for row in summary.get("evaluation", [])
        if row["train_backend"] == _backend
        and row["eval_backend"] == _backend
        and row["domain"] in {"nominal", "combined_in"}
        and _metric in row["metrics"]
    ]
    if not _records:
        _checkpoint_plot = mo.md(
            "Intermediate physical evaluations are not available yet."
        )
    else:
        _domains = ["nominal", "combined_in"]
        _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        _checkpoints = summary.get("protocol", {}).get("checkpoints", [])
        for _ax, _domain in zip(_axes, _domains):
            for _bundle in variant_order:
                _selected = [
                    row
                    for row in _records
                    if row["bundle"] == _bundle and row["domain"] == _domain
                ]
                for _seed in sorted({row["seed"] for row in _selected}):
                    _seed_values = {
                        row["checkpoint_iteration"]: row["metrics"][_metric]["mean"]
                        for row in _selected
                        if row["seed"] == _seed
                    }
                    _ax.plot(
                        _checkpoints,
                        [
                            _seed_values.get(iteration, np.nan)
                            for iteration in _checkpoints
                        ],
                        color=variant_colors[_bundle],
                        alpha=0.18,
                        linewidth=1,
                    )
                _means = []
                _errors = []
                for _iteration in _checkpoints:
                    _values = np.asarray(
                        [
                            row["metrics"][_metric]["mean"]
                            for row in _selected
                            if row["checkpoint_iteration"] == _iteration
                        ],
                        dtype=float,
                    )
                    _means.append(float(np.mean(_values)) if len(_values) else np.nan)
                    _errors.append(
                        float(np.std(_values, ddof=1)) if len(_values) > 1 else 0.0
                    )
                if _checkpoints:
                    _ax.errorbar(
                        _checkpoints,
                        _means,
                        yerr=_errors,
                        color=variant_colors[_bundle],
                        marker="o",
                        linewidth=2,
                        capsize=2,
                        label=_bundle,
                    )
            _ax.set_title(_domain.replace("_", " "))
            _ax.set_xlabel("checkpoint iteration")
            _ax.grid(alpha=0.2)
        _definition = next(
            row["metrics"][_metric] for row in _records if _metric in row["metrics"]
        )
        _axes[0].set_ylabel(f"{_metric} [{_definition['unit']}]")
        _axes[-1].legend(ncol=2, frameon=False)
        _fig.suptitle(f"Native physical progression on {_backend}")
        _fig.tight_layout()
        _checkpoint_plot = _fig
    mo.vstack(
        [
            mo.md(
                "## Intermediate checkpoints\n\nThese are deterministic physical "
                "evaluations, not training reward. This progression uses each "
                "metric's environment mean. Thin lines are individual seeds; bold "
                "points are seed means with one sample-SD whiskers. "
                "Only completed checkpoints are drawn, so gaps remain visible."
            ),
            _checkpoint_plot,
        ]
    )
    return


@app.cell
def _(mo):
    robustness_statistic = mo.ui.dropdown(
        options={"mean": "mean", "worst decile": "worst_decile_mean"},
        value="mean",
        label="final/transfer environment statistic",
    )
    robustness_view = mo.ui.dropdown(
        options={"paired benefit vs off": "benefit", "absolute value": "absolute"},
        value="paired benefit vs off",
        label="final robustness view",
    )
    mo.hstack([robustness_statistic, robustness_view], justify="start")
    return robustness_statistic, robustness_view


@app.cell
def _(
    direction_benefit,
    mo,
    np,
    plt,
    robustness_eval_backend,
    robustness_metric,
    robustness_statistic,
    robustness_train_backend,
    robustness_view,
    summary,
    variant_order,
):
    _statistic = robustness_statistic.value
    _view = robustness_view.value
    _records = [
        row
        for row in summary.get("evaluation", [])
        if row["checkpoint_iteration"]
        == summary.get("protocol", {}).get("train_iterations")
        and row["train_backend"] == robustness_train_backend.value
        and row["eval_backend"] == robustness_eval_backend.value
        and robustness_metric.value in row["metrics"]
    ]
    _domains = list(summary.get("protocol", {}).get("eval_domains", {}))
    _bundles = variant_order[1:] if _view == "benefit" else variant_order
    _matrix = np.full((len(_bundles), len(_domains)), np.nan)
    _std = np.full_like(_matrix, np.nan)
    _counts = np.zeros_like(_matrix, dtype=int)
    _definition = next(
        (
            row["metrics"][robustness_metric.value]
            for row in _records
            if robustness_metric.value in row["metrics"]
        ),
        {"unit": "", "direction": "context"},
    )
    for _column, _domain in enumerate(_domains):
        _off = {
            row["seed"]: row["metrics"][robustness_metric.value]
            for row in _records
            if row["bundle"] == "off" and row["domain"] == _domain
        }
        for _row, _bundle in enumerate(_bundles):
            _values = []
            for _record in _records:
                if _record["bundle"] != _bundle or _record["domain"] != _domain:
                    continue
                _metric = _record["metrics"][robustness_metric.value]
                if _view == "benefit":
                    if _record["seed"] not in _off:
                        continue
                    _baseline = _off[_record["seed"]]
                    _values.append(
                        direction_benefit(
                            _metric[_statistic],
                            _baseline[_statistic],
                            _metric["direction"],
                        )
                    )
                else:
                    _values.append(_metric[_statistic])
            if _values:
                _matrix[_row, _column] = np.mean(_values)
                _std[_row, _column] = (
                    np.std(_values, ddof=1) if len(_values) > 1 else np.nan
                )
                _counts[_row, _column] = len(_values)
    if not np.isfinite(_matrix).any():
        robustness_plot = mo.md(
            "Final robustness results for this view are not available yet."
        )
    else:
        _fig, _ax = plt.subplots(figsize=(12, 4.2))
        if _view == "benefit":
            _bound = np.nanmax(np.abs(_matrix)) or 1.0
            _image = _ax.imshow(
                _matrix, cmap="RdYlGn", vmin=-_bound, vmax=_bound, aspect="auto"
            )
            _color_label = "paired change; positive is better"
        else:
            _cmap = "viridis_r" if _definition["direction"] == "lower" else "viridis"
            _image = _ax.imshow(_matrix, cmap=_cmap, aspect="auto")
            _color_label = f"absolute {_statistic} [{_definition['unit']}]"
        for _row in range(_matrix.shape[0]):
            for _column in range(_matrix.shape[1]):
                if np.isfinite(_matrix[_row, _column]):
                    _spread = _std[_row, _column]
                    _spread_text = "" if np.isnan(_spread) else f" ± {_spread:.2g}"
                    _ax.text(
                        _column,
                        _row,
                        f"{_matrix[_row, _column]:+.3g}{_spread_text}\n"
                        f"n={_counts[_row, _column]}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
        _ax.set_xticks(range(len(_domains)), _domains, rotation=35, ha="right")
        _ax.set_yticks(range(len(_bundles)), _bundles)
        _title = "Paired benefit over DR-off" if _view == "benefit" else "Absolute"
        _ax.set_title(
            f"{_title} {_statistic}: {robustness_metric.value} "
            f"[{_definition['unit']}]\n"
            f"train {robustness_train_backend.value} → "
            f"eval {robustness_eval_backend.value}"
        )
        _fig.colorbar(_image, ax=_ax, label=_color_label)
        _fig.tight_layout()
        robustness_plot = _fig
    if _view == "benefit":
        _explanation = (
            "Cells are paired seed-level changes relative to DR-off, oriented "
            "so green/positive is better. `n` is the number of completed seed "
            "pairs."
        )
    else:
        _explanation = (
            "Cells are absolute values across completed seeds, including the "
            "DR-off baseline. `n` is the number of completed seeds; color follows "
            "the metric's favorable direction."
        )
    mo.vstack(
        [
            mo.md(
                "## Final robustness\n\n"
                f"{_explanation} Choose mean or worst-decile behavior; annotations "
                "show descriptive mean ± sample SD."
            ),
            robustness_plot,
        ]
    )
    return


@app.cell
def _(
    mo,
    np,
    plt,
    robustness_metric,
    robustness_statistic,
    summary,
):
    _final = summary.get("protocol", {}).get("train_iterations")
    _records = [
        row
        for row in summary.get("evaluation", [])
        if row["checkpoint_iteration"] == _final
        and row["bundle"] == "all"
        and row["domain"] == "combined_in"
        and robustness_metric.value in row["metrics"]
    ]
    _backends = [
        item["label"] for item in summary.get("protocol", {}).get("backends", [])
    ]
    _matrix = np.full((len(_backends), len(_backends)), np.nan)
    _counts = np.zeros_like(_matrix, dtype=int)
    _definition = next(
        (
            row["metrics"][robustness_metric.value]
            for row in _records
            if robustness_metric.value in row["metrics"]
        ),
        {"unit": "", "direction": "context"},
    )
    for _row, _train in enumerate(_backends):
        for _column, _evaluation in enumerate(_backends):
            _values = [
                record["metrics"][robustness_metric.value][robustness_statistic.value]
                for record in _records
                if record["train_backend"] == _train
                and record["eval_backend"] == _evaluation
            ]
            if _values:
                _matrix[_row, _column] = np.mean(_values)
                _counts[_row, _column] = len(_values)
    if not np.isfinite(_matrix).any():
        transfer_plot = mo.md("Cross-backend final results are not available yet.")
    else:
        _fig, _ax = plt.subplots(figsize=(5.8, 4.8))
        _cmap = "viridis_r" if _definition["direction"] == "lower" else "viridis"
        _image = _ax.imshow(_matrix, cmap=_cmap, aspect="equal")
        for _row in range(len(_backends)):
            for _column in range(len(_backends)):
                if np.isfinite(_matrix[_row, _column]):
                    _rgba = _image.cmap(_image.norm(_matrix[_row, _column]))
                    _luminance = (
                        0.2126 * _rgba[0] + 0.7152 * _rgba[1] + 0.0722 * _rgba[2]
                    )
                    _ax.text(
                        _column,
                        _row,
                        f"{_matrix[_row, _column]:.3g}\nn={_counts[_row, _column]}",
                        color="black" if _luminance > 0.55 else "white",
                        ha="center",
                        va="center",
                    )
                if _row == _column:
                    _ax.add_patch(
                        plt.Rectangle(
                            (_column - 0.49, _row - 0.49),
                            0.98,
                            0.98,
                            fill=False,
                            edgecolor="white",
                            linewidth=2,
                        )
                    )
        _ax.set_xticks(range(len(_backends)), _backends)
        _ax.set_yticks(range(len(_backends)), _backends)
        _ax.set_xlabel("evaluation backend")
        _ax.set_ylabel("training backend")
        _ax.set_title(
            "All-DR transfer · combined in-range\n"
            f"{robustness_metric.value} {robustness_statistic.value} "
            f"[{_definition['unit']}]"
        )
        _fig.colorbar(
            _image,
            ax=_ax,
            label=f"{_definition['direction']} is favorable",
        )
        _fig.tight_layout()
        transfer_plot = _fig
    mo.vstack(
        [
            mo.md(
                "## Backend transfer\n\nRows are training engines and columns are "
                "evaluation engines; the outlined diagonal is native evaluation. "
                "This view reveals policies that look robust only inside their "
                "training simulator. Values are the selected absolute environment "
                "statistic averaged across seeds; `n` is the number of completed "
                "seeds, and color follows the selected metric's direction."
            ),
            transfer_plot,
        ]
    )
    return


@app.cell
def _(mo, summary):
    _final = summary.get("protocol", {}).get("train_iterations")
    _requested_seeds = set(summary.get("protocol", {}).get("seeds", []))
    _native = [
        row
        for row in summary.get("evaluation", [])
        if row["checkpoint_iteration"] == _final
        and row["train_backend"] == row["eval_backend"]
        and row["domain"] in {"nominal", "combined_in"}
        and "survival" in row["metrics"]
    ]
    _messages = []
    _backends = [
        item["label"] for item in summary.get("protocol", {}).get("backends", [])
    ]
    _bundles = [
        name for name in summary.get("protocol", {}).get("bundles", {}) if name != "off"
    ]
    for _backend in _backends:
        for _bundle in _bundles:
            if f"{_backend}:{_bundle}" in summary.get("protocol", {}).get(
                "excluded_training", []
            ):
                continue
            _selected = [
                row
                for row in _native
                if row["train_backend"] == _backend and row["bundle"] == _bundle
            ]
            _off = [
                row
                for row in _native
                if row["train_backend"] == _backend and row["bundle"] == "off"
            ]
            _nominal = {
                row["seed"]: row["metrics"]["survival"]["mean"]
                for row in _selected
                if row["domain"] == "nominal"
            }
            _off_nominal = {
                row["seed"]: row["metrics"]["survival"]["mean"]
                for row in _off
                if row["domain"] == "nominal"
            }
            _paired = [
                _nominal[seed] - _off_nominal[seed]
                for seed in _nominal.keys() & _off_nominal.keys()
            ]
            if _paired:
                _delta = sum(_paired) / len(_paired)
                if len(_paired) < len(_requested_seeds):
                    _flag = "preliminary FAIL" if _delta < -0.05 else "pending"
                else:
                    _flag = "FAIL" if _delta < -0.05 else "pass"
                    if len(_requested_seeds) < 3:
                        _flag = f"preliminary {_flag}"
                _messages.append(
                    f"- **{_backend}/{_bundle}:** {_flag} nominal-survival gate "
                    f"(Δ {_delta:+.3f}, n={len(_paired)}/{len(_requested_seeds)})"
                )
    if not summary.get("evaluation", []):
        _body = (
            "**No policy evaluations have completed yet.** Training progress "
            "alone cannot establish robustness or select a DR bundle. "
            "The policy-evaluation panels remain empty until results are available."
        )
    else:
        _body = (
            "\n".join(_messages)
            if _messages
            else "Final native seed pairs are pending."
        )
    mo.md(
        f"""
        ## Predeclared decision check

        A bundle is flagged if mean paired nominal survival falls by more than
        0.05. Passing this guard is necessary, not sufficient: the robustness
        heatmap must also show a physically coherent held-out benefit.

        {_body}
        """
    )
    return


if __name__ == "__main__":
    app.run()
