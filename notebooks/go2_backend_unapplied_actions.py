"""Explore the VSim/MuJoCo comparison with policy actions left unapplied.

Generate the paired artifacts first:

    uv run --env-file .env.vsim \
        scripts/compare_backend_unapplied_actions.py \
        --checkpoint logs/go2trot_vsim_transfer_20260829/\
Aug29_22-29-40_/model_1000.pt \
        --output logs/backend_unapplied_actions/vsim_seed37_model1000_500hz_pd

The script makes the control frequency equal the 500 Hz simulation frequency,
so every notebook sample is one physics step. Add ``--disable-motors`` for the
zero-torque free-fall/contact discriminator.

Then open this notebook:

    uv run --frozen marimo edit \
        notebooks/go2_backend_unapplied_actions.py

Set ``Q2_UNAPPLIED_ACTIONS_DIR`` to select another artifact directory.
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

    def load_npz(path):
        with np.load(path, allow_pickle=False) as source:
            return {key: source[key] for key in source.files}

    def rmse_over_time(reference, candidate):
        axes = tuple(range(1, reference.ndim))
        return np.sqrt(np.mean(np.square(reference - candidate), axis=axes))

    def first_true(values):
        return int(np.flatnonzero(values)[0])

    def mark_events(axis, impact_time, termination_times):
        axis.axvline(
            impact_time,
            color="#2ca02c",
            linestyle=":",
            linewidth=1.5,
            label="first contact",
        )
        for label, event_time, color in termination_times:
            axis.axvline(
                event_time,
                color=color,
                linestyle="--",
                linewidth=1.2,
                label=f"{label} termination",
            )

    colors = {"VSim": "#1f77b4", "MuJoCo": "#ff7f0e"}

    return (
        Path,
        colors,
        first_true,
        json,
        load_npz,
        mark_events,
        mo,
        np,
        os,
        plt,
        rmse_over_time,
    )


@app.cell
def _(Path, mo, os):
    _requested = os.environ.get("Q2_UNAPPLIED_ACTIONS_DIR")
    _roots = sorted(
        {
            path.parent
            for path in Path("logs/backend_unapplied_actions").rglob("vsim.npz")
            if (path.parent / "mujoco.npz").is_file()
        },
        reverse=True,
    )
    if _requested:
        _requested_root = Path(_requested)
        if _requested_root not in _roots:
            _roots.insert(0, _requested_root)
        _default_root = str(_requested_root)
    else:
        _default_root = str(_roots[0]) if _roots else None

    mo.stop(
        not _roots,
        mo.md(
            "**No paired artifacts found.** Run the command in this "
            "notebook's docstring first."
        ),
    )
    artifact_picker = mo.ui.dropdown(
        options={str(path): str(path) for path in _roots},
        value=_default_root,
        label="artifact directory",
        full_width=True,
    )
    artifact_picker
    return (artifact_picker,)


@app.cell
def _(Path, artifact_picker, json, load_npz, np):
    artifact_root = Path(artifact_picker.value)
    vsim = load_npz(artifact_root / "vsim.npz")
    mujoco = load_npz(artifact_root / "mujoco.npz")
    comparison_path = artifact_root / "comparison.json"
    comparison = json.loads(comparison_path.read_text())

    np.testing.assert_array_equal(vsim["time"], mujoco["time"])
    np.testing.assert_array_equal(
        vsim["actor_observation_names"], mujoco["actor_observation_names"]
    )
    np.testing.assert_array_equal(vsim["body_names"], mujoco["body_names"])
    np.testing.assert_array_equal(vsim["dof_names"], mujoco["dof_names"])

    return artifact_root, comparison, mujoco, vsim


@app.cell
def _(first_true, mujoco, np, rmse_over_time, vsim):
    time = vsim["time"]
    body_names = [str(name) for name in vsim["body_names"]]
    foot_names = [str(name) for name in vsim["foot_names"]]
    foot_indices = [body_names.index(name) for name in foot_names]

    vsim_contact_norm = np.linalg.vector_norm(vsim["contact_forces"], axis=2)
    mujoco_contact_norm = np.linalg.vector_norm(mujoco["contact_forces"], axis=2)
    vsim_impact_step = first_true(vsim_contact_norm.max(axis=1) > 1.0e-4)
    mujoco_impact_step = first_true(mujoco_contact_norm.max(axis=1) > 1.0e-4)
    earliest_impact_step = min(vsim_impact_step, mujoco_impact_step)
    vsim_termination_step = first_true(vsim["terminated"])
    mujoco_termination_step = first_true(mujoco["terminated"])

    divergence = {
        "Actor observation": rmse_over_time(
            vsim["actor_observations"], mujoco["actor_observations"]
        ),
        "Queried action": rmse_over_time(
            vsim["queried_actions"], mujoco["queried_actions"]
        ),
        "All-body contact force [N]": rmse_over_time(
            vsim["contact_forces"], mujoco["contact_forces"]
        ),
        "Foot contact force [N]": rmse_over_time(
            vsim["contact_forces"][:, foot_indices],
            mujoco["contact_forces"][:, foot_indices],
        ),
        "Joint position [rad]": rmse_over_time(
            vsim["dof_position"], mujoco["dof_position"]
        ),
        "Joint velocity [rad/s]": rmse_over_time(
            vsim["dof_velocity"], mujoco["dof_velocity"]
        ),
        "Root state": rmse_over_time(vsim["root_state"], mujoco["root_state"]),
        "Rigid-body position [m]": rmse_over_time(
            vsim["rigid_body_state"][..., :3],
            mujoco["rigid_body_state"][..., :3],
        ),
        "Rigid-body velocity [m/s]": rmse_over_time(
            vsim["rigid_body_state"][..., 7:10],
            mujoco["rigid_body_state"][..., 7:10],
        ),
        "Joint acceleration [rad/s²]": rmse_over_time(
            vsim["dof_acceleration"], mujoco["dof_acceleration"]
        ),
        "Contact impulse [N s]": rmse_over_time(
            vsim["contact_impulse"], mujoco["contact_impulse"]
        ),
    }
    termination_times = (
        ("VSim", float(time[vsim_termination_step]), "#1f77b4"),
        ("MuJoCo", float(time[mujoco_termination_step]), "#ff7f0e"),
    )
    return (
        body_names,
        divergence,
        earliest_impact_step,
        foot_indices,
        foot_names,
        mujoco_contact_norm,
        mujoco_impact_step,
        mujoco_termination_step,
        termination_times,
        time,
        vsim_contact_norm,
        vsim_impact_step,
        vsim_termination_step,
    )


@app.cell
def _(
    artifact_root,
    comparison,
    earliest_impact_step,
    mo,
    mujoco,
    mujoco_impact_step,
    mujoco_termination_step,
    np,
    time,
    vsim,
    vsim_impact_step,
    vsim_termination_step,
):
    _applied_max_vsim = float(np.abs(vsim["environment_actions"]).max())
    _applied_max_mujoco = float(np.abs(mujoco["environment_actions"]).max())
    _initial = comparison["initial"]
    _motion_source = (
        "open-loop gait plus PD control"
        if bool(vsim["motors_enabled"])
        else "gravity and contact only (zero applied torque)"
    )
    mo.vstack(
        [
            mo.md(
                f"""
                # VSim / MuJoCo unapplied-policy comparison

                One deterministic Go2Trot robot is initialized identically in
                each backend. The policy is queried at every control sample,
                but its output is never assigned to the environment. Motion
                comes from **{_motion_source}**.

                **Checkpoint:** iteration
                `{int(vsim["checkpoint_iteration"])}` ·
                `{str(vsim["checkpoint_sha256"])[:12]}…`

                **Protocol:** `{float(vsim["duration_s"]):g} s` · control/sim
                `{float(vsim["control_frequency_hz"]):g}` /
                `{float(vsim["simulation_frequency_hz"]):g} Hz` · decimation
                `{int(vsim["decimation"])}` · motors
                `{bool(vsim["motors_enabled"])}` · command
                `{vsim["command"].tolist()}` · DR
                `{str(vsim["domain_randomization"])}`

                **Artifact:** `{artifact_root}`
                """
            ),
            mo.callout(
                mo.md(
                    f"**Action-path check:** maximum applied action is "
                    f"`{_applied_max_vsim:g}` in VSim and "
                    f"`{_applied_max_mujoco:g}` in MuJoCo. Initial observation "
                    f"RMSE is `{_initial['observation_rmse']:.3e}`."
                ),
                kind="success",
            ),
            mo.md(
                f"""
                **Events:** first contact occurs at
                `{float(time[vsim_impact_step]):.3f} s` in VSim and
                `{float(time[mujoco_impact_step]):.3f} s` in MuJoCo. The shared
                comparison event is `{float(time[earliest_impact_step]):.3f} s`.
                MuJoCo terminates at
                `{float(time[mujoco_termination_step]):.3f} s`; VSim terminates
                at `{float(time[vsim_termination_step]):.3f} s`.
                """
            ),
        ]
    )
    return


@app.cell
def _(body_names, comparison, mo, mujoco, np, plt, vsim):
    _mass_difference = vsim["link_mass_kg"] - mujoco["link_mass_kg"]
    _inertia_difference = np.max(
        np.abs(
            np.sort(vsim["link_inertia_diagonal"], axis=1)
            - np.sort(mujoco["link_inertia_diagonal"], axis=1)
        ),
        axis=1,
    )
    _order = np.argsort(np.abs(_mass_difference))[::-1][:8]
    _rows = [
        {
            "body": body_names[_index],
            "VSim mass [kg]": float(vsim["link_mass_kg"][_index]),
            "MuJoCo mass [kg]": float(mujoco["link_mass_kg"][_index]),
            "difference [kg]": float(_mass_difference[_index]),
        }
        for _index in _order
    ]
    _figure, _axes = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, constrained_layout=True
    )
    _axes[0].bar(body_names, _mass_difference, color="#4c72b0")
    _axes[0].set_ylabel("VSim − MuJoCo mass [kg]")
    _axes[1].bar(body_names, _inertia_difference, color="#dd8452")
    _axes[1].set_ylabel("max principal-moment difference [kg m²]")
    _axes[1].tick_params(axis="x", rotation=75, labelsize=7)
    for _axis in _axes:
        _axis.grid(axis="y", alpha=0.25)
    _figure.suptitle("Effective inertial-model comparison")
    mo.vstack(
        [
            mo.md(
                "## Static model sanity check\n\n"
                f"Total mass is `{float(vsim['robot_mass_kg']):.6g} kg` in "
                f"VSim and `{float(mujoco['robot_mass_kg']):.6g} kg` in "
                "MuJoCo. Principal moments are sorted before comparison "
                "because the engines may choose different inertial-axis order."
            ),
            _figure,
            mo.ui.table(_rows, selection=None),
        ]
    )
    return


@app.cell
def _(
    divergence,
    earliest_impact_step,
    mark_events,
    plt,
    termination_times,
    time,
):
    def plot_divergence_overview():
        _figure, _axes = plt.subplots(
            5, 2, figsize=(13, 15), sharex=True, constrained_layout=True
        )
        _specs = (
            ("Actor observation", "normalized", "Observation divergence"),
            ("Queried action", "normalized", "Policy response divergence"),
            (
                "All-body contact force [N]",
                "N",
                "All-body contact divergence",
            ),
            ("Foot contact force [N]", "N", "Foot contact divergence"),
            ("Joint position [rad]", "rad", "Joint-position divergence"),
            (
                "Joint velocity [rad/s]",
                "rad/s",
                "Joint-velocity divergence",
            ),
            (
                "Rigid-body position [m]",
                "m",
                "Body-origin position divergence",
            ),
            (
                "Rigid-body velocity [m/s]",
                "m/s",
                "Body-origin velocity divergence",
            ),
            (
                "Joint acceleration [rad/s²]",
                "rad/s²",
                "Joint-acceleration divergence",
            ),
            (
                "Contact impulse [N s]",
                "N s",
                "Cumulative body-impulse divergence",
            ),
        )
        for _axis, (_key, _unit, _title) in zip(_axes.flat, _specs):
            _axis.plot(time, divergence[_key], color="#6a3d9a")
            mark_events(
                _axis,
                float(time[earliest_impact_step]),
                termination_times,
            )
            _axis.set(title=_title, ylabel=f"RMSE [{_unit}]")
            _axis.grid(alpha=0.25)
        for _axis in _axes[-1]:
            _axis.set_xlabel("time [s]")
        _axes[0, 0].legend(fontsize=8)
        _figure.suptitle("Aligned trajectory divergence")
        return _figure

    plot_divergence_overview()
    return


@app.cell
def _(colors, earliest_impact_step, mujoco, np, plt, time, vsim):
    _transition_time = vsim["transition_time"]
    _torque_difference = vsim["applied_torque"] - mujoco["applied_torque"]
    _torque_rmse = np.sqrt(np.mean(np.square(_torque_difference), axis=1))
    _impact_transition = max(0, earliest_impact_step - 1)
    _joint_index = int(np.argmax(np.abs(_torque_difference[_impact_transition])))
    _joint_name = str(vsim["actuated_dof_names"][_joint_index])
    _mask = np.abs(_transition_time - time[earliest_impact_step]) <= 0.05
    _figure, _axes = plt.subplots(
        1, 2, figsize=(13, 4), constrained_layout=True
    )
    _axes[0].plot(_transition_time, _torque_rmse, color="#6a3d9a")
    _axes[0].set(title="Applied-torque divergence", ylabel="RMSE [N m]")
    _axes[1].plot(
        _transition_time[_mask],
        vsim["applied_torque"][_mask, _joint_index],
        label="VSim",
        color=colors["VSim"],
    )
    _axes[1].plot(
        _transition_time[_mask],
        mujoco["applied_torque"][_mask, _joint_index],
        "--",
        label="MuJoCo",
        color=colors["MuJoCo"],
    )
    _axes[1].set(
        title=f"{_joint_name}: ±50 ms around impact",
        ylabel="applied torque [N m]",
    )
    for _axis in _axes:
        _axis.axvline(
            time[earliest_impact_step], color="black", linestyle=":", linewidth=1
        )
        _axis.set_xlabel("time [s]")
        _axis.grid(alpha=0.25)
    _axes[1].legend(fontsize=8)
    _figure.suptitle(
        "Controller output (policy actions remain unapplied; motors may be disabled)"
    )
    _figure
    return


@app.cell
def _(colors, comparison, mujoco, np, plt, vsim):
    def plot_initial_values():
        _figure, _axes = plt.subplots(
            1, 3, figsize=(15, 4.2), constrained_layout=True
        )
        _observation_index = np.arange(vsim["actor_observations"].shape[1])
        _action_index = np.arange(vsim["queried_actions"].shape[1])
        _axes[0].plot(
            _observation_index,
            vsim["actor_observations"][0],
            label="VSim",
            color=colors["VSim"],
        )
        _axes[0].plot(
            _observation_index,
            mujoco["actor_observations"][0],
            "--",
            label="MuJoCo",
            color=colors["MuJoCo"],
        )
        _axes[0].set(
            title="Actor observations at t=0",
            xlabel="component index",
            ylabel="normalized value",
        )
        _axes[1].plot(
            _action_index,
            vsim["queried_actions"][0],
            marker="o",
            label="VSim",
            color=colors["VSim"],
        )
        _axes[1].plot(
            _action_index,
            mujoco["queried_actions"][0],
            "--x",
            label="MuJoCo",
            color=colors["MuJoCo"],
        )
        _axes[1].set(
            title="Queried actions at t=0",
            xlabel="actuator index",
            ylabel="normalized value",
        )
        _observation_difference = np.sort(
            np.abs(
                vsim["actor_observations"][0] - mujoco["actor_observations"][0]
            )
        )[::-1]
        _action_difference = np.sort(
            np.abs(vsim["queried_actions"][0] - mujoco["queried_actions"][0])
        )[::-1]
        _axes[2].semilogy(
            _observation_difference,
            label="observation components",
            color="#6a3d9a",
        )
        _axes[2].semilogy(
            _action_difference,
            label="action components",
            color="#33a02c",
        )
        _axes[2].set(
            title="Sorted absolute differences at t=0",
            xlabel="component rank",
            ylabel="absolute difference",
        )
        for _axis in _axes:
            _axis.grid(alpha=0.25)
            _axis.legend(fontsize=8)
        _figure.suptitle(
            "Initial agreement — observation RMSE "
            f"{comparison['initial']['observation_rmse']:.2e}, action RMSE "
            f"{comparison['initial']['action_rmse']:.2e}"
        )
        return _figure

    plot_initial_values()
    return


@app.cell
def _(
    earliest_impact_step,
    mo,
    mujoco_impact_step,
    time,
    vsim_impact_step,
):
    _impact_options = {
            f"earliest ({float(time[earliest_impact_step]):.3f} s)": (
                earliest_impact_step
            ),
            f"VSim ({float(time[vsim_impact_step]):.3f} s)": vsim_impact_step,
            f"MuJoCo ({float(time[mujoco_impact_step]):.3f} s)": (
                mujoco_impact_step
            ),
        }
    impact_reference = mo.ui.dropdown(
        options=_impact_options,
        value=next(iter(_impact_options)),
        label="impact reference",
    )
    pre_impact_steps = mo.ui.slider(
        1,
        min(5, earliest_impact_step),
        value=1,
        step=1,
        show_value=True,
        label="steps before impact",
    )
    post_impact_steps = mo.ui.slider(
        0,
        20,
        value=0,
        step=1,
        show_value=True,
        label="steps after first contact",
    )
    impact_window_ms = mo.ui.dropdown(
        options={"±50 ms": 50, "±100 ms": 100, "±250 ms": 250},
        value="±100 ms",
        label="local window",
    )
    mo.vstack(
        [
            mo.md(
                "## Pre- versus post-impact visualizer\n\n"
                "Choose the event and samples used below. With the default "
                "offsets, **pre** is the last airborne sample and **post** is "
                "the first contact-bearing sample."
            ),
            mo.hstack(
                [
                    impact_reference,
                    pre_impact_steps,
                    post_impact_steps,
                    impact_window_ms,
                ],
                justify="start",
                gap=1,
            ),
        ]
    )
    return (
        impact_reference,
        impact_window_ms,
        post_impact_steps,
        pre_impact_steps,
    )


@app.cell
def _(
    impact_reference,
    mujoco,
    post_impact_steps,
    pre_impact_steps,
    time,
    vsim,
):
    impact_step = int(impact_reference.value)
    pre_impact_step = impact_step - int(pre_impact_steps.value)
    post_impact_step = min(
        impact_step + int(post_impact_steps.value), len(time) - 1
    )
    pre_contact_forces = {
        "VSim": vsim["contact_forces"][pre_impact_step],
        "MuJoCo": mujoco["contact_forces"][pre_impact_step],
    }
    post_contact_forces = {
        "VSim": vsim["contact_forces"][post_impact_step],
        "MuJoCo": mujoco["contact_forces"][post_impact_step],
    }
    return (
        impact_step,
        post_contact_forces,
        post_impact_step,
        pre_contact_forces,
        pre_impact_step,
    )


@app.cell
def _(
    body_names,
    impact_step,
    mujoco,
    np,
    plt,
    post_contact_forces,
    post_impact_step,
    pre_contact_forces,
    pre_impact_step,
    time,
    vsim,
):
    def plot_impact_force_transition():
        _force_limit = max(
            1.0,
            float(
                np.max(
                    np.abs(
                        np.stack(
                            [
                                *pre_contact_forces.values(),
                                *post_contact_forces.values(),
                            ]
                        )
                    )
                )
            ),
        )
        _figure, _axes = plt.subplots(
            2, 3, figsize=(11, 14), sharex=True, sharey=True, constrained_layout=True
        )
        _rows = (
            ("PRE", pre_impact_step, pre_contact_forces),
            ("POST", post_impact_step, post_contact_forces),
        )
        _image = None
        for _row_index, (_phase, _step, _forces) in enumerate(_rows):
            _panels = (
                ("VSim", _forces["VSim"]),
                ("MuJoCo", _forces["MuJoCo"]),
                ("VSim − MuJoCo", _forces["VSim"] - _forces["MuJoCo"]),
            )
            for _column_index, (_label, _values) in enumerate(_panels):
                _axis = _axes[_row_index, _column_index]
                _image = _axis.imshow(
                    _values,
                    aspect="auto",
                    cmap="RdBu_r",
                    vmin=-_force_limit,
                    vmax=_force_limit,
                )
                _relative_ms = 1000.0 * (time[_step] - time[impact_step])
                _axis.set_title(
                    f"{_phase}: {_label}\n"
                    f"t={time[_step]:.3f} s ({_relative_ms:+.0f} ms)"
                )
                _axis.set_xticks(range(3), ["Fx", "Fy", "Fz"])
        for _axis in _axes[:, 0]:
            _axis.set_yticks(range(len(body_names)), body_names, fontsize=7)
        _figure.colorbar(
            _image,
            ax=_axes,
            label="world-frame contact force [N]",
            shrink=0.7,
        )
        _figure.suptitle(
            "Canonical body-force state immediately around impact",
            fontsize=15,
        )
        return _figure

    plot_impact_force_transition()
    return


@app.cell
def _(
    divergence,
    mo,
    post_impact_step,
    pre_impact_step,
    time,
):
    _metric_specs = (
        ("Actor observation", "normalized"),
        ("Queried action", "normalized"),
        ("All-body contact force [N]", "N"),
        ("Foot contact force [N]", "N"),
        ("Joint position [rad]", "rad"),
        ("Joint velocity [rad/s]", "rad/s"),
        ("Rigid-body position [m]", "m"),
        ("Rigid-body velocity [m/s]", "m/s"),
        ("Joint acceleration [rad/s²]", "rad/s²"),
        ("Contact impulse [N s]", "N s"),
        ("Root state", "mixed"),
    )
    impact_rows = [
        {
            "signal": _name,
            "pre-impact RMSE": f"{divergence[_name][pre_impact_step]:.6g}",
            "post-impact RMSE": f"{divergence[_name][post_impact_step]:.6g}",
            "unit": _unit,
        }
        for _name, _unit in _metric_specs
    ]
    mo.vstack(
        [
            mo.md(
                f"### Selected transition statistics\n\n"
                f"Pre-impact sample: `{time[pre_impact_step]:.3f} s`; "
                f"post-impact sample: `{time[post_impact_step]:.3f} s`."
            ),
            mo.ui.table(impact_rows, selection=None),
        ]
    )
    return


@app.cell
def _(
    divergence,
    impact_step,
    mark_events,
    mujoco,
    np,
    plt,
    post_impact_step,
    pre_impact_step,
    time,
    vsim,
):
    def plot_pre_post_metrics():
        _specs = (
            ("Actor observation", "normalized"),
            ("Queried action", "normalized"),
            ("All-body contact force [N]", "N"),
            ("Foot contact force [N]", "N"),
            ("Joint position [rad]", "rad"),
            ("Joint velocity [rad/s]", "rad/s"),
            ("Rigid-body velocity [m/s]", "m/s"),
            ("Joint acceleration [rad/s²]", "rad/s²"),
        )
        _figure, _axes = plt.subplots(
            2, 4, figsize=(15, 7), constrained_layout=True
        )
        for _axis, (_name, _unit) in zip(_axes.flat, _specs):
            _values = divergence[_name][[pre_impact_step, post_impact_step]]
            _axis.bar(
                ["pre", "post"],
                _values,
                color=["#b2df8a", "#e31a1c"],
            )
            _axis.set(title=_name, ylabel=f"RMSE [{_unit}]")
            _axis.grid(axis="y", alpha=0.25)
        _figure.suptitle(
            f"Divergence before and after impact at t={time[impact_step]:.3f} s"
        )
        return _figure

    plot_pre_post_metrics()
    return


@app.cell
def _(
    divergence,
    impact_step,
    impact_window_ms,
    mujoco,
    np,
    plt,
    time,
    vsim,
):
    def plot_local_impact_window():
        _relative_ms = 1000.0 * (time - time[impact_step])
        _window = int(impact_window_ms.value)
        _mask = np.abs(_relative_ms) <= _window
        _force_difference = np.linalg.vector_norm(
            vsim["contact_forces"] - mujoco["contact_forces"], axis=2
        )
        _impact_body_index = int(np.argmax(_force_difference[impact_step]))
        _impact_body = str(vsim["body_names"][_impact_body_index])
        _figure, _axes = plt.subplots(
            2, 2, figsize=(13, 7), sharex=True, constrained_layout=True
        )
        _axes[0, 0].plot(
            _relative_ms[_mask],
            divergence["Actor observation"][_mask],
            label="observation RMSE",
        )
        _axes[0, 0].plot(
            _relative_ms[_mask],
            divergence["Queried action"][_mask],
            label="queried-action RMSE",
        )
        _axes[0, 0].set(title="Policy inputs and outputs", ylabel="normalized")
        _axes[0, 1].plot(
            _relative_ms[_mask],
            divergence["Joint velocity [rad/s]"][_mask],
            label="velocity",
        )
        _axes[0, 1].plot(
            _relative_ms[_mask],
            divergence["Joint position [rad]"][_mask],
            label="position",
        )
        _axes[0, 1].set(title="Joint-state RMSE", ylabel="rad or rad/s")
        _axes[1, 0].plot(
            _relative_ms[_mask],
            vsim["contact_forces"][_mask, _impact_body_index, 2],
            label="VSim",
        )
        _axes[1, 0].plot(
            _relative_ms[_mask],
            mujoco["contact_forces"][_mask, _impact_body_index, 2],
            "--",
            label="MuJoCo",
        )
        _axes[1, 0].set(
            title=f"{_impact_body} vertical force", ylabel="world Fz [N]"
        )
        _axes[1, 1].plot(
            _relative_ms[_mask],
            divergence["All-body contact force [N]"][_mask],
            label="all bodies",
        )
        _axes[1, 1].plot(
            _relative_ms[_mask],
            divergence["Foot contact force [N]"][_mask],
            label="feet",
        )
        _axes[1, 1].set(title="Contact-force RMSE", ylabel="N")
        for _axis in _axes.flat:
            _axis.axvline(0.0, color="black", linestyle=":", linewidth=1.2)
            _axis.set_xlabel("time relative to impact [ms]")
            _axis.grid(alpha=0.25)
            _axis.legend(fontsize=8)
        _figure.suptitle("Impact-localized trajectory response")
        return _figure

    plot_local_impact_window()
    return


@app.cell
def _(mo):
    signal_source = mo.ui.dropdown(
        options={
            "actor observations": "observations",
            "queried policy actions": "actions",
            "joint position": "position",
            "joint velocity": "velocity",
            "root state": "root",
        },
        value="actor observations",
        label="signal family",
    )
    mo.vstack(
        [
            mo.md(
                "## Signal explorer\n\n"
                "Inspect named values in the two backends and their aligned "
                "difference. Policy outputs remain diagnostic only: they were "
                "not applied."
            ),
            signal_source,
        ]
    )
    return (signal_source,)


@app.cell
def _(mujoco, signal_source, vsim):
    _root_names = (
        "position.x",
        "position.y",
        "position.z",
        "orientation.x",
        "orientation.y",
        "orientation.z",
        "orientation.w",
        "linear_velocity.x",
        "linear_velocity.y",
        "linear_velocity.z",
        "angular_velocity.x",
        "angular_velocity.y",
        "angular_velocity.z",
    )
    _sources = {
        "observations": (
            vsim["actor_observations"],
            mujoco["actor_observations"],
            [str(name) for name in vsim["actor_observation_names"]],
            "normalized",
        ),
        "actions": (
            vsim["queried_actions"],
            mujoco["queried_actions"],
            [str(name) for name in vsim["action_names"]],
            "normalized",
        ),
        "position": (
            vsim["dof_position"],
            mujoco["dof_position"],
            [str(name) for name in vsim["dof_names"]],
            "rad",
        ),
        "velocity": (
            vsim["dof_velocity"],
            mujoco["dof_velocity"],
            [str(name) for name in vsim["dof_names"]],
            "rad/s",
        ),
        "root": (
            vsim["root_state"],
            mujoco["root_state"],
            list(_root_names),
            "mixed",
        ),
    }
    selected_vsim_signal, selected_mujoco_signal, signal_names, signal_unit = (
        _sources[signal_source.value]
    )
    return (
        selected_mujoco_signal,
        selected_vsim_signal,
        signal_names,
        signal_unit,
    )


@app.cell
def _(
    impact_step,
    mo,
    np,
    selected_mujoco_signal,
    selected_vsim_signal,
    signal_names,
):
    _difference = np.abs(
        selected_vsim_signal[impact_step] - selected_mujoco_signal[impact_step]
    )
    _default_indices = np.argsort(_difference)[::-1][: min(4, len(signal_names))]
    _defaults = [signal_names[index] for index in _default_indices]
    selected_signals = mo.ui.multiselect(
        options=signal_names,
        value=_defaults,
        max_selections=6,
        label="components (up to 6)",
        full_width=True,
    )
    selected_signals
    return (selected_signals,)


@app.cell
def _(
    colors,
    earliest_impact_step,
    mark_events,
    mo,
    np,
    plt,
    selected_mujoco_signal,
    selected_signals,
    selected_vsim_signal,
    signal_names,
    signal_unit,
    termination_times,
    time,
):
    def plot_selected_signals():
        _names = selected_signals.value
        if not _names:
            return mo.md("_Select at least one component._")
        _figure, _axes = plt.subplots(
            len(_names),
            2,
            figsize=(13, 2.5 * len(_names)),
            sharex=True,
            constrained_layout=True,
            squeeze=False,
        )
        for _row, _name in zip(_axes, _names):
            _index = signal_names.index(_name)
            _row[0].plot(
                time,
                selected_vsim_signal[:, _index],
                label="VSim",
                color=colors["VSim"],
            )
            _row[0].plot(
                time,
                selected_mujoco_signal[:, _index],
                "--",
                label="MuJoCo",
                color=colors["MuJoCo"],
            )
            _row[1].plot(
                time,
                selected_vsim_signal[:, _index]
                - selected_mujoco_signal[:, _index],
                color="#6a3d9a",
            )
            _row[0].set(title=_name, ylabel=signal_unit)
            _row[1].set(title=f"VSim − MuJoCo: {_name}", ylabel=signal_unit)
            for _axis in _row:
                mark_events(
                    _axis,
                    float(time[earliest_impact_step]),
                    termination_times,
                )
                _axis.grid(alpha=0.25)
        _axes[0, 0].legend(fontsize=8)
        for _axis in _axes[-1]:
            _axis.set_xlabel("time [s]")
        return _figure

    plot_selected_signals()
    return


@app.cell
def _(body_names, foot_names, mo):
    contact_component = mo.ui.dropdown(
        options={"vertical Fz": 2, "fore-aft Fx": 0, "lateral Fy": 1, "norm": 3},
        value="vertical Fz",
        label="contact component",
    )
    selected_bodies = mo.ui.multiselect(
        options=body_names,
        value=foot_names,
        max_selections=8,
        label="canonical bodies",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md(
                "## Contact-force explorer\n\n"
                "Forces use canonical body order, world axes, and Newtons."
            ),
            mo.hstack(
                [contact_component, selected_bodies], justify="start", gap=1
            ),
        ]
    )
    return contact_component, selected_bodies


@app.cell
def _(
    body_names,
    colors,
    contact_component,
    earliest_impact_step,
    mark_events,
    mo,
    mujoco,
    np,
    plt,
    selected_bodies,
    termination_times,
    time,
    vsim,
):
    def plot_contact_forces():
        _bodies = selected_bodies.value
        if not _bodies:
            return mo.md("_Select at least one body._")
        _figure, _axes = plt.subplots(
            len(_bodies),
            1,
            figsize=(12, 2.5 * len(_bodies)),
            sharex=True,
            constrained_layout=True,
            squeeze=False,
        )
        for _axis, _body in zip(_axes[:, 0], _bodies):
            _index = body_names.index(_body)
            if contact_component.value == 3:
                _vsim_values = np.linalg.vector_norm(
                    vsim["contact_forces"][:, _index], axis=1
                )
                _mujoco_values = np.linalg.vector_norm(
                    mujoco["contact_forces"][:, _index], axis=1
                )
            else:
                _vsim_values = vsim["contact_forces"][
                    :, _index, contact_component.value
                ]
                _mujoco_values = mujoco["contact_forces"][
                    :, _index, contact_component.value
                ]
            _axis.plot(time, _vsim_values, label="VSim", color=colors["VSim"])
            _axis.plot(
                time,
                _mujoco_values,
                "--",
                label="MuJoCo",
                color=colors["MuJoCo"],
            )
            mark_events(
                _axis,
                float(time[earliest_impact_step]),
                termination_times,
            )
            _axis.set(title=_body, ylabel="force [N]")
            _axis.grid(alpha=0.25)
        _axes[0, 0].legend(fontsize=8)
        _axes[-1, 0].set_xlabel("time [s]")
        return _figure

    plot_contact_forces()
    return


@app.cell
def _(comparison, mo, vsim):
    _pre = comparison["pre_termination_peak"]
    _first = comparison["first_contact"]
    _post = comparison["first_post_step"]
    _mode = (
        "With the gait/PD controller active"
        if bool(vsim["motors_enabled"])
        else "With every applied torque forced to zero"
    )
    mo.callout(
        mo.md(
            f"""
            ## Interpretation

            {_mode}, reset agreement is effectively exact. After one 2 ms step,
            joint-velocity RMSE is `{_post['dof_velocity_rmse_rad_s']:.3g} rad/s`
            and body-origin position RMSE is
            `{_post['rigid_body_position_rmse_m']:.3g} m`. First contact occurs
            at `{_first['vsim_time_s']:.3f} s` in VSim and
            `{_first['mujoco_time_s']:.3f} s` in MuJoCo. At the earliest impact,
            joint-velocity RMSE is
            `{_first['dof_velocity_rmse_rad_s']:.3g} rad/s`, foot-force RMSE is
            `{_first['foot_contact_rmse_n']:.3g} N`, and cumulative contact-
            impulse difference is
            `{_first['cumulative_contact_impulse_difference_ns']:.3g} N s`.

            The zero-torque artifact is the clean discriminator: if its
            airborne rows remain near machine precision and diverge only at
            collision, controller and policy paths are ruled out. The next
            discriminators should therefore vary collision geometry/contact
            offsets, restitution/compliance, and solver settings one at a time.
            This notebook localizes the discrepancy; it does not yet identify
            which engine's contact response is physically preferable.
            """
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
