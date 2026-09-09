"""Inspect deterministic Go2Trot trajectories with and without friction DR.

Generate the six one-robot artifacts first:

    uv run --env-file .env.vsim \
        scripts/eval_domain_randomization_trajectory.py

Then open the report:

    uv run marimo edit notebooks/go2_domain_randomization_trajectory.py

Set ``GO2_DR_TRAJECTORY_DIR`` to inspect a non-default artifact directory.
"""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return Path, mo, np, os, plt


@app.cell
def _(Path, np, os):
    artifact_root = Path(
        os.environ.get(
            "GO2_DR_TRAJECTORY_DIR",
            "logs/domain_randomization_trajectory/Aug11_23-14-45_model_1000",
        )
    )

    artifacts = {}
    if artifact_root.exists():
        for _path in sorted(artifact_root.glob("*_dr_*.npz")):
            with np.load(_path, allow_pickle=False) as _data:
                _artifact = {_key: _data[_key] for _key in _data.files}
            _backend = str(_artifact["backend_label"])
            _mode = str(_artifact["dr_mode"])
            artifacts.setdefault(_backend, {})[_mode] = _artifact

    complete_backends = [
        _backend
        for _backend, _modes in artifacts.items()
        if {"off", "on"} <= set(_modes)
    ]
    return artifact_root, artifacts, complete_backends


@app.cell
def _(artifact_root, complete_backends, mo):
    mo.stop(
        not complete_backends,
        mo.md(
            f"**No complete DR-on/off pairs found in `{artifact_root}`.** "
            "Run the command in this notebook's docstring first."
        ),
    )
    backend = mo.ui.dropdown(
        options=complete_backends,
        value=complete_backends[0],
        label="backend",
    )
    backend
    return (backend,)


@app.cell
def _(artifacts, backend, np):
    off = artifacts[backend.value]["off"]
    on = artifacts[backend.value]["on"]
    _initial_keys = (
        "initial_root_state",
        "initial_dof_position",
        "initial_dof_velocity",
    )
    initial_max_difference = max(
        float(np.max(np.abs(off[_key] - on[_key]))) for _key in _initial_keys
    )
    return initial_max_difference, off, on


@app.cell
def _(artifact_root, backend, initial_max_difference, mo, off, on):
    mo.md(
        f"""
        # Deterministic Go2Trot friction trajectory

        **Backend:** `{backend.value}`

        **Checkpoint:** `{off["checkpoint_path"]}` at iteration
        `{int(off["checkpoint_iteration"])}`

        **Protocol:** one robot, `{float(off["duration_s"]):g} s`, fixed command
        `{off["command"].tolist()}`, policy inference mode
        `{bool(off["inference_mode"])}`

        **Friction:** DR off `μ={float(off["contact_friction"]):g}`; DR on uses
        the deterministic representative realization
        `μ={float(on["contact_friction"]):g}`

        **Initial root/joint position/velocity maximum difference:**
        `{initial_max_difference:.3e}`

        Artifact directory: `{artifact_root}`

        Each plot compares DR off and on only within the selected backend.
        Once the states diverge, subsequent policy-action differences are the
        deterministic closed-loop response to that physical divergence.
        """
    )
    return


@app.cell
def _(mo, off):
    foot_names = [str(_name) for _name in off["foot_names"]]
    foot = mo.ui.dropdown(options=foot_names, value=foot_names[0], label="foot")
    foot
    return foot, foot_names


@app.cell
def _(foot_names, off, on, plt):
    def _plot_foot_trajectories():
        _fig, _axes = plt.subplots(
            len(foot_names),
            2,
            figsize=(13, 2.7 * len(foot_names)),
            sharex=True,
            constrained_layout=True,
        )
        _time = off["time"]
        for _foot_index, (_foot_name, _row) in enumerate(zip(foot_names, _axes)):
            for _mode, _data, _style in (
                ("DR off", off, "-"),
                ("DR on", on, "--"),
            ):
                _row[0].plot(
                    _time,
                    _data["foot_position_body"][:, _foot_index, 0],
                    _style,
                    label=_mode,
                )
                _row[1].plot(
                    _time,
                    _data["foot_position_body"][:, _foot_index, 2],
                    _style,
                    label=_mode,
                )
            _row[0].set(ylabel=f"{_foot_name}\nx rel. CoG (m)")
            _row[1].set(ylabel="z rel. CoG (m)")
            for _axis in _row:
                _axis.grid(alpha=0.2)
        _axes[0, 0].set_title("Fore-aft foot trajectory in base frame")
        _axes[0, 1].set_title("Vertical foot trajectory in base frame")
        _axes[-1, 0].set_xlabel("time (s)")
        _axes[-1, 1].set_xlabel("time (s)")
        _axes[0, 0].legend()
        return _fig

    _plot_foot_trajectories()
    return


@app.cell
def _(off, on, plt):
    def _plot_cog_motion():
        _fig, _axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        _time = off["time"]
        _specs = (
            ("height", "cog_position", 2, "m"),
            ("forward velocity", "cog_velocity", 0, "m/s"),
            ("lateral velocity", "cog_velocity", 1, "m/s"),
            ("vertical velocity", "cog_velocity", 2, "m/s"),
        )
        for _axis, (_title, _key, _index, _unit) in zip(_axes.flat, _specs):
            _axis.plot(_time, off[_key][:, _index], label="DR off")
            _axis.plot(_time, on[_key][:, _index], "--", label="DR on")
            _axis.set(
                title=f"CoG {_title}",
                xlabel="time (s)",
                ylabel=_unit,
            )
            _axis.grid(alpha=0.2)
        _axes[0, 0].legend()
        return _fig

    _plot_cog_motion()
    return


@app.cell
def _(foot_names, off, on, plt):
    def _plot_yaw_and_grf():
        _fig, _axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
        _time = off["time"]
        _axes[0, 0].plot(_time, off["base_ang_velocity_body"][:, 2], label="DR off")
        _axes[0, 0].plot(_time, on["base_ang_velocity_body"][:, 2], "--", label="DR on")
        _axes[0, 0].set(title="Yaw velocity", ylabel="rad/s")
        _axes[0, 1].plot(
            _time, off["foot_grf_world"][:, :, 2].sum(axis=1), label="DR off"
        )
        _axes[0, 1].plot(
            _time,
            on["foot_grf_world"][:, :, 2].sum(axis=1),
            "--",
            label="DR on",
        )
        _axes[0, 1].set(title="Total vertical GRF", ylabel="N")
        for _index, (_foot_name, _axis) in enumerate(zip(foot_names, _axes.flat[2:])):
            _axis.plot(
                _time,
                off["foot_grf_world"][:, _index, 2],
                label="DR off",
            )
            _axis.plot(
                _time,
                on["foot_grf_world"][:, _index, 2],
                "--",
                label="DR on",
            )
            _axis.set(title=_foot_name, ylabel="vertical GRF (N)")
        for _axis in _axes.flat:
            _axis.set_xlabel("time (s)")
            _axis.grid(alpha=0.2)
        _axes[0, 0].legend()
        return _fig

    _plot_yaw_and_grf()
    return


@app.cell
def _(np, off, on, plt):
    def _plot_trajectory_differences():
        _fig, _axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        _time = off["time"]
        _cog_position = np.linalg.norm(
            on["cog_position"] - off["cog_position"], axis=-1
        )
        _cog_velocity = np.linalg.norm(
            on["cog_velocity"] - off["cog_velocity"], axis=-1
        )
        _foot = np.sqrt(
            np.mean(
                np.square(on["foot_position_body"] - off["foot_position_body"]),
                axis=(1, 2),
            )
        )
        _grf = np.sqrt(
            np.mean(
                np.square(on["foot_grf_world"] - off["foot_grf_world"]),
                axis=(1, 2),
            )
        )
        _series = (
            (_cog_position * 1000.0, "CoG position difference", "mm"),
            (_cog_velocity, "CoG velocity difference", "m/s"),
            (_foot * 1000.0, "Foot-position RMSE", "mm"),
            (_grf, "Foot-GRF RMSE", "N"),
        )
        for _axis, (_values, _title, _unit) in zip(_axes.flat, _series):
            _axis.plot(_time, _values)
            _axis.set(title=_title, xlabel="time (s)", ylabel=_unit)
            _axis.grid(alpha=0.2)
        return _fig

    _plot_trajectory_differences()
    return


@app.cell
def _(np, off, on, plt):
    def _plot_closed_loop_response():
        _fig, _axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        _time = off["time"]
        _action_time = off["action_time"]
        _action_difference = np.sqrt(
            np.mean(np.square(on["policy_action"] - off["policy_action"]), axis=1)
        )
        _torque_difference = np.sqrt(
            np.mean(np.square(on["joint_torque"] - off["joint_torque"]), axis=1)
        )
        _axes[0].plot(_time, off["base_tilt_deg"], label="DR off")
        _axes[0].plot(_time, on["base_tilt_deg"], "--", label="DR on")
        _axes[0].set(title="Base tilt", ylabel="deg")
        _axes[1].plot(_action_time, _action_difference)
        _axes[1].set(title="Policy-action RMSE", ylabel="normalized action")
        _axes[2].plot(_time, _torque_difference, label="torque RMSE")
        _axes[2].plot(
            _time,
            np.abs(on["mechanical_power"] - off["mechanical_power"]),
            label="|power difference|",
        )
        _axes[2].set(title="Actuator response", ylabel="N·m or W")
        for _axis in _axes:
            _axis.set_xlabel("time (s)")
            _axis.grid(alpha=0.2)
        _axes[0].legend()
        _axes[2].legend()
        return _fig

    _plot_closed_loop_response()
    return


@app.cell
def _(mo, np, off, on):
    def _scalar_stats(_off_values, _on_values):
        _difference = np.asarray(_on_values) - np.asarray(_off_values)
        return (
            float(np.mean(_off_values)),
            float(np.mean(_on_values)),
            float(np.sqrt(np.mean(np.square(_difference)))),
            float(np.max(np.abs(_difference))),
        )

    _signals = (
        ("CoG height", off["cog_position"][:, 2], on["cog_position"][:, 2], "m"),
        (
            "CoG velocity",
            np.linalg.norm(off["cog_velocity"], axis=-1),
            np.linalg.norm(on["cog_velocity"], axis=-1),
            "m/s",
        ),
        (
            "Yaw velocity",
            off["base_ang_velocity_body"][:, 2],
            on["base_ang_velocity_body"][:, 2],
            "rad/s",
        ),
        (
            "Foot position",
            off["foot_position_body"].reshape(-1),
            on["foot_position_body"].reshape(-1),
            "m",
        ),
        (
            "Vertical GRF",
            off["foot_grf_world"][:, :, 2].reshape(-1),
            on["foot_grf_world"][:, :, 2].reshape(-1),
            "N",
        ),
        ("Base tilt", off["base_tilt_deg"], on["base_tilt_deg"], "deg"),
        (
            "Policy action",
            off["policy_action"].reshape(-1),
            on["policy_action"].reshape(-1),
            "normalized",
        ),
        (
            "Joint torque",
            off["joint_torque"].reshape(-1),
            on["joint_torque"].reshape(-1),
            "N·m",
        ),
        (
            "Mechanical power",
            off["mechanical_power"],
            on["mechanical_power"],
            "W",
        ),
    )
    _rows = []
    for _name, _off_values, _on_values, _unit in _signals:
        _off_mean, _on_mean, _rmse, _maximum = _scalar_stats(_off_values, _on_values)
        _rows.append(
            f"| {_name} | {_off_mean:.4g} | {_on_mean:.4g} | "
            f"{_rmse:.4g} | {_maximum:.4g} | {_unit} |"
        )
    mo.md(
        "## Whole-trajectory difference statistics\n\n"
        "These statistics compare aligned samples from the deterministic "
        "DR-on and DR-off trajectories.\n\n"
        "| Signal | mean off | mean on | aligned RMSE | max abs diff | unit |\n"
        "|---|---:|---:|---:|---:|---|\n" + "\n".join(_rows)
    )
    return


if __name__ == "__main__":
    app.run()
