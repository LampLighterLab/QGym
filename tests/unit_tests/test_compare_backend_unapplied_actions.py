from types import SimpleNamespace

import numpy as np

from scripts.compare_backend_unapplied_actions import (
    BACKENDS,
    _first_active_step,
    cell_command,
    rmse_over_time,
    validate_protocol,
)


def test_rmse_over_time_reduces_every_non_time_axis():
    reference = np.zeros((2, 2, 3), dtype=np.float32)
    candidate = np.zeros_like(reference)
    candidate[1] = 2.0

    np.testing.assert_array_equal(rmse_over_time(reference, candidate), [0.0, 2.0])


def test_first_active_step_reports_first_true_sample():
    assert _first_active_step(np.asarray([False, False, True, True])) == 2
    assert _first_active_step(np.asarray([False, False])) is None


def test_cell_command_preserves_unapplied_action_protocol(tmp_path):
    args = SimpleNamespace(
        checkpoint=tmp_path / "model_1000.pt",
        output=tmp_path / "output",
        duration=5.0,
        seed=123,
        command=(0.0, 0.0, -0.75),
        mujoco_njmax=256,
        mujoco_solref=None,
        disable_motors=False,
    )

    command = cell_command(args, BACKENDS[0])

    assert command[command.index("--command") + 1 : command.index("--command") + 4] == [
        "0.0",
        "0.0",
        "-0.75",
    ]
    assert command[command.index("--cell") + 1] == "vsim"
    assert "--resume" not in command
    assert "--disable-motors" not in command


def test_cell_command_propagates_zero_torque_mode(tmp_path):
    args = SimpleNamespace(
        checkpoint=tmp_path / "model_1000.pt",
        output=tmp_path / "output",
        duration=1.0,
        seed=123,
        command=(0.0, 0.0, 0.0),
        mujoco_njmax=256,
        mujoco_solref=(0.005, 1.0),
        disable_motors=True,
    )

    command = cell_command(args, BACKENDS[1])

    assert "--disable-motors" in command
    assert command[command.index("--mujoco-solref") + 1 :] == ["0.005", "1.0"]


def test_protocol_validation_rejects_nonzero_environment_actions():
    artifact = {
        "schema_version": np.int64(1),
        "task": np.asarray("go2trot"),
        "checkpoint_sha256": np.asarray("hash"),
        "checkpoint_iteration": np.int64(1000),
        "original_cfg": np.bool_(True),
        "inference_mode": np.bool_(True),
        "actions_applied": np.bool_(False),
        "motors_enabled": np.bool_(True),
        "domain_randomization": np.asarray("off"),
        "seed": np.int64(123),
        "command": np.asarray([0.0, 0.0, -0.75]),
        "duration_s": np.float64(5.0),
        "ctrl_dt": np.float64(0.01),
        "sim_dt": np.float64(0.01),
        "control_frequency_hz": np.float64(100.0),
        "simulation_frequency_hz": np.float64(100.0),
        "decimation": np.int64(1),
        "mujoco_solref": np.asarray([0.005, 1.0]),
        "time": np.asarray([0.0, 0.01]),
        "transition_time": np.asarray([0.0]),
        "actor_observation_fields": np.asarray(["obs"]),
        "actor_observation_names": np.asarray(["obs"]),
        "actor_observation_scales": np.asarray([1.0]),
        "action_fields": np.asarray(["action"]),
        "action_names": np.asarray(["action"]),
        "action_scales": np.asarray([1.0]),
        "body_names": np.asarray(["body"]),
        "foot_names": np.asarray(["body"]),
        "dof_names": np.asarray(["joint"]),
        "actuated_dof_names": np.asarray(["joint"]),
        "environment_actions": np.zeros((2, 1), dtype=np.float32),
    }
    candidate = {key: value.copy() for key, value in artifact.items()}
    candidate["environment_actions"][1, 0] = 0.1

    try:
        validate_protocol(artifact, candidate)
    except ValueError as error:
        assert "candidate environment action buffers are nonzero" in str(error)
    else:
        raise AssertionError("nonzero environment action was accepted")
