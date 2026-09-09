"""Pendulum regression uses real task, optimizer, and checkpoint behavior."""

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from gym.envs.pendulum.pendulum_config import PendulumCfg, PendulumRunnerCfg
from gym.utils.task_registry import task_registry
from learning.algorithms import PPO2
from learning.runners.on_policy_runner import OnPolicyRunner
from scripts import regression_pendulum_training as regression


@pytest.fixture
def registered_pendulum_runner(tmp_path):
    # Do not use the regression profile here: ordinary registered task configs
    # must construct the active runner without a test-only rollout field repair.
    cfg, train_cfg = PendulumCfg(), PendulumRunnerCfg()
    cfg.seed = 7
    cfg.env.num_envs = 4
    cfg.control.ctrl_frequency = cfg.control.desired_sim_frequency = 100
    train_cfg.actor.frequency = 100
    train_cfg.log_dir = str(tmp_path / "registered")
    task_registry.convert_frequencies_to_params(cfg, train_cfg)
    env = task_registry.make_env("pendulum", cfg, "cpu", True, "mujoco")
    try:
        yield task_registry.make_alg_runner(env, train_cfg)
    finally:
        env._backend.close()


def test_registered_pendulum_constructs_current_ppo_runner(registered_pendulum_runner):
    runner = registered_pendulum_runner
    assert isinstance(runner, OnPolicyRunner)
    assert isinstance(runner.alg, PPO2)
    assert (
        runner.num_steps_per_env * runner.env.num_envs == runner.alg_cfg["rollout_size"]
    )
    assert runner.get_inference_actions().shape == (4, 1)


def test_initial_networks_do_not_depend_on_task_rng_consumption():
    cfg = PendulumRunnerCfg()
    first = regression.initial_state(7, cfg)
    torch.rand(93)
    before = torch.random.get_rng_state().clone()
    second = regression.initial_state(7, cfg)
    assert torch.equal(torch.random.get_rng_state(), before)
    different_seed = regression.initial_state(17, cfg)
    for name in ("actor", "critic"):
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        assert regression.tensor_hash(first[name]) != regression.tensor_hash(
            different_seed[name]
        )


def test_catch_requires_full_final_second_and_wraps_angle():
    theta = np.zeros((101, 4))
    omega = np.zeros_like(theta)
    theta[:, 0] = 2 * math.pi
    theta[:51, 1] = 0.2  # Final pose succeeds but the final second does not.
    omega[:, 2] = 0.51
    theta[0, 3] = math.pi

    metrics, caught, times = regression.catch_metrics(theta, omega, 0.01)

    np.testing.assert_array_equal(caught, [True, False, False, True])
    np.testing.assert_allclose(times[[0, 3]], [0, 0.01])
    assert np.isnan(times[[1, 2]]).all()
    assert metrics["catch_rate"] == 0.5
    assert metrics["catch_time_mean_s"] == 0.005


def test_evaluation_records_actual_torque_and_exact_initial_grid(
    registered_pendulum_runner, tmp_path
):
    runner = registered_pendulum_runner
    env = runner.env
    env._reset_state = env.reset_to_uniform
    with torch.no_grad():
        for parameter in runner.alg.actor.NN.parameters():
            parameter.zero_()
        runner.alg.actor.NN[-1].bias.fill_(20)
    output = tmp_path / "constant.npz"

    metrics = regression.evaluate(runner, 1.0, output)

    with np.load(output) as artifact:
        assert float(artifact["dt"]) == 0.01
        np.testing.assert_array_equal(artifact["raw_actions"], 20)
        np.testing.assert_array_equal(artifact["applied_torques"], 5)
        expected = np.asarray(
            [[-math.pi, -5], [-math.pi, 5], [math.pi, -5], [math.pi, 5]]
        )
        np.testing.assert_allclose(artifact["state"][0], expected, atol=1e-7)
        assert not np.allclose(artifact["state"][-1], artifact["state"][0])
        weighted = (artifact["reward_terms"] * artifact["reward_weights"]).sum(axis=-1)
        assert metrics["weighted_reward_mean"] == pytest.approx(
            weighted.mean(), abs=2e-6
        )
        assert metrics["torque_saturation_fraction"] == 1
        assert metrics["all_finite"]


@pytest.mark.parametrize("require_learning", [False, True])
def test_cpu_worker_trains_evaluates_and_resumes_real_checkpoint(
    tmp_path, require_learning
):
    output = tmp_path / "cpu"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.regression_pendulum_training",
            "--backend",
            "mujoco",
            "--device",
            "cpu",
            "--iterations",
            "2",
            "--num-envs",
            "4",
            "--rollout-size",
            "32",
            "--batch-size",
            "16",
            "--gradient-steps",
            "2",
            "--eval-envs",
            "4",
            "--eval-seconds",
            "1",
            "--cpu-threads",
            "1",
            "--output",
            str(output),
            *(["--require-learning"] if require_learning else []),
        ],
        cwd=regression.ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if require_learning:
        assert completed.returncode != 0
        assert "learning regression failed" in completed.stderr
    else:
        assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert manifest["mode"] == ("regression" if require_learning else "calibration")
    # This tiny optimizer/checkpoint smoke cannot qualify as swing-up learning.
    # A blocking run must retain its artifacts and fail after measuring that.
    assert not summary["learning_passed"]
    assert manifest["checkpoints"] == [0, 2]
    assert manifest["env_cfg"]["sim_dt"] == 0.01
    assert manifest["runner_cfg"]["actor"]["frequency"] == 100
    assert manifest["runner_cfg"]["algorithm"]["gamma"] == 0.9875
    assert manifest["runner_cfg"]["algorithm"]["lam"] == 0.995
    assert summary["training"]["collected_samples"] == 64
    assert summary["checkpoint_reload_and_resume_passed"]
    assert summary["resumed_iteration"] == 3
    assert summary["training"]["algorithm_actor_all_finite"]
    run_dir = Path(summary["run_dir"])
    initial = torch.load(run_dir / "model_0.pt", weights_only=True)
    trained = torch.load(run_dir / "model_2.pt", weights_only=True)
    resumed = torch.load(output / "resume" / "model_3.pt", weights_only=True)
    for name in ("actor", "critic"):
        key = f"{name}_state_dict"
        assert regression.tensor_hash(initial[key]) == manifest["initial_hashes"][name]
        assert regression.tensor_hash(initial[key]) != regression.tensor_hash(
            trained[key]
        )
    for name in ("optimizer_state_dict", "critic_optimizer_state_dict"):
        assert trained[name]["state"]
        for parameter, state in trained[name]["state"].items():
            assert resumed[name]["state"][parameter]["step"] == state["step"] + 2
    for iteration in (0, 2):
        with np.load(output / f"evaluation_{iteration}.npz") as artifact:
            assert artifact["checkpoint_iteration"] == iteration
            assert artifact["state"].shape == (101, 4, 2)
            assert np.isfinite(artifact["values"]).all()
            assert np.abs(artifact["applied_torques"]).max() <= 5
