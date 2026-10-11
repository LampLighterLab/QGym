"""SAC task alignment and real optimizer/checkpoint/resume behavior."""

import copy
import json

import pytest
import torch

from gym.envs.pendulum.pendulum_config import PendulumCfg, PendulumRunnerCfg
from gym.envs.pendulum.pendulum_SAC_config import PendulumSACCfg, PendulumSACRunnerCfg
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from learning.runners.off_policy_runner import storage


@pytest.fixture
def runner(tmp_path):
    cfg, train_cfg = PendulumSACCfg(), PendulumSACRunnerCfg()
    cfg.seed = 7
    cfg.env.num_envs = 4
    train_cfg.log_dir = str(tmp_path)
    train_cfg.algorithm.initial_fill = 2
    train_cfg.algorithm.storage_size = 32
    train_cfg.algorithm.batch_size = 8
    train_cfg.algorithm.max_gradient_steps = 2
    train_cfg.runner.num_steps_per_env = 2
    train_cfg.runner.max_iterations = 2
    train_cfg.runner.save_interval = 1
    task_registry.convert_frequencies_to_params(cfg, train_cfg)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env = task_registry.make_env("sac_pendulum", cfg, "cpu", True, "mujoco")
    try:
        yield task_registry.make_alg_runner(env, train_cfg)
    finally:
        env._backend.close()
        torch.set_num_threads(old_threads)


def test_sac_uses_ppo_physical_task_and_rewards(runner):
    sac_env, ppo_env = PendulumSACCfg(), PendulumCfg()
    # Parallel population is learner geometry, not a physical task setting.
    sac_env.env.num_envs = ppo_env.env.num_envs
    assert class_to_dict(sac_env) == class_to_dict(ppo_env)
    sac, ppo = PendulumSACRunnerCfg(), PendulumRunnerCfg()
    assert class_to_dict(sac.critic.reward) == class_to_dict(ppo.critic.reward)
    assert sac.critic.obs == ppo.critic.obs
    assert sac.critic.normalize_obs == ppo.critic.normalize_obs
    assert sac.actor.frequency == ppo.actor.frequency
    assert sac.actor.obs == ppo.actor.obs
    assert class_to_dict(sac.actor.noise) == class_to_dict(ppo.actor.noise)
    assert sum(isinstance(m, torch.nn.ReLU) for m in runner.alg.actor.modules()) == 2
    assert not any(
        isinstance(m, torch.nn.LayerNorm) for m in runner.alg.actor.modules()
    )
    with torch.no_grad():
        actions = runner.alg.act(runner.get_obs(runner.actor_cfg["obs"]))
    assert actions.abs().max() <= 1
    runner.set_actions(runner.actor_cfg["actions"], torch.ones(4, 1))
    torch.testing.assert_close(runner.env.tau_ff, torch.full((4, 1), 5.0))


def test_training_restores_targets_temperature_and_optimizer_in_place(runner, tmp_path):
    runner.learn()
    assert runner.it == 2
    assert storage.get_data().shape == (6, 4)  # Two warmup + four learned steps.
    assert len((tmp_path / "vitals.jsonl").read_text().splitlines()) == 2
    vitals = json.loads((tmp_path / "vitals.jsonl").read_text().splitlines()[-1])
    assert vitals["algorithm/alpha"] > 0

    batch = storage.get_data().clone()
    alpha_parameter = runner.alg.log_alpha
    with torch.no_grad():
        runner.alg.log_alpha.fill_(3)
        for critic in (runner.alg.target_critic_1, runner.alg.target_critic_2):
            for parameter in critic.parameters():
                parameter.fill_(123)
    runner.load(tmp_path / "model_2.pt")
    assert runner.alg.log_alpha is alpha_parameter
    assert (
        runner.alg.log_alpha_optimizer.param_groups[0]["params"][0] is alpha_parameter
    )
    saved = torch.load(tmp_path / "model_2.pt", weights_only=True)
    torch.testing.assert_close(runner.alg.log_alpha, saved["log_alpha"])
    for name in ("target_critic_1", "target_critic_2"):
        torch.testing.assert_close(
            getattr(runner.alg, name).state_dict(), saved[f"{name}_state_dict"]
        )

    # A restored checkpoint must reproduce the next update on identical replay
    # data and RNG state, including Adam moments and entropy temperature.
    torch.manual_seed(73)
    runner.alg.update(batch)
    expected = {
        name: copy.deepcopy(getattr(runner.alg, name).state_dict())
        for name in (
            "actor",
            "critic_1",
            "critic_2",
            "target_critic_1",
            "target_critic_2",
        )
    }
    expected_alpha = runner.alg.log_alpha.detach().clone()
    runner.load(tmp_path / "model_2.pt")
    torch.manual_seed(73)
    runner.alg.update(batch)
    for name, state in expected.items():
        torch.testing.assert_close(
            getattr(runner.alg, name).state_dict(), state, rtol=0, atol=0
        )
    torch.testing.assert_close(runner.alg.log_alpha, expected_alpha, rtol=0, atol=0)

    runner.load(tmp_path / "model_2.pt")
    runner.num_learning_iterations = 1
    runner.learn()
    assert runner.it == 3
    resumed = torch.load(tmp_path / "model_3.pt", weights_only=True)
    for name in ("actor", "critic_1", "critic_2", "log_alpha"):
        key = f"{name}_optimizer_state_dict"
        for parameter, state in saved[key]["state"].items():
            assert resumed[key]["state"][parameter]["step"] == state["step"] + 2


def test_warmup_samples_configured_action_range(runner):
    runner.alg_cfg["initial_fill"] = 12
    runner.alg_cfg["storage_size"] = 64
    runner.alg.action_min = 0.25
    runner.alg.action_max = 0.5
    runner.alg.action_delta = 0.125
    runner.alg.action_offset = 0.375
    runner.num_learning_iterations = 0
    runner.learn()
    warmup_actions = storage.get_data()["actions"]
    assert warmup_actions.shape == (12, 4, 1)
    assert warmup_actions.min() >= 0.25
    assert warmup_actions.max() <= 0.5


@pytest.mark.parametrize("time_limit", [True, False])
def test_replay_preserves_final_state_and_bootstraps_only_time_limits(
    runner, time_limit
):
    env = runner.env

    def initial_state(mask):
        env.dof_pos[mask] = 1.0
        env.dof_vel[mask] = 0.0

    def reset_state(mask):
        env.dof_pos[mask] = 2.0
        env.dof_vel[mask] = 0.0

    def end_episode():
        env.timed_out.fill_(time_limit)
        # Pendulum also sets terminated on its time limit.
        env.terminated.fill_(True)

    env._reset_state = initial_state
    env.reset()
    env._reset_state = reset_state
    env._check_terminations_and_timeouts = end_episode
    runner.alg_cfg["initial_fill"] = 0
    runner.num_learning_iterations = 1
    runner.learn()
    data = storage.get_data()

    assert (data["dones"] == (not time_limit)).all()
    assert (data["timed_out"] == time_limit).all()
    # Final physical state remains near 1 rad; the next collected transition
    # starts from the exact reset at 2 rad. Neither may replace the other.
    next_obs = data["next_critic_obs"][0]
    final_angle = torch.atan2(next_obs[:, 0], next_obs[:, 1])
    assert ((final_angle > 0.9) & (final_angle < 1.1)).all()
    current_obs = data["critic_obs"][1]
    reset_angle = torch.atan2(current_obs[:, 0], current_obs[:, 1])
    torch.testing.assert_close(reset_angle, torch.full((4,), 2.0))
    if time_limit:
        assert (data["rewards"] > 0).all()
    else:
        assert (data["rewards"] == 0).all()
