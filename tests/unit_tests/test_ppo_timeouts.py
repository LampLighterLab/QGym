"""PPO collection preserves the physical successor at episode boundaries."""

import pytest
import torch

from gym.envs.pendulum.pendulum_config import PendulumCfg, PendulumRunnerCfg
from gym.utils.task_registry import task_registry


@pytest.mark.parametrize("time_limit", [True, False])
def test_ppo_preserves_final_observations_and_time_limit_reward(tmp_path, time_limit):
    cfg, train_cfg = PendulumCfg(), PendulumRunnerCfg()
    cfg.seed = 7
    cfg.env.num_envs = 4
    train_cfg.log_dir = str(tmp_path)
    train_cfg.algorithm.rollout_size = 8
    train_cfg.algorithm.batch_size = 8
    train_cfg.algorithm.max_gradient_steps = 1
    train_cfg.runner.max_iterations = 1
    task_registry.convert_frequencies_to_params(cfg, train_cfg)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env = task_registry.make_env("pendulum", cfg, "cpu", True, "mujoco")
    try:
        runner = task_registry.make_alg_runner(env, train_cfg)

        def initial_state(mask):
            env.dof_pos[mask] = 1.0
            env.dof_vel[mask] = 0.0

        def reset_state(mask):
            env.dof_pos[mask] = 2.0
            env.dof_vel[mask] = 0.0

        def end_episode():
            env.timed_out.fill_(time_limit)
            # Pendulum historically also marks a time limit as terminated.
            env.terminated.fill_(True)

        env._reset_state = initial_state
        env.reset()
        env._reset_state = reset_state
        env._check_terminations_and_timeouts = end_episode
        captured = []
        update = runner.alg.update

        def capture_rollout(data):
            captured.append(data.clone())
            update(data)

        runner.alg.update = capture_rollout
        runner.learn()
        data = captured[0]

        assert data["dones"].all()
        assert (data["timed_out"] == time_limit).all()
        assert (data["terminated"] == (not time_limit)).all()
        for key in ("next_actor_obs", "next_critic_obs"):
            final_angle = torch.atan2(data[key][0, :, 0], data[key][0, :, 1])
            assert ((final_angle > 0.9) & (final_angle < 1.1)).all()
        for key in ("actor_obs", "critic_obs"):
            reset_angle = torch.atan2(data[key][1, :, 0], data[key][1, :, 1])
            torch.testing.assert_close(reset_angle, torch.full((4,), 2.0))
        if time_limit:
            assert (data["rewards"] > 0).all()
        else:
            assert (data["rewards"] == 0).all()
    finally:
        env._backend.close()
        torch.set_num_threads(old_threads)
