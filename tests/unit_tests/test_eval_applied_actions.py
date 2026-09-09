"""Evaluation artifacts distinguish policy samples from applied commands."""

from pathlib import Path
import sys

import numpy as np
import torch

from gym import GYM_ROOT_DIR
from gym.envs.base.domain_randomization import apply_domain_randomization_override
from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg
from gym.utils.helpers import class_to_dict
from gym.utils.logging_and_saving.local_code_save_helper import (
    save_original_cfg_to_logs,
)
from gym.utils.task_registry import task_registry
from learning.modules import Actor, Critic
from scripts import eval_policy


def test_eval_saves_bounded_applied_commands_and_raw_policy_samples(
    tmp_path, monkeypatch
):
    env_cfg = Go2TrotCfg()
    train_cfg = Go2TrotRunnerCfg()
    env_cfg.seed = 7
    env_cfg.env.num_envs = 2
    env_cfg.control.ctrl_frequency = 100
    env_cfg.control.desired_sim_frequency = 100
    apply_domain_randomization_override(env_cfg, "off")
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    env = task_registry.make_env("go2trot", env_cfg, "cpu", True, "mujoco")
    checkpoint = tmp_path / "model_5.pt"
    samples = torch.linspace(-1e4, 1e4, env.num_actuators)
    try:
        actor = Actor(
            env.get_states(train_cfg.actor.obs).shape[-1],
            env.num_actuators,
            **class_to_dict(train_cfg.actor),
        )
        critic = Critic(
            env.get_states(train_cfg.critic.obs).shape[-1],
            **class_to_dict(train_cfg.critic),
        )
        with torch.no_grad():
            for parameter in actor.NN.parameters():
                parameter.zero_()
            actor.NN[-1].bias.copy_(samples)
        torch.save(
            {
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "iter": 5,
            },
            checkpoint,
        )
        limits = env.dof_pos_limits[env.actuated_dof_indices].numpy().copy()
        defaults = env.default_dof_pos[:, env.actuated_dof_indices].numpy().copy()
        joint_names = env.actuated_dof_names
        phase_offsets = np.repeat(
            [
                env_cfg.control.gait_phase_offsets[name]
                for name in env.robot_layout.body_groups["feet"]
            ],
            3,
        )
    finally:
        env._backend.close()
    save_original_cfg_to_logs(tmp_path, Path(GYM_ROOT_DIR) / "gym" / "envs")
    output = tmp_path / "evaluation.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_policy.py",
            "--task",
            "go2trot",
            "--ckpt",
            str(checkpoint),
            "--train_label",
            "constant-policy",
            "--eval_backend",
            "mujoco",
            "--eval_device",
            "cpu",
            "--num_envs",
            "2",
            "--t_end",
            "0.03",
            "--settling_time",
            "0",
            "--reset_mode",
            "reset_to_basic",
            "--domain-randomization",
            "off",
            "--record_policy_io",
            "--original_cfg",
            "--out",
            str(output),
        ],
    )

    eval_policy.main()

    with np.load(output) as artifact:
        assert float(artifact["ctrl_hz"]) == 100
        raw = artifact["policy_actions"]
        applied = artifact["applied_actions"]
        np.testing.assert_array_equal(raw, np.broadcast_to(samples.numpy(), (3, 2, 12)))
        actor_names = artifact["actor_observation_names"].tolist()
        actor_obs = artifact["actor_observations"]
        phase = np.arctan2(
            actor_obs[..., actor_names.index("phase_obs.sin")],
            actor_obs[..., actor_names.index("phase_obs.cos")],
        )
        reference = np.asarray(env_cfg.control.gait_joint_offsets) + np.asarray(
            env_cfg.control.gait_joint_amplitudes
        ) * np.sin(phase[..., None] + 2 * np.pi * phase_offsets)
        bound = np.where(samples.numpy() > 0, limits[:, 1], limits[:, 0])
        expected = bound - defaults - reference
        np.testing.assert_allclose(applied, expected, atol=5e-7, rtol=0)
        assert np.isfinite(applied).all()
        # Observations stay before the current action/step: the first target is
        # zero, and the second is the previously applied command in task units.
        for prefix in ("actor", "critic"):
            names = artifact[f"{prefix}_observation_names"].tolist()
            columns = [names.index(f"dof_pos_target.{name}") for name in joint_names]
            target_obs = artifact[f"{prefix}_observations"][..., columns]
            np.testing.assert_array_equal(target_obs[0], np.zeros((2, 12)))
            np.testing.assert_allclose(
                target_obs[1] * artifact["action_scales"], applied[0], atol=5e-7
            )
