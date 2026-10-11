"""Pendulum rewards must respect the periodic state seen by the policy."""

import math

import pytest
import torch

from gym.envs.pendulum.pendulum import Pendulum


def equilibrium_reward(angles, velocities):
    pendulum = Pendulum.__new__(Pendulum)
    pendulum.dof_state = torch.stack(
        (torch.tensor(angles), torch.tensor(velocities)), dim=-1
    )
    pendulum.dof_pos = pendulum.dof_state[:, :1]
    pendulum.dof_vel = pendulum.dof_state[:, 1:]
    pendulum.scales = {"dof_pos": 2 * math.pi, "dof_vel": 5.0}
    return pendulum._reward_equilibrium()


@pytest.mark.parametrize("angle", [0.0, 0.2, math.pi])
def test_equilibrium_reward_is_invariant_to_full_revolutions(angle):
    angles = [angle + turns * 2 * math.pi for turns in (-2, -1, 0, 1, 2)]
    rewards = equilibrium_reward(angles, [0.3] * len(angles))

    torch.testing.assert_close(rewards, rewards[2].expand_as(rewards))


def test_equilibrium_reward_peaks_at_every_stationary_upright():
    rewards = equilibrium_reward([0.0, 2 * math.pi, -2 * math.pi], [0.0] * 3)

    torch.testing.assert_close(rewards, torch.ones(3))


def test_equilibrium_reward_decreases_with_angle_or_speed_error():
    rewards = equilibrium_reward([0.0, 0.2, 0.4, 0.0, 0.0], [0.0, 0.0, 0.0, 0.2, 0.4])

    assert 1.0 == rewards[0] > rewards[1] > rewards[2] > 0.0
    assert rewards[0] > rewards[3] > rewards[4] > 0.0
