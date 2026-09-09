from math import sqrt
import torch
import numpy as np

from gym.envs.base.fixed_robot import FixedRobot
from gym.utils.sampling import masked_update


class Pendulum(FixedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.dof_pos_obs = torch.zeros(self.num_envs, 2, device=self.device)

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self.dof_pos_obs = torch.cat([self.dof_pos.sin(), self.dof_pos.cos()], dim=1)

    def _reset_system(self, reset_mask):
        super()._reset_system(reset_mask)
        masked_update(
            self.dof_pos_obs,
            torch.cat([self.dof_pos.sin(), self.dof_pos.cos()], dim=1),
            reset_mask,
        )

    def _check_terminations_and_timeouts(self):
        super()._check_terminations_and_timeouts()
        self.terminated.copy_(self.timed_out)

    def reset_to_uniform(self, reset_mask):
        grid_points = int(sqrt(self.num_envs))
        lin_pos = torch.linspace(
            self.dof_pos_range[0, 0],
            self.dof_pos_range[0, 1],
            grid_points,
            device=self.device,
        )
        lin_vel = torch.linspace(
            self.dof_vel_range[0, 0],
            self.dof_vel_range[0, 1],
            grid_points,
            device=self.device,
        )
        grid = torch.cartesian_prod(lin_pos, lin_vel)
        masked_update(self.dof_pos, grid[:, 0].unsqueeze(-1), reset_mask)
        masked_update(self.dof_vel, grid[:, 1].unsqueeze(-1), reset_mask)

    def _reward_theta(self):
        theta_err = 1.0 - torch.cos(self.dof_pos[:, 0])
        return self._sqrdexp(theta_err)

    def _reward_omega(self):
        omega_rwd = torch.square(self.dof_vel[:, 0] / self.scales["dof_vel"])
        return self._sqrdexp(omega_rwd)

    def _reward_equilibrium(self):
        # todo compare alternatives
        error = torch.abs(self.dof_state)
        error[:, 0] /= self.scales["dof_pos"]
        error[:, 1] /= self.scales["dof_vel"]
        return self._sqrdexp(torch.mean(error, dim=1), sigma=0.01)

    def _reward_torques(self):
        """Penalize torques"""
        return self._sqrdexp(torch.mean(torch.square(self.torques), dim=1), sigma=0.2)

    def _reward_energy(self):
        kinetic_energy = (
            0.5
            * self.cfg.asset.mass
            * self.cfg.asset.length**2
            * torch.square(self.dof_vel[:, 0])
        )
        potential_energy = (
            self.cfg.asset.mass
            * 9.81
            * self.cfg.asset.length
            * torch.cos(self.dof_pos[:, 0])
        )
        desired_energy = self.cfg.asset.mass * 9.81 * self.cfg.asset.length
        energy_error = kinetic_energy + potential_energy - desired_energy
        return self._sqrdexp(energy_error / desired_energy)

    def _normalize_theta(self):
        # normalize to range [-pi, pi]
        theta = self.dof_pos[:, 0]
        return ((theta + np.pi) % (2 * np.pi)) - np.pi
