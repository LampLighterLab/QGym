from types import SimpleNamespace

import torch

from gym.envs.go2.go2trot import Go2Trot


def test_contact_strength_uses_each_environments_applied_total_mass():
    task = Go2Trot.__new__(Go2Trot)
    task.feet_indices = torch.arange(4)
    task.contact_forces = torch.zeros(2, 4, 3)
    task.contact_forces[:, :, 2] = 98.1
    task._backend = SimpleNamespace(link_mass=torch.tensor([[5.0, 5.0], [10.0, 10.0]]))

    strength = task._foot_contact_strength()

    torch.testing.assert_close(strength[0], torch.ones(4))
    torch.testing.assert_close(strength[1], torch.full((4,), 0.5))
