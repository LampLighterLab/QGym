"""Viewer arrows follow the task's live, scalar-last root pose."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import torch

visualizer_module = import_module("gym.utils.interfaces.CommandVisualizer")


def test_arrows_follow_canonical_root_pose_without_native_simulator_data(monkeypatch):
    env = SimpleNamespace(
        _backend=SimpleNamespace(_viewer_overlay_fn=None),
        commands=torch.tensor([[2.0, -1.0, 0.0]]),
        root_states=torch.zeros(1, 13),
    )
    env.root_states[0, :3] = torch.tensor([1.0, 2.0, 0.4])
    env.root_states[0, 5:7] = 2**-0.5  # xyzw: 90 degrees about world z.
    viewer = SimpleNamespace(user_scn=SimpleNamespace(ngeom=0))
    arrows = []

    def record_connector(viewer, geom_type, width, start, end, rgba):
        arrows.append((start.copy(), end.copy()))

    monkeypatch.setattr(visualizer_module, "_add_connector", record_connector)
    visualizer_module.CommandVisualizer(env)
    env._backend._viewer_overlay_fn(viewer)

    assert len(arrows) == 2
    np.testing.assert_allclose(arrows[0], [[1.0, 2.0, 0.8], [1.0, 2.6, 0.8]])
    np.testing.assert_allclose(arrows[1], [[1.0, 2.0, 0.8], [1.3, 2.0, 0.8]])

    env.root_states[0, 0] = 3.0
    env.root_states[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    arrows.clear()
    env._backend._viewer_overlay_fn(viewer)
    np.testing.assert_allclose(arrows[0], [[3.0, 2.0, 0.8], [3.6, 2.0, 0.8]])
