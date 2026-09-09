"""CPU setup/reset caches are current without stepping, at 100 Hz."""

import math

import mujoco
import numpy as np
import pytest
import torch

from tests.unit_tests.conftest import _make_mini_cheetah_cfg, _make_pendulum_cfg

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend


NUM_ENVS = 4


@pytest.fixture
def backend():
    instance = MuJocoCPUBackend()
    try:
        instance.setup(_make_mini_cheetah_cfg(sim_dt=0.01), NUM_ENVS, "cpu")
        yield instance
    finally:
        instance.close()


def _cached(backend):
    return {
        "root": backend.root_states,
        "dof": backend.dof_state.view(NUM_ENVS, backend.num_dof, 2),
        "body": backend.rigid_body_states.view(NUM_ENVS, backend.num_bodies, 13),
        "contact": backend.contact_forces,
    }


def _native_reference(backend, env_id):
    model = backend._model_for_env(env_id)
    native = backend._datas[env_id]
    reference = mujoco.MjData(model)
    reference.qpos[:] = native.qpos
    reference.qvel[:] = native.qvel
    reference.qfrc_applied[:] = native.qfrc_applied
    reference.xfrc_applied[:] = native.xfrc_applied
    reference.qacc_warmstart[:] = native.qacc_warmstart
    mujoco.mj_forward(model, reference)
    mujoco.mj_rnePostConstraint(model, reference)
    return model, reference


def _assert_cached_matches_native(backend, cached, env_ids):
    for env_id in env_ids:
        model, reference = _native_reference(backend, env_id)
        dof_order = backend._canonical_to_native_dof_np
        expected_dof = np.stack(
            [
                reference.qpos[backend._qpos_offset :][dof_order],
                reference.qvel[backend._qvel_offset :][dof_order],
            ],
            axis=-1,
        )
        np.testing.assert_allclose(cached["dof"][env_id], expected_dof, atol=1e-6)
        for body_id, name in enumerate(backend.body_names):
            native_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(
                model, reference, mujoco.mjtObj.mjOBJ_XBODY, native_id, velocity, 0
            )
            expected_body = np.concatenate(
                [
                    reference.xpos[native_id],
                    reference.xquat[native_id][[1, 2, 3, 0]],
                    velocity[3:],
                    velocity[:3],
                ]
            )
            np.testing.assert_allclose(
                cached["body"][env_id, body_id], expected_body, atol=2e-6, rtol=1e-5
            )
            np.testing.assert_allclose(
                cached["contact"][env_id, body_id],
                reference.cfrc_ext[native_id, 3:6],
                atol=1e-4,
                rtol=1e-5,
            )
            if backend._has_free_joint and name == "base":
                np.testing.assert_allclose(
                    cached["root"][env_id], expected_body, atol=2e-6, rtol=1e-5
                )


def _place_in_air(backend):
    backend.root_states.zero_()
    backend.root_states[:, 2] = 3.0
    backend.root_states[:, 6] = 1.0
    backend.dof_pos.zero_()
    backend.dof_vel.zero_()
    backend.reset_state(torch.ones(NUM_ENVS, dtype=torch.bool))


def test_setup_publishes_current_native_state(backend):
    _assert_cached_matches_native(backend, _cached(backend), range(NUM_ENVS))
    assert all(data.time == 0 for data in backend._datas)


def test_fixed_base_setup_publishes_body_poses():
    backend = MuJocoCPUBackend()
    try:
        backend.setup(_make_pendulum_cfg(sim_dt=0.01), NUM_ENVS, "cpu")
        _assert_cached_matches_native(backend, _cached(backend), range(NUM_ENVS))
    finally:
        backend.close()


@pytest.mark.parametrize("root_only", [False, True], ids=["reset", "root_only"])
def test_reset_refreshes_cached_state_immediately(backend, root_only, monkeypatch):
    _place_in_air(backend)
    cached = _cached(backend)
    pointers = {name: tensor.data_ptr() for name, tensor in cached.items()}
    native_dof = [
        (
            data.qpos[backend._qpos_offset :].copy(),
            data.qvel[backend._qvel_offset :].copy(),
        )
        for data in backend._datas
    ]
    forward_calls = []
    original_forward = mujoco.mj_forward

    def record_forward(model, data):
        forward_calls.append(next(i for i, d in enumerate(backend._datas) if d is data))
        original_forward(model, data)

    monkeypatch.setattr(mujoco, "mj_forward", record_forward)
    backend.root_states[:, :3] = torch.tensor([0.4, -0.3, 5.0])
    backend.root_states[:, 3:7] = torch.tensor([0.0, math.sin(0.3), 0.0, math.cos(0.3)])
    backend.root_states[:, 7:13] = torch.tensor([0.2, -0.1, 0.3, 0.4, -0.2, 0.1])
    if root_only:
        backend.set_all_root_states()
        for data, before in zip(backend._datas, native_dof, strict=True):
            np.testing.assert_array_equal(data.qpos[backend._qpos_offset :], before[0])
            np.testing.assert_array_equal(data.qvel[backend._qvel_offset :], before[1])
    else:
        backend.dof_pos[:] = torch.linspace(-0.1, 0.1, backend.num_dof)
        backend.dof_vel[:] = torch.linspace(-0.2, 0.2, backend.num_dof)
        backend.reset_state(torch.ones(NUM_ENVS, dtype=torch.bool))
    assert forward_calls == list(range(NUM_ENVS))
    monkeypatch.undo()
    # Inspect saved references before touching any property getter again.
    _assert_cached_matches_native(backend, cached, range(NUM_ENVS))
    assert pointers == {
        name: tensor.data_ptr() for name, tensor in _cached(backend).items()
    }
    assert all(data.time == 0 for data in backend._datas)


def test_reset_refreshes_contact_forces_without_stepping(backend):
    _place_in_air(backend)
    cached = _cached(backend)
    mask = torch.tensor([True, False, False, False])
    backend.root_states[0, 2] = 0.3
    backend.reset_state(mask)
    assert cached["contact"][0].abs().max() > 1.0
    _assert_cached_matches_native(backend, cached, [0])

    backend.root_states[0, 2] = 3.0
    backend.reset_state(mask)
    torch.testing.assert_close(
        cached["contact"][0], torch.zeros_like(cached["contact"][0])
    )
    _assert_cached_matches_native(backend, cached, [0])


@pytest.mark.parametrize("selected", [[], [1, 3]])
def test_reset_only_refreshes_selected_environments(backend, monkeypatch, selected):
    _place_in_air(backend)
    backend.step(torch.zeros(NUM_ENVS, backend.num_dof))
    cached = _cached(backend)
    before = {name: tensor.clone() for name, tensor in cached.items()}
    native_before = [
        (data.qpos.copy(), data.qvel.copy(), data.cfrc_ext.copy(), data.time)
        for data in backend._datas
    ]
    calls = {
        name: [] for name in ("mj_forward", "mj_rnePostConstraint", "mj_kinematics")
    }
    for name in calls:
        original = getattr(mujoco, name)

        def record(model, data, *, operation=original, operation_name=name):
            calls[operation_name].append(
                next(i for i, d in enumerate(backend._datas) if d is data)
            )
            return operation(model, data)

        monkeypatch.setattr(mujoco, name, record)
    mask = torch.zeros(NUM_ENVS, dtype=torch.bool)
    mask[selected] = True
    backend.root_states[mask, 2] = 6.0
    backend.reset_state(mask)

    assert calls["mj_forward"] == selected
    assert calls["mj_rnePostConstraint"] == selected
    assert calls["mj_kinematics"] == []  # forward already refreshed kinematics
    monkeypatch.undo()
    for i in set(range(NUM_ENVS)) - set(selected):
        for name, tensor in cached.items():
            torch.testing.assert_close(tensor[i], before[name][i], rtol=0, atol=0)
        data = backend._datas[i]
        for value, expected in zip(
            (data.qpos, data.qvel, data.cfrc_ext), native_before[i][:3], strict=True
        ):
            np.testing.assert_array_equal(value, expected)
        assert data.time == native_before[i][3]
    _assert_cached_matches_native(backend, cached, selected)
