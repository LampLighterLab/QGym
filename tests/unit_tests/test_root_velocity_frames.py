"""World-frame floating-root velocities at 100 Hz, including rotated resets.

MuJoCo free-joint angular qvel is body-local. A public write/read round trip
alone can therefore hide matching errors on both sides of the backend boundary.
Use native object velocities and rotation increments as independent oracles.
"""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from tests.unit_tests.conftest import _make_mini_cheetah_cfg, vsim_guard

from gym.envs.base.mujoco_backend_base import MuJocoBackendBase


NUM_ENVS = 4
DT = 0.01
BACKENDS = [
    "cpu",
    pytest.param("warp", marks=pytest.mark.warp),
    pytest.param("vsim", marks=pytest.mark.vsim),
]


@pytest.fixture(params=BACKENDS)
def backend(request):
    cfg = _make_mini_cheetah_cfg(sim_dt=DT)
    cfg.sim.gravity = [0.0, 0.0, 0.0]
    cfg.asset.joint_damping = 0.0
    cfg.init_state = SimpleNamespace(pos=[0.0, 0.0, 5.0], rot=[0.0, 0.0, 0.0, 1.0])
    if request.param == "cpu":
        from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

        instance, device = MuJocoCPUBackend(), "cpu"
    elif request.param == "warp":
        from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

        instance, device = MuJocoWarpBackend(), "cuda:0"
    else:
        vsim_guard()
        from gym.envs.base.vsim_backend import VSimBackend

        instance, device = VSimBackend(), "cuda:0"
    try:
        instance.setup(cfg, num_envs=NUM_ENVS, device=device, task=None)
        yield instance
    finally:
        instance.close()


def _orientations():
    # Construct rotations independently of the quaternion helpers used by tasks
    # and backends. Yaw tests expose swapped axes; tilt also exposes yaw errors.
    result = []
    for roll, pitch, yaw in (
        (0.0, 0.0, math.pi / 2),
        (0.0, 0.0, -math.pi / 2),
        (0.4, -0.35, 0.6),
        (-0.3, 0.45, -0.8),
    ):
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, (rz @ ry @ rx).ravel())
        result.append(quat[[1, 2, 3, 0]])
    return np.array(result)


def _rotation(quaternion):
    matrix = np.empty(9)
    quat = np.asarray(quaternion, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quat[[3, 0, 1, 2]])
    return matrix.reshape(3, 3)


def _write_rotated_states(backend):
    state = backend.root_states
    state.zero_()
    state[:, 2] = 5.0
    state[:, 3:7] = torch.as_tensor(
        _orientations(), dtype=state.dtype, device=state.device
    )
    state[:, 7:10] = torch.tensor([0.2, -0.1, 0.3], device=state.device)
    state[:, 10:13] = torch.tensor(
        [[0.3, -0.2, 0.1], [-0.2, 0.1, 0.3], [0.1, 0.3, -0.2], [-0.1, -0.2, 0.3]],
        device=state.device,
    )
    backend.dof_vel.zero_()
    return state.clone()


def _all_mask(backend):
    return torch.ones(NUM_ENVS, dtype=torch.bool, device=backend.device)


def _assert_mujoco_native_velocity(backend, expected):
    # VSim is checked with an engine-independent orientation increment below.
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    if not isinstance(backend, MuJocoBackendBase):
        return
    if isinstance(backend, MuJocoCPUBackend):
        pairs = [(backend._model_for_env(i), d) for i, d in enumerate(backend._datas)]
    else:
        # Reconstruct CPU kinematics from Warp's native generalized state,
        # independently of Warp's assembled public root and body tensors.
        pairs = []
        qpos, qvel = backend._qpos_t.cpu().numpy(), backend._qvel_t.cpu().numpy()
        for i in range(NUM_ENVS):
            data = mujoco.MjData(backend._mjm)
            data.qpos[:] = qpos[i]
            data.qvel[:] = qvel[i]
            mujoco.mj_forward(backend._mjm, data)
            pairs.append((backend._mjm, data))
    for i, (model, data) in enumerate(pairs):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        velocity = np.empty(6)
        # XBODY is the body-origin frame; BODY uses the inertial/COM origin.
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_XBODY, body_id, velocity, 0
        )
        np.testing.assert_allclose(
            velocity,
            expected[i, [10, 11, 12, 7, 8, 9]].cpu().numpy(),
            atol=2e-6,
            rtol=1e-5,
        )


def _assert_root_body_velocity(backend, root, bodies):
    base_id = backend.body_names.index("base")
    torch.testing.assert_close(
        root[:, 7:13], bodies[:, base_id, 7:13], atol=2e-6, rtol=1e-5
    )


def _assert_world_rotation_increment(before, after):
    before = before.detach().cpu().numpy()
    after = after.detach().cpu().numpy()
    for previous, current in zip(before, after, strict=True):
        # R_next R_previous^T is a WORLD-frame increment. Reversing this
        # multiplication would measure body-local angular velocity instead.
        delta = _rotation(current[3:7]) @ _rotation(previous[3:7]).T
        omega = np.array(
            [
                delta[2, 1] - delta[1, 2],
                delta[0, 2] - delta[2, 0],
                delta[1, 0] - delta[0, 1],
            ]
        ) / (2 * DT)
        # This one-step derivative differs from instantaneous velocity because
        # of integration and the articulated robot's angular acceleration.
        # At these modest rates, 0.005 rad/s admits that discretization while
        # excluding the 0.1--0.5 rad/s wrong-frame errors in the chosen poses.
        np.testing.assert_allclose(omega, previous[10:13], atol=0.005, rtol=0)


def _step_and_check(backend, before):
    root = backend.root_states
    bodies = backend.rigid_body_states.view(NUM_ENVS, backend.num_bodies, 13)
    backend.step(torch.zeros(NUM_ENVS, backend.num_dof, device=backend.device))
    _assert_world_rotation_increment(before, root)
    _assert_root_body_velocity(backend, root, bodies)
    _assert_mujoco_native_velocity(backend, root)


@pytest.mark.parametrize("commit", ["reset_state", "set_all_root_states"])
def test_rotated_root_reset_accepts_world_velocity(backend, commit):
    expected = _write_rotated_states(backend)
    if commit == "reset_state":
        backend.reset_state(_all_mask(backend))
    else:
        backend.set_all_root_states()
    _assert_mujoco_native_velocity(backend, expected)
    torch.testing.assert_close(backend.root_states[:, 7:13], expected[:, 7:13])
    _assert_root_body_velocity(
        backend,
        backend.root_states,
        backend.rigid_body_states.view(NUM_ENVS, backend.num_bodies, 13),
    )
    _step_and_check(backend, expected)


def test_partial_reset_uses_new_orientation_and_preserves_other_environments(backend):
    _write_rotated_states(backend)
    backend.reset_state(_all_mask(backend))
    backend.step(torch.zeros(NUM_ENVS, backend.num_dof, device=backend.device))
    cached_root = backend.root_states
    expected = cached_root.clone()
    selected = torch.tensor([0, 2], device=backend.device)
    new_quaternions = torch.as_tensor(
        _orientations()[[1, 0, 3, 2]], dtype=expected.dtype, device=backend.device
    )
    expected[selected, 2] += 1.0
    expected[selected, 3:7] = new_quaternions[selected]
    expected[selected, 7:13] *= -1.0
    cached_root[selected] = expected[selected]
    mask = torch.zeros(NUM_ENVS, dtype=torch.bool, device=backend.device)
    mask[selected] = True
    backend.reset_state(mask)

    _assert_mujoco_native_velocity(backend, expected)
    torch.testing.assert_close(cached_root, expected, atol=2e-6, rtol=1e-5)
    _step_and_check(backend, expected)


def test_step_updates_cached_world_root_and_body_velocities(backend):
    expected = _write_rotated_states(backend)
    backend.reset_state(_all_mask(backend))
    cached_root = backend.root_states
    cached_bodies = backend.rigid_body_states.view(NUM_ENVS, backend.num_bodies, 13)
    root_ptr, body_ptr = cached_root.data_ptr(), cached_bodies.data_ptr()
    before = expected
    for _ in range(3):
        backend.step(torch.zeros(NUM_ENVS, backend.num_dof, device=backend.device))
        # Do not access state properties before inspecting the saved references:
        # a getter-side refresh must not rescue a stale public tensor.
        assert not torch.allclose(cached_root[:, 3:7], before[:, 3:7])
        _assert_world_rotation_increment(before, cached_root)
        _assert_root_body_velocity(backend, cached_root, cached_bodies)
        _assert_mujoco_native_velocity(backend, cached_root)
        before = cached_root.clone()
    assert backend.root_states.data_ptr() == root_ptr
    assert backend.rigid_body_states.data_ptr() == body_ptr
