"""COM positions/velocities with model-axis orientations, independent of adapters.

The asymmetric two-link asset separates COM, joint and principal-inertia frames.
Canonical order deliberately puts the floating base last. Kinematic derivatives
use prescribed nearby poses, not a time integrator or production tensor helpers.
"""

import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from tests.unit_tests.conftest import _make_mini_cheetah_cfg, vsim_guard

from gym.envs.base.mujoco_backend_base import MuJocoBackendBase


NUM_ENVS = 4
BODY_NAMES = ["arm", "base"]
BASE_COM = np.array([0.13, -0.09, 0.07])
ARM_COM = np.array([0.18, 0.04, -0.11])
HINGE_POSITION = np.array([0.4, 0.05, -0.03])
COLLISION_OFFSETS = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("warp", marks=pytest.mark.warp),
        pytest.param("vsim", marks=pytest.mark.vsim),
    ]
)
def backend(request, tmp_path):
    cfg = _make_mini_cheetah_cfg(sim_dt=0.002)
    cfg.asset.file = str(Path(__file__).parent / "assets" / "offset_com.urdf")
    cfg.asset.foot_name = None
    cfg.asset.joint_damping = 0.0
    cfg.asset.penalize_contacts_on = []
    cfg.asset.terminate_after_contacts_on = []
    cfg.asset.robot_layout = SimpleNamespace(
        version="offset_com_v1",
        dof_names=["hinge"],
        body_names=BODY_NAMES,
    )
    cfg.sim.gravity = [0.0, 0.0, 0.0]
    # Allocate per-world models so the same fixture also exercises mass DR.
    cfg.domain_randomization.startup.link_mass_scale_range = [0.5, 2.0]
    if request.param == "cpu":
        from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

        instance, device = MuJocoCPUBackend(), "cpu"
    elif request.param == "warp":
        from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

        instance, device = MuJocoWarpBackend(), "cuda:0"
    else:
        vsim_guard()
        from gym.envs.base.vsim_backend import VSimBackend

        # Conversion writes a sibling vsim directory; keep generated assets out
        # of the source tree without changing the URDF's physical definition.
        asset_directory = tmp_path / "urdf"
        asset_directory.mkdir()
        asset_path = asset_directory / "offset_com.urdf"
        shutil.copyfile(cfg.asset.file, asset_path)
        cfg.asset.file = str(asset_path)
        initial = _make_root_states()[0]
        cfg.init_state = SimpleNamespace(
            pos=initial[:3].tolist(), rot=initial[3:7].tolist()
        )
        instance, device = VSimBackend(), "cuda:0"
    try:
        instance.setup(cfg, NUM_ENVS, device)
        yield instance
    finally:
        instance.close()


def _make_rotation(vector):
    """Rodrigues rotation; vector direction is the world rotation axis."""
    angle = np.linalg.norm(vector)
    if angle == 0:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross


def _get_rotation(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion)[[3, 0, 1, 2]])
    return matrix.reshape(3, 3)


def _make_root_states():
    roots = np.zeros((NUM_ENVS, 13))
    for index in range(NUM_ENVS):
        roots[index, :3] = [index * 0.3, -index * 0.2, 4.0 + index * 0.1]
        quaternion = np.empty(4)
        rotation = _make_rotation(np.array([0.2, -0.3, 0.4]) * (index + 1))
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        roots[index, 3:7] = quaternion[[1, 2, 3, 0]]
        roots[index, 7:10] = [0.2, -0.1, 0.3]
        roots[index, 10:13] = [0.4, -0.2, 0.1]
    return roots


def _write_states(backend, roots, angles, speeds):
    backend.root_states[:] = torch.tensor(roots, device=backend.device)
    backend.dof_pos[:, 0] = torch.tensor(angles, device=backend.device)
    backend.dof_vel[:, 0] = torch.tensor(speeds, device=backend.device)


def _reset_all(backend):
    backend.reset_state(torch.ones(NUM_ENVS, dtype=torch.bool, device=backend.device))


def _compute_poses(root, angle, speed=0.0, time=0.0):
    # Float32 state snapshots cannot resolve the 1e-5 s derivative perturbation.
    root = np.asarray(root, dtype=np.float64)
    angle, speed = float(angle), float(speed)
    rotation = _make_rotation(time * root[10:13]) @ _get_rotation(root[3:7])
    base_COM = root[:3] + time * root[7:10]
    base_origin = base_COM - rotation @ BASE_COM
    arm_rotation = rotation @ _make_rotation(np.array([0.0, angle + time * speed, 0.0]))
    arm_origin = base_origin + rotation @ HINGE_POSITION
    positions = np.stack([arm_origin + arm_rotation @ ARM_COM, base_COM])
    return (
        positions,
        np.stack([arm_rotation, rotation]),
        np.stack([arm_origin, base_origin]),
    )


def _get_native_data(backend):
    if backend.device == "cpu":
        return list(zip(backend._models, backend._datas, strict=True))
    # Reconstruct kinematics from Warp generalized state, not its public caches.
    pairs = []
    for position, velocity in zip(
        backend._qpos_t.cpu().numpy(), backend._qvel_t.cpu().numpy(), strict=True
    ):
        data = mujoco.MjData(backend._mjm)
        data.qpos[:] = position
        data.qvel[:] = velocity
        mujoco.mj_forward(backend._mjm, data)
        pairs.append((backend._mjm, data))
    return pairs


def _get_vsim_native_states(backend):
    """Query native frames separately, without using assembled adapter buffers."""
    import vlearn as v

    model = torch.empty(NUM_ENVS, 2, 7, device=backend.device)
    center = torch.empty_like(model)
    velocity = torch.empty(NUM_ENVS, 2, 6, device=backend.device)
    transforms = [
        backend._grp.create_link_transform_command(
            v.wrap_gpu_buffer(buffer),
            backend._art_h,
            (0, 2),
            transform_type=frame,
        )
        for buffer, frame in (
            (model, v.TransformType.MODEL),
            (center, v.TransformType.COM),
        )
    ]
    velocities = backend._grp.create_link_velocity_command(
        v.wrap_gpu_buffer(velocity), backend._art_h, (0, 2)
    )
    backend._gym.get_link_transforms(backend._gym.create_gpu_array(transforms))
    backend._gym.get_link_velocities(backend._gym.create_gpu_array([velocities]))
    order = [backend._native_body_names.index(name) for name in BODY_NAMES]
    return tuple(buffer.cpu().numpy()[:, order] for buffer in (model, center, velocity))


def _assert_native_states(backend):
    bodies = backend.rigid_body_states.view(NUM_ENVS, 2, 13).cpu().numpy()
    if not isinstance(backend, MuJocoBackendBase):
        model, center, velocity = _get_vsim_native_states(backend)
        np.testing.assert_allclose(bodies[..., :3], center[..., 4:7], atol=2e-6)
        np.testing.assert_allclose(
            bodies[..., 7:13], velocity[..., [3, 4, 5, 0, 1, 2]], atol=3e-6, rtol=1e-5
        )
        for state, origin, inertia in zip(
            bodies.reshape(-1, 13),
            model.reshape(-1, 7),
            center.reshape(-1, 7),
            strict=True,
        ):
            np.testing.assert_allclose(
                _get_rotation(state[3:7]), _get_rotation(origin[:4]), atol=2e-6
            )
            assert not np.allclose(
                _get_rotation(origin[:4]), _get_rotation(inertia[:4]), atol=0.01
            )
        np.testing.assert_allclose(backend.root_states.cpu(), bodies[:, 1], atol=2e-6)
        return
    for env_id, (model, data) in enumerate(_get_native_data(backend)):
        for canonical_id, name in enumerate(BODY_NAMES):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            state = bodies[env_id, canonical_id]
            np.testing.assert_allclose(state[:3], data.xipos[body_id], atol=2e-6)
            np.testing.assert_allclose(
                _get_rotation(state[3:7]), data.xmat[body_id].reshape(3, 3), atol=2e-6
            )
            # Inertia axes intentionally differ: copying xiquat would be wrong.
            assert not np.allclose(data.xmat[body_id], data.ximat[body_id], atol=0.01)
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
            )
            np.testing.assert_allclose(
                state[7:13], np.r_[velocity[3:], velocity[:3]], atol=3e-6, rtol=1e-5
            )
    np.testing.assert_allclose(backend.root_states.cpu(), bodies[:, 1], atol=2e-6)


def _assert_analytic_poses(backend, body_states, roots, angles):
    bodies = body_states.view(NUM_ENVS, 2, 13).cpu().numpy()
    is_mujoco = isinstance(backend, MuJocoBackendBase)
    native_data = (
        _get_native_data(backend) if is_mujoco else _get_vsim_native_states(backend)
    )
    for index, (root, angle) in enumerate(zip(roots, angles, strict=True)):
        positions, rotations, origins = _compute_poses(root, angle)
        np.testing.assert_allclose(bodies[index, :, :3], positions, atol=2e-6)
        for body, rotation in zip(bodies[index], rotations, strict=True):
            np.testing.assert_allclose(_get_rotation(body[3:7]), rotation, atol=2e-6)
        if not is_mujoco:
            # VSim exposes link MODEL transforms, not individual collision poses.
            # Their positions locate the original collision/joint frames directly.
            np.testing.assert_allclose(
                native_data[0][index, :, 4:7], origins, atol=2e-6
            )
            continue
        model, data = native_data[index]
        for name, origin, rotation, collision_offset in zip(
            BODY_NAMES, origins, rotations, COLLISION_OFFSETS, strict=True
        ):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            np.testing.assert_allclose(data.xpos[body_id], origin, atol=2e-6)
            geometry_id = np.flatnonzero(model.geom_bodyid == body_id)[0]
            np.testing.assert_allclose(
                data.geom_xpos[geometry_id],
                origin + rotation @ collision_offset,
                atol=2e-6,
            )


def test_com_poses_keep_model_axes_and_canonical_order(backend):
    assert backend.body_names == BODY_NAMES
    _assert_native_states(backend)  # setup must publish COM state before any reset
    if not isinstance(backend, MuJocoBackendBase):
        # VSim consumes cfg.init_state during native construction. Its requested
        # COM pose must locate the same physical model before any reset call.
        initial = np.repeat(_make_root_states()[0:1], NUM_ENVS, axis=0)
        _assert_analytic_poses(
            backend, backend.rigid_body_states, initial, np.zeros(NUM_ENVS)
        )
    roots, angles = _make_root_states(), np.array([0.2, -0.4, 0.6, -0.1])
    _write_states(backend, roots, angles, np.zeros(NUM_ENVS))
    _reset_all(backend)
    _assert_analytic_poses(backend, backend.rigid_body_states, roots, angles)
    backend.step(torch.zeros_like(backend.dof_pos))
    _assert_native_states(backend)


def test_moving_joint_com_velocities_match_kinematic_derivatives(backend):
    roots = _make_root_states()
    angles, speeds = np.array([0.2, -0.4, 0.6, -0.1]), np.array([0.7, -0.6, 0.2, -0.5])
    _write_states(backend, roots, angles, speeds)
    _reset_all(backend)
    epsilon = 1e-5
    for _ in range(2):
        bodies = backend.rigid_body_states.view(NUM_ENVS, 2, 13).cpu().numpy()
        for index, (root, angle, speed) in enumerate(
            zip(roots, angles, speeds, strict=True)
        ):
            ahead = _compute_poses(root, angle, speed, epsilon)[0]
            behind = _compute_poses(root, angle, speed, -epsilon)[0]
            np.testing.assert_allclose(
                bodies[index, :, 7:10], (ahead - behind) / (2 * epsilon), atol=3e-6
            )
        _assert_native_states(backend)
        backend.step(torch.zeros_like(backend.dof_pos))
        roots = backend.root_states.cpu().numpy().copy()
        angles = backend.dof_pos[:, 0].cpu().numpy().copy()
        speeds = backend.dof_vel[:, 0].cpu().numpy().copy()


def test_com_reset_is_atomic_and_preserves_physical_geometry(backend):
    roots = _make_root_states()
    angles, speeds = np.array([0.2, -0.4, 0.6, -0.1]), np.array([0.7, -0.6, 0.2, -0.5])
    _write_states(backend, roots, angles, speeds)
    _reset_all(backend)
    cached_root, cached_bodies = backend.root_states, backend.rigid_body_states
    cached_dofs = backend.dof_state
    pointers = [state.data_ptr() for state in (cached_root, cached_bodies, cached_dofs)]
    bodies = cached_bodies.view(NUM_ENVS, 2, 13)
    # VSim's compute_kinematics alone rounds native COM/MODEL poses and velocities
    # again (~0.1--0.5 micro-units on this fixture, also before the COM migration).
    # Allow two float32 ULP at 4 m only for recomputed kinematics; joints stay exact.
    isolation_tolerance = 0 if isinstance(backend, MuJocoBackendBase) else 1e-6
    unchanged = bodies[[1, 3]].clone()
    unchanged_dofs = cached_dofs[[1, 3]].clone()
    roots[[0, 2], :3] += [0.5, -0.4, 0.3]
    roots[[0, 2], 3:7] = roots[[1, 3], 3:7]
    roots[[0, 2], 7:13] *= -1
    angles[[0, 2]] += 0.3
    speeds[[0, 2]] *= -1
    _write_states(backend, roots, angles, speeds)
    backend.reset_state(torch.tensor([True, False, True, False], device=backend.device))
    # Inspect cached tensors before a property getter could conceal stale state.
    _assert_analytic_poses(backend, cached_bodies, roots, angles)
    _assert_native_states(backend)
    torch.testing.assert_close(
        bodies[[1, 3]], unchanged, atol=isolation_tolerance, rtol=0
    )
    torch.testing.assert_close(cached_dofs[[1, 3]], unchanged_dofs, atol=0, rtol=0)
    np.testing.assert_allclose(
        cached_dofs.view(NUM_ENVS, 1, 2)[:, 0].cpu(), np.c_[angles, speeds], atol=2e-6
    )

    unchanged = bodies[[0, 2, 3]].clone()
    unchanged_dofs = cached_dofs[[0, 2, 3]].clone()
    roots[1, :3] += [0.1, 0.3, -0.2]
    roots[1, 3:7] = roots[2, 3:7]
    angles[1] -= 0.2
    speeds[1] *= -1
    _write_states(backend, roots, angles, speeds)
    backend.reset_state(
        torch.tensor([False, True, False, False], device=backend.device)
    )
    _assert_analytic_poses(backend, cached_bodies, roots, angles)
    _assert_native_states(backend)
    torch.testing.assert_close(
        bodies[[0, 2, 3]], unchanged, atol=isolation_tolerance, rtol=0
    )
    torch.testing.assert_close(cached_dofs[[0, 2, 3]], unchanged_dofs, atol=0, rtol=0)

    cached = (cached_root, cached_bodies, cached_dofs)
    before = tuple(state.clone() for state in cached)
    backend.reset_state(torch.zeros(NUM_ENVS, dtype=torch.bool, device=backend.device))
    for actual, expected in zip(cached[:2], before[:2], strict=True):
        torch.testing.assert_close(actual, expected, atol=isolation_tolerance, rtol=0)
    torch.testing.assert_close(cached_dofs, before[2], atol=0, rtol=0)
    _assert_analytic_poses(backend, cached_bodies, roots, angles)

    # Root-only updates must use the newly requested orientation and keep joints.
    roots[:, 3:7] = roots[::-1, 3:7].copy()
    roots[:, :3] += [0.2, 0.1, -0.1]
    cached_root[:] = torch.tensor(roots, device=backend.device)
    backend.set_all_root_states()
    _assert_analytic_poses(backend, cached_bodies, roots, angles)
    _assert_native_states(backend)
    np.testing.assert_allclose(
        cached_dofs.view(NUM_ENVS, 1, 2)[:, 0].cpu(), np.c_[angles, speeds], atol=2e-6
    )
    assert pointers == [
        state.data_ptr()
        for state in (backend.root_states, backend.rigid_body_states, backend.dof_state)
    ]


def test_nonuniform_mass_randomization_does_not_redefine_link_coms(backend):
    scales = torch.tensor([[1, 1], [2, 1], [1, 2], [0.5, 1.5]], device=backend.device)
    backend.set_link_mass_scale(torch.arange(NUM_ENVS, device=backend.device), scales)
    roots = np.repeat(_make_root_states()[0:1], NUM_ENVS, axis=0)
    angles, speeds = np.full(NUM_ENVS, 0.4), np.full(NUM_ENVS, 0.7)
    _write_states(backend, roots, angles, speeds)
    _reset_all(backend)
    _assert_analytic_poses(backend, backend.rigid_body_states, roots, angles)
    _assert_native_states(backend)
    bodies = backend.rigid_body_states.view(NUM_ENVS, 2, 13)
    torch.testing.assert_close(
        bodies, bodies[0:1].expand_as(bodies), atol=2e-6, rtol=1e-5
    )
    masses = backend.link_mass
    aggregate_COM = (masses[..., None] * bodies[..., :3]).sum(1) / masses.sum(
        1, keepdim=True
    )
    assert torch.linalg.vector_norm(aggregate_COM[0] - aggregate_COM[1]) > 0.02
    backend.step(torch.zeros_like(backend.dof_pos))
    _assert_native_states(backend)
