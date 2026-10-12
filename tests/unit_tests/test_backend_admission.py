"""Engine-neutral floating-base checks for admitting an additional backend.

These complement the shared pendulum contract: they exercise simultaneous root
and joint writes, nonidentity orientations, reference-point velocities, and cached
state after integration. Add a real fixture to FLOATING_BACKENDS to reuse them;
CPU runs by default, Warp/VSim only through their explicit execution groups.
Contact-rich and Go2 task qualification remain separate gates.
"""

import torch

from tests.unit_tests.conftest import MINI_CHEETAH_SIM_DT


def _place_airborne(backend):
    backend.dof_pos.zero_()
    backend.dof_vel.zero_()
    backend.root_states.zero_()
    backend.root_states[:, 2] = 5.0
    orientation = torch.tensor([0.2, -0.3, 0.4, 0.8], device=backend.device)
    backend.root_states[:, 3:7] = orientation / orientation.norm()
    mask = torch.ones(4, dtype=torch.bool, device=backend.device)
    backend.reset_state(mask)
    return mask


def _get_public_tensors(backend):
    return (
        backend.dof_state,
        backend.dof_pos,
        backend.dof_vel,
        backend.root_states,
        backend.rigid_body_states,
        backend.contact_forces,
    )


def test_masked_atomic_reset_preserves_aliases_and_unselected_worlds(floating_backend):
    backend = floating_backend
    _place_airborne(backend)
    torques = torch.zeros_like(backend.dof_pos)
    backend.step(torques)

    cached = _get_public_tensors(backend)
    pointers = tuple(tensor.data_ptr() for tensor in cached)
    state = cached[0].view(4, backend.num_dof, 2)
    bodies = cached[4].view(4, backend.num_bodies, 13)
    mask = torch.tensor([True, False, True, False], device=backend.device)
    before = tuple(
        tensor[~mask].clone() for tensor in (state, backend.root_states, bodies)
    )

    # Test both write directions, not merely equal values copied on refresh.
    state[mask, :, 0] = torch.linspace(
        -0.1, 0.1, backend.num_dof, device=backend.device
    )
    torch.testing.assert_close(backend.dof_pos, state[..., 0], atol=0, rtol=0)
    backend.dof_vel[mask] = 0.2
    torch.testing.assert_close(state[..., 1], backend.dof_vel, atol=0, rtol=0)
    backend.root_states[mask, 2] = 3.0
    backend.root_states[mask, 7:10] = torch.tensor(
        [0.3, -0.2, 0.1], device=backend.device
    )
    requested_joints = state[mask].clone()
    requested_roots = backend.root_states[mask].clone()

    # One call must commit joints AND roots without an intermediate refresh
    # clobbering either pending write. Noncontiguous selection exercises masks.
    backend.reset_state(mask)
    torch.testing.assert_close(state[mask], requested_joints, atol=1e-5, rtol=0)
    torch.testing.assert_close(
        backend.root_states[mask], requested_roots, atol=1e-5, rtol=0
    )
    for actual, expected in zip((state, backend.root_states, bodies), before):
        torch.testing.assert_close(actual[~mask], expected, atol=2e-6, rtol=0)
    assert (
        tuple(tensor.data_ptr() for tensor in _get_public_tensors(backend)) == pointers
    )

    backend.step(torques)
    assert (
        tuple(tensor.data_ptr() for tensor in _get_public_tensors(backend)) == pointers
    )
    # Cached references, not refreshed property reads, must see the next step.
    assert not torch.equal(state[mask], requested_joints)
    for original, current in zip(cached, _get_public_tensors(backend)):
        torch.testing.assert_close(original, current, atol=0, rtol=0)
    torch.testing.assert_close(state[1], state[3], atol=2e-6, rtol=0)
    torch.testing.assert_close(cached[3][1], cached[3][3], atol=2e-6, rtol=0)


def test_rotated_root_and_body_velocities_match_published_points(floating_backend):
    backend = floating_backend
    mask = _place_airborne(backend)
    backend.dof_pos[:] = torch.linspace(
        -0.1, 0.1, backend.num_dof, device=backend.device
    )
    backend.root_states[:, 7:10] = torch.tensor([0.2, -0.1, 0.3], device=backend.device)
    backend.root_states[:, 10:13] = torch.tensor(
        [0.4, -0.2, 0.1], device=backend.device
    )
    requested_roots = backend.root_states.clone()
    backend.reset_state(mask)

    root = backend.root_states
    bodies = backend.rigid_body_states.view(4, backend.num_bodies, 13)
    torch.testing.assert_close(root, requested_roots, atol=2e-6, rtol=0)
    torch.testing.assert_close(
        bodies[:, backend.find_body_index("base")], root, atol=2e-6, rtol=0
    )

    # With zero joint velocities, every link instantaneously moves as a rigid
    # assembly: v_link = v_root + omega_world x (p_link - p_root). This tests
    # consistency at the published points. Independent CoM oracles are needed
    # too: the identity also holds for consistently exposed model origins.
    angular_velocity = root[:, None, 10:13].expand_as(bodies[:, :, 10:13])
    expected_linear = root[:, None, 7:10] + torch.cross(
        angular_velocity, bodies[:, :, :3] - root[:, None, :3], dim=-1
    )
    torch.testing.assert_close(bodies[:, :, 7:10], expected_linear, atol=2e-5, rtol=0)
    torch.testing.assert_close(bodies[:, :, 10:13], angular_velocity, atol=2e-5, rtol=0)


def test_free_fall_updates_cached_body_state_at_completed_step(floating_backend):
    backend = floating_backend
    mask = _place_airborne(backend)
    backend.root_states[:, 7:10] = torch.tensor([0.3, -0.2, 0.1], device=backend.device)
    backend.reset_state(mask)
    root = backend.root_states
    bodies = backend.rigid_body_states.view(4, backend.num_bodies, 13)
    contacts = backend.contact_forces
    initial_root = root.clone()
    initial_body_positions = bodies[:, :, :3].clone()
    torques = torch.zeros_like(backend.dof_pos)
    steps = 20
    for _ in range(steps):
        backend.step(torques)

    duration = steps * MINI_CHEETAH_SIM_DT
    gravity = torch.tensor([0.0, 0.0, -9.81], device=backend.device)
    expected_velocity = initial_root[:, 7:10] + gravity * duration
    expected_position = (
        initial_root[:, :3]
        + initial_root[:, 7:10] * duration
        + 0.5 * gravity * duration**2
    )
    # Allow the analytic-to-first-order integration error, not engine-specific
    # trajectory drift: |position error| <= g*T*dt/2 + float32 roundoff.
    position_tolerance = 0.5 * 9.81 * duration * MINI_CHEETAH_SIM_DT + 5e-6
    torch.testing.assert_close(
        root[:, :3], expected_position, atol=position_tolerance, rtol=0
    )
    torch.testing.assert_close(root[:, 7:10], expected_velocity, atol=2e-5, rtol=0)
    torch.testing.assert_close(
        bodies[:, :, :3] - initial_body_positions,
        (root[:, None, :3] - initial_root[:, None, :3]).expand_as(
            initial_body_positions
        ),
        atol=2e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        bodies[:, :, 7:10],
        expected_velocity[:, None, :].expand_as(bodies[:, :, 7:10]),
        atol=2e-5,
        rtol=0,
    )
    torch.testing.assert_close(contacts, torch.zeros_like(contacts), atol=2e-5, rtol=0)
