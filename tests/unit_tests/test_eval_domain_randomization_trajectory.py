from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from gym.envs.mini_cheetah.mini_cheetah_config import MiniCheetahCfg
from scripts.eval_domain_randomization_trajectory import (
    BACKENDS,
    _record_state,
    _state_buffers,
    cell_command,
    system_cog,
    validate_pair,
)
from scripts.mini_cheetah_fidelity import (
    _contact_trace_buffers,
    _record_contact_trace,
    compare,
)


def test_system_cog_weights_each_worlds_applied_link_masses():
    body_states = torch.zeros(2, 2, 13)
    body_states[..., 6] = 1.0
    body_states[:, 0, 0:3] = torch.tensor([0.0, 0.0, 1.0])
    body_states[:, 1, 0:3] = torch.tensor([2.0, 0.0, 1.0])
    body_states[:, 0, 7:10] = torch.tensor([0.0, 2.0, 0.0])
    body_states[:, 0, 10:13] = torch.tensor([0.0, 0.0, 2.0])
    body_mass = torch.tensor([[1.0, 3.0], [3.0, 1.0]])

    position, velocity = system_cog(body_states, body_mass)

    torch.testing.assert_close(
        position, torch.tensor([[1.5, 0.0, 1.0], [0.5, 0.0, 1.0]])
    )
    torch.testing.assert_close(
        velocity, torch.tensor([[0.0, 0.5, 0.0], [0.0, 1.5, 0.0]])
    )


def test_evaluators_recover_physical_com_once(legged_cpu_backend):
    backend = legged_cpu_backend
    num_envs = backend.root_states.shape[0]
    backend.root_states[:, :3] = torch.tensor([0.2, -0.3, 1.0])
    backend.root_states[:, 3:7] = torch.tensor([0.2, -0.3, 0.1, 0.9])
    backend.root_states[:, 3:7] /= backend.root_states[:, 3:7].norm(dim=1)[:, None]
    backend.root_states[:, 7:13] = torch.tensor([0.2, 0.3, -0.1, 1.0, -0.4, 0.7])
    backend.dof_vel[:] = torch.linspace(-0.5, 0.6, backend.num_dof)
    backend.reset_state(torch.ones(num_envs, dtype=torch.bool))

    native_com_positions = []
    native_com_velocities = []
    for data in backend._datas:
        native_com_positions.append(data.xipos[backend._canonical_to_native_body_np])
        velocities = []
        for native_index in backend._canonical_to_native_body_np:
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(
                backend._mjm, data, mujoco.mjtObj.mjOBJ_BODY, native_index, velocity, 0
            )
            velocities.append(velocity[3:].copy())
        native_com_velocities.append(velocities)

    # Consumers use the common state contract, not the backend's Python type.
    source = SimpleNamespace(
        rigid_body_states=backend.rigid_body_states,
        body_names=backend.body_names,
        num_bodies=backend.num_bodies,
        link_mass=backend.link_mass,
    )
    env = SimpleNamespace(
        _backend=source,
        cfg=MiniCheetahCfg(),
        device="cpu",
        robot_layout=backend.robot_layout,
        num_envs=num_envs,
        num_dof=backend.num_dof,
        num_bodies=backend.num_bodies,
        feet_indices=torch.tensor(
            [
                index
                for index, name in enumerate(backend.body_names)
                if name.endswith("foot")
            ]
        ),
        root_states=backend.root_states,
        dof_pos=backend.dof_pos,
        dof_vel=backend.dof_vel,
        actuated_dof_indices=torch.arange(backend.num_dof),
        torques=torch.zeros_like(backend.dof_pos),
        base_lin_vel=torch.zeros(num_envs, 3),
        base_ang_vel=torch.zeros(num_envs, 3),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]).expand(num_envs, -1),
        contact_forces=backend.contact_forces,
        terminated=torch.zeros(num_envs, dtype=torch.bool),
        timed_out=torch.zeros(num_envs, dtype=torch.bool),
    )
    masses = backend.link_mass[0]
    assert np.abs(backend._mjm.body_ipos).max() > 0.01
    expected_position = np.average(native_com_positions, axis=1, weights=masses.numpy())
    expected_velocity = np.average(
        native_com_velocities, axis=1, weights=masses.numpy()
    )

    trajectory = _state_buffers(
        1, len(env.feet_indices), backend.num_dof, backend.num_dof
    )
    _record_state(env, trajectory, 0)
    np.testing.assert_allclose(
        trajectory["cog_position"][0], expected_position[0], atol=2e-7
    )
    np.testing.assert_allclose(
        trajectory["cog_velocity"][0], expected_velocity[0], atol=2e-7
    )

    contact_trace = _contact_trace_buffers(env, 0)
    _record_contact_trace(env, contact_trace, 0)
    np.testing.assert_allclose(
        contact_trace["system_com_pos"][0], expected_position, atol=2e-7
    )
    np.testing.assert_allclose(
        contact_trace["foot_collision_offset"],
        np.zeros((len(env.feet_indices), 3)),
        atol=1e-8,
    )


def test_fidelity_comparison_rejects_mixed_body_reference_points(tmp_path):
    files = []
    for label, reference_point in [("mujoco-cpu", "com"), ("vsim", "model_origin")]:
        path = tmp_path / f"{label}.npz"
        np.savez(
            path,
            label=label,
            probe="impact",
            body_state_reference_point=reference_point,
        )
        files.append(path)

    with pytest.raises(ValueError, match="different reference points"):
        compare(SimpleNamespace(files=files))


@pytest.mark.parametrize("reference_point", ["com", "model_origin"])
def test_trajectory_pair_requires_current_com_convention(tmp_path, reference_point):
    args = SimpleNamespace(dr_friction=0.75, command=(1.0, 0.0, 0.0))
    for mode, friction in [("off", 1.0), ("on", args.dr_friction)]:
        np.savez(
            tmp_path / f"vsim_dr_{mode}.npz",
            body_state_reference_point=reference_point,
            inference_mode=True,
            checkpoint_iteration=1000,
            contact_friction=friction,
            command=args.command,
            initial_root_state=np.zeros(13),
            initial_dof_position=np.zeros(12),
            initial_dof_velocity=np.zeros(12),
        )
    if reference_point == "com":
        validate_pair(tmp_path, BACKENDS[2], args)
    else:
        with pytest.raises(ValueError, match="incompatible body-state reference point"):
            validate_pair(tmp_path, BACKENDS[2], args)


def test_campaign_cells_fix_command_friction_and_inference_protocol(tmp_path):
    checkpoint = tmp_path / "model_1000.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = SimpleNamespace(
        checkpoint=checkpoint,
        output=tmp_path,
        duration=5.0,
        seed=7,
        dr_friction=0.75,
        command=(1.0, 0.0, 0.0),
        mujoco_njmax=256,
    )

    off = cell_command(args, BACKENDS[0], "off")
    on = cell_command(args, BACKENDS[2], "on")

    assert off[off.index("--command") + 1 : off.index("--command") + 4] == [
        "1.0",
        "0.0",
        "0.0",
    ]
    assert on[on.index("--dr-friction") + 1] == "0.75"
    assert on[on.index("--dr-mode") + 1] == "on"
    assert on[on.index("--cell") + 1] == "vsim"
    assert "--resume" not in on
