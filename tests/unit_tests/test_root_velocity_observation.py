"""Task observations must express native root motion in the robot frame."""

import math

import mujoco
import numpy as np
import pytest
import torch

from gym.envs.go2.go2trot import Go2Trot
from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg
from gym.utils.task_registry import select_backend, task_registry


def _native_local_velocities(backend, device):
    """Compute an oracle from native state, independent of public root tensors."""
    if device == "cpu":
        qpos = np.stack([data.qpos.copy() for data in backend._datas])
        qvel = np.stack([data.qvel.copy() for data in backend._datas])
    else:
        qpos = backend._qpos_t.cpu().numpy()
        qvel = backend._qvel_t.cpu().numpy()

    model = backend._mjm
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    assert base_id >= 0
    velocities = np.empty((len(qpos), 6))
    for index, (position, velocity) in enumerate(zip(qpos, qvel, strict=True)):
        data = mujoco.MjData(model)
        data.qpos[:] = position
        data.qvel[:] = velocity
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        mujoco.mj_objectVelocity(
            model,
            data,
            # BODY selects the inertial/COM frame, whose principal axes can
            # differ from the link axes. XBODY selects the body-origin frame
            # used by the task's quaternion and body-local observations.
            mujoco.mjtObj.mjOBJ_XBODY,
            base_id,
            velocities[index],
            1,  # Local body frame; angular components precede linear components.
        )
    return torch.tensor(velocities, dtype=torch.float, device=device)


@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.warp, id="warp"))
)
def test_go2trot_root_velocity_observations_match_native_body_frame(device):
    if device != "cpu" and not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)

    cfg = Go2TrotCfg()
    runner_cfg = Go2TrotRunnerCfg()
    cfg.seed = 7
    cfg.env.num_envs = 2
    cfg.control.ctrl_frequency = 100
    cfg.control.desired_sim_frequency = 100
    cfg.push_robots.toggle = False
    cfg.init_state.reset_mode = "reset_to_basic"
    cfg.init_state.pos = [0.0, 0.0, 3.0]
    cfg.domain_randomization.startup.contact_friction_range = None
    cfg.domain_randomization.startup.link_mass_scale_range = None
    cfg.domain_randomization.episode.scale_ranges = {}
    # No runner/policy is constructed, so observation noise and normalization
    # cannot obscure the task's physical-to-scaled observation boundary.
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = select_backend(cfg, device, "mujoco")
    try:
        env = Go2Trot(cfg, device, True, backend)
        posture = {
            f"{leg}_{joint}_joint": angle
            for leg in ("FL", "FR", "RL", "RR")
            for joint, angle in (("hip", 0.0), ("thigh", 0.66), ("calf", -1.36))
        }
        env.dof_pos[:] = torch.tensor(
            [posture[name] for name in env.dof_names], device=device
        )
        env.dof_vel.zero_()
        env.root_states.zero_()
        env.root_states[:, 2] = 3.0
        # A quarter turn plus a compound rotation prevent local/world angular
        # velocity confusion from hiding behind upright or yaw-only motion.
        env.root_states[:, 3:7] = torch.tensor(
            [
                [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
                [0.2, -0.3, 0.4, math.sqrt(0.71)],
            ],
            device=device,
        )
        env.root_states[:, 7:10] = torch.tensor(
            [[0.3, -0.2, 0.1], [-0.5, 0.4, -0.2]], device=device
        )
        env.root_states[:, 10:13] = torch.tensor(
            [[1.1, -0.6, 0.4], [-0.5, 1.4, -0.8]], device=device
        )
        backend.reset_state(torch.ones(2, dtype=torch.bool, device=device))
        cached_root = env.root_states
        cached_ang_vel = env.base_ang_vel
        cached_lin_vel = env.base_lin_vel
        initial_orientation = cached_root[:, 3:7].clone()

        for _ in range(3):
            env.step()
            # Read the cached task quantities before the independent native
            # oracle; a getter must not repair stale state during the assertion.
            actual_ang_vel = cached_ang_vel.clone()
            actual_lin_vel = cached_lin_vel.clone()
            scaled_ang_vel = env.get_state("base_ang_vel")
            scaled_lin_vel = env.get_state("base_lin_vel")
            expected = _native_local_velocities(backend, device)
            torch.testing.assert_close(
                actual_ang_vel, expected[:, :3], atol=3e-6, rtol=2e-5
            )
            torch.testing.assert_close(
                actual_lin_vel, expected[:, 3:], atol=3e-6, rtol=2e-5
            )
            torch.testing.assert_close(
                scaled_ang_vel,
                expected[:, :3] / cfg.scaling.base_ang_vel,
                atol=1e-5,
                rtol=2e-5,
            )
            torch.testing.assert_close(
                scaled_lin_vel,
                expected[:, 3:] / cfg.scaling.base_lin_vel,
                atol=1e-5,
                rtol=2e-5,
            )
            assert env.base_ang_vel is cached_ang_vel
            assert env.base_lin_vel is cached_lin_vel
            assert env.root_states is cached_root

        assert not torch.allclose(cached_root[:, 3:7], initial_orientation)
        assert torch.count_nonzero(env.contact_forces) == 0
        assert cached_root.data_ptr() == backend.root_states.data_ptr()
    finally:
        backend.close()
