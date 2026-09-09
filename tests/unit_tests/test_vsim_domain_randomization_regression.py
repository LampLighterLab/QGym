"""VSim DR topology must preserve nominal dynamics and task-facing units.

These licensed tests use current Go2Trot code at 100 Hz, without historical
checkpoints or throughput thresholds. Worlds are created and closed sequentially
because VSim owns a process singleton. Run with scripts/run_vsim_tests.sh.
"""

from contextlib import contextmanager

import pytest
import torch

from gym.envs.go2.go2trot import Go2Trot
from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg
from gym.utils.helpers import set_seed
from gym.utils.task_registry import select_backend, task_registry
from gym.utils.torch_quat import quat_rotate_inverse
from tests.unit_tests.conftest import vsim_guard


pytestmark = pytest.mark.vsim
NUM_ENVS = 4
SAMPLE_STEPS = (1, 5, 10, 20, 40)


@contextmanager
def _go2trot(mode, *, randomized=False):
    vsim_guard()
    cfg = Go2TrotCfg()
    runner_cfg = Go2TrotRunnerCfg()
    cfg.seed = 7
    cfg.env.num_envs = NUM_ENVS
    cfg.control.ctrl_frequency = 100
    cfg.control.desired_sim_frequency = 100
    cfg.push_robots.toggle = False
    cfg.commands.resampling_time = 100.0
    cfg.init_state.reset_mode = "reset_to_basic"
    dr = cfg.domain_randomization
    nominal_friction = cfg.terrain.dynamic_friction
    dr.startup.contact_friction_range = (
        ([0.5, 1.0] if randomized else [nominal_friction, nominal_friction])
        if mode in ("friction", "all")
        else None
    )
    dr.startup.link_mass_scale_range = (
        ([0.8, 1.2] if randomized else [1.0, 1.0]) if mode in ("mass", "all") else None
    )
    dr.episode.scale_ranges = (
        {
            "p_gains": [0.8, 1.2] if randomized else [1.0, 1.0],
            "d_gains": [0.8, 1.2] if randomized else [1.0, 1.0],
        }
        if mode in ("pd", "all")
        else {}
    )
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    set_seed(cfg.seed)
    backend = select_backend(cfg, "cuda:0", "vsim")
    try:
        env = Go2Trot(cfg, "cuda:0", True, backend)
        _restore_initial_state(env)
        yield env, runner_cfg
    finally:
        backend.close()


def _restore_initial_state(env):
    # Distinct worlds expose accidental broadcast/routing; legal joint angles
    # avoid making this a test of the task's historical all-zero reset posture.
    posture = {
        f"{leg}_{joint}_joint": angle
        for leg in ("FL", "FR", "RL", "RR")
        for joint, angle in (("hip", 0.0), ("thigh", 0.66), ("calf", -1.36))
    }
    env.dof_pos[:] = torch.tensor(
        [posture[name] for name in env.dof_names], device=env.device
    )
    env.dof_vel.zero_()
    env.root_states.zero_()
    env.root_states[:, 2] = torch.tensor([0.43, 0.45, 0.47, 0.49], device=env.device)
    env.root_states[:, 6] = 1.0
    env.root_states[:, 7] = torch.tensor([0.0, 0.1, -0.1, 0.2], device=env.device)
    env._backend.reset_state(torch.ones(NUM_ENVS, dtype=torch.bool, device=env.device))
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
    env.dof_pos_obs[:] = env.dof_pos - env.default_dof_pos
    env.dof_pos_target.zero_()
    env.dof_pos_history.zero_()
    env.commands.zero_()
    env.phase[:, 0] = torch.tensor([0.0, 0.2, 0.4, 0.6], device=env.device)
    env.phase_frequency.fill_(2.0)
    env._update_phase_observation()
    env._update_gait_reference()
    env.episode_length_buf.zero_()
    env._reset_buffers()


def _step(env, step):
    offsets = torch.arange(env.num_actuators, device=env.device)
    env.dof_pos_target[:] = 0.025 * torch.sin(0.2 * step + offsets)
    env.step()


def _snapshot(env, runner_cfg):
    backend = env._backend
    values = {
        "root": backend.root_states,
        "dof": backend.dof_state.view(NUM_ENVS, env.num_dof, 2),
        "body": backend.rigid_body_states.view(NUM_ENVS, backend.num_bodies, 13),
        "contact": backend.contact_forces,
        "actor_obs": env.get_states(runner_cfg.actor.obs),
        "contact_strength": env._foot_contact_strength(),
        "mass": backend.link_mass,
        "inertia": backend.link_inertia,
        "friction": backend.contact_friction,
        "p_gains": env.p_gains,
        "d_gains": env.d_gains,
        "phase": env.phase,
        "phase_frequency": env.phase_frequency,
        "history": env.dof_pos_history,
        "commands": env.commands,
        "episode_length": env.episode_length_buf,
    }
    return {name: value.detach().cpu().clone() for name, value in values.items()}


def _assert_same_state(actual, expected, *, rows=slice(None), following_step=False):
    for name, reference in expected.items():
        if following_step and name in ("dof", "root", "body"):
            candidate, reference = actual[name][rows], reference[rows]
            if name == "dof":
                pose, velocity = candidate[..., 0], candidate[..., 1]
                ref_pose, ref_velocity = reference[..., 0], reference[..., 1]
            else:
                pose, velocity = candidate[..., :7], candidate[..., 7:]
                ref_pose, ref_velocity = reference[..., :7], reference[..., 7:]
            torch.testing.assert_close(
                pose,
                ref_pose,
                rtol=2e-5,
                atol=2e-5,
                msg=lambda message: f"{name} pose: {message}",
            )
            # Extra reset refreshes round kinematics before the next contact
            # solve. Bound velocity separately in m/s or rad/s: 2e-4 would
            # integrate to only 2e-6 m/rad over this test's 10 ms step. Keep
            # immediate-reset and pose checks tighter to catch state leakage.
            torch.testing.assert_close(
                velocity,
                ref_velocity,
                rtol=2e-5,
                atol=2e-4,
                msg=lambda message: f"{name} velocity: {message}",
            )
            continue
        # Refreshing VSim kinematics can round floating-point state even when
        # no world was reset. Contacts have force units, not position units.
        atol = 5e-3 if name == "contact" else 1e-4 if name == "actor_obs" else 2e-5
        torch.testing.assert_close(
            actual[name][rows],
            reference[rows],
            rtol=2e-5,
            atol=atol,
            msg=lambda message: f"{name}: {message}",
        )


def _trajectory(mode):
    with _go2trot(mode) as (env, runner_cfg):
        samples = []
        for step in range(1, max(SAMPLE_STEPS) + 1):
            _step(env, step)
            if step in SAMPLE_STEPS:
                samples.append(_snapshot(env, runner_cfg))
        assert any(sample["contact"].abs().max() > 1.0 for sample in samples), (
            "topology comparison must exercise collision contacts"
        )
        return samples


@pytest.fixture(scope="module")
def nominal_trajectory():
    return _trajectory("off")


@pytest.mark.parametrize(
    ("mode", "expected_sets"),
    (
        ("off", 1),
        ("pd", 1),
        ("friction", NUM_ENVS),
        ("mass", NUM_ENVS),
        ("all", NUM_ENVS),
    ),
)
def test_native_set_topology_matches_enabled_physical_axes(mode, expected_sets):
    with _go2trot(mode) as (env, _):
        assert env.dt == 0.01
        assert env.cfg.sim_dt == 0.01
        assert env._backend._grp.get_num_environment_sets() == expected_sets
        expected_sizes = [NUM_ENVS] if expected_sets == 1 else [1] * NUM_ENVS
        assert env._backend._grp.get_num_environments() == expected_sizes


@pytest.mark.parametrize("mode", ("friction", "mass", "all"))
def test_nominal_dr_preserves_dynamics_and_scaled_observations(
    mode, nominal_trajectory
):
    for actual, expected in zip(_trajectory(mode), nominal_trajectory, strict=True):
        _assert_same_state(actual, expected)


@pytest.mark.parametrize("mode", ("off", "all"))
def test_sparse_and_empty_resets_preserve_unselected_worlds(mode):
    # A no-reset replay also checks native state after the next physics step,
    # rather than accepting unchanged public caches as sufficient evidence.
    with _go2trot(mode, randomized=True) as (env, runner_cfg):
        for step in range(1, 22):
            _step(env, step)
        next_without_reset = _snapshot(env, runner_cfg)

    with _go2trot(mode, randomized=True) as (env, runner_cfg):
        for step in range(1, 21):
            _step(env, step)
        before = _snapshot(env, runner_cfg)
        backend = env._backend
        pointers = tuple(
            value.data_ptr()
            for value in (
                backend.root_states,
                backend.dof_state,
                backend.contact_forces,
            )
        )
        empty = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.device)
        env._reset_idx(empty)
        _assert_same_state(_snapshot(env, runner_cfg), before)

        reset_mask = torch.tensor([False, True, False, False], device=env.device)
        env._reset_idx(reset_mask)
        after_reset = _snapshot(env, runner_cfg)
        untouched = torch.tensor([0, 2, 3])
        _assert_same_state(after_reset, before, rows=untouched)
        assert env.episode_length_buf[1] == 0
        torch.testing.assert_close(
            env.root_states[1, 2],
            torch.tensor(env.cfg.init_state.pos[2], device=env.device),
            rtol=0,
            atol=2e-5,
        )
        for name in ("mass", "inertia", "friction"):
            torch.testing.assert_close(after_reset[name], before[name], rtol=0, atol=0)
        assert pointers == tuple(
            value.data_ptr()
            for value in (
                backend.root_states,
                backend.dof_state,
                backend.contact_forces,
            )
        )
        _step(env, 21)
        _assert_same_state(
            _snapshot(env, runner_cfg),
            next_without_reset,
            rows=untouched,
            following_step=True,
        )


def test_contact_strength_normalizes_by_each_worlds_applied_weight():
    with _go2trot("all", randomized=True) as (env, _):
        weight = 9.81 * env._backend.link_mass.sum(dim=1, keepdim=True)
        assert weight.max() - weight.min() > 1e-3
        # Equal fractions of each world's own weight must give equal reward
        # inputs, including saturation above body weight and at negative load.
        fractions = torch.tensor([-0.25, 0.25, 0.5, 2.0], device=env.device)
        env.contact_forces.zero_()
        env.contact_forces[:, env.feet_indices, 2] = weight * fractions
        expected = torch.tensor([0.0, 0.15625, 0.5, 1.0], device=env.device)
        torch.testing.assert_close(
            env._foot_contact_strength(), expected.expand(NUM_ENVS, -1)
        )
