import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from gym import GYM_ROOT_DIR
from tests.unit_tests.conftest import vsim_guard


def _friction_cfg():
    return SimpleNamespace(
        seed=11,
        asset=SimpleNamespace(
            file=(
                f"{GYM_ROOT_DIR}/resources/robots/friction_sled/urdf/friction_sled.urdf"
            ),
            joint_damping=1.0,
            rotor_inertia=0.0,
            disable_gravity=False,
            fix_base_link=False,
            penalize_contacts_on=[],
            terminate_after_contacts_on=[],
        ),
        init_state=SimpleNamespace(
            pos=[0.0, 0.0, 0.105],
            rot=[0.0, 0.0, 0.0, 1.0],
        ),
        terrain=SimpleNamespace(
            mesh_type="plane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        domain_randomization=SimpleNamespace(contact_friction_range=[0.2, 1.0]),
        sim=SimpleNamespace(gravity=[5.0, 0.0, -9.81]),
        sim_dt=0.002,
    )


def _set_explicit_friction(backend):
    cached = backend.contact_friction
    backend.set_contact_friction(
        torch.tensor([0, 1], device=backend.device),
        torch.tensor([0.2, 0.8], device=backend.device),
    )
    assert backend.contact_friction is cached
    torch.testing.assert_close(backend.contact_friction.cpu(), torch.tensor([0.2, 0.8]))

    backend.set_contact_friction(
        torch.tensor([1], device=backend.device),
        torch.tensor([0.7], device=backend.device),
    )
    torch.testing.assert_close(backend.contact_friction.cpu(), torch.tensor([0.2, 0.7]))


def _assert_friction_changes_motion(backend):
    backend.set_contact_friction(
        torch.tensor([0, 1], device=backend.device),
        torch.tensor([0.2, 0.8], device=backend.device),
    )
    env_ids = torch.arange(2, device=backend.device)
    backend.dof_pos.zero_()
    backend.dof_vel.zero_()
    backend.reset_dof_state(env_ids)
    backend.root_states.zero_()
    backend.root_states[:, 2] = 0.105
    backend.root_states[:, 6] = 1.0
    backend.reset_root_state(env_ids)

    torques = torch.zeros(2, backend.num_dof, device=backend.device)
    for _ in range(500):
        backend.step(torques)

    displacement = backend.root_states[:, 0].cpu()
    assert displacement[0] > 0.5, displacement
    assert displacement[1].abs() < 0.15, displacement
    assert displacement[0] - displacement[1] > 0.5, displacement


def _build_randomized_task(device, backend_name="mujoco"):
    from gym.envs.mini_cheetah.mini_cheetah import MiniCheetah
    from gym.envs.mini_cheetah.mini_cheetah_config import (
        MiniCheetahCfg,
        MiniCheetahRunnerCfg,
    )
    from gym.utils.task_registry import select_backend, task_registry

    cfg = MiniCheetahCfg()
    runner_cfg = MiniCheetahRunnerCfg()
    cfg.env.num_envs = 4
    cfg.seed = 17
    cfg.push_robots.toggle = False
    cfg.domain_randomization.contact_friction_range = [0.5, 1.0]
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    backend = select_backend(cfg, device, backend_name)
    return MiniCheetah(cfg, device, True, backend)


def _assert_task_reset_randomization(device, backend_name="mujoco"):
    env = _build_randomized_task(device, backend_name)
    try:
        generator = torch.Generator(device=device).manual_seed(17)
        expected_initial = 0.5 + 0.5 * torch.rand(4, generator=generator, device=device)
        assert env.domain_randomizer._generator.device == torch.device(device)
        torch.testing.assert_close(
            env.domain_randomizer.contact_friction, expected_initial
        )

        env._reset_idx(torch.tensor([1, 3], device=device))
        expected_reset = 0.5 + 0.5 * torch.rand(2, generator=generator, device=device)
        expected_initial[[1, 3]] = expected_reset
        torch.testing.assert_close(
            env.domain_randomizer.contact_friction, expected_initial
        )
    finally:
        env._backend.close()


def test_mujoco_cpu_applies_friction_per_environment(monkeypatch):
    import mujoco

    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    backend = MuJocoCPUBackend()
    backend.setup(_friction_cfg(), num_envs=2, device="cpu", task=None)
    try:
        assert all(data.model is backend._mjm for data in backend._datas)

        _set_explicit_friction(backend)
        active = {"step": [], "forward": [], "rne": []}
        original_step = mujoco.mj_step
        original_forward = mujoco.mj_forward
        original_rne = mujoco.mj_rnePostConstraint

        def record_step(model, data):
            active["step"].append(float(model.geom_friction[0, 0]))
            original_step(model, data)

        def record_forward(model, data):
            active["forward"].append(float(model.geom_friction[0, 0]))
            original_forward(model, data)

        def record_rne(model, data):
            active["rne"].append(float(model.geom_friction[0, 0]))
            original_rne(model, data)

        monkeypatch.setattr(mujoco, "mj_step", record_step)
        monkeypatch.setattr(mujoco, "mj_forward", record_forward)
        monkeypatch.setattr(mujoco, "mj_rnePostConstraint", record_rne)
        env_ids = torch.arange(2)
        backend.reset_dof_state(env_ids)
        backend.reset_root_state(env_ids)
        backend.step(torch.zeros(2, backend.num_dof))
        assert active["step"] == pytest.approx([0.2, 0.7])
        assert active["rne"] == pytest.approx([0.2, 0.7])
        assert active["forward"] == pytest.approx([0.2, 0.7, 0.2, 0.7])
    finally:
        backend.close()


def test_mujoco_cpu_friction_partial_update_changes_only_selected_value():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    backend = MuJocoCPUBackend()
    backend.setup(_friction_cfg(), num_envs=3, device="cpu", task=None)
    try:
        backend.set_contact_friction(torch.tensor([1]), torch.tensor([0.4]))
        torch.testing.assert_close(
            backend.contact_friction, torch.tensor([1.0, 0.4, 1.0])
        )
    finally:
        backend.close()


def test_mujoco_cpu_without_dr_rejects_friction_updates():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    cfg = _friction_cfg()
    cfg.domain_randomization.contact_friction_range = None
    backend = MuJocoCPUBackend()
    backend.setup(cfg, num_envs=2, device="cpu", task=None)
    try:
        assert all(data.model is backend._mjm for data in backend._datas)
        with pytest.raises(RuntimeError, match="was not enabled"):
            backend.set_contact_friction(torch.tensor([0]), torch.tensor([0.5]))
    finally:
        backend.close()


def test_go2_disables_crashing_mujoco_multiccd_pose():
    code = textwrap.dedent(
        """
        import mujoco
        import numpy as np

        from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
        from gym.envs.go2.go2trot_config import Go2TrotCfg

        qpos = np.array([
            53.8714493, 143.839690, 0.151433455,
            0.651931015, 0.474724545, 0.354051379, 0.473571725,
            -0.736025268, 0.0852538293, -1.37272838,
            0.392659753, -0.143433949, -1.06821755,
            -0.119921852, -0.125259463, -1.09467047,
            0.356757622, 0.819555286, -1.06409946,
        ])

        backend = MuJocoCPUBackend()
        backend.setup(Go2TrotCfg(), num_envs=1, device="cpu", task=None)
        model, data = backend._mjm, backend._datas[0]
        assert model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD
        data.qpos[:] = qpos
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
        backend.close()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mujoco_cpu_friction_has_predicted_physical_effect():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    backend = MuJocoCPUBackend()
    backend.setup(_friction_cfg(), num_envs=2, device="cpu", task=None)
    try:
        _assert_friction_changes_motion(backend)
    finally:
        backend.close()


def test_mujoco_cpu_task_randomizes_only_reset_environments():
    _assert_task_reset_randomization("cpu")


@pytest.mark.warp
def test_mujoco_warp_applies_and_consumes_friction_per_world():
    if not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)
    from gym.envs.base.mujoco_warp_backend import MuJocoWarpBackend

    backend = MuJocoWarpBackend()
    backend.setup(_friction_cfg(), num_envs=2, device="cuda:0", task=None)
    try:
        _set_explicit_friction(backend)
        expected = backend.contact_friction[:, None].expand_as(
            backend._geom_friction_t[:, :, 0]
        )
        torch.testing.assert_close(backend._geom_friction_t[:, :, 0], expected)
        _assert_friction_changes_motion(backend)
    finally:
        backend.close()


@pytest.mark.warp
def test_mujoco_warp_task_randomizes_only_reset_environments():
    if not torch.cuda.is_available():
        pytest.fail("Warp tests requested but CUDA is not available", pytrace=False)
    _assert_task_reset_randomization("cuda:0")


@pytest.mark.vsim
def test_vsim_applies_and_consumes_friction_per_environment_set():
    vsim_guard()
    from gym.envs.base.vsim_backend import VSimBackend

    backend = VSimBackend()
    backend.setup(_friction_cfg(), num_envs=2, device="cuda:0", task=None)
    try:
        assert backend._grp.get_num_environment_sets() == 2
        assert backend._grp.get_num_environments() == [1, 1]
        _set_explicit_friction(backend)

        backend._property_mask.fill_(True)
        backend._static_friction.fill_(float("nan"))
        backend._dynamic_friction.fill_(float("nan"))
        backend._gym.get_rigid_material_properties(backend._friction_set_arr)
        torch.testing.assert_close(
            backend._static_friction.cpu(), backend.contact_friction.cpu()
        )
        torch.testing.assert_close(
            backend._dynamic_friction.cpu(), backend.contact_friction.cpu()
        )
        _assert_friction_changes_motion(backend)
    finally:
        backend.close()


@pytest.mark.vsim
def test_vsim_task_randomizes_only_reset_environments():
    vsim_guard()
    _assert_task_reset_randomization("cuda:0", "vsim")
