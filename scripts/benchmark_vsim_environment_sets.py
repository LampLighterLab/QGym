"""Measure VSim environment-set topology at fixed 100 Hz and nominal physics.

Run each set count in a fresh process with the VSim process environment:

    uv run --frozen --env-file .env.vsim -m \
        scripts.benchmark_vsim_environment_sets --sets 4096 \
        --output logs/vsim_sets/4096.json

The proxy changes native sharing topology without enabling physical DR. This is
a performance discriminator, not a pooled-domain training implementation. The
isolated component measurements are not additive, and only step profiles advance
physics. No license or CUDA initialization is needed to import this module.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda:0"
FREQUENCY_HZ = 100
WARMUP_STEPS = 25
SETTLE_STEPS = 200
SOURCE_PATHS = (
    "scripts/benchmark_vsim_environment_sets.py",
    "gym/envs/base/vsim_backend.py",
    "gym/envs/base/vsim_asset.py",
    "gym/envs/base/sim_backend.py",
    "gym/envs/base/legged_robot.py",
    "gym/envs/base/domain_randomization.py",
    "gym/envs/go2/go2trot.py",
    "gym/envs/go2/go2trot_config.py",
    "resources/robots/go2/urdf/go2.urdf",
)


def environment_set_sizes(num_envs: int, num_sets: int) -> list[int]:
    if num_envs <= 0 or num_sets <= 0 or num_envs % num_sets:
        raise ValueError("sets must be a positive divisor of positive num-envs")
    return [num_envs // num_sets] * num_sets


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--sets", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        environment_set_sizes(args.num_envs, args.sets)
    except ValueError as error:
        parser.error(str(error))
    if args.steps <= 0 or args.repeats <= 0:
        parser.error("steps and repeats must be positive")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    return args


def configure_task(num_envs: int, seed: int, graph_captures: bool):
    from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg
    from gym.utils.task_registry import task_registry

    cfg, runner_cfg = Go2TrotCfg(), Go2TrotRunnerCfg()
    cfg.seed = runner_cfg.seed = seed
    cfg.env.num_envs = num_envs
    cfg.control.ctrl_frequency = FREQUENCY_HZ
    cfg.control.desired_sim_frequency = FREQUENCY_HZ
    cfg.domain_randomization.startup.contact_friction_range = None
    cfg.domain_randomization.startup.link_mass_scale_range = None
    cfg.domain_randomization.episode.scale_ranges = {}
    cfg.terrain.static_friction = cfg.terrain.dynamic_friction = 1.0
    cfg.push_robots.toggle = False
    cfg.commands.resampling_time = 10000.0
    cfg.init_state.reset_mode = "reset_to_basic"
    cfg.vsim_attributes = SimpleNamespace(enable_graph_captures=graph_captures)
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    return cfg


class NominalSetGym:
    """Forward gym operations, replacing only the one nominal group topology."""

    def __init__(self, native, sizes: list[int]):
        self.native = native
        self.sizes = list(sizes)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def create_environment_group(self, env_def, sizes):
        # The production backend must request its ordinary nominal topology;
        # accepting a physical-DR group would confound this experiment.
        if list(sizes) != [sum(self.sizes)]:
            raise ValueError("topology probe requires one nominal environment set")
        return self.native.create_environment_group(env_def, self.sizes)


def source_provenance():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS
        },
    }


def write_result(path: Path, result: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def synchronize():
    # Device-wide synchronization includes VSim's private CUDA streams.
    torch.cuda.synchronize(DEVICE)


def check_finite(backend):
    for name, value in (
        ("root_states", backend.root_states),
        ("dof_state", backend.dof_state),
        ("rigid_body_states", backend.rigid_body_states),
        ("contact_forces", backend.contact_forces),
    ):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"non-finite {name} during topology benchmark")


def measure_component(operation, restore, backend, steps: int, repeats: int):
    durations = []
    for _ in range(repeats):
        restore()
        for _ in range(WARMUP_STEPS):
            operation()
        synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            operation()
        synchronize()
        durations.append(time.perf_counter() - start)
        # Native-step measurements deliberately omit public-state refresh.
        backend._refresh_state()
        check_finite(backend)
    median = statistics.median(durations)
    return {
        "durations_s": durations,
        "median_us_per_call": median * 1e6 / steps,
        "median_env_calls_per_s": backend._num_envs * steps / median,
    }


def run_profiles(env, backend, args, result):
    native = backend._gym.native
    all_mask = torch.ones(args.num_envs, dtype=torch.bool, device=DEVICE)
    empty_mask = torch.zeros_like(all_mask)
    zero = torch.zeros(args.num_envs, backend.num_dof, device=DEVICE)
    nominal_pose = {
        f"{leg}_{joint}_joint": value
        for leg in ("FL", "FR", "RL", "RR")
        for joint, value in (("hip", 0.0), ("thigh", 0.66), ("calf", -1.36))
    }
    positions = torch.tensor(
        [nominal_pose[name] for name in backend.dof_names], device=DEVICE
    )

    def restore():
        backend.dof_pos[:] = positions
        backend.dof_vel.zero_()
        backend.root_states.zero_()
        backend.root_states[:, 2] = 0.46
        backend.root_states[:, 6] = 1.0
        backend.reset_state(all_mask)
        backend._motor_buf.zero_()
        native.set_motor_forces(backend._motor_arr)
        for _ in range(SETTLE_STEPS):
            backend.step(zero)

    def gather():
        native.get_articulation_kinematic_states(backend._aks_get_arr)
        native.get_link_transforms(backend._lt_arr)
        native.get_link_velocities(backend._lv_arr)
        native.get_sensor_forces(backend._fs_arr)

    operations = {
        "backend_step": lambda: backend.step(zero),
        "native_step": native.step,
        "refresh": backend._refresh_state,
        "native_kinematics": native.compute_kinematics,
        "native_gathers": gather,
        "assembly": backend._sync_assembled_states,
        "empty_backend_reset": lambda: backend.reset_state(empty_mask),
        "empty_task_reset": lambda: env._reset_idx(empty_mask),
    }
    result["profiles"] = {}
    for name, operation in operations.items():
        measurement = measure_component(
            operation, restore, backend, args.steps, args.repeats
        )
        result["profiles"][name] = measurement
        print(f"{name}: {measurement['median_us_per_call']:.2f} us/call", flush=True)
        write_result(args.output, result)

    restore()
    check_finite(backend)
    result["settled_state"] = {
        "root_mean": backend.root_states.mean(dim=0).tolist(),
        "total_contact_z_mean_n": (
            backend.contact_forces[..., 2].sum(dim=1).mean().item()
        ),
        "robot_weight_mean_n": 9.81 * backend.link_mass.sum(dim=1).mean().item(),
        "finite": True,
    }
    if args.profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            for _ in range(10):
                with torch.profiler.record_function("q2_backend_step"):
                    backend.step(zero)
            synchronize()
        profiler.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
        args.output.with_suffix(".profile.txt").write_text(
            profiler.key_averages().table(
                sort_by="self_device_time_total", row_limit=50
            ),
            encoding="utf-8",
        )


def main(argv=None):
    args = get_args(argv)
    # VSim is optional and licensed; keep it behind the executable boundary.
    import vlearn

    from gym.envs.base.vsim_backend import VSimBackend
    from gym.envs.go2.go2trot import Go2Trot
    from gym.utils.helpers import class_to_dict, set_seed

    cfg = configure_task(args.num_envs, args.seed, not args.no_graphs)
    sizes = environment_set_sizes(args.num_envs, args.sets)
    set_seed(args.seed)
    create_gym = vlearn.create_gym

    def create_override(**kwargs):
        return NominalSetGym(create_gym(**kwargs), sizes)

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "physics_hz": FREQUENCY_HZ,
        "control_hz": FREQUENCY_HZ,
        "num_envs": args.num_envs,
        "sets": args.sets,
        "set_sizes": sizes,
        "physical_dr": False,
        "graph_captures": not args.no_graphs,
        "steps": args.steps,
        "repeats": args.repeats,
        "warmup_steps": WARMUP_STEPS,
        "settle_steps_per_trial": SETTLE_STEPS,
        "vlearn": importlib.metadata.version("vlearn"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(DEVICE),
        "seed": args.seed,
        "provenance": source_provenance(),
        "config": class_to_dict(cfg),
        "timing": (
            "device-wide synchronized wall time; isolated component timings "
            "are not additive; only backend_step and native_step advance physics"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_result(args.output, result)
    backend = VSimBackend()
    try:
        synchronize()
        free_before = torch.cuda.mem_get_info(DEVICE)[0]
        start = time.perf_counter()
        with patch.object(vlearn, "create_gym", create_override):
            env = Go2Trot(cfg, DEVICE, True, backend)
        synchronize()
        result["setup_s"] = time.perf_counter() - start
        result["setup_cuda_bytes"] = free_before - torch.cuda.mem_get_info(DEVICE)[0]
        native_sizes = backend._grp.get_num_environments()
        if list(native_sizes) != sizes:
            raise RuntimeError(f"unexpected native set sizes: {native_sizes}")
        result["native_sets"] = backend._grp.get_num_environment_sets()
        result["native_set_sizes"] = list(native_sizes)
        result["solver_iterations"] = backend._gym.get_num_solver_iterations()
        torch.testing.assert_close(
            backend.contact_friction, torch.ones(args.num_envs, device=DEVICE)
        )
        run_profiles(env, backend, args, result)
        result["status"] = "complete"
        write_result(args.output, result)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
