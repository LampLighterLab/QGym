"""Measure domain-randomization setup and reset-path overhead for one backend.

Run one cell per process. The campaign orchestrators construct backend/DR
matrices and call this worker with isolated process state.
"""

import argparse
import copy
from contextlib import contextmanager
import importlib.metadata
import json
import math
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gym.envs.base.domain_randomization import (
    STARTUP_AXES,
    apply_domain_randomization_override,
    get_domain_randomization_range,
    set_domain_randomization_range,
)
from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry


SCHEMA_VERSION = 2
FRICTION_RANGE = [0.5, 1.0]
BUNDLES = ("off", "friction-fixed", "friction", "pd", "mass", "all")
BUNDLE_MODES = {
    "off": "off",
    "friction-fixed": "friction-only",
    "friction": "friction-only",
    "pd": "pd-only",
    "mass": "mass-only",
    "all": "config",
    # Compatibility with the original contact-friction campaign.
    "on": "friction-only",
}
DEFAULT_MIN_STEPS = 50
DEFAULT_MIN_WARMUP_STEPS = 25


@contextmanager
def trace_range(name, device):
    """Add a stable range label when the worker is captured with Nsight."""
    if device.startswith("cuda"):
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if device.startswith("cuda"):
            torch.cuda.nvtx.range_pop()


def timeout_reset_counts(num_envs, episode_steps, num_steps):
    """Return an exactly rate-matched deterministic timeout reset schedule."""
    cumulative = np.floor(
        np.arange(num_steps + 1, dtype=np.float64) * num_envs / episode_steps
    ).astype(np.int64)
    return np.diff(cumulative)


def default_warmup_steps(num_envs):
    """Choose a bounded warmup long enough to expose lazy backend allocation."""
    return max(DEFAULT_MIN_WARMUP_STEPS, min(50, math.ceil(8192 / num_envs)))


def make_reset_schedule(profile, num_envs, episode_steps, num_steps, device):
    """Prebuild persistent-shape reset masks outside the timed region."""
    empty = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if profile == "none":
        return [empty] * num_steps
    if profile == "all":
        all_envs = torch.ones(num_envs, dtype=torch.bool, device=device)
        return [all_envs] * num_steps
    if profile != "timeout":
        raise ValueError(f"unknown reset profile {profile!r}")

    counts = timeout_reset_counts(num_envs, episode_steps, num_steps)
    schedule = []
    cursor = 0
    for count in counts:
        if count == 0:
            schedule.append(empty)
            continue
        ids = torch.arange(cursor, cursor + int(count), device=device) % num_envs
        reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        reset_mask[ids] = True
        schedule.append(reset_mask)
        cursor = (cursor + int(count)) % num_envs
    return schedule


def synchronize(backend_label, device):
    """Wait for both simulator and Torch work before reading wall time."""
    if backend_label == "warp":
        import warp as wp

        wp.init()
        wp.synchronize_device(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def package_versions():
    versions = {}
    for distribution in ("torch", "mujoco", "mujoco-warp", "vlearn"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def git_metadata():
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def configure_task(task, num_envs, seed, dr_bundle):
    import gym.envs  # noqa: F401 — registers tasks

    registered_env_cfg, registered_train_cfg = task_registry.get_cfgs(task)
    env_cfg = copy.deepcopy(registered_env_cfg)
    train_cfg = copy.deepcopy(registered_train_cfg)
    env_cfg.env.num_envs = num_envs
    env_cfg.seed = seed
    train_cfg.seed = seed
    env_cfg.init_state.reset_mode = "reset_to_basic"
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False
    apply_domain_randomization_override(env_cfg, BUNDLE_MODES[dr_bundle])
    if dr_bundle == "friction-fixed":
        nominal = float(env_cfg.terrain.dynamic_friction)
        set_domain_randomization_range(
            env_cfg,
            "contact_friction_range",
            [nominal, nominal],
        )
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    return env_cfg, train_cfg


def cuda_memory_snapshot(device):
    if not device.startswith("cuda"):
        return None
    free, total = torch.cuda.mem_get_info(torch.device(device))
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "torch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def profile_trials(env, backend_label, profile, steps, repeats, warmup_steps):
    device = env.device
    episode_steps = int(env.max_episode_length)
    warmup_schedule = make_reset_schedule(
        profile, env.num_envs, episode_steps, warmup_steps, device
    )
    with trace_range(f"warmup/{profile}", device):
        for reset_mask in warmup_schedule:
            env.step()
            env._reset_idx(reset_mask)
        synchronize(backend_label, device)

    schedule = make_reset_schedule(profile, env.num_envs, episode_steps, steps, device)
    reset_count = sum(int(mask.count_nonzero().item()) for mask in schedule)
    elapsed = []
    checksums = []
    all_envs = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    for repeat in range(repeats):
        with trace_range(f"trial/{profile}/{repeat}", device):
            env._reset_idx(all_envs)
            synchronize(backend_label, device)
            start = time.perf_counter()
            for reset_mask in schedule:
                env.step()
                env._reset_idx(reset_mask)
            synchronize(backend_label, device)
            elapsed.append(time.perf_counter() - start)

        finite = (
            torch.isfinite(env.root_states).all()
            & torch.isfinite(env.dof_state).all()
            & torch.isfinite(env.contact_forces).all()
        )
        if not bool(finite.item()):
            raise RuntimeError(f"non-finite state after {profile!r} profile")
        checksums.append(
            float(
                (
                    env.root_states[:, 2].mean()
                    + env.dof_pos.mean()
                    + env.contact_forces[..., 2].mean()
                ).item()
            )
        )

    elapsed_array = np.asarray(elapsed, dtype=np.float64)
    throughput = env.num_envs * steps / elapsed_array
    physics_throughput = throughput * int(env.cfg.control.decimation)
    reset_throughput = reset_count / elapsed_array
    return {
        "profile": profile,
        "steps_per_trial": steps,
        "repeats": repeats,
        "warmup_steps": warmup_steps,
        "reset_envs_per_trial": int(reset_count),
        "elapsed_seconds": elapsed,
        "env_control_steps_per_s": throughput.tolist(),
        "env_physics_steps_per_s": physics_throughput.tolist(),
        "reset_envs_per_s": reset_throughput.tolist(),
        "median_env_control_steps_per_s": float(np.median(throughput)),
        "p10_env_control_steps_per_s": float(np.quantile(throughput, 0.1)),
        "p90_env_control_steps_per_s": float(np.quantile(throughput, 0.9)),
        "std_env_control_steps_per_s": float(np.std(throughput)),
        "state_checksums": checksums,
    }


def backend_selection(label):
    return {
        "cpu": ("mujoco", "cpu"),
        "warp": ("mujoco", "cuda:0"),
        "vsim": ("vsim", "cuda:0"),
    }[label]


def tensor_stats(value):
    if value is None:
        return None
    value = value.detach()
    return {
        "min": float(value.min()),
        "mean": float(value.mean()),
        "max": float(value.max()),
        "std": float(value.std(unbiased=False)),
    }


def native_topology_details(env, backend_label):
    """Return measured topology facts rather than inferring them from config."""
    if backend_label != "vsim":
        return None
    sizes = env._backend._grp.get_num_environments()
    return {
        "environment_set_count": env._backend._grp.get_num_environment_sets(),
        "total_environment_count": sum(sizes),
        "min_environments_per_set": min(sizes),
        "max_environments_per_set": max(sizes),
    }


def validate_applied_randomization(env, bundle):
    randomizer = env.domain_randomizer
    values = {
        "contact_friction": randomizer.contact_friction,
        "stiffness_scale": randomizer.episode_scale("p_gains"),
        "damping_scale": randomizer.episode_scale("d_gains"),
        "link_mass_scale": randomizer.link_mass_scale,
    }
    enabled = {
        "off": set(),
        "friction-fixed": {"contact_friction"},
        "friction": {"contact_friction"},
        "pd": {"stiffness_scale", "damping_scale"},
        "mass": {"link_mass_scale"},
        "all": {
            "contact_friction",
            "stiffness_scale",
            "damping_scale",
            "link_mass_scale",
        },
    }[bundle]
    for name, value in values.items():
        if name in enabled:
            if value is None:
                raise RuntimeError(f"{bundle} did not allocate {name}")
            if bundle == "friction-fixed" and name == "contact_friction":
                nominal = float(env.cfg.terrain.dynamic_friction)
                torch.testing.assert_close(value, torch.full_like(value, nominal))
            elif value.numel() > 1 and float(value.max() - value.min()) == 0.0:
                raise RuntimeError(f"{bundle} produced no spread in {name}")
        elif name == "contact_friction":
            nominal = float(env.cfg.terrain.dynamic_friction)
            torch.testing.assert_close(value, torch.full_like(value, nominal))
        elif value is not None:
            raise RuntimeError(f"{bundle} unexpectedly allocated {name}")
    return {name: tensor_stats(value) for name, value in values.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="go2trot")
    parser.add_argument("--backend", choices=["cpu", "warp", "vsim"], required=True)
    parser.add_argument(
        "--dr",
        choices=[*BUNDLES, "on"],
        required=True,
        help="DR bundle; 'on' is retained as a friction-only compatibility alias",
    )
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--target_env_steps",
        type=int,
        default=32768,
        help="approximately constant timed work across environment counts",
    )
    parser.add_argument("--min_steps", type=int, default=DEFAULT_MIN_STEPS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("none", "timeout", "all"),
        default=("none", "timeout", "all"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.num_envs <= 0 or args.target_env_steps <= 0 or args.min_steps <= 0:
        parser.error("environment and step counts must be positive")
    if args.repeats <= 0:
        parser.error("repeats must be positive")

    physics_backend, device = backend_selection(args.backend)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(args.backend, device)
    memory_before = cuda_memory_snapshot(device)
    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    dr_bundle = "friction" if args.dr == "on" else args.dr
    env_cfg, _ = configure_task(
        args.task,
        args.num_envs,
        args.seed,
        dr_bundle=dr_bundle,
    )
    set_seed(args.seed)
    with trace_range(f"setup/{dr_bundle}", device):
        setup_start = time.perf_counter()
        env = task_registry.make_env(
            args.task,
            env_cfg,
            device=device,
            headless=True,
            backend=physics_backend,
        )
        synchronize(args.backend, device)
        setup_seconds = time.perf_counter() - setup_start
    memory_after_setup = cuda_memory_snapshot(device)
    rss_after_setup_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    steps = max(args.min_steps, math.ceil(args.target_env_steps / args.num_envs))
    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = default_warmup_steps(args.num_envs)

    try:
        profiles = [
            profile_trials(
                env,
                args.backend,
                profile,
                steps,
                args.repeats,
                warmup_steps,
            )
            for profile in args.profiles
        ]
        synchronize(args.backend, device)
        memory_after_profiles = cuda_memory_snapshot(device)
        rss_after_profiles_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        applied = validate_applied_randomization(env, dr_bundle)

        result = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git": git_metadata(),
            "system": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "packages": package_versions(),
                "cuda_device": (
                    torch.cuda.get_device_name(torch.device(device))
                    if device.startswith("cuda")
                    else None
                ),
            },
            "protocol": {
                "task": args.task,
                "backend_label": args.backend,
                "physics_backend": physics_backend,
                "device": device,
                "num_envs": args.num_envs,
                "seed": args.seed,
                "target_env_steps": args.target_env_steps,
                "min_steps": args.min_steps,
                "steps_per_trial": steps,
                "repeats": args.repeats,
                "warmup_steps": warmup_steps,
                "reset_profiles": list(args.profiles),
                "dr_enabled": dr_bundle != "off",
                "dr_bundle": dr_bundle,
                "domain_randomization": {
                    "startup": {
                        name: get_domain_randomization_range(env_cfg, name)
                        for name in STARTUP_AXES
                    },
                    "episode": {
                        "scale_ranges": dict(
                            env_cfg.domain_randomization.episode.scale_ranges
                        )
                    },
                },
                "ctrl_dt": float(env.cfg.control.ctrl_dt),
                "sim_dt": float(env.cfg.sim_dt),
                "decimation": int(env.cfg.control.decimation),
                "episode_steps": int(env.max_episode_length),
                "native_topology": {
                    "cpu": (
                        "compact private model per environment"
                        if dr_bundle in {"mass", "all"}
                        else "shared-model serial worlds"
                    ),
                    "warp": f"batched {dr_bundle} parameter fields",
                    "vsim": (
                        f"{args.num_envs} environment sets of one"
                        if dr_bundle in {"friction", "mass", "all"}
                        else f"one environment set of {args.num_envs}"
                    ),
                }[args.backend],
                "native_topology_details": native_topology_details(env, args.backend),
            },
            "setup_seconds": setup_seconds,
            "memory": {
                "cuda_before": memory_before,
                "cuda_after_setup": memory_after_setup,
                "cuda_after_profiles": memory_after_profiles,
                "cuda_used_delta_bytes": (
                    None
                    if memory_before is None
                    else memory_before["free_bytes"] - memory_after_setup["free_bytes"]
                ),
                "cuda_used_after_profiles_delta_bytes": (
                    None
                    if memory_before is None
                    else memory_before["free_bytes"]
                    - memory_after_profiles["free_bytes"]
                ),
                "host_max_rss_before_kb": rss_before_kb,
                "host_max_rss_after_setup_kb": rss_after_setup_kb,
                "host_max_rss_after_profiles_kb": rss_after_profiles_kb,
                "host_max_rss_delta_kb": rss_after_setup_kb - rss_before_kb,
                "host_max_rss_after_profiles_delta_kb": (
                    rss_after_profiles_kb - rss_before_kb
                ),
            },
            "applied_domain_randomization": applied,
            "profiles": profiles,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        env._backend.close()


if __name__ == "__main__":
    main()
