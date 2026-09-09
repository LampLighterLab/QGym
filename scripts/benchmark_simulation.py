"""Controlled 100 Hz simulation timing; run each cell in a fresh process.

``run`` records warmed batches and physics evidence. ``compare`` consumes paired
fresh-process results, never treating batches from one process as independent.
GPU captures use this same workload through scripts.profile_simulation.
"""

import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch

from gym.utils.helpers import class_to_dict, set_seed
from gym.utils.task_registry import task_registry
from scripts.benchmark_domain_randomization import (
    cuda_memory_snapshot,
    git_metadata,
    make_reset_schedule,
    package_versions,
    synchronize,
    trace_range,
)
from scripts.benchmark_vsim_environment_sets import NominalSetGym, environment_set_sizes


ROOT = Path(__file__).resolve().parents[1]
PROFILES = (
    "task_empty",
    "task_timeout",
    "backend_step",
    *(
        f"{layer}_reset_{mask}"
        for layer in ("backend", "task")
        for mask in ("empty", "one", "sparse", "all")
    ),
)


def configure(task, num_envs, seed):
    import gym.envs  # noqa: F401 — register tasks

    cfg, runner = copy.deepcopy(task_registry.get_cfgs(task))
    cfg.seed = runner.seed = seed
    cfg.env.num_envs = num_envs
    cfg.env.episode_length_s = 10.0
    cfg.control.ctrl_frequency = cfg.control.desired_sim_frequency = 100
    runner.actor.frequency = 100
    cfg.init_state.reset_mode = "reset_to_basic"
    cfg.domain_randomization.startup.contact_friction_range = None
    cfg.domain_randomization.startup.link_mass_scale_range = None
    cfg.domain_randomization.episode.scale_ranges = {}
    if task == "go2trot":
        cfg.push_robots.toggle = False
        cfg.commands.resampling_time = 10000.0
        cfg.terrain.static_friction = cfg.terrain.dynamic_friction = 1.0
    task_registry.convert_frequencies_to_params(cfg, runner)
    return cfg


class Workload:
    def __init__(self, env, backend_label, task, seed, profile, steps):
        self.env, self.backend = env, env._backend
        self.device, self.backend_label = env.device, backend_label
        self.task, self.seed, self.profile = task, seed, profile
        self.all_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.masks = make_reset_schedule(
            "timeout" if profile == "task_timeout" else "none",
            env.num_envs,
            1000,
            steps,
            env.device,
        )
        if "_reset_" in profile:
            count = {
                "empty": 0,
                "one": 1,
                "sparse": max(1, env.num_envs // 100),
                "all": env.num_envs,
            }[profile.rsplit("_", 1)[1]]
            mask = torch.arange(env.num_envs, device=env.device) < count
            self.masks = [mask] * steps
        # A short, nonzero periodic input avoids allocating a rollout-sized tensor.
        phase = torch.arange(32, device=env.device)[:, None, None] * (2 * torch.pi / 32)
        joint = torch.arange(env.num_dof, device=env.device)[None, None, :] * 0.3
        self.inputs = (0.2 * torch.sin(phase + joint)).expand(-1, env.num_envs, -1)
        self.restore()

    def restore(self):
        """Restore task/RNG state and exposed native buffers outside timing.

        VSim does not expose contact/solver history; repeat-state evidence records
        whether that opaque history affects these measured trajectories.
        """
        env = self.env
        set_seed(self.seed)
        if self.backend_label == "cpu":
            for data in self.backend._datas:
                data.qacc_warmstart.fill(0)
                data.qfrc_applied.fill(0)
                data.time = 0.0
        elif self.backend_label == "warp":
            self.backend._d.qacc_warmstart.zero_()
            self.backend._d.qfrc_applied.zero_()
            self.backend._d.time.zero_()
        else:
            self.backend._motor_buf.zero_()
            self.backend._gym.set_motor_forces(self.backend._motor_arr)
        env.tau_ff.zero_()
        env.dof_pos_target.zero_()
        env.dof_vel_target.zero_()
        env.torques.zero_()
        env._reset_idx(self.all_mask)
        env.dof_pos_history.zero_()
        env.common_step_counter = 0
        env._reset_buffers()
        if self.task == "pendulum":
            env.dof_pos.fill_(0.4)
            env.dof_vel.zero_()
            self.backend.reset_state(self.all_mask)
            env.dof_pos_obs.copy_(
                torch.cat((torch.sin(env.dof_pos), torch.cos(env.dof_pos)), dim=-1)
            )
        self.reset_pos = env.dof_pos.clone()
        self.reset_vel = env.dof_vel.clone()
        self.reset_root = env.root_states.clone()

    def step(self, index):
        profile, env = self.profile, self.env
        mask = self.masks[index % len(self.masks)]
        if profile == "backend_step":
            self.backend.step(self.inputs[index % 32])
        elif profile.startswith("backend_reset_"):
            # Public-state writes are part of the atomic reset contract.
            env.dof_pos.copy_(self.reset_pos)
            env.dof_vel.copy_(self.reset_vel)
            env.root_states.copy_(self.reset_root)
            self.backend.reset_state(mask)
        elif profile.startswith("task_reset_"):
            env._reset_idx(mask)
        else:
            if self.task == "pendulum":
                env.tau_ff.copy_(self.inputs[index % 32])
            else:
                env.dof_pos_target.copy_(self.inputs[index % 32] * 0.1)
            env.step()
            env._reset_idx(mask)

    def close(self):
        self.backend.close()


def build_workload(task, backend_label, num_envs, seed, sets, profile, steps):
    cfg = configure(task, num_envs, seed)
    device = "cpu" if backend_label == "cpu" else "cuda:0"
    backend = "vsim" if backend_label == "vsim" else "mujoco"
    context = nullcontext()
    if backend_label == "vsim":
        import vlearn

        create_gym = vlearn.create_gym
        sizes = environment_set_sizes(num_envs, sets)
        context = patch.object(
            vlearn, "create_gym", lambda **kw: NominalSetGym(create_gym(**kw), sizes)
        )
    elif sets != 1:
        raise ValueError("--sets applies only to the VSim topology probe")
    with context:
        env = task_registry.make_env(
            task, cfg, device=device, backend=backend, headless=True
        )
    return Workload(env, backend_label, task, seed, profile, steps)


def run_steps(workload, steps):
    for index in range(steps):
        workload.step(index)


def profiled_steps(workload, steps):
    """Dedicated stack frame for filtering host samples to the timed region."""
    run_steps(workload, steps)
    synchronize(workload.backend_label, workload.device)


def instrument_capture(workload):
    """Label owned task/backend methods only in a requested Nsight capture."""

    def instrument(owner, name, label):
        original = getattr(owner, name)

        def traced(*args, **kwargs):
            with trace_range(label, workload.device):
                return original(*args, **kwargs)

        setattr(owner, name, traced)

    for owner, name, label in (
        (workload.env, "step", "task_step"),
        (workload.env, "_reset_idx", "task_reset"),
        (workload.env, "_post_physics_step", "derived_state"),
        (workload.backend, "step", "backend_step"),
        (workload.backend, "reset_state", "backend_reset"),
    ):
        instrument(owner, name, label)


def state_evidence(workload):
    backend = workload.backend
    result = {}
    for name in ("root_states", "dof_pos", "dof_vel", "contact_forces"):
        value = getattr(backend, name)
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"non-finite {name} after measured workload")
        result[name] = value[:4].detach().cpu().tolist()
    return result


def hardware(device):
    info = {
        "platform": platform.platform(),
        "cpu": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }
    if device.startswith("cuda"):
        info["gpu"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    return info


def source_hashes():
    paths = subprocess.check_output(
        ["rg", "--files", "gym", "learning", "scripts"], cwd=ROOT, text=True
    ).splitlines()
    paths += ["pyproject.toml", "uv.lock"]
    paths += subprocess.check_output(
        ["rg", "--files", "resources/robots/pendulum", "resources/robots/go2"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    return {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in sorted(paths)
        if p.endswith((".py", ".toml", ".lock", ".urdf", ".xml"))
    }


def measure(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    provenance, hashes = git_metadata(), source_hashes()
    start = time.perf_counter()
    workload = build_workload(
        args.task,
        args.backend,
        args.num_envs,
        args.seed,
        args.sets,
        args.profile,
        max(args.steps, args.warmup),
    )
    try:
        synchronize(args.backend, workload.device)
        setup_seconds = time.perf_counter() - start
        if args.capture:
            instrument_capture(workload)
        start = time.perf_counter()
        run_steps(workload, args.warmup)
        synchronize(args.backend, workload.device)
        warmup_seconds = time.perf_counter() - start
        durations, states = [], []
        for _ in range(args.repeats):
            workload.restore()
            initial = state_evidence(workload)
            synchronize(args.backend, workload.device)
            if args.capture:
                torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push("q2_measure")
            try:
                start = time.perf_counter()
                profiled_steps(workload, args.steps)
                durations.append(time.perf_counter() - start)
            finally:
                if args.capture:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.profiler.stop()
            states.append(state_evidence(workload))
            if args.profile in ("backend_step", "task_empty", "task_timeout"):
                if np.allclose(
                    states[-1]["dof_pos"], initial["dof_pos"], atol=1e-8, rtol=0
                ):
                    raise RuntimeError(
                        "measured stepping did not change joint positions"
                    )
        native_topology = None
        if args.backend == "vsim":
            group = workload.backend._grp
            native_topology = {
                "sets": group.get_num_environment_sets(),
                "environments": group.get_num_environments(),
            }
        return {
            "schema": 1,
            "protocol": {
                key: getattr(args, key)
                for key in (
                    "task",
                    "backend",
                    "num_envs",
                    "sets",
                    "seed",
                    "profile",
                    "steps",
                    "warmup",
                    "repeats",
                    "threads",
                )
            },
            "config": class_to_dict(workload.env.cfg),
            "hardware": hardware(workload.device),
            "packages": {**package_versions(), "torch_cuda": torch.version.cuda},
            "provenance": provenance,
            "source_sha256": hashes,
            "setup_s": setup_seconds,
            "warmup_s": warmup_seconds,
            "cuda_memory": cuda_memory_snapshot(workload.device),
            "native_topology": native_topology,
            "capture": args.capture,
            "durations_s": durations,
            "median_s": statistics.median(durations),
            "median_env_steps_per_s": args.steps
            * args.num_envs
            / statistics.median(durations),
            "reset_count": sum(int(m.sum()) for m in workload.masks[: args.steps]),
            "states": states,
            "initial_state": initial,
            "state_repeat_max_abs_delta": {
                key: float(
                    np.ptp(np.asarray([state[key] for state in states]), axis=0).max()
                )
                for key in states[0]
            },
        }
    finally:
        workload.close()


def compare_pairs(references, candidates, threshold=0.05):
    ratios = []
    for reference, candidate in zip(references, candidates, strict=True):
        for key in (
            "schema",
            "protocol",
            "config",
            "hardware",
            "packages",
            "native_topology",
            "reset_count",
        ):
            if reference[key] != candidate[key] or reference[key] != references[0][key]:
                raise ValueError(f"incomparable benchmark {key}")
        if reference["capture"] or candidate["capture"]:
            raise ValueError("profiler timings cannot be used as performance evidence")
        # Contacts are derived solver outputs, not requested initial conditions.
        # GPU reduction roundoff belongs in physics diagnostics, not this identity.
        for name in ("root_states", "dof_pos", "dof_vel"):
            if not np.allclose(
                reference["initial_state"][name],
                candidate["initial_state"][name],
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError(f"incomparable initial state: {name}")
        ratios.append(candidate["median_s"] / reference["median_s"])
    # Resample independent paired processes, not correlated within-process batches.
    rng = np.random.default_rng(7)
    samples = rng.choice(ratios, size=(10000, len(ratios)), replace=True)
    low, high = np.quantile(np.median(samples, axis=1), [0.025, 0.975])
    median = statistics.median(ratios)
    status = (
        "regression"
        if low > 1 + threshold
        else "pass"
        if high <= 1 + threshold
        else "inconclusive"
    )
    if len(ratios) < 5:
        status = "insufficient_pairs"
    return {
        "status": status,
        "paired_ratios": ratios,
        "median_ratio": median,
        "interval_95": [float(low), float(high)],
        "threshold": threshold,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--task", choices=("pendulum", "go2trot"), default="go2trot")
    run.add_argument("--backend", choices=("cpu", "warp", "vsim"), required=True)
    run.add_argument("--num-envs", type=int, default=4096)
    run.add_argument("--profile", choices=PROFILES, default="task_timeout")
    run.add_argument("--steps", type=int, default=250)
    run.add_argument("--warmup", type=int, default=100)
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--sets", type=int, default=1)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--capture", action="store_true")
    run.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--reference", nargs="+", type=Path, required=True)
    compare.add_argument("--candidate", nargs="+", type=Path, required=True)
    compare.add_argument("--threshold", type=float, default=0.05)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = measure(args)
    else:
        result = compare_pairs(
            [json.loads(p.read_text()) for p in args.reference],
            [json.loads(p.read_text()) for p in args.candidate],
            args.threshold,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, default=lambda x: x.tolist()) + "\n"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    ("median_s", "median_env_steps_per_s")
                    if args.command == "run"
                    else ("status", "median_ratio", "interval_95")
                )
            }
        )
    )
    if args.command == "compare" and result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
