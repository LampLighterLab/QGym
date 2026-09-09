#!/usr/bin/env python3
"""Run Q2's complete multi-axis domain-randomization campaign.

The campaign is intentionally large and resumable. CPU children run in a
small parallel pool while Warp and VSim share one serialized GPU lane. Launch
the parent with the VSim environment so every child inherits the license and
loader configuration:

    uv run --env-file .env.vsim \
        scripts/run_full_domain_randomization_campaign.py

Interrupted training cells restart from their original seed. Q2 checkpoints
do not contain simulator state or every RNG, so continuing a partial run would
not be the same controlled experiment.
"""

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO_ROOT / "logs"
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Backend:
    label: str
    physics_backend: str
    device: str


BACKENDS = (
    Backend("cpu", "mujoco", "cpu"),
    Backend("warp", "mujoco", "cuda:0"),
    Backend("vsim", "vsim", "cuda:0"),
)

BUNDLES = {
    "off": "off",
    "friction": "friction-only",
    "pd": "pd-only",
    "mass": "mass-only",
    "all": "config",
}

EVAL_DOMAINS = {
    "nominal": {"mode": "off"},
    "friction_in": {
        "mode": "friction-only",
        "contact_friction": [0.5, 1.0],
    },
    "pd_in": {
        "mode": "pd-only",
        "stiffness_scale": [0.9, 1.1],
        "damping_scale": [0.9, 1.1],
    },
    "mass_in": {
        "mode": "mass-only",
        "link_mass_scale": [0.9, 1.1],
    },
    "combined_in": {
        "mode": "config",
        "contact_friction": [0.5, 1.0],
        "stiffness_scale": [0.9, 1.1],
        "damping_scale": [0.9, 1.1],
        "link_mass_scale": [0.9, 1.1],
    },
    "friction_stress": {
        "mode": "friction-only",
        "contact_friction": [0.35, 1.15],
    },
    "pd_stress": {
        "mode": "pd-only",
        "stiffness_scale": [0.8, 1.2],
        "damping_scale": [0.8, 1.2],
    },
    "mass_stress": {
        "mode": "mass-only",
        "link_mass_scale": [0.8, 1.2],
    },
    "combined_stress": {
        "mode": "config",
        "contact_friction": [0.35, 1.15],
        "stiffness_scale": [0.8, 1.2],
        "damping_scale": [0.8, 1.2],
        "link_mass_scale": [0.8, 1.2],
    },
}

ENGINE_OVERFLOW_MARKERS = (
    "nefc overflow",
    "njmax_nnz overflow",
    "broadphase overflow",
    "narrowphase overflow",
)

EXECUTION_SOURCES = (
    "pyproject.toml",
    "uv.lock",
    "scripts/train.py",
    "scripts/train_domain_randomization.py",
    "scripts/eval_policy.py",
    "scripts/benchmark_domain_randomization.py",
    "scripts/run_full_domain_randomization_campaign.py",
    "resources/robots/go2/urdf/go2.urdf",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def provenance_paths():
    paths = [Path(name) for name in EXECUTION_SOURCES]
    for root_name in ("gym", "learning"):
        root = REPO_ROOT / root_name
        paths.extend(
            path.relative_to(REPO_ROOT)
            for path in root.rglob("*.py")
            if not path.name.startswith("test_") and "tests" not in path.parts
        )
    return tuple(sorted(set(paths)))


def snapshot_sources(campaign_dir, copy_sources):
    records = {}
    for relative in provenance_paths():
        source = REPO_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"campaign source is missing: {source}")
        digest = sha256_file(source)
        snapshot = campaign_dir / "provenance" / relative
        if copy_sources:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, snapshot)
            if sha256_file(snapshot) != digest:
                raise RuntimeError(f"source changed while copying: {source}")
        records[relative.as_posix()] = {
            "sha256": digest,
            "bytes": source.stat().st_size,
            "snapshot": str(Path("provenance") / relative),
        }
    return records


def backend_by_label(label):
    return next(backend for backend in BACKENDS if backend.label == label)


def protocol_backends(protocol):
    return tuple(backend_by_label(item["label"]) for item in protocol["backends"])


def apply_bundle(cfg, bundle):
    from gym.envs.base.domain_randomization import apply_domain_randomization_override

    apply_domain_randomization_override(cfg, BUNDLES[bundle])


def resolved_config(task, backend, seed, num_envs, iterations, save_interval, bundle):
    import gym.envs  # noqa: F401
    from gym.utils.helpers import class_to_dict
    from gym.utils.task_registry import task_registry

    registered_env, registered_train = task_registry.get_cfgs(task)
    env_cfg = copy.deepcopy(registered_env)
    train_cfg = copy.deepcopy(registered_train)
    env_cfg.seed = seed
    env_cfg.env.num_envs = num_envs
    train_cfg.seed = seed
    train_cfg.runner.device = backend.device
    train_cfg.runner.max_iterations = iterations
    train_cfg.runner.save_interval = save_interval
    apply_bundle(env_cfg, bundle)
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    return {
        "environment": class_to_dict(env_cfg),
        "training": class_to_dict(train_cfg),
    }


def make_protocol(args):
    selected_backends = tuple(backend_by_label(label) for label in args.backends)
    template = resolved_config(
        args.task,
        selected_backends[0],
        args.seeds[0],
        args.train_num_envs,
        args.train_iterations,
        args.save_interval,
        "all",
    )
    algorithm = template["training"]["algorithm"]
    gait_frequency = template["environment"]["control"]["gait_freq"]
    rollout_size = int(algorithm["rollout_size"])
    if rollout_size % args.train_num_envs:
        raise ValueError(
            f"rollout_size {rollout_size} must divide across "
            f"{args.train_num_envs} environments"
        )
    checkpoints = sorted(set(args.checkpoints))
    if checkpoints[-1] != args.train_iterations:
        raise ValueError("the final checkpoint must equal train_iterations")
    invalid = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint <= 0
        or checkpoint > args.train_iterations
        or checkpoint % args.save_interval
    ]
    if invalid:
        raise ValueError(
            "checkpoints must be positive save-interval multiples no later than "
            f"the final iteration; invalid={invalid}"
        )
    # In-range evaluations must cover the actual frozen training support.
    # Stress domains retain their explicitly declared ranges.
    domains = copy.deepcopy(EVAL_DOMAINS)
    dr = template["environment"]["domain_randomization"]
    startup = dr["startup"]
    gains = dr["episode"]["scale_ranges"]
    in_ranges = {
        "contact_friction": startup["contact_friction_range"],
        "stiffness_scale": gains["p_gains"],
        "damping_scale": gains["d_gains"],
        "link_mass_scale": startup["link_mass_scale_range"],
    }
    for name in ("friction_in", "pd_in", "mass_in", "combined_in"):
        for axis in domains[name].keys() - {"mode"}:
            domains[name][axis] = copy.deepcopy(in_ranges[axis])
    return {
        "task": args.task,
        "backends": [asdict(backend) for backend in selected_backends],
        "bundles": {name: BUNDLES[name] for name in args.bundles},
        "excluded_training": list(args.exclude_training),
        "include_speed": not args.skip_speed,
        "evaluation_schedule": (
            "after_training_cell"
            if args.evaluate_after_training
            else "after_training_stage"
        ),
        "seeds": list(args.seeds),
        "speed_common_num_envs": args.speed_common_num_envs,
        "speed_production_num_envs": args.speed_production_num_envs,
        "speed_target_env_steps": args.speed_target_env_steps,
        "speed_min_steps": args.speed_min_steps,
        "speed_repeats": args.speed_repeats,
        "train_num_envs": args.train_num_envs,
        "train_iterations": args.train_iterations,
        "save_interval": args.save_interval,
        "checkpoints": checkpoints,
        "rollout_size": rollout_size,
        "rollout_steps_per_env": rollout_size // args.train_num_envs,
        "optimizer_batch_size": int(algorithm["batch_size"]),
        "max_gradient_steps": int(algorithm["max_gradient_steps"]),
        "eval_num_envs": args.eval_num_envs,
        "eval_duration_s": args.eval_duration,
        "eval_settling_time_s": args.eval_settling_time,
        "eval_seed": args.eval_seed,
        "eval_contact_threshold_n": args.eval_contact_threshold,
        "eval_phase_frequency_hz": 0.5 * sum(map(float, gait_frequency)),
        "eval_domains": {name: domains[name] for name in args.eval_domains},
        "intermediate_domains": ["nominal", "combined_in"],
        "final_cross_backend": True,
        "evaluation_sampling": {
            "contact_friction": "evenly spaced scalar levels per command case",
            "pd_gains": (
                "deterministic independent per-actuator uniform samples, reused "
                "across command cases"
            ),
            "link_mass": (
                "deterministic independent per-link uniform samples, reused "
                "across command cases"
            ),
        },
        "environment_config_template": template["environment"],
        "training_config_template": template["training"],
        "interpretation": {
            "training_reward": "diagnostic learning signal, not policy quality",
            "seed_evidence": (
                f"{len(args.seeds)} paired seed(s); fewer than three is a pilot, "
                "not promotion evidence"
            ),
            "nominal_gate": "flag survival regression larger than 0.05 absolute",
            "robustness": (
                "advance a bundle only if held-out robustness improves without "
                "unexplained nominal collapse"
            ),
            "resume": (
                "interrupted training restarts from seed because checkpoints omit "
                "simulator and complete RNG state"
            ),
        },
    }


def speed_cases(protocol):
    if not protocol["include_speed"]:
        return []
    cases = []
    for backend in protocol_backends(protocol):
        counts = [protocol["speed_common_num_envs"]]
        if backend.device.startswith("cuda"):
            counts.append(protocol["speed_production_num_envs"])
        for num_envs in dict.fromkeys(counts):
            for bundle in protocol["bundles"]:
                label = f"{backend.label}_{num_envs}_{bundle}"
                cases.append(
                    {
                        "stage": "speed",
                        "label": label,
                        "backend": backend.label,
                        "bundle": bundle,
                        "num_envs": num_envs,
                        "artifact": f"speed/{label}.json",
                        "summary": f"cell_summaries/speed/{label}.json",
                    }
                )
    return cases


def training_cases(protocol):
    return [
        {
            "stage": "train",
            "label": f"{backend.label}_{bundle}_seed{seed}",
            "backend": backend.label,
            "bundle": bundle,
            "seed": seed,
            "artifact": f"training/{backend.label}_{bundle}_seed{seed}.json",
            "summary": (
                f"cell_summaries/train/{backend.label}_{bundle}_seed{seed}.json"
            ),
        }
        for seed in protocol["seeds"]
        for bundle in protocol["bundles"]
        for backend in protocol_backends(protocol)
        if f"{backend.label}:{bundle}" not in protocol["excluded_training"]
    ]


def evaluation_cases(protocol):
    cases = []
    final_checkpoint = protocol["train_iterations"]
    for training in training_cases(protocol):
        training_backend = backend_by_label(training["backend"])
        for checkpoint in protocol["checkpoints"]:
            if checkpoint == final_checkpoint:
                eval_backends = protocol_backends(protocol)
                domains = protocol["eval_domains"]
            else:
                eval_backends = (training_backend,)
                domains = {
                    name: protocol["eval_domains"][name]
                    for name in protocol["intermediate_domains"]
                }
            for eval_backend in eval_backends:
                for domain in domains:
                    label = (
                        f"{training['label']}__ckpt{checkpoint}__"
                        f"{eval_backend.label}__{domain}"
                    )
                    cases.append(
                        {
                            "stage": "eval",
                            "label": label,
                            "training_label": training["label"],
                            "train_backend": training["backend"],
                            "bundle": training["bundle"],
                            "seed": training["seed"],
                            "checkpoint_iteration": checkpoint,
                            "eval_backend": eval_backend.label,
                            "domain": domain,
                            "artifact": f"evaluation/{label}.npz",
                            "summary": f"cell_summaries/eval/{label}.json",
                        }
                    )
    return cases


def campaign_cases(protocol):
    return [
        *speed_cases(protocol),
        *training_cases(protocol),
        *evaluation_cases(protocol),
    ]


def cell_key(case):
    return f"{case['stage']}:{case['label']}"


def create_manifest(campaign_dir, protocol, provenance):
    cells = {}
    for case in campaign_cases(protocol):
        cells[cell_key(case)] = {
            **case,
            "status": "pending",
            "attempts": [],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "running",
        "git": git_metadata(),
        "protocol": protocol,
        "source_provenance": provenance,
        "cells": cells,
    }


def load_or_create_manifest(campaign_dir, protocol, resume):
    manifest_path = campaign_dir / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"{campaign_dir} exists; pass --resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("campaign schema mismatch")
        if manifest["protocol"] != protocol:
            raise ValueError("cannot resume with a changed scientific protocol")
        if not source_provenance_valid(campaign_dir, manifest["source_provenance"]):
            raise ValueError("cannot resume after execution sources changed")
        for cell in manifest["cells"].values():
            if cell["status"] == "running":
                cell["status"] = "pending"
                cell["interrupted_at"] = utc_now()
        return manifest

    campaign_dir.mkdir(parents=True, exist_ok=False)
    provenance = snapshot_sources(campaign_dir, copy_sources=True)
    manifest = create_manifest(campaign_dir, protocol, provenance)
    write_json(manifest_path, manifest)
    return manifest


def source_provenance_valid(campaign_dir, expected):
    if snapshot_sources(campaign_dir, copy_sources=False) != expected:
        return False
    for record in expected.values():
        snapshot = campaign_dir / record["snapshot"]
        if (
            not snapshot.is_file()
            or snapshot.stat().st_size != record["bytes"]
            or sha256_file(snapshot) != record["sha256"]
        ):
            return False
    return True


def checkpoint_valid(path, expected_iteration):
    import torch

    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    except Exception:
        return False
    required = {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "critic_optimizer_state_dict",
        "iter",
    }
    try:
        return (
            required <= set(checkpoint)
            and int(checkpoint["iter"]) == expected_iteration
            and checkpoint_tensors_finite(checkpoint)
        )
    except (TypeError, ValueError):
        return False


def checkpoint_tensors_finite(value):
    import torch

    if isinstance(value, torch.Tensor):
        if not (value.is_floating_point() or value.is_complex()):
            return True
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(checkpoint_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(checkpoint_tensors_finite(item) for item in value)
    return True


def numeric_values_finite(value):
    if isinstance(value, dict):
        return all(numeric_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(numeric_values_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    return True


def checkpoint_record_valid(record):
    try:
        path = Path(record["path"])
        return (
            checkpoint_valid(path, int(record["iteration"]))
            and path.stat().st_size == int(record["bytes"])
            and sha256_file(path) == record["sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def validate_speed(path, case):
    if not path.is_file():
        return False
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        protocol = artifact["protocol"]
        profiles = artifact["profiles"]
    except (KeyError, json.JSONDecodeError):
        return False
    return (
        protocol.get("backend_label") == case["backend"]
        and protocol.get("dr_bundle") == case["bundle"]
        and int(protocol.get("num_envs", -1)) == case["num_envs"]
        and {profile.get("profile") for profile in profiles}
        == {"none", "timeout", "all"}
    )


def validate_training(path, case, protocol):
    if not path.is_file():
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if any(
        marker.get(name) != case[name]
        for name in ("label", "backend", "bundle", "seed")
    ):
        return False
    run_dir = Path(marker.get("run_dir", ""))
    vitals = run_dir / "vitals.jsonl"
    if not run_dir.is_dir() or not vitals.is_file():
        return False
    try:
        records = [
            json.loads(line)
            for line in vitals.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (json.JSONDecodeError, OSError):
        return False
    if (
        not records
        or not numeric_values_finite(records)
        or int(records[-1].get("iteration", -1)) != protocol["train_iterations"]
    ):
        return False
    resolved_path = Path(marker.get("resolved_config", ""))
    try:
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    identity = {name: case[name] for name in ("label", "backend", "bundle", "seed")}
    expected_resolved = {
        **identity,
        **resolved_config(
            protocol["task"],
            backend_by_label(case["backend"]),
            case["seed"],
            protocol["train_num_envs"],
            protocol["train_iterations"],
            protocol["save_interval"],
            case["bundle"],
        ),
    }
    if resolved != expected_resolved:
        return False
    checkpoints = marker.get("checkpoints", [])
    if [item.get("iteration") for item in checkpoints] != protocol["checkpoints"]:
        return False
    return all(checkpoint_record_valid(item) for item in checkpoints)


def training_checkpoint(campaign_dir, case):
    marker_path = campaign_dir / "training" / f"{case['training_label']}.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    return next(
        item
        for item in marker["checkpoints"]
        if item["iteration"] == case["checkpoint_iteration"]
    )


def validate_evaluation(path, case, protocol, campaign_dir):
    if not path.is_file():
        return False
    try:
        checkpoint = training_checkpoint(campaign_dir, case)
        if not checkpoint_record_valid(checkpoint):
            return False
        with np.load(path, allow_pickle=False) as artifact:
            domain = protocol["eval_domains"][case["domain"]]
            grid_names = {
                "contact_friction": "contact_friction_grid",
                "stiffness_scale": "stiffness_scale_range",
                "damping_scale": "damping_scale_range",
                "link_mass_scale": "link_mass_scale_range",
            }
            grids_match = all(
                np.allclose(
                    artifact[artifact_name],
                    domain.get(parameter_name, [np.nan, np.nan]),
                    equal_nan=True,
                )
                for parameter_name, artifact_name in grid_names.items()
            )
            return (
                int(artifact["checkpoint_iteration"]) == case["checkpoint_iteration"]
                and int(artifact["num_envs"]) == protocol["eval_num_envs"]
                and int(artifact["seed"]) == protocol["eval_seed"]
                and str(artifact["task"]) == protocol["task"]
                and str(artifact["train_label"]) == case["training_label"]
                and str(artifact["eval_label"]) == case["eval_backend"]
                and Path(str(artifact["checkpoint_path"])).resolve()
                == Path(checkpoint["path"]).resolve()
                and str(artifact["checkpoint_sha256"]) == checkpoint["sha256"]
                and str(artifact["reset_mode"]) == "reset_to_basic"
                and str(artifact["command_profile"]) == "go2"
                and np.isclose(
                    float(artifact["duration_s"]), protocol["eval_duration_s"]
                )
                and np.isclose(
                    float(artifact["settling_time_s"]),
                    protocol["eval_settling_time_s"],
                )
                and np.isclose(
                    float(artifact["contact_threshold_n"]),
                    protocol["eval_contact_threshold_n"],
                )
                and np.allclose(
                    artifact["phase_frequency_hz"],
                    protocol["eval_phase_frequency_hz"],
                )
                and str(artifact["domain_randomization"]) == domain["mode"]
                and bool(artifact["original_cfg"])
                and grids_match
                and np.isfinite(artifact["contact_friction"]).all()
                and np.isfinite(artifact["stiffness_scale"]).all()
                and np.isfinite(artifact["damping_scale"]).all()
                and np.isfinite(artifact["link_mass_scale"]).all()
                and artifact["body_names"].shape[0]
                == artifact["link_mass_scale"].shape[1]
            )
    except (KeyError, OSError, StopIteration, ValueError):
        return False


def validate_cell(campaign_dir, cell, protocol):
    path = campaign_dir / cell["artifact"]
    if cell["stage"] == "speed":
        return validate_speed(path, cell)
    if cell["stage"] == "train":
        return validate_training(path, cell, protocol)
    return validate_evaluation(path, cell, protocol, campaign_dir)


def subprocess_environment(cell):
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLBACKEND"] = "Agg"
    backend = cell.get("backend", cell.get("eval_backend"))
    environment["OMP_NUM_THREADS"] = "4" if backend == "cpu" else "2"
    environment["MKL_NUM_THREADS"] = environment["OMP_NUM_THREADS"]
    return environment


def speed_command(campaign_dir, cell, protocol):
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "benchmark_domain_randomization.py"),
        "--task",
        protocol["task"],
        "--backend",
        cell["backend"],
        "--dr",
        cell["bundle"],
        "--num_envs",
        str(cell["num_envs"]),
        "--seed",
        str(protocol["seeds"][0]),
        "--target_env_steps",
        str(protocol["speed_target_env_steps"]),
        "--min_steps",
        str(protocol["speed_min_steps"]),
        "--repeats",
        str(protocol["speed_repeats"]),
        "--out",
        str(campaign_dir / cell["artifact"]),
    ]


def training_experiment(campaign_dir, cell):
    relative = campaign_dir.relative_to(LOG_ROOT)
    return relative / "training_runs" / cell["label"]


def write_resolved_training_config(campaign_dir, cell, protocol):
    backend = backend_by_label(cell["backend"])
    config = resolved_config(
        protocol["task"],
        backend,
        cell["seed"],
        protocol["train_num_envs"],
        protocol["train_iterations"],
        protocol["save_interval"],
        cell["bundle"],
    )
    path = campaign_dir / "training" / f"{cell['label']}.resolved.json"
    identity = {name: cell[name] for name in ("label", "backend", "bundle", "seed")}
    write_json(path, {**identity, **config})
    return path


def training_command(campaign_dir, cell, protocol):
    backend = backend_by_label(cell["backend"])
    write_resolved_training_config(campaign_dir, cell, protocol)
    return [
        sys.executable,
        "-X",
        "faulthandler",
        "-m",
        "scripts.train_domain_randomization",
        "--task",
        protocol["task"],
        "--backend",
        backend.physics_backend,
        "--device",
        backend.device,
        "--num_envs",
        str(protocol["train_num_envs"]),
        "--max_iterations",
        str(protocol["train_iterations"]),
        "--save_interval",
        str(protocol["save_interval"]),
        "--seed",
        str(cell["seed"]),
        "--experiment_name",
        str(training_experiment(campaign_dir, cell)),
        "--dr-bundle",
        cell["bundle"],
        "--headless",
        "--disable_wandb",
    ]


def completed_training_marker(campaign_dir, training_label):
    path = campaign_dir / "training" / f"{training_label}.json"
    if not path.is_file():
        raise FileNotFoundError(f"training marker is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def domain_arguments(domain, domains=EVAL_DOMAINS):
    definition = domains[domain]
    arguments = ["--domain-randomization", definition["mode"]]
    option_names = {
        "contact_friction": "--contact_friction_grid",
        "stiffness_scale": "--stiffness-scale-range",
        "damping_scale": "--damping-scale-range",
        "link_mass_scale": "--link-mass-scale-range",
    }
    for name, option in option_names.items():
        if name in definition:
            arguments.extend([option, *map(str, definition[name])])
    return arguments


def evaluation_command(campaign_dir, cell, protocol):
    marker = completed_training_marker(campaign_dir, cell["training_label"])
    checkpoint = next(
        item["path"]
        for item in marker["checkpoints"]
        if item["iteration"] == cell["checkpoint_iteration"]
    )
    backend = backend_by_label(cell["eval_backend"])
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_policy.py"),
        "--task",
        protocol["task"],
        "--ckpt",
        checkpoint,
        "--train_label",
        cell["training_label"],
        "--eval_backend",
        backend.physics_backend,
        "--eval_device",
        backend.device,
        "--eval_label",
        backend.label,
        "--num_envs",
        str(protocol["eval_num_envs"]),
        "--t_end",
        str(protocol["eval_duration_s"]),
        "--seed",
        str(protocol["eval_seed"]),
        "--reset_mode",
        "reset_to_basic",
        "--command_profile",
        "go2",
        "--settling_time",
        str(protocol["eval_settling_time_s"]),
        "--contact_threshold",
        str(protocol["eval_contact_threshold_n"]),
        "--original_cfg",
        "--mujoco_njmax",
        "256",
        *domain_arguments(cell["domain"], protocol["eval_domains"]),
        "--out",
        str(campaign_dir / cell["artifact"]),
    ]


def command_for_cell(campaign_dir, cell, protocol):
    if cell["stage"] == "speed":
        return speed_command(campaign_dir, cell, protocol)
    if cell["stage"] == "train":
        return training_command(campaign_dir, cell, protocol)
    return evaluation_command(campaign_dir, cell, protocol)


def find_training_run(campaign_dir, cell, protocol):
    experiment_dir = LOG_ROOT / training_experiment(campaign_dir, cell)
    final_name = f"model_{protocol['train_iterations']}.pt"
    candidates = sorted(
        (
            checkpoint.parent
            for checkpoint in experiment_dir.glob(f"*/{final_name}")
            if checkpoint_valid(checkpoint, protocol["train_iterations"])
        ),
        key=lambda path: (path / final_name).stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no valid {final_name} under {experiment_dir}")
    return candidates[0]


def training_summary(cell, marker):
    records = [
        json.loads(line)
        for line in (Path(marker["run_dir"]) / "vitals.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    curve_keys = (
        "iteration",
        "rewards/total_rewards",
        "episode_time",
        "steps_per_s",
        "t_collection",
        "t_learning",
        "algorithm/mean_value_loss",
        "algorithm/mean_surrogate_loss",
        "algorithm/learning_rate",
        "actor/action_std",
    )
    return {
        **{name: cell[name] for name in ("label", "backend", "bundle", "seed")},
        "run_dir": marker["run_dir"],
        "vitals_path": str(Path(marker["run_dir"]) / "vitals.jsonl"),
        "checkpoints": marker["checkpoints"],
        "curve": [
            {name: record.get(name) for name in curve_keys if name in record}
            for record in records
        ],
    }


def finalize_training(campaign_dir, cell, protocol):
    run_dir = find_training_run(campaign_dir, cell, protocol)
    checkpoints = []
    for iteration in protocol["checkpoints"]:
        path = run_dir / f"model_{iteration}.pt"
        if not checkpoint_valid(path, iteration):
            raise RuntimeError(f"invalid required checkpoint: {path}")
        checkpoints.append(
            {
                "iteration": iteration,
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    marker = {
        **{name: cell[name] for name in ("label", "backend", "bundle", "seed")},
        "run_dir": str(run_dir),
        "resolved_config": str(
            campaign_dir / "training" / f"{cell['label']}.resolved.json"
        ),
        "checkpoints": checkpoints,
    }
    write_json(campaign_dir / cell["artifact"], marker)
    write_json(campaign_dir / cell["summary"], training_summary(cell, marker))


def descriptive(values, direction):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    count = max(1, math.ceil(0.1 * len(values)))
    if direction == "higher":
        tail = np.partition(values, count - 1)[:count]
    elif direction == "closer_to_one":
        indices = np.argpartition(np.abs(values - 1.0), -count)[-count:]
        tail = values[indices]
    else:
        tail = np.partition(values, len(values) - count)[-count:]
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "p10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "worst_decile_mean": float(tail.mean()),
    }


def summarize_evaluation(path, cell):
    with np.load(path, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["hardware_metric_metadata"]))
        metrics = {}
        for name in map(str, artifact["hardware_metric_names"]):
            definition = metadata[name]
            statistics = descriptive(
                artifact[f"metric_{name}"], definition["direction"]
            )
            if statistics is not None:
                metrics[name] = {**definition, **statistics}
        metrics["reward_diagnostic"] = {
            "unit": "reward/step",
            "direction": "higher",
            "description": "Training reward terms, excluding termination.",
            **descriptive(artifact["mean_reward"], "higher"),
        }
        return {
            **{
                name: cell[name]
                for name in (
                    "label",
                    "training_label",
                    "train_backend",
                    "bundle",
                    "seed",
                    "checkpoint_iteration",
                    "eval_backend",
                    "domain",
                )
            },
            "artifact": str(path),
            "metrics": metrics,
        }


def finalize_speed(campaign_dir, cell):
    artifact = json.loads((campaign_dir / cell["artifact"]).read_text(encoding="utf-8"))
    summary = {
        **{name: cell[name] for name in ("label", "backend", "bundle", "num_envs")},
        "setup_seconds": artifact["setup_seconds"],
        "memory": artifact["memory"],
        "profiles": artifact["profiles"],
        "applied_domain_randomization": artifact["applied_domain_randomization"],
    }
    write_json(campaign_dir / cell["summary"], summary)


def finalize_evaluation(campaign_dir, cell):
    path = campaign_dir / cell["artifact"]
    write_json(campaign_dir / cell["summary"], summarize_evaluation(path, cell))


def refresh_cell_summary(campaign_dir, cell):
    if cell["stage"] == "speed":
        finalize_speed(campaign_dir, cell)
    elif cell["stage"] == "train":
        marker = json.loads(
            (campaign_dir / cell["artifact"]).read_text(encoding="utf-8")
        )
        write_json(campaign_dir / cell["summary"], training_summary(cell, marker))
    else:
        finalize_evaluation(campaign_dir, cell)


def reject_engine_overflow(log_path):
    output = log_path.read_text(encoding="utf-8", errors="replace")
    for marker in ENGINE_OVERFLOW_MARKERS:
        if marker in output:
            raise RuntimeError(f"physics capacity overflow in child log: {marker}")


def finalize_cell(campaign_dir, cell, protocol):
    if cell["stage"] == "speed":
        finalize_speed(campaign_dir, cell)
    elif cell["stage"] == "train":
        finalize_training(campaign_dir, cell, protocol)
    else:
        finalize_evaluation(campaign_dir, cell)
    if not validate_cell(campaign_dir, cell, protocol):
        raise RuntimeError(f"artifact validation failed for {cell_key(cell)}")


class Campaign:
    def __init__(self, directory, manifest):
        self.directory = directory
        self.manifest_path = directory / "manifest.json"
        self.manifest = manifest
        self.protocol = manifest["protocol"]
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.process_lock = threading.Lock()
        self.processes = {}

    def request_stop(self):
        self.stop_event.set()
        with self.process_lock:
            processes = list(self.processes.values())
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    def save(self):
        self.manifest["updated_at"] = utc_now()
        write_json(self.manifest_path, self.manifest)

    def update(self, key, **changes):
        with self.lock:
            self.manifest["cells"][key].update(changes)
            self.save()

    def run_cell(self, key):
        if self.stop_event.is_set():
            return False
        cell = self.manifest["cells"][key]
        try:
            artifact_valid = validate_cell(self.directory, cell, self.protocol)
        except Exception as error:
            artifact_valid = False
            print(f"[audit] {key}: {error}", flush=True)
        if artifact_valid:
            try:
                refresh_cell_summary(self.directory, cell)
            except Exception as error:
                self.update(key, status="failed", error=str(error))
                print(f"[failed] {key}: {error}", flush=True)
                return False
            self.update(key, status="complete", reused=True, error=None)
            return True
        attempt = len(cell["attempts"]) + 1
        log_path = (
            self.directory
            / "stdout"
            / f"{cell['stage']}_{cell['label']}.attempt{attempt}.log"
        )
        started = utc_now()
        attempt_record = {
            "attempt": attempt,
            "started_at": started,
            "finished_at": None,
            "wall_seconds": None,
            "returncode": None,
            "log": str(log_path),
        }
        with self.lock:
            cell["attempts"].append(attempt_record)
            self.save()
        try:
            if not source_provenance_valid(
                self.directory, self.manifest["source_provenance"]
            ):
                raise RuntimeError(
                    "execution sources changed after the campaign was created"
                )
            command = command_for_cell(self.directory, cell, self.protocol)
        except Exception as error:
            attempt_record["finished_at"] = utc_now()
            with self.lock:
                cell.update(status="failed", error=str(error))
                self.save()
            print(f"[failed] {key}: {error}", flush=True)
            return False
        self.update(
            key,
            status="running",
            started_at=started,
            command=command,
            log=str(log_path),
        )
        print(f"[start] {key}", flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=subprocess_environment(cell),
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with self.process_lock:
                self.processes[key] = process
                stop_requested = self.stop_event.is_set()
            if stop_requested:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            try:
                returncode = process.wait()
            finally:
                with self.process_lock:
                    self.processes.pop(key, None)
        elapsed = time.perf_counter() - start_time
        attempt_record.update(
            finished_at=utc_now(),
            wall_seconds=elapsed,
            returncode=returncode,
        )
        if returncode and self.stop_event.is_set():
            with self.lock:
                cell.update(
                    status="pending",
                    error="interrupted; this controlled cell restarts from its seed",
                )
                self.save()
            print(f"[interrupted] {key}", flush=True)
            return False
        try:
            if returncode:
                raise RuntimeError(f"child exited with return code {returncode}")
            reject_engine_overflow(log_path)
            if not source_provenance_valid(
                self.directory, self.manifest["source_provenance"]
            ):
                raise RuntimeError("execution sources changed while the child ran")
            finalize_cell(self.directory, cell, self.protocol)
        except Exception as error:
            with self.lock:
                cell.update(status="failed", error=str(error))
                self.save()
            print(f"[failed] {key}: {error}", flush=True)
            return False
        with self.lock:
            cell.update(
                status="complete",
                finished_at=attempt_record["finished_at"],
                wall_seconds=elapsed,
                error=None,
            )
            self.save()
        print(f"[done] {key} ({elapsed:.1f} s)", flush=True)
        return True


def audit_completed_cells(campaign):
    invalid = []
    for key, cell in campaign.manifest["cells"].items():
        if cell["status"] == "complete" and not validate_cell(
            campaign.directory, cell, campaign.protocol
        ):
            invalid.append(key)
    if invalid:
        with campaign.lock:
            for key in invalid:
                campaign.manifest["cells"][key].update(
                    status="pending",
                    error="completed artifact failed resume validation",
                )
            campaign.save()
    return invalid


def write_combined_summary(campaign):
    with campaign.lock:
        summaries = {"speed": [], "train": [], "eval": []}
        for cell in campaign.manifest["cells"].values():
            if cell["status"] != "complete":
                continue
            summary_path = campaign.directory / cell["summary"]
            if summary_path.is_file():
                summaries[cell["stage"]].append(
                    json.loads(summary_path.read_text(encoding="utf-8"))
                )
        for values in summaries.values():
            values.sort(key=lambda row: row["label"])
        status = {}
        for cell in campaign.manifest["cells"].values():
            stage_status = status.setdefault(cell["stage"], {})
            stage_status[cell["status"]] = stage_status.get(cell["status"], 0) + 1
        write_json(
            campaign.directory / "summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "updated_at": utc_now(),
                "protocol": campaign.protocol,
                "progress": status,
                "speed": summaries["speed"],
                "training": summaries["train"],
                "evaluation": summaries["eval"],
            },
        )


def mark_campaign_stopped(campaign, reason):
    campaign.request_stop()
    with campaign.lock:
        for cell in campaign.manifest["cells"].values():
            if cell["status"] == "running":
                cell.update(
                    status="pending",
                    interrupted_at=utc_now(),
                    error="interrupted; this controlled cell restarts from its seed",
                )
        campaign.manifest.update(
            status="incomplete",
            stopped_at=utc_now(),
            stop_reason=reason,
        )
        campaign.save()
    write_combined_summary(campaign)


def run_pool(campaign, keys, workers):
    results = []
    key_iterator = iter(keys)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {}
        for _ in range(workers):
            if campaign.stop_event.is_set():
                break
            try:
                key = next(key_iterator)
            except StopIteration:
                break
            futures[pool.submit(campaign.run_cell, key)] = key
        completed_since_summary = 0
        while futures:
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                futures.pop(future)
                try:
                    results.append(bool(future.result()))
                except Exception as error:
                    results.append(False)
                    print(f"[failed] worker: {error}", flush=True)
                completed_since_summary += 1
                if campaign.stop_event.is_set():
                    continue
                try:
                    key = next(key_iterator)
                except StopIteration:
                    continue
                futures[pool.submit(campaign.run_cell, key)] = key
            if completed_since_summary >= 10:
                write_combined_summary(campaign)
                completed_since_summary = 0
    except KeyboardInterrupt:
        campaign.request_stop()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return results


def run_stage(campaign, stage, cpu_workers):
    keys = [
        key
        for key, cell in campaign.manifest["cells"].items()
        if cell["stage"] == stage
    ]
    dependencies_ready = True
    if stage == "eval":
        training_valid = {}
        ready = []
        for key in keys:
            cell = campaign.manifest["cells"][key]
            training_key = f"train:{cell['training_label']}"
            training = campaign.manifest["cells"][training_key]
            if training_key not in training_valid:
                training_valid[training_key] = training[
                    "status"
                ] == "complete" and validate_cell(
                    campaign.directory, training, campaign.protocol
                )
                if not training_valid[training_key]:
                    campaign.update(
                        training_key,
                        status="failed",
                        error="training dependency is missing or invalid",
                    )
            if training_valid[training_key]:
                ready.append(key)
        dependencies_ready = len(ready) == len(keys)
        keys = ready
    if stage == "speed":
        results = run_pool(campaign, keys, workers=1)
        write_combined_summary(campaign)
        return all(results) and dependencies_ready
    cpu_keys = [
        key
        for key in keys
        if (
            campaign.manifest["cells"][key].get("backend") == "cpu"
            or campaign.manifest["cells"][key].get("eval_backend") == "cpu"
        )
    ]
    gpu_keys = [key for key in keys if key not in cpu_keys]
    results = []
    lanes = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        cpu_future = lanes.submit(run_pool, campaign, cpu_keys, cpu_workers)
        gpu_future = lanes.submit(run_pool, campaign, gpu_keys, 1)
        results.extend(cpu_future.result())
        results.extend(gpu_future.result())
    except KeyboardInterrupt:
        campaign.request_stop()
        lanes.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        lanes.shutdown(wait=True)
    write_combined_summary(campaign)
    return all(results) and dependencies_ready


def run_training_and_evaluation(campaign, cpu_workers):
    """Prioritize ready evaluations without sharing the CPU and GPU queues."""
    cells = campaign.manifest["cells"]

    def lane(key):
        cell = cells[key]
        backend = cell["backend"] if cell["stage"] == "train" else cell["eval_backend"]
        return "cpu" if backend == "cpu" else "gpu"

    training = {name: deque() for name in ("cpu", "gpu")}
    evaluations = {name: deque() for name in training}
    dependents = {}
    for key, cell in cells.items():
        if cell["stage"] == "train":
            training[lane(key)].append(key)
        elif cell["stage"] == "eval":
            dependents.setdefault(cell["training_label"], []).append(key)
    capacities = {"cpu": cpu_workers, "gpu": 1}
    pools = {
        name: concurrent.futures.ThreadPoolExecutor(max_workers=count)
        for name, count in capacities.items()
    }
    active = dict.fromkeys(pools, 0)
    futures = {}
    results = []
    try:
        while True:
            for name, pool in pools.items():
                while (
                    active[name] < capacities[name] and not campaign.stop_event.is_set()
                ):
                    queue = evaluations[name] or training[name]
                    if not queue:
                        break
                    key = queue.popleft()
                    futures[pool.submit(campaign.run_cell, key)] = (key, name)
                    active[name] += 1
            if not futures:
                break
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                key, name = futures.pop(future)
                active[name] -= 1
                try:
                    success = bool(future.result())
                except Exception as error:
                    success = False
                    print(f"[failed] worker {key}: {error}", flush=True)
                results.append(success)
                if cells[key]["stage"] == "train":
                    if success:
                        # run_cell validates even reused training artifacts before
                        # returning success; a status flag alone is insufficient.
                        for evaluation in dependents.pop(cells[key]["label"], []):
                            evaluations[lane(evaluation)].append(evaluation)
                    write_combined_summary(campaign)
                elif len(results) % 10 == 0:
                    write_combined_summary(campaign)
    except KeyboardInterrupt:
        campaign.request_stop()
        raise
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
    write_combined_summary(campaign)
    return (
        all(results)
        and not campaign.stop_event.is_set()
        and all(
            cell["status"] == "complete"
            for cell in cells.values()
            if cell["stage"] in ("train", "eval")
        )
    )


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="go2trot")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=[backend.label for backend in BACKENDS],
        default=[backend.label for backend in BACKENDS],
        help="training and final evaluation backends",
    )
    parser.add_argument(
        "--bundles",
        nargs="+",
        choices=list(BUNDLES),
        default=list(BUNDLES),
        help="training DR bundles",
    )
    parser.add_argument(
        "--eval-domains",
        nargs="+",
        choices=list(EVAL_DOMAINS),
        default=list(EVAL_DOMAINS),
        help="evaluation domains; in-range values follow the saved training config",
    )
    parser.add_argument(
        "--exclude-training",
        nargs="+",
        default=[],
        choices=[
            f"{backend.label}:{bundle}" for backend in BACKENDS for bundle in BUNDLES
        ],
        help="omit backend:bundle training cells and their dependent evaluations",
    )
    parser.add_argument(
        "--skip-speed",
        action="store_true",
        help="omit runtime benchmark cells from the campaign plan",
    )
    parser.add_argument(
        "--evaluate-after-training",
        action="store_true",
        help="prioritize evaluations as each training cell completes",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["speed", "train", "eval", "summarize"],
        default=["speed", "train", "eval", "summarize"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--speed-common-num-envs", type=int, default=256)
    parser.add_argument("--speed-production-num-envs", type=int, default=4096)
    parser.add_argument("--speed-target-env-steps", type=int, default=32768)
    parser.add_argument("--speed-min-steps", type=int, default=50)
    parser.add_argument("--speed-repeats", type=int, default=5)
    parser.add_argument("--train-num-envs", type=int, default=4096)
    parser.add_argument("--train-iterations", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument(
        "--checkpoints", type=int, nargs="+", default=[100, 250, 500, 750, 1000]
    )
    parser.add_argument("--eval-num-envs", type=int, default=200)
    parser.add_argument("--eval-duration", type=float, default=5.0)
    parser.add_argument("--eval-settling-time", type=float, default=0.5)
    parser.add_argument("--eval-contact-threshold", type=float, default=5.0)
    parser.add_argument("--eval-seed", type=int, default=1701)
    parser.add_argument("--cpu-workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = LOG_ROOT / f"dr_full_{stamp}"
    else:
        args.output = args.output.resolve()
    try:
        args.output.relative_to(LOG_ROOT)
    except ValueError:
        parser.error(f"--output must be under {LOG_ROOT}")
    positive = (
        args.speed_common_num_envs,
        args.speed_production_num_envs,
        args.speed_target_env_steps,
        args.speed_min_steps,
        args.speed_repeats,
        args.train_num_envs,
        args.train_iterations,
        args.save_interval,
        args.eval_num_envs,
        args.eval_duration,
        args.eval_contact_threshold,
        args.cpu_workers,
    )
    if any(value <= 0 for value in positive):
        parser.error("counts, intervals, durations, and worker counts must be positive")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be unique")
    if args.eval_num_envs % 10:
        parser.error("eval-num-envs must be divisible by the 10 Go2 command cases")
    for name in ("backends", "bundles", "eval_domains"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"{name.replace('_', '-')} must be unique")
    missing_intermediate = {"nominal", "combined_in"} - set(args.eval_domains)
    if missing_intermediate:
        parser.error(
            "eval domains must include intermediate domains "
            f"{sorted(missing_intermediate)}"
        )
    pairs = {
        f"{backend}:{bundle}" for backend in args.backends for bundle in args.bundles
    }
    exclusions = set(args.exclude_training)
    if len(exclusions) != len(args.exclude_training) or not exclusions <= pairs:
        parser.error("training exclusions must be unique selected backend:bundle pairs")
    if exclusions == pairs:
        parser.error("training exclusions remove every training cell")
    return args


def main(argv=None):
    args = get_args(argv)
    protocol = make_protocol(args)
    cases = campaign_cases(protocol)
    counts = {
        stage: sum(case["stage"] == stage for case in cases)
        for stage in ("speed", "train", "eval")
    }
    print(json.dumps({"output": str(args.output), "cells": counts}, indent=2))
    if args.dry_run:
        return
    manifest = load_or_create_manifest(args.output, protocol, args.resume)
    campaign = Campaign(args.output, manifest)
    if args.resume:
        with campaign.lock:
            campaign.manifest.pop("stopped_at", None)
            campaign.manifest.pop("stop_reason", None)
            campaign.manifest["status"] = "running"
            campaign.save()
        invalid = audit_completed_cells(campaign)
        if invalid:
            print(f"[audit] {len(invalid)} completed artifacts need repair", flush=True)
    write_combined_summary(campaign)
    success = True
    try:
        interleaved = args.evaluate_after_training and {"train", "eval"} <= set(
            args.stages
        )
        for stage in ("speed", "train", "eval"):
            if stage not in args.stages or (interleaved and stage == "eval"):
                continue
            if interleaved and stage == "train":
                success = (
                    run_training_and_evaluation(campaign, args.cpu_workers) and success
                )
            else:
                success = run_stage(campaign, stage, args.cpu_workers) and success
    except KeyboardInterrupt:
        mark_campaign_stopped(campaign, "interrupted by user")
        print("[stopped] campaign is incomplete and safe to resume", flush=True)
        raise SystemExit(130) from None
    if "summarize" in args.stages:
        write_combined_summary(campaign)
    statuses = {cell["status"] for cell in campaign.manifest["cells"].values()}
    if statuses == {"complete"}:
        campaign.manifest["status"] = "complete"
    elif "failed" in statuses or not success:
        campaign.manifest["status"] = "incomplete"
    else:
        campaign.manifest["status"] = "running"
    campaign.save()
    if not success:
        raise SystemExit("one or more campaign cells failed; rerun with --resume")


if __name__ == "__main__":
    main()
