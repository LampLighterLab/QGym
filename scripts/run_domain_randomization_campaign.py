"""Run the contact-friction DR speed, training, and evaluation campaign.

The parent process launches every simulator cell in a fresh child process.
Invoke it with the VSim environment so CUDA/VSim children inherit activation:

    uv run --env-file .env.vsim scripts/run_domain_randomization_campaign.py

Results are incremental and resumable under ``logs/dr_contact_friction_*``.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gym.envs.base.domain_randomization import (
    get_domain_randomization_range,
    set_domain_randomization_range,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO_ROOT / "logs"
FRICTION_RANGE = [0.5, 1.0]
EVAL_DOMAINS = {
    "nominal": [1.0, 1.0],
    "in_range": [0.5, 1.0],
    "low_friction": [0.35, 0.35],
}
FIXED_PROVENANCE_FILES = (
    "pyproject.toml",
    "uv.lock",
    "scripts/run_domain_randomization_campaign.py",
    "scripts/benchmark_domain_randomization.py",
    "scripts/eval_policy.py",
    "scripts/train.py",
    "scripts/train_domain_randomization.py",
    "resources/robots/go2/urdf/go2.urdf",
)
PROVENANCE_SOURCE_ROOTS = ("gym", "learning")


@dataclass(frozen=True)
class Backend:
    label: str
    physics_backend: str
    device: str


@dataclass(frozen=True)
class SpeedCase:
    backend: Backend
    num_envs: int
    dr_mode: str

    stage = "speed"

    @property
    def label(self):
        return f"{self.backend.label}_{self.num_envs}_{self.dr_mode}"


@dataclass(frozen=True)
class TrainingCase:
    backend: Backend
    seed: int
    dr_mode: str

    stage = "train"

    @property
    def label(self):
        return f"{self.backend.label}_{self.dr_mode}_seed{self.seed}"


@dataclass(frozen=True)
class EvaluationCase:
    training: TrainingCase
    backend: Backend
    domain: str
    friction_range: tuple[float, float]

    stage = "eval"

    @property
    def label(self):
        return f"{self.training.label}__{self.backend.label}__{self.domain}"


@dataclass(frozen=True)
class CellSpec:
    case: SpeedCase | TrainingCase | EvaluationCase
    command: tuple[str, ...]
    artifact: Path
    stdout_log: Path


@dataclass
class CampaignState:
    directory: Path
    manifest_path: Path
    manifest: dict
    resume: bool

    @property
    def protocol(self):
        return self.manifest["protocol"]


BACKENDS = (
    Backend("warp", "mujoco", "cuda:0"),
    Backend("vsim", "vsim", "cuda:0"),
    Backend("cpu", "mujoco", "cpu"),
)


def protocol_backends(protocol):
    return tuple(Backend(**backend) for backend in protocol["backends"])


def speed_cases(protocol):
    production = set(
        protocol.get(
            "speed_production_backends",
            [
                backend.label
                for backend in protocol_backends(protocol)
                if backend.device.startswith("cuda")
            ],
        )
    )
    cases = []
    for backend in protocol_backends(protocol):
        num_envs_values = [protocol["speed_common_num_envs"]]
        if backend.label in production:
            num_envs_values.append(protocol["speed_production_num_envs"])
        for num_envs in dict.fromkeys(num_envs_values):
            for dr_mode in ("off", "on"):
                cases.append(SpeedCase(backend, num_envs, dr_mode))
    return tuple(cases)


def training_cases(protocol):
    return tuple(
        TrainingCase(backend, seed, dr_mode)
        for seed in protocol["seeds"]
        for backend in protocol_backends(protocol)
        for dr_mode in ("off", "on")
    )


def evaluation_backends(protocol, training_backend):
    backends = protocol_backends(protocol)
    if protocol["cross_backend_eval"]:
        return backends
    return tuple(backend for backend in backends if backend.label == training_backend)


def evaluation_cases(protocol):
    return tuple(
        EvaluationCase(training, backend, domain, tuple(friction_range))
        for training in training_cases(protocol)
        for backend in evaluation_backends(protocol, training.backend.label)
        for domain, friction_range in protocol["eval_domains"].items()
    )


def campaign_cases(protocol):
    return (
        *speed_cases(protocol),
        *training_cases(protocol),
        *evaluation_cases(protocol),
    )


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_paths():
    paths = [Path(path) for path in FIXED_PROVENANCE_FILES]
    for root_name in PROVENANCE_SOURCE_ROOTS:
        root = REPO_ROOT / root_name
        paths.extend(
            path.relative_to(REPO_ROOT)
            for path in root.rglob("*.py")
            if not path.name.startswith("test_") and "tests" not in path.parts
        )
    return tuple(sorted(set(paths)))


def source_provenance(campaign_dir, *, copy_sources):
    """Hash the execution sources and optionally preserve byte-for-byte copies."""
    snapshot_root = campaign_dir / "provenance"
    files = {}
    for relative in provenance_paths():
        source = REPO_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"campaign provenance source is missing: {source}")
        snapshot = snapshot_root / relative
        digest = sha256_file(source)
        if copy_sources:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, snapshot)
            if sha256_file(snapshot) != digest:
                raise RuntimeError(f"source changed while snapshotting: {source}")
        files[relative.as_posix()] = {
            "sha256": digest,
            "bytes": (snapshot if copy_sources else source).stat().st_size,
            "snapshot": str(Path("provenance") / relative),
        }
    return {"snapshot_root": "provenance", "files": files}


def requested_protocol(args):
    """Resolve every scientific campaign setting before the first cell runs."""
    import gym.envs  # noqa: F401 — registers tasks
    from gym.utils.helpers import class_to_dict
    from gym.utils.task_registry import task_registry

    registered_env_cfg, registered_train_cfg = task_registry.get_cfgs(args.task)
    env_cfg = copy.deepcopy(registered_env_cfg)
    train_cfg = copy.deepcopy(registered_train_cfg)
    env_cfg.env.num_envs = args.train_num_envs
    train_cfg.runner.max_iterations = args.train_iterations
    set_domain_randomization_range(env_cfg, "p_gains", None)
    set_domain_randomization_range(env_cfg, "d_gains", None)
    set_domain_randomization_range(env_cfg, "link_mass_scale_range", None)
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)

    configured_range = get_domain_randomization_range(env_cfg, "contact_friction_range")
    if configured_range is None or list(configured_range) != FRICTION_RANGE:
        raise ValueError(
            f"task {args.task!r} must configure contact_friction_range="
            f"{FRICTION_RANGE}, got {configured_range!r}"
        )
    rollout_size = int(train_cfg.algorithm.rollout_size)
    if rollout_size % args.train_num_envs:
        raise ValueError(
            "training rollout_size must be divisible by train_num_envs: "
            f"{rollout_size} % {args.train_num_envs} != 0"
        )

    return {
        "task": args.task,
        "backends": [asdict(backend) for backend in BACKENDS],
        "seeds": list(args.seeds),
        "speed_common_num_envs": args.speed_common_num_envs,
        "speed_production_num_envs": args.speed_production_num_envs,
        "speed_production_backends": ["warp", "vsim"],
        "speed_target_env_steps": args.speed_target_env_steps,
        "speed_repeats": args.speed_repeats,
        "speed_min_control_steps": args.speed_min_steps,
        "speed_warmup_steps": "max(25, min(50, ceil(8192 / num_envs)))",
        "speed_reset_profiles": ["none", "timeout", "all"],
        "train_num_envs": args.train_num_envs,
        "train_iterations": args.train_iterations,
        "rollout_size": rollout_size,
        "rollout_steps_per_env": rollout_size // args.train_num_envs,
        "optimizer_batch_size": int(train_cfg.algorithm.batch_size),
        "max_gradient_steps": int(train_cfg.algorithm.max_gradient_steps),
        "control_frequency_hz": float(env_cfg.control.ctrl_frequency),
        "control_dt_s": float(env_cfg.control.ctrl_dt),
        "sim_dt_s": float(env_cfg.sim_dt),
        "decimation": int(env_cfg.control.decimation),
        "eval_num_envs": args.eval_num_envs,
        "eval_duration_s": args.eval_duration,
        "eval_settling_time_s": args.eval_settling_time,
        "eval_seed": args.eval_seed,
        "eval_reset_mode": "reset_to_basic",
        "eval_command_profile": "go2",
        "eval_contact_threshold_n": 20.0,
        "eval_velocity_impulse_m_per_s": 0.0,
        "eval_record_dof": False,
        "eval_record_tracking": False,
        "eval_record_policy_io": False,
        "eval_domains": EVAL_DOMAINS,
        "contact_friction_dr_range": FRICTION_RANGE,
        "cross_backend_eval": args.cross_backend_eval,
        "environment_config_template": class_to_dict(env_cfg),
        "training_config_template": class_to_dict(train_cfg),
        "per_training_cell_overrides": [
            "environment.seed",
            "training.seed",
            "training.runner.device",
            "environment.domain_randomization.startup.contact_friction_range",
        ],
        "active_task_note": (
            "This worktree's Go2Trot uses the active single phase oscillator; "
            "the four-oscillator design in oscillator.md is not active code."
        ),
        "interpretation": (
            "Uniform [0.5, 1.0] changes both friction variance and mean "
            "friction relative to the 1.0 baseline. One seed is a screen; "
            "at least three seeds are required for an advancement decision."
        ),
        "predeclared_screen": {
            "speed": "report paired ON/OFF ratios; no hard pass threshold",
            "nominal_policy": "flag a nominal survival loss greater than 0.05 absolute",
            "robustness": (
                "look for improved in-range and low-friction survival and "
                "tracking without unexplained nominal collapse"
            ),
        },
    }


def changed_top_level_values(persisted, requested):
    def compact(value):
        encoded = json.dumps(json_safe(value), sort_keys=True).encode()
        if len(encoded) <= 500:
            return value
        return {
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "json_bytes": len(encoded),
        }

    keys = sorted(set(persisted) | set(requested))
    return {
        key: {
            "persisted": compact(persisted.get(key)),
            "requested": compact(requested.get(key)),
        }
        for key in keys
        if persisted.get(key) != requested.get(key)
    }


def load_manifest(path, args, protocol, provenance):
    if path.exists():
        if not args.resume:
            raise FileExistsError(
                f"campaign already exists at {path.parent}; pass --resume"
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        evidence_stages = {"speed", "train", "eval"} & set(args.stages)
        if manifest.get("schema_version") != 2:
            raise ValueError("unsupported campaign schema; expected schema_version 2")

        protocol_changes = changed_top_level_values(manifest["protocol"], protocol)
        provenance_changes = changed_top_level_values(
            manifest["source_provenance"]["files"], provenance["files"]
        )
        if evidence_stages and (protocol_changes or provenance_changes):
            details = {
                "protocol": protocol_changes,
                "source_provenance": provenance_changes,
            }
            raise ValueError(
                "cannot resume evidence-producing stages with a changed protocol "
                "or source tree:\n" + json.dumps(details, indent=2)
            )
        return manifest
    return {
        "schema_version": 2,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "git": git_metadata(),
        "protocol": protocol,
        "source_provenance": provenance,
        "cells": [],
    }


def validate_speed_artifact(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    schema_version = result.get("schema_version")
    if schema_version != 2:
        return False, "unsupported or missing speed schema_version"
    if not isinstance(result.get("protocol"), dict):
        return False, "speed protocol is missing"
    setup_seconds = result.get("setup_seconds")
    if not isinstance(setup_seconds, (int, float)) or not math.isfinite(setup_seconds):
        return False, "speed setup_seconds is not finite"
    profiles = result.get("profiles")
    if not isinstance(profiles, list):
        return False, "speed profiles are missing"
    memory = result.get("memory", {})
    required_memory = {
        "cuda_after_profiles",
        "cuda_used_delta_bytes",
        "cuda_used_after_profiles_delta_bytes",
        "host_max_rss_after_profiles_kb",
        "host_max_rss_after_profiles_delta_kb",
    }
    missing_memory = sorted(required_memory - set(memory))
    if missing_memory:
        return False, f"speed post-profile memory is missing {missing_memory}"
    if {profile.get("profile") for profile in profiles} != {"none", "timeout", "all"}:
        return False, "speed profiles must contain none, timeout, and all"
    for profile in profiles:
        repeats = profile.get("repeats")
        elapsed = profile.get("elapsed_seconds")
        throughput = profile.get("env_control_steps_per_s")
        median = profile.get("median_env_control_steps_per_s")
        if not isinstance(repeats, int) or repeats <= 0:
            return (
                False,
                f"invalid repeats for speed profile {profile.get('profile')!r}",
            )
        if not isinstance(elapsed, list) or len(elapsed) != repeats:
            return False, f"incomplete elapsed timings for {profile['profile']!r}"
        if not isinstance(throughput, list) or len(throughput) != repeats:
            return False, f"incomplete throughput timings for {profile['profile']!r}"
        values = np.asarray(elapsed + throughput + [median], dtype=np.float64)
        if not np.isfinite(values).all() or (values <= 0.0).any():
            return (
                False,
                f"non-positive or non-finite timings for {profile['profile']!r}",
            )
    return True, None


def checkpoint_iteration(path):
    prefix = "model_"
    if not path.name.startswith(prefix) or path.suffix != ".pt":
        raise ValueError(f"checkpoint does not follow model_<iteration>.pt: {path}")
    return int(path.stem.removeprefix(prefix))


def validate_training_artifact(path):
    marker = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "label",
        "backend",
        "dr_mode",
        "seed",
        "run_dir",
        "checkpoint",
        "resolved_config",
    }
    missing = sorted(required - set(marker))
    if missing:
        return False, f"training marker is missing {missing}"
    run_dir = Path(marker["run_dir"])
    checkpoint = Path(marker["checkpoint"])
    resolved_config = Path(marker["resolved_config"])
    if not run_dir.is_dir():
        return False, f"training run directory is missing: {run_dir}"
    if checkpoint.parent != run_dir or not checkpoint.is_file():
        return False, f"training checkpoint is missing or outside run_dir: {checkpoint}"
    if checkpoint.stat().st_size == 0:
        return False, f"training checkpoint is empty: {checkpoint}"
    iteration = checkpoint_iteration(checkpoint)
    if not resolved_config.is_file():
        return False, f"resolved training config is missing: {resolved_config}"
    resolved = json.loads(resolved_config.read_text(encoding="utf-8"))
    try:
        configured_iterations = int(resolved["training"]["runner"]["max_iterations"])
    except (KeyError, TypeError, ValueError):
        return False, "resolved config has no valid training.runner.max_iterations"
    if configured_iterations != iteration:
        return False, (
            "checkpoint iteration does not match resolved max_iterations: "
            f"{iteration} != {configured_iterations}"
        )
    vitals_path = run_dir / "vitals.jsonl"
    if not vitals_path.is_file():
        return False, f"training vitals are missing: {vitals_path}"
    records = [
        line for line in vitals_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if not records:
        return False, "training vitals contain no records"
    final_record = json.loads(records[-1])
    if final_record.get("iteration") != iteration:
        return False, (
            "final vitals iteration does not match checkpoint: "
            f"{final_record.get('iteration')} != {iteration}"
        )
    return True, None


def validate_evaluation_artifact(path):
    required = {
        "task",
        "checkpoint_iteration",
        "num_envs",
        "contact_friction",
        "survived",
        "ep_len",
        "mean_reward",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            return False, f"evaluation artifact is missing {missing}"
        num_envs = int(data["num_envs"])
        if num_envs <= 0:
            return False, "evaluation num_envs must be positive"
        if int(data["checkpoint_iteration"]) < 0:
            return False, "evaluation checkpoint_iteration must be non-negative"
        if not str(data["task"]):
            return False, "evaluation task must be non-empty"
        for name in ("contact_friction", "survived", "ep_len", "mean_reward"):
            values = np.asarray(data[name])
            if values.shape != (num_envs,):
                return False, (
                    f"evaluation {name} has shape {values.shape}, expected "
                    f"({num_envs},)"
                )
            if name != "survived" and not np.isfinite(values).all():
                return False, f"evaluation {name} contains non-finite values"
    return True, None


def validate_artifact(stage, path):
    """Return whether a stage artifact is complete enough to reuse as evidence."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False, f"artifact is missing or empty: {path}"
    validators = {
        "speed": validate_speed_artifact,
        "train": validate_training_artifact,
        "eval": validate_evaluation_artifact,
    }
    if stage not in validators:
        raise ValueError(f"unknown campaign stage {stage!r}")
    try:
        return validators[stage](path)
    except (
        EOFError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        return False, f"could not parse {stage} artifact {path}: {error}"


def validate_speed_identity(case, protocol, path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    artifact_protocol = result["protocol"]
    expected_steps = max(
        protocol["speed_min_control_steps"],
        math.ceil(protocol["speed_target_env_steps"] / case.num_envs),
    )
    expected_warmup = max(25, min(50, math.ceil(8192 / case.num_envs)))
    expected = {
        "task": protocol["task"],
        "backend_label": case.backend.label,
        "physics_backend": case.backend.physics_backend,
        "device": case.backend.device,
        "num_envs": case.num_envs,
        "seed": protocol["seeds"][0],
        "target_env_steps": protocol["speed_target_env_steps"],
        "min_steps": protocol["speed_min_control_steps"],
        "steps_per_trial": expected_steps,
        "repeats": protocol["speed_repeats"],
        "warmup_steps": expected_warmup,
        "dr_enabled": case.dr_mode == "on",
    }
    mismatches = {
        key: (artifact_protocol.get(key), value)
        for key, value in expected.items()
        if artifact_protocol.get(key) != value
    }
    if mismatches:
        return False, f"speed artifact does not match cell: {mismatches}"
    expected_range = (
        protocol["contact_friction_dr_range"] if case.dr_mode == "on" else None
    )
    artifact_range = (
        artifact_protocol.get("domain_randomization", {})
        .get("startup", {})
        .get("contact_friction_range")
    )
    if artifact_range != expected_range:
        return False, "speed artifact has the wrong contact-friction range"
    for profile in result["profiles"]:
        if profile["repeats"] != protocol["speed_repeats"]:
            return False, "speed artifact has the wrong repeat count"
        if profile["steps_per_trial"] != expected_steps:
            return False, "speed artifact has the wrong timed step count"
        if profile["warmup_steps"] != expected_warmup:
            return False, "speed artifact has the wrong warmup step count"
    return True, None


def validate_training_identity(case, protocol, path):
    marker = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_marker = {
        "label": case.label,
        "backend": asdict(case.backend),
        "dr_mode": case.dr_mode,
        "seed": case.seed,
    }
    mismatches = {
        key: (marker.get(key), value)
        for key, value in expected_marker.items()
        if marker.get(key) != value
    }
    if mismatches:
        return False, f"training marker does not match cell: {mismatches}"

    resolved = json.loads(Path(marker["resolved_config"]).read_text(encoding="utf-8"))
    expected_range = (
        protocol["contact_friction_dr_range"] if case.dr_mode == "on" else None
    )
    expected_resolved = {
        "task": (resolved.get("task"), protocol["task"]),
        "backend": (resolved.get("backend"), asdict(case.backend)),
        "dr_mode": (resolved.get("dr_mode"), case.dr_mode),
        "environment.seed": (resolved["environment"].get("seed"), case.seed),
        "environment.env.num_envs": (
            resolved["environment"]["env"].get("num_envs"),
            protocol["train_num_envs"],
        ),
        "environment.domain_randomization.startup.contact_friction_range": (
            resolved["environment"]["domain_randomization"]["startup"].get(
                "contact_friction_range"
            ),
            expected_range,
        ),
        "training.seed": (resolved["training"].get("seed"), case.seed),
        "training.runner.device": (
            resolved["training"]["runner"].get("device"),
            case.backend.device,
        ),
        "training.runner.max_iterations": (
            resolved["training"]["runner"].get("max_iterations"),
            protocol["train_iterations"],
        ),
    }
    mismatches = {
        key: values
        for key, values in expected_resolved.items()
        if values[0] != values[1]
    }
    if mismatches:
        return False, f"resolved training config does not match cell: {mismatches}"
    return True, None


def validate_evaluation_identity(case, protocol, path):
    with np.load(path, allow_pickle=False) as data:
        expected_scalars = {
            "task": protocol["task"],
            "train_label": case.training.label,
            "eval_label": f"{case.backend.label}-{case.domain}",
            "checkpoint_iteration": protocol["train_iterations"],
            "reset_mode": protocol["eval_reset_mode"],
            "command_profile": protocol["eval_command_profile"],
            "num_envs": protocol["eval_num_envs"],
            "seed": protocol["eval_seed"],
        }
        mismatches = {}
        for key, expected in expected_scalars.items():
            actual = np.asarray(data[key]).item()
            if actual != expected:
                mismatches[key] = (actual, expected)
        for key, expected in (
            ("duration_s", protocol["eval_duration_s"]),
            ("settling_time_s", protocol["eval_settling_time_s"]),
            ("contact_threshold_n", protocol["eval_contact_threshold_n"]),
        ):
            actual = float(data[key])
            if not math.isclose(actual, expected):
                mismatches[key] = (actual, expected)
        if mismatches:
            return False, f"evaluation artifact does not match cell: {mismatches}"

        expected_range = np.asarray(case.friction_range, dtype=np.float32)
        if not np.allclose(data["contact_friction_grid"], expected_range):
            return False, "evaluation artifact has the wrong friction grid"
        friction = np.asarray(data["contact_friction"])
        if friction.min() < expected_range[0] or friction.max() > expected_range[1]:
            return False, "evaluation friction escaped the requested grid"
    return True, None


def validate_cell_artifact(case, protocol, path):
    valid, reason = validate_artifact(case.stage, path)
    if not valid:
        return valid, reason
    validators = {
        "speed": validate_speed_identity,
        "train": validate_training_identity,
        "eval": validate_evaluation_identity,
    }
    try:
        return validators[case.stage](case, protocol, path)
    except (
        EOFError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        return False, f"artifact identity validation failed for {case.label}: {error}"


def cell_key(stage, label):
    return f"{stage}:{label}"


def completed_cells(manifest):
    return {
        cell_key(cell["stage"], cell["label"]): cell
        for cell in manifest["cells"]
        if cell["status"] == "complete"
    }


def record_cell(state, cell):
    key = cell_key(cell["stage"], cell["label"])
    state.manifest["cells"] = [
        existing
        for existing in state.manifest["cells"]
        if cell_key(existing["stage"], existing["label"]) != key
    ]
    state.manifest["cells"].append(cell)
    state.manifest["updated_at"] = utc_now()
    write_json(state.manifest_path, state.manifest)


def command_text(command):
    return " ".join(str(part) for part in command)


def attempt_summary(cell):
    keys = (
        "status",
        "started_at",
        "finished_at",
        "wall_seconds",
        "returncode",
        "artifact_error",
        "command",
        "artifact",
        "stdout_log",
    )
    return {key: cell[key] for key in keys if key in cell}


def assert_source_provenance_unchanged(state):
    persisted = state.manifest["source_provenance"]
    current = source_provenance(state.directory, copy_sources=False)
    changes = changed_top_level_values(persisted["files"], current["files"])
    if changes:
        raise RuntimeError(
            "campaign source changed after its immutable snapshot:\n"
            + json.dumps(changes, indent=2)
        )
    for metadata in persisted["files"].values():
        snapshot = state.directory / metadata["snapshot"]
        if not snapshot.is_file() or sha256_file(snapshot) != metadata["sha256"]:
            raise RuntimeError(
                f"campaign source snapshot is missing or changed: {snapshot}"
            )


def run_cell(state, spec, *, prepare=None, finalize=None):
    case = spec.case
    key = cell_key(case.stage, case.label)
    previous = next(
        (
            cell
            for cell in state.manifest["cells"]
            if cell_key(cell["stage"], cell["label"]) == key
        ),
        None,
    )
    if state.resume and previous is not None and previous["status"] == "complete":
        valid, reason = validate_cell_artifact(
            case, state.protocol, previous["artifact"]
        )
        if valid:
            print(f"[skip] {case.stage} {case.label}", flush=True)
            return True
        print(
            f"[retry] {case.stage} {case.label}; invalid prior artifact: {reason}",
            flush=True,
        )

    assert_source_provenance_unchanged(state)

    cell = {
        "stage": case.stage,
        "label": case.label,
        "status": "running",
        "started_at": utc_now(),
        "attempt": 1 if previous is None else previous.get("attempt", 1) + 1,
        "command": list(spec.command),
        "artifact": str(spec.artifact),
        "stdout_log": str(spec.stdout_log),
    }
    if previous is not None:
        cell["history"] = [
            *previous.get("history", []),
            attempt_summary(previous),
        ]
    record_cell(state, cell)
    spec.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {case.stage} {case.label}", flush=True)
    print(f"      {command_text(spec.command)}", flush=True)

    prepared = None
    try:
        if prepare is not None:
            prepared = prepare()
    except (OSError, TypeError, ValueError) as error:
        cell["finished_at"] = utc_now()
        cell["status"] = "failed"
        cell["artifact_error"] = f"could not prepare cell: {error}"
        record_cell(state, cell)
        print(f"[failed] {case.stage} {case.label}; {error}", flush=True)
        return False

    start = time.perf_counter()
    try:
        with spec.stdout_log.open("w", encoding="utf-8") as output:
            completed = subprocess.run(
                spec.command,
                cwd=REPO_ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
            )
    except OSError as error:
        cell["finished_at"] = utc_now()
        cell["wall_seconds"] = time.perf_counter() - start
        cell["status"] = "failed"
        cell["artifact_error"] = f"could not launch child: {error}"
        record_cell(state, cell)
        print(f"[failed] {case.stage} {case.label}; {error}", flush=True)
        return False

    cell["finished_at"] = utc_now()
    cell["wall_seconds"] = time.perf_counter() - start
    cell["returncode"] = completed.returncode
    if completed.returncode:
        cell["status"] = "failed"
        record_cell(state, cell)
        tail = spec.stdout_log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-30:]
        print(
            f"[failed] {case.stage} {case.label}; tail of {spec.stdout_log}:",
            flush=True,
        )
        print("\n".join(tail), flush=True)
        return False

    try:
        if finalize is not None:
            cell.update(finalize(prepared) or {})
        valid, reason = validate_cell_artifact(case, state.protocol, spec.artifact)
    except (OSError, TypeError, ValueError) as error:
        valid, reason = False, f"could not finalize cell: {error}"
    if not valid:
        cell["status"] = "failed"
        cell["artifact_error"] = reason
        record_cell(state, cell)
        print(f"[failed] {case.stage} {case.label}; {reason}", flush=True)
        return False

    cell["status"] = "complete"
    record_cell(state, cell)
    print(
        f"[done] {case.stage} {case.label} ({cell['wall_seconds']:.1f} s)",
        flush=True,
    )
    return True


def write_resolved_training_config(path, protocol, case):
    import gym.envs  # noqa: F401 — registers tasks
    from gym.utils.helpers import class_to_dict
    from gym.utils.task_registry import task_registry

    registered_env_cfg, registered_train_cfg = task_registry.get_cfgs(protocol["task"])
    env_cfg = copy.deepcopy(registered_env_cfg)
    train_cfg = copy.deepcopy(registered_train_cfg)
    env_cfg.env.num_envs = protocol["train_num_envs"]
    env_cfg.seed = case.seed
    train_cfg.seed = case.seed
    set_domain_randomization_range(
        env_cfg,
        "contact_friction_range",
        list(protocol["contact_friction_dr_range"]) if case.dr_mode == "on" else None,
    )
    set_domain_randomization_range(env_cfg, "p_gains", None)
    set_domain_randomization_range(env_cfg, "d_gains", None)
    set_domain_randomization_range(env_cfg, "link_mass_scale_range", None)
    train_cfg.runner.max_iterations = protocol["train_iterations"]
    train_cfg.runner.device = case.backend.device
    task_registry.convert_frequencies_to_params(env_cfg, train_cfg)
    write_json(
        path,
        {
            "task": protocol["task"],
            "backend": asdict(case.backend),
            "dr_mode": case.dr_mode,
            "environment": class_to_dict(env_cfg),
            "training": class_to_dict(train_cfg),
        },
    )


def training_checkpoint_signatures(experiment_dir, iterations):
    if not experiment_dir.is_dir():
        return {}
    signatures = {}
    for run_dir in experiment_dir.iterdir():
        checkpoint = run_dir / f"model_{iterations}.pt"
        if run_dir.is_dir() and checkpoint.is_file():
            stat = checkpoint.stat()
            signatures[checkpoint] = (stat.st_mtime_ns, stat.st_size)
    return signatures


def find_training_run(experiment_dir, iterations, previous_checkpoints=None):
    previous_checkpoints = previous_checkpoints or {}
    current = training_checkpoint_signatures(experiment_dir, iterations)
    candidates = sorted(
        (
            checkpoint.parent
            for checkpoint, signature in current.items()
            if previous_checkpoints.get(checkpoint) != signature
        ),
        key=lambda path: (path / f"model_{iterations}.pt").stat().st_mtime_ns,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no new model_{iterations}.pt checkpoint under {experiment_dir}"
        )
    return candidates[-1]


def run_speed_stage(state, fail_fast):
    protocol = state.protocol
    success = True
    for case in speed_cases(protocol):
        artifact = state.directory / "speed" / f"{case.label}.json"
        spec = CellSpec(
            case,
            (
                sys.executable,
                str(REPO_ROOT / "scripts" / "benchmark_domain_randomization.py"),
                "--task",
                protocol["task"],
                "--backend",
                case.backend.label,
                "--dr",
                case.dr_mode,
                "--num_envs",
                str(case.num_envs),
                "--seed",
                str(protocol["seeds"][0]),
                "--target_env_steps",
                str(protocol["speed_target_env_steps"]),
                "--repeats",
                str(protocol["speed_repeats"]),
                "--min_steps",
                str(protocol["speed_min_control_steps"]),
                "--out",
                str(artifact),
            ),
            artifact,
            state.directory / "stdout" / f"speed_{case.label}.log",
        )
        okay = run_cell(state, spec)
        success = okay and success
        if not okay and fail_fast:
            return False
    return success


def run_training_stage(state, fail_fast):
    protocol = state.protocol
    success = True
    relative_campaign = state.directory.relative_to(LOG_ROOT)
    for case in training_cases(protocol):
        experiment = relative_campaign / "training" / case.label
        experiment_dir = LOG_ROOT / experiment
        completion_marker = state.directory / "training" / f"{case.label}.json"
        resolved_config = (
            state.directory / "training" / f"{case.label}.resolved_config.json"
        )
        spec = CellSpec(
            case,
            (
                sys.executable,
                "-X",
                "faulthandler",
                "-m",
                "scripts.train_domain_randomization",
                "--task",
                protocol["task"],
                "--backend",
                case.backend.physics_backend,
                "--device",
                case.backend.device,
                "--num_envs",
                str(protocol["train_num_envs"]),
                "--max_iterations",
                str(protocol["train_iterations"]),
                "--seed",
                str(case.seed),
                "--headless",
                "--disable_wandb",
                "--experiment_name",
                str(experiment),
                "--dr-bundle",
                "friction" if case.dr_mode == "on" else "off",
            ),
            completion_marker,
            state.directory / "stdout" / f"train_{case.label}.log",
        )

        def prepare_training():
            write_resolved_training_config(resolved_config, protocol, case)
            return training_checkpoint_signatures(
                experiment_dir, protocol["train_iterations"]
            )

        def finalize_training(previous_checkpoints):
            run_dir = find_training_run(
                experiment_dir,
                protocol["train_iterations"],
                previous_checkpoints,
            )
            checkpoint = run_dir / f"model_{protocol['train_iterations']}.pt"
            write_json(
                completion_marker,
                {
                    "label": case.label,
                    "backend": asdict(case.backend),
                    "dr_mode": case.dr_mode,
                    "seed": case.seed,
                    "run_dir": str(run_dir),
                    "checkpoint": str(checkpoint),
                    "resolved_config": str(resolved_config),
                },
            )
            return {"run_dir": str(run_dir), "checkpoint": str(checkpoint)}

        okay = run_cell(
            state,
            spec,
            prepare=prepare_training,
            finalize=finalize_training,
        )
        success = okay and success
        if not okay and fail_fast:
            return False
    return success


def training_artifacts(manifest):
    protocol = manifest["protocol"]
    complete = completed_cells(manifest)
    artifacts = []
    for case in training_cases(protocol):
        cell = complete.get(cell_key(case.stage, case.label))
        if cell is None:
            continue
        artifact = Path(cell["artifact"])
        valid, _ = validate_cell_artifact(case, protocol, artifact)
        if valid:
            artifacts.append(json.loads(artifact.read_text(encoding="utf-8")))
    return artifacts


def run_evaluation_stage(state, fail_fast):
    protocol = state.protocol
    training_by_label = {
        artifact["label"]: artifact for artifact in training_artifacts(state.manifest)
    }
    success = True
    for case in evaluation_cases(protocol):
        training = training_by_label.get(case.training.label)
        if training is None:
            continue
        low, high = case.friction_range
        artifact = state.directory / "evaluation" / f"{case.label}.npz"
        spec = CellSpec(
            case,
            (
                sys.executable,
                str(REPO_ROOT / "scripts" / "eval_policy.py"),
                "--task",
                protocol["task"],
                "--ckpt",
                training["checkpoint"],
                "--train_label",
                training["label"],
                "--eval_backend",
                case.backend.physics_backend,
                "--eval_device",
                case.backend.device,
                "--eval_label",
                f"{case.backend.label}-{case.domain}",
                "--num_envs",
                str(protocol["eval_num_envs"]),
                "--t_end",
                str(protocol["eval_duration_s"]),
                "--seed",
                str(protocol["eval_seed"]),
                "--reset_mode",
                protocol["eval_reset_mode"],
                "--command_profile",
                protocol["eval_command_profile"],
                "--settling_time",
                str(protocol["eval_settling_time_s"]),
                "--contact_threshold",
                str(protocol["eval_contact_threshold_n"]),
                "--velocity_impulse",
                str(protocol["eval_velocity_impulse_m_per_s"]),
                "--contact_friction_grid",
                str(low),
                str(high),
                "--domain-randomization",
                "friction-only",
                "--out",
                str(artifact),
            ),
            artifact,
            state.directory / "stdout" / f"eval_{case.label}.log",
        )
        okay = run_cell(state, spec)
        success = okay and success
        if not okay and fail_fast:
            return False
    return success


def finite_mean(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def descriptive_stats(values):
    """Describe finite seed-level values without implying inferential evidence."""
    finite = np.asarray(
        [value for value in values if value is not None], dtype=np.float64
    )
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "mean": None, "sample_std": None}
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "sample_std": float(finite.std(ddof=1)) if len(finite) > 1 else None,
    }


def summarize_training(training):
    run_dir = Path(training["run_dir"])
    vitals_path = run_dir / "vitals.jsonl"
    records = [
        json.loads(line)
        for line in vitals_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps_per_s = [record.get("steps_per_s", math.nan) for record in records[2:]]
    tail = records[-min(10, len(records)) :]
    return {
        **training,
        "iterations": len(records),
        "reward_final_10_mean": finite_mean(
            [record.get("rewards/total_rewards", math.nan) for record in tail]
        ),
        "episode_time_final_10_mean": finite_mean(
            [record.get("episode_time", math.nan) for record in tail]
        ),
        "median_logged_steps_per_s_after_warmup": (
            float(np.nanmedian(steps_per_s)) if steps_per_s else None
        ),
    }


def friction_quartile_labels(friction):
    friction = np.asarray(friction)
    if np.ptp(friction) == 0.0:
        return np.full(len(friction), "fixed", dtype="<U8")
    quantiles = np.quantile(friction, [0.25, 0.5, 0.75])
    return np.asarray(
        [f"q{index + 1}" for index in np.digitize(friction, quantiles)],
        dtype="<U8",
    )


def summarize_evaluation(cell):
    artifact = Path(cell["artifact"])
    with np.load(artifact, allow_pickle=False) as data:
        friction = data["contact_friction"]
        quartiles = friction_quartile_labels(friction)
        arrays = {
            "mean_reward": data["mean_reward"],
            "survival": data["survived"].astype(np.float32),
            "episode_duration": data["ep_len"],
        }
        for metric in (
            "tracking_vx_rmse",
            "tracking_vy_rmse",
            "tracking_yaw_rmse",
            "base_tilt_rms",
            "foot_slip_speed_rms",
            "grf_balance_cv",
            "grf_min_leg_mean",
            "gait_trot_classified",
            "foot_contact_phase_match",
            "torque_utilization_rms",
            "torque_saturation_fraction",
            "unsafe_contact_fraction",
        ):
            key = f"metric_{metric}"
            if key in data:
                arrays[metric] = data[key]
        overall = {name: finite_mean(values) for name, values in arrays.items()}
        by_friction = {
            label: {
                name: finite_mean(values[quartiles == label])
                for name, values in arrays.items()
            }
            for label in dict.fromkeys(quartiles.tolist())
        }
        return {
            "label": cell["label"],
            "artifact": str(artifact),
            "train_label": str(data["train_label"]),
            "eval_label": str(data["eval_label"]),
            "friction_low": float(friction.min()),
            "friction_high": float(friction.max()),
            "overall": overall,
            "by_friction_quartile": by_friction,
        }


def speed_rows(manifest):
    protocol = manifest["protocol"]
    complete = completed_cells(manifest)
    rows = []
    for case in speed_cases(protocol):
        cell = complete.get(cell_key(case.stage, case.label))
        if cell is None:
            continue
        artifact = Path(cell["artifact"])
        valid, _ = validate_cell_artifact(case, protocol, artifact)
        if not valid:
            continue
        result = json.loads(artifact.read_text(encoding="utf-8"))
        artifact_protocol = result["protocol"]
        for profile in result["profiles"]:
            rows.append(
                {
                    "backend": artifact_protocol["backend_label"],
                    "num_envs": artifact_protocol["num_envs"],
                    "dr": "on" if artifact_protocol["dr_enabled"] else "off",
                    "profile": profile["profile"],
                    "setup_seconds": result["setup_seconds"],
                    "env_control_steps_per_s": profile[
                        "median_env_control_steps_per_s"
                    ],
                    "cuda_used_delta_bytes": result["memory"]["cuda_used_delta_bytes"],
                    "cuda_used_after_profiles_delta_bytes": result["memory"].get(
                        "cuda_used_after_profiles_delta_bytes"
                    ),
                    "host_max_rss_after_profiles_delta_kb": result["memory"].get(
                        "host_max_rss_after_profiles_delta_kb"
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["backend"],
            row["num_envs"],
            row["profile"],
            row["dr"],
        ),
    )


def paired_speed_effects(rows):
    grouped = {}
    for row in rows:
        key = (row["backend"], row["num_envs"], row["profile"])
        grouped.setdefault(key, {})[row["dr"]] = row
    effects = []
    for (backend, num_envs, profile), pair in sorted(grouped.items()):
        if set(pair) != {"off", "on"}:
            continue
        effects.append(
            {
                "backend": backend,
                "num_envs": num_envs,
                "profile": profile,
                "off_env_control_steps_per_s": pair["off"]["env_control_steps_per_s"],
                "on_env_control_steps_per_s": pair["on"]["env_control_steps_per_s"],
                "on_over_off": pair["on"]["env_control_steps_per_s"]
                / pair["off"]["env_control_steps_per_s"],
                "setup_on_minus_off_seconds": pair["on"]["setup_seconds"]
                - pair["off"]["setup_seconds"],
            }
        )
    return effects


def paired_eval_effects(evaluations):
    grouped = {}
    for evaluation in evaluations:
        train_label = evaluation["train_label"]
        training_backend, mode, seed = parse_training_label(train_label)
        key = (training_backend, seed, evaluation["eval_label"])
        grouped.setdefault(key, {})[mode] = evaluation
    effects = []
    for (training_backend, seed, eval_label), pair in sorted(grouped.items()):
        if set(pair) != {"off", "on"}:
            continue
        metrics = set(pair["off"]["overall"]) & set(pair["on"]["overall"])
        delta = {}
        for metric in sorted(metrics):
            off = pair["off"]["overall"][metric]
            on = pair["on"]["overall"][metric]
            delta[metric] = None if off is None or on is None else on - off
        effects.append(
            {
                "pair": f"{training_backend}_seed{seed}",
                "training_backend": training_backend,
                "seed": seed,
                "eval_label": eval_label,
                "delta_on_minus_off": delta,
            }
        )
    return effects


def parse_training_label(label):
    """Parse campaign labels such as ``warp_on_seed7`` without guessing mode."""
    prefix, separator, seed_text = label.rpartition("_seed")
    training_backend, mode_separator, mode = prefix.rpartition("_")
    if not separator or not mode_separator or mode not in {"off", "on"}:
        raise ValueError(f"invalid campaign training label: {label!r}")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise ValueError(f"invalid campaign training label: {label!r}") from error
    return training_backend, mode, seed


def paired_training_effects(training):
    """Return one raw ON-minus-OFF comparison for every completed seed pair."""
    grouped = {}
    for row in training:
        key = (row["backend"]["label"], int(row["seed"]))
        grouped.setdefault(key, {})[row["dr_mode"]] = row

    metrics = (
        "reward_final_10_mean",
        "episode_time_final_10_mean",
        "median_logged_steps_per_s_after_warmup",
    )
    effects = []
    for (backend, seed), pair in sorted(grouped.items()):
        if set(pair) != {"off", "on"}:
            continue
        deltas = {}
        for metric in metrics:
            off = pair["off"].get(metric)
            on = pair["on"].get(metric)
            deltas[metric] = None if off is None or on is None else on - off
        off_throughput = pair["off"].get("median_logged_steps_per_s_after_warmup")
        on_throughput = pair["on"].get("median_logged_steps_per_s_after_warmup")
        throughput_ratio = (
            None
            if off_throughput in {None, 0.0} or on_throughput is None
            else on_throughput / off_throughput
        )
        effects.append(
            {
                "training_backend": backend,
                "seed": seed,
                "delta_on_minus_off": deltas,
                "throughput_on_over_off": throughput_ratio,
            }
        )
    return effects


def aggregate_paired_training_effects(effects, protocol):
    """Pool seed-paired training effects by backend using descriptive stats."""
    requested_seeds = sorted(set(int(seed) for seed in protocol["seeds"]))
    rows = []
    for backend in (item.label for item in protocol_backends(protocol)):
        selected = [row for row in effects if row["training_backend"] == backend]
        paired_seeds = sorted(row["seed"] for row in selected)
        metric_names = sorted(
            {metric for row in selected for metric in row["delta_on_minus_off"]}
        )
        rows.append(
            {
                "training_backend": backend,
                "requested_seed_count": len(requested_seeds),
                "paired_seed_count": len(paired_seeds),
                "paired_seeds": paired_seeds,
                "missing_seed_pairs": sorted(set(requested_seeds) - set(paired_seeds)),
                "delta_on_minus_off": {
                    metric: descriptive_stats(
                        [row["delta_on_minus_off"].get(metric) for row in selected]
                    )
                    for metric in metric_names
                },
                "throughput_on_over_off": descriptive_stats(
                    [row["throughput_on_over_off"] for row in selected]
                ),
            }
        )
    return rows


def aggregate_paired_eval_effects(effects, protocol):
    """Pool seed-paired evaluation deltas for each train/evaluation domain."""
    requested_seeds = sorted(set(int(seed) for seed in protocol["seeds"]))
    rows = []
    groups = dict.fromkeys(
        (
            case.training.backend.label,
            f"{case.backend.label}-{case.domain}",
        )
        for case in evaluation_cases(protocol)
    )
    for training_backend, eval_label in groups:
        selected = [
            row
            for row in effects
            if row["training_backend"] == training_backend
            and row["eval_label"] == eval_label
        ]
        paired_seeds = sorted(row["seed"] for row in selected)
        metric_names = sorted(
            {metric for row in selected for metric in row["delta_on_minus_off"]}
        )
        rows.append(
            {
                "training_backend": training_backend,
                "eval_label": eval_label,
                "requested_seed_count": len(requested_seeds),
                "paired_seed_count": len(paired_seeds),
                "paired_seeds": paired_seeds,
                "missing_seed_pairs": sorted(set(requested_seeds) - set(paired_seeds)),
                "delta_on_minus_off": {
                    metric: descriptive_stats(
                        [row["delta_on_minus_off"].get(metric) for row in selected]
                    )
                    for metric in metric_names
                },
            }
        )
    return rows


def expected_campaign_cells(protocol):
    """Return the unique cells implied by the persisted campaign protocol."""
    expected = {"speed": set(), "train": set(), "eval": set()}
    for case in campaign_cases(protocol):
        expected[case.stage].add(case.label)
    return expected


def campaign_completion(manifest):
    """Count completed, failed, and not-yet-complete planned cells by stage."""
    protocol = manifest["protocol"]
    planned = campaign_cases(protocol)
    actual = {
        cell_key(cell["stage"], cell["label"]): cell for cell in manifest["cells"]
    }
    stages = {}
    for stage in ("speed", "train", "eval"):
        cases = [case for case in planned if case.stage == stage]
        counts = {
            "expected": len(cases),
            "complete": 0,
            "failed": 0,
            "incomplete": 0,
            "failed_labels": [],
            "incomplete_labels": [],
        }
        for case in sorted(cases, key=lambda item: item.label):
            cell = actual.get(cell_key(stage, case.label))
            if cell is None:
                counts["incomplete"] += 1
                counts["incomplete_labels"].append(case.label)
            elif cell["status"] == "failed":
                counts["failed"] += 1
                counts["failed_labels"].append(case.label)
            elif cell["status"] == "complete" and cell.get("artifact"):
                valid, _ = validate_cell_artifact(case, protocol, cell["artifact"])
                if valid:
                    counts["complete"] += 1
                else:
                    counts["incomplete"] += 1
                    counts["incomplete_labels"].append(case.label)
            else:
                counts["incomplete"] += 1
                counts["incomplete_labels"].append(case.label)
        stages[stage] = counts
    overall = {
        key: sum(stage[key] for stage in stages.values())
        for key in ("expected", "complete", "failed", "incomplete")
    }
    return {"overall": overall, "by_stage": stages}


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_descriptive(stats, *, signed=True):
    if stats is None or stats["mean"] is None:
        return "n/a"
    sign = "+" if signed else ""
    mean = f"{stats['mean']:{sign}.3f}"
    if stats["sample_std"] is None:
        return f"{mean} (n={stats['count']})"
    return f"{mean} ± {stats['sample_std']:.3f}"


def markdown_report(summary, protocol):
    seed_count = len(protocol["seeds"])
    friction_range = protocol["contact_friction_dr_range"]
    nominal_friction = protocol["environment_config_template"]["terrain"][
        "dynamic_friction"
    ]
    completion = summary["completion"]
    lines = [
        "# Contact-friction domain-randomization campaign",
        "",
        f"This is a descriptive paired comparison across {seed_count} requested "
        f"training seed{'s' if seed_count != 1 else ''}. Mean ± sample SD is "
        "computed only from completed ON/OFF seed pairs. These summaries do not "
        "establish statistical significance or generalization to other seeds.",
        "",
        (
            "Fewer than three seeds were requested, so all policy comparisons are "
            "screening evidence only."
            if seed_count < 3
            else "The observed seed variation is reported descriptively; no "
            "hypothesis test or confidence claim is made."
        ),
        "",
        f"The ON condition samples friction uniformly from `{friction_range}`; "
        f"the OFF condition uses `{nominal_friction}`. This changes both variance "
        "and mean friction.",
        "",
        "## Campaign completion",
        "",
        "| stage | expected | complete | failed | incomplete |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stage in ("speed", "train", "eval"):
        counts = completion["by_stage"][stage]
        lines.append(
            f"| {stage} | {counts['expected']} | {counts['complete']} | "
            f"{counts['failed']} | {counts['incomplete']} |"
        )
    counts = completion["overall"]
    lines.extend(
        [
            f"| **total** | **{counts['expected']}** | **{counts['complete']}** | "
            f"**{counts['failed']}** | **{counts['incomplete']}** |",
            "",
            "Failed and incomplete cell labels are retained in `summary.json`.",
            "",
            "## Speed",
            "",
            "| backend | envs | reset profile | off env-steps/s | on env-steps/s "
            "| on/off | setup Δ s |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["speed_effects"]:
        lines.append(
            f"| {row['backend']} | {row['num_envs']} | {row['profile']} | "
            f"{row['off_env_control_steps_per_s']:.0f} | "
            f"{row['on_env_control_steps_per_s']:.0f} | "
            f"{row['on_over_off']:.3f} | "
            f"{row['setup_on_minus_off_seconds']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired training effects across seeds",
            "",
            "Δ is ON minus OFF. Values are mean ± sample SD across completed "
            "seed pairs; `n=1` has no sample SD.",
            "",
            "| backend | pairs | reward Δ | episode-time Δ s | throughput on/off |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["training_effects_aggregate"]:
        metrics = row["delta_on_minus_off"]
        lines.append(
            f"| {row['training_backend']} | "
            f"{row['paired_seed_count']}/{row['requested_seed_count']} | "
            f"{format_descriptive(metrics.get('reward_final_10_mean'))} | "
            f"{format_descriptive(metrics.get('episode_time_final_10_mean'))} | "
            f"{format_descriptive(row['throughput_on_over_off'], signed=False)} |"
        )
    lines.extend(
        [
            "",
            "## Paired policy-evaluation effects across seeds",
            "",
            "Δ is ON minus OFF. Positive reward/survival and negative tracking, "
            "slip, or tilt errors are favorable. These are descriptive seed-level "
            "differences, not confidence intervals.",
            "",
            "| train backend | eval domain | pairs | reward Δ | survival Δ | "
            "vx RMSE Δ | slip Δ | tilt Δ |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["evaluation_effects_aggregate"]:
        metrics = row["delta_on_minus_off"]
        lines.append(
            f"| {row['training_backend']} | {row['eval_label']} | "
            f"{row['paired_seed_count']}/{row['requested_seed_count']} | "
            f"{format_descriptive(metrics.get('mean_reward'))} | "
            f"{format_descriptive(metrics.get('survival'))} | "
            f"{format_descriptive(metrics.get('tracking_vx_rmse'))} | "
            f"{format_descriptive(metrics.get('foot_slip_speed_rms'))} | "
            f"{format_descriptive(metrics.get('base_tilt_rms'))} |"
        )
    lines.extend(
        [
            "",
            "Raw per-run training rows and curves, per-evaluation summaries, "
            "friction-quartile breakdowns, and seed-level paired deltas are retained "
            "in `summary.json`. Raw per-environment friction and physical metrics "
            "remain in each evaluation NPZ.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_campaign(campaign_dir, manifest):
    protocol = manifest["protocol"]
    speed = speed_rows(manifest)
    training = [
        summarize_training(artifact) for artifact in training_artifacts(manifest)
    ]
    complete = completed_cells(manifest)
    evaluations = []
    for case in evaluation_cases(protocol):
        cell = complete.get(cell_key(case.stage, case.label))
        if cell is None:
            continue
        valid, _ = validate_cell_artifact(case, protocol, cell["artifact"])
        if valid:
            evaluations.append(summarize_evaluation(cell))
    training_effects = paired_training_effects(training)
    evaluation_effects = paired_eval_effects(evaluations)
    summary = {
        "created_at": utc_now(),
        "protocol": protocol,
        "completion": campaign_completion(manifest),
        "speed": speed,
        "speed_effects": paired_speed_effects(speed),
        "training": training,
        "training_effects": training_effects,
        "training_effects_aggregate": aggregate_paired_training_effects(
            training_effects, protocol
        ),
        "evaluation": evaluations,
        "evaluation_effects": evaluation_effects,
        "evaluation_effects_aggregate": aggregate_paired_eval_effects(
            evaluation_effects,
            protocol,
        ),
    }
    write_json(campaign_dir / "summary.json", summary)
    write_csv(campaign_dir / "speed.csv", speed)
    (campaign_dir / "report.md").write_text(
        markdown_report(summary, protocol),
        encoding="utf-8",
    )
    print(f"wrote {campaign_dir / 'report.md'}", flush=True)
    return summary


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="go2trot")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["speed", "train", "eval", "summarize"],
        default=["speed", "train", "eval", "summarize"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--speed_common_num_envs", type=int, default=256)
    parser.add_argument("--speed_production_num_envs", type=int, default=4096)
    parser.add_argument("--speed_target_env_steps", type=int, default=32768)
    parser.add_argument("--speed_repeats", type=int, default=5)
    parser.add_argument("--speed_min_steps", type=int, default=50)
    parser.add_argument("--train_num_envs", type=int, default=4096)
    parser.add_argument("--train_iterations", type=int, default=100)
    parser.add_argument("--eval_num_envs", type=int, default=200)
    parser.add_argument("--eval_duration", type=float, default=5.0)
    parser.add_argument("--eval_settling_time", type=float, default=0.5)
    parser.add_argument("--eval_seed", type=int, default=1701)
    parser.add_argument("--cross_backend_eval", action="store_true")
    args = parser.parse_args(argv)
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = LOG_ROOT / f"dr_contact_friction_{stamp}"
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
        args.speed_repeats,
        args.speed_min_steps,
        args.train_num_envs,
        args.train_iterations,
        args.eval_num_envs,
    )
    if any(value <= 0 for value in positive):
        parser.error("environment, iteration, repeat, and step counts must be positive")
    if args.eval_duration <= 0:
        parser.error("--eval_duration must be positive")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must not contain duplicates")
    return args


def main(argv=None):
    args = get_args(argv)
    campaign_dir = args.output
    manifest_path = campaign_dir / "manifest.json"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    protocol = requested_protocol(args)
    provenance = source_provenance(
        campaign_dir, copy_sources=not manifest_path.exists()
    )
    manifest = load_manifest(manifest_path, args, protocol, provenance)
    write_json(manifest_path, manifest)
    state = CampaignState(campaign_dir, manifest_path, manifest, args.resume)

    success = True
    if "speed" in args.stages:
        success &= run_speed_stage(state, args.fail_fast)
    if "train" in args.stages:
        success &= run_training_stage(state, args.fail_fast)
    if "eval" in args.stages:
        success &= run_evaluation_stage(state, args.fail_fast)
    if "summarize" in args.stages:
        summarize_campaign(campaign_dir, manifest)
    if not success:
        raise SystemExit("one or more campaign cells failed; inspect manifest.json")


if __name__ == "__main__":
    main()
