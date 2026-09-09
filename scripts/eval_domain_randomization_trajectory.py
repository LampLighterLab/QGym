#!/usr/bin/env python3
"""Record deterministic one-robot DR-on/off policy trajectories.

The public command launches six fresh child processes: MuJoCo CPU, MuJoCo
Warp, and VSim, each at nominal friction and at one fixed representative DR
realization.  The fixed realization is necessary for a one-robot trajectory:
sampling a random coefficient would confound the comparison.

    uv run --env-file .env.vsim \
        scripts/eval_domain_randomization_trajectory.py
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from gym import GYM_ROOT_DIR
from gym.utils.policy_io import state_component_names
from gym.utils.torch_quat import quat_rotate_inverse

if __package__:
    from .eval_policy import build
else:
    from eval_policy import build


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "logs/go2trot/Aug11_23-14-45_/model_1000.pt"
DEFAULT_OUTPUT = (
    REPO_ROOT / "logs/domain_randomization_trajectory/Aug11_23-14-45_model_1000"
)


@dataclass(frozen=True)
class Backend:
    label: str
    backend: str
    device: str


BACKENDS = (
    Backend("mujoco", "mujoco", "cpu"),
    Backend("mjx", "mujoco", "cuda:0"),
    Backend("vsim", "vsim", "cuda:0"),
)
DR_MODES = ("off", "on")


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--dr-friction", type=float, default=0.75)
    parser.add_argument("--command", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    parser.add_argument("--mujoco-njmax", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cell", choices=[backend.label for backend in BACKENDS])
    parser.add_argument("--dr-mode", choices=DR_MODES)
    return parser.parse_args(argv)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quat_rotate_xyzw(quat, vector):
    q_xyz = quat[..., :3]
    twice_cross = 2.0 * torch.cross(q_xyz, vector, dim=-1)
    return (
        vector + quat[..., 3:4] * twice_cross + torch.cross(q_xyz, twice_cross, dim=-1)
    )


def canonical_body_inertial_properties(env):
    """Return canonical body masses and body-frame center-of-mass offsets."""
    asset_path = env.cfg.asset.file.format(GYM_ROOT_DIR=GYM_ROOT_DIR)
    links = {
        link.get("name"): link
        for link in ET.parse(asset_path).getroot().findall("link")
    }
    masses = []
    local_com = []
    for name in env.robot_layout.body_names:
        inertial = links[name].find("inertial")
        if inertial is None:
            masses.append(0.0)
            local_com.append([0.0, 0.0, 0.0])
            continue
        mass = inertial.find("mass")
        origin = inertial.find("origin")
        masses.append(float(mass.get("value")) if mass is not None else 0.0)
        xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        local_com.append([float(value) for value in xyz.split()])
    return (
        torch.tensor(masses, device=env.device),
        torch.tensor(local_com, device=env.device),
    )


def system_cog(body_states, body_mass, body_local_com):
    """Compute whole-robot CoG position and velocity in the world frame."""
    local_com = body_local_com.unsqueeze(0).expand(body_states.shape[0], -1, -1)
    offset_world = _quat_rotate_xyzw(body_states[..., 3:7], local_com)
    body_com_position = body_states[..., 0:3] + offset_world
    body_com_velocity = body_states[..., 7:10] + torch.cross(
        body_states[..., 10:13], offset_world, dim=-1
    )
    mass = body_mass[None, :, None]
    total_mass = body_mass.sum()
    return (
        torch.sum(body_com_position * mass, dim=1) / total_mass,
        torch.sum(body_com_velocity * mass, dim=1) / total_mass,
    )


def _state_buffers(num_samples, num_feet, num_dof, num_actuators):
    return {
        "root_state": np.empty((num_samples, 13), dtype=np.float32),
        "cog_position": np.empty((num_samples, 3), dtype=np.float32),
        "cog_velocity": np.empty((num_samples, 3), dtype=np.float32),
        "base_lin_velocity_body": np.empty((num_samples, 3), dtype=np.float32),
        "base_ang_velocity_body": np.empty((num_samples, 3), dtype=np.float32),
        "base_tilt_deg": np.empty(num_samples, dtype=np.float32),
        "foot_position_world": np.empty((num_samples, num_feet, 3), dtype=np.float32),
        "foot_position_body": np.empty((num_samples, num_feet, 3), dtype=np.float32),
        "foot_velocity_world": np.empty((num_samples, num_feet, 3), dtype=np.float32),
        "foot_grf_world": np.empty((num_samples, num_feet, 3), dtype=np.float32),
        "dof_position": np.empty((num_samples, num_dof), dtype=np.float32),
        "dof_velocity": np.empty((num_samples, num_dof), dtype=np.float32),
        "joint_torque": np.empty((num_samples, num_actuators), dtype=np.float32),
        "mechanical_power": np.empty(num_samples, dtype=np.float32),
        "terminated": np.empty(num_samples, dtype=np.bool_),
        "timed_out": np.empty(num_samples, dtype=np.bool_),
    }


def _record_state(env, buffers, index, body_mass, body_local_com):
    body_states = env._backend.rigid_body_states.view(env.num_envs, env.num_bodies, 13)
    cog_position, cog_velocity = system_cog(body_states, body_mass, body_local_com)
    feet = env.feet_indices
    foot_position = body_states[:, feet, 0:3]
    foot_relative_world = foot_position - cog_position[:, None, :]
    base_quat = env.root_states[:, 3:7]
    foot_relative_body = quat_rotate_inverse(
        base_quat[:, None, :].expand(-1, len(feet), -1).reshape(-1, 4),
        foot_relative_world.reshape(-1, 3),
    ).reshape(env.num_envs, len(feet), 3)
    actuated_velocity = env.dof_vel.index_select(1, env.actuated_dof_indices)
    torque = env.torques

    buffers["root_state"][index] = env.root_states[0].detach().cpu().numpy()
    buffers["cog_position"][index] = cog_position[0].detach().cpu().numpy()
    buffers["cog_velocity"][index] = cog_velocity[0].detach().cpu().numpy()
    buffers["base_lin_velocity_body"][index] = (
        env.base_lin_vel[0].detach().cpu().numpy()
    )
    buffers["base_ang_velocity_body"][index] = (
        env.base_ang_vel[0].detach().cpu().numpy()
    )
    buffers["base_tilt_deg"][index] = (
        torch.rad2deg(torch.acos(torch.clamp(-env.projected_gravity[0, 2], -1.0, 1.0)))
        .detach()
        .cpu()
    )
    buffers["foot_position_world"][index] = foot_position[0].detach().cpu().numpy()
    buffers["foot_position_body"][index] = foot_relative_body[0].detach().cpu().numpy()
    buffers["foot_velocity_world"][index] = (
        body_states[0, feet, 7:10].detach().cpu().numpy()
    )
    buffers["foot_grf_world"][index] = (
        env.contact_forces[0, feet].detach().cpu().numpy()
    )
    buffers["dof_position"][index] = env.dof_pos[0].detach().cpu().numpy()
    buffers["dof_velocity"][index] = env.dof_vel[0].detach().cpu().numpy()
    buffers["joint_torque"][index] = torque[0].detach().cpu().numpy()
    buffers["mechanical_power"][index] = (
        torch.sum(torch.abs(torque[0] * actuated_velocity[0])).detach().cpu()
    )
    buffers["terminated"][index] = bool(env.terminated[0])
    buffers["timed_out"][index] = bool(env.timed_out[0])


def _backend_for_label(label):
    return next(backend for backend in BACKENDS if backend.label == label)


def run_cell(args):
    backend = _backend_for_label(args.cell)
    friction = 1.0 if args.dr_mode == "off" else args.dr_friction
    friction_grid = None if args.dr_mode == "off" else [friction, friction]
    env, runner = build(
        task="go2trot",
        eval_backend=backend.backend,
        eval_device=backend.device,
        num_envs=1,
        t_end=args.duration,
        ckpt=args.checkpoint,
        reset_mode="reset_to_basic",
        seed=args.seed,
        contact_friction_grid=friction_grid,
        contact_friction_dr=args.dr_mode,
        original_cfg=True,
        mujoco_njmax=args.mujoco_njmax,
    )
    if runner.alg.actor.training or runner.alg.critic.training:
        raise RuntimeError("policy runner did not enter evaluation mode")

    env.commands[0, :3] = torch.tensor(
        args.command, dtype=env.commands.dtype, device=env.device
    )
    if hasattr(env, "_update_cmd_switch"):
        env._update_cmd_switch()
    if args.dr_mode == "on":
        env._backend.set_contact_friction(
            torch.tensor([0], dtype=torch.long, device=env.device),
            torch.tensor([friction], device=env.device),
        )

    n_steps = int(round(args.duration * float(env.cfg.control.ctrl_frequency)))
    num_feet = len(env.feet_indices)
    buffers = _state_buffers(n_steps, num_feet, env.num_dof, env.num_actuators)
    policy_actions = np.empty((n_steps, runner.alg.actor.num_actions), dtype=np.float32)
    position_targets = np.empty((n_steps, env.num_actuators), dtype=np.float32)
    body_mass, body_local_com = canonical_body_inertial_properties(env)
    initial_root_state = env.root_states[0].detach().cpu().numpy().copy()
    initial_dof_position = env.dof_pos[0].detach().cpu().numpy().copy()
    initial_dof_velocity = env.dof_vel[0].detach().cpu().numpy().copy()

    with torch.inference_mode():
        for step in range(n_steps):
            actions = runner.get_inference_actions()
            if step == 0:
                repeated = runner.get_inference_actions()
                torch.testing.assert_close(actions, repeated, rtol=0.0, atol=0.0)
            runner.set_actions(
                runner.actor_cfg["actions"],
                actions,
                runner.actor_cfg["disable_actions"],
            )
            policy_actions[step] = actions[0].detach().cpu().numpy()
            position_targets[step] = env.dof_pos_target[0].detach().cpu().numpy()
            env.step()
            _record_state(env, buffers, step, body_mass, body_local_com)

    applied_friction = float(env.domain_randomizer.contact_friction[0].cpu())
    actor_training = runner.alg.actor.training
    critic_training = runner.alg.critic.training
    env._backend.close()

    artifact_path = args.output / f"{backend.label}_dr_{args.dr_mode}.npz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        schema_version=np.int64(1),
        task="go2trot",
        backend_label=backend.label,
        backend=backend.backend,
        device=backend.device,
        dr_mode=args.dr_mode,
        contact_friction=np.float32(applied_friction),
        command=np.asarray(args.command, dtype=np.float32),
        checkpoint_path=str(args.checkpoint.resolve()),
        checkpoint_sha256=_sha256(args.checkpoint),
        checkpoint_iteration=np.int64(runner.it),
        original_cfg=np.bool_(True),
        inference_mode=np.bool_(not actor_training and not critic_training),
        seed=np.int64(args.seed),
        duration_s=np.float64(args.duration),
        ctrl_dt=np.float64(env.dt),
        time=(np.arange(n_steps, dtype=np.float64) + 1.0) * float(env.dt),
        action_time=np.arange(n_steps, dtype=np.float64) * float(env.dt),
        initial_root_state=initial_root_state,
        initial_dof_position=initial_dof_position,
        initial_dof_velocity=initial_dof_velocity,
        foot_names=np.asarray(env.robot_layout.body_groups["feet"]),
        dof_names=np.asarray(env.dof_names),
        action_names=np.asarray(
            state_component_names(env, runner.actor_cfg["actions"])
        ),
        policy_action=policy_actions,
        dof_position_target=position_targets,
        **buffers,
    )
    print(f"wrote {artifact_path}")


def cell_command(args, backend, dr_mode):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--output",
        str(args.output.resolve()),
        "--duration",
        str(args.duration),
        "--seed",
        str(args.seed),
        "--dr-friction",
        str(args.dr_friction),
        "--command",
        *[str(value) for value in args.command],
        "--mujoco-njmax",
        str(args.mujoco_njmax),
        "--cell",
        backend.label,
        "--dr-mode",
        dr_mode,
    ]


def validate_pair(output, backend, args):
    artifacts = {}
    for dr_mode in DR_MODES:
        path = output / f"{backend.label}_dr_{dr_mode}.npz"
        with np.load(path, allow_pickle=False) as data:
            artifacts[dr_mode] = {key: data[key] for key in data.files}
        artifact = artifacts[dr_mode]
        if not bool(artifact["inference_mode"]):
            raise ValueError(f"{path}: policy was not in inference mode")
        if int(artifact["checkpoint_iteration"]) != 1000:
            raise ValueError(f"{path}: wrong checkpoint iteration")
        expected_friction = 1.0 if dr_mode == "off" else args.dr_friction
        if float(artifact["contact_friction"]) != expected_friction:
            raise ValueError(f"{path}: wrong friction")
        if not np.array_equal(artifact["command"], np.asarray(args.command)):
            raise ValueError(f"{path}: wrong command")

    off = artifacts["off"]
    on = artifacts["on"]
    for key in (
        "initial_root_state",
        "initial_dof_position",
        "initial_dof_velocity",
    ):
        np.testing.assert_allclose(off[key], on[key], rtol=0.0, atol=0.0)


def run_campaign(args):
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for backend in BACKENDS:
        for dr_mode in DR_MODES:
            label = f"{backend.label}_dr_{dr_mode}"
            artifact = args.output / f"{label}.npz"
            log_path = args.output / f"{label}.log"
            command = cell_command(args, backend, dr_mode)
            if args.resume and artifact.is_file():
                records.append({"label": label, "status": "reused"})
                continue
            print(f"\n[{label}] {' '.join(command)}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_file.write(line)
                returncode = process.wait()
            if returncode != 0:
                raise RuntimeError(f"{label} failed; see {log_path}")
            if "overflow" in log_path.read_text(encoding="utf-8").lower():
                raise RuntimeError(f"{label} emitted an overflow; see {log_path}")
            records.append({"label": label, "status": "complete"})
        validate_pair(args.output, backend, args)

    manifest = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "duration_s": args.duration,
        "num_envs": 1,
        "seed": args.seed,
        "command": list(args.command),
        "dr_off_friction": 1.0,
        "dr_on_friction": args.dr_friction,
        "reset_mode": "reset_to_basic",
        "original_cfg": True,
        "cells": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output / 'manifest.json'}")


def main():
    args = get_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.duration <= 0:
        raise ValueError("duration must be positive")
    if args.dr_friction < 0:
        raise ValueError("dr-friction cannot be negative")
    if (args.cell is None) != (args.dr_mode is None):
        raise ValueError("--cell and --dr-mode must be supplied together")
    if args.cell is not None:
        run_cell(args)
    else:
        run_campaign(args)


if __name__ == "__main__":
    main()
