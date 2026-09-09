#!/usr/bin/env python3
"""Compare VSim and MuJoCo while policy actions are queried but not applied.

Each backend runs in a fresh child process. Both robots start from the same
deterministic state and evolve under Go2Trot's open-loop gait and PD controller.
The policy is evaluated at every control step, but its output is never copied
into the environment's action buffers.

Run through the VSim environment so both child processes inherit its license:

    uv run --env-file .env.vsim \
        scripts/compare_backend_unapplied_actions.py \
        --checkpoint logs/go2trot/<run>/model_1000.pt
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from gym.utils.helpers import class_to_dict
from gym.utils.policy_io import state_component_names, state_component_scales

if __package__:
    from .eval_policy import build, resolve_ckpt
else:
    from eval_policy import build, resolve_ckpt


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Backend:
    label: str
    backend: str
    device: str


BACKENDS = (
    Backend("vsim", "vsim", "cuda:0"),
    Backend("mujoco", "mujoco", "cuda:0"),
)
CONTACT_ACTIVE_N = 1.0e-4


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/backend_unapplied_actions"),
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        default=(0.0, 0.0, -0.75),
        metavar=("VX", "VY", "YAW"),
    )
    parser.add_argument("--mujoco-njmax", type=int, default=256)
    parser.add_argument(
        "--mujoco-solref",
        type=float,
        nargs=2,
        metavar=("TIMECONST", "DAMPING_RATIO"),
        help="override every MuJoCo geom's solref after loading original config",
    )
    parser.add_argument(
        "--disable-motors",
        action="store_true",
        help="send zero torque while retaining the same state and policy queries",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cell", choices=[backend.label for backend in BACKENDS])
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def backend_for_label(label):
    return next(backend for backend in BACKENDS if backend.label == label)


def cell_command(args, backend):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(args.output),
        "--duration",
        str(args.duration),
        "--seed",
        str(args.seed),
        "--command",
        *[str(value) for value in args.command],
        "--mujoco-njmax",
        str(args.mujoco_njmax),
        "--cell",
        backend.label,
    ]
    if args.disable_motors:
        command.append("--disable-motors")
    if args.mujoco_solref is not None:
        command.extend(("--mujoco-solref", *map(str, args.mujoco_solref)))
    return command


def _allocate_samples(num_samples, env, runner):
    actor_width = sum(getattr(env, name).shape[1] for name in runner.actor_cfg["obs"])
    action_width = sum(
        getattr(env, name).shape[1] for name in runner.actor_cfg["actions"]
    )
    return {
        "actor_observations": np.empty((num_samples, actor_width), dtype=np.float32),
        "queried_actions": np.empty((num_samples, action_width), dtype=np.float32),
        "environment_actions": np.empty((num_samples, action_width), dtype=np.float32),
        "contact_forces": np.empty((num_samples, env.num_bodies, 3), dtype=np.float32),
        "dof_position": np.empty((num_samples, env.num_dof), dtype=np.float32),
        "dof_velocity": np.empty((num_samples, env.num_dof), dtype=np.float32),
        "root_state": np.empty((num_samples, 13), dtype=np.float32),
        "rigid_body_state": np.empty(
            (num_samples, env.num_bodies, 13), dtype=np.float32
        ),
        "base_linear_velocity_body": np.empty((num_samples, 3), dtype=np.float32),
        "base_angular_velocity_body": np.empty((num_samples, 3), dtype=np.float32),
        "projected_gravity": np.empty((num_samples, 3), dtype=np.float32),
        "total_contact_force": np.empty((num_samples, 3), dtype=np.float32),
        "gait_reference": np.empty((num_samples, env.num_actuators), dtype=np.float32),
        "phase": np.empty((num_samples, env.phase.shape[1]), dtype=np.float32),
        "phase_frequency": np.empty(
            (num_samples, env.phase_frequency.shape[1]), dtype=np.float32
        ),
        "terminated": np.empty(num_samples, dtype=np.bool_),
        "timed_out": np.empty(num_samples, dtype=np.bool_),
    }


def _allocate_transitions(num_steps, env):
    shape = (num_steps, env.num_actuators)
    return {
        "target_position": np.empty(shape, dtype=np.float32),
        "position_error": np.empty(shape, dtype=np.float32),
        "velocity_error": np.empty(shape, dtype=np.float32),
        "position_torque": np.empty(shape, dtype=np.float32),
        "damping_torque": np.empty(shape, dtype=np.float32),
        "feedforward_torque": np.empty(shape, dtype=np.float32),
        "unclipped_torque": np.empty(shape, dtype=np.float32),
        "controller_torque": np.empty(shape, dtype=np.float32),
        "applied_torque": np.empty(shape, dtype=np.float32),
        "torque_saturated": np.empty(shape, dtype=np.bool_),
    }


def _record_controller_transition(transitions, index, env):
    position = env.dof_pos.index_select(1, env.actuated_dof_indices)
    velocity = env.dof_vel.index_select(1, env.actuated_dof_indices)
    default = env.default_dof_pos.index_select(1, env.actuated_dof_indices)
    target = env.gait_reference + env.dof_pos_target + default
    position_error = target - position
    velocity_error = env.dof_vel_target - velocity
    position_torque = env.p_gains * position_error
    damping_torque = env.d_gains * velocity_error
    unclipped_torque = position_torque + damping_torque + env.tau_ff
    controller_torque = env._compute_torques()

    transitions["target_position"][index] = target[0].cpu().numpy()
    transitions["position_error"][index] = position_error[0].cpu().numpy()
    transitions["velocity_error"][index] = velocity_error[0].cpu().numpy()
    transitions["position_torque"][index] = position_torque[0].cpu().numpy()
    transitions["damping_torque"][index] = damping_torque[0].cpu().numpy()
    transitions["feedforward_torque"][index] = env.tau_ff[0].cpu().numpy()
    transitions["unclipped_torque"][index] = unclipped_torque[0].cpu().numpy()
    transitions["controller_torque"][index] = controller_torque[0].cpu().numpy()
    transitions["torque_saturated"][index] = (
        (unclipped_torque[0].abs() > env.actuated_torque_limits).cpu().numpy()
    )


def _record_sample(samples, index, env, runner):
    observations = runner.get_obs(runner.actor_cfg["obs"])
    actions = runner.get_inference_actions()
    environment_actions = torch.cat(
        [getattr(env, name) for name in runner.actor_cfg["actions"]], dim=-1
    )

    samples["actor_observations"][index] = observations[0].cpu().numpy()
    samples["queried_actions"][index] = actions[0].cpu().numpy()
    samples["environment_actions"][index] = environment_actions[0].cpu().numpy()
    samples["contact_forces"][index] = env.contact_forces[0].cpu().numpy()
    samples["dof_position"][index] = env.dof_pos[0].cpu().numpy()
    samples["dof_velocity"][index] = env.dof_vel[0].cpu().numpy()
    samples["root_state"][index] = env.root_states[0].cpu().numpy()
    samples["rigid_body_state"][index] = (
        env._rigid_body_state.view(env.num_envs, env.num_bodies, 13)[0].cpu().numpy()
    )
    samples["base_linear_velocity_body"][index] = env.base_lin_vel[0].cpu().numpy()
    samples["base_angular_velocity_body"][index] = env.base_ang_vel[0].cpu().numpy()
    samples["projected_gravity"][index] = env.projected_gravity[0].cpu().numpy()
    samples["total_contact_force"][index] = (
        env.contact_forces[0].sum(dim=0).cpu().numpy()
    )
    samples["gait_reference"][index] = env.gait_reference[0].cpu().numpy()
    samples["phase"][index] = env.phase[0].cpu().numpy()
    samples["phase_frequency"][index] = env.phase_frequency[0].cpu().numpy()
    samples["terminated"][index] = bool(env.terminated[0])
    samples["timed_out"][index] = bool(env.timed_out[0])


def run_cell(args):
    backend = backend_for_label(args.cell)
    checkpoint = Path(resolve_ckpt(args.checkpoint)).resolve()
    env, runner = build(
        task="go2trot",
        eval_backend=backend.backend,
        eval_device=backend.device,
        num_envs=1,
        t_end=args.duration,
        ckpt=checkpoint,
        reset_mode="reset_to_basic",
        seed=args.seed,
        original_cfg=True,
        mujoco_njmax=args.mujoco_njmax,
        domain_randomization="off",
        control_at_sim_frequency=True,
        mujoco_geom_solref=args.mujoco_solref,
    )
    try:
        if runner.alg.actor.training or runner.alg.critic.training:
            raise RuntimeError("policy runner did not enter evaluation mode")

        env.cfg.asset.disable_motors = args.disable_motors

        env.commands[0, :3] = torch.tensor(
            args.command, dtype=env.commands.dtype, device=env.device
        )
        n_steps = int(round(args.duration * env.cfg.control.ctrl_frequency))
        num_samples = n_steps + 1
        samples = _allocate_samples(num_samples, env, runner)
        transitions = _allocate_transitions(n_steps, env)

        with torch.inference_mode():
            for step in range(num_samples):
                env._pre_decimation_step()
                _record_sample(samples, step, env, runner)
                if step < n_steps:
                    _record_controller_transition(transitions, step, env)
                    # Deliberately omit runner.set_actions(...). The queried
                    # policy output cannot influence the simulated trajectory.
                    env.step()
                    transitions["applied_torque"][step] = env.torques[0].cpu().numpy()

        actor_names = state_component_names(env, runner.actor_cfg["obs"])
        action_names = state_component_names(env, runner.actor_cfg["actions"])
        scales = class_to_dict(env.cfg.scaling)
        actor_scales = state_component_scales(env, runner.actor_cfg["obs"], scales)
        action_scales = state_component_scales(env, runner.actor_cfg["actions"], scales)
        robot_mass = float(env._backend.link_mass[0].sum().cpu())
        ctrl_dt = float(env.dt)
        sim_dt = float(env.cfg.sim_dt)
        decimation = int(env.cfg.control.decimation)
        checkpoint_iteration = runner.it
        body_names = np.asarray(env.robot_layout.body_names)
        foot_names = np.asarray(env.robot_layout.body_groups["feet"])
        dof_names = np.asarray(env.dof_names)
        actuated_dof_names = np.asarray(env.actuated_dof_names)
        link_mass = env._backend.link_mass[0].cpu().numpy().copy()
        link_inertia = env._backend.link_inertia[0].cpu().numpy().copy()
        contact_friction = float(env._backend.contact_friction[0].cpu())
        p_gains = env.p_gains[0].cpu().numpy().copy()
        d_gains = env.d_gains[0].cpu().numpy().copy()
        torque_limits = env.actuated_torque_limits.cpu().numpy().copy()
        default_dof_position = env.default_dof_pos[0].cpu().numpy().copy()
    finally:
        env._backend.close()

    time = np.arange(num_samples, dtype=np.float64) * ctrl_dt
    dof_acceleration = np.gradient(samples["dof_velocity"], ctrl_dt, axis=0)
    rigid_body_linear_acceleration = np.gradient(
        samples["rigid_body_state"][..., 7:10], ctrl_dt, axis=0
    )
    contact_impulse = np.zeros_like(samples["contact_forces"])
    contact_impulse[1:] = np.cumsum(samples["contact_forces"][1:] * ctrl_dt, axis=0)
    total_contact_impulse = contact_impulse.sum(axis=1)

    artifact_path = args.output / f"{backend.label}.npz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output,
            schema_version=np.int64(3),
            task="go2trot",
            backend_label=backend.label,
            backend=backend.backend,
            device=backend.device,
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=sha256_file(checkpoint),
            checkpoint_iteration=np.int64(checkpoint_iteration),
            original_cfg=np.bool_(True),
            inference_mode=np.bool_(True),
            actions_applied=np.bool_(False),
            motors_enabled=np.bool_(not args.disable_motors),
            domain_randomization="off",
            seed=np.int64(args.seed),
            command=np.asarray(args.command, dtype=np.float32),
            duration_s=np.float64(args.duration),
            ctrl_dt=np.float64(ctrl_dt),
            sim_dt=np.float64(sim_dt),
            control_frequency_hz=np.float64(1.0 / ctrl_dt),
            simulation_frequency_hz=np.float64(1.0 / sim_dt),
            decimation=np.int64(decimation),
            mujoco_solref=np.asarray(
                args.mujoco_solref if args.mujoco_solref is not None else (),
                dtype=np.float64,
            ),
            time=time,
            transition_time=time[:-1],
            actor_observation_fields=np.asarray(runner.actor_cfg["obs"]),
            actor_observation_names=np.asarray(actor_names),
            actor_observation_scales=np.asarray(actor_scales, dtype=np.float32),
            action_fields=np.asarray(runner.actor_cfg["actions"]),
            action_names=np.asarray(action_names),
            action_scales=np.asarray(action_scales, dtype=np.float32),
            body_names=body_names,
            foot_names=foot_names,
            dof_names=dof_names,
            actuated_dof_names=actuated_dof_names,
            robot_mass_kg=np.float32(robot_mass),
            link_mass_kg=link_mass,
            link_inertia_diagonal=link_inertia,
            contact_friction=np.float32(contact_friction),
            p_gains=p_gains,
            d_gains=d_gains,
            torque_limits=torque_limits,
            default_dof_position=default_dof_position,
            joint_damping=np.asarray(env.cfg.asset.joint_damping, dtype=np.float32),
            rotor_inertia=np.asarray(env.cfg.asset.rotor_inertia, dtype=np.float32),
            dof_acceleration=dof_acceleration,
            rigid_body_linear_acceleration=rigid_body_linear_acceleration,
            contact_impulse=contact_impulse,
            total_contact_impulse=total_contact_impulse,
            **samples,
            **transitions,
        )
    temporary.replace(artifact_path)
    print(f"wrote {artifact_path}")


def load_artifact(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def validate_protocol(reference, candidate):
    for key in (
        "schema_version",
        "task",
        "checkpoint_sha256",
        "checkpoint_iteration",
        "original_cfg",
        "inference_mode",
        "actions_applied",
        "motors_enabled",
        "domain_randomization",
        "seed",
        "command",
        "duration_s",
        "ctrl_dt",
        "sim_dt",
        "control_frequency_hz",
        "simulation_frequency_hz",
        "decimation",
        "mujoco_solref",
        "time",
        "transition_time",
        "actor_observation_fields",
        "actor_observation_names",
        "actor_observation_scales",
        "action_fields",
        "action_names",
        "action_scales",
        "body_names",
        "foot_names",
        "dof_names",
        "actuated_dof_names",
    ):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"backend artifacts differ in {key!r}")
    if bool(reference["actions_applied"]):
        raise ValueError("reference artifact applied policy actions")
    if np.any(reference["environment_actions"]):
        raise ValueError("reference environment action buffers are nonzero")
    if np.any(candidate["environment_actions"]):
        raise ValueError("candidate environment action buffers are nonzero")


def rmse_over_time(reference, candidate):
    difference = reference - candidate
    components = tuple(range(1, difference.ndim))
    return np.sqrt(np.mean(np.square(difference), axis=components))


def _top_component_rows(reference, candidate, names, step, limit=10):
    difference = np.abs(reference[step] - candidate[step])
    order = np.argsort(difference)[::-1][:limit]
    return [
        {
            "name": str(names[index]),
            "reference": float(reference[step, index]),
            "candidate": float(candidate[step, index]),
            "absolute_difference": float(difference[index]),
        }
        for index in order
    ]


def _first_active_step(values):
    active = np.flatnonzero(values)
    return None if len(active) == 0 else int(active[0])


def comparison_summary(reference, candidate):
    feet = [str(name) for name in reference["foot_names"]]
    body_names = [str(name) for name in reference["body_names"]]
    foot_indices = [body_names.index(name) for name in feet]
    observation_rmse = rmse_over_time(
        reference["actor_observations"], candidate["actor_observations"]
    )
    action_rmse = rmse_over_time(
        reference["queried_actions"], candidate["queried_actions"]
    )
    contact_rmse = rmse_over_time(
        reference["contact_forces"], candidate["contact_forces"]
    )
    foot_contact_rmse = rmse_over_time(
        reference["contact_forces"][:, foot_indices],
        candidate["contact_forces"][:, foot_indices],
    )
    dof_position_rmse = rmse_over_time(
        reference["dof_position"], candidate["dof_position"]
    )
    dof_velocity_rmse = rmse_over_time(
        reference["dof_velocity"], candidate["dof_velocity"]
    )
    root_state_rmse = rmse_over_time(reference["root_state"], candidate["root_state"])
    applied_torque_rmse = rmse_over_time(
        reference["applied_torque"], candidate["applied_torque"]
    )
    rigid_body_position_rmse = rmse_over_time(
        reference["rigid_body_state"][..., 0:3],
        candidate["rigid_body_state"][..., 0:3],
    )
    rigid_body_linear_velocity_rmse = rmse_over_time(
        reference["rigid_body_state"][..., 7:10],
        candidate["rigid_body_state"][..., 7:10],
    )
    dof_acceleration_rmse = rmse_over_time(
        reference["dof_acceleration"], candidate["dof_acceleration"]
    )
    total_contact_impulse_difference = np.linalg.vector_norm(
        reference["total_contact_impulse"] - candidate["total_contact_impulse"],
        axis=1,
    )
    contact_body_difference = np.sqrt(
        np.mean(
            np.square(reference["contact_forces"] - candidate["contact_forces"]),
            axis=2,
        )
    )
    first_post_step = 1
    reference_contact_step = _first_active_step(
        np.max(np.linalg.vector_norm(reference["contact_forces"], axis=2), axis=1)
        > CONTACT_ACTIVE_N
    )
    candidate_contact_step = _first_active_step(
        np.max(np.linalg.vector_norm(candidate["contact_forces"], axis=2), axis=1)
        > CONTACT_ACTIVE_N
    )
    if reference_contact_step is None or candidate_contact_step is None:
        raise RuntimeError("both backend trajectories must reach contact")
    first_contact_step = min(reference_contact_step, candidate_contact_step)
    reference_termination_step = _first_active_step(reference["terminated"])
    candidate_termination_step = _first_active_step(candidate["terminated"])
    if reference_termination_step is None or candidate_termination_step is None:
        raise RuntimeError("both backend trajectories must terminate")
    first_termination_step = min(
        step
        for step in (reference_termination_step, candidate_termination_step)
        if step is not None
    )
    pre_termination = slice(0, first_termination_step)
    body_order = np.argsort(contact_body_difference[first_contact_step])[::-1][:10]
    return {
        "protocol": {
            "checkpoint_sha256": str(reference["checkpoint_sha256"]),
            "checkpoint_iteration": int(reference["checkpoint_iteration"]),
            "duration_s": float(reference["duration_s"]),
            "ctrl_dt": float(reference["ctrl_dt"]),
            "command": reference["command"].tolist(),
            "actions_applied": bool(reference["actions_applied"]),
            "motors_enabled": bool(reference["motors_enabled"]),
            "control_frequency_hz": float(reference["control_frequency_hz"]),
            "simulation_frequency_hz": float(reference["simulation_frequency_hz"]),
            "decimation": int(reference["decimation"]),
        },
        "robot_mass_kg": {
            "vsim": float(reference["robot_mass_kg"]),
            "mujoco": float(candidate["robot_mass_kg"]),
        },
        "static_model": {
            "link_mass_rmse_kg": float(
                rmse_over_time(
                    reference["link_mass_kg"][None],
                    candidate["link_mass_kg"][None],
                )[0]
            ),
            "link_mass_max_abs_kg": float(
                np.max(np.abs(reference["link_mass_kg"] - candidate["link_mass_kg"]))
            ),
            # Principal-axis order is backend-specific. Sorting preserves the
            # three principal moments while removing a pure axis permutation.
            "sorted_principal_inertia_rmse_kg_m2": float(
                rmse_over_time(
                    np.sort(reference["link_inertia_diagonal"], axis=1)[None],
                    np.sort(candidate["link_inertia_diagonal"], axis=1)[None],
                )[0]
            ),
        },
        "initial": {
            "observation_rmse": float(observation_rmse[0]),
            "observation_max_abs": float(
                np.max(
                    np.abs(
                        reference["actor_observations"][0]
                        - candidate["actor_observations"][0]
                    )
                )
            ),
            "action_rmse": float(action_rmse[0]),
            "action_max_abs": float(
                np.max(
                    np.abs(
                        reference["queried_actions"][0]
                        - candidate["queried_actions"][0]
                    )
                )
            ),
            "contact_rmse_n": float(contact_rmse[0]),
            "contact_max_abs_n": float(
                np.max(
                    np.abs(
                        reference["contact_forces"][0] - candidate["contact_forces"][0]
                    )
                )
            ),
        },
        "first_post_step": {
            "time_s": float(reference["time"][first_post_step]),
            "observation_rmse": float(observation_rmse[first_post_step]),
            "action_rmse": float(action_rmse[first_post_step]),
            "contact_rmse_n": float(contact_rmse[first_post_step]),
            "foot_contact_rmse_n": float(foot_contact_rmse[first_post_step]),
            "dof_position_rmse_rad": float(dof_position_rmse[first_post_step]),
            "dof_velocity_rmse_rad_s": float(dof_velocity_rmse[first_post_step]),
            "root_state_rmse": float(root_state_rmse[first_post_step]),
            "rigid_body_position_rmse_m": float(
                rigid_body_position_rmse[first_post_step]
            ),
            "rigid_body_linear_velocity_rmse_m_s": float(
                rigid_body_linear_velocity_rmse[first_post_step]
            ),
            "dof_acceleration_rmse_rad_s2": float(
                dof_acceleration_rmse[first_post_step]
            ),
            "applied_torque_rmse_nm": float(applied_torque_rmse[0]),
        },
        "first_contact": {
            "vsim_time_s": float(reference["time"][reference_contact_step]),
            "mujoco_time_s": float(candidate["time"][candidate_contact_step]),
            "comparison_time_s": float(reference["time"][first_contact_step]),
            "comparison_step": first_contact_step,
            "observation_rmse": float(observation_rmse[first_contact_step]),
            "action_rmse": float(action_rmse[first_contact_step]),
            "contact_rmse_n": float(contact_rmse[first_contact_step]),
            "foot_contact_rmse_n": float(foot_contact_rmse[first_contact_step]),
            "dof_position_rmse_rad": float(dof_position_rmse[first_contact_step]),
            "dof_velocity_rmse_rad_s": float(dof_velocity_rmse[first_contact_step]),
            "rigid_body_position_rmse_m": float(
                rigid_body_position_rmse[first_contact_step]
            ),
            "rigid_body_linear_velocity_rmse_m_s": float(
                rigid_body_linear_velocity_rmse[first_contact_step]
            ),
            "dof_acceleration_rmse_rad_s2": float(
                dof_acceleration_rmse[first_contact_step]
            ),
            "preceding_applied_torque_rmse_nm": float(
                applied_torque_rmse[max(0, first_contact_step - 1)]
            ),
            "cumulative_contact_impulse_difference_ns": float(
                total_contact_impulse_difference[first_contact_step]
            ),
        },
        "first_termination": {
            "vsim_time_s": float(reference["time"][reference_termination_step]),
            "mujoco_time_s": float(candidate["time"][candidate_termination_step]),
        },
        "pre_termination_peak": {
            "observation_rmse": float(np.max(observation_rmse[pre_termination])),
            "action_rmse": float(np.max(action_rmse[pre_termination])),
            "contact_rmse_n": float(np.max(contact_rmse[pre_termination])),
            "foot_contact_rmse_n": float(np.max(foot_contact_rmse[pre_termination])),
            "dof_position_rmse_rad": float(np.max(dof_position_rmse[pre_termination])),
            "dof_velocity_rmse_rad_s": float(
                np.max(dof_velocity_rmse[pre_termination])
            ),
            "rigid_body_position_rmse_m": float(
                np.max(rigid_body_position_rmse[pre_termination])
            ),
            "rigid_body_linear_velocity_rmse_m_s": float(
                np.max(rigid_body_linear_velocity_rmse[pre_termination])
            ),
            "dof_acceleration_rmse_rad_s2": float(
                np.max(dof_acceleration_rmse[pre_termination])
            ),
            "applied_torque_rmse_nm": float(
                np.max(applied_torque_rmse[:first_termination_step])
            ),
            "cumulative_contact_impulse_difference_ns": float(
                np.max(total_contact_impulse_difference[pre_termination])
            ),
        },
        "peak": {
            "observation_rmse": float(np.max(observation_rmse)),
            "action_rmse": float(np.max(action_rmse)),
            "contact_rmse_n": float(np.max(contact_rmse)),
            "foot_contact_rmse_n": float(np.max(foot_contact_rmse)),
            "dof_position_rmse_rad": float(np.max(dof_position_rmse)),
            "dof_velocity_rmse_rad_s": float(np.max(dof_velocity_rmse)),
            "root_state_rmse": float(np.max(root_state_rmse)),
        },
        "final": {
            "observation_rmse": float(observation_rmse[-1]),
            "action_rmse": float(action_rmse[-1]),
            "contact_rmse_n": float(contact_rmse[-1]),
            "foot_contact_rmse_n": float(foot_contact_rmse[-1]),
            "dof_position_rmse_rad": float(dof_position_rmse[-1]),
            "dof_velocity_rmse_rad_s": float(dof_velocity_rmse[-1]),
            "root_state_rmse": float(root_state_rmse[-1]),
        },
        "largest_first_post_step_observation_differences": _top_component_rows(
            reference["actor_observations"],
            candidate["actor_observations"],
            reference["actor_observation_names"],
            first_post_step,
        ),
        "largest_first_post_step_action_differences": _top_component_rows(
            reference["queried_actions"],
            candidate["queried_actions"],
            reference["action_names"],
            first_post_step,
        ),
        "largest_first_contact_differences": [
            {
                "body": body_names[index],
                "vector_rmse_n": float(
                    contact_body_difference[first_contact_step, index]
                ),
            }
            for index in body_order
        ],
        "timeseries": {
            "time_s": reference["time"].tolist(),
            "observation_rmse": observation_rmse.tolist(),
            "action_rmse": action_rmse.tolist(),
            "contact_rmse_n": contact_rmse.tolist(),
            "foot_contact_rmse_n": foot_contact_rmse.tolist(),
            "dof_position_rmse_rad": dof_position_rmse.tolist(),
            "dof_velocity_rmse_rad_s": dof_velocity_rmse.tolist(),
            "root_state_rmse": root_state_rmse.tolist(),
            "rigid_body_position_rmse_m": rigid_body_position_rmse.tolist(),
            "rigid_body_linear_velocity_rmse_m_s": (
                rigid_body_linear_velocity_rmse.tolist()
            ),
            "dof_acceleration_rmse_rad_s2": dof_acceleration_rmse.tolist(),
            "transition_time_s": reference["transition_time"].tolist(),
            "applied_torque_rmse_nm": applied_torque_rmse.tolist(),
            "cumulative_contact_impulse_difference_ns": (
                total_contact_impulse_difference.tolist()
            ),
        },
    }


def _plot_first_values(output, reference, candidate, first_contact_step):
    time = reference["time"]
    observations = reference["actor_observation_names"]
    actions = reference["action_names"]
    body_names = reference["body_names"]
    for step, label in (
        (0, "initial"),
        (1, "first_post_step"),
        (first_contact_step, "first_contact"),
    ):
        figure, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
        axes[0].plot(reference["actor_observations"][step], label="VSim")
        axes[0].plot(
            candidate["actor_observations"][step], label="MuJoCo", linestyle="--"
        )
        axes[0].set_ylabel("normalized observation")
        axes[0].set_xlabel(f"actor component index (0..{len(observations) - 1})")
        axes[0].legend()

        x = np.arange(len(actions))
        width = 0.42
        axes[1].bar(
            x - width / 2,
            reference["queried_actions"][step],
            width,
            label="VSim",
        )
        axes[1].bar(
            x + width / 2,
            candidate["queried_actions"][step],
            width,
            label="MuJoCo",
        )
        axes[1].set_xticks(x, actions, rotation=55, ha="right", fontsize=8)
        axes[1].set_ylabel("queried policy action")
        axes[1].legend()

        axes[2].plot(reference["contact_forces"][step, :, 2], label="VSim", marker="o")
        axes[2].plot(
            candidate["contact_forces"][step, :, 2],
            label="MuJoCo",
            marker="x",
            linestyle="--",
        )
        axes[2].set_xticks(
            np.arange(len(body_names)), body_names, rotation=75, ha="right", fontsize=7
        )
        axes[2].set_ylabel("body contact force z [N]")
        axes[2].legend()
        for axis in axes:
            axis.grid(alpha=0.25)
        figure.suptitle(f"Values at t={time[step]:.2f} s")
        figure.savefig(output / f"{label}_values.png", dpi=180)
        plt.close(figure)


def _plot_contact_heatmaps(output, reference, candidate, first_contact_step):
    body_names = reference["body_names"]
    for step, label in (
        (0, "initial"),
        (1, "first_post_step"),
        (first_contact_step, "first_contact"),
    ):
        values = (
            reference["contact_forces"][step],
            candidate["contact_forces"][step],
            reference["contact_forces"][step] - candidate["contact_forces"][step],
        )
        limit = max(1.0, *(float(np.max(np.abs(value))) for value in values))
        figure, axes = plt.subplots(1, 3, figsize=(11, 12), constrained_layout=True)
        images = []
        for axis, value, title in zip(
            axes, values, ("VSim", "MuJoCo", "VSim - MuJoCo"), strict=True
        ):
            image = axis.imshow(
                value,
                aspect="auto",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            images.append(image)
            axis.set_title(title)
            axis.set_xticks((0, 1, 2), ("Fx", "Fy", "Fz"))
            axis.set_yticks(np.arange(len(body_names)), body_names, fontsize=7)
        figure.colorbar(images[-1], ax=axes, label="contact force [N]")
        figure.suptitle(
            f"All canonical body contact forces at t={reference['time'][step]:.2f} s"
        )
        figure.savefig(output / f"{label}_contact_forces.png", dpi=180)
        plt.close(figure)


def _plot_divergence(output, summary):
    series = summary["timeseries"]
    time = np.asarray(series["time_s"])
    figure, axes = plt.subplots(
        2, 2, figsize=(10, 7), sharex=True, constrained_layout=True
    )
    plots = (
        ("observation_rmse", "actor-observation RMSE"),
        ("action_rmse", "queried-action RMSE"),
        ("contact_rmse_n", "all-body contact-force RMSE [N]"),
        ("foot_contact_rmse_n", "foot contact-force RMSE [N]"),
    )
    for axis, (key, label) in zip(axes.flat, plots, strict=True):
        axis.plot(time, series[key], color="#c44e52")
        axis.axvline(
            summary["first_contact"]["comparison_time_s"],
            color="#4c9f70",
            linestyle=":",
            linewidth=1,
            label="first contact",
        )
        axis.axvline(
            summary["first_termination"]["mujoco_time_s"],
            color="#dd8452",
            linestyle="--",
            linewidth=1,
            label="MuJoCo termination",
        )
        axis.axvline(
            summary["first_termination"]["vsim_time_s"],
            color="#4c72b0",
            linestyle="--",
            linewidth=1,
            label="VSim termination",
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("time [s]")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Divergence with policy actions queried but not applied")
    figure.savefig(output / "divergence_over_time.png", dpi=180)
    plt.close(figure)


def _plot_foot_contacts(output, reference, candidate):
    body_names = [str(name) for name in reference["body_names"]]
    foot_names = [str(name) for name in reference["foot_names"]]
    time = reference["time"]
    figure, axes = plt.subplots(
        2, 2, figsize=(10, 7), sharex=True, constrained_layout=True
    )
    for axis, foot_name in zip(axes.flat, foot_names, strict=True):
        index = body_names.index(foot_name)
        axis.plot(time, reference["contact_forces"][:, index, 2], label="VSim")
        axis.plot(
            time,
            candidate["contact_forces"][:, index, 2],
            label="MuJoCo",
            linestyle="--",
        )
        axis.set_title(foot_name)
        axis.set_ylabel("vertical contact force [N]")
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("time [s]")
    axes[0, 0].legend()
    figure.suptitle("Foot contact forces during open-loop evolution")
    figure.savefig(output / "foot_contact_forces_over_time.png", dpi=180)
    plt.close(figure)


def _plot_dynamics_diagnostics(output, summary):
    series = summary["timeseries"]
    time = np.asarray(series["time_s"])
    transition_time = np.asarray(series["transition_time_s"])
    figure, axes = plt.subplots(
        3, 2, figsize=(11, 10), sharex=True, constrained_layout=True
    )
    plots = (
        (time, "rigid_body_position_rmse_m", "body-origin position RMSE [m]"),
        (
            time,
            "rigid_body_linear_velocity_rmse_m_s",
            "body-origin linear-velocity RMSE [m/s]",
        ),
        (time, "dof_acceleration_rmse_rad_s2", "joint-acceleration RMSE [rad/s²]"),
        (
            transition_time,
            "applied_torque_rmse_nm",
            "applied-torque RMSE [N m]",
        ),
        (
            time,
            "cumulative_contact_impulse_difference_ns",
            "cumulative contact-impulse difference [N s]",
        ),
        (time, "root_state_rmse", "root-state RMSE [mixed units]"),
    )
    for axis, (x, key, label) in zip(axes.flat, plots, strict=True):
        axis.plot(x, series[key], color="#6a3d9a")
        axis.axvline(
            summary["first_contact"]["comparison_time_s"],
            color="#4c9f70",
            linestyle=":",
            linewidth=1,
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("time [s]")
    figure.suptitle("Dynamics diagnostics at one sample per physics step")
    figure.savefig(output / "dynamics_diagnostics.png", dpi=180)
    plt.close(figure)


def _plot_static_model(output, reference, candidate):
    names = reference["body_names"]
    mass_difference = reference["link_mass_kg"] - candidate["link_mass_kg"]
    reference_inertia = np.sort(reference["link_inertia_diagonal"], axis=1)
    candidate_inertia = np.sort(candidate["link_inertia_diagonal"], axis=1)
    inertia_difference = np.max(np.abs(reference_inertia - candidate_inertia), axis=1)
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    axes[0].bar(names, mass_difference, color="#4c72b0")
    axes[0].set_ylabel("VSim − MuJoCo mass [kg]")
    axes[1].bar(names, inertia_difference, color="#dd8452")
    axes[1].set_ylabel("maximum principal-moment difference [kg m²]")
    for axis in axes:
        axis.tick_params(axis="x", rotation=75, labelsize=7)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Effective per-link inertial-property differences")
    figure.savefig(output / "static_model_differences.png", dpi=180)
    plt.close(figure)


def _markdown_rows(rows, key):
    lines = [
        "| component | VSim | MuJoCo | absolute difference |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row[key]} | {row.get('reference', 0.0):.5g} | "
            f"{row.get('candidate', 0.0):.5g} | "
            f"{row.get('absolute_difference', row.get('vector_rmse_n', 0.0)):.5g} |"
        )
    return "\n".join(lines)


def write_report(output, summary):
    initial = summary["initial"]
    first = summary["first_post_step"]
    first_contact = summary["first_contact"]
    first_termination = summary["first_termination"]
    pre_termination_peak = summary["pre_termination_peak"]
    peak = summary["peak"]
    final = summary["final"]
    static_model = summary["static_model"]
    observation_rows = summary["largest_first_post_step_observation_differences"]
    action_rows = summary["largest_first_post_step_action_differences"]
    contact_rows = summary["largest_first_contact_differences"]
    contact_table = "\n".join(
        [
            "| body | vector RMSE [N] |",
            "|---|---:|",
            *[
                f"| {row['body']} | {row['vector_rmse_n']:.5g} |"
                for row in contact_rows
            ],
        ]
    )
    rollout_rows = "\n".join(
        f"| {label} | {peak[key]:.5g} | {final[key]:.5g} |"
        for label, key in (
            ("Actor-observation RMSE", "observation_rmse"),
            ("Queried-action RMSE", "action_rmse"),
            ("All-body contact-force RMSE [N]", "contact_rmse_n"),
            ("Foot contact-force RMSE [N]", "foot_contact_rmse_n"),
            ("Joint-position RMSE [rad]", "dof_position_rmse_rad"),
            ("Joint-velocity RMSE [rad/s]", "dof_velocity_rmse_rad_s"),
        )
    )
    report = f"""# VSim / MuJoCo unapplied-policy comparison

The same policy is queried at each control step, but its actions are never
copied into the environment action buffers. Physics is sampled at the native
simulation frequency with one physics step per control step.

## Protocol

- Checkpoint iteration: {summary["protocol"]["checkpoint_iteration"]}
- Checkpoint SHA-256: `{summary["protocol"]["checkpoint_sha256"]}`
- Duration: {summary["protocol"]["duration_s"]:.3g} s
- Control timestep: {summary["protocol"]["ctrl_dt"]:.3g} s
- Control/simulation frequency: {summary["protocol"]["control_frequency_hz"]:.3g}
  / {summary["protocol"]["simulation_frequency_hz"]:.3g} Hz
- Decimation: {summary["protocol"]["decimation"]}
- Command: {summary["protocol"]["command"]}
- Policy actions applied: {summary["protocol"]["actions_applied"]}
- Motors enabled: {summary["protocol"]["motors_enabled"]}
- Robot mass: VSim {summary["robot_mass_kg"]["vsim"]:.6g} kg, MuJoCo
  {summary["robot_mass_kg"]["mujoco"]:.6g} kg

## Effective model comparison

- Per-link mass RMSE: {static_model["link_mass_rmse_kg"]:.6g} kg
- Maximum per-link mass difference: {static_model["link_mass_max_abs_kg"]:.6g} kg
- Sorted principal-moment RMSE:
  {static_model["sorted_principal_inertia_rmse_kg_m2"]:.6g} kg m²

## Initial state, t=0

- Actor-observation RMSE: {initial["observation_rmse"]:.6g}
- Maximum actor-observation difference: {initial["observation_max_abs"]:.6g}
- Queried-action RMSE: {initial["action_rmse"]:.6g}
- Maximum queried-action difference: {initial["action_max_abs"]:.6g}
- All-body contact-force RMSE: {initial["contact_rmse_n"]:.6g} N
- Maximum contact-force component difference: {initial["contact_max_abs_n"]:.6g} N

## First post-step state, t={first["time_s"]:.3g} s

- Actor-observation RMSE: {first["observation_rmse"]:.6g}
- Queried-action RMSE: {first["action_rmse"]:.6g}
- All-body contact-force RMSE: {first["contact_rmse_n"]:.6g} N
- Foot contact-force RMSE: {first["foot_contact_rmse_n"]:.6g} N
- Joint-position RMSE: {first["dof_position_rmse_rad"]:.6g} rad
- Joint-velocity RMSE: {first["dof_velocity_rmse_rad_s"]:.6g} rad/s
- Rigid-body position RMSE: {first["rigid_body_position_rmse_m"]:.6g} m
- Rigid-body origin-velocity RMSE:
  {first["rigid_body_linear_velocity_rmse_m_s"]:.6g} m/s
- Joint-acceleration RMSE: {first["dof_acceleration_rmse_rad_s2"]:.6g} rad/s²
- Applied-torque RMSE: {first["applied_torque_rmse_nm"]:.6g} N m

No body is in contact at t=0 or t={first["time_s"]:.3g} s.

## First contact

- First contact time: VSim {first_contact["vsim_time_s"]:.3g} s, MuJoCo
  {first_contact["mujoco_time_s"]:.3g} s
- Actor-observation RMSE: {first_contact["observation_rmse"]:.6g}
- Queried-action RMSE: {first_contact["action_rmse"]:.6g}
- All-body contact-force RMSE: {first_contact["contact_rmse_n"]:.6g} N
- Foot contact-force RMSE: {first_contact["foot_contact_rmse_n"]:.6g} N
- Joint-position RMSE: {first_contact["dof_position_rmse_rad"]:.6g} rad
- Joint-velocity RMSE: {first_contact["dof_velocity_rmse_rad_s"]:.6g} rad/s
- Joint-acceleration RMSE: {first_contact["dof_acceleration_rmse_rad_s2"]:.6g}
  rad/s²
- Preceding applied-torque RMSE:
  {first_contact["preceding_applied_torque_rmse_nm"]:.6g} N m
- Cumulative contact-impulse difference:
  {first_contact["cumulative_contact_impulse_difference_ns"]:.6g} N s
- First termination: VSim {first_termination["vsim_time_s"]:.3g} s, MuJoCo
  {first_termination["mujoco_time_s"]:.3g} s

## Peak divergence before either robot terminates

- Actor-observation RMSE: {pre_termination_peak["observation_rmse"]:.6g}
- Queried-action RMSE: {pre_termination_peak["action_rmse"]:.6g}
- All-body contact-force RMSE: {pre_termination_peak["contact_rmse_n"]:.6g} N
- Foot contact-force RMSE: {pre_termination_peak["foot_contact_rmse_n"]:.6g} N
- Joint-position RMSE: {pre_termination_peak["dof_position_rmse_rad"]:.6g} rad
- Joint-velocity RMSE: {pre_termination_peak["dof_velocity_rmse_rad_s"]:.6g} rad/s
- Rigid-body position RMSE: {pre_termination_peak["rigid_body_position_rmse_m"]:.6g}
  m
- Rigid-body linear-velocity RMSE:
  {pre_termination_peak["rigid_body_linear_velocity_rmse_m_s"]:.6g} m/s
- Joint-acceleration RMSE:
  {pre_termination_peak["dof_acceleration_rmse_rad_s2"]:.6g} rad/s²
- Applied-torque RMSE: {pre_termination_peak["applied_torque_rmse_nm"]:.6g} N m
- Cumulative contact-impulse difference:
  {pre_termination_peak["cumulative_contact_impulse_difference_ns"]:.6g} N s

## Rollout divergence

| quantity | peak | final |
|---|---:|---:|
{rollout_rows}

## Largest first post-step observation differences

{_markdown_rows(observation_rows, "name")}

## Largest first post-step action differences

{_markdown_rows(action_rows, "name")}

## Largest first-contact body-force differences

{contact_table}

## Plots

- `initial_values.png`: observations, queried actions, and vertical contact
  forces at t=0.
- `first_post_step_values.png`: the same values after one control step.
- `initial_contact_forces.png` and `first_post_step_contact_forces.png`: full
  canonical-body XYZ contact tensors.
- `first_contact_values.png` and `first_contact_contact_forces.png`: values at
  the first collision-bearing control sample.
- `divergence_over_time.png`: observation, action, and contact-force RMSE.
- `foot_contact_forces_over_time.png`: per-foot vertical-force trajectories.
- `dynamics_diagnostics.png`: body kinematics, acceleration, applied torque,
  contact impulse, and root-state divergence.
- `static_model_differences.png`: canonical per-link mass and principal-moment
  differences.
"""
    (output / "report.md").write_text(report)


def compare_artifacts(output):
    reference = load_artifact(output / "vsim.npz")
    candidate = load_artifact(output / "mujoco.npz")
    validate_protocol(reference, candidate)
    summary = comparison_summary(reference, candidate)
    (output / "comparison.json").write_text(json.dumps(summary, indent=2))
    first_contact_step = summary["first_contact"]["comparison_step"]
    _plot_first_values(output, reference, candidate, first_contact_step)
    _plot_contact_heatmaps(output, reference, candidate, first_contact_step)
    _plot_divergence(output, summary)
    _plot_foot_contacts(output, reference, candidate)
    _plot_dynamics_diagnostics(output, summary)
    _plot_static_model(output, reference, candidate)
    write_report(output, summary)
    return summary


def run_comparison(args):
    checkpoint = Path(resolve_ckpt(args.checkpoint)).resolve()
    args.checkpoint = checkpoint
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    for backend in BACKENDS:
        artifact = args.output / f"{backend.label}.npz"
        if args.resume and artifact.is_file():
            print(f"reusing {artifact}")
            continue
        command = cell_command(args, backend)
        print(" ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    summary = compare_artifacts(args.output)
    print(
        "initial observation RMSE "
        f"{summary['initial']['observation_rmse']:.6g}; first post-step "
        f"{summary['first_post_step']['observation_rmse']:.6g}"
    )
    print(f"wrote {args.output / 'report.md'}")


def main():
    args = get_args()
    if args.duration <= 0.0:
        raise ValueError("duration must be positive")
    if args.cell is None:
        run_comparison(args)
    else:
        run_cell(args)


if __name__ == "__main__":
    main()
