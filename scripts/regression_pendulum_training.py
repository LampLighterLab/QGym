"""Calibrate or check real pendulum learning at 100 Hz.

Run one backend/seed per process, without overlapping GPU jobs:

    uv run --frozen -m scripts.regression_pendulum_training \
        --backend mujoco --device cuda:0 --output logs/streamlining/warp_seed7
    uv run --frozen --env-file .env.vsim -m scripts.regression_pendulum_training \
        --backend vsim --device cuda:0 --output logs/streamlining/vsim_seed7

The default 400-update calibration evaluates checkpoints 0/100/200/400, then
checks save/load and one additional optimizer update. Add --require-learning
only when using a frozen regression profile. Calibration records the proposed
80% catch / 50 percentage-point improvement targets without claiming they pass.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import torch

from gym.envs.pendulum.pendulum_config import PendulumCfg, PendulumRunnerCfg
from gym.utils.helpers import class_to_dict, randomize_episode_counters, set_seed
from gym.utils.task_registry import task_registry
from learning.modules import Actor, Critic
from scripts import train


ROOT = Path(__file__).resolve().parents[1]
FREQUENCY_HZ = 100
CATCH_ANGLE = 0.14
CATCH_VELOCITY = 0.5


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mujoco", "vsim"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--rollout-size", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--gradient-steps", type=int, default=24)
    parser.add_argument("--eval-envs", type=int, default=256)
    parser.add_argument("--eval-seconds", type=float, default=10.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--require-learning", action="store_true")
    parser.add_argument("--minimum-catch-rate", type=float, default=0.8)
    parser.add_argument("--minimum-improvement", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        min(
            args.iterations,
            args.num_envs,
            args.rollout_size,
            args.batch_size,
            args.gradient_steps,
            args.cpu_threads,
            args.eval_envs,
        )
        <= 0
    ):
        parser.error(
            "iteration, environment, sample and thread counts must be positive"
        )
    if args.rollout_size % args.num_envs or args.rollout_size % args.batch_size:
        parser.error("rollout-size must be divisible by num-envs and batch-size")
    if math.isqrt(args.eval_envs) ** 2 != args.eval_envs:
        parser.error("eval-envs must be a perfect square for the fixed grid")
    if args.seed < 0 or args.eval_seconds < 1:
        parser.error("seed must be nonnegative and evaluation must last at least 1s")
    return args


def configure(args):
    """One resolved profile is used for training, evaluation, and resume."""
    cfg, runner_cfg = PendulumCfg(), PendulumRunnerCfg()
    cfg.seed = runner_cfg.seed = args.seed
    cfg.env.num_envs = args.num_envs
    cfg.control.ctrl_frequency = FREQUENCY_HZ
    cfg.control.desired_sim_frequency = FREQUENCY_HZ
    cfg.domain_randomization.startup.contact_friction_range = None
    cfg.domain_randomization.startup.link_mass_scale_range = None
    cfg.domain_randomization.episode.scale_ranges = {}
    runner_cfg.actor.frequency = FREQUENCY_HZ
    runner_cfg.actor.normalize_obs = runner_cfg.critic.normalize_obs = False
    runner_cfg.actor.noise.dof_pos_obs = 0.0
    runner_cfg.actor.noise.dof_vel = 0.0
    runner_cfg.algorithm.rollout_size = args.rollout_size
    runner_cfg.algorithm.batch_size = args.batch_size
    runner_cfg.algorithm.max_gradient_steps = args.gradient_steps
    runner_cfg.algorithm.discount_horizon = 0.8
    runner_cfg.algorithm.GAE_bootstrap_horizon = 2.0
    runner_cfg.runner.max_iterations = args.iterations
    runner_cfg.runner.save_interval = 100
    runner_cfg.runner.device = args.device
    runner_cfg.runner.resume = False
    task_registry.convert_frequencies_to_params(cfg, runner_cfg)
    return cfg, runner_cfg


def initial_state(seed, runner_cfg):
    """CPU construction decouples initial weights from backend/task RNG use."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        actor = Actor(3, 1, **class_to_dict(runner_cfg.actor))
        critic = Critic(3, **class_to_dict(runner_cfg.critic))
    return {"actor": actor.state_dict(), "critic": critic.state_dict()}


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def reset_training_state(env, seed):
    set_seed(seed)
    env.tau_ff.zero_()
    env.reset()
    randomize_episode_counters(env)


def probe_observations(device):
    angles = torch.linspace(-torch.pi, torch.pi, 16, device=device)
    velocity = torch.linspace(-5, 5, 16, device=device)
    state = torch.cartesian_prod(angles, velocity)
    return torch.stack((state[:, 0].sin(), state[:, 0].cos(), state[:, 1] / 5), dim=1)


def finite_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    for name in ("actor_state_dict", "critic_state_dict"):
        for key, tensor in checkpoint[name].items():
            assert torch.isfinite(tensor).all(), f"{path}: nonfinite {name}.{key}"
    for name in ("optimizer_state_dict", "critic_optimizer_state_dict"):
        for parameter, state in checkpoint[name]["state"].items():
            for key, tensor in state.items():
                assert torch.isfinite(tensor).all(), (
                    f"{path}: nonfinite {name}.{parameter}.{key}"
                )
    return checkpoint


def checkpoint_iterations(iterations):
    return sorted({0, iterations, *(i for i in (100, 200, 400) if i <= iterations)})


def catch_metrics(theta, omega, dt):
    """Physical catch definition shared with pendulum_fidelity's final-second gate.

    Arrays include the exact initial state followed by post-step states. Catch
    time is the start of the final uninterrupted upright interval, not a
    controller-specific mode switch.
    """
    wrapped = np.arctan2(np.sin(theta), np.cos(theta))
    good = (np.abs(wrapped) < CATCH_ANGLE) & (np.abs(omega) < CATCH_VELOCITY)
    caught = good[-round(1 / dt) :].all(axis=0)
    last_bad = np.where(~good, np.arange(len(good))[:, None], -1).max(axis=0)
    catch_time = np.where(caught, (last_bad + 1) * dt, np.nan)
    return (
        {
            "catch_rate": float(caught.mean()),
            "caught": int(caught.sum()),
            "catch_time_mean_s": float(catch_time[caught].mean())
            if caught.any()
            else None,
            "final_angle_rmse_rad": float(np.sqrt(np.square(wrapped[-1]).mean())),
            "final_velocity_rmse_rad_s": float(np.sqrt(np.square(omega[-1]).mean())),
        },
        caught,
        catch_time,
    )


@torch.no_grad()
def evaluate(runner, seconds, output):
    env = runner.env
    runner.switch_to_eval()
    env.tau_ff.zero_()
    env._reset_idx(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    env._reset_buffers()
    n_steps = round(seconds / env.dt)
    state = torch.empty(n_steps + 1, env.num_envs, 2, device=env.device)
    state[0] = env.dof_state.view(env.num_envs, 2)
    raw_actions = torch.empty(n_steps, env.num_envs, device=env.device)
    torques = torch.empty_like(raw_actions)
    values = torch.empty_like(raw_actions)
    weights = runner.critic_cfg["reward"]["weights"]
    rewards = torch.empty(n_steps, env.num_envs, len(weights), device=env.device)
    for step in range(n_steps):
        actions = runner.get_inference_actions()
        raw_actions[step] = actions[:, 0]
        values[step] = runner.alg.critic.evaluate(
            runner.get_obs(runner.critic_cfg["obs"])
        )
        runner.set_actions(runner.actor_cfg["actions"], actions)
        env.step()
        state[step + 1] = env.dof_state.view(env.num_envs, 2)
        torques[step] = env.torques[:, 0]
        for column, name in enumerate(weights):
            rewards[step, :, column] = runner.reward_functions[name]()
    for name, tensor in (
        ("state", state),
        ("raw_actions", raw_actions),
        ("torques", torques),
        ("values", values),
        ("rewards", rewards),
    ):
        assert torch.isfinite(tensor).all(), f"nonfinite evaluation {name}"
    assert not env.timed_out.any(), "evaluation must not cross an episode boundary"
    arrays = {
        "state": state.cpu().numpy(),
        "raw_actions": raw_actions.cpu().numpy(),
        "applied_torques": torques.cpu().numpy(),
        "values": values.cpu().numpy(),
        "reward_terms": rewards.cpu().numpy(),
    }
    metrics, caught, catch_time = catch_metrics(
        arrays["state"][..., 0], arrays["state"][..., 1], env.dt
    )
    limit = env.torque_limits[0].item()
    metrics.update(
        raw_action_peak=float(raw_actions.abs().max()),
        applied_torque_peak_nm=float(torques.abs().max()),
        torque_saturation_fraction=float(
            (torques.abs() >= limit - 1e-6).float().mean()
        ),
        value_peak=float(values.abs().max()),
        per_term_mean={
            name: float(rewards[..., column].mean())
            for column, name in enumerate(weights)
        },
        weighted_reward_mean=float(
            sum(
                weight * rewards[..., column].mean()
                for column, weight in enumerate(weights.values())
            )
        ),
        all_finite=True,
    )
    np.savez_compressed(
        output,
        **arrays,
        caught=caught,
        catch_time_s=catch_time,
        reward_names=np.asarray(list(weights)),
        reward_weights=np.asarray(list(weights.values())),
        dt=env.dt,
        torque_limit_nm=limit,
        checkpoint_iteration=runner.it,
    )
    return metrics


def restore_and_check(runner, checkpoint_path, expected_actions):
    checkpoint = finite_checkpoint(checkpoint_path, runner.device)
    runner.load(checkpoint_path, load_optimizer=True)
    for key, current in (
        ("actor_state_dict", runner.alg.actor.state_dict()),
        ("critic_state_dict", runner.alg.critic.state_dict()),
        ("optimizer_state_dict", runner.alg.optimizer.state_dict()),
        ("critic_optimizer_state_dict", runner.alg.critic_optimizer.state_dict()),
    ):
        torch.testing.assert_close(current, checkpoint[key], rtol=0, atol=0)
    runner.switch_to_eval()
    with torch.no_grad():
        actual = runner.alg.actor.act_inference(probe_observations(runner.device))
    torch.testing.assert_close(actual.cpu(), expected_actions, rtol=1e-6, atol=1e-7)
    assert runner.it == checkpoint["iter"]


def summarize_vitals(path, rollout_size):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    # Episodic averages can be NaN before any episode ends. Optimization and
    # policy statistics must be finite from the first completed update.
    for record in records:
        for key, value in record.items():
            if key.startswith(("algorithm/", "actor/")):
                assert math.isfinite(value), (
                    f"nonfinite iteration {record['iteration']} {key}"
                )
    steady = records[10:]
    return {
        "iterations": len(records),
        "collected_samples": len(records) * rollout_size,
        "warmup_iterations_excluded": 10,
        "steady_median_s": {
            key: statistics.median(record[key] for record in steady) if steady else None
            for key in ("t_collection", "t_learning")
        },
        "steady_collection_samples_per_s": (
            rollout_size
            / statistics.median(record["t_collection"] for record in steady)
            if steady
            else None
        ),
        "algorithm_actor_all_finite": True,
    }


def source_provenance():
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "gym",
            "learning",
            "scripts",
            "resources/robots/pendulum",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in sorted(paths)
        },
    }


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run(args):
    torch.set_num_threads(args.cpu_threads)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg, runner_cfg = configure(args)
    shared = initial_state(args.seed, runner_cfg)
    torch.save(shared, output / "initial_state.pt")
    manifest = {
        "schema_version": 1,
        "mode": "regression" if args.require_learning else "calibration",
        "backend": args.backend,
        "device": args.device,
        "seed": args.seed,
        "checkpoints": checkpoint_iterations(args.iterations),
        "env_cfg": class_to_dict(cfg),
        "runner_cfg": class_to_dict(runner_cfg),
        "evaluation": {
            "num_envs": args.eval_envs,
            "seconds": args.eval_seconds,
            "reset_mode": "reset_to_uniform",
            "hold_seconds": 1,
            "angle_bound_rad": CATCH_ANGLE,
            "velocity_bound_rad_s": CATCH_VELOCITY,
        },
        "acceptance": {
            "minimum_catch_rate": args.minimum_catch_rate,
            "minimum_improvement": args.minimum_improvement,
        },
        "initial_hashes": {name: tensor_hash(state) for name, state in shared.items()},
        "cpu_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gpu": torch.cuda.get_device_name(args.device)
        if args.device != "cpu"
        else None,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "engine_version": importlib.metadata.version(
            "vlearn" if args.backend == "vsim" else "mujoco"
        ),
        "source": source_provenance(),
    }
    if args.backend == "mujoco" and args.device != "cpu":
        manifest["mujoco_warp_version"] = importlib.metadata.version("mujoco-warp")
    write_json(output / "manifest.json", manifest)
    train_args = train.get_train_args(
        [
            "--task",
            "pendulum",
            "--backend",
            args.backend,
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--headless",
            "--disable_wandb",
            "--experiment_name",
            str(output / "training"),
        ]
    )
    start = time.perf_counter()
    runner_cfg, runner = train.setup(train_args, cfg, runner_cfg)
    setup_seconds = time.perf_counter() - start
    try:
        runner.alg.actor.load_state_dict(shared["actor"])
        runner.alg.critic.load_state_dict(shared["critic"])
        reset_training_state(runner.env, args.seed)
        np.savez_compressed(
            output / "training_initial_state.npz",
            dof_state=runner.env.dof_state.cpu().numpy(),
            episode_counters=runner.env.episode_length_buf.cpu().numpy(),
        )
        start = time.perf_counter()
        train.train(runner_cfg, runner)
        train_seconds = time.perf_counter() - start
        run_dir = Path(runner.log_dir)
        final_checkpoint = run_dir / f"model_{args.iterations}.pt"
        runner.switch_to_eval()
        with torch.no_grad():
            expected_actions = runner.alg.actor.act_inference(
                probe_observations(args.device)
            ).cpu()
    finally:
        runner.env._backend.close()

    evaluation_cfg = copy.deepcopy(cfg)
    evaluation_cfg.env.num_envs = args.eval_envs
    evaluation_cfg.env.episode_length_s = args.eval_seconds + 1
    evaluation_cfg.init_state.reset_mode = "reset_to_uniform"
    evaluation_runner_cfg = copy.deepcopy(runner_cfg)
    evaluation_runner_cfg.log_dir = None
    env = task_registry.make_env(
        "pendulum", evaluation_cfg, args.device, True, args.backend
    )
    evaluations = {}
    try:
        evaluator = task_registry.make_alg_runner(env, evaluation_runner_cfg)
        for iteration in manifest["checkpoints"]:
            checkpoint = run_dir / f"model_{iteration}.pt"
            finite_checkpoint(checkpoint, args.device)
            evaluator.load(checkpoint, load_optimizer=False)
            evaluations[str(iteration)] = evaluate(
                evaluator, args.eval_seconds, output / f"evaluation_{iteration}.npz"
            )
            evaluations[str(iteration)]["checkpoint_sha256"] = hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            write_json(output / "evaluations.json", evaluations)
    finally:
        env._backend.close()

    resume_cfg = copy.deepcopy(runner_cfg)
    resume_cfg.log_dir = str(output / "resume")
    resume_cfg.runner.max_iterations = 1
    env = task_registry.make_env("pendulum", cfg, args.device, True, args.backend)
    try:
        resumed = task_registry.make_alg_runner(env, resume_cfg)
        restore_and_check(resumed, final_checkpoint, expected_actions)
        reset_training_state(env, args.seed)
        train.train(resume_cfg, resumed)
        assert resumed.it == args.iterations + 1
        finite_checkpoint(output / "resume" / f"model_{resumed.it}.pt", args.device)
    finally:
        env._backend.close()

    final_rate = evaluations[str(args.iterations)]["catch_rate"]
    improvement = final_rate - evaluations["0"]["catch_rate"]
    learning_passed = (
        final_rate >= args.minimum_catch_rate
        and improvement >= args.minimum_improvement
    )
    summary = {
        "mode": manifest["mode"],
        "learning_passed": learning_passed,
        "catch_rate_improvement": improvement,
        "evaluations": evaluations,
        "run_dir": str(run_dir),
        "setup_seconds": setup_seconds,
        "training_seconds": train_seconds,
        "training": summarize_vitals(run_dir / "vitals.jsonl", args.rollout_size),
        "checkpoint_reload_and_resume_passed": True,
        "resumed_iteration": args.iterations + 1,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    if args.require_learning:
        assert learning_passed, (
            f"learning regression failed; see {output / 'summary.json'}"
        )
    return summary


if __name__ == "__main__":
    run(get_args())
