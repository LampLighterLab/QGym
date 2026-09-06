# Backend Progression and Parity Plan

## Current scope

Q2 supports three execution targets behind one task-facing contract:

- MuJoCo CPU for portable development, CI, and interactive playback.
- MuJoCo Warp for batched CUDA training.
- Optional licensed VSim for CUDA training and cross-engine evaluation.

The supported world is currently a flat ground plane. Periodic `push_robots`
disturbances remain part of legged-task behavior. Heightfields, trimeshes, and
projectiles are outside the current scope.

Domain randomization is returning as a new backend-neutral feature. The first
milestone—independent episode-level contact friction—is implemented for
MuJoCo CPU, MuJoCo Warp, and VSim. `DR.md` defines the semantics, evidence, and
remaining progression. The final speed matrix and backend-specific tests pass.
A deterministic MuJoCo 3.11 crash on a valid fallen Go2 pose was isolated to
general convex multi-contact CCD; Go2 disables that path and retains primitive
multi-point contacts, with the captured pose covered by a subprocess
regression. Do not reintroduce legacy engine-specific callbacks.

## Architecture

```text
SimBackend
├── MuJocoCPUBackend
├── MuJocoWarpBackend
└── VSimBackend

TaskSkeleton
└── BaseTask
    ├── FixedRobot
    └── LeggedRobot
        └── concrete tasks
```

`gym/envs/base/sim_backend.py` and its contract tests are the executable
specification. Tasks consume canonical tensors and named robot semantics;
engine-native ordering and representation stay behind each backend boundary.

Core invariants:

- Public state tensors are valid after setup and refreshed in place after
  every step and reset.
- `dof_pos` and `dof_vel` are writable views into persistent `dof_state`.
- Root and DOF state is written first and committed atomically through one
  backend reset operation.
- Task-facing quaternions are scalar-last `[x, y, z, w]`.
- Public DOFs, bodies, torques, contacts, and state tensors use canonical
  `RobotLayout` order.
- Contact forces are robot-body net collision forces in world coordinates and
  Newtons.

## Correctness gates

### Portable gate

The default suite combines pure unit behavior, MuJoCo CPU backend contracts,
task registration, and a one-action-step construction test for every declared
task:

```bash
uv run --frozen python -m pytest -q
```

Small deterministic implementation tests remain beside their code and run
explicitly:

```bash
uv run --frozen python -m pytest gym -q
uv run --frozen python -m pytest learning -q
```

### CUDA and licensed gates

```bash
uv run --frozen python -m pytest -q -m warp
bash scripts/run_vsim_tests.sh
```

A deselected or skipped hardware group is not passing evidence. Record the
device, backend, seed, environment count, and exact executed tests.

### Repository gate

```bash
uv run --frozen ruff check .
uv build --no-sources
```

GitHub CI runs the portable and colocated suites, Ruff, and the package build.
Smoke training and hardware-specific groups remain explicit local gates.

## Physics and parity evidence

### Pendulum

The analytic energy-pump plus LQR probe exercises a deterministic grid of
initial angles and velocities. MuJoCo CPU, MuJoCo Warp, and VSim all reached a
100% catch rate with a mean catch time of approximately 1.08 seconds in the
recorded 1024-environment campaign. Angular divergence RMS against MuJoCo CPU
was approximately `9e-7 rad` for Warp and `5e-4 rad` for VSim.

Use `scripts/pendulum_fidelity.py` to rerun the probe. Treat these figures as
recorded evidence, not permanent thresholds detached from the current code.

### Mini Cheetah reference tracking

MuJoCo CPU and Warp now agree closely through free flight and contact after the
root-reset ordering and contact refresh fixes. VSim agrees before contact and
has bounded solver-specific differences during contact.

Important invalid evidence remains visible:

- Warp checkpoints produced before the floating-base reset ordering fix are
  invalid. A DOF reset refreshed assembled root state before the pending root
  reset was committed, spawning robots at the wrong height.
- Early contact probes included an unintended initialization step before
  recording. Only probes that restore the requested state immediately before
  capture are comparable.
- A prior VSim contact path returned link-frame vector components despite an
  environment-frame request. The backend now rotates those components into
  world axes. A settled Mini Cheetah supports approximately its `81.3 N`
  weight along world `+z`.
- The last recorded cross-engine transfer campaign remained incomplete: a
  VSim-trained checkpoint failed on both MuJoCo targets even though the other
  tested transfer directions passed. Do not present that campaign as full
  policy parity.
- A later Go2 observation audit found that limited-joint resets were not
  equivalent: VSim projected an out-of-range calf position onto the URDF
  limit, while MuJoCo retained the invalid requested value until stepping.
  MuJoCo CPU and Warp now clamp limited scalar joints during reset and expose
  the applied value through the public state tensor. Pre-fix Go2 transfer
  artifacts are invalid for initial-observation parity. With the corrected
  reset, initial policy inputs and actions agree to numerical precision, but
  the rollout still diverges. The resulting audit also found two MuJoCo public
  rigid-body-state bugs: pose/velocity fields were one completed physics step
  stale, and COM-based `cvel` translation was published as the velocity of the
  body origin. Both CPU and Warp now refresh post-integration kinematics and
  publish body-origin velocity; pre-fix rigid-body trajectory plots are invalid.
  The corrected 500 Hz unapplied-policy probe queries the policy once per
  physics step but leaves both action buffers at zero. In its zero-torque mode,
  VSim and MuJoCo Warp remain near numerical precision through free fall (at
  `0.064 s`, rigid-body position RMSE is `6.0e-7 m` and joint-velocity RMSE is
  `3.8e-7 rad/s`). VSim first contacts at `0.066 s` and MuJoCo at `0.068 s`;
  at the earlier sample, foot-force RMSE is `129 N` and joint-velocity RMSE is
  `2.60 rad/s`. With the gait/PD controller active, both first contact at
  `0.056 s`, but foot-force RMSE is `76.0 N` despite only `0.064 N m` preceding
  applied-torque RMSE. A zero-torque timestep sweep at `250`, `500`, `1000`,
  and `2000 Hz` shows VSim responding exactly one physics step before MuJoCo
  in every case (a `4`, `2`, `1`, and `0.5 ms` lead). At `500 Hz`, the last
  common airborne state has collision-sphere clearance matched within
  `1.2e-6 m` and predicts impact `0.43 ms` into the next step. MuJoCo reaches
  `0.066 s` with `1.024 mm` penetration but no contact from that step's
  pre-integration collision stage; evaluating collision at the unchanged
  post-step state finds all four contacts. This rules out an imported-geometry
  threshold as the cause of the one-step onset lag and identifies different
  contact/integration staging. Onset-aligned force magnitudes still differ, so
  contact material/compliance and solver-response discriminators remain.
  An onset-aligned zero-torque fit selected MuJoCo `solref=[0.005, 1.0]`
  instead of its default `[0.02, 1.0]`. Over the first `32 ms` of contact at
  `500 Hz`, this reduces total foot-force RMSE from `202 N` to `65.6 N`,
  cumulative-impulse RMSE from `0.840 N s` to `0.152 N s`, joint-velocity
  RMSE from `1.21` to `0.275 rad/s`, and foot vertical-velocity RMSE from
  `0.252` to `0.0530 m/s`. With the gait/PD controller active, first-contact
  foot-force RMSE falls from `76.0 N` to `19.1 N`, and the MuJoCo termination
  moves from `0.516 s` to `0.600 s` versus VSim's `0.616 s`. At `1000 Hz`,
  the fitted setting also improves the `32 ms` cumulative-impulse RMSE from
  `0.962` to `0.357 N s` and joint-velocity RMSE from `1.38` to
  `0.547 rad/s`. The one-step contact-response lead remains by design; the
  fit changes the force response actually applied by MuJoCo and does not
  synthesize a post-step contact.
- A later VSim training run ended because the host GPU fell off the PCIe bus,
  not because the policy diverged. Its partial checkpoint is not promotion
  evidence.

Use `scripts/mini_cheetah_fidelity.py`, `scripts/eval_policy.py`,
`scripts/compare_backend_unapplied_actions.py`, and the checked-in benchmark
wrappers for new evidence. The paired Go2 artifact can be explored with
`notebooks/go2_backend_unapplied_actions.py`. Keep checkpoint selection,
commands, reset distribution, episode duration, and rollout geometry fixed
across transfer cells.

## Promotion criteria

A backend or policy is ready for promotion only when all applicable checks
pass:

1. Shared contract, liveness, reset, routing, contact, and task tests pass.
2. A policy-free invariant or fidelity probe explains physics agreement and
   any accepted cross-engine tolerance.
3. Short training remains finite and produces a valid checkpoint.
4. Deterministic evaluation passes the declared command and reset suite.
5. Cross-backend transfer results are reported for every intended direction;
   failed and invalid cells remain visible.
6. Environment count, timestep, rollout horizon, optimizer batch geometry,
   device, seed, warnings, and checkpoint-selection rule are recorded.

Aggregate reward or a viewer impression alone is not promotion evidence.

## Planned learning-stack pruning

The learning package currently mixes the supported PPO path with configured
research paths and code that no registered task can select. Pruning must follow
runtime reachability and an explicit support decision; an import or historical
checkpoint alone does not make a path supported.

### Inventory

| Status | Runtime chain | Current consumer |
|---|---|---|
| Core | `OnPolicyRunner` → `PPO2` → `Actor`/`SmoothActor` + `Critic` → `DictStorage` | All standard PPO tasks |
| Configured research | `OffPolicyRunner` → `SAC` → `ChimeraActor` + `Critic` → `ReplayBuffer` | `sac_pendulum`, `sac_mini_cheetah` |
| Configured research | `PSACRunner` → `SAC` → `ChimeraActor` + `DenseSpectralLatent` → `ReplayBuffer` | `psd_pendulum` |
| Unconfigured | `CustomCriticRunner`, `MyRunner`, `DataLoggingRunner` | Exported by `learning.runners`, but selected by no registered task |
| Legacy | `OldPolicyRunner` → deprecated `PPO` → `ActorCritic` → `RolloutStorage` | No registered task |
| Orphaned | `StateEstimator` → `StateEstimatorNN` → `SERolloutStorage` | No registered task |
| Partially reachable | Other classes in `QRCritics.py` | Only `DenseSpectralLatent` is selected by a registered task |

Environment-agnostic utilities are not removal candidates merely because a
runner does not call them. Keep normalization, logging, dictionary utilities,
and the colocated PBRS implementation/tutorial unless their own behavior or
tests show that they are obsolete.

### Pruning sequence

1. Keep the core PPO chain as the supported baseline. Add no compatibility
   aliases for removed selectable class names; registry/config failures should
   remain explicit.
2. Decide whether SAC and PSD-SAC are supported research features. For each of
   `sac_pendulum`, `sac_mini_cheetah`, and `psd_pendulum`, run a reduced CPU
   smoke that crosses replay-buffer initialization, performs updates, saves a
   checkpoint, resumes it, and produces finite deterministic inference. Keep
   and test a chain only if that evidence passes and the feature is still
   wanted; otherwise unregister its tasks and configs before deleting it.
3. Remove the definitely unreachable chains in separate reviewable changes:
   first `OldPolicyRunner`/`PPO`/`ActorCritic`/`RolloutStorage`, then the state
   estimator chain, then the three unconfigured runner variants. Update package
   exports and delete tests that exist only for a removed path in the same
   change.
4. After the SAC/PSD decision, reduce `QRCritics.py` to retained classes and
   their demonstrated dependencies. If PSD-SAC stays, give its selected critic
   focused math and checkpoint tests rather than preserving every historical
   critic architecture.
5. Normalize module names only after reachability is settled: for example,
   rename `BaseRunner.py` to `base_runner.py` and any retained custom-critic
   module to snake case. This avoids renaming code immediately before deleting
   it.
6. For every retained runner family, require a registered task, focused unit
   coverage, a tiny train/save/resume/inference smoke, and explicit inclusion
   in developer documentation. Finish each pruning slice with the portable,
   colocated, lint, and package-build gates.

Pruning is complete when every selectable runner and algorithm has a declared
consumer and evidence, every registered learning task uses a supported chain,
and the package no longer exports unreachable implementations.

## Domain randomization progression

1. **Complete:** backend-neutral seeded startup sampling for physical friction,
   link mass, and inertia. Native routing, full-task flow, and predicted
   physics invariants have been exercised on all three backend paths.
2. **Complete:** backend-neutral stiffness and damping scaling from stored
   nominal gains.
3. **Complete:** physically consistent startup link mass and inertia scaling,
   including derived engine constants and body-weight-dependent reward
   normalization.
4. Add measured delay, bias, and noise axes only with explicit schedules.
5. Keep motor-strength scaling low priority; if added, apply it to final torque
   rather than treating PD gain uncertainty as equivalent.
6. Compare nominal and randomized policies with identical rollout and
   evaluation geometry after each axis.

See `DR.md` for detailed semantics, backend topology, test gates, and the
literature survey.

## Common commands

```bash
# CPU smoke train
uv run --frozen scripts/train.py \
  --task pendulum \
  --backend mujoco \
  --device cpu \
  --num_envs 8 \
  --max_iterations 2 \
  --headless \
  --disable_wandb

# CUDA training
uv run --frozen scripts/train.py \
  --task mini_cheetah \
  --backend mujoco \
  --device cuda:0 \
  --headless

# Interactive CPU playback
uv run --frozen scripts/play.py --task mini_cheetah --device cpu
```
