# Backend Progression and Parity Plan

## Current scope

Q2 supports three execution targets behind one task-facing contract:

- MuJoCo CPU for portable development, CI, and interactive playback.
- MuJoCo Warp for batched CUDA training.
- Optional licensed VSim for CUDA training and cross-engine evaluation.

The supported world is currently a flat ground plane. Periodic `push_robots`
disturbances remain part of legged-task behavior. Heightfields, trimeshes, and
projectiles are outside the current scope.

Domain randomization implements startup contact friction and link mass/inertia,
plus episode-level PD-gain scaling on MuJoCo CPU, MuJoCo Warp, and VSim.
`DR.md` defines the semantics, evidence, and remaining progression. The angular-
velocity contract and CPU reset-state caches are corrected. The reduced 100 Hz
pilot completed its four GPU training cells, but CPU nominal training diverged
before iteration 500. A fixed-policy CPU probe isolated runaway previous-target
feedback; Go2Trot now bounds full position commands to joint limits. Task reset
observations are also refreshed immediately. Validate fresh CPU training and
the remaining transfer failures before expanding the campaign.
A deterministic MuJoCo 3.11 crash on a valid fallen Go2 pose was isolated to
general convex multi-contact CCD; Go2 disables that path and retains primitive
multi-point contacts, with the captured pose covered by a subprocess
regression. Do not reintroduce legacy engine-specific callbacks.

MuJoCo CPU and Warp now share a single explicit `cfg.mujoco` section for
constraint capacity, CCD settings, and contact `solref`. The registered tasks'
compiled settings are unchanged. Configs must use the unified section;
automatic migration of older saved-run `mjspec_*` sections is out of scope.

Go2 VSim conversion replaces unavailable DAE visuals with OBJ parts from the
local SDK before native conversion. All nonvisual URDF elements and visual
origins are preserved; the vendor's physical model is not adopted. XML/path
tests cover this without vendor binaries. Native VSim rendering still needs
verification on a CUDA-capable host.

## Main promotion preparation (2026-10-03)

The candidate is QGym's `bigred` branch at `ebcfd8b9`, plus the preparation
changes described here. The target is `LampLighterLab/QGym:main` at `f0424249`.
BigRedGym's classroom-only pruning is outside this promotion. Keep the research
evaluators, campaign/profiling tools, VSim launcher, and root developer guidance.
Existing mini-cheetah reward weights and the adaptive pendulum learning-rate
schedule are retained; this preparation does not change training or physics.

Preparation enables the portable CI gate on `bigred`/`cdx`, adds manual runs,
and makes file-size checks cover the full push/PR range. The required `uv.lock`
is explicitly exempt from the 100 KiB limit in CI and pre-commit. Cloud
notebooks now default to QGym `main` and use frozen installation without an
unlocked fallback. SDK setup instructions and the VSim launcher explicitly
select their locked optional dependencies. A focused zero-interval threading
regression restores coverage for the CSV writer's timer-free loop.

Fresh validation:

| Check | Result |
| --- | --- |
| Portable suite | 309 passed; one existing strict deployment-parity xfail |
| Colocated `gym` / `learning` | 32 / 6 passed |
| MuJoCo Warp | 31 passed; existing MULTICCD fixture warnings |
| Licensed VSim | 47 passed |
| Unitree SDK / threading | 17 passed, including the new zero-interval case |
| Fresh frozen install without vendor sources | Passed; portable suite also passed in that clone |
| Ruff / wheel and source-distribution build | Passed |
| CI configuration | YAML/shell checks and 12 file-size scenarios passed locally |

Unitree checks used the pinned SDK and real Cyclone DDS 0.10.2 in an isolated
temporary environment, with no robot connection or changes to the working
environment. Hosted CI still excludes the optional backend/SDK groups.

The portable suite exercises CPU PPO updates, checkpoint reload, and resume.
Separate Warp/VSim smokes each completed two updates and resumed through update
three, with finite model, inference, and evaluation tensors. They used seed 7,
16 environments, 100 Hz policy/control/physics, 128 rollout samples, 64-sample
minibatches, and two optimizer steps per update. Artifacts are under
`logs/main_promotion_20261003/`; these small smokes establish execution, not
learning quality. The failed learning calibration below remains unresolved.

### Merge handoff

`main` is not an ancestor of `bigred`. An isolated ordinary merge of the audited
`main` tip produced 11 content/add-add conflicts and five modify/delete conflicts
in legacy interfaces and tests. Main's relevant fixes are already represented
in the modern implementation. Resolving those conflicts with the existing
`bigred` files/deletions produced a tree **identical to `bigred`**, verified with
`git diff --cached --exit-code HEAD`. The fresh-install test above also ran on
that resolved tree.

After committing and pushing the preparation changes, reconcile `origin/main`
into `bigred`, preserving the modern implementations and intentional legacy
deletions, then promote through a normal PR. Recheck if either remote tip has
moved. Promotion to `main` remains a separate step; this preparation does not
change `main` or commit a merge.

Outstanding work remains separate: full-batch startup DR, deployment
residual-action observation parity (the explicit xfail), learning-quality
calibration, and the incomplete DR/transfer campaign. A passing source gate
does not close those research and deployment questions.

## Streamlining regression calibration

The 100 Hz regression push is tracked in `STREAMLINING_PLAN.md`; the behavior
and cost review is in `tests/TEST_CATALOG.md`. Pendulum now declares the PPO
runner's required `rollout_size`; unused PPO `storage_size` declarations and
the unsupported `collapse_fixed_joints` field are removed.

The first controlled pendulum calibration used seed 7, matching initial
actor/critic weights, 512 environments, 65,536 samples per update, and 400
updates on each GPU backend. Both engines stayed finite and passed fresh-runner
checkpoint/optimizer restoration plus update 401, but both caught and held only
38/256 evaluation starts (14.84%), against the proposed 80% target. This is
failed learning evidence, not a usable learning reference. Preserve the
artifacts under `logs/streamlining/pendulum/` and the offline diagnosis there.

Most reward improvement came from suppressing angular velocity. The saved
trajectories do not establish the known unwrapped-angle reward alias as the
main cause. Before changing rewards or lowering a threshold, compare the
existing analytic swing-up controller's physical success and discounted return
on the same initial grid. The shared learning failure does not invalidate the
separate Go2 DR campaign evidence. Simulation speed calibration and Warp/VSim
host/CUDA profiling proceed independently; broad learner refactoring waits for
a meaningful physical learning gate.

## Pendulum algorithm validation (2026-10-07)

Current control-learning implementations and their pendulum coverage:

| Algorithm / variant | Registered task and runner | Pendulum learning evidence |
|---|---|---|
| `PPO2` | `pendulum` / `OnPolicyRunner` | CPU swing-up and stabilization validated below |
| `SAC` | `sac_pendulum` / `OffPolicyRunner` | CPU swing-up/stabilization validated below with PPO2's physical task and rewards |
| SAC with PSD critic | `psd_pendulum` / `PSACRunner` / `DenseSpectralLatent` | Pending; shares the SAC update algorithm |
| `StateEstimator` | No registered task | Supervised state estimation, not a pendulum controller |

The PPO2 tuning followed scaling, initialization, reward weights, then
hyperparameters. All CPU comparisons used the user's 25 Hz control / 50 Hz
simulation settings, 256 environments, 4,096 samples per update (16 steps per
environment, or 0.64 s), full-rollout optimizer batches, 24 gradient steps,
and 200 updates. This is a separate profile from the failed 100 Hz calibration
above; those earlier results remain failed evidence. W&B was disabled, and
PyTorch used one CPU thread per process.

Evaluation uses deterministic actions for 15 s, requiring `abs(wrapped_angle)
< 0.14 rad` and `abs(angular_velocity) < 0.5 rad/s` throughout the final 2 s.
The fixed suite contains a 17x17 angle/velocity grid over `[-pi, pi] x [-5, 5]`,
17 hanging starts at `pi +/- 0.1 rad` with zero velocity, and 17 near-upright
starts in `[-0.2, 0.2] rad` with zero velocity. Final checkpoint 200 is the
reported checkpoint; intermediate evaluations were diagnostic only.

Sequential seed-7 comparisons held initial network weights and rollout
geometry constant. Each row includes preceding changes:

| Change | Grid success at update 200 | Hanging success | Finding |
|---|---:|---:|---|
| Original rewards/scales, reduced CPU sample count | 14.9% | 0% | Learns to settle downward |
| Action scale 1 → 5 Nm; explicit sin/cos scale 1 | 18.0% | 0% | Initial torque clipping rises to 32.0% |
| Initial action standard deviation 1 → 0.5 | 24.9% | 0% | Initial clipping falls to 4.9%; scaling/init alone do not solve swing-up |
| Velocity penalty weight 0.1 → 0.001 | 0% | 0% | Swings up, but final equilibrium is biased by 0.150 rad; earlier checkpoints passed and are not substituted for the final one |
| Periodic equilibrium reward | 100% | 100% | Equivalent angles at 0 and 2pi now earn equal reward; updates 50/100/150/200 all pass |
| Adaptive actor LR ceiling 0.01 → 0.001 | 100% | 100% | Avoids the original first-update 100x learning-rate rise |

Observed scaled velocities initially span approximately [-1.40, 1.37] at the
1st/99th percentiles; sin/cos already lie within [-1, 1]. Network architecture,
weight initialization, uniform reset ranges, gamma=0.95, lambda=0.98, and
entropy weight remain unchanged. At swing-up energy near the bottom, the old
velocity term contributes about -3.82 versus +0.49 from the energy term.
Reducing that penalty permits the energy-building phase. Separately, the old
equilibrium reward used unwrapped angle despite periodic observations; its
full-revolution invariance now has focused regression coverage. An analytic
swing-up controller catches all 81 starts of a smaller reference grid at
these same physics/control settings.

The final settings were also trained through the ordinary `train.setup` /
`OnPolicyRunner.learn` initialization path, without replacing the initial
weights or reseeding the environment after construction:

| Seed | Grid / hanging / near-upright success | Final angle RMSE | Mean catch time |
|---|---|---:|---:|
| 7 | 100% / 100% / 100% | 0.0104 rad | 1.81 s |
| 17 | 100% / 100% / 100% | 0.0226 rad | 1.73 s |
| 27 | 100% / 100% / 100% | 0.0073 rad | 1.82 s |

All three runs pass every diagnostic checkpoint from update 50 onward.
Each final run collects 819,200 transitions; the pendulum config now declares
the tested 256-environment/4,096-sample geometry explicitly. This evidence is
for PPO2 on MuJoCo CPU and these three seeds; it does not establish PSD-SAC or
Warp/VSim learning quality. SAC's subsequent validation is recorded separately
below.

Artifacts remain gitignored under `logs/pendulum_tuning/`: per-run resolved
configs, source snapshots, checkpoints, evaluation trajectories, reward-term
distributions, and rollout mean/std/quantiles for observations, actions,
rewards, and returns. `task_config_seed{7,17,27}/summary.json` and
`validation_summary.json` contain the final scores. `learning_comparison.png`
shows the ablations; `policy_distributions.png` shows validation trajectories,
torque histograms, final angles, and weighted reward contributions. The local
`probe.py` and `plot_results.py` retain the experiment and plotting commands.

Validation: focused reward-invariance and pendulum training/resume tests,
the portable and both colocated suites, and Ruff pass. No PPO2 update math,
normalizer implementation, or physics backend was changed.

### SAC with the same physical task and rewards

`PendulumSACCfg` inherits PPO2's physical settings, scales, reset distribution,
and 10-second time limit. Observations, observation noise, and reward methods
and weights remain shared. Both act at 25 Hz with 50 Hz physics, observe
`[sin(theta), cos(theta), omega/5]`, and can apply +/-5 Nm. Environment population
and learner settings are SAC-specific; copying PPO2's optimizer geometry was
not an appropriate sample-efficiency comparison.

Reference settings were checked against primary implementations and results:

- [RL Zoo's published SAC Pendulum-v1 agent](https://huggingface.co/sb3/sac-Pendulum-v1)
  reports a 20,000-transition training budget and learning rate 1e-3. The
  [maintained configuration](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml)
  retains those settings. That pendulum has different physics/rewards; its
  reported score is not this project's physical acceptance criterion.
- [Stable Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
  uses 256-sample minibatches, gamma 0.99, tau 0.005, and automatic entropy
  tuning. Its [policy implementation](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/sac/policies.html)
  uses 256/256 ReLU networks by default.
- [Spinning Up SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
  explicitly maintains one optimizer update per new transition. The
  [SAC algorithms and applications paper, Appendix D](https://arxiv.org/pdf/1812.05905)
  also specifies 256-sample batches, 256/256 ReLU networks, gamma 0.99,
  target updates every gradient step, and tau 0.005; its general learning
  rate is 3e-4.

The reference-informed Q2 profile collects one transition from each of 16
parallel environments, then performs 16 optimizer steps with 256-sample replay
batches. It uses a one-million-transition replay capacity, 1,024 uniform-action
warmup transitions, gamma 0.99, Polyak retention 0.995, learning rates 1e-3,
and target entropy -1. Initial temperature 0.01 is adapted to Q2's dt-integrated
reward scale. The actor uses the existing `ChimeraActor` with two shared 256-unit
ReLU layers, its linear latent projection, and mean/log-std heads; critics use
256/256 ReLU networks. Log standard deviation is bounded to [-20, 2]. This is
an adaptation of the reference settings, not an exact SB3 implementation.
The selected final checkpoint is update 1,500: 24,000 learned transitions plus
1,024 warmup transitions, or 25,024 collected samples.

Correctness fixes are separate from that configuration:

- The actor's nested dictionaries now use `activations`; the former
  `activation` key was silently ignored by `create_MLP`.
- Target critics update after every gradient step, rather than once after an
  entire group of optimizer steps. With the original aligned profile and
  seed 7, restoring the old cadence yields 0/323 successful starts versus
  323/323 with the fix, at identical sample/update budgets.
- SAC preserves the final physical observation before reset and bootstraps
  across time limits. True failures remain terminal. Time-limit transitions
  retain their ordinary reward instead of being routed through failure reward
  masking. A regression covers pre-reset versus reset state and both end types.
- Checkpoints save target critics and restore temperature in place, preserving
  the optimizer's parameter reference. Temperature is created on the requested
  device, warmup respects action bounds, and local vitals are recorded.

Replay contents and RNG state are not checkpointed: resume rebuilds replay,
so it is not an exact continuation of sampling. Older SAC checkpoints lacking
saved target critics remain usable for inference but cannot fully resume.
The unit regression verifies identical next optimizer updates when a checkpoint
is given the same replay data and RNG state.

The same 323-start physical suite as PPO2 is used: deterministic actions for
15 s, with angle below 0.14 rad and speed below 0.5 rad/s throughout the final
2 s. Evaluation runs at the training policy frequency and uses clean
observations. Final checkpoints are chosen before evaluating the runs.

Earlier comparisons remain visible; none changes task physics or reward weights:

| Profile | Samples | Successful starts, seeds 7 / 17 / 27 | Finding |
|---|---:|---|---|
| PPO2-style collection/batches, SAC LR 1e-4 | 819,200 | 323 / 322 / 311 | Only 24 optimizer updates per 4,096 new samples; hanging swing-up can be slow |
| Same, actor/critic LR 3e-4 | 819,200 | 261 / 259 / 259 | Rejected; all hanging starts fail |
| Same baseline, initial alpha 0.001 | 819,200 | 0 / 246 / 323 | Rejected; substantially more squash saturation and inconsistent learning |
| Same baseline, initial std 0.65 | 819,200 | 319 / 319 / 318 | Matches initial physical torque spread more closely, but does not solve reliability |
| Baseline extended to update 400 | 1,638,400 | 323 / 323 / 323 | Works, but inefficient learner geometry remains |
| Reference-informed profile, before timeout correction | 25,024 | 323 / 323 / 0 | Seed 27 stabilizes at 0.239 rad, outside the physical gate |

The reference-informed profile already passed all starts at 9,024 samples for
seed 7, but that run temporarily regressed to a 0.149-rad equilibrium at
17,024 samples. Intermediate success does not replace the preselected final
checkpoint or establish reliability.

With corrected time-limit handling, all three seeds pass all 323 starts at the
preselected final checkpoint, using 25,024 samples and the ordinary production
config and initialization:

| Seed | Grid / hanging / upright success | Final angle RMSE (rad) | Mean catch time (s) | Applied torque near limit |
|---|---|---:|---:|---:|
| 7 | 100% / 100% / 100% | 0.0288 | 2.04 | 2.90% |
| 17 | 100% / 100% / 100% | 0.0552 | 2.10 | 2.73% |
| 27 | 100% / 100% / 100% | 0.0228 | 1.45 | 3.51% |

All three also pass at the preceding 17,024-sample checkpoint; only seed 7
passes all starts at 9,024 samples. This establishes learning for these three
seeds on MuJoCo CPU. Warp, VSim, and PSD-SAC are not covered. The earlier PPO2
checkpoints used a coarser evaluation schedule, so this is not a measurement
of either algorithm's minimum sample requirement.

A public-CLI resume of seed 7 from update 1,500 through 1,510 also passes all
323 starts (final angle RMSE 0.0757 rad). The portable gate passes 330 tests
with one existing expected failure; colocated gym and learning suites pass
32 and 7 tests respectively. Ruff passes.

Scaling diagnostics distinguish actuator clipping from tanh saturation.
On a common 17x17 observation grid with 128 stochastic draws per state, the
original nominal-std-0.5 comparison gives PPO2 2.37-2.39 Nm applied torque
standard deviation and 4.5-5.9% hard clipping, versus SAC 1.90-2.04 Nm and
zero hard clipping. SAC's near-limit fraction (|torque| > 4.9 Nm) rises from
roughly 3-4% to 14-18% in the rejected low-temperature trials. These fixed-state
probes are separate from closed-loop trajectory statistics. Initial reference
exploration instead starts with uniformly sampled bounded actions.

The final policies have zero hard actuator clipping. Their deterministic
evaluation trajectories spend 2.7-3.5% of samples above 4.9 Nm in magnitude.
On the broad fixed-state stochastic probe, the same policies instead have
49-57% near-limit actions: strong actions far from the goal are compatible
with successful swing-up. Saturation alone does not diagnose a bad policy.

Artifacts remain gitignored under `logs/sac_pendulum_validation/`: resolved
configs, source snapshots, checkpoints, per-term vitals, rollout observation /
action / pre-squash / Q statistics, and full evaluation trajectories. The local
`probe.py`, `evaluate_saved.py`, and plotting scripts retain the experiment
recipes. `action_scaling.json` records command/applied torque quantiles,
clipping, near-limit fraction, and the squash derivative; `action_scaling.png`
visualizes the initial comparison and rejected temperature trial.
`timeouts_distributions.png` shows the final learning curves, hanging-start
trajectories, torque distributions, final angles, and per-term rewards.
`validation_summary.json` records the final profile, all three seeds, and the
resume check; the corresponding raw trials are in `timeouts_seed{7,17,27}/`.

### Go2Trot PPO minibatch alignment (2026-10-08)

Decision: retain the Go2Trot baseline of 32,768 samples per optimizer minibatch,
32 gradient steps per network, and 65,536 fresh transitions per update (16 steps
per environment at 4,096 environments). The RSL-RL-matched experiment below
regressed sample efficiency and yaw tracking. The base config is also restored
to its original 32,768-sample minibatches, 24 gradient steps, and 65,536-sample
rollout; Go2Trot retains its 32-step override.

The experiment used 24,576 samples per optimizer minibatch and 20 gradient
steps per network. These match RSL-RL 5.5.1's four minibatches and five epochs
when using 4,096 environments and its documented 24-step example. The matched
rollout collected 98,304 fresh transitions per update, or 24 steps per
environment. Task physics, rewards, observation/action scales, domain
randomization, networks, learning rates, and the 550-update budget were
unchanged. Observation normalizers were disabled. The experiment did not
address the separate normalization or actor-only adaptive-learning-rate findings.

With the matched geometry, the minibatch generator drew four complete
24,576-sample batches per shuffle. Twenty steps processed 491,520 samples per
network, using every fresh transition exactly five times. The retained baseline
uses a 65,536-sample rollout, 32,768-sample minibatches, and 32 steps:
1,048,576 sample presentations per network, with 16 uses each.

The paired validation uses seeds 7, 17, and 27 on MuJoCo Warp / RTX 5080,
4,096 environments, 100 Hz policy/control/physics, and 550 updates. Each seed
uses identical initial actor and critic weights across the two profiles. Final
checkpoint selection is fixed at update 550. That gives the new profile 50%
more fresh transitions. An additional comparison holds samples equal:
baseline update 450 versus new update 300, both 29,491,200 transitions.
Evaluation uses 100 environments across ten balanced commands, both basic
and randomized resets, five seconds per episode, and a 0.5-second settling
exclusion for tracking metrics.
Evaluation is deterministic, disables pushes, and retains configured physical
randomization. Native Warp evaluation was added after the initial CPU survival
regression; both backends' results are retained. Device RNG differences mean
CPU-versus-Warp comparisons do not isolate physics alone.

Before the runs, the acceptance screen requires finite model/loss tensors,
final-20-update mean training reward at least 90% of baseline, survival loss
at most five percentage points overall / ten points per command, and each
forward/lateral/yaw RMSE no greater than
`max(1.10 * baseline, baseline + 0.05)` in physical units. Survival accompanies
tracking because the latter is measured only until failure. Gait, actuator,
and per-term reward diagnostics are retained alongside the screen.

The initial optimizer-only comparison retained the 65,536-sample rollout:
two full minibatches per shuffle, a randomly omitted 16,384-sample remainder,
and 7.5 average uses per transition. All three seeds failed CPU survival checks.
Seed 7 improved final training reward from 5.729 to 5.869, but randomized-reset
survival fell from 98% to 87%. For seed 17, forward-0.5-m/s survival fell from
100% to 80%; for seed 27, forward-3-m/s survival fell from 100% to 60%.
The initial seed-7 failure prompted the full-rollout experiment. These failures
remain under `logs/go2trot_rsl_optimizer_20261008/`, including seed-7 diagnostic
checkpoints 0, 100, and 250.

The completed 98,304-sample rollout experiment also fails the declared
learning-quality screen. At update 550, native Warp randomized-reset results
are below; arrows denote baseline to matched geometry. The screen includes
both reset modes and individual commands, beyond the aggregate values shown.

| Seed | Survival | Forward RMSE (m/s) | Yaw RMSE (rad/s) | Native Warp screen |
| --- | --- | --- | --- | --- |
| 7 | 96% → 98% | 0.265 → 0.255 | 0.552 → 0.594 | Pass |
| 17 | 97% → 100% | 0.229 → 0.231 | 0.466 → 0.561 | Fail: yaw, both reset modes |
| 27 | 100% → 99% | 0.297 → 0.252 | 0.551 → 0.636 | Fail: yaw, both reset modes |

CPU evaluation reproduces the yaw failures for seeds 17 and 27. Seed 7 also
fails its CPU randomized forward-3-m/s survival check (100% to 80%). Final
training rewards retain 95.11%, 98.47%, and 99.38% of the paired baselines,
and all model/loss checks remain finite. Thus comparable aggregate reward and
high survival do not establish equivalent tracking or gait quality. Mean
native randomized contact-phase agreement falls from 63.82% to 59.25%; actuator
diagnostics show no consistent improvement across seeds.

Final-update comparisons use 54,067,200 fresh transitions for the new profile
versus 36,044,800 for baseline. At exactly 29,491,200 samples, every seed fails
the screen on both evaluation backends. Mean randomized forward RMSE rises
from 0.273 to 0.399 m/s on native Warp (CPU: 0.278 to 0.388 m/s). The matched
geometry still learns, but requires more samples for comparable forward
tracking and retains worse yaw tracking at update 550. Similar learning quality
is not established, so the baseline defaults are restored. Overlapping GPU
workloads make these runs unsuitable for comparative speed claims.

Resolved configs, checkpoints, finite-value checks, per-term training rewards,
evaluation trajectories, and reproduction scripts remain gitignored under
`logs/go2trot_rsl_rollout_20261008/`. `comparison.json` and
`comparison_warp.json` record the CPU and native Warp screens; their PNGs show
final-update curves and diagnostics. `same_samples.png` and
`same_samples_warp.png` show the equal-sample comparison. Initial actor/critic
weights and physical configs match exactly within each seed.

Validation after restoring the baseline: 330 portable tests pass (95 deselected,
one existing expected failure), plus 32 colocated gym tests and seven learning
tests. Ruff passes. The campaign's CPU/GPU execution smokes passed, and a
generator probe confirmed exactly 20 minibatches and five complete passes for
the experimental configuration.

## Rebase verification (2026-09-08)

Compared `cdx_freeze` with `cdx` and restored the reference Warp step,
keyboard dispatch, canonical viewer pose, yaw reward, constraint capacity,
VSim solver setting, and matching lockfile. The pre-rebase auto-stash contained
70 additional project files, including the missing friction fixture and the
SDK/streamlining work; those sources are recovered, with the stash retained
and archived vendor binaries/licenses excluded.

Validation: 309 portable tests passed with one existing expected failure,
31 Warp tests passed, 47 VSim tests passed, and 38 colocated tests passed.
Lock validation, Ruff, and package builds passed. Matching seed-7 pendulum
smokes at 100 Hz completed two updates and checkpoint reload/resume on both
GPU backends. These short smokes establish execution correctness, not learning
quality or a speed baseline. The prior failed learning calibration above
remains unresolved.

Reference-gait weights remain deliberate tuning differences: `reference_traj`
is 3.0 versus 0.5 in `cdx`, and `swing_grf`/`stance_grf` are each 1.5 versus
0.5. Their effect on trained policies has not been established by this audit.
The existing deployment residual-action clipping mismatch remains an expected
failure. The 16 optional Unitree integration tests were not run because the
native Cyclone DDS dependency is unavailable in the current environment;
the SDK fetch helper verified the configured source revision.

Commands, original diffs, recovery manifest, and test/training artifacts are
under `logs/rebase_audit/20260908/README.md`.

## Revised immediate plan

1. **Complete:** correct MuJoCo CPU/Warp free-root angular velocity at the
   backend boundary:
   publish world-frame velocity and convert world-frame reset/root-update writes
   into native body coordinates using the requested orientation. Verify at
   **100 Hz** with rotated poses, independent native/rotation-increment oracles,
   selective resets, cached tensors, and real Go2Trot observations. VSim serves
   as an unchanged contract reference.
2. **Complete:** close the separate MuJoCo CPU reset-liveness gap. Setup and
   root/reset commits publish current DOF, root, body, and contact state. Sparse
   resets forward/refresh only selected environments; empty masks do no work.
3. **Incomplete:** retrain nominal CPU/Warp/VSim baselines from scratch and evaluate
   their cross-backend transfer with fixed standing, translation, yaw, and combined
   commands. Report per-command survival and tracking before attributing remaining
   transfer failure to contact physics or DR. An older VSim policy remains an
   available diagnostic reference, not a required training restart.
4. **Frozen:** a fresh 100 Hz pilot with source hashes, applied parameter arrays,
   training support, held-out domains, rollout geometry, seeds, and checkpoint
   selection. Preserve old manifests rather than resuming them across the fix.
5. Evaluate the paired nominal/DR checkpoints as they become available. Expand
   only after finite training and coherent physical behavior; require three
   complete paired seeds for promotion. Set pooling remains a separate, explicit
   performance/DR-sampling decision, informed by the vendor benchmark.

### Reduced restart at 100 Hz

The restart uses five fresh Go2Trot training cells: CPU nominal, Warp nominal,
VSim nominal, Warp all-DR, and VSim all-DR. Each uses seed 7, 500 iterations,
4,096 environments, 65,536 rollout samples (16 steps/environment), 32,768-sample
optimizer minibatches, and 32 gradient steps. Physics and control are both
100 Hz. The user's source-config iteration limit of 550 remains unchanged;
the campaign explicitly overrides it with 500.

Checkpoints 100 and 250 receive native nominal and combined-in-range evaluation;
checkpoint 500 receives both domains on all three backends. This is 50
evaluation cells with 200 environments, ten balanced command cases, and five
seconds per evaluation. CPU full-DR training is excluded; the small CPU
evaluation cells still check cross-backend transfer. Runtime benchmark cells
are omitted. Evaluations are scheduled after their training cell finishes,
without waiting for the CPU training run to finish first.

In-range domains now derive from the frozen training config: friction
`[0.5, 1.0]`, stiffness `[0.9, 1.1]`, damping `[0.8, 1.2]`, and link mass/inertia
`[0.9, 1.2]`. Earlier evaluation constants under-covered damping and mass.
This single-seed pilot can expose regressions and justify further work; it
cannot establish a DR winner or replace the three-seed promotion gate.

```bash
uv run --frozen --env-file .env.vsim \
  scripts/run_full_domain_randomization_campaign.py \
  --output logs/baselines_100hz_20260906 \
  --backends cpu warp vsim --bundles off all --exclude-training cpu:all \
  --skip-speed --seeds 7 --train-num-envs 4096 --train-iterations 500 \
  --save-interval 50 --checkpoints 100 250 500 \
  --eval-domains nominal combined_in --cpu-workers 2 \
  --stages train eval summarize --evaluate-after-training
```

The manifest freezes source/config hashes and the reduced cell plan. Schema 2
adds explicit training exclusions, speed-stage inclusion, and evaluation
scheduling. Old campaigns remain historical; do not resume them across these
corrections.

The campaign stopped under `logs/baselines_100hz_20260906/`; its `manifest.json`
and `summary.json` record four complete training cells, one failed training
cell, 40 complete evaluations, and ten pending evaluations. Controller records
are under `logs/baselines_100hz_20260906_control/`. Revalidation found all 44
completed cells valid and execution sources unchanged from the frozen snapshot.

### Reduced pilot results

All four GPU runs completed 500 iterations with finite logged metrics. CPU
nominal logged through iteration 444, then failed during the next PPO actor
update with NaN distribution means. Instability was already visible at iteration
242 (value loss 36.2), rising to 6.67e15 at 264 before recovering to 0.031 at
300. A second wave reached infinity at 439 and NaN at 444. Action-rate penalties
also exploded, but they average completed episodes, so their logged timing
cannot establish which quantity diverged first. Checkpoint 400 has finite
model/optimizer tensors; it does not certify the preceding training as healthy.
There is no CPU checkpoint 500 or CPU-trained evaluation result in this pilot;
its ten dependent cells remain pending. The subsequent investigation below
isolates a sufficient failure mechanism without recovering the exact first
trigger of the original run. Keep the failed run visible.

Final-checkpoint survival below is **nominal / combined-in-range**, in percent.
Each cell uses 200 environments, ten commands with twenty trials each, and a
five-second horizon at 100 Hz.

| Training | CPU evaluation | Warp evaluation | VSim evaluation |
| --- | ---: | ---: | ---: |
| Warp, DR off | 97.5 / 99.5 | 97 / 100 | 94.5 / 95.5 |
| Warp, full DR | 100 / 100 | 100 / 100 | 100 / 100 |
| VSim, DR off | 100 / 94 | 87.5 / 93.5 | 100 / 99.5 |
| VSim, full DR | 95 / 94 | 93.5 / 95 | 100 / 100 |

Warp DR removes every observed final-checkpoint failure, including sideways-
right failures in the nominal policy. Native forward tracking RMSE worsens
from 0.246 to 0.354 m/s nominal and 0.211 to 0.345 m/s under combined variation.
The DR policy is still improving between checkpoints 250 and 500, so these
results compare a fixed training budget, not converged policies.

VSim DR improves native forward tracking RMSE from 0.358 to 0.318 m/s nominal
and 0.368 to 0.318 m/s under combined variation. It resolves the nominal
policy's backward failure on Warp (1/20 to 20/20 survival), but fast-forward
survival remains only 10/20 on CPU and 7/20 on Warp in the nominal domain.
The corresponding DR-off policy survives fast-forward in 20/20 CPU and 14/20
Warp trials. This is a transfer tradeoff, not uniform robustness improvement.

Tracking metrics exclude the first 0.5 seconds and samples after termination;
interpret them alongside survival. This single-seed, short-horizon, in-range
pilot does not establish a general winner. Next priorities are CPU numerical
stability, controlled CPU/Warp fast-forward and backward trajectory comparison,
then additional paired GPU seeds and targeted stress tests if those checks are
coherent. CPU full-DR training remains excluded.

### CPU instability investigation and command correction

Artifacts are under `logs/cpu_instability_20260907/`. A fixed checkpoint-250
policy reproduces runaway targets on actual CPU physics without any optimizer
updates: 256 environments, 500 control/physics steps at 100 Hz, fixed seed and
initialization. Deterministic raw actions reach 2.47e7, residual targets 2.08e7
radians, unweighted action-rate penalties 5.41e11, and critic predictions
7.27e4. Applied torques remain capped at 45.43 Nm, recorded physical state stays
finite, and native MuJoCo warning counts remain zero.

Holding physical observations fixed and advancing only previous-target/history
feedback also reproduces growth with checkpoint 250. That probe produces GAE
returns of 8.86e12 and mean squared value error of 4.02e23 with both networks
unchanged. Checkpoints 200 and 400 stay bounded in the same frozen-observation
probe. A separately reinitialized checkpoint-400 continuation completes 50
updates with maximum mean value loss 0.0414. It is not an exact simulator/RNG
resume and does not certify the failed campaign as healthy.

The task reset path had a separate defect: immediately after resetting a fallen
robot, actor/critic inputs could retain its old local base velocities and
projected gravity. Those selected derived caches now refresh from the committed
root state, and base height is a persistent root-state view. This correction
changes rollout trajectories but does not remove the checkpoint-250 runaway.

Go2Trot now constrains the **full** desired joint position (default offset plus
gait reference plus policy residual) to canonical actuator joint limits before
PD control. The applied residual stays in `dof_pos_target` and history, keeping
observations and rewards consistent with the controller. PPO retains the raw
sampled action for likelihood calculation. This uses the robot's existing
limits; there is no reward, optimizer, or normalization tuning.

The same fixed-policy experiment with that boundary reduces deterministic peak
actions from 2.47e7 to 9.06, peak action-rate penalty from 5.41e11 to 2.59, and
peak critic value from 7.27e4 to 7.03. Stochastic/noisy peak actions fall from
228.65 to 11.39. Full commanded-position violations are at most 2.4e-7 radians
(float32 roundoff). This isolates the stabilizing intervention, but the old
policy's termination count increases: deterministic 23 to 32 and stochastic
12 to 37. Retraining and evaluation are required; numerical stability is not
evidence of improved gait quality.

Regression tests cover immediate reset actor/critic observations, selective and
empty resets, legal commands, both joint limits with nonzero phase/default
offsets, passive/permuted DOF routing, raw PPO sample preservation, and finite
applied histories/rewards under huge finite samples. Removing the reset fix
fails two CPU cases; removing the command projection fails three boundary
cases. VSim native-root preservation allows only 2e-7 quaternion refresh
roundoff; derived caches and actor/critic observation preservation remain exact.

The original campaign remains a record of the earlier command semantics.
Further CPU collection uses a fresh seed-7, 500-iteration run with the original
4096-environment/16-step rollout geometry and 100 Hz physics/control. Diagnostic
instrumentation records per-field inputs, rewards, values/returns, native warning
counts, and gradient norms; it captures the first large value-loss update and
stops on nonfinite gradients/losses or an update exception. Do not merge those
results into the original frozen manifest.

The fresh CPU run launched at `2026-09-07T04:17:14Z` under
`logs/cpu_instability_20260907/fresh_cpu500/`. Training status and per-update
diagnostics are in that directory; controller status and its ten-cell evaluation
plan are under `fresh_cpu500_control/`. After successful training, checkpoints
100/250 receive native CPU nominal/combined-in-range evaluation, and checkpoint
500 receives both domains on CPU/Warp/VSim. Execution-source changes block
evaluation, and the controller checks artifact shape, frequency, balanced
commands, finite core traces, and checkpoint identity before marking a result
complete. Training and all ten evaluations completed by
`2026-09-07T05:36:47Z`. All 500 recorded updates have finite inputs, physical
state, losses, and gradients, with zero native warning counts. Maximum mean
value loss is 0.266 and the final value is 0.0355. All 105 recorded execution
source hashes and all ten evaluation checkpoint hashes were revalidated.

Final survival is 100% nominal/combined on CPU and Warp, and 98.5%/99% on
VSim. Nominal forward tracking RMSE is 0.151/0.156/0.332 m/s on CPU/Warp/VSim;
yaw RMSE is 0.480/0.479/0.510 rad/s. Thus stability and forward transfer are
strong, while yaw tracking and VSim fast-forward accuracy still need work.
Earlier GPU policies used different command semantics, so these are not yet
matched-policy-training comparisons across the corrected task.

### Backend specificity and cost of the corrections

The unstable CPU-trained checkpoint 250 also runs away on CUDA with the same
recorded observations and precomputed stochastic noise. Deterministic CPU/CUDA
actions exceed 100 at the same step 75 and reach approximately 8.27e8 after
500 steps. Stochastic thresholds also agree. Final action differences are only
about 2.9e-6 of peak amplitude. All eight Warp/VSim checkpoints at iterations
250/500 remain bounded on the same CPU replay. The instability follows the
learned weights; CPU arithmetic is not required for this failure mechanism.

An initialization confound is confirmed: all four original GPU runs have
identical initial actor weights, but the seed-7 CPU actor differs. The trainer
constructs the environment before the network. CPU task initialization consumes
the CPU RNG subsequently used by `nn.Linear`, while GPU task sampling uses the
CUDA RNG. Future cross-device comparisons should explicitly match initial model
state or separate learner initialization from task RNG. Which training
difference first triggered the original unstable policy remains unresolved.

At 4096 environments and 100 Hz, isolated added command-projection work costs
approximately 30 microseconds on CPU and 16-17 microseconds on CUDA. Added
reset-derived-state refresh costs 239 microseconds on CPU and 138-143
microseconds on CUDA. Relative to current open-loop task steps with eight
selected resets per step, their summed cost corresponds to roughly 0.09% CPU,
0.8% Warp, and 3.3% VSim. These percentages indicate scale, not a paired
end-to-end regression measurement. The fixes do not change native physics
settings, solver work, or environment-set topology.

Actual CPU training collection medians over iterations 100-400 are 4.406 seconds
before and 4.420 seconds after (+0.33%); whole iteration medians are 9.693 and
9.342 seconds. Different trajectories and added diagnostics prevent attributing
those differences solely to the patch. Detailed timings and replay evidence are
in `logs/cpu_instability_20260907/timing_and_backend_specificity.json`.

The final correctness gate passes 265 portable tests, 38 colocated tests, all
31 Warp tests, and 18 focused licensed VSim tests. Ruff and the package build
pass. A two-update CPU smoke uses the production 4096-environment/16-step
geometry and completes with finite losses/gradients and no native warnings.
An end-to-end evaluation regression verifies that raw samples and pre-step
observations retain their timing while recorded applied commands reflect the
post-projection task values. Logs are in the investigation directory.

### CPU reset-state cache correction

Setup, atomic resets, and root-only updates now copy current native state into
the persistent public buffers immediately. Each selected reset performs one
native forward and contact-force assembly, followed by one environment's cache
copy. Empty masks perform no native work, and unselected native/public state
remains unchanged. The physics-step staging is unchanged.

Seven new 100 Hz regressions cover setup, immediate root/body/DOF/contact state,
root-only writes, sparse-reset isolation, and native call counts. Five failed
before the correction. The full portable gate now passes 257 tests, the
colocated suites pass 38, and the focused GPU checks pass five Warp and
15 VSim cases. Ruff and package build pass. A nominal eight-environment CPU
smoke completed two updates and a deterministic evaluation at 100 Hz. These
are integration checks, not the 500-iteration baseline results. Logs are in
`logs/cpu_reset_liveness_20260906/` and `logs/cpu_reset_smoke_20260906/`.

Cleanup removed approximately 4.5 MB of generated build/cache files and the
superseded Ant `initial_dr_smoke`/`initial_summary_check` outputs. The removal
inventory is `logs/cleanup_20260906/removed.json`. Historical campaign data,
checkpoints, referenced experiment workers, and the final vendor submission
remain available.

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
- Root linear and angular velocities are in world coordinates on both reads
  and writes. Tasks rotate them into body coordinates exactly once.
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

On 2026-09-06, the local VSim dependency was upgraded from `0.3.12` to
`0.3.14+cu130`, matching the locked CUDA 13.0 PyTorch build. Import/license
preflight and all 29 licensed VSim tests passed on the RTX 5080. The portable
suite (223 tests), colocated suites (38 tests), Ruff, and package build also
passed. A Go2Trot smoke with all configured DR axes, seed 7, 4,096 environments,
100 Hz control/physics, 16 rollout steps per environment, and two PPO updates
produced finite metrics and checkpoint tensors under
`logs/vsim0314_upgrade_smoke/Sep06_17-59-14_/`. The headless smoke emitted
unresolved visual DAE resource warnings; viewer assets were not validated.
This establishes upgrade compatibility. The suspected normalization regression
with DR environment sets and full policy transfer remain separate checks;
earlier campaign results retain their original engine version.

### Repository gate

```bash
uv run --frozen ruff check .
```

GitHub CI runs the portable and colocated suites and Ruff, including a small
CPU training/checkpoint/resume smoke. Package builds are not a CI gate for the
repository-checkout workflow; hardware-specific groups remain local gates.

## Physics and parity evidence

### Angular-velocity frame correction

On 2026-09-06, MuJoCo CPU and Warp were found to copy free-joint body-local
angular `qvel[3:6]` directly into public world-frame root state and to make the
inverse mistake on reset/root writes. The task's world-to-body rotation was
correct for the contract, so the wrong backend output was rotated a second
time. Both backend boundaries now convert explicitly; VSim's convention and
the already world-frame rigid-body angular velocities are unchanged.

The 100 Hz CPU probe in `logs/angular_velocity_frame_20260906/` uses 90° yaw
and requests world-X rotation at 1 rad/s. Before the fix, native physics rotated
about world Y and post-step public/native world angular velocity differed by
1.00000036 rad/s. After the fix, it rotates about world X and the post-step
error is `1.72e-8 rad/s`. This independent native check is necessary: matching
read/write mistakes can pass a public-state round trip.

Older saved trajectories independently confirm the error. In
`logs/backend_unapplied_actions/vsim_seed37_model1000_500hz_pd_current/`,
MuJoCo root/body world angular velocities differed by up to 0.6463 rad/s before
termination; rotating the published root angular vector into world axes reduced
the discrepancy below `1e-6 rad/s`. VSim's root/body angular values agreed
directly. These are reexamined historical data, not new 500 Hz runs.

The defect predates DR and affects locomotion observations, angular rewards,
and reset/push semantics. Old MuJoCo policies are historical checkpoints with
the old observation contract, not corrected-campaign baselines. Existing VSim
checkpoints remain useful for reevaluation on corrected MuJoCo. In
`logs/dr_full_new`, 1,039 of 1,260 completed evaluations involve MuJoCo training
or evaluation and cannot establish corrected-contract robustness/transfer;
221 VSim-to-VSim cells are unaffected by this particular defect. Preserve all
artifacts and failures. Native parameter readbacks, policy-free probes with
unaffected inputs, and VSim topology timings retain their documented scope.

Validation: the new `test_root_velocity_frames.py` and
`test_root_velocity_observation.py` execute 14 cases at 100 Hz: five CPU,
five Warp, and four VSim. They cover both root-write APIs, sparse resets with
a changed orientation, native world/body velocity, world-axis orientation
increments, cached state liveness after stepping, and Go2Trot observation
scaling. The CPU reset cases failed before the fix; an isolated restoration
of the old readback also makes the observation regression fail. All 14 pass,
as do the 11 existing 100 Hz VSim DR/set regressions, 241 portable tests,
38 colocated tests, Ruff, and the package build. Detailed GPU/portable logs are
retained beside the before/after probe. The older complete Warp/VSim groups were not rerun;
the GPU gate here is the focused 100 Hz selection. No fresh training or policy
transfer result is claimed by this correction.

### VSim pre-DR speed and normalization comparison

On 2026-09-06, pre-DR revision `7d82334` (parent of the first environment-set
commit `7af97ca`) was compared with current revision `d164440` using the same
new `vlearn 0.3.14+cu130` binary and RTX 5080. The old source ran in an isolated
worktree; the current checkout was preserved. Both used 4,096 Go2Trot
environments, seed 7, 100 Hz control, matched 88-input actor observations,
disabled observation noise/normalizers, 65,536 rollout samples (16 steps per
environment), 32,768 optimizer minibatches, and 32 gradient steps. Each fresh
training ran 60 iterations; steady-state medians exclude the first 10.

| Physics frequency | Pre-DR, one set | Current, one set | Current, 4,096 nominal sets |
|---|---:|---:|---:|
| 500 Hz | 146,556 samples/s | 149,465 samples/s | 132,086 samples/s |
| 100 Hz | 314,747 samples/s | 327,763 samples/s | 303,941 samples/s |

These are synchronized PPO collection-plus-optimization rates. The many-set
case enabled friction DR with its value fixed at nominal, isolating topology
from physical variation. Relative to current one-set code, many sets reduced
PPO throughput by 11.6% at 500 Hz and 7.3% at 100 Hz. Backend-step throughput
(including eager state refresh) fell 16.6% and 14.4%, respectively; setup
increased from about 1.8 to 4.4 s.
Enabling all DR axes at fixed nominal values gave similar speed at 500 Hz.
Current one-set backend throughput remained within 1.2% of pre-DR. A separate
100 Hz empty-reset microbenchmark was 11.6% slower than pre-DR because current
code still commits/refreshes empty selections; timeout-reset throughput and
short PPO throughput improved in the same comparison.

Current one-set and nominal many-set probes had bitwise-identical sampled
state, contact, observation, and diagnostic policy-output arrays at both
frequencies. Old/current initial observations and policy outputs agreed within
`1.7e-7` and `1.3e-6`, respectively; later old/current contact trajectories were
not identical. RunningMeanStd source is unchanged and disabled in this config.
Contact-strength scaling did change independently of set count: current code
uses native imported mass (about 16.3063 kg), versus the old configured
16.087 kg. VSim's converted asset supplies 0.21928 kg across eight calf
collision links without explicit URDF inertials. This increases the weight
denominator 1.36%; the source revisions share the same conversion pipeline.

All seven short trainings retained finite metrics and checkpoint tensors.
Licensed test gates passed: 24 tests on pre-DR and 29 on current source.
Evidence supports targeting set-layout and empty-reset costs for optimization.
These timings do not establish converged training quality or campaign transfer,
and both revisions used the new binary, so they do not isolate a VSim version
effect. Reset RNG consumption differs across revisions. Protocol, raw timing
and state arrays, source hashes, workers, mass audit, and the concise report
are retained under `logs/vsim_pre_dr_comparison_20260906/`.

### VSim environment-set profiling at 100 Hz

The vendor SDK/examples supplied on 2026-09-06 use the same topology as Q2:
`train/envs/ant_environment_domain_randomization.py:183–184` creates
`[1] * num_envs`. The vendor getting-started guide documents that static
properties are shared within a set; independent physical DR uses one set per
environment. Its DR PPO config also uses 4,096 environments. Vendor Ant has
fewer links/DOFs/sensors than Q2 Go2 and uses different solver settings, so
its absolute speed is not a matched performance reference. Q2 applies physical
DR at startup; repeated property writes are not the source of its ongoing
set-layout cost. The supplied release includes Python/bindings and native
binaries, without the native solver implementation.

New profiling uses only **100 Hz physics/control**, current Q2, VSim
`0.3.14+cu130`, RTX 5080, 4,096 Go2 environments, seed 7, eight native solver
iterations, and fixed nominal parameters. This measures a zero-torque robot
that has fallen into ground contact. A diagnostic proxy changes set
count with all physical DR paths disabled. Each trial restores the initial
pose, settles for 200 steps, warms the measured operation for 25 calls, then
times 100 calls with device-wide synchronization; medians use three repeats.

| Set count | Native step | Full backend step | State refresh |
|---|---:|---:|---:|
| 1 | 3.923 ms | 4.116 ms | 0.221 ms |
| 64 | 3.862 ms | 4.053 ms | 0.225 ms |
| 4,096 | 4.495 ms | 4.729 ms | 0.298 ms |

These isolated component timings are not additive. Most added time is inside
native simulation; standalone canonical tensor assembly was about 82–83 us
at all three counts. Disabling CUDA graphs retained the native-step gap
(4.013 ms versus 4.604 ms). GPU traces of ten full backend steps had identical
kernel-name/call counts (3,070 launches, 128 unique names), with increased
execution time in several native articulation kernels. In particular,
`artiStage_11` had median durations of 28.3 versus 50.2 us across the same
80 calls (means 28.5/59.5 us, with outliers in the many-set trace). More
separate static-property data and reduced data sharing are a plausible cause;
the exact cache/memory mechanism is not established by these timings.

The 64-set result motivates a future controlled comparison of shared sampled
physical configurations. Such pooling changes the DR sampling scheme and must
preserve explicit environment-to-set mapping. No production DR semantics or
backend physics were changed in this investigation. Empty-reset refresh cost
remains a separate optimization target.

The reusable `scripts/benchmark_vsim_environment_sets.py` retains the 100 Hz
protocol without hardware-dependent pass/fail speed thresholds. Licensed
regressions in `test_vsim_domain_randomization_regression.py` cover native set
selection, nominal trajectories/scaled observations, reset isolation, and
contact-strength normalization using each environment's applied mass.
Profiling artifacts and native kernel comparisons are under
`logs/vsim_set_investigation_20260906/`.

Validation: all 11 new licensed cases passed at 100 Hz; the portable suite
passed 236 tests and the colocated suites passed 38. Ruff and touched-file
format checks passed. The finished benchmark CLI completed all eight profiles
with eight environments in two sets. Initial reset tests used one tolerance
for pose and velocity; diagnostics found exact unselected DOF state immediately
after reset and about `1e-4 rad/s` velocity differences after the next contact
solve. The final test preserves tight immediate-reset/pose checks and uses a
separate `2e-4` velocity tolerance for that next step. Initial failure logs and
per-field deltas remain in the artifact directory. Ruff now excludes all of
`thirdparty/`, including the newly supplied vendor backup.

### Vendor Ant environment-set cross-check

On 2026-09-06, the vendor's `AntEnvironmentGpu` was benchmarked directly,
without importing Q2 environment/backend code. Both layouts used the same
vendor class, nominal Ant asset/parameters, 4,096 environments, 100 Hz,
eight solver iterations, seed 7, zero actions/reset noise, and identical
flattened 64-by-64 world placement. A scoped setup wrapper changed only the
partition from one set of 4,096 environments to 4,096 singleton sets; vendor
stepping, observation, reward, and reset methods were unchanged. Comparing
the regular and DR classes directly would also change solver/material settings
and reset bookkeeping, so this comparison holds the implementation constant.

Two fresh processes per layout ran in one/many/many/one order. Each profile
used five timed batches of 500 steps per process with warmup and device-wide
CUDA synchronization. Across ten batches per layout, native stepping increased
from 1.188 to 1.302 ms (8.8% less throughput); the full vendor environment loop
increased from 1.311 to 1.430 ms (8.3% less throughput). Native timing used a
zero-torque settled-contact workload; full-environment timing included its
ordinary termination/reset cycle. Sampled state/observation arrays were finite
and bitwise identical across all four processes. The partition therefore has
a measurable speed cost in the vendor integration too. These are environment
throughput measurements, without PPO optimization. Raw results, source hashes,
the exact worker, and a report are in `logs/vendor_ant_sets_20260906/`.

A vendor-facing reproducer now lives at
`thirdparty/vlearn/train/benchmark_ant_sets.py`, with instructions beside it.
It uses the original non-DR Ant class for both layouts and preserves its
stepping/reset methods. The default protocol sets the episode timeout to
100 steps, ensuring resets within each timed 500-step batch at 100 Hz;
this differs from the earlier probe's default 1,000-step timeout. With the
same hardware, VSim version, and 4,096 environments, ten batches per layout
measured 1.236 ms for one shared set and 1.339 ms for singleton sets,
or 7.6% lower throughput. All four processes exercised resets, and world
placement, sampled preflight states, and corresponding timed trial final
states matched bitwise. Raw results are in the vendor checkout's
`benchmarks/ant_sets/`; the script requires only the normal vendor setup.

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

## Learning-stack pruning

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
| Orphaned | `StateEstimator` → `StateEstimatorNN` → `SERolloutStorage` | No registered task |
| Partially reachable | Other classes in `QRCritics.py` | Only `DenseSpectralLatent` is selected by a registered task |

Environment-agnostic utilities are not removal candidates merely because a
runner does not call them. Keep normalization, logging, dictionary utilities,
and the colocated PBRS implementation/tutorial unless their own behavior or
tests show that they are obsolete.

### First pruning slice

Removed the unconfigured legacy chain `OldPolicyRunner` → `PPO` →
`ActorCritic` → `RolloutStorage`, including its package exports, selectable
registry names, and old-runner-only inference test. Repository references
confirmed that this chain had no registered task or retained runner consumer.
The unused `policy_class_name` config fields were also removed; current runners
construct actor and critic modules directly. `PPO2` retains its public name,
training behavior, and checkpoint format. SAC/PSD-SAC and the other pruning
candidates remain in the inventory above.

The pendulum regression's source manifest now hashes existing worktree files
while retaining tracked deletions in its recorded Git status. This allows its
train/save/resume smoke to run during pruning before deletions are staged.

Validation for this slice: the portable suite, both colocated suites, Ruff,
and source/wheel builds pass. The portable gate includes registered task
construction and the real CPU PPO2 train/evaluate/checkpoint/resume smoke;
the built wheel excludes the four removed modules. These checks establish
retained execution behavior, not learning quality or optional GPU-backend
coverage.

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
3. After the first legacy PPO removal, remove the remaining unreachable chains
   in separate reviewable changes: the state estimator chain, then the three
   unconfigured runner variants. Update package exports and delete tests that
   exist only for a removed path in the same change.
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
