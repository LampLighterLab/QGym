# Domain Randomization

## Current status

The MuJoCo angular-velocity frame and CPU reset-state liveness corrections are
in place. The reduced restart in `MIGRATION_PLAN.md` completed four 500-iteration
GPU runs and 40 evaluations at 100 Hz. CPU nominal training diverged, logging
through iteration 444 before failing with NaN policy outputs; its ten evaluation
cells remain pending. CPU full-DR training is excluded. Warp DR eliminated all
observed final-checkpoint failures at a forward-tracking cost; VSim DR improved
native tracking but left fast-forward transfer failures. See the reduced pilot
results in `MIGRATION_PLAN.md` and `logs/baselines_100hz_20260906/summary.json`.
The CPU investigation subsequently reproduced runaway previous-target feedback
with a fixed policy and no learning. Go2Trot now bounds full joint-position
commands and keeps applied residuals in history; task reset observations also
refresh immediately. The fresh CPU validation completed 500 finite updates and
ten evaluations: final survival is 100% on CPU/Warp and 98.5% nominal/99%
combined on VSim. Yaw tracking and VSim fast-forward accuracy still need work.
Same-policy replay reproduces the old runaway on both CPU and CUDA, while the
original CPU/GPU runs also began with different weights despite sharing a seed.
See the CPU investigation and backend-specificity evidence in
`MIGRATION_PLAN.md`. This single-seed pilot is a regression screen, not promotion
evidence, and its results describe the earlier command semantics.
Previously trained MuJoCo policies used the wrong angular observation/reward
inputs and must not be treated as corrected baselines. Existing VSim policies
remain useful for reevaluation; their native results are unaffected by this
particular defect.

The latest full artifact, `logs/dr_full_new`, contains 25 runtime cells,
36 valid training cells, and 1,260 completed evaluations, with nine failed
training cells and 315 pending evaluations. Of the completed evaluations,
1,039 involve MuJoCo training or evaluation and 221 are VSim-to-VSim. Preserve
the campaign as historical evidence and use a fresh directory for corrected
collection; do not resume its scientific manifest across the frame correction.

The first three domain-randomization axes are implemented on MuJoCo CPU,
MuJoCo Warp, and VSim. The current `Go2TrotCfg` enables effective contact
friction in `[0.5, 1.0]`, stiffness scales in `[0.9, 1.1]`, damping scales in
`[0.8, 1.2]`, and independent per-link mass scales in `[0.9, 1.2]`. Link mass
and diagonal inertia always receive the same scale; link CoM is unchanged.
Old logged configs containing the inert `domain_rand` block remain inert.

The config separates sampling cadence deliberately:

| Config group | Implemented axes | Cadence |
| --- | --- | --- |
| `domain_randomization.startup` | Contact friction; per-link mass and diagonal inertia | Sample every parallel environment once during task construction, then keep that environment's physical identity fixed. |
| `domain_randomization.episode.scale_ranges` | `p_gains` and `d_gains` multiplicative scales | Resample only the environments whose episodes reset. |

With thousands of parallel environments, startup sampling already provides a
large physical distribution. Keeping model properties fixed avoids native
property writes and derived-constant recomputation during frequent asynchronous
resets. The split is structural rather than a per-axis boolean, so adding an
axis requires an explicit cadence decision in both config and implementation.
The public trainer consumes this config as written and has no DR override.
Benchmark, evaluation, and campaign tools mutate private config copies when an
ablation needs a different set of axes.

Each axis has a dedicated seeded generator on the task device, so CUDA resets
do not transfer values from the host, one axis does not shift another axis's
sample stream, and DR does not perturb command or reset-state RNG. Runs are
reproducible for a fixed device, seed, and reset sequence; exact CPU/CUDA
sample equality is not required. Applied friction, PD scales, link scales,
absolute link masses, and diagonal inertias remain available in persistent
canonical tensors. Episode resets change only the selected environments' PD
gains; friction, link mass, and link inertia remain fixed until a new task is
constructed.

The stochastic features that do work are separate concerns:

- reset-state and command sampling vary the task distribution;
- `push_robots` applies a periodic disturbance;
- neither changes persistent physical or control parameters.

Before the Isaac Gym removal, friction was selected from 64 buckets while
actors were created, and a random amount was added to the base mass. Those
values stayed fixed for each environment. That engine-specific implementation
was deliberately removed. Its old config fields have now been deleted rather
than reactivated with changed semantics.

## Backend audit

Q2 has two backend families, but three execution paths that must be checked:
plain MuJoCo (`mujoco` on CPU), MuJoCo Warp (`mujoco` on CUDA, called `mjx` by
the evaluation script), and VSim (`vsim`).

| Path | What exists underneath | DR implementation |
| --- | --- | --- |
| MuJoCo CPU | Serial worlds | Startup friction DR keeps one shared model and activates each environment's fixed coefficient before native operations. Startup mass DR uses one compact private model per environment because derived inertial constants differ; headless models discard visual assets to avoid duplicating the roughly 5 MB rendered model. |
| MuJoCo Warp | One Warp model and batched per-world data | Startup friction, mass, inertia, subtree mass, and inverse-weight fields are batched. The one startup mass application runs `mujoco_warp.set_const` so all derived inertial fields remain consistent. |
| VSim | Environment-set topology selected during setup | Any startup physical DR uses `[1] * num_envs`, because material and link-property commands operate on sets. Persistent commands apply friction, mass, and diagonal inertia once after setup. PD-only DR retains the faster `[num_envs]` topology. |

VSim's domain-randomization documentation page is unfinished, but the shipped
examples are useful:

- `demos/401_link_mass.py` creates a `LinkProperty.MASS` command with one row
  per environment set.
- `demos/402_rigid_material.py` changes static and dynamic friction with
  `RigidMaterialProperty` commands.
- `train/envs/ant_environment_domain_randomization.py` creates one environment
  per set, applies masked mass and dynamic-friction updates on reset, and adds
  action noise, observation noise, and pushes separately.

The installed licensed VSim build has executed local headless probes assigning
different friction, link mass, and diagonal inertia to individual environment
sets. Q2 uses explicit mass and inertia commands rather than relying on the
vendor's implicit inertia scaling side effect.

Q2 defines one portable effective
sliding Coulomb coefficient. MuJoCo has one sliding coefficient, whereas VSim
distinguishes static and dynamic friction. It maps the portable value to
MuJoCo sliding friction and to both VSim values. A VSim-only interpretation
would make nominally identical experiments different.

## 2026-08-10 contact-friction campaign

The artifacts under `logs/dr_contact_friction_20260810/` are the first
performance and short-training screen of the implemented friction axis. They
were collected from dirty commit `7af97ca46dc5abc1d4e00c687ea9718361a86e8a`;
the manifest records that state, resolved configs preserve the parameters, and
the training run directories contain source snapshots.

### Synchronized speed protocol

Each backend/DR cell ran in a fresh process because enabling friction changes
native setup topology. The worker used `Go2Trot`, basic resets, disabled
pushes, seed 7, 32,768 environment-control-steps per profile, and five timed
repeats. The 256-environment cells warmed up for 32 control steps; the
4,096-environment cells warmed up for five. Timing used `perf_counter` around
complete task steps and reset commits. CUDA cells synchronized Torch before
and after every region, and Warp additionally synchronized its device. State
finiteness and the applied friction distribution were checked outside the
timed region.

The three reset profiles were: no resets; a deterministic timeout-rate
schedule matching the 500-control-step episode; and resetting every
environment after every control step. Values below are the median environment
control steps per second across the five repeats, rounded to the nearest whole
step; the per-cell JSON retains every raw timing.

| Backend | Envs | Profile | DR off | DR on | On/off |
| --- | ---: | --- | ---: | ---: | ---: |
| MuJoCo CPU | 256 | none | 4,710 | 4,369 | 0.928 |
| MuJoCo CPU | 256 | timeout | 4,643 | 4,352 | 0.937 |
| MuJoCo CPU | 256 | all | 3,071 | 2,887 | 0.940 |
| MuJoCo Warp | 256 | none | 12,196 | 11,877 | 0.974 |
| MuJoCo Warp | 256 | timeout | 10,135 | 9,904 | 0.977 |
| MuJoCo Warp | 256 | all | 8,733 | 8,527 | 0.976 |
| MuJoCo Warp | 4,096 | none | 97,490 | 95,773 | 0.982 |
| MuJoCo Warp | 4,096 | timeout | 67,482 | 65,950 | 0.977 |
| MuJoCo Warp | 4,096 | all | 65,934 | 64,084 | 0.972 |
| VSim | 256 | none | 61,537 | 60,653 | 0.986 |
| VSim | 256 | timeout | 56,146 | 55,235 | 0.984 |
| VSim | 256 | all | 54,721 | 53,840 | 0.984 |
| VSim | 4,096 | none | 229,665 | 191,718 | 0.835 |
| VSim | 4,096 | timeout | 219,036 | 181,715 | 0.830 |
| VSim | 4,096 | all | 225,501 | 186,258 | 0.826 |

Setup time was 0.349 to 0.353 seconds for CPU at 256 environments, 1.252
to 1.240 seconds for Warp at 256, 1.376 to 1.349 seconds for Warp at 4,096,
0.293 to 0.417 seconds for VSim at 256, and 1.640 to 4.162 seconds for VSim at
4,096, where each pair is DR off to on. The small negative Warp deltas are
measurement noise, not a claimed speedup.

The result separates the backend costs clearly. Warp's batched model fields
cost roughly 2--3% throughput and no measurable setup penalty. CPU's shared
model activation costs roughly 6--7% throughput at 256 environments. VSim is
also cheap at 256 environments, but its required change from one set of 4,096
environments to 4,096 one-environment sets costs 16.5--17.4% throughput,
2.52 seconds of setup, and about 177 MiB of additional observed CUDA memory at
4,096 environments. Independent VSim sets remain functionally correct, but
that scaling cost must be considered before adding another set-level axis.

### Short policy screen

Training used 4,096 environments, a 65,536-sample rollout, optimizer batch
size 32,768, 32 gradient steps, and 50 iterations for seeds 7, 17, and 27.
This is only 16 control steps per environment per iteration. Warp and VSim
completed all three paired seeds; CPU completed all three DR-off runs but only
DR-on seeds 17 and 27. The first CPU+DR seed-7 process exited natively with
return code `-11` while collecting iteration 40, after writing 39 complete
iteration records. Exact reproductions reached the same point. The failure was
subsequently diagnosed and fixed as described below; the original policy
screen still contains only two paired CPU seeds and is reported as collected.

| Backend | DR | Completed seeds | Final-10 training reward | Final-10 episode s |
| --- | --- | ---: | ---: | ---: |
| MuJoCo CPU | off | 3 | 4.048 | 4.535 |
| MuJoCo CPU | on | 2 | 4.052 | 4.487 |
| MuJoCo Warp | off | 3 | 4.023 | 4.434 |
| MuJoCo Warp | on | 3 | 4.027 | 4.438 |
| VSim | off | 3 | 4.111 | 4.422 |
| VSim | on | 3 | 4.206 | 4.689 |

These short training means are descriptive, especially the unequal CPU rows;
they are not evidence of convergence or a statistically resolved DR effect.

Each completed policy was evaluated only on its training backend, with 200
environments for five seconds at evaluation seed 1701. The explicit crossed
friction domains were nominal `1.0`, in-range `[0.5, 1.0]`, and fixed
out-of-range low friction `0.35`. Evaluation stored the applied coefficient
and per-environment physical metrics. Mean paired DR-on-minus-off effects were:

| Backend | Paired seeds | Domain | Reward | Survival | Vx RMSE | Slip RMS | Tilt RMS |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| MuJoCo CPU | 2 | nominal | -0.161 | +0.000 | +0.002 | -0.022 | +1.106 deg |
| MuJoCo CPU | 2 | in-range | -0.165 | +0.002 | +0.002 | -0.025 | +0.622 deg |
| MuJoCo CPU | 2 | low | -0.149 | +0.010 | +0.001 | -0.012 | -0.883 deg |
| MuJoCo Warp | 3 | nominal | -0.343 | +0.000 | +0.015 | -0.016 | +5.101 deg |
| MuJoCo Warp | 3 | in-range | -0.365 | +0.000 | +0.018 | -0.025 | +5.168 deg |
| MuJoCo Warp | 3 | low | -0.308 | +0.000 | +0.019 | -0.056 | +3.696 deg |
| VSim | 3 | nominal | -0.078 | +0.005 | -0.023 | +0.012 | +2.635 deg |
| VSim | 3 | in-range | -0.091 | +0.000 | -0.024 | -0.001 | +3.559 deg |
| VSim | 3 | low | -0.057 | +0.000 | -0.027 | -0.017 | +3.390 deg |

Policy-effect values are rounded to three decimals; the campaign summary and
evaluation NPZ files retain the underlying per-seed and per-environment data.

This is mixed evidence, not a robustness result. Survival was already near
ceiling. DR usually reduced slip and VSim forward-velocity error, but reduced
reward and often increased tilt; individual seeds varied substantially. The
ON range also changes mean friction from `1.0` to `0.75`, so it does not isolate
variance. The campaign was short, did not perform cross-backend transfer
evaluation, and used the currently active single phase oscillator plus fixed
leg offsets in `Go2Trot`. The four coupled oscillators proposed in
`oscillator.md` are not active code. No friction range or DR bundle should be
advanced from this screen alone.

### MuJoCo CPU crash resolution and final speed check

The CPU seed-7 failure was not caused by DR sampling or the shared-model
virtual domains. A journaled reproduction captured the exact environment and
state immediately before the native crash. The state was finite, its free
quaternion was valid, contact/constraint storage had ample capacity, and the
same pose crashed in an isolated one-step MuJoCo 3.11 subprocess at both the
sampled coefficient and nominal friction. It crashed in `mj_collision` while
processing overlapping lower-leg cylinders. Clearing solver warm-start state
did not help; disabling MuJoCo's general convex multi-contact CCD path did.

`Go2Cfg` and `Go2TrotCfg` therefore set `mjDSBL_MULTICCD`. Primitive
multi-point colliders, including foot/ground contacts, remain enabled. An
isolated subprocess regression exercises the captured pose so a regression is
reported as a failed test instead of taking down pytest. The exact 4,096-env,
seed-7, DR-on training command then completed all 50 iterations and wrote
`model_50.pt`, crossing the previously deterministic failure point.

A per-environment compact-model layout was also tested for friction alone
during diagnosis. It produced the same trajectory and the same crash, while
adding memory and dispatch cost, so friction-only DR retains the simpler
shared model with serial per-environment activation. Link-mass DR still needs
private models because its derived inertial constants differ by environment.

The final implementation was benchmarked again under
`logs/dr_contact_friction_20260811_final_speed/`. Every one of the ten cells
completed on an NVIDIA GeForce RTX 5080. Each profile used five synchronized
repeats and at least 50 control steps per repeat; the table shows median
environment control steps per second and the DR-on/off ratio.

| Backend | Envs | Profile | DR off | DR on | On/off |
| --- | ---: | --- | ---: | ---: | ---: |
| MuJoCo CPU | 256 | none | 4,575 | 4,308 | 0.942 |
| MuJoCo CPU | 256 | timeout | 4,525 | 4,274 | 0.945 |
| MuJoCo CPU | 256 | all | 2,999 | 2,874 | 0.959 |
| MuJoCo Warp | 256 | none | 12,211 | 11,701 | 0.958 |
| MuJoCo Warp | 256 | timeout | 10,131 | 9,796 | 0.967 |
| MuJoCo Warp | 256 | all | 8,688 | 8,454 | 0.973 |
| MuJoCo Warp | 4,096 | none | 96,096 | 93,181 | 0.970 |
| MuJoCo Warp | 4,096 | timeout | 67,351 | 65,514 | 0.973 |
| MuJoCo Warp | 4,096 | all | 65,629 | 63,733 | 0.971 |
| VSim | 256 | none | 61,425 | 60,909 | 0.992 |
| VSim | 256 | timeout | 56,118 | 55,358 | 0.986 |
| VSim | 256 | all | 54,466 | 53,928 | 0.990 |
| VSim | 4,096 | none | 219,721 | 184,858 | 0.841 |
| VSim | 4,096 | timeout | 210,200 | 176,186 | 0.838 |
| VSim | 4,096 | all | 223,653 | 185,017 | 0.827 |

Setup changed from 0.320 to 0.332 seconds on CPU, 1.339 to 1.355
seconds on Warp at 4,096 environments, and 1.764 to 4.307 seconds on VSim at
4,096 environments. The final result is consistent with the initial screen:
CPU pays roughly 4--6%, Warp roughly 3%, and VSim's required 4,096-set topology
roughly 16--17%. These are end-to-end costs of the enabled `[0.5, 1.0]`
distribution, not a pure topology-only microbenchmark.

## Should the current instantiation be replaced?

No. The `SimBackend` separation, canonical `RobotLayout`, and task lifecycle
remain the right architecture. Only native parameter storage changes when an
enabled physical axis requires it.

There are three credible layouts plus one rejected baseline:

| Layout | Benefit | Cost / limitation |
| --- | --- | --- |
| One domain per environment | Clean semantics, independently sampled startup identities, simple tests | Native Warp rows and VSim sets scale with environment count. At 4,096 environments, the measured VSim set topology added 2.52 seconds of setup, about 177 MiB, and 16.5--17.4% throughput cost. |
| Serial virtual domains on MuJoCo CPU | Independent friction values with one shared model; exact because worlds already step serially | Parameters requiring different derived constants cannot share this model. Mass DR therefore uses compact per-environment models. |
| A fixed pool of `K` domain buckets | Bounds VSim set count and resembles the old 64 friction buckets | Startup-only sampling removes reset coupling, but buckets quantize the distribution and make multiple environments share a physical identity. Keep independent environments as the reference unless measured scaling forces this trade-off. |
| One shared global domain | Minimal implementation work | Parallel environments do not cover a distribution. Reject as a DR implementation. |

Independent domains are the semantic reference. MuJoCo CPU implements them by
serial activation, Warp by per-world rows, and VSim by one environment per set.
Do not introduce an invisible bucket fallback. If a declared VSim throughput
or memory budget makes buckets necessary, expose the quantization and number of
buckets in the config and campaign artifacts.

Random torque/force injection is another useful robustness method, but it is
not a replacement for the physical-parameter contract. It is naturally
backend-neutral and should be evaluated later as a separate regularizer.

## What the literature suggests

There is no universally best list: randomize uncertainty that exists on the
target hardware, and use measured ranges where possible.

- [Tan et al. (RSS 2018)](https://roboticsproceedings.org/rss14/p10.html)
  randomized mass, inertia, motor friction and strength, control step, latency,
  battery voltage, contact friction, and IMU bias/noise. Their experiments also
  found accurate actuator and latency models essential, and showed the
  robustness-versus-peak-performance trade-off from broad randomization.
- [Margolis et al. (IJRR 2024)](https://journals.sagepub.com/doi/10.1177/02783649231224053)
  used ground friction/restitution, payload, center-of-mass displacement, and
  motor strength, together with a history-conditioned policy. They identify
  unmodeled lag as a likely remaining gap and emphasize that adaptation avoids
  forcing one conservative behavior to serve every domain.
- [Campanaro et al. (L4DC 2024)](https://proceedings.mlr.press/v242/campanaro24a.html)
  obtained competitive transfer with per-step random joint torques plus
  episodic torque offsets. It is evidence for a small, deliberate DR set, and
  also warns that base-force perturbations can change the learned gait.
- [Hu et al. (2025)](https://arxiv.org/abs/2503.01255) found that omitted joint
  static friction could dominate their transfer gap. This supports measuring
  actuator/joint friction instead of assuming generic damping covers it.
- [Sobanbabu et al. (CoRL 2025)](https://proceedings.mlr.press/v305/sobanbabu25a.html)
  show the other side of the trade-off: heuristic DR can be conservative, and
  targeted system identification can outperform it. DR ranges should therefore
  be logged, tested by ablation, and narrowed as hardware evidence improves.
- [Cha et al. (2025)](https://arxiv.org/abs/2504.06585) compare conventional DR
  over friction, link mass/CoM, armature, damping, motor constant, delay, noise,
  and pushes with joint-torque-space perturbations. Their results motivate
  keeping interpretable DR and unmodeled-dynamics perturbations as separately
  measurable approaches.

For Q2 locomotion, the practical priority is:

1. contact friction and backend-neutral PD-gain uncertainty;
2. loading: physically consistent link mass/inertia scaling, then a separately
   defined payload/center-of-mass model;
3. measured control delay, sensor bias/noise, and observation latency;
4. measured joint friction/damping/armature uncertainty;
5. motor-strength scaling, kept low priority until the higher-value axes are
   evaluated. If added, it must scale final clipped torque, not stand in for
   independent stiffness and damping uncertainty.

Terrain geometry, pushes, initial states, and command distributions should
remain separate named features. Keeping them separate makes ablations and run
comparisons interpretable.

### Cadence for future physical parameters

Physical identity parameters should normally join `startup`, even when a
backend could update them cheaply. They describe a particular robot or surface,
not a property that changes when that robot falls:

| Future physical axis | Planned cadence | Reason |
| --- | --- | --- |
| Added payload, link CoM offsets, and corresponding inertia | Startup only | Changing any of these requires physically consistent mass-property updates and derived constants. |
| Joint armature / rotor inertia | Startup only | It is a fixed actuator property and may participate in compiled or derived dynamics fields. |
| Passive joint damping, Coulomb friction, and stiction | Startup only | These represent mechanical variation between robots; backend write support differs. |
| Contact restitution and material compliance | Startup only | These are surface/material identities and VSim exposes them through set-level properties. |
| Collision dimensions or foot geometry | Startup only | Geometry changes can require collision-structure rebuilds and should never occur during an episode reset. |
| Motor strength, torque limit, and motor constant | Startup only | Hardware capability stays fixed. This remains low priority and is distinct from episode-resampled controller gains. |

Terrain shape, slope, or heightfield selection also belongs at startup when it
changes native scene geometry, but remains a separate terrain feature rather
than being folded into the physical-parameter DR block. Sensor calibration
bias is naturally startup-only; per-step sensor noise, action delay, pushes,
and reset-state sampling have different lifecycles and should stay separate.

## Work proposal and checks

### 1. Effective contact friction — complete

- The new config and sampler are resolved before backend setup. Friction is an
  absolute, nonnegative, startup-sampled value. The randomizer validates only
  its two-value range; terrain construction and material mapping belong to the
  selected backend.
- Sampling and native application are separate, and enabled unsupported
  application fails instead of falling back to a global value.
- A resolved `seed=-1` is written back to the environment config. DR uses its
  own generator on the task device and never stages reset samples through CPU.
- MuJoCo CPU uses serial virtual domains; Warp batches native model fields;
  VSim uses stable material-command buffers and one environment per set.

Evidence on 2026-08-09: pure validation/reproducibility tests, strict routing
tests, native engine readback, full-task reset integration, and a
shared tilted-gravity sled test all passed on MuJoCo CPU, CUDA Warp, and the
licensed CUDA VSim backend. Existing Warp and VSim legged state/physics
contracts also passed with the new topology.

The 2026-08-10 campaign completed the initial setup, memory, throughput, and
applied-value logging checks. The deterministic CPU seed-7 crash was reproduced,
isolated to MuJoCo 3.11 multi-contact CCD, fixed in the Go2 model options, and
covered by a subprocess regression. The exact 50-iteration run and the final
ten-cell speed matrix then passed. A longer policy campaign and cross-backend
evaluation remain necessary before changing the friction range or declaring a
robustness benefit.

### 2. PD-gain randomization — complete

- `episode.scale_ranges` maps explicitly bound task tensors to positive
  multiplicative ranges. The current config binds `p_gains` and `d_gains`.
  Independent per-axis generators mean enabling one axis does not shift
  another axis's sample stream.
- `p_gains` and `d_gains` are already `[num_envs, num_actuators]`. The
  `DomainRandomizer` clones the nominal value only for configured targets,
  stores the applied scale, and rebuilds selected rows from that nominal;
  `LeggedRobot` carries no DR-specific nominal-gain buffers.
- Tasks pass an explicit name-to-tensor binding to the randomizer. Config names
  never trigger arbitrary attribute lookup. The bound values are persistent
  floating-point tensors whose leading dimension is `num_envs`; because the
  task constructs them immediately before binding, the randomizer uses them
  directly instead of recasting or revalidating those internal invariants.
- Application is in canonical actuated-DOF order in the common task layer, so
  it uses the same path on all backends and preserves VSim's fast grouping when
  no physical axis is enabled.

Checks completed: gain reconstruction from nominal values, reproducible and
independent streams, partial-reset isolation, full-task tests on every backend,
and one-iteration training smokes on every selectable backend.

### 3. Link mass and inertia scaling — complete

- Every canonical robot link receives an independent positive scale. Mass and
  diagonal inertia are rebuilt from nominal values with that same scale; CoM
  stays fixed. It is sampled once at startup. A future added payload remains a
  separate mass/CoM/inertia model.
- MuJoCo CPU keeps derived constants in per-environment compact models. Warp
  batches the source and derived fields and calls `set_const`. VSim receives
  explicit masked mass and diagonal-inertia commands.
- Go2Trot's contact saturation and evaluation GRF normalization use effective
  per-environment total mass.

Checks completed: doubling mass and inertia halves pendulum acceleration under
equal torque on all three engines; native property readback, canonical routing,
startup persistence across episode resets, combined-axis Go2Trot stepping, full
optional backend groups, and one-iteration training smokes all pass. At 4,096
environments, a
small partial reset under the former episodic implementation took about 28 ms
with Warp mass DR versus 18 ms without it; VSim remained about 1.1 ms in both
cases. Startup-only mass DR removes that recurring property update. A fresh
controlled end-to-end throughput campaign is still required before quantifying
the new lifecycle.

### 4. Add delay/noise only with deployment evidence

Implement action delay as a control-step buffer and observation bias/noise near
observation assembly. Keep persistent bias distinct from per-step white noise.
Start with conservative ranges from measured Go2 control timing and sensors,
not copied paper ranges.

Checks: exact delayed-action sequence tests; bias persistence across a rollout;
noise statistics and seed reproducibility; clean inference when disabled.

### 5. Evaluate before expanding the parameter set

Run at least three controlled seeds for nominal training and each deliberate DR
bundle. Evaluate every policy on:

- nominal physics;
- a held-out grid for one parameter at a time;
- combined in-range samples;
- modest out-of-range stress tests;
- both MuJoCo implementations and VSim with the same explicit test domains.

Report mean return, command-tracking error, fall rate, energy/torque metrics,
and worst-decile (CVaR-style) performance. A DR bundle advances only if it
improves held-out robustness without an unexplained collapse in nominal
performance. Add parameters one at a time or in justified bundles; do not begin
with a large omnibus randomization range.

The full campaign is encoded in
`scripts/run_full_domain_randomization_campaign.py`. Its frozen protocol uses:

- DR off, friction only, PD gains only, link mass only, and all three axes;
- MuJoCo CPU, MuJoCo Warp, and VSim training at 4,096 environments;
- seeds 7, 17, and 27 through iteration 1000, with checkpoints at 100, 250,
  500, 750, and 1000;
- native nominal and combined-in-range evaluation at intermediate checkpoints;
- all three evaluation backends and nine one-axis/combined domains at the
  final checkpoint;
- 200 robots balanced across ten commands, 5 seconds per evaluation, a 5 N
  contact threshold, and a fixed 2 Hz oscillator frequency.

Friction evaluation uses evenly spaced scalar levels per command case. PD and
mass evaluation use deterministic independent per-actuator/per-link samples
from the declared range, reused across command cases; they are distributions,
not one-dimensional endpoint sweeps. The manifest binds every result to its
checkpoint hash, saved config, source snapshot, and exact applied parameter
arrays. Interrupted training restarts from its seed because checkpoints do not
contain simulator state and every RNG.

Run or resume it with:

```bash
uv run --env-file .env.vsim \
    scripts/run_full_domain_randomization_campaign.py \
    --output logs/dr_full_new [--resume]
```

The first full launch on 2026-08-13 was stopped during its initial Warp policy
after revealing that Go2's old `njmax=130` constraint allocation was too small:
contact-rich states requested as many as 200 rows and MuJoCo Warp truncates work
above the allocation. Those policy results are invalid evidence. Both Go2
configs now use 256 rows, matching the capacity already used by deterministic
evaluation. The campaign must start in a fresh directory after this model
change; the stopped pilot remains under `logs/dr_full_20260813` for diagnosis.

The live Marimo report is `notebooks/go2_domain_randomization_campaign.py`.
It shows completion, synchronized runtime cost, learning curves, physical
checkpoint progression, paired final robustness, backend transfer, and the
predeclared nominal-survival gate. It never treats the reward diagnostic as an
episodic return or a partial seed pair as a passing result.

#### Closed partial result: `dr_full_20260813_nj256`

Collection was intentionally stopped on 2026-08-13 to review the result before
spending several more days on the full matrix. The closed artifact contains all
25 runtime cells, 23 valid 1,000-iteration training cells, and no held-out policy
evaluations. It includes every DR bundle for seeds 7 and 17 on Warp and VSim,
plus CPU off/PD for seed 7 and Warp friction for seed 27. One technically
completed run, Warp off/seed 27, diverged after iteration 935 and produced a
non-finite value loss; artifact validation rejects it and all summaries exclude
it.

The engineering result is useful and bounded:

- PD-gain randomization changes throughput by at most about 2% in the measured
  production cells.
- Under the former episodic lifecycle, link-mass randomization dominated
  reset-path cost: roughly 15% on Warp and 17--19% on VSim at 4,096
  environments, with a similar CPU reset-stress cost at 256 environments.
  This result motivated moving physical-property sampling to startup.
- Friction costs about 2--3% on Warp, 7--8% on CPU, and 16--17% on VSim at
  4,096 environments because VSim needs one environment per set.
- Every bundle trained successfully for the common Warp/VSim seeds 7 and 17
  and reached nearly the full five-second episode horizon. This establishes
  trainability, not robustness.

No policy evaluation cells were run, so this campaign cannot rank DR bundles
or support claims about nominal regression, held-out robustness, stress
behavior, or cross-backend transfer. The decision is therefore: engineering
characterization complete; robustness verdict deferred. The closed directory
is not resumed after the wrap-up source changes; start a fresh output directory
for any follow-up evidence campaign.

### Low-priority backlog: motor strength

If hardware evidence later motivates motor-strength DR, scale the final
torque command and its effective limit in the common control path. Randomizing
stiffness and damping is useful but is not equivalent once feed-forward torque
or clipping is active. Implement this only after the milestones above.
