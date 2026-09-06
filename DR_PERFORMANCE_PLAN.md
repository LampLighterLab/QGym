# Domain Randomization Performance Plan

## Objective

Make domain randomization use fixed-size, device-resident operations while
keeping one clear task-facing contract across MuJoCo CPU, MuJoCo Warp, and
VSim. The refactor should simplify the implementation, remove avoidable CUDA
synchronization, and preserve the distinction between:

- startup physical identity: friction, link mass, and inertia are sampled once;
- episodic task tensors: currently PD gains are resampled at episode reset.

Correctness and physical semantics take priority over throughput. A backend
must not silently reduce the requested randomization distribution.

## Current evidence

The current physical-DR slowdown in VSim is primarily an environment-set
topology cost, not a repeated index-to-mask conversion. A synchronized local
control on an RTX 5080 used 4,096 Go2Trot environments, fixed friction `1.0`,
50 warm-up steps, and five 100-step trials:

| VSim layout | Median throughput | Setup time |
| --- | ---: | ---: |
| One set containing 4,096 environments | 217k environment-steps/s | 1.75 s |
| 4,096 sets containing one environment each | 184k environment-steps/s | 4.31 s |

The 4,096-set layout was 15.1% slower despite having no friction variation.
This reproduces the existing 15--17% physical-DR gap while removing sampling
and contact-distribution differences.

The VSim feedback still identifies real inefficiencies in the reset path:

- the runner already has a boolean done mask but converts it with
  `torch.nonzero`, which synchronizes CUDA with the host;
- task code uses dynamically sized index tensors for sampling and assignment;
- VSim reconstructs persistent boolean command masks from those indices;
- a floating-base VSim reset currently commits and refreshes the combined root
  and joint state once through `reset_dof_state` and again through
  `reset_root_state`.

An isolated 4,096-environment Torch probe measured about 14.5 us for
`nonzero`, 11.1 us to clear and scatter an index list into a mask, and 5.9 us
for a full `[4096, 12]` `torch.where`. These are only discriminating
microbenchmarks: in a real rollout, `nonzero` can also wait for previously
queued simulator work.

## Target design

Use two small interfaces rather than a generic selection wrapper.

### Startup physical parameters

Startup always covers every environment. The randomizer samples full tensors
and the backend consumes them directly:

```python
randomizer.randomize_startup()
backend.set_contact_friction(coefficients)  # [num_envs]
backend.set_link_mass_scale(scales)         # [num_envs, num_bodies]
```

There is no startup environment index or mask. VSim property commands are
unmasked and run once; Warp updates complete batched model arrays; MuJoCo CPU
iterates over its serial worlds. Internally generated tensors are not recast or
revalidated immediately before use.

### Episodic task tensors

Episode updates accept one persistent boolean mask shaped `[num_envs]`.
Configured targets retain full-size candidate, nominal, and applied-scale
tensors. Each axis samples a full candidate and applies it with fixed-shape
operations such as `torch.where`; boolean indexing is not a substitute because
it also creates dynamically sized work.

The existing explicit name-to-tensor binding remains. Configuration must not
trigger arbitrary environment attribute lookup.

### Reset contract

The task owns one persistent reset mask and updates it in place:

```text
terminated | timed_out
        -> persistent reset mask
        -> episodic DR and task reset sampling
        -> one backend state commit
```

GPU backends consume the mask directly. MuJoCo CPU may convert its CPU mask to
indices locally because that operation cannot stall a CUDA device. Avoid a
`Selection(mask, indices)` abstraction: lazily requesting its indices would
retain the synchronization while hiding where it occurs.

## Milestone 0: freeze the performance discriminator

Before changing APIs, turn the current evidence into a repeatable profiling
protocol.

Work:

- Add an explicit topology-control case: physical DR enabled with every value
  fixed to its nominal value.
- Run each cell in a fresh process because VSim set topology and Warp model
  batching are fixed during setup.
- Capture Nsight Systems traces for VSim with:
  1. one set, DR off;
  2. 4,096 sets, fixed friction `1.0`;
  3. 4,096 sets, sampled startup friction;
  4. one set, episodic PD randomization.
- Record task, seed, environment count, control/simulation frequencies,
  warm-up, timed steps, reset profile, device, package versions, commit, and
  worktree state.

Checks:

- Synchronize VSim, CUDA, and Warp work at every timing boundary.
- Report setup separately from steady stepping and reset profiles.
- Confirm fixed and sampled physical cases select the same native topology.
- Confirm the fixed-friction case applies exactly `1.0` to every environment.
- Preserve raw Nsight reports so the VSim team can inspect the same capture.

Exit gate: the one-set versus many-set comparison is reproducible, and the
trace distinguishes Torch sampling/update work from VSim simulation work.

### Milestone 0 result (2026-08-31)

The repeatable worker and capture driver now cover the four declared VSim
cases. The run used the current Go2Trot configuration: 4,096 environments,
seed 7, 100 Hz control and simulation, decimation 1, 25 warm-up steps, and five
50-step trials per reset profile on an RTX 5080 with VSim 0.3.12. This protocol
is not an absolute-throughput comparison with the earlier 500 Hz evidence.

| Case | Setup | Steady | Timeout resets | Reset-all stress |
| --- | ---: | ---: | ---: | ---: |
| One set, DR off | 1.72 s | 1.057 M env-steps/s | 0.865 M | 0.905 M |
| 4,096 sets, friction fixed at 1.0 | 4.28 s | 0.903 M | 0.739 M | 0.769 M |
| 4,096 sets, sampled friction | 4.26 s | 0.901 M | 0.739 M | 0.766 M |
| One set, episodic PD DR | 1.73 s | 1.061 M | 0.859 M | 0.901 M |

The fixed-friction case retained exactly `1.0` in all 4,096 environments. Its
native topology exactly matched the sampled-friction case. Sampled/fixed
throughput ratios were `0.998`, `1.001`, and `0.996` for steady, timeout, and
reset-all profiles. The many-set topology therefore accounts for the measured
14.5--15.4% throughput loss; startup sampling does not explain it. PD-only
remained within 0.6% of DR-off.

The one-repeat Nsight captures reproduce the same separation. The fixed and
sampled many-set setup ranges took 4.92 and 4.94 s, versus 1.99 and 2.00 s for
DR-off and PD-only. Within setup, CUDA kernel summaries attribute about
137--139 ms to non-Torch kernels for the many-set cases versus 2.7 ms for the
one-set cases. Torch kernels account for 0.76--0.79 ms in every case. Raw
reports and synchronized JSON artifacts are under
`logs/dr_performance_milestone0_20260831/`; generated artifacts remain
gitignored. `scripts/profile_vsim_domain_randomization.py` reproduces them.

Milestone 0 exit gate: **met**. Proceed to the mask-native selection contract
as a separate review step.

## Milestone 1: mask-native selection contract

Make a persistent boolean mask the task-facing selection type for episodic DR,
task reset work, and backend state commits. This directly addresses the
avoidable CUDA synchronization identified by the VSim team. It does not force
MuJoCo CPU to scan every world: that backend may derive a short index list
locally because its mask is already on the CPU.

This milestone changes selection representation without also combining root
and DOF commits. Keeping atomic reset as a later step makes state-liveness
regressions easier to isolate.

Work:

- Reuse one persistent task-owned boolean reset mask shaped `[num_envs]` and
  update it in place rather than rebinding it.
- Pass the mask from active runners to `_reset_idx` without calling `nonzero`.
  Remove the unused task-local index conversion when the runner owns reset.
- Change episodic DR from `randomize_episode(env_ids)` to
  `randomize_episode(reset_mask)`.
- Allocate one persistent full-size candidate tensor per configured episodic
  target. Sample full tensors and update targets from retained nominals with
  fixed-shape `torch.where` operations.
- Convert reset-state, command, phase, history, and task-specific resampling to
  full-size candidates and fixed-shape masked updates. Do not replace index
  tensors with `tensor[mask]`; boolean indexing still creates dynamically sized
  work.
- Change backend reset methods to accept the mask. VSim either wraps that
  persistent mask directly or copies it into one persistent command mask;
  Warp uses fixed-shape masked tensor operations. Neither GPU backend rebuilds
  a mask from indices.
- Let MuJoCo CPU call `nonzero` locally on its CPU mask and iterate only the
  selected native `MjData` objects. Indices remain a private implementation
  detail rather than part of the shared contract.
- Do not branch on `reset_mask.any()` in CUDA code. An all-false mask must be a
  valid fixed-size no-op without a device-to-host readback.
- Audit every active runner and task override, especially Go2Trot,
  MiniCheetah variants, fixed-base tasks, periodic command resampling, and
  timeout-specific reset behavior.
- Leave logger episode compaction and rollout-storage statistics alone. Their
  dynamic selections are learning-stack concerns, not reset or DR semantics.

Checks:

- All-false, all-true, and sparse-mask tests for episodic DR and every reset
  buffer touched by the base tasks.
- Repeated episodic updates rebuild values from the nominal rather than
  accumulating a random walk, and independent DR axes retain independent RNG
  streams.
- Sparse reset isolation for DOF, root, command, phase, PD scale, history, and
  episode-length buffers.
- Shared fixed- and floating-base reset contract tests on MuJoCo CPU, MuJoCo
  Warp, and VSim, including canonical routing and joint-limit clamps.
- Initial all-environment reset and all-false steady steps remain valid.
- A focused CUDA profiler trace contains no reset/DR `nonzero`, boolean
  indexing, or index-to-mask reconstruction. A source audit permits `nonzero`
  only inside the MuJoCo CPU implementation for this path.
- Repeat the PD-only reset benchmark. Throughput must not regress by more than
  2% relative to the synchronized Milestone 0 baseline.
- Tiny CPU, Warp, and VSim train/save/inference smokes complete with finite
  states, rewards, and checkpoint tensors.

Exit gate: the active task, runner, episodic-DR, and backend reset interfaces
use one persistent mask; the GPU reset path performs no dynamically sized
selection or index-to-mask conversion; MuJoCo CPU alone derives private local
indices; and all reset-liveness and sparse-isolation checks pass.

### Milestone 1 result (2026-09-01)

The selection refactor is implemented. The task owns one persistent boolean
reset mask, active runners fill it in place, episodic DR and task-specific
resampling consume full-size candidates, and both GPU backends consume the mask
directly. MuJoCo CPU is the only reset implementation that derives a private
index list. Sparse reset tests now cover base state, DOF state, commands, Go2
phase and frequency, PD scales, target/history tensors, and episode length.

The synchronized 4,096-environment A/B on the current committed Go2Trot
frequencies (100 Hz control, 500 Hz simulation) found no measurable PD-DR
penalty:

| Backend | Bundle | Steady | Timeout resets | Reset-all stress |
| --- | --- | ---: | ---: | ---: |
| VSim | off | 1.050 M | 1.048 M | 1.125 M |
| VSim | PD | 1.054 M | 1.057 M | 1.130 M |
| Warp | off | 0.334 M | 0.330 M | 0.351 M |
| Warp | PD | 0.331 M | 0.330 M | 0.351 M |

These are physics-step rates; each cell used seed 7, 25 warm-up control steps,
and five 50-control-step trials. PD/off ratios were `1.004`, `1.008`, and
`1.004` on VSim, and `0.992`, `0.999`, and `1.000` on Warp.

An exact repeat of the Milestone 0 100 Hz simulation protocol exposed a
separate all-false-mask cost. Relative to the old index implementation, VSim
PD throughput changed from `1.061/0.859/0.901 M` to
`0.882/0.881/0.933 M` for steady/timeout/reset-all. The realistic timeout and
reset-all paths improved by 2.5% and 3.6%, but repeatedly asking the task to
reset an empty mask was 16.9% slower. An isolated breakdown measured
`1.053 M` for stepping alone, `1.042 M` with episodic DR only, about `0.99 M`
with either one VSim state commit, and `0.880 M` with the complete current task
reset. The cost is therefore the two unconditional state commit/refresh calls,
not mask sampling or index reconstruction.

The focused Nsight range contains no `nonzero`, masked-select, or scalar-read
selection kernel. The two existing state commits remain visible and are the
declared subject of Milestone 3. CPU, Warp, and VSim train/save/inference
smokes completed with finite states, actions, rewards, and checkpoint tensors.

Milestone 1 implementation and correctness checks are complete. Its original
no-reset performance clause is not met; retain that result rather than hiding
it in the production reset profile. Do not begin Milestone 2 until this review
point is accepted. Milestone 3 must remove the duplicate state commit and
repeat the all-false discriminator.

## Milestone 2: full-batch startup DR

Remove selection from startup physical randomization. Milestone 0 showed that
the persistent VSim cost comes from the required one-environment-per-set
topology, not from sampling or index-to-mask conversion. This milestone is
therefore a contract and lifecycle simplification; it is not expected to
recover that roughly 15% steady-state VSim cost.

Work:

- Change `DomainRandomizer.randomize_startup` to take no selection argument.
- Sample full friction and link-mass-scale tensors once after backend setup.
- Remove the separate `bind_link_masses` lifecycle step. Create the persistent
  applied link-mass-scale tensor inside `randomize_startup`, when backend link
  masses and their canonical shape are available.
- Change backend physical setters to consume complete tensors:
  `set_contact_friction(coefficients)` receives `[num_envs]`, and
  `set_link_mass_scale(scales)` receives `[num_envs, num_bodies]`.
- Remove `SimBackend` helpers that cast, flatten, and compare internally
  generated update tensors.
- Remove VSim's startup friction and link-property mask buffers. Keep stable
  full-size property-value buffers and unmasked command objects for graph
  capture.
- Keep VSim's one-environment-per-set topology while independent physical
  identities are requested.
- Route deterministic evaluation overrides through `DomainRandomizer` rather
  than modifying both its buffers and the backend separately. Overrides remain
  full-domain operations, so the recorded applied values cannot drift from the
  backend state.
- Keep the ownership boundary explicit: `DomainRandomizer` owns sampled and
  applied parameter values, including link-mass scales; each backend owns the
  engine-native nominal mass and inertia needed to apply those scales and
  recompute derived physics state. No nominal physical tensor belongs in the
  task environment.
- Delete partial startup-property APIs and call sites, including tests whose
  only purpose is to update selected physical domains. Episode DR remains the
  only selective path.

Checks:

- Colocated sampler tests: seeded full-batch sampling, independent axis
  streams, range bounds, disabled-axis behavior, and exactly one startup
  application. Replace subset-startup tests rather than preserving an unused
  capability.
- Recording-backend lifecycle test: construction invokes each enabled physical
  setter once and episode resets never invoke either setter.
- Backend integration tests: complete native readback of friction, mass, and
  inertia on CPU, Warp, and VSim.
- Canonical-routing sentinel: distinct per-link scales reach the correct
  native link on every backend.
- Physics invariant: mass and inertia scaling retains the existing predicted
  acceleration relationship.
- Episode-reset persistence: startup properties do not change after partial or
  full resets.
- Evaluation override: deterministic full-domain values replace all startup
  samples through the randomizer, and recorded values match native backend
  readback.
- Call-site audit: no startup physical-property call constructs an all-true
  mask merely to select every environment, and no supported call passes any
  selection argument to a physical setter.
- Repeat the Milestone 0 4,096-environment VSim cells (`off`, fixed friction,
  sampled friction, and PD-only). Fixed and sampled friction must retain the
  same singleton-set topology and remain within 2% of each other; no new
  steady-state or setup regression is accepted. This is a regression check,
  not a requirement to recover the topology cost.

Exit gate: startup application has no environment selection or VSim property
mask, the task contains no startup-DR nominal buffers or binding step, all
native readbacks and physics checks pass, the randomizer and backend remain in
sync after evaluation overrides, and no supported path depends on a partial
physical-property update.

## Milestone 3: atomic backend reset

Combine the now-mask-native state reset into one backend operation. This is a
separate correctness/performance change from replacing indices with masks.

Work:

- Replace separate backend DOF/root commits with one `reset_state(reset_mask)`
  operation. Preserve write-then-commit semantics and public tensor identity.
- Make the single operation clamp limited joints, commit the complete root and
  DOF state, run the required engine forward operation, and refresh public
  state once.
- Make VSim issue one state SET/refresh sequence per floating-base reset.
- Keep MuJoCo CPU's mask-to-index conversion local and perform one native
  forward per selected environment.
- Audit fixed-base behavior and `set_all_root_states` so they use the same
  complete-state contract without fabricating selection indices.

Checks:

- Shared fixed- and floating-base atomic-reset contract tests on all three
  backends.
- Root and DOF writes are committed together and remain live after reset.
- VSim issues exactly one state SET/refresh sequence per floating-base reset.
- MuJoCo Warp and CPU preserve canonical/native routing and joint-limit clamps.
- Sparse, all-false, and all-true resets preserve unaffected public state.
- Repeat reset-profile benchmarks to distinguish the atomic-commit gain from
  the mask-selection gain measured after Milestone 1.

Exit gate: every backend exposes one complete-state reset operation, root and
DOF writes remain live and correctly ordered, and no backend performs duplicate
forward/refresh work for a single task reset.

### Milestone 3 result (2026-09-02)

The backend contract now exposes one `reset_state(reset_mask)` transaction.
Legged and fixed-base tasks call it once; VSim performs one `_commit_state`,
MuJoCo Warp performs one `mjw.forward` and public-state refresh, and MuJoCo CPU
performs one `mj_forward` per selected environment. The root-only push path
remains separate.

The exact 4,096-environment, 100 Hz Milestone 1 discriminator improved on
VSim from `0.882/0.881/0.933 M` to `0.933/0.935/0.987 M` environment-steps/s
for steady/timeout/reset-all profiles: gains of 5.8%, 6.1%, and 5.8%. PD/off
ratios were `1.000`, `1.000`, and `0.997`. Warp PD/off ratios were `1.005`,
`0.997`, and `0.999`, so episodic PD sampling remains within the 2% target.

Shared CPU, Warp, and VSim reset contracts pass, including sparse selection,
joint-limit clamping, configured floating-base height, state liveness, and
canonical routing. The CPU friction regression now explicitly observes one
native forward per selected environment. Milestone 3 exit gate: **met**.

Logger episode compaction and storage statistics also use dynamic selections
today. They are not DR semantics and remain outside these milestones. Profile
them separately and create a focused learning-stack change if they remain a
material synchronization source after the reset refactor.

## Milestone 4: performance and parity closure

Re-run controlled evidence after the three implementation milestones.

Matrix:

- backends: MuJoCo CPU, MuJoCo Warp, VSim;
- bundles: off, PD only, fixed nominal physical topology, sampled friction,
  sampled mass, and all configured axes;
- environment counts: 256 on every backend and 4,096 on GPU backends;
- reset profiles: none, production timeout rate, and all-environment stress;
- fresh process per cell, identical seed and task/control geometry.

Checks:

- Report setup time, median and p10--p90 throughput, reset rate, host RSS, and
  CUDA memory.
- Verify applied distributions and state finiteness outside timed regions.
- Capture post-change Nsight traces for the same VSim cells as Milestone 0.
- Confirm no physical property command executes during episode reset.
- Confirm fixed and sampled physical cases have similar steady throughput;
  their common gap from DR-off is reported as topology cost.
- Run the full portable, colocated `gym`, colocated `learning`, Warp, VSim,
  Ruff, and package-build gates.
- Update `DR.md` and `MIGRATION_PLAN.md`; remove the stale description of
  friction as episode-resampled.

Targets:

- no backend or DR bundle regresses more than 2% against its comparable
  pre-change path, excluding the already isolated VSim set-topology cost;
- PD-only overhead remains within 2% of DR-off;
- fixed-nominal and sampled physical VSim throughput differ by at most 2%;
- reset traces contain no DR/reset `nonzero` or index-to-mask conversion;
- state, physics, and training-smoke checks pass on every supported backend.

Exit gate: performance claims are supported by synchronized artifacts, native
parameter semantics still match, and the remaining VSim cost is attributed to
environment-set topology rather than Q2 mask construction.

## Optional Milestone 5: bounded physical-domain pool

Do not start this milestone unless Milestone 4 confirms that VSim's set cost is
still unacceptable and the VSim profile/vendor review provides no lower-cost
independent-domain layout.

Possible design:

```python
domain_randomization.startup.num_physical_domains = 64
```

`None` means one independently sampled physical identity per environment. A
finite value `K` samples `K` complete identities and assigns environments to
them. One identity contains every startup physical axis together; VSim cannot
independently bucket friction and mass because an environment set owns their
combined static properties. CPU and Warp must use the same assignment in
controlled comparisons even though they do not require it technically.

Checks before adoption:

- Sweep `K = 1, 32, 64, 256, 1024, 4096` for VSim setup, memory, and throughput.
- Verify balanced assignment, persistent per-environment applied-value logging,
  and combined-axis native readback.
- Quantify unique physical identities and parameter correlations in artifacts.
- Run a controlled policy study before claiming that reduced diversity is an
  acceptable substitute for independent environments.
- Expose `K` in config and every campaign artifact; never enable a bucket
  fallback implicitly for VSim.

Exit gate: either retain independent domains with their measured cost, or
adopt one explicit cross-backend `K` based on both performance and policy
evidence.

## Review boundaries

Keep the work reviewable in this order:

1. profiling protocol only;
2. episodic mask application;
3. task/backend reset contract;
4. full-batch startup API;
5. performance evidence and documentation;
6. optional physical-domain pool as a separate feature decision.

Do not combine physical-range tuning, reward changes, policy architecture
changes, or new DR axes with these milestones. Those changes would make the
performance comparison uninterpretable.
