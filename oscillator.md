# Go2Trot four-leg CPG plan

## Goal

Replace the single Go2Trot phase with four coupled phase oscillators, ordered by
`RobotLayout.body_groups["feet"]` as `(FL, FR, RL, RR)`. The network has:

- a true standing fixed point at zero frequency, with every phase at
  \(\theta_s = 3\pi/2\); and
- a stable trotting relative-phase equilibrium while the phases rotate.

The policy retains its 12 joint residuals and gains one action controlling the
common oscillator frequency.

## Dynamics

Let \(\theta_i\) be a leg phase, \(f \ge 0\) the policy-selected frequency in
Hz, and \(\Omega = 2\pi f\). The desired trot offsets are

\[
\delta = (0,\ \pi,\ \pi,\ 0).
\]

Blend standing and trotting with

\[
b(f) = S\!\left(\operatorname{clip}(f/f_{\rm lock}, 0, 1)\right),
\qquad S(x)=x^2(3-2x).
\]

Use the all-to-all phase dynamics

\[
\dot\theta_i = \Omega
+ b(f)\frac{k_t}{4}\sum_j
  \sin\!\left((\theta_j-\theta_i)-(\delta_j-\delta_i)\right)
+ (1-b(f))k_s\sin(\theta_s-\theta_i).
\]

`k_t` and `k_s` are in rad/s. Integrate once per physics substep with
`sim_dt = ctrl_dt / decimation`, then wrap phases into \([0,2\pi)\).
Start with `f_max = 3 Hz`, `f_lock = 1 Hz`, and `k_t = k_s = 10 rad/s`;
validate these in a phase-only probe before training. This first version has
no contact feedback, amplitude state, or process noise.

Use the same blend for the open-loop joint amplitude,

\[
q_{\mathrm{ref},i}=q_{\mathrm{stand}}+b(f)a\sin\theta_i.
\]

Thus zero frequency gives the static configured standing posture, while the
existing sinusoidal gait has full amplitude at and above `f_lock`.

### Equilibria

At \(f=0\), \(b=0\) and

\[
\dot\theta_i=k_s\sin(\theta_s-\theta_i).
\]

Thus \(\theta_i=\theta_s\) is a fixed point. For a perturbation
\(\epsilon_i=\theta_i-\theta_s\),
\(\dot\epsilon_i\approx-k_s\epsilon_i\), so it is locally stable.

For \(f\ge f_{\rm lock}\), \(b=1\). The solution

\[
\theta_i(t)=\psi(t)+\delta_i,\qquad \dot\psi=\Omega
\]

zeros every coupling term. Linearized relative-phase errors obey

\[
\dot\epsilon_i=k_t(\bar\epsilon-\epsilon_i),
\]

so all relative modes decay at rate \(k_t\). The common phase is neutral and
rotates at \(\Omega\); this is a stable relative equilibrium, not a fixed
absolute phase.

The standing and trotting solutions are parameter-dependent regimes, not two
simultaneous attractors at the same frequency.

## State and policy interface

- Change `phase` from `[num_envs, 1]` to `[num_envs, 4]`.
- Change `phase_obs` to eight interleaved values:
  `(sin FL, cos FL, ..., sin RR, cos RR)`.
- Keep `gait_frequency` as a `[num_envs, 1]` observation and add it to actor
  actions after the 12 joint residuals. Scale it by `f_max`, then clamp to
  `[0, f_max]`; negative policy output therefore selects exactly zero.
- Start without frequency filtering. Add a slew-rate limit only if evaluation
  shows action chatter.
- Build each leg's three-joint reference directly from its oscillator phase;
  remove the duplicated per-leg phase-offset expansion.

This changes actor input and output dimensions, so it starts a new experiment;
old checkpoints are not resumable under this interface.

## Standing and reward behavior

- At reset, initialize stopped-command environments at
  `(3π/2, 3π/2, 3π/2, 3π/2)` and zero frequency. Initialize moving-command
  environments on the trot manifold with a randomized common phase.
- Gate `trot_support` and `swing_contact` by a moving-command mask. At stand,
  all four feet should support the body and must not be classified as swing.
- Add a small reward for tracking a command-dependent cadence. Commands below
  the planar-speed deadband target zero; above it, the target rises linearly
  from `f_lock` to `f_max`. Velocity tracking can still override this weak
  prior.

  \[
  f_{\rm cmd}=\begin{cases}
  0, & \|v_{xy}^{\rm cmd}\|\le v_{\rm dead},\\
  f_{\rm lock}+(f_{\rm max}-f_{\rm lock})
  \operatorname{clip}\!\left(
  \frac{\|v_{xy}^{\rm cmd}\|-v_{\rm dead}}{v_{\rm max}-v_{\rm dead}},0,1
  \right), & \text{otherwise.}
  \end{cases}
  \]

- Keep the residual joint policy active at stand. If it still steps after the
  CPG reaches the stop equilibrium, address that separately with the existing
  stand-still/action-rate terms.

## Implementation sequence

1. Add the four-phase tensors, coupling equation, named config parameters, and
   deterministic equilibrium tests.
2. Route the four phases into the reference trajectory, contact rewards, reset
   logic, and eight-value observation.
3. Add the clipped frequency action and command-conditioned cadence reward.
4. Update policy-I/O names, deterministic evaluation setup, phase plots, and
   frequency/phase logging.
5. Run a CPU task step and tiny PPO smoke test, then exercise MuJoCo Warp and
   VSim because the CPG itself should be backend-independent.

## Checks before tuning

- At `f=0`, all phase derivatives are zero at `3π/2` and perturbations decay.
- Above `f_lock`, trot-relative phase errors decay while common phase advances
  at `2πf`.
- Frequency actions clip correctly and zero is exactly reachable.
- The standing reference is identical for all four legs; the rotating trot
  reproduces the current diagonal sequence.
- Stopped evaluations report frequency near zero, phase error to `3π/2`, low
  joint motion, and four-foot support.
- Moving evaluations report low trot relative-phase error, balanced scheduled
  stance use, low swing contact, and unchanged tracking/survival protocols.
