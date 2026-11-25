# Training configuration

```bash
sprint-train --rl_device auto --num_envs 4096 --num_threads 10
```

Edit [SPRINT configuration](../legged_gym/envs/g1_sprint/config.py) before starting a new run. `--resume` restores the checkpoint's saved configuration and curriculum, including its original rewards; local reward edits apply to new runs or `--warm_start`. Use TensorBoard (`tensorboard --logdir logs`) to inspect rewards, commands and curriculum progress.

The terminal uses the standard rsl_rl layout **every iteration**, including on resume: throughput, losses, learning rate, mean reward, episode length (steps), action standard deviation, all reward terms, curriculum speed and timing. Iteration numbers match TensorBoard and checkpoint indices. Use `--log_interval 10` for less frequent output. TensorBoard retains all metrics every iteration, including tracking and per-joint diagnostics; its Text tab contains model structure and settings. The Viser monitor also updates every iteration.

Training uses forward commands with foot-crossing prevention. Reward changes require a new run or `--warm_start`; older models remain playable with their saved settings.

- Exploration: initial action standard deviation **0.3**, bounded to **0.1–0.5** after every optimizer step; PPO entropy coefficient **0.001**.
- Rollouts: **96 steps** per environment (1.92 s). The actor observes no gait phase, as in the paper, so its LSTM must infer phase. A shorter window would not span one 0.61 Hz stride (1.64 s) during truncated backpropagation through time. The critic adds a `[256, 128]` head after its LSTM, and both networks normalize observations. SPRINT publishes neither setting.
- Residual targets: **±0.5 rad**, narrowed to **±0.15 rad** for shoulder roll/yaw. Wrists retain fixed PD targets.
- PD uses reference phase velocity (`reference_velocity`) with the configured gains; residuals remain position offsets. Targets update at 50 Hz and feedback at 200 Hz in training and reference playback. This is an implementation choice; SPRINT does not publish PD gains or a velocity-target rule. Changed controllers require a new policy run. High-speed ground tracking remains to be validated.
- `reference_lookahead` advances the held position target along the reference velocity at each 200 Hz substep. The P and D terms then agree, and a zero residual reaches `q_ref(t+dt)` instead of lagging it. In zero-residual probes, this raised the same-instant prior kernel at every tested speed (for example, 0.746 to 0.773 at 0.6 m/s).
- Torque limits follow the MJCF actuator ranges. Ankles and waist roll/pitch use **50 Nm**; the inherited 25 Nm default is not used.
- Pushes and control latency are disabled for baseline gait learning. Observation noise and mass/friction randomization remain enabled. Enable and validate pushes/latency separately before robustness or hardware testing.

To initialize a new run from a compatible policy:

```bash
sprint-train --warm_start "<model>.pt" --rl_device auto --num_envs 4096 --num_threads 10
```

`--warm_start` transfers the actor's mean policy and uses the new training settings. Critic, optimizer, exploration, iteration counter and curriculum start fresh. The robot, controller, action bounds and spectral prior must match. This is an optional starting point, not a guarantee of faster convergence. Use `--resume` only to continue an existing compatible run with its saved settings.

## Rewards

[SPRINT’s reward terms and 13 published weights](https://arxiv.org/html/2605.28549v1#S3.SS3) are retained. Unpublished scales are tuned from a [RuN](https://arxiv.org/html/2509.20696v1#S3.SS3)-informed starting point:

| Term | Weight | Setting |
| --- | ---: | --- |
| Forward velocity | 2.0 | Exponential-kernel denominator 0.1 |
| Yaw velocity | 1.0 | Exponential-kernel denominator 0.25 |
| Prior positions | 0.3 per joint | Denominator 0.03; ten kernels, maximum total 3; `q` and `q_ref` at the same instant (Eq. 11) |
| Torso pitch | −2.0 | SPRINT’s speed-dependent bounds |
| Swing deficit | −0.75 | Heuristic unit penalty budget; no direct RuN equivalent |

These are tuning defaults, not the SPRINT authors’ unpublished settings or RuN’s exact reward function. Long-run convergence remains to be validated.

**Foot-crossing adaptation:** the close-feet term uses `max(0, 0.12 - d_y)` with weight −100, where `d_y` is left-minus-right foot separation along the robot's heading-frame y axis. Unlike the paper's absolute distance, this also penalizes reversed foot ordering. All 18 reward weights are retained.

**Swing-cap adaptation:** `air_height_swing_cap` credits foot clearance only while the foot's air time is within the Eq. 12 swing duration (`0.4 / f`). Without the cap, a foot held up indefinitely earns up to 5 per second, more than velocity tracking and survival combined. The kernel and weight follow the paper; the cap is an addition.

**Speed calibration (deviation):** `reference.speed_calibration = "g1_kinematic"` changes which speed label the prior receives. The prior gets the label whose generated cycle moves G1 at the commanded speed without foot slip. The prior's speed labels are human speeds. Retargeted onto G1's shorter legs, the same cycle moves the pelvis 15–35% slower. For example, the 0.66 m/s clip gives about 0.53 m/s. Without calibration, the reference undershoots the command, and the narrow prior kernel penalizes the residual needed to close the gap. In a 197M-step run, both earlier configurations plateaued near 0.5 m/s, and the curriculum stalled below 1 m/s. Calibration runs at environment start. It uses forward kinematics with a level, fixed pelvis, and each foot's stance is the lowest 60% of its cycle. Cadence rises 8–16% above the paper's map, for example 0.66 Hz instead of 0.61 Hz at 0.6 m/s. The calibrated prior saturates near 4.1 m/s; faster commands rely on the residual. The knots are part of the checkpoint contract. `g1_sprint_paper_map` keeps the paper's map for comparison. Run `sprint-diagnose speed` to print the label and frequency used per command.

**Standing adaptation:** `air_height_command_gate` fades the foot-lift reward with the prior's speed blend: zero at rest and full strength from 0.4 m/s. Lateral or yaw commands retain the reward for stepping. This command condition is an addition to the paper's reward definition; existing checkpoints retain their original behavior.

An episode terminates if `d_y < −0.02 m` persists for **0.10 s** (five consecutive control steps). A brief crossing can recover, but any command segment crossing that margin cannot promote the curriculum. This is a true failure, including at a time limit; PPO does not bootstrap it as a timeout. These are implementation additions, not claims about the paper's termination rules. No hip joint is locked.

## Command curriculum

The [RLvRL-based curriculum](../legged_gym/envs/curriculums/grid_adaptive.py) expands forward velocity with lateral and yaw commands fixed at zero:

- Initial forward range: **0–0.4 m/s**, plus 5% standing samples. This follows the paper's expansion "from static standing". With this controller, a 0.4–0.8 m/s start failed to keep the robot upright: episodes still averaged 2.6 s after 190M steps. A 0–0.4 m/s start reached 7.6 s episodes by 50M steps. The cost is a double-support shuffle before stepping returns (single support fell to 0.08, then recovered).
- Target forward range: **0–6 m/s**; yaw stays exactly **0 rad/s** throughout training.
- Grid: **30 forward cells of 0.2 m/s**, one fixed yaw cell; commands resample every 10 seconds.
- Promotion requires the normalized forward tracking score ≥0.8, at least two seconds after the acceleration ramp settles, and no fall in the segment. Dedicated standing samples do not promote.
- A physical-progress gate additionally requires command speed ≥0.2 m/s and mean signed forward speed ≥50% of the command over the settled samples. This project-specific addition prevents standing from unlocking neighboring cells. An earlier forward-MAE ≤0.1 m/s condition is no longer used: it blocked a policy tracking 1.0 m/s at 0.90 m/s, whose RLvRL score (0.87) passed. `g1_sprint_legacy` keeps it.
- Successful cells and their immediate forward neighbors gain 0.2 sampling weight, capped at 1. Previously learned cells remain available.
- Optional focus band (off by default; not in the paper or RLvRL): `--curriculum-focus LO HI FRACTION` samples that fraction of commands uniformly in [LO, HI] m/s once every cell overlapping the band is active. Promotion rules are unchanged, and the setting is saved in the run snapshot, so `--resume` keeps it; `--curriculum-focus 0 0 0` turns an inherited band off. Use it to practice at a failure boundary, e.g. `--resume --checkpoint_path <model>.pt --curriculum-focus 1.25 1.30 0.25`.

Yaw tracking reward remains active at a zero target, with its weight and kernel unchanged. Yaw score/error do not gate forward promotion in this experiment. Zero yaw command does not physically lock rotation: inspect yaw error and trajectory heading alongside speed, because forward velocity is measured in the robot's heading frame. This configuration does not train turning or backward locomotion. The prior takes phase and speed-derived frequency, not a turn command.

`Commands/*` shows command distribution and prior activation; `Curriculum/*` shows coverage and tracking success. If expansion stalls, inspect tracking scores, episode duration and reward contributions. A wider command range alone does not demonstrate successful locomotion.

`Control/*` reports residual size and clipping; `TorqueSaturation/*` reports each joint's fraction of physics steps at ≥99% of its torque limit. For each policy-controlled joint, `ResidualRMS/*` reports bounded residual RMS in radians, `ResidualLimitFraction/*` reports time at ≥99% of the effective residual limit, and `ActionClipFraction/*` reports the fraction of requested actions clipped. These are measured before joint-position clipping; held wrists are excluded. `Tracking/*` reports prior, forward and yaw errors. `Gait/*` reports support fractions, signed foot separation and crossed-foot time fraction. `Episode/feet_crossing_fraction` reports the fraction of completed episodes terminated by persistent crossing. Inspect these alongside speed and survival; diagnostics use actual episode durations, including failed episodes.

## Comparing configurations

Three tasks are registered:

- `g1_sprint`: the default configuration.
- `g1_sprint_paper_map`: the same, but with the paper's speed→frequency map instead of the speed calibration.
- `g1_sprint_legacy`: the pre-diagnosis settings. These are the one-step-offset prior kernel, no PD lookahead, 25 Nm ankle/waist limits, the uncapped foot-lift reward, 32-step rollouts, a linear critic head and no observation normalization.

Checkpoints trained before these changes keep their saved settings when resumed or played.

```bash
sprint-diagnose probe --tasks g1_sprint_legacy g1_sprint --speeds 0 0.6 1.5 3   # zero-residual rollouts
sprint-diagnose speed                                                           # prior label/frequency per command
bash scripts/compare.sh                                                         # train, then summarize
```

`scripts/compare.sh` trains every task in `TASKS` for `STEPS_M` million environment steps per seed. Iterations are derived from each task's rollout length, so the sample budgets are equal. Completed runs are skipped, and the script ends with `sprint-diagnose compare`. Defaults: `TASKS="g1_sprint_paper_map g1_sprint"`, `SEEDS="1 2 3"`, `STEPS_M=400`, `LOG_ROOT=logs/comparison_v3`. `NUM_ENVS`, `NUM_THREADS` and `RL_DEVICE` can also be overridden. On a 10-core Apple-silicon CPU with 4096 environments, `g1_sprint` collects about 9,300 steps/s, roughly 12 h per 400M-step run.

For each task, `compare` reports the environment steps (millions) and wall-clock hours needed to reach three milestones, as mean ± std over seeds:

- episode length ≥ 15 s
- curriculum maximum command ≥ 2 m/s
- curriculum maximum command ≥ 4 m/s

Each milestone uses a 10-iteration rolling mean. `compare` also reports final tracking, gait and posture metrics. `Tracking/prior_mae_rad` always uses the same-instant reference, so all tasks are comparable.

Run `sprint-reference --headless` for one second of reference playback. Use
`--source prior --mode pd` to exercise torque-limited suspended PD tracking.

`control.target_interp` selects the position target within each 50 Hz policy step.
The default `None` retains the checkpoint's `reference_lookahead` setting. In
the stock `g1_sprint` task this advances the held target by 5, 10, 15 and 20 ms
along the reference velocity. Explicit `zoh` holds the position fixed.
Explicit `linear` extrapolates the issued reference at the same four offsets
before adding the residual and enforcing joint limits. `exact` evaluates the
frozen prior at each offset and uses a 5 ms centered difference there for the
velocity target. A delayed command retains its original reference and residual.

### Residual dynamics

Set `control.residual_mode` to `position` (the default), `lowpass`, or
`integrator`. In position and lowpass modes, `action_scale * action` commands
a joint offset in radians. In integrator mode it commands rad/s. The integrator
uses `residual_leak` in s⁻¹, including zero for a pure integrator. A lowpass
controller requires a positive leak and approaches the commanded offset with
time constant `1 / residual_leak`.

The controller updates the state every 5 ms. It limits stored offsets to
`residual_limits` and restricts each outward increment to the available joint
range and PD torque. An increment toward the interior can proceed while the
opposite direction is saturated. Each episode starts with zero residual and
empty input history. A delayed controller consumes the reference, its velocity
and the residual command from the same policy tick.

`control.residual_obs=True` appends the applied offset divided by the joint's
residual limit to the actor observation. `residual_feedforward` multiplies
the realized residual velocity in the D target. For a filtered state this
uses the accepted 5 ms increment. For an absolute position command it uses
the difference between successive delayed commands over the 20 ms policy
period. `residual_mix_direct` sets the fraction of a lowpass target applied
directly alongside the filtered state. Its direct jump is excluded from
the velocity target.

### Integrated action increments

For C1-env, set `control.residual_mode="integrator"`, `control.residual_leak=0`
and `control.increment_kappa` to the desired fraction of each joint's residual
limit per policy interval. At 50 Hz the simulator clips the scaled Gaussian
sample to that increment, divides by 20 ms to obtain rad/s, and integrates it
at 200 Hz. Its action observation contains the bounded requested increment in
policy action units. Torque and joint limits can reduce the realized increment.

For C1-mean, set `train.policy.integrate_mean=True`, `control.residual_mode="position"`
and `control.increment_kappa`. The actor adds its bounded increment to the
previous clean mean, caps the result at the joint's residual limit, and samples
an absolute action from the usual Gaussian. Exploration standard deviation uses
the same action units as A0. Each PPO transition stores the clean previous mean
alongside its observations. The actor clears that history on episode reset.
Deterministic playback applies the clean mean directly. Both C1 arms use the
experiment grid `increment_kappa` ∈ {0.1, 0.2, 0.4}.

### Phase modulation and learned leak

`reference.phase_obs=True` adds the sine and cosine of the current oscillator
phase to the actor. With `reference.phase_modulation=0.1`, append one scalar
phase action after the joint actions. It multiplies the current phase rate by
`1 + 0.1 * tanh(action)`. The spectral prior retains its speed-conditioned
waveform. Phase advance, substep target timing and reward timing use the new
cadence.

With `control.learned_leak=True`, append a scalar leak action last. Its rate
in s⁻¹ is `control.lambda_min + softplus(action)`. The scalar and joint
commands undergo the same input delay. Set the initial `residual_leak` above
`lambda_min` (default 0). The decoder initializes the leak channel to that
rate for every observation. Joint observations, bounds and controller metrics
continue to use 23 physical joints.

### Train a residual controller arm

Run `sprint-train --headless --arm A0 --seed 0` for the stock controller.
Each named arm writes under `logs/g1_sprint_<arm>/`. The timestamped run
directory contains the resolved environment and PPO settings in `run_config.json`.
Use a fresh training run for a changed arm. Playback and evaluation read the
controller, observation layout and actor architecture beside the checkpoint.

| Arm | Controller changes | Grid option |
| --- | --- | --- |
| A0 | Stock substep reference lookahead and direct residual position | |
| A1 | Evaluate exact reference position and velocity at each substep | |
| A1-R | A1 with a penalty on the joint-mean squared difference between consecutive bounded actions in one episode | `--action-rate-weight W` (positive, default 0.0001) |
| F | A1 with a lowpass residual, λ = 2π times cutoff in Hz | `--cutoff-hz {5,10,20}` (default 10) |
| B-obs, B-ff, B | F with residual state observation, unit residual velocity feedforward, or both | Same cutoff as F |
| Hyb | Selected B-family controller plus a direct position fraction | `--base-arm`, `--beta {0.2,0.4}` (default 0.2) |
| D-ctrl, D | Add phase observation to the selected parent. D enables 10% phase-rate modulation. | `--base-arm` and the parent's grid options |
| C1-env, C1-mean | Integrate bounded sampled increments in physics or integrate the clean Gaussian mean in the actor | `--base-arm`, `--kappa {0.1,0.2,0.4}` (default 0.2) |
| C2 | Learn one leak rate around the parent controller's initial bandwidth | `--base-arm` and its cutoff |

Hyb and C variants accept B-obs, B-ff or B as a parent (default B).
D-ctrl and D also accept A0, A1, F and Hyb. For a Hyb parent, pass its B-family
choice through `--hyb-base-arm`. For example, these two commands use the same
parent and train at one tick of input delay.

```bash
sprint-train --headless --arm D-ctrl --base-arm Hyb --hyb-base-arm B-ff --cutoff-hz 5 --beta 0.4 --latency-steps 1
sprint-train --headless --arm D      --base-arm Hyb --hyb-base-arm B-ff --cutoff-hz 5 --beta 0.4 --latency-steps 1
```

Use `--residual-scale VALUE` to set the physical action scale for an RMS-matched
retraining. `--latency-steps {0,1,2}` sets a fixed training delay in policy
ticks. Preserve the checkpoint's adjacent configuration and prior for playback.
Older checkpoints retain their recorded control settings and receive the A0 arm label.

### Choose a training horizon for high-speed comparisons

Read the command curriculum and evaluation results from earlier runs with the
same task, reward, prior, rollout length and number of environments. Record
the first promotion out of the initial command range and the number of PPO
updates required to survive at the intended speed. A flat command limit before
the first promotion measures the time spent in the initial range. Use a
checkpoint evaluation to establish what speeds the policy can sustain.

For a low-cost controller check, start with a saved policy that survives the
target command. Evaluate its existing controller and the intervention with
the same weights, reference, commands and reset seeds using the frozen-policy
options in `docs/evaluation.md`. Compare high-frequency torque only where both
conditions meet the study's survival and observation requirements. To test
learning with a new actor, use independent training seeds, matched rollout
budgets and the same evaluation gate. If the budget buys only initial learning,
report the high-speed trained-policy question as inconclusive.
