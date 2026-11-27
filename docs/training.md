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

## Unitree actuator model

Three switches, all off by default, apply the actuator model that Unitree
publishes in `unitree_rl_lab` (`assets/robots/unitree_actuators.py`).

| Setting | Effect |
| --- | --- |
| `torque_speed_envelope` | Clip PD torque at every physics step. Driving torque (same sign as joint velocity) is limited to `y1`, braking torque to `y2`. Both are flat up to `x1` and fall linearly to zero at `x2`. The effort limit stays an outer bound. `Control/torque_saturation` counts samples at 99% of this envelope. |
| `actuator_friction` | Subtract `fs * tanh(v / 0.01) + fd * v` after clipping. |
| `actuator_armature` | Use each actuator's reflected inertia instead of the uniform `armature`. |

| Actuator | Joints | `y1` / `y2` (Nm) | `x1` / `x2` (rad/s) | `fs` (Nm), `fd` (Nm s/rad) | armature (kg m²) |
| --- | --- | --- | --- | --- | --- |
| N7520-14.3 | hip pitch, hip yaw, waist yaw | 71 / 83.3 | 22.63 / 35.52 | 1.6, 0.16 | 0.0102 |
| N7520-22.5 | hip roll, knee | 111 / 131 | 14.5 / 22.7 | 2.4, 0.24 | 0.0251 |
| N5020-16 | shoulder, elbow, wrist roll | 24.8 / 31.9 | 30.86 / 40.13 | 0.6, 0.06 | 0.0036 |
| N5020-16x2 | ankle, waist roll/pitch | 49.6 / 63.8 | 30.86 / 40.13 | 1.2, 0.12 | 0.0072 |
| W4010-25 | wrist pitch, wrist yaw | 4.8 / 8.6 | 15.3 / 24.76 | 0.6, 0.06 | 0.0043 |

`actuator_models` holds this table and `joint_actuators` maps joints to it by
regex. N5020-16x2 approximates the ankle and waist roll/pitch linkages as two
N5020-16 in parallel, as Unitree's G1 config does for effort and armature.
Some G1 builds use N7520-22.5 for hip pitch; change `joint_actuators` for those.
`envelope_velocity_scale` multiplies `x1` and `x2`; `envelope_torque_scale`
multiplies `y1` and `y2`.

`--torque-speed-envelope`, `--actuator-friction`, `--actuator-armature`
(`on|off`), `--envelope-velocity-scale` and `--envelope-torque-scale` override
these settings in training and evaluation. Checkpoints saved without them load
with all three off, and a saved policy may be evaluated, resumed or warm-started
under any actuator model.

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

`Commands/*` shows command distribution and prior activation; `Curriculum/*` shows coverage and tracking success. A single success opens a neighbor partly, so with thousands of parallel segments cells open long before they are mastered. `Curriculum/frontier_success_rate` is the recent success rate of the fastest open cell and `Curriculum/mastered_command` the fastest cell with at least 20 recent segments and a 50% recent success rate, over roughly each cell's last 100 segments; both are printed on the console. If expansion stalls, inspect these, episode duration and reward contributions. A wider command range alone does not demonstrate successful locomotion.

`--initial-command-range LO HI` sets the forward range the curriculum starts with, for example when a warm-started actor can already run. A warm start may switch between `g1_sprint` and `g1_sprint_paper_map` (speed calibration on or off); a resume may not.

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
