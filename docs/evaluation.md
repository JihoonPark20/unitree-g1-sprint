# Evaluation

## Spectral prior

```bash
spectral-prior evaluate --prior outputs/prior.pt \
  --library outputs/reference/library.npz --output outputs/prior_evaluation.json
```

Use the same library that trained the prior; a hash mismatch is rejected.

| Metric | Meaning |
| --- | --- |
| `L_rec` | Joint-angle MSE over five references, ten joints and 600 timestamps |
| `E_BA` | Peak-to-peak amplitude difference at 0.6/2.3 Hz against the generated boundary-reference frequencies |
| `FID_joint` | Fréchet distance between generated and reference joint-angle distributions |

## Policy

```bash
sprint-evaluate --checkpoint_path "<model>.pt" \
  --output outputs/policy_evaluation.json --save-trajectories
```

Keep `run_config.json` and `spectral_prior.pt` beside `<model>.pt`.

Defaults: **61 forward commands from 0 to 6 m/s**, spaced by 0.1 m/s; zero lateral/yaw commands; seeds 42/43/44; eight environments per seed and command. Each trial uses deterministic actions, an eight-second warmup with the default acceleration limit, then 20 seconds of measurement. Observation noise, pushes, mass and friction randomization are disabled; motor latency and reset-state variation remain enabled.

| Metric | Meaning |
| --- | --- |
| `E_qpos` | Mean absolute position error across the ten reference joints |
| `E_vel` | Mean absolute forward-velocity error against the requested command |
| `FID_joint` | Fréchet distance between actual and reference joint-angle distributions |
| Survival | Fraction of trials completed without failure |

A fall ends the trial; automatic resets do not replace failed samples. Incomplete trials produce `null` full aggregate scores and separate observed-only diagnostics. Check survival and `paper_grid_complete` alongside errors. Saved reports include settings and checkpoint hashes; `--save-trajectories` also exports NPZ traces.

Use `--speeds`, `--seconds`, `--warmup-seconds`, `--seeds` and `--num_envs` for shorter checks. Compare policies using identical settings. The metrics follow [SPRINT Eqs. 15–19](https://arxiv.org/html/2605.28549v1#S4.SS1), but use this project's MuJoCo protocol. `FID_joint` uses raw joint angles and is not established as numerically comparable to the paper’s FID.

### Controller measurements

Each policy evaluation also reports command bands `0-2` (including zero), `2-4` and `4-6` (including six). The upper endpoint belongs to the next band at 2 and 4 m/s. In `speed_bins`, check `trials`, `survival`, `samples`, covered speeds and `welch_windows` alongside each estimate. An empty band or unavailable estimate has a null value. The experiment compares tracking and spectral errors only in bands with survival of at least 0.9 in both arms.

| Field | Definition |
| --- | --- |
| `E_hi` | Welch power from 20 through 100 Hz divided by total power, computed per joint then averaged across the bilateral hip pitch, knee and ankle pitch joints. Applied torque is sampled at 200 Hz. Each trial uses complete one-second Hann windows, half overlap and constant detrending. Windows are pooled within the band. |
| `action_rate`, `action_acc` | Joint mean squared first and second differences of the clipped joint action at 50 Hz, respectively. |
| `joint_jerk` | Mean absolute second difference of the controlled joints' post-step angular velocities divided by control period squared (rad/s³). |
| `residual_rms` | RMS applied residual across substeps and controlled joints (rad). |
| `action_clip_fraction`, `anti_windup_fraction` | Fraction of clipped policy joint actions and fraction of joint integration updates restricted by anti-windup. |
| `torque_saturation` | Fraction of substep-joint samples at or above 99% of the actuator effort limit. |
| `p_jump_reference_rms_per_joint`, `p_jump_residual_rms_per_joint` | RMS target-induced P jump (Nm) between the previous last and next first substep. Joint order is in `protocol.control_joint_names`. The zero-residual reference target and the remaining applied offset include the effect of clipping. |
| `prior_reward` | Measured mean of the prior reward kernel before reward weighting. |
| `push_recovery` | Fraction of trials surviving the measurement horizon among those alive to receive a nonzero impulse. The report also identifies all warmup failures. |

Action differences and joint jerk use the valid-sample weighting of the SuperMimic benchmark. Each difference uses two or three samples from the same trial after warmup. A trial's terminal sample is included. [SuperMimic benchmark definitions](https://github.com/JihoonPark20/SuperMimic-Gym/blob/41340afdce0ff151cbdcd2d7d9adefe42fc750da/legged_gym/supermimic/benchmark.py#L45-L60)

With `--save-trajectories`, each seed/command NPZ includes `requested_torque`, `applied_torque` (Nm) and `joint_velocity` (rad/s), with axes [measured policy sample, physics substep, joint]. Use `env_index` and `time_s` to select an individual trajectory.

### Latency and impulse tests

`--latency-steps {0,1,2}` fixes motor-input latency to exactly that many policy ticks (20 ms each) in an evaluation. The same training flag applies a fixed delay to all training environments.

```bash
sprint-evaluate --checkpoint_path "<model>.pt" --latency-steps 2 \
  --output "$HOME/sprint-ode-runs/delay2.json"
sprint-evaluate --checkpoint_path "<A0-model>.pt" --calibrate-push \
  --output "$HOME/sprint-ode-runs/push_calibration.json"
sprint-evaluate --checkpoint_path "<model>.pt" --speeds 3 --warmup-seconds 8 \
  --push-impulse 4 --output "$HOME/sprint-ode-runs/push4.json"
```

Signed per-joint `p_jump_reference` and `p_jump_residual` arrays use axes [measured policy sample, joint]. `p_jump_valid` excludes the undefined boundary after reset.

The impulse in the last command is an example magnitude in N·s. Each exposed pelvis receives one heading-frame lateral impulse at its center of mass over the first physics substep after warmup. Signs alternate by environment and evaluation seed. Calibration holds the 3 m/s command, checkpoint, initial conditions and evaluation horizon fixed. It tests zero, doubles the impulse through 256 N·s until unconditional survival crosses 50%, then refines the interval. The output records every probe. A calibrated magnitude requires observed survival within one trial of half the cohort. Calibration records a null result if zero-impulse survival is already at or below 50%, the target is unbracketed or the tested impulses miss the tolerance.
