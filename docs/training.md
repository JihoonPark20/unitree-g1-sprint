# Training configuration

```bash
sprint-train --rl_device auto --num_envs 4096 --num_threads 10
```

Edit [SPRINT configuration](../legged_gym/envs/g1_sprint/config.py) before starting a new run. `--resume` restores the checkpoint's saved configuration and curriculum, including its original rewards; local reward edits apply to new runs or `--warm_start`. Use TensorBoard (`tensorboard --logdir logs`) to inspect rewards, commands and curriculum progress.

The terminal uses the standard rsl_rl layout **every iteration**, including on resume: throughput, losses, learning rate, mean reward, episode length (steps), action standard deviation, all reward terms, curriculum speed and timing. Iteration numbers match TensorBoard and checkpoint indices. Use `--log_interval 10` for less frequent output. TensorBoard retains all metrics every iteration, including tracking and per-joint diagnostics; its Text tab contains model structure and settings. The Viser monitor also updates every iteration.

Training uses forward commands with foot-crossing prevention. Reward changes require a new run or `--warm_start`; older models remain playable with their saved settings.

- Exploration: initial action standard deviation **0.3**, bounded to **0.1–0.5** after every optimizer step; PPO entropy coefficient **0.001**.
- Residual targets: **±0.5 rad**, narrowed to **±0.15 rad** for shoulder roll/yaw. Wrists retain fixed PD targets.
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
| Prior positions | 0.3 per joint | Denominator 0.03; ten kernels, maximum total 3 |
| Torso pitch | −2.0 | SPRINT’s speed-dependent bounds |
| Swing deficit | −0.75 | Heuristic unit penalty budget; no direct RuN equivalent |

These are tuning defaults, not the SPRINT authors’ unpublished settings or RuN’s exact reward function. Long-run convergence remains to be validated.

**Foot-crossing adaptation:** the close-feet term uses `max(0, 0.12 - d_y)` with weight −100, where `d_y` is left-minus-right foot separation along the robot's heading-frame y axis. Unlike the paper's absolute distance, this also penalizes reversed foot ordering. All 18 reward weights are retained.

**Standing adaptation:** `air_height_command_gate` fades the foot-lift reward with the prior's speed blend: zero at rest and full strength from 0.4 m/s. Lateral or yaw commands retain the reward for stepping. This command condition is an addition to the paper's reward definition; existing checkpoints retain their original behavior.

An episode terminates if `d_y < −0.02 m` persists for **0.10 s** (five consecutive control steps). A brief crossing can recover, but any command segment crossing that margin cannot promote the curriculum. This is a true failure, including at a time limit; PPO does not bootstrap it as a timeout. These are implementation additions, not claims about the paper's termination rules. No hip joint is locked.

## Command curriculum

The [RLvRL-based curriculum](../legged_gym/envs/curriculums/grid_adaptive.py) expands forward velocity with lateral and yaw commands fixed at zero:

- Initial forward range: **0.4–0.8 m/s**, plus 5% standing samples.
- Target forward range: **0–6 m/s**; yaw stays exactly **0 rad/s** throughout training.
- Grid: **30 forward cells of 0.2 m/s**, one fixed yaw cell; commands resample every 10 seconds.
- Promotion requires the normalized forward tracking score ≥0.8, at least two seconds after the acceleration ramp settles, and no fall in the segment. Dedicated standing samples do not promote.
- A physical-progress gate additionally requires command speed ≥0.2 m/s, mean signed forward speed ≥50% of the command and forward MAE ≤0.1 m/s over the settled samples. This project-specific addition prevents standing from unlocking neighboring cells.
- Successful cells and their immediate forward neighbors gain 0.2 sampling weight, capped at 1. Previously learned cells remain available.

Yaw tracking reward remains active at a zero target, with its weight and kernel unchanged. Yaw score/error do not gate forward promotion in this experiment. Zero yaw command does not physically lock rotation: inspect yaw error and trajectory heading alongside speed, because forward velocity is measured in the robot's heading frame. This configuration does not train turning or backward locomotion. The prior takes phase and speed-derived frequency, not a turn command.

`Commands/*` shows command distribution and prior activation; `Curriculum/*` shows coverage and tracking success. If expansion stalls, inspect tracking scores, episode duration and reward contributions. A wider command range alone does not demonstrate successful locomotion.

`Control/*` reports residual size and clipping; `TorqueSaturation/*` reports each joint's fraction of physics steps at ≥99% of its torque limit. `Tracking/*` reports prior, forward and yaw errors. `Gait/*` reports support fractions, signed foot separation and crossed-foot time fraction. `Episode/feet_crossing_fraction` reports the fraction of completed episodes terminated by persistent crossing. Inspect these alongside speed and survival; diagnostics use actual episode durations, including failed episodes.

한국어: 기본 설정은 yaw·횡방향 명령을 0으로 유지하며 처음부터 학습합니다. 발 간격의 부호를 보존해 교차 자세를 벌점 처리하고, 2 cm 이상 순서가 뒤집힌 상태가 0.1초 지속되면 종료합니다. 교차한 명령 구간은 승급하지 않습니다. 이 부분은 논문 식·종료 조건에 대한 구현상의 변경입니다. 속도·생존율과 교차 발생률을 함께 확인하세요.
