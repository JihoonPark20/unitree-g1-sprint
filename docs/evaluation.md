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

한국어: 정책은 기본적으로 0–6 m/s의 61개 속도에서 평가합니다. 낙상한 실행은 실패로 기록하며 전체 점수는 `null`이 됩니다. 같은 설정으로 생존율과 오차를 함께 비교하세요.
