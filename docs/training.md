# Training configuration

```bash
sprint-train --rl_device auto --num_envs 4096 --num_threads 10
```

Edit [SPRINT configuration](../legged_gym/envs/g1_sprint/config.py) before starting a new run. Checkpoints retain their configuration and curriculum for `--resume`; incompatible configurations require a new run. Use TensorBoard (`tensorboard --logdir logs`) to inspect rewards, commands and curriculum progress.

## Rewards

[SPRINT’s reward terms and 13 published weights](https://arxiv.org/html/2605.28549v1#S3.SS3) are retained. Unpublished scales use [RuN](https://arxiv.org/html/2509.20696v1#S3.SS3) as a starting point:

| Term | Weight | Setting |
| --- | ---: | --- |
| Forward velocity | 2.0 | Exponential-kernel denominator 0.5 |
| Yaw velocity | 0.5 | Exponential-kernel denominator 1.0 |
| Prior positions | 0.1 per joint | Denominator 1/6; ten kernels total |
| Torso pitch | −2.0 | SPRINT’s speed-dependent bounds |
| Swing deficit | −0.75 | Heuristic unit penalty budget; no direct RuN equivalent |

Prior scaling matches RuN’s peak and small-error sensitivity on the ten-joint subset, not its complete reward function. These are tuning defaults, not the SPRINT authors’ unpublished settings.

## Command curriculum

The [RLvRL-based grid curriculum](../legged_gym/envs/curriculums/grid_adaptive.py) samples forward velocity and yaw rate jointly:

- Initial range: **0–0.8 m/s, ±0.1 rad/s**, plus 5% standing samples.
- Target range: **0–6 m/s, ±0.7 rad/s**; lateral velocity stays zero.
- Grid resolution: **0.2 m/s × 0.2 rad/s**; commands resample every 10 seconds.
- Promotion requires both normalized tracking scores ≥0.8, at least two seconds after the acceleration ramp settles, and no fall in the segment. Dedicated standing samples do not promote.
- Successful cells and their immediate rectangular neighbors gain 0.2 sampling weight, capped at 1. Previously learned cells remain available.

`Commands/*` shows command distribution and prior activation; `Curriculum/*` shows coverage and tracking success. If expansion stalls, inspect tracking scores, episode duration and reward contributions. A wider command range alone does not demonstrate successful locomotion.

한국어: 설정 파일을 수정한 뒤 새 학습을 시작하세요. 보상은 SPRINT 항목을 유지하며 미공개 스케일에 RuN을 참고합니다. 커리큘럼은 구간별 추종 성능에 따라 확장하므로 명령 범위와 실제 생존·추종 성능을 함께 확인하세요.
