# SPRINT

[English](README.md)

Unitree G1을 위한 [SPRINT](https://arxiv.org/abs/2605.28549)의 비공식 구현입니다. `legged_gym` 구조를 따르며, MuJoCo CPU 시뮬레이션과 CPU/MPS/CUDA 정책 학습을 사용합니다. 논문 수준의 성능과 실기 성능은 검증하지 않았습니다.

데이터셋은 **LAFAN1 모션을 GMR로 G1에 리타게팅하고 NPZ로 가공한 데이터**입니다. 선별된 reference library와 사전 학습 prior는 `outputs/`에 포함됩니다. 원본 LAFAN1 데이터와 GMR 도구는 포함하지 않습니다.

## 설치

Python 3.10–3.13 환경에서 저장소 루트를 기준으로 실행합니다.

```bash
python -m pip install -e .
```

## 학습

```bash
sprint-train --rl_device auto --num_envs 4096 --num_threads 10
sprint-train --resume --max_iterations 1000
```

정책은 23개 관절을 제어하며, 손목 6개 관절은 일정한 PD 목표 각도를 유지합니다(roll: 왼쪽 +0.15/오른쪽 −0.15 rad, pitch/yaw: 0). 손목 고정은 구현 선택입니다.

학습은 SPRINT 보상 항목을 유지하고 **미공개 스케일에 RuN을 참고**하며, **RLvRL 격자 커리큘럼**으로 저속 보행부터 0–6 m/s·±0.7 rad/s 범위까지 확장합니다. 자세한 내용은 [학습 설정](docs/training.md)을 참고하세요. `--resume`은 호환되는 체크포인트의 설정과 커리큘럼을 복원합니다.

**http://127.0.0.1:8080**에서 Viser 모니터를 엽니다(실제 주소는 실행 시 출력). 로봇, 명령·램프 적용·실제 속도, 참조 주파수와 학습 그래프를 표시합니다. `--headless`를 붙이면 모니터를 끕니다.

학습 기본값은 `num_envs=4096`, `num_threads=10`입니다. 결과는 `logs/g1_sprint/`에 저장됩니다. `--max_iterations`는 이번 실행에서 추가로 학습할 횟수입니다. Prior를 다시 학습하려면 `spectral-prior train` 실행 후 PPO 학습에 `--prior_checkpoint outputs/prior.pt`를 지정합니다. Library 재생성은 [데이터 준비](docs/data.md)를 참고하세요.

## 재생·평가

골반을 공중에 고정한 상태로 reference 모션을 확인합니다(Viser, 정책 불필요).

```bash
python -m legged_gym.scripts.reference
```

5개 library 모션 또는 생성 모션인 `prior`를 선택합니다. `kinematic`은 데이터 자세를 그대로 표시하고, `pd`는 토크 제한이 있는 추종을 확인합니다. Library에는 10개 관절만 있으며 나머지는 기본 자세입니다. 원본 관절 데이터는 `--motion /path/to/gmr.npz`로 열고, 손목 모션을 보려면 **Hold wrists**를 해제하세요. 패키지 재설치 후에는 `sprint-reference`도 사용할 수 있습니다.

`pd`에서 **Gravity compensation (diagnostic)**을 켜면 모터 토크 한계 내에서 중력을 보상합니다(`--gravity-comp`). 뷰어 전용 옵션으로 학습 제어기는 바뀌지 않으며, 관성·게인·토크 포화로 인한 추종 오차는 남을 수 있습니다.

```bash
sprint-play --checkpoint_path "<model>.pt" --headless --velocity 3 0 0
spectral-prior evaluate
sprint-evaluate --checkpoint_path "<model>.pt"
```

`<model>.pt`를 학습한 정책 모델의 경로로 바꾸세요.

화면을 보려면 `--headless`를 제거합니다. macOS에서는 `mjpython -m legged_gym.scripts.play`를 사용합니다. 학습 그래프: `tensorboard --logdir logs`.

라이선스: `LICENSE`, `licenses/`, `resources/robots/UNITREE_LICENSE`.

[평가 프로토콜](docs/evaluation.md)은 prior 지표와 61개 속도 명령의 정책 평가를 제공합니다. 포함된 데이터는 논문의 30→60 Hz 전처리를 따르지만, 측정된 기준점은 저자의 원본과 일부 다릅니다.
