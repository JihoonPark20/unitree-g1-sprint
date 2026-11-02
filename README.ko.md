# SPRINT

[English](README.md)

Unitree G1을 위한 [SPRINT](https://arxiv.org/abs/2605.28549)의 비공식 구현입니다. `legged_gym` 구조를 따르며 MuJoCo CPU 시뮬레이션과 CPU/MPS/CUDA 정책 학습을 사용합니다. 논문 수준의 성능과 실기 성능은 검증하지 않았습니다.

데이터셋은 **LAFAN1 모션을 GMR로 G1에 리타게팅한 데이터**입니다. 가공된 reference library와 사전 학습 motion prior는 `outputs/`에 포함됩니다. 원본 LAFAN1 데이터와 GMR 도구는 포함하지 않습니다.

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

기본값은 환경 4096개, 시뮬레이션 스레드 10개입니다. 결과는 `logs/g1_sprint/`에 저장됩니다. `--resume`은 호환되는 체크포인트를 복원하며, `--max_iterations`는 추가로 학습할 횟수입니다.

출력된 Viser 주소(기본 **http://127.0.0.1:8080**)에서 학습을 확인합니다. `--headless`로 화면을 끌 수 있습니다. 상세 지표: `tensorboard --logdir logs`.

## 재생·평가

```bash
# 정책 없이 공중에 고정한 reference 모션 확인
sprint-reference

# 학습한 정책 재생
sprint-play --checkpoint_path "<model>.pt" --velocity 3 0 0

# 평가
spectral-prior evaluate
sprint-evaluate --checkpoint_path "<model>.pt"
```

`<model>.pt`를 학습한 모델의 경로로 바꾸세요. macOS 정책 재생은 같은 인자로 `mjpython -m legged_gym.scripts.play`를 사용합니다. 화면 없이 재생하려면 `--headless`를 추가합니다.

로봇 추적 중에도 드래그로 회전·이동하고 스크롤로 확대·축소할 수 있습니다. 정책 재생에서 **F**는 로봇 중심으로 이동, **R**은 시점 초기화입니다. 초기 시점은 `--camera-distance`, `--camera-azimuth`, `--camera-elevation`, `--camera-height`로 지정합니다. Viser에서는 **Camera**에서 시점과 방향 프리셋을 조절합니다.

Reference 재생은 `library` / `prior`, `kinematic` / `pd` 모드를 제공합니다. Prior 재학습: `spectral-prior train` 실행 후 `sprint-train --prior_checkpoint outputs/prior.pt`.

자세한 내용: [학습](docs/training.md) · [데이터 준비](docs/data.md) · [평가](docs/evaluation.md).

라이선스: `LICENSE`, `licenses/`, `resources/robots/UNITREE_LICENSE`.
