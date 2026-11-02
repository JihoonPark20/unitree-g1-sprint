# SPRINT

[한국어](README.ko.md)

Unofficial implementation of [SPRINT](https://arxiv.org/abs/2605.28549) for Unitree G1, following the `legged_gym` structure. Uses MuJoCo CPU simulation and CPU/MPS/CUDA policy training. Paper-level and hardware performance are not validated.

The dataset uses **LAFAN1 motions retargeted to G1 with GMR**. The processed reference library and pretrained motion prior are included in `outputs/`; raw LAFAN1 data and GMR tools are not included.

## Install

Python 3.10–3.13; run from the repository root.

```bash
python -m pip install -e .
```

## Train

```bash
sprint-train --rl_device auto --num_envs 4096 --num_threads 10
sprint-train --resume --max_iterations 1000
```

Defaults: 4096 environments, 10 simulation threads. Runs are saved to `logs/g1_sprint/`. `--resume` restores a compatible checkpoint; `--max_iterations` specifies additional iterations.

Open the printed Viser URL (default **http://127.0.0.1:8080**) to monitor training, or use `--headless` to disable it. Detailed metrics: `tensorboard --logdir logs`.

## View and evaluate

```bash
# Suspended reference motions, without a policy
sprint-reference

# Trained policy
sprint-play --checkpoint_path "<model>.pt" --velocity 3 0 0

# Evaluation
spectral-prior evaluate
sprint-evaluate --checkpoint_path "<model>.pt"
```

Replace `<model>.pt` with your trained model path. For policy playback on macOS, use `mjpython -m legged_gym.scripts.play` with the same arguments. Add `--headless` for playback without a window.

Drag to orbit/pan and scroll to zoom while following the robot. In policy playback, **F** centers on the robot and **R** resets the view; `--camera-distance`, `--camera-azimuth`, `--camera-elevation`, and `--camera-height` set the initial view. Viser provides these controls and direction presets in **Camera**.

Reference playback offers `library` / `prior` and `kinematic` / `pd` modes. To retrain the prior: `spectral-prior train`, then `sprint-train --prior_checkpoint outputs/prior.pt`.

Details: [training](docs/training.md) · [data preparation](docs/data.md) · [evaluation](docs/evaluation.md).

Licenses: `LICENSE`, `licenses/`, and `resources/robots/UNITREE_LICENSE`.
