# SPRINT

[한국어](README.ko.md)

Unofficial implementation of [SPRINT](https://arxiv.org/abs/2605.28549) for Unitree G1, following the `legged_gym` structure. Uses MuJoCo CPU simulation and CPU/MPS/CUDA policy training. Paper-level and hardware performance are not validated.

The dataset is built from **LAFAN1 motions retargeted to G1 with GMR** and exported as NPZ. A curated reference library and pretrained spectral prior are included in `outputs/`. Raw LAFAN1 data and GMR tools are not included.

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

The policy controls 23 joints; six wrist joints retain fixed PD targets (roll: left +0.15/right −0.15 rad, pitch/yaw: 0). Wrist holding is an implementation choice.

Training uses SPRINT rewards with **RuN-informed unpublished scales** and an **RLvRL grid curriculum**, expanding from slow walking toward 0–6 m/s and ±0.7 rad/s. See [training settings](docs/training.md). `--resume` requires a compatible checkpoint and restores its saved configuration and curriculum.

Open **http://127.0.0.1:8080** for the Viser monitor (the actual URL is printed at startup). It shows the robot, commanded/ramped/actual speed, reference frequency and training curves. Use `--headless` to disable it.

Training defaults: `num_envs=4096`, `num_threads=10`. Runs are saved to `logs/g1_sprint/`. `--max_iterations` specifies additional iterations for this invocation. To retrain the prior, run `spectral-prior train`, then train PPO with `--prior_checkpoint outputs/prior.pt`. See [data preparation](docs/data.md) to rebuild the library.

## Play and evaluate

Inspect the bundled reference motions with the pelvis fixed in midair (Viser, no policy):

```bash
python -m legged_gym.scripts.reference
```

Select one of five library clips, or `prior` for generated motion. `kinematic` shows dataset poses directly; `pd` checks torque-limited tracking. The library contains ten joints; others keep their default pose. Use `--motion /path/to/gmr.npz` for source joint data and turn off **Hold wrists** to inspect its wrist motion. `sprint-reference` is also available after reinstalling the package.

In `pd`, **Gravity compensation (diagnostic)** adds gravity torque within the motor limits (`--gravity-comp`). This viewer option does not change the training controller; tracking error can remain because of inertia, gains and saturation.

```bash
sprint-play --checkpoint_path "<model>.pt" --headless --velocity 3 0 0
spectral-prior evaluate
sprint-evaluate --checkpoint_path "<model>.pt"
```

Replace `<model>.pt` with the path to your trained policy model.

For a viewer, omit `--headless`; on macOS use `mjpython -m legged_gym.scripts.play`. Training curves: `tensorboard --logdir logs`.

The [evaluation protocol](docs/evaluation.md) covers prior metrics and the 61-speed policy sweep. The bundled library follows the paper's 30→60 Hz preprocessing; its measured anchors differ from the authors' references.

Licenses: `LICENSE`, `licenses/`, and `resources/robots/UNITREE_LICENSE`.
