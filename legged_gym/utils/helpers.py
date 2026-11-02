"""Configuration, CLI and device helpers for the MuJoCo training path."""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch


def class_to_dict(obj):
    if isinstance(obj, dict):
        return {key: class_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [class_to_dict(value) for value in obj]
    if not hasattr(obj, "__dict__"):
        return obj
    return {
        key: class_to_dict(getattr(obj, key))
        for key in dir(obj)
        if not key.startswith("_") and not callable(getattr(obj, key))
    }


def update_class_from_dict(obj, values, *, strict=True):
    for key, value in values.items():
        if not hasattr(obj, key):
            if not strict:
                continue
            raise ValueError(f"Unknown configuration field: {key}")
        target = getattr(obj, key)
        if isinstance(value, dict) and hasattr(target, "__dict__"):
            update_class_from_dict(target, value, strict=strict)
        else:
            setattr(obj, key, value)


def set_seed(seed):
    if seed < 0:
        seed = random.SystemRandom().randrange(2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def resolve_rl_device(device="auto"):
    """Validate the learner device and canonicalize CUDA to an explicit ordinal.

    An explicit unavailable device raises instead of silently moving training.
    Automatic selection is the default and prefers CUDA, then MPS, then CPU.
    """
    if str(device) == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("rl_device must be mps, cpu, cuda[:N] or auto") from error
    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install a CUDA-enabled PyTorch build or use --rl_device cpu/auto")
        index = torch.cuda.current_device() if parsed.index is None else parsed.index
        count = torch.cuda.device_count()
        if not 0 <= index < count:
            raise ValueError(f"CUDA index {index} is out of range for {count} visible device(s)")
        return f"cuda:{index}"
    if parsed.type not in ("mps", "cpu") or parsed.index not in (None, 0):
        raise ValueError("rl_device must be mps, cpu, cuda[:N] or auto")
    if parsed.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; select --rl_device cuda, cpu or auto")
    return parsed.type


def get_load_path(root, load_run=-1, checkpoint=-1):
    root = Path(root)
    if str(load_run) == "-1":
        runs = sorted((p for p in root.glob("*") if p.is_dir() and any(p.glob("model_*.pt"))), key=lambda p: p.name)
        if not runs:
            raise FileNotFoundError(f"No checkpoints under {root}")
        run = runs[-1]
    else:
        run = root / str(load_run)
    if checkpoint == -1:
        models = [p for p in run.glob("model_*.pt") if p.stem.removeprefix("model_").isdigit()]
        if not models:
            raise FileNotFoundError(f"No checkpoints in {run}")
        return str(max(models, key=lambda p: int(p.stem.split("_")[-1])))
    path = run / f"model_{checkpoint}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def get_args(argv=None, *, mode="train"):
    parser = argparse.ArgumentParser(description=f"SPRINT {mode}: MuJoCo CPU physics + residual recurrent PPO")
    parser.add_argument("--task", default="g1_sprint", choices=["g1_sprint"])
    parser.add_argument("--rl_device", "--rl-device", default="auto")
    parser.add_argument("--num_envs", "--num-envs", type=int, help="Parallel environments (training default: 4096)")
    parser.add_argument("--num_threads", "--num-threads", type=int, help="MuJoCo worker threads (default: 10)")
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prior_checkpoint", "--prior", help="Frozen spectral prior checkpoint")
    parser.add_argument("--checkpoint_path", "--checkpoint-path", metavar="<model>.pt", help="Path to the policy model")
    parser.add_argument("--load_run", default="-1")
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--experiment_name")
    parser.add_argument("--headless", action="store_true", help="Disable the viewer (Viser during training)")
    if mode == "train":
        parser.add_argument("--max_iterations", "--max-iterations", type=int)
        parser.add_argument("--max_grad_norm", type=float)
        parser.add_argument("--desired_kl", type=float)
        parser.add_argument("--learning_rate", type=float)
        parser.add_argument("--run_name")
        parser.add_argument("--log_interval", "--log-interval", type=int, default=1,
                            help="Terminal progress interval in iterations (default: 1); TensorBoard logs every iteration")
        parser.add_argument("--log-dir", help="Exact new run directory")
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--warm_start", "--warm-start", metavar="<model>.pt",
                            help="Initialize actor weights in a new run; reset critic, optimizer, exploration and curriculum")
    elif mode == "play":
        parser.add_argument("--steps", type=int, default=1000)
        parser.add_argument("--velocity", type=float, nargs=3, metavar=("VX", "VY", "WZ"))
        parser.add_argument("--camera", choices=["follow", "fixed"], default="follow")
        parser.add_argument("--camera-azimuth", type=float, default=135.0, help="Initial orbit angle in degrees")
        parser.add_argument("--camera-elevation", type=float, default=-20.0, help="Initial MuJoCo elevation in degrees")
        parser.add_argument("--camera-distance", type=float, default=3.0, help="Initial camera distance in meters")
        parser.add_argument("--camera-height", type=float, default=0.0, help="Look-at height relative to the pelvis")
    elif mode == "evaluate":
        parser.add_argument("--speeds", nargs="+", type=float, default=[i / 10 for i in range(61)])
        parser.add_argument("--seconds", type=float, default=20, help="Measurement duration after warmup")
        parser.add_argument("--warmup-seconds", type=float, help="Default: max command / acceleration limit + 2 s")
        parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
        parser.add_argument("--save-trajectories", action="store_true", help="Save aligned samples next to the JSON")
        parser.add_argument("--output", default="outputs/policy_evaluation.json")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    args = parser.parse_args(argv)
    if getattr(args, "warm_start", None) and (getattr(args, "resume", False) or args.checkpoint_path):
        parser.error("--warm_start starts a new run and cannot be combined with --resume or --checkpoint_path")
    args.play = mode != "train"
    if args.play:
        args.resume = True
    for name in ("num_envs", "max_iterations", "torch_threads", "steps", "log_interval"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    for name in ("max_grad_norm", "desired_kl", "learning_rate"):
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{name} must be finite and positive")
    if args.num_threads is not None and args.num_threads < 0:
        parser.error("num_threads must be nonnegative")
    if mode == "play" and (
        not all(math.isfinite(v) for v in (
            args.camera_azimuth, args.camera_elevation, args.camera_distance, args.camera_height
        ))
        or args.camera_distance <= 0
        or not -89 <= args.camera_elevation <= 89
    ):
        parser.error("Camera values must be finite, distance positive, and elevation within [-89, 89]")
    if (
        mode == "play"
        and args.velocity
        and not (
            all(math.isfinite(v) for v in args.velocity)
            and 0 <= args.velocity[0] <= 6
            and args.velocity[1] == 0
            and abs(args.velocity[2]) <= 0.7
        )
    ):
        parser.error("Velocity requires VX in [0,6], VY=0, WZ in [-0.7,0.7]")
    if mode == "evaluate" and (
        not math.isfinite(args.seconds)
        or args.seconds <= 0
        or any(not 0 <= v <= 6 for v in args.speeds)
        or (args.warmup_seconds is not None and (not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0))
        or len(set(args.speeds)) != len(args.speeds)
        or len(set(args.seeds)) != len(args.seeds)
        or any(seed < 0 for seed in args.seeds)
    ):
        parser.error(
            "Require positive finite seconds, nonnegative warmup/seeds, unique speeds in [0,6] and unique seeds"
        )
    try:
        args.rl_device = resolve_rl_device(args.rl_device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    torch.set_num_threads(args.torch_threads)
    return args


def update_cfg_from_args(env_cfg, train_cfg, args):
    for cfg in (env_cfg, train_cfg):
        if cfg is not None and getattr(args, "seed", None) is not None:
            cfg.seed = args.seed
    if env_cfg is not None:
        if getattr(args, "num_envs", None) is not None:
            env_cfg.env.num_envs = args.num_envs
        if getattr(args, "num_threads", None) is not None:
            env_cfg.sim.nthread = args.num_threads
    if train_cfg is not None:
        for key in ("max_grad_norm", "desired_kl", "learning_rate"):
            value = getattr(args, key, None)
            if value is not None:
                setattr(train_cfg.algorithm, key, value)
        for key in ("max_iterations", "experiment_name", "run_name", "log_interval"):
            value = getattr(args, key, None)
            if value is not None:
                setattr(train_cfg.runner, key, value)
    return env_cfg, train_cfg
