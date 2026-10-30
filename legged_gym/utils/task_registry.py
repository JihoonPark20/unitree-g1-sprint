# SPDX-License-Identifier: BSD-3-Clause
# Based on legged_gym task_registry, Copyright (c) 2021 ETH Zurich, Nikita Rudin.
"""Task construction and the rsl-rl runner configuration boundary."""

import json
import shutil
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR

from .helpers import (
    class_to_dict,
    get_args,
    get_load_path,
    resolve_rl_device,
    set_seed,
    update_cfg_from_args,
    update_class_from_dict,
)

LOG_DIR = Path(LEGGED_GYM_ROOT_DIR) / "logs"


def policy_contract(env):
    """Fields that must match when a saved policy is applied to this task."""
    contract = {
        "version": 2,
        "robot": env.cfg.asset.file,
        "joint_names": env.dof_names,
        "action_joint_names": env.action_dof_names,
        "held_joint_control": "default_pose_pd",
        "policy_obs": env.num_obs,
        "critic_obs": env.num_privileged_obs,
        "history_length": env.cfg.env.history_length,
        "history_order": "term_oldest_first",
        "sim_dt": env.cfg.sim.dt,
        "decimation": env.cfg.control.decimation,
        "action_scale": env.cfg.control.action_scale,
        "clip_actions": env.cfg.control.clip_actions,
        "default_joint_pos": env.default_dof_pos.tolist(),
        "kp": env.p_gains.tolist(),
        "kd": env.d_gains.tolist(),
        "effort_limits": env.torque_limits.tolist(),
        "torque_speed_curve": False,
        "armature": env.cfg.control.armature,
        "frictionloss": env.cfg.control.frictionloss,
    }
    return contract


def runner_config(cfg):
    policy = class_to_dict(cfg.policy)
    algorithm = class_to_dict(cfg.algorithm)
    return {
        "seed": cfg.seed,
        "num_steps_per_env": cfg.runner.num_steps_per_env,
        "save_interval": cfg.runner.save_interval,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": policy["actor_hidden_dims"],
            "activation": policy["activation"],
            "obs_normalization": policy.get("obs_normalization", False),
            "distribution_cfg": {
                "class_name": "rsl_rl.modules:GaussianDistribution",
                "std_type": "scalar",
                "init_std": policy["init_noise_std"],
            },
        },
        "critic": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": policy["critic_hidden_dims"],
            "activation": policy["activation"],
            "obs_normalization": policy.get("obs_normalization", False),
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms:PPO",
            "rnd_cfg": None,
            "symmetry_cfg": None,
            **algorithm,
        },
        "logger": "tensorboard",
        "check_for_nan": True,
    }


class LeggedOnPolicyRunner(OnPolicyRunner):
    """Upstream PPO, with policy-contract and curriculum checkpoint metadata."""

    def save(self, path, infos=None):
        metadata = dict(infos or {})
        metadata["legged_gym"] = {
            "contract": policy_contract(self.env),
            "training_state": self.env.training_state_dict(),
            "next_iteration": self.current_learning_iteration + 1,
        }
        super().save(path, metadata)

    def load(self, path, load_cfg=None, strict=True, map_location=None, *, restore_env=True):
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        metadata = (checkpoint.get("infos") or {}).get("legged_gym")
        if metadata is None or any(
            metadata.get("contract", {}).get(key) != value for key, value in policy_contract(self.env).items()
        ):
            raise ValueError("Checkpoint policy contract does not match this robot/configuration")
        del checkpoint
        # Load on CPU, then let model/optimizer state_dict loading place tensors
        # on the constructed learner device. This also keeps Adam step scalars portable.
        infos = super().load(path, load_cfg, strict, map_location or "cpu")
        if restore_env:
            self.current_learning_iteration = infos["legged_gym"].get(
                "next_iteration", self.current_learning_iteration + 1
            )
            self.env.load_training_state_dict(infos["legged_gym"]["training_state"])
        return infos


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name, task_class, env_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name):
        if name not in self.task_classes:
            raise ValueError(f"Unknown task {name!r}; available: {', '.join(self.task_classes)}")
        return self.task_classes[name]

    def get_cfgs(self, name, args=None):
        self.get_task_class(name)
        env_cfg = deepcopy(self.env_cfgs[name])
        train_cfg = deepcopy(self.train_cfgs[name])
        if args is not None and (getattr(args, "resume", False) or getattr(args, "checkpoint_path", None)):
            if not getattr(args, "checkpoint_path", None):
                experiment = getattr(args, "experiment_name", None) or train_cfg.runner.experiment_name
                root = LOG_DIR / experiment
                args.checkpoint_path = get_load_path(root, args.load_run, args.checkpoint)
            snapshot_path = Path(args.checkpoint_path).resolve().parent / "run_config.json"
            if snapshot_path.is_file():
                snapshot = json.loads(snapshot_path.read_text())
                if snapshot.get("task") not in (None, name):
                    raise ValueError("Checkpoint belongs to a different task")
                if snapshot["env"]["asset"]["file"] != env_cfg.asset.file:
                    raise ValueError("Checkpoint snapshot belongs to a different robot/configuration")
                if (
                    snapshot["env"]["env"]["num_actions"] != env_cfg.env.num_actions
                    or snapshot["env"]["asset"].get("action_joint_names") != env_cfg.asset.action_joint_names
                ):
                    raise ValueError(
                        "Checkpoint action joints do not match the wrist-held policy (23 actions). "
                        "Old 29-action checkpoints cannot be resumed or played with this configuration. "
                        "Start a new training run without --resume or --checkpoint_path."
                    )
                legacy_rewards = "tracking_ang_sigma" not in snapshot["env"]["rewards"]
                if not getattr(args, "play", False) and (
                    snapshot["env"]["commands"].get("curriculum_kind") != "rlvrl_grid"
                    or snapshot["env"]["rewards"].get("preset") != "sprint_run_scales_v1"
                ):
                    raise ValueError(
                        "This checkpoint predates the RuN-scale rewards / RLvRL grid curriculum. "
                        "It can still be played/evaluated, but start a new training run without "
                        "--resume or --checkpoint_path to use the new setup."
                    )
                update_class_from_dict(env_cfg, snapshot["env"], strict=False)
                update_class_from_dict(train_cfg, snapshot["train"], strict=False)
                if legacy_rewards:
                    # Old policies used one shared width for both tracking terms.
                    env_cfg.rewards.tracking_ang_sigma = env_cfg.rewards.tracking_sigma
                    env_cfg.rewards.preset = "legacy_checkpoint"
                train_cfg.runner.experiment_name = name
            else:
                raise ValueError("Checkpoint requires adjacent run_config.json")
        if args is not None:
            update_cfg_from_args(env_cfg, train_cfg, args)
        env_cfg.seed = train_cfg.seed
        prior = env_cfg.reference.checkpoint
        if args is not None and getattr(args, "checkpoint_path", None):
            bundled = Path(args.checkpoint_path).resolve().parent / "spectral_prior.pt"
            if bundled.is_file():
                prior = bundled
        if args is not None and getattr(args, "prior_checkpoint", None):
            prior = args.prior_checkpoint
        env_cfg.reference.checkpoint = str(Path(prior).resolve())
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None):
        args = get_args() if args is None else args
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name, args)
        else:
            env_cfg = deepcopy(env_cfg)
            update_cfg_from_args(env_cfg, None, args)
        env_cfg.seed = set_seed(env_cfg.seed)
        env = self.get_task_class(name)(
            cfg=env_cfg,
            physics_engine=getattr(args, "physics_engine", "mujoco"),
            sim_device=getattr(args, "sim_device", "cpu"),
            headless=True,
        )
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default"):
        if version("rsl-rl-lib") != "5.5.2":
            raise RuntimeError(
                "This project requires rsl-rl-lib 5.5.2. Install dependencies with: python -m pip install -e ."
            )
        args = get_args() if args is None else args
        if train_cfg is None:
            if name is None:
                raise ValueError("Provide a task name or train_cfg")
            _, train_cfg = self.get_cfgs(name, args)
        else:
            train_cfg = deepcopy(train_cfg)
            update_cfg_from_args(None, train_cfg, args)
        train_cfg.seed = env.cfg.seed
        device = resolve_rl_device(getattr(args, "rl_device", "auto"))
        if log_root == "default":
            log_root = LOG_DIR / train_cfg.runner.experiment_name
        log_dir = None
        if log_root is not None:
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            if train_cfg.runner.run_name:
                run_name += "_" + train_cfg.runner.run_name
            directory = Path(args.log_dir) if getattr(args, "log_dir", None) else Path(log_root) / run_name
            directory.mkdir(parents=True, exist_ok=False)
            log_dir = str(directory)
            shutil.copy2(env.cfg.reference.checkpoint, directory / "spectral_prior.pt")
            env.cfg.reference.checkpoint = str((directory / "spectral_prior.pt").resolve())
            (directory / "run_config.json").write_text(
                json.dumps(
                    {
                        "task": name,
                        "env": class_to_dict(env.cfg),
                        "train": class_to_dict(train_cfg),
                        "rl_device": device,
                        "dependencies": {key: version(key) for key in ("rsl-rl-lib", "mujoco", "torch")},
                        "reward_definition": "SPRINT Eq.14 torso bound uses raw commanded vx",
                        "contract": policy_contract(env),
                    },
                    indent=2,
                )
                + "\n"
            )
        from legged_gym.runners.on_policy_runner import SprintRunner, sprint_runner_config

        runner = SprintRunner(env, sprint_runner_config(train_cfg), log_dir, device=device)
        if getattr(args, "resume", False) or getattr(args, "checkpoint_path", None):
            path = getattr(args, "checkpoint_path", None)
            if path is None:
                raise ValueError("Resolve the resume checkpoint through get_cfgs before constructing the runner")
            runner.load(path, restore_env=not getattr(args, "play", False))
            # Loading Adam restores its saved LR. An explicit CLI override must
            # update both Adam and PPO's adaptive scheduler after that restore.
            learning_rate = getattr(args, "learning_rate", None)
            if learning_rate is not None:
                runner.alg.learning_rate = learning_rate
                for group in runner.alg.optimizer.param_groups:
                    group["lr"] = learning_rate
        return runner, train_cfg


task_registry = TaskRegistry()
