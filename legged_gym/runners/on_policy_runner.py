"""rsl-rl PPO plus spectral-prior identity and environment checkpoint contracts."""

import io
import math
from contextlib import redirect_stdout
from pathlib import Path

import torch

from legged_gym.utils.task_registry import LeggedOnPolicyRunner, policy_contract, runner_config
from legged_gym.utils.training_logger import TrainingLogger
from spectral_prior import JOINT_NAMES
from spectral_prior.data import sha256


def sprint_runner_config(train, env=None):
    config = runner_config(train)
    config["algorithm"]["class_name"] = "legged_gym.algorithms.ppo:SprintPPO"
    recurrent = getattr(train.policy, "recurrent", True)
    if recurrent:
        for key in ("actor", "critic"):
            config[key].update(
                class_name="legged_gym.modules.actor_critic:SprintRecurrentModel",
                feature_dim=256,
                rnn_hidden_dim=64,
                rnn_type="lstm",
                rnn_num_layers=1,
            )
    if getattr(train.policy, "integrate_mean", False):
        if env is None or env.cfg.control.residual_mode != "position" or env.cfg.control.increment_kappa is None:
            raise ValueError("An integrated-mean actor requires position control and an increment_kappa")
        if env.num_actions != len(env.action_dof_names):
            raise ValueError("An integrated-mean actor requires one output per action joint")
        model = "IntegratedMeanLSTM" if recurrent else "IntegratedMeanMLP"
        config["actor"].update(
            class_name=f"legged_gym.modules.actor_critic:{model}",
            action_limits=(env.residual_limits / env.cfg.control.action_scale).tolist(),
            increment_kappa=env.cfg.control.increment_kappa,
        )
        config["algorithm"]["class_name"] = "legged_gym.algorithms.ppo:IntegratedMeanPPO"
    elif env is not None and env.cfg.control.increment_kappa is not None and env.cfg.control.residual_mode == "position":
        raise ValueError("Position control with increment_kappa requires policy.integrate_mean")
    return config


class SprintRunner(LeggedOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        # Upstream prints the full actor/critic during construction. Capture
        # only setup, before the monitor starts; never redirect training output.
        setup = io.StringIO()
        try:
            with redirect_stdout(setup):
                super().__init__(env, train_cfg, log_dir, device)
        except BaseException:
            print(setup.getvalue(), end="")
            raise
        if env.cfg.control.learned_leak:
            initial = env.cfg.control.residual_leak - env.cfg.control.lambda_min
            bias = initial + math.log(-math.expm1(-initial))
            head = self.alg.actor.mlp[-1]
            with torch.no_grad():
                head.weight[-1].zero_()
                head.bias[-1].fill_(bias)
        self.logger = TrainingLogger(
            log_dir=log_dir, cfg=self.cfg, env_cfg=env.cfg, num_envs=env.num_envs,
            is_distributed=self.is_distributed, gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank, device=self.device,
            control_dt=env.dt, setup_output=setup.getvalue(),
        )

    def sprint_contract(self):
        contract = {
            "version": 2,
            "action_joint_names": self.env.action_dof_names,
            "held_joint_control": "default_pose_pd",
            "prior_sha256": sha256(self.env.cfg.reference.checkpoint),
            "reference_joint_names": list(JOINT_NAMES),
            "prior_config": self.env.reference.prior.config.__dict__,
            "blend_speed": self.env.cfg.reference.blend_speed,
            "acceleration_limit": self.env.cfg.commands.acceleration_limit,
            "max_latency_steps": self.env.cfg.control.max_latency_steps,
            "observation": "angular_gravity_filtered_commands_q_dq_previous_residual",
            "critic_extra": "base_linear_velocity_reference_10",
        }
        if self.env.auxiliary_actions:
            contract["auxiliary_actions"] = self.env.auxiliary_actions
        if self.env.cfg.reference.phase_modulation:
            contract["phase_modulation"] = self.env.cfg.reference.phase_modulation
        if self.env.cfg.reference.phase_obs:
            contract["phase_obs"] = True
        if self.env.cfg.control.learned_leak:
            contract["lambda_min"] = self.env.cfg.control.lambda_min
            contract["residual_leak_initial"] = self.env.cfg.control.residual_leak
        if self.env.cfg.control.increment_kappa is not None:
            contract["increment_kappa"] = self.env.cfg.control.increment_kappa
            contract["increment_mode"] = self.env.cfg.control.residual_mode
        return contract

    def save(self, path, infos=None):
        metadata = dict(infos or {})
        metadata["sprint"] = self.sprint_contract()
        if getattr(self, "warm_start_info", None) is not None:
            metadata["warm_start"] = self.warm_start_info
        super().save(path, metadata)

    def warm_start(self, path):
        """Transfer an actor's mean policy under an identical control contract.

        The new reward / curriculum may differ. Value function, optimizer,
        iteration counter and exploration parameters remain freshly initialized.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("infos") or {}
        if metadata.get("sprint") != self.sprint_contract() or metadata.get("legged_gym", {}).get("contract") != policy_contract(self.env):
            raise ValueError("Warm start requires matching robot, action bounds, observations, controller and spectral prior")
        source = dict(checkpoint["actor_state_dict"])
        for key, value in self.alg.actor.state_dict().items():
            if key.startswith("distribution."):
                source[key] = value.clone()
        self.alg.actor.load_state_dict(source, strict=True)
        self.alg.actor.reset()
        self.warm_start_info = {"checkpoint": str(Path(path).resolve()), "sha256": sha256(path), "transfer": "actor_mean_only"}

    def load(self, path, *args, **kwargs):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (checkpoint.get("infos") or {}).get("sprint") != self.sprint_contract():
            raise ValueError("Checkpoint spectral prior or SPRINT control contract mismatch")
        infos = super().load(path, *args, **kwargs)
        self.warm_start_info = (checkpoint.get("infos") or {}).get("warm_start")
        return infos
