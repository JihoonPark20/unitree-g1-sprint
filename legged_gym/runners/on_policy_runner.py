"""rsl-rl PPO plus spectral-prior identity and environment checkpoint contracts."""

import torch

from legged_gym.utils.task_registry import LeggedOnPolicyRunner, runner_config
from spectral_prior import JOINT_NAMES
from spectral_prior.data import sha256


def sprint_runner_config(train):
    config = runner_config(train)
    config["algorithm"]["class_name"] = "legged_gym.algorithms.ppo:SprintPPO"
    for key in ("actor", "critic"):
        config[key].update(
            class_name="legged_gym.modules.actor_critic:SprintRecurrentModel",
            feature_dim=256,
            rnn_hidden_dim=64,
            rnn_type="lstm",
            rnn_num_layers=1,
        )
    return config


class SprintRunner(LeggedOnPolicyRunner):
    def sprint_contract(self):
        return {
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

    def save(self, path, infos=None):
        metadata = dict(infos or {})
        metadata["sprint"] = self.sprint_contract()
        super().save(path, metadata)

    def load(self, path, *args, **kwargs):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (checkpoint.get("infos") or {}).get("sprint") != self.sprint_contract():
            raise ValueError("Checkpoint spectral prior or SPRINT control contract mismatch")
        return super().load(path, *args, **kwargs)
