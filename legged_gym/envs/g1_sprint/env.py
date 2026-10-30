"""SPRINT task adapter for the integrated MuJoCo legged_gym interface.

Step/reset flow adapted from legged_gym; see licenses/legged_gym_LICENSE.
"""

import numpy as np
import torch

from legged_gym.envs.base.legged_robot import SprintRobot, rotate_inverse
from legged_gym.envs.curriculums import GridAdaptiveCurriculum
from legged_gym.sim.mujoco_batch import joint_parameters
from spectral_prior.reference import ReferenceGenerator


class SprintEnv(SprintRobot):
    def __init__(self, cfg, **kwargs):
        default = joint_parameters(cfg.asset.joint_names, cfg.init_state.default_joint_angles, 0)
        self.reference = ReferenceGenerator(
            cfg.reference.checkpoint, cfg.asset.joint_names, default, cfg.env.num_envs, cfg.reference.blend_speed
        )
        self.q_ref = np.tile(default, (cfg.env.num_envs, 1))
        self.frequency = np.full(cfg.env.num_envs, 0.6)
        self.filtered_speed = np.zeros(cfg.env.num_envs)
        self.target_position = self.q_ref.copy()
        self._latency = np.zeros(cfg.env.num_envs, dtype=int)
        self._targets = np.tile(self.q_ref, (cfg.control.max_latency_steps + 1, 1, 1))
        self.command_curriculum = None
        if cfg.commands.curriculum:
            if cfg.commands.curriculum_kind != "rlvrl_grid":
                raise ValueError("SPRINT training requires the rlvrl_grid command curriculum")
            self.command_curriculum = GridAdaptiveCurriculum(
                np.asarray(cfg.commands.limit_ranges)[[0, 2]], cfg.commands.curriculum_bin_widths,
                np.asarray(cfg.commands.ranges)[[0, 2]], cfg.commands.curriculum_thresholds,
                cfg.commands.curriculum_increment, cfg.commands.curriculum_local_range,
            )
            if (
                not np.isfinite([cfg.commands.acceleration_limit, cfg.commands.curriculum_min_settled_s,
                                 cfg.commands.curriculum_settle_tolerance, cfg.commands.resampling_time]).all()
                or cfg.commands.acceleration_limit <= 0 or cfg.commands.curriculum_min_settled_s <= 0
                or cfg.commands.curriculum_settle_tolerance < 0
                or cfg.commands.resampling_time < cfg.commands.curriculum_min_settled_s
            ):
                raise ValueError("Curriculum needs positive acceleration and a valid settled measurement window")
        self._command_bins = np.full(cfg.env.num_envs, -1, dtype=int)
        self._command_score_sums = np.zeros((cfg.env.num_envs, 2))
        self._command_settled_steps = np.zeros(cfg.env.num_envs, dtype=int)
        self.record_evaluation = False
        super().__init__(cfg, **kwargs)
        self._foot_quat = self.sim.batch.bind("xquat")

    def _update_reference(self):
        self.q_ref[:], self.frequency[:] = self.reference.evaluate(self.filtered_speed)

    def reset_idx(self, ids):
        super().reset_idx(ids)
        self.filtered_speed[ids] = 0
        self.reference.reset(ids)
        self._latency[ids] = self.rng.integers(0, self.cfg.control.max_latency_steps + 1, len(ids))
        self._update_reference()
        self.target_position[ids] = self.q_ref[ids]
        self._targets[:, ids] = self.q_ref[ids]

    def _compute_torques(self, actions):
        # Target is held over decimation; PD feedback is recomputed at 200 Hz.
        pos = self.sim.qpos[:, self.sim.qpos_ids]
        vel = self.sim.qvel[:, self.sim.qvel_ids]
        torque = self.p_gains * (self.target_position - pos) - self.d_gains * vel
        return np.clip(torque, -self.torque_limits, self.torque_limits)

    def step(self, actions):
        if self._closed:
            raise RuntimeError("Environment is closed")
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.shape != self.actions.shape or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite residual actions of shape {self.actions.shape}")
        self.actions[:] = np.clip(actions, -self.cfg.control.clip_actions, self.cfg.control.clip_actions)
        target = self.q_ref.copy()
        target[:, self.action_dof_indices] += self.cfg.control.action_scale * self.actions
        # Wrist joints remain physical: PD holds a fixed pose under external loads.
        target[:, self.held_dof_indices] = self.default_dof_pos[self.held_dof_indices]
        # Joint constraints apply to the combined reference+residual, not either separately.
        limits = self.sim.model.jnt_range[self.sim.joint_ids]
        self._targets[1:] = self._targets[:-1].copy()
        self._targets[0] = np.clip(target, limits[:, 0], limits[:, 1])
        self.target_position[:] = self._targets[self._latency, np.arange(self.num_envs)]
        self._push_robots()
        self._first_contact.fill(False)
        self._touchdown_air_time.fill(0)
        for substep in range(self.cfg.control.decimation):
            self.torques[:] = self._compute_torques(self.actions)
            self.sim.step(self.torques, observe=substep == self.cfg.control.decimation - 1)
            self._update_contacts()
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self._refresh_state()
        self.check_termination()
        # Evaluate the reference and command that actually produced this transition.
        self.compute_reward()
        self._record_command_scores()
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.reference.advance(self.filtered_speed, self.dt)
        resample_steps = max(1, round(self.cfg.commands.resampling_time / self.dt))
        resample = (self.episode_length_buf.numpy() % resample_steps == 0) & ~self.reset_buf
        # Close each segment against its original bin BEFORE resampling/reset.
        self._finish_command_segments(np.flatnonzero(resample | self.reset_buf))
        self._resample_commands(np.flatnonzero(resample))
        delta = self.cfg.commands.acceleration_limit * self.dt
        self.filtered_speed += np.clip(self.commands[:, 0] - self.filtered_speed, -delta, delta)
        self._update_reference()
        self.compute_observations()
        dones = torch.from_numpy(self.reset_buf.copy())
        rewards = torch.from_numpy(self.rew_buf.astype(np.float32))
        extras = {"time_outs": torch.from_numpy(self.time_out_buf.copy())}
        if self.record_evaluation:
            # Snapshot t+dt before auto-reset, aligned with the oscillator at t+dt.
            w, x, y, z = self.base_quat.T
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            extras["evaluation"] = {
                "joint_pos": self.dof_pos[:, self.reference.indices].copy(),
                "reference_joint_pos": self.q_ref[:, self.reference.indices].copy(),
                "forward_velocity": (np.cos(yaw) * self.sim.qvel[:, 0] + np.sin(yaw) * self.sim.qvel[:, 1]).copy(),
            }
        ids = np.flatnonzero(self.reset_buf)
        if ids.size:
            extras["terminal_observations"] = self.obs_buf[torch.from_numpy(ids)].clone()
            extras["terminal_env_ids"] = torch.from_numpy(ids)
            extras["log"] = self._episode_metrics(ids)
            self.reset_idx(ids)
            self._refresh_state()
            self.compute_observations(ids)
        if not self.headless:
            self.render()
        return self.obs_buf, rewards, dones, extras

    def _episode_metrics(self, ids):
        metrics = super()._episode_metrics(ids)
        # Population snapshots, independent of the environment shown in Viser.
        metrics.update({
            "Commands/mean_vx": float(self.commands[:, 0].mean()),
            "Commands/moving_fraction": float(np.mean(self.commands[:, 0] > 0)),
            "Commands/mean_filtered_vx": float(self.filtered_speed.mean()),
            "Commands/mean_frequency_hz": float(self.frequency.mean()),
            "Commands/full_prior_fraction": float(np.mean(self.filtered_speed >= self.cfg.reference.blend_speed)),
        })
        if self.command_curriculum is not None and self.cfg.commands.curriculum:
            metrics.update(self.command_curriculum.metrics())
        return metrics

    def _resample_commands(self, ids):
        if not len(ids):
            return
        self._command_score_sums[ids] = 0
        self._command_settled_steps[ids] = 0
        self._command_bins[ids] = -1
        if self.command_curriculum is None or not self.cfg.commands.curriculum:
            # Playback/evaluation use exact externally requested commands.
            return super()._resample_commands(ids)
        sampled, bins = self.command_curriculum.sample(self.rng, len(ids))
        self.commands[ids] = 0
        self.commands[np.ix_(ids, [0, 2])] = sampled
        self._command_bins[ids] = bins
        standing = ids[self.rng.random(len(ids)) < self.cfg.commands.standing_fraction]
        self.commands[standing] = 0
        self._command_bins[standing] = -1  # Standing never unlocks walking bins.
        self.command_ranges[[0, 2]] = self.command_curriculum.active_ranges

    def _record_command_scores(self):
        if self.command_curriculum is None or not self.cfg.commands.curriculum:
            return
        settled = (self._command_bins >= 0) & (
            np.abs(self.filtered_speed - self.commands[:, 0]) <= self.cfg.commands.curriculum_settle_tolerance
        )
        self._command_settled_steps[settled] += 1
        # Unweighted kernels avoid coupling success thresholds to reward tuning.
        self._command_score_sums[settled, 0] += self._reward_tracking_lin_vel()[settled]
        self._command_score_sums[settled, 1] += self._reward_tracking_ang_vel()[settled]

    def _finish_command_segments(self, ids):
        if not len(ids) or self.command_curriculum is None or not self.cfg.commands.curriculum:
            return
        counts = self._command_settled_steps[ids]
        scores = self._command_score_sums[ids] / np.maximum(counts[:, None], 1)
        eligible = (counts * self.dt + 1e-12 >= self.cfg.commands.curriculum_min_settled_s) & ~self.terminated_buf[ids]
        self.command_curriculum.update(self._command_bins[ids], scores, eligible)
        self.command_ranges[[0, 2]] = self.command_curriculum.active_ranges
        self._command_score_sums[ids] = 0
        self._command_settled_steps[ids] = 0
        self._command_bins[ids] = -1

    def _observation_terms(self, critic):
        terms = super()._observation_terms(critic)
        # Three task commands are [filtered vx, vy=0, raw yaw rate].
        commands = self.commands.copy()
        commands[:, 0] = self.filtered_speed
        terms["commands"] = commands
        if critic:
            terms["reference"] = self.q_ref[:, self.reference.indices]
        return terms

    def _reward_tracking_lin_vel(self):
        # Heading frame excludes torso pitch from the measured forward speed.
        w, x, y, z = self.base_quat.T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        v = np.cos(yaw) * self.sim.qvel[:, 0] + np.sin(yaw) * self.sim.qvel[:, 1]
        return np.exp(-np.square(self.filtered_speed - v) / self.cfg.rewards.tracking_sigma)

    def _reward_prior(self):
        ids = self.reference.indices
        return np.exp(-np.square(self.dof_pos[:, ids] - self.q_ref[:, ids]) / self.cfg.rewards.prior_sigma).sum(1)

    def _reward_tracking_ang_vel(self):
        return np.exp(-np.square(self.commands[:, 2] - self.base_ang_vel[:, 2]) / self.cfg.rewards.tracking_ang_sigma)

    def _reward_close_feet(self):
        relative = self.feet_pos[:, 0] - self.feet_pos[:, 1]
        w, x, y, z = self.base_quat.T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        lateral = -np.sin(yaw) * relative[:, 0] + np.cos(yaw) * relative[:, 1]
        return np.maximum(0, self.cfg.rewards.min_feet_distance - np.abs(lateral))

    def _reward_air_height(self):
        height = self.feet_pos[:, :, 2] - self.feet_ground_height
        return (
            (~self._feet_contact)
            * np.exp(-np.abs(height - self.cfg.rewards.feet_clearance_target) / self.cfg.rewards.feet_clearance_std)
        ).sum(1)

    def _reward_low_speed_air(self):
        return (self.filtered_speed < 1.5) * (~self._feet_contact).all(1)

    def _reward_high_speed_ground(self):
        return (self.filtered_speed > 3) * self._feet_contact.all(1)

    def _reward_feet_parallel(self):
        q = self._foot_quat[:, self.sim.feet_ids].reshape(-1, 4)
        gravity = rotate_inverse(q, np.tile([0.0, 0.0, -1.0], (len(q), 1))).reshape(self.num_envs, 2, 3)
        return (np.exp(-np.linalg.norm(gravity[:, :, :2], axis=-1)) * self._feet_contact).sum(1)

    def _reward_swing_deficit(self):
        # Eq.13 is a positive deficit; a negative weight avoids rewarding short swings.
        target = self.cfg.rewards.swing_duty / self.frequency
        return ((~self._feet_contact) * np.maximum(0, target[:, None] - self.feet_air_time)).sum(1)

    def _reward_dof_pos_limits(self):
        return ((self.dof_pos < self.dof_pos_limits[:, 0]) | (self.dof_pos > self.dof_pos_limits[:, 1])).sum(1)

    def _reward_torso_pitch(self):
        gx = self.projected_gravity[:, 0]
        # Eq. 14 uses the commanded speed, before the reference acceleration ramp.
        forward = 0.1 + 0.05 * self.commands[:, 0]
        return np.square(np.maximum(0, gx - forward) + np.maximum(0, -gx))

    def training_state_dict(self):
        state = super().training_state_dict()
        state["reward_preset"] = self.cfg.rewards.preset
        state["sprint_curriculum"] = self.command_curriculum.state_dict() if self.command_curriculum is not None else None
        state["command_rng_state"] = self.rng.bit_generator.state
        return state

    def load_training_state_dict(self, state):
        if state.get("reward_preset") != self.cfg.rewards.preset or state.get("sprint_curriculum") is None:
            raise ValueError("Legacy reward/curriculum checkpoint cannot resume this setup; start a new training run")
        if self.command_curriculum is None:
            raise ValueError("Cannot restore training curriculum into an evaluation environment")
        self.command_curriculum.load_state_dict(state["sprint_curriculum"])
        self.rng.bit_generator.state = state["command_rng_state"]
        # Physics is reset on resume; partial segments cannot be credited to a
        # fresh trajectory. Only completed statistics / sampling weights persist.
        super().load_training_state_dict(state)

    def render(self):
        """Refresh viewer kinematics after copying a nonpersistent batch state."""
        import mujoco
        import mujoco.viewer

        if self._viewer is None:
            self._viewer_model = self.sim.batch.get_model(0)
            self._viewer_data = self.sim.batch.get_data(0)
            # Nonpersistent Batch.get_data copies integration state only.
            # The viewer consumes derived xpos/xquat/geom_xpos, not just qpos.
            mujoco.mj_forward(self._viewer_model, self._viewer_data)
            self._viewer = mujoco.viewer.launch_passive(self._viewer_model, self._viewer_data)
            with self._viewer.lock():
                self._viewer.cam.distance = 3.0
                self._viewer.cam.elevation = -20
                self._viewer.cam.lookat[:] = self.base_pos[0]
        if self._viewer.is_running():
            with self._viewer.lock():
                self.sim.batch.get_data(0, self._viewer_data)
                mujoco.mj_forward(self._viewer_model, self._viewer_data)
                if getattr(self, "follow_camera", True):
                    self._viewer.cam.lookat[:] = self.base_pos[0]
            self._viewer.sync()

    def close(self):
        self._foot_quat = None
        super().close()
