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
        roll_scale = cfg.rewards.scales.get("torso_roll", 0.0)
        roll_deadzone = getattr(cfg.rewards, "torso_roll_deadzone", 0.1)
        if not np.isfinite([roll_scale, roll_deadzone]).all() or roll_scale > 0 or not 0 <= roll_deadzone < 1:
            raise ValueError("torso_roll requires a nonpositive scale and a projected-gravity dead zone in [0, 1)")
        if (
            not np.isfinite([cfg.control.action_scale, cfg.control.clip_actions]).all()
            or cfg.control.action_scale <= 0 or cfg.control.clip_actions <= 0
        ):
            raise ValueError("Action scale and action clip must be finite and positive")
        bounds = getattr(cfg.control, "residual_limits", None)
        self.residual_limits = None if bounds is None else joint_parameters(cfg.asset.action_joint_names, bounds)
        if self.residual_limits is not None and np.any(self.residual_limits <= 0):
            raise ValueError("Residual joint limits must be positive radians")
        self._terminate_on_feet_crossing = getattr(cfg.termination, "terminate_on_feet_crossing", False)
        if self._terminate_on_feet_crossing and (
            not np.isfinite([cfg.termination.feet_crossing_margin, cfg.termination.feet_crossing_duration_s]).all()
            or cfg.termination.feet_crossing_margin < 0 or cfg.termination.feet_crossing_duration_s <= 0
        ):
            raise ValueError("Foot crossing needs a nonnegative margin and positive duration")
        self._feet_crossing_steps = np.zeros(cfg.env.num_envs, dtype=np.int32)
        self.feet_crossing_buf = np.zeros(cfg.env.num_envs, dtype=bool)
        self._command_crossed = np.zeros(cfg.env.num_envs, dtype=bool)
        self._diagnostic_names = (
            "Control/residual_rms_rad", "Control/action_clip_fraction", "Control/target_clip_fraction",
            "Tracking/prior_mae_rad", "Tracking/forward_mae_m_s", "Tracking/yaw_mae_rad_s",
            "Gait/both_ground_fraction", "Gait/single_support_fraction", "Gait/both_air_fraction",
            "Gait/signed_foot_separation_m", "Gait/feet_crossed_fraction",
            "Posture/roll_rad", "Posture/roll_abs_rad",
        )
        self._diagnostic_sums = np.zeros((cfg.env.num_envs, len(self._diagnostic_names)))
        self._joint_residual_sums = np.zeros((cfg.env.num_envs, 3, len(cfg.asset.action_joint_names)))
        self._torque_saturation_sums = np.zeros((cfg.env.num_envs, len(cfg.asset.joint_names)))
        default = joint_parameters(cfg.asset.joint_names, cfg.init_state.default_joint_angles, 0)
        self.reference = ReferenceGenerator(
            cfg.reference.checkpoint, cfg.asset.joint_names, default, cfg.env.num_envs, cfg.reference.blend_speed
        )
        self.q_ref = np.tile(default, (cfg.env.num_envs, 1))
        self.dq_ref = np.zeros_like(self.q_ref)
        self.frequency = np.full(cfg.env.num_envs, 0.6)
        self.filtered_speed = np.zeros(cfg.env.num_envs)
        self.target_position = self.q_ref.copy()
        self.target_velocity = np.zeros_like(self.q_ref)
        self._latency = np.zeros(cfg.env.num_envs, dtype=int)
        self._targets = np.tile(self.q_ref, (cfg.control.max_latency_steps + 1, 1, 1))
        self._target_velocities = np.zeros_like(self._targets)
        self.command_curriculum = None
        if cfg.commands.curriculum:
            if cfg.commands.curriculum_kind != "rlvrl_grid":
                raise ValueError("SPRINT training requires the rlvrl_grid command curriculum")
            self.command_curriculum = GridAdaptiveCurriculum(
                np.asarray(cfg.commands.limit_ranges)[[0, 2]], cfg.commands.curriculum_bin_widths,
                np.asarray(cfg.commands.ranges)[[0, 2]], cfg.commands.curriculum_thresholds,
                cfg.commands.curriculum_increment, cfg.commands.curriculum_local_range,
                progress=getattr(cfg.commands, "curriculum_progress", None),
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
        self._command_motion_sums = np.zeros((cfg.env.num_envs, 3))
        self._command_settled_steps = np.zeros(cfg.env.num_envs, dtype=int)
        self.record_evaluation = False
        self.record_failure_trace = False
        super().__init__(cfg, **kwargs)
        self._foot_quat = self.sim.batch.bind("xquat")

    def _update_reference(self):
        if getattr(self.cfg.control, "reference_velocity", False):
            self.q_ref[:], self.dq_ref[:], self.frequency[:] = self.reference.evaluate_with_velocity(
                self.filtered_speed, self.cfg.sim.dt,
            )
        else:
            self.q_ref[:], self.frequency[:] = self.reference.evaluate(self.filtered_speed)
            self.dq_ref.fill(0)

    def reset_idx(self, ids):
        super().reset_idx(ids)
        self._diagnostic_sums[ids] = 0
        self._joint_residual_sums[ids] = 0
        self._torque_saturation_sums[ids] = 0
        self._feet_crossing_steps[ids] = 0
        self.feet_crossing_buf[ids] = False
        self.filtered_speed[ids] = 0
        self.reference.reset(ids)
        self._latency[ids] = self.rng.integers(0, self.cfg.control.max_latency_steps + 1, len(ids))
        self._update_reference()
        self.target_position[ids] = self.q_ref[ids]
        self._targets[:, ids] = self.q_ref[ids]
        self.target_velocity[ids] = 0
        self._target_velocities[:, ids] = 0

    def check_termination(self):
        super().check_termination()
        if self._terminate_on_feet_crossing:
            crossed = self._signed_foot_separation() < -self.cfg.termination.feet_crossing_margin
            self._feet_crossing_steps[:] = np.where(crossed, self._feet_crossing_steps + 1, 0)
            self._command_crossed |= crossed
            self.feet_crossing_buf[:] = (
                self._feet_crossing_steps * self.dt + 1e-12 >= self.cfg.termination.feet_crossing_duration_s
            )
            self.terminated_buf |= self.feet_crossing_buf
            # Crossing is a true failure even when it coincides with a timeout.
            self.time_out_buf &= ~self.terminated_buf
            self.reset_buf = self.terminated_buf | self.time_out_buf

    def _compute_torques(self, actions):
        # Target is held over decimation; PD feedback is recomputed at 200 Hz.
        pos = self.sim.qpos[:, self.sim.qpos_ids]
        vel = self.sim.qvel[:, self.sim.qvel_ids]
        torque = self.p_gains * (self.target_position - pos) + self.d_gains * (self.target_velocity - vel)
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
        if self.residual_limits is not None:
            action_limits = self.residual_limits / self.cfg.control.action_scale
            self.actions[:] = np.clip(self.actions, -action_limits, action_limits)
        # Previous-action observations describe the bounded residual actually
        # requested from PD, while PPO retains its original sampled actions.
        residual = self.cfg.control.action_scale * self.actions
        clipped = np.abs(actions - self.actions) > 1e-7
        bound = self.cfg.control.action_scale * self.cfg.control.clip_actions
        if self.residual_limits is not None:
            bound = np.minimum(bound, self.residual_limits)
        self._joint_residual_sums[:, 0] += np.square(residual)
        self._joint_residual_sums[:, 1] += np.abs(residual) >= 0.99 * bound
        self._joint_residual_sums[:, 2] += clipped
        self._diagnostic_sums[:, 0] += np.sqrt(np.square(residual).mean(1))
        self._diagnostic_sums[:, 1] += clipped.mean(1)
        target = self.q_ref.copy()
        target[:, self.action_dof_indices] += residual
        # Wrist joints remain physical: PD holds a fixed pose under external loads.
        target[:, self.held_dof_indices] = self.default_dof_pos[self.held_dof_indices]
        # Joint constraints apply to the combined reference+residual, not either separately.
        limits = self.sim.model.jnt_range[self.sim.joint_ids]
        self._diagnostic_sums[:, 2] += ((target < limits[:, 0]) | (target > limits[:, 1])).mean(1)
        self._targets[1:] = self._targets[:-1].copy()
        self._targets[0] = np.clip(target, limits[:, 0], limits[:, 1])
        velocity = self.dq_ref.copy()
        velocity[:, self.held_dof_indices] = 0
        velocity[((target <= limits[:, 0]) & (velocity < 0)) | ((target >= limits[:, 1]) & (velocity > 0))] = 0
        self._target_velocities[1:] = self._target_velocities[:-1].copy()
        self._target_velocities[0] = velocity
        self.target_position[:] = self._targets[self._latency, np.arange(self.num_envs)]
        self.target_velocity[:] = self._target_velocities[self._latency, np.arange(self.num_envs)]
        self._push_robots()
        self._first_contact.fill(False)
        self._touchdown_air_time.fill(0)
        if self.record_failure_trace:
            step_saturation = np.zeros_like(self.torques)
            step_torque_peak = np.zeros_like(self.torques)
        for substep in range(self.cfg.control.decimation):
            self.torques[:] = self._compute_torques(self.actions)
            if self.record_failure_trace:
                p_torque = self.p_gains * (self.target_position - self.sim.qpos[:, self.sim.qpos_ids])
                d_torque = self.d_gains * (self.target_velocity - self.sim.qvel[:, self.sim.qvel_ids])
                requested_torque = p_torque + d_torque
                step_saturation += (np.abs(self.torques) >= 0.99 * self.torque_limits) / self.cfg.control.decimation
                np.maximum(step_torque_peak, np.abs(self.torques), out=step_torque_peak)
            self._torque_saturation_sums += (
                np.abs(self.torques) >= 0.99 * self.torque_limits
            ) / self.cfg.control.decimation
            self.sim.step(self.torques, observe=substep == self.cfg.control.decimation - 1)
            self._update_contacts()
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self._refresh_state()
        self.check_termination()
        # Evaluate the reference and command that actually produced this transition.
        self.compute_reward()
        self._record_tracking_diagnostics()
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
        # check_termination replaces these arrays on the next step, and
        # reset_idx does not mutate them, so retained flags remain stable.
        dones = torch.from_numpy(self.reset_buf)
        rewards = torch.from_numpy(self.rew_buf.astype(np.float32))
        extras = {
            "time_outs": torch.from_numpy(self.time_out_buf),
            "feet_crossing": torch.from_numpy(self.feet_crossing_buf.copy()),
        }
        if self.record_evaluation:
            # Snapshot t+dt before auto-reset, aligned with the oscillator at t+dt.
            w, x, y, z = self.base_quat.T
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            extras["evaluation"] = {
                "joint_pos": self.dof_pos[:, self.reference.indices].copy(),
                "reference_joint_pos": self.q_ref[:, self.reference.indices].copy(),
                "forward_velocity": (np.cos(yaw) * self.sim.qvel[:, 0] + np.sin(yaw) * self.sim.qvel[:, 1]).copy(),
            }
            if self.record_failure_trace:
                gravity = self.projected_gravity
                tilt = np.arccos(np.clip(-gravity[:, 2], -1, 1))
                extras["evaluation"]["failure_trace"] = {
                    "roll_rad": np.arctan2(-gravity[:, 1], -gravity[:, 2]),
                    "pitch_rad": np.arcsin(np.clip(gravity[:, 0], -1, 1)),
                    "tilt_rad": tilt,
                    "angular_velocity": self.base_ang_vel.copy(),
                    "base_height_m": self.base_height.copy(),
                    "feet_contact": self._feet_contact.copy(),
                    "feet_height_m": self.feet_pos[:, :, 2].copy(),
                    "signed_foot_separation_m": self._signed_foot_separation().copy(),
                    "phase": self.reference.phase.numpy().copy(),
                    "frequency_hz": self.frequency.copy(),
                    "forward_velocity": extras["evaluation"]["forward_velocity"].copy(),
                    "issued_residual_rad": residual.copy(),
                    "joint_pos_rad": self.dof_pos[:, self.action_dof_indices].copy(),
                    "pd_target_rad": self.target_position[:, self.action_dof_indices].copy(),
                    "joint_velocity_rad_s": self.dof_vel[:, self.action_dof_indices].copy(),
                    "pd_target_velocity_rad_s": self.target_velocity[:, self.action_dof_indices].copy(),
                    "torque_last_substep_Nm": self.torques[:, self.action_dof_indices].copy(),
                    "requested_torque_last_substep_Nm": requested_torque[:, self.action_dof_indices].copy(),
                    "p_torque_last_substep_Nm": p_torque[:, self.action_dof_indices].copy(),
                    "d_torque_last_substep_Nm": d_torque[:, self.action_dof_indices].copy(),
                    "torque_peak_abs_Nm": step_torque_peak[:, self.action_dof_indices].copy(),
                    "torque_saturation_fraction": step_saturation[:, self.action_dof_indices].copy(),
                    "foot_contact_force_norm_N": np.linalg.norm(self.contact_forces[:, self.feet_indices], axis=-1),
                    "feet_position_world_m": self.feet_pos.copy(),
                    "pelvis_position_world_m": self.base_pos.copy(),
                    "failure_flags": np.column_stack((
                        tilt > self.cfg.termination.max_tilt,
                        self.base_height < self.cfg.termination.min_height,
                        (self._contact_peak[:, self._root_contact_index] > self.cfg.rewards.contact_threshold)
                        & bool(self.cfg.termination.base_contact), self.feet_crossing_buf,
                    )),
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
            "Episode/feet_crossing_fraction": float(self.feet_crossing_buf[ids].mean()),
            "Commands/mean_vx": float(self.commands[:, 0].mean()),
            "Commands/moving_fraction": float(np.mean(self.commands[:, 0] > 0)),
            "Commands/mean_filtered_vx": float(self.filtered_speed.mean()),
            "Commands/mean_frequency_hz": float(self.frequency.mean()),
            "Commands/full_prior_fraction": float(np.mean(self.filtered_speed >= self.cfg.reference.blend_speed)),
        })
        if self.command_curriculum is not None and self.cfg.commands.curriculum:
            metrics.update(self.command_curriculum.metrics())
        # Normalize by actual episode length, not the nominal timeout, so early
        # falls cannot hide actuator saturation or tracking errors.
        lengths = np.maximum(self.episode_length_buf.numpy()[ids, None], 1)
        means = (self._diagnostic_sums[ids] / lengths).mean(0)
        metrics.update(dict(zip(self._diagnostic_names, map(float, means))))
        saturation = (self._torque_saturation_sums[ids] / lengths).mean(0)
        metrics["Control/torque_saturation"] = float(saturation.mean())
        metrics.update({f"TorqueSaturation/{name}": float(value) for name, value in zip(self.dof_names, saturation)})
        joint_means = self._joint_residual_sums[ids] / lengths[:, :, None]
        joint_means[:, 0] = np.sqrt(joint_means[:, 0])
        for tag, values in zip(
            ("ResidualRMS", "ResidualLimitFraction", "ActionClipFraction"), joint_means.mean(0),
        ):
            metrics.update({f"{tag}/{name}": float(value) for name, value in zip(self.action_dof_names, values)})
        return metrics

    def _record_tracking_diagnostics(self):
        indices = self.reference.indices
        self._diagnostic_sums[:, 3] += np.abs(self.dof_pos[:, indices] - self.q_ref[:, indices]).mean(1)
        forward = self._forward_velocity()
        self._diagnostic_sums[:, 4] += np.abs(self.filtered_speed - forward)
        self._diagnostic_sums[:, 5] += np.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        contacts = self._feet_contact.sum(1)
        self._diagnostic_sums[:, 6] += contacts == 2
        self._diagnostic_sums[:, 7] += contacts == 1
        self._diagnostic_sums[:, 8] += contacts == 0
        separation = self._signed_foot_separation()
        self._diagnostic_sums[:, 9] += separation
        self._diagnostic_sums[:, 10] += separation < 0
        roll = np.arctan2(-self.projected_gravity[:, 1], -self.projected_gravity[:, 2])
        self._diagnostic_sums[:, 11] += roll
        self._diagnostic_sums[:, 12] += np.abs(roll)

    def _resample_commands(self, ids):
        if not len(ids):
            return
        self._command_score_sums[ids] = 0
        self._command_motion_sums[ids] = 0
        self._command_settled_steps[ids] = 0
        self._command_bins[ids] = -1
        self._command_crossed[ids] = False
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
        forward = self._forward_velocity()[settled]
        self._command_motion_sums[settled, 0] += forward
        self._command_motion_sums[settled, 1] += np.abs(self.commands[settled, 0] - forward)
        self._command_motion_sums[settled, 2] += np.abs(self.commands[settled, 2] - self.base_ang_vel[settled, 2])

    def _finish_command_segments(self, ids):
        if not len(ids) or self.command_curriculum is None or not self.cfg.commands.curriculum:
            return
        counts = self._command_settled_steps[ids]
        scores = self._command_score_sums[ids] / np.maximum(counts[:, None], 1)
        eligible = (counts * self.dt + 1e-12 >= self.cfg.commands.curriculum_min_settled_s) & ~self.terminated_buf[ids]
        # A crossing near a segment boundary must not promote before the
        # consecutive-step termination timer has had time to expire.
        eligible &= ~self._command_crossed[ids]
        motion = np.column_stack((self.commands[ids, 0], self._command_motion_sums[ids] / np.maximum(counts[:, None], 1)))
        self.command_curriculum.update(self._command_bins[ids], scores, eligible, motion=motion)
        self.command_ranges[[0, 2]] = self.command_curriculum.active_ranges
        self._command_score_sums[ids] = 0
        self._command_motion_sums[ids] = 0
        self._command_settled_steps[ids] = 0
        self._command_bins[ids] = -1
        self._command_crossed[ids] = False

    def _observation_terms(self, critic):
        terms = super()._observation_terms(critic)
        # Three task commands are [filtered vx, vy=0, raw yaw rate].
        commands = self.commands.copy()
        commands[:, 0] = self.filtered_speed
        terms["commands"] = commands
        if critic:
            terms["reference"] = self.q_ref[:, self.reference.indices]
        return terms

    def _forward_velocity(self):
        # Heading frame excludes torso pitch from the measured forward speed.
        w, x, y, z = self.base_quat.T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.cos(yaw) * self.sim.qvel[:, 0] + np.sin(yaw) * self.sim.qvel[:, 1]

    def _reward_tracking_lin_vel(self):
        return np.exp(-np.square(self.filtered_speed - self._forward_velocity()) / self.cfg.rewards.tracking_sigma)

    def _reward_prior(self):
        ids = self.reference.indices
        return np.exp(-np.square(self.dof_pos[:, ids] - self.q_ref[:, ids]) / self.cfg.rewards.prior_sigma).sum(1)

    def _reward_tracking_ang_vel(self):
        return np.exp(-np.square(self.commands[:, 2] - self.base_ang_vel[:, 2]) / self.cfg.rewards.tracking_ang_sigma)

    def _signed_foot_separation(self):
        # Left-minus-right distance along the robot's heading-frame y axis.
        relative = self.feet_pos[:, 0] - self.feet_pos[:, 1]
        w, x, y, z = self.base_quat.T
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return -np.sin(yaw) * relative[:, 0] + np.cos(yaw) * relative[:, 1]

    def _reward_close_feet(self):
        separation = self._signed_foot_separation()
        if not getattr(self.cfg.rewards, "signed_feet_distance", False):
            separation = np.abs(separation)  # Original paper / legacy checkpoint behavior.
        return np.maximum(0, self.cfg.rewards.min_feet_distance - separation)

    def _reward_air_height(self):
        height = self.feet_pos[:, :, 2] - self.feet_ground_height
        reward = (
            (~self._feet_contact)
            * np.exp(-np.abs(height - self.cfg.rewards.feet_clearance_target) / self.cfg.rewards.feet_clearance_std)
        ).sum(1)
        if getattr(self.cfg.rewards, "air_height_command_gate", False):
            # Match the reference blend during acceleration/deceleration.
            # A zero forward command can still request lateral motion/turning.
            blend = np.clip(np.abs(self.filtered_speed) / self.cfg.reference.blend_speed, 0.0, 1.0)
            blend = np.square(blend) * (3.0 - 2.0 * blend)
            blend[np.any(np.abs(self.commands[:, 1:]) > 1e-6, axis=1)] = 1.0
            reward *= blend
        return reward

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

    def _reward_torso_roll(self):
        deadzone = getattr(self.cfg.rewards, "torso_roll_deadzone", 0.1)
        return np.square(np.maximum(np.abs(self.projected_gravity[:, 1]) - deadzone, 0.0))

    def training_state_dict(self):
        state = super().training_state_dict()
        state["sprint_curriculum"] = self.command_curriculum.state_dict() if self.command_curriculum is not None else None
        state["command_rng_state"] = self.rng.bit_generator.state
        return state

    def load_training_state_dict(self, state):
        if state.get("sprint_curriculum") is None:
            raise ValueError("Checkpoint has no SPRINT curriculum state; start a new training run")
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
            self._camera_request = None

            def camera_key(key):
                if key in (70, 82):  # GLFW F / R; applied under the viewer lock below.
                    self._camera_request = key

            self._viewer = mujoco.viewer.launch_passive(
                self._viewer_model, self._viewer_data, key_callback=camera_key
            )
            self._camera_root = self.base_pos[0].copy()
            with self._viewer.lock():
                self._reset_camera()
        if self._viewer.is_running():
            with self._viewer.lock():
                self.sim.batch.get_data(0, self._viewer_data)
                mujoco.mj_forward(self._viewer_model, self._viewer_data)
                if getattr(self, "follow_camera", True):
                    self._viewer.cam.lookat[:] += self.base_pos[0] - self._camera_root
                self._camera_root = self.base_pos[0].copy()
                request, self._camera_request = self._camera_request, None
                if request == 82:
                    self._reset_camera()
                elif request == 70:
                    self._viewer.cam.lookat[:] = self.base_pos[0] + np.array([
                        0, 0, getattr(self, "camera_height", 0.0)
                    ])
            self._viewer.sync()

    def _reset_camera(self):
        self._viewer.cam.distance = getattr(self, "camera_distance", 3.0)
        self._viewer.cam.azimuth = getattr(self, "camera_azimuth", 135.0)
        self._viewer.cam.elevation = getattr(self, "camera_elevation", -20.0)
        self._viewer.cam.lookat[:] = self.base_pos[0] + np.array([0, 0, getattr(self, "camera_height", 0.0)])

    def close(self):
        self._foot_quat = None
        super().close()
