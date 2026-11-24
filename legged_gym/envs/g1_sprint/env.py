"""SPRINT task adapter for the integrated MuJoCo legged_gym interface.

Step/reset flow adapted from legged_gym; see licenses/legged_gym_LICENSE.
"""

import numpy as np
import torch

from legged_gym.envs.base.legged_robot import SprintRobot, rotate_inverse
from legged_gym.envs.curriculums import GridAdaptiveCurriculum
from legged_gym.envs.g1_sprint.calibration import kinematic_speed_calibration
from legged_gym.envs.g1_sprint.residual import ResidualDynamics
from legged_gym.sim.mujoco_batch import joint_parameters, load_model
from spectral_prior.reference import ReferenceGenerator


class SprintEnv(SprintRobot):
    def __init__(self, cfg, **kwargs):
        if cfg.reference.phase_modulation not in (0, 0.1):
            raise ValueError("phase_modulation must be 0 or 0.1")
        self.auxiliary_actions = []
        if cfg.reference.phase_modulation:
            self.auxiliary_actions.append("phase")
        if cfg.control.learned_leak:
            if (cfg.control.residual_mode == "position" or not np.isfinite(cfg.control.lambda_min)
                    or not 0 <= cfg.control.lambda_min < cfg.control.residual_leak):
                raise ValueError("Learned leak requires residual dynamics with 0 <= lambda_min < initial residual_leak")
            self.auxiliary_actions.append("leak")
        joint_count = len(cfg.asset.action_joint_names)
        if self.auxiliary_actions and cfg.env.num_actions == joint_count:
            cfg.env.num_actions = joint_count + len(self.auxiliary_actions)
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
        if cfg.control.residual_mode != "position":
            self._diagnostic_names += ("Control/anti_windup_fraction",)
        self._diagnostic_sums = np.zeros((cfg.env.num_envs, len(self._diagnostic_names)))
        self._joint_residual_sums = np.zeros((cfg.env.num_envs, 3, len(cfg.asset.action_joint_names)))
        self._torque_saturation_sums = np.zeros((cfg.env.num_envs, len(cfg.asset.joint_names)))
        default = joint_parameters(cfg.asset.joint_names, cfg.init_state.default_joint_angles, 0)
        self.reference = ReferenceGenerator(
            cfg.reference.checkpoint, cfg.asset.joint_names, default, cfg.env.num_envs, cfg.reference.blend_speed
        )
        calibration = getattr(cfg.reference, "speed_calibration", None)
        if calibration == "g1_kinematic":
            self.reference.speed_calibration = kinematic_speed_calibration(
                load_model(cfg), cfg.asset.joint_names, default, self.reference, cfg.asset.feet,
                1.0 - cfg.rewards.swing_duty,
            )
        elif calibration is not None:
            raise ValueError(f"Unknown reference.speed_calibration: {calibration}")
        self._target_interp = getattr(cfg.control, "target_interp", None)
        if self._target_interp not in (None, "zoh", "linear", "exact"):
            raise ValueError("target_interp must be None, zoh, linear, or exact")
        mode = cfg.control.residual_mode
        leak, ff, mix = cfg.control.residual_leak, cfg.control.residual_feedforward, cfg.control.residual_mix_direct
        if mode not in ("position", "lowpass", "integrator"):
            raise ValueError("residual_mode must be position, lowpass, or integrator")
        if not np.isfinite([leak, ff, mix]).all() or leak < 0 or ff < 0 or not 0 <= mix <= 1:
            raise ValueError("Residual leak and feedforward must be nonnegative and direct mix in [0, 1]")
        if mode == "lowpass" and leak <= 0:
            raise ValueError("Lowpass residual dynamics require a positive leak")
        if mix and mode != "lowpass":
            raise ValueError("Direct position mixing requires a lowpass target in radians")
        kappa = cfg.control.increment_kappa
        self._increment_action_limits = None
        if kappa is not None:
            if not np.isfinite(kappa) or kappa <= 0 or self.residual_limits is None:
                raise ValueError("increment_kappa requires a positive finite fraction and per-joint residual limits")
            if mode == "integrator" and leak == 0:
                self._increment_action_limits = kappa * self.residual_limits / cfg.control.action_scale
            elif mode != "position":
                raise ValueError("Bounded increments require pure integration or an integrated-mean position policy")
        self._uses_residual_state = mode != "position" or ff != 0 or cfg.control.residual_obs
        self.residual_dynamics = None
        if self._uses_residual_state:
            if self.residual_limits is None:
                raise ValueError("Residual state requires per-joint residual_limits")
            if self._target_interp is None and (mode != "position" or ff):
                self._target_interp = "linear" if cfg.control.reference_lookahead else "zoh"
        self._legacy_lookahead = self._target_interp is None and getattr(cfg.control, "reference_lookahead", False)
        self._substep_history = None
        self.q_ref = np.tile(default, (cfg.env.num_envs, 1))
        self.dq_ref = np.zeros_like(self.q_ref)
        self.q_ref_aligned = self.q_ref.copy()
        self.frequency = np.full(cfg.env.num_envs, 0.6)
        self.filtered_speed = np.zeros(cfg.env.num_envs)
        self.target_position = self.q_ref.copy()
        self.target_velocity = np.zeros_like(self.q_ref)
        self._latency = np.zeros(cfg.env.num_envs, dtype=int)
        self._lookahead_s = 0.0
        self._pd_target = self.q_ref.copy()
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
                focus=getattr(cfg.commands, "curriculum_focus", None),
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
        # Cycle-averaged linear score: open-cycle speed error / steps, and steps of the segment's completed cycles.
        self._cycle_error_sums = np.zeros(cfg.env.num_envs)
        self._cycle_steps = np.zeros(cfg.env.num_envs, dtype=int)
        self._command_cycle_steps = np.zeros(cfg.env.num_envs, dtype=int)
        self.evaluation_latency_steps = None
        self.evaluation_impulse = None
        self._evaluation_history = None
        self._last_eval_reference = None
        self.record_evaluation = False
        self.record_failure_trace = False
        super().__init__(cfg, **kwargs)
        self._foot_quat = self.sim.batch.bind("xquat")

    def _update_reference(self):
        if getattr(self.cfg.control, "reference_velocity", False) or self._target_interp in ("linear", "exact"):
            self.q_ref[:], self.dq_ref[:], self.frequency[:] = self.reference.evaluate_with_velocity(
                self.filtered_speed, self.cfg.sim.dt,
            )
        else:
            self.q_ref[:], self.frequency[:] = self.reference.evaluate(self.filtered_speed)
            self.dq_ref.fill(0)

    def reset_idx(self, ids):
        super().reset_idx(ids)
        if self._uses_residual_state:
            if self.residual_dynamics is None:
                self.residual_dynamics = ResidualDynamics(self)
            self.residual_dynamics.reset(ids)
        self._diagnostic_sums[ids] = 0
        self._joint_residual_sums[ids] = 0
        self._torque_saturation_sums[ids] = 0
        self._feet_crossing_steps[ids] = 0
        self.feet_crossing_buf[ids] = False
        self.filtered_speed[ids] = 0
        self.reference.reset(ids)
        delay = self.evaluation_latency_steps
        if delay is None:
            delay = getattr(self.cfg.control, "fixed_latency_steps", None)
        if delay is None:
            self._latency[ids] = self.rng.integers(0, self.cfg.control.max_latency_steps + 1, len(ids))
        else:
            if delay not in (0, 1, 2):
                raise ValueError("Fixed latency must be 0, 1, or 2 policy steps")
            if delay >= len(self._targets):
                extra = delay + 1 - len(self._targets)
                self._targets = np.concatenate((self._targets, np.repeat(self._targets[-1:], extra, axis=0)))
                self._target_velocities = np.concatenate((
                    self._target_velocities, np.repeat(self._target_velocities[-1:], extra, axis=0),
                ))
                self._evaluation_history = None
            self._latency[ids] = delay
        self._update_reference()
        self.target_position[ids] = self.q_ref[ids]
        self._targets[:, ids] = self.q_ref[ids]
        self.target_velocity[ids] = 0
        self._target_velocities[:, ids] = 0
        if self._evaluation_history is not None:
            ref, vel, residual = self._evaluation_history
            ref[:, ids] = self.q_ref[ids]
            vel[:, ids] = 0
            residual[:, ids] = 0
        if self._last_eval_reference is not None:
            self._last_eval_reference[ids] = np.nan
            self._last_eval_offset[ids] = np.nan

        if self._target_interp is not None and len(self._targets) > 1:
            depth = len(self._targets)
            if self._substep_history is None:
                shape = (depth, self.num_envs, self.cfg.control.decimation, self.num_dof)
                self._substep_history = (np.zeros(shape), np.zeros(shape), np.zeros((depth, *self.actions.shape)))
                if self.cfg.control.learned_leak:
                    self._substep_history += (np.full((depth, self.num_envs, 1), self.cfg.control.residual_leak),)
            elif len(self._substep_history[0]) < depth:
                self._substep_history = tuple(
                    np.concatenate((h, np.repeat(h[-1:], depth - len(h), axis=0))) for h in self._substep_history
                )
            positions, velocities, inputs = self._substep_history[:3]
            if self.cfg.control.learned_leak:
                self._substep_history[3][:, ids] = self.cfg.control.residual_leak
            positions[:, ids] = self.q_ref[ids, None]
            velocities[:, ids] = 0
            inputs[:, ids] = 0

    def _issue_substep_targets(self, control_input, leak=None):
        count = self.cfg.control.decimation
        dt = self.cfg.sim.dt
        if self._target_interp == "exact":
            position, velocity = self.reference.evaluate_substeps(self.filtered_speed, dt, count)
        else:
            shape = (self.num_envs, count, self.num_dof)
            position = np.broadcast_to(self.q_ref[:, None], shape)
            velocity = np.broadcast_to(self.dq_ref[:, None], shape)
            if self._target_interp == "linear":
                times = (dt * np.arange(1, count + 1))[None, :, None]
                position = position + times * velocity
        packet = (position, velocity, control_input)
        if leak is not None:
            packet += (leak,)
        if self._substep_history is None:
            return packet
        rows = np.arange(self.num_envs)
        for history, value in zip(self._substep_history, packet):
            history[1:] = history[:-1].copy()
            history[0] = value
        return tuple(history[self._latency, rows] for history in self._substep_history)

    def _apply_substep_target(self, reference, velocity, residual, limits):
        self.target_position[:] = reference
        self.target_position[:, self.action_dof_indices] += residual
        self.target_position[:, self.held_dof_indices] = self.default_dof_pos[self.held_dof_indices]
        self.target_velocity[:] = velocity
        if self.cfg.control.residual_feedforward:
            self.target_velocity[:, self.action_dof_indices] += (
                self.cfg.control.residual_feedforward * self.residual_dynamics.rate
            )
        self.target_velocity[:, self.held_dof_indices] = 0
        lower, upper = limits.T
        outside = (self.target_position < lower) | (self.target_position > upper)
        self._diagnostic_sums[:, 2] += outside.mean(1) / self.cfg.control.decimation
        self.target_velocity[((self.target_position <= lower) & (self.target_velocity < 0))
                             | ((self.target_position >= upper) & (self.target_velocity > 0))] = 0
        np.clip(self.target_position, lower, upper, out=self.target_position)

    def _record_applied_residual(self, offset, count):
        square = np.square(offset)
        self._joint_residual_sums[:, 0] += square / count
        self._joint_residual_sums[:, 1] += (np.abs(offset) >= 0.99 * self.residual_limits) / count
        self._diagnostic_sums[:, 0] += np.sqrt(square.mean(1)) / count

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
        target = self.target_position
        if self._legacy_lookahead:
            # Move the held target along the reference so the P and D terms agree within the control interval.
            limits = self.sim.model.jnt_range[self.sim.joint_ids]
            target = np.clip(target + self.target_velocity * self._lookahead_s, limits[:, 0], limits[:, 1])
        self._pd_target = target
        torque = self.p_gains * (target - pos) + self.d_gains * (self.target_velocity - vel)
        return np.clip(torque, -self.torque_limits, self.torque_limits)

    def _hold_target(self, target, limits):
        self._targets[1:] = self._targets[:-1].copy()
        self._targets[0] = np.clip(target, limits[:, 0], limits[:, 1])
        velocity = self.dq_ref.copy()
        velocity[:, self.held_dof_indices] = 0
        velocity[((target <= limits[:, 0]) & (velocity < 0)) | ((target >= limits[:, 1]) & (velocity > 0))] = 0
        self._target_velocities[1:] = self._target_velocities[:-1].copy()
        self._target_velocities[0] = velocity
        self.target_position[:] = self._targets[self._latency, np.arange(self.num_envs)]
        self.target_velocity[:] = self._target_velocities[self._latency, np.arange(self.num_envs)]

    def _evaluation_inputs(self, residual, limits):
        if self._evaluation_history is None:
            shape = (*self._targets.shape[:2], residual.shape[-1])
            self._evaluation_history = (self._targets.copy(), self._target_velocities.copy(), np.zeros(shape))
        reference = self.q_ref.copy()
        reference[:, self.held_dof_indices] = self.default_dof_pos[self.held_dof_indices]
        velocity = self.dq_ref.copy()
        velocity[:, self.held_dof_indices] = 0
        velocity[((reference <= limits[:, 0]) & (velocity < 0))
                 | ((reference >= limits[:, 1]) & (velocity > 0))] = 0
        inputs = (np.clip(reference, limits[:, 0], limits[:, 1]), velocity, residual)
        for history, value in zip(self._evaluation_history, inputs):
            history[1:] = history[:-1].copy()
            history[0] = value
        rows = np.arange(self.num_envs)
        return tuple(history[self._latency, rows] for history in self._evaluation_history)

    def _apply_evaluation_impulse(self):
        impulse = np.asarray(self.evaluation_impulse, dtype=float)
        if impulse.shape != (self.num_envs,) or not np.isfinite(impulse).all():
            raise ValueError("Expected one finite signed lateral impulse (N s) per environment")
        self.evaluation_impulse = None
        w, x, y, z = self.sim.qpos[:, 3:7].T
        yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
        force = self.sim.batch.bind("xfrc_applied")[:, self.sim.model.body(self.cfg.asset.root_body).id, :2]
        original = force.copy()
        force += np.column_stack((-np.sin(yaw), np.cos(yaw))) * (impulse / self.cfg.sim.dt)[:, None]
        return force, original

    def step(self, actions):
        if self._closed:
            raise RuntimeError("Environment is closed")
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        action_shape = (self.num_envs, self.num_actions)
        if actions.shape != action_shape or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite policy actions of shape {action_shape}")
        leak = None
        if self.auxiliary_actions:
            joint_count = len(self.action_dof_names)
            if self.cfg.reference.phase_modulation:
                scale = 1 + self.cfg.reference.phase_modulation * np.tanh(actions[:, joint_count])
                self.reference.phase_rate_scale = torch.as_tensor(scale, dtype=torch.float32)
                self._update_reference()
            if self.cfg.control.learned_leak:
                leak = self.cfg.control.lambda_min + np.logaddexp(0, actions[:, -1:].astype(float))
            actions = actions[:, :joint_count]
        self.actions[:] = np.clip(actions, -self.cfg.control.clip_actions, self.cfg.control.clip_actions)
        if self.residual_limits is not None and self.cfg.control.residual_mode != "integrator":
            action_limits = self.residual_limits / self.cfg.control.action_scale
            self.actions[:] = np.clip(self.actions, -action_limits, action_limits)
        if self._increment_action_limits is not None:
            self.actions[:] = np.clip(self.actions, -self._increment_action_limits, self._increment_action_limits)
        # Previous-action observations describe the bounded residual actually
        # requested from PD, while PPO retains its original sampled actions.
        residual = self.cfg.control.action_scale * self.actions
        if self._increment_action_limits is not None:
            residual = residual / self.dt
        clipped = np.abs(actions - self.actions) > 1e-7
        bound = self.cfg.control.action_scale * self.cfg.control.clip_actions
        if self.residual_limits is not None:
            bound = np.minimum(bound, self.residual_limits)
        if self.residual_dynamics is None:
            self._joint_residual_sums[:, 0] += np.square(residual)
            self._joint_residual_sums[:, 1] += np.abs(residual) >= 0.99 * bound
            self._diagnostic_sums[:, 0] += np.sqrt(np.square(residual).mean(1))
        self._joint_residual_sums[:, 2] += clipped
        self._diagnostic_sums[:, 1] += clipped.mean(1)
        limits = self.sim.model.jnt_range[self.sim.joint_ids]
        packet = None
        if self._target_interp is None:
            target = self.q_ref.copy()
            target[:, self.action_dof_indices] += residual
            target[:, self.held_dof_indices] = self.default_dof_pos[self.held_dof_indices]
            self._diagnostic_sums[:, 2] += ((target < limits[:, 0]) | (target > limits[:, 1])).mean(1)
        else:
            packet = self._issue_substep_targets(residual, leak)
        if packet is None and (self.record_evaluation or self.residual_dynamics is not None):
            eval_ref, eval_velocity, eval_residual = self._evaluation_inputs(residual, limits)
            if self.residual_dynamics is not None:
                self.residual_dynamics.applied[:] = eval_residual
                self._record_applied_residual(eval_residual, 1)
        if self.record_evaluation:
            if self._last_eval_reference is None:
                self._last_eval_reference = np.full_like(self.q_ref, np.nan)
                self._last_eval_offset = np.full_like(self.q_ref, np.nan)
            if packet is not None:
                eval_residual = packet[2]
            shape = (self.cfg.control.decimation, *self.torques.shape)
            trace = {
                "requested_torque": np.empty(shape),
                "applied_torque": np.empty(shape),
                "joint_velocity": np.empty(shape),
                "residual": np.repeat(eval_residual[None], shape[0], axis=0),
                "action": self.actions.copy(),
                "action_clipped": clipped.copy(),
                "anti_windup": np.zeros((shape[0], *eval_residual.shape), dtype=bool),
            }
        if packet is None:
            self._hold_target(target, limits)
        self._push_robots()
        self._first_contact.fill(False)
        self._touchdown_air_time.fill(0)
        if self.record_failure_trace:
            step_saturation = np.zeros_like(self.torques)
            step_torque_peak = np.zeros_like(self.torques)
        for substep in range(self.cfg.control.decimation):
            self._lookahead_s = (substep + 1) * self.cfg.sim.dt
            if packet is not None:
                reference = packet[0][:, substep]
                velocity = packet[1][:, substep]
                offset = packet[2]
                dynamics = self.residual_dynamics
                if dynamics is not None:
                    applied_leak = packet[3] if self.cfg.control.learned_leak else None
                    dynamics.advance(packet[2], reference, velocity, self.sim, first=substep == 0, leak=applied_leak)
                    offset = dynamics.applied
                    count = self.cfg.control.decimation
                    self._record_applied_residual(offset, count)
                    if dynamics.mode != "position":
                        self._diagnostic_sums[:, -1] += dynamics.blocked.mean(1) / count
                    if self.record_evaluation:
                        trace["residual"][substep] = offset
                        trace["anti_windup"][substep] = dynamics.blocked
                self._apply_substep_target(reference, velocity, offset, limits)
            self.torques[:] = self._compute_torques(self.actions)
            if self.record_evaluation or self.record_failure_trace:
                joint_velocity = self.sim.qvel[:, self.sim.qvel_ids]
                p_torque = self.p_gains * (self._pd_target - self.sim.qpos[:, self.sim.qpos_ids])
                d_torque = self.d_gains * (self.target_velocity - joint_velocity)
                requested_torque = p_torque + d_torque
            if self.record_evaluation:
                trace["requested_torque"][substep] = requested_torque
                trace["applied_torque"][substep] = self.torques
                trace["joint_velocity"][substep] = joint_velocity
                if packet is not None:
                    neutral = np.clip(reference, limits[:, 0], limits[:, 1])
                    neutral[:, self.held_dof_indices] = self.default_dof_pos[self.held_dof_indices]
                elif self._legacy_lookahead:
                    neutral = np.clip(eval_ref + eval_velocity * self._lookahead_s, limits[:, 0], limits[:, 1])
                else:
                    neutral = eval_ref
                offset = self._pd_target - neutral
                if substep == 0:
                    trace["p_jump_reference"] = self.p_gains * (neutral - self._last_eval_reference)
                    trace["p_jump_residual"] = self.p_gains * (offset - self._last_eval_offset)
                if substep == self.cfg.control.decimation - 1:
                    self._last_eval_reference[:] = neutral
                    self._last_eval_offset[:] = offset
            if self.record_failure_trace:
                step_saturation += (np.abs(self.torques) >= 0.99 * self.torque_limits) / self.cfg.control.decimation
                np.maximum(step_torque_peak, np.abs(self.torques), out=step_torque_peak)
            self._torque_saturation_sums += (
                np.abs(self.torques) >= 0.99 * self.torque_limits
            ) / self.cfg.control.decimation
            observe = substep == self.cfg.control.decimation - 1
            if substep == 0 and self.evaluation_impulse is not None:
                force, original = self._apply_evaluation_impulse()
                try:
                    self.sim.step(self.torques, observe=observe)
                finally:
                    force[:] = original
            else:
                self.sim.step(self.torques, observe=observe)
            self._update_contacts()
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self._refresh_state()
        self.check_termination()
        # Reference at the post-step state's instant (t+dt), before the phase advances.
        self.q_ref_aligned = self.reference.evaluate(self.filtered_speed, time_offset=self.dt)[0]
        # Evaluate the reference and command that actually produced this transition.
        self.compute_reward()
        if self.record_evaluation:
            trace["prior_reward"] = self._reward_prior()
            trace["joint_velocity_end"] = self.dof_vel.copy()
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
                **trace,
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
                    ("issued_residual_rate_rad_s" if self.cfg.control.residual_mode == "integrator"
                     else "issued_residual_rad"): residual.copy(),
                    "applied_residual_rad": trace["residual"][-1].copy(),
                    "joint_pos_rad": self.dof_pos[:, self.action_dof_indices].copy(),
                    "pd_target_rad": self._pd_target[:, self.action_dof_indices].copy(),
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
        # Same-instant error regardless of the reward setting, so runs stay comparable.
        self._diagnostic_sums[:, 3] += np.abs(self.dof_pos[:, indices] - self.q_ref_aligned[:, indices]).mean(1)
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
        self._cycle_error_sums[ids] = 0
        self._cycle_steps[ids] = 0
        self._command_cycle_steps[ids] = 0
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
        if getattr(self.cfg.commands, "curriculum_cycle_average", False):
            self._cycle_error_sums[settled] += (self.filtered_speed - self._forward_velocity())[settled]
            self._cycle_steps[settled] += 1
            self._cycle_error_sums[~settled] = 0  # A scored cycle must be settled throughout.
            self._cycle_steps[~settled] = 0
            # The phase wraps on this step's advance: credit the cycle-mean kernel to each of its steps.
            cadence = self.reference.modulated_frequency(self.reference.frequency(self.filtered_speed))
            wraps = self.reference.phase.numpy() + cadence.numpy() * self.dt >= 1
            done = settled & wraps
            steps = self._cycle_steps[done]
            mean_error = self._cycle_error_sums[done] / steps
            self._command_score_sums[done, 0] += steps * np.exp(-np.square(mean_error) / self.cfg.rewards.tracking_sigma)
            self._command_cycle_steps[done] += steps
            self._cycle_error_sums[done] = 0
            self._cycle_steps[done] = 0
        else:
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
        # Cycle mode: the linear score covers completed cycles only (none -> 0, cannot promote).
        linear_counts = self._command_cycle_steps[ids] if getattr(self.cfg.commands, "curriculum_cycle_average", False) else counts
        scores = self._command_score_sums[ids] / np.maximum(np.column_stack((linear_counts, counts)), 1)
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
        self._cycle_error_sums[ids] = 0
        self._cycle_steps[ids] = 0
        self._command_cycle_steps[ids] = 0
        self._command_bins[ids] = -1
        self._command_crossed[ids] = False

    def _observation_terms(self, critic):
        terms = super()._observation_terms(critic)
        # Three task commands are [filtered vx, vy=0, raw yaw rate].
        commands = self.commands.copy()
        commands[:, 0] = self.filtered_speed
        terms["commands"] = commands
        if self.cfg.reference.phase_obs and not critic:
            angle = 2 * np.pi * self.reference.phase.numpy()
            terms["phase"] = np.column_stack((np.sin(angle), np.cos(angle)))
        if self.cfg.control.residual_obs and not critic:
            terms["residual"] = self.residual_dynamics.applied / self.residual_limits
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

    def _reward_action_rate(self):
        valid = self.episode_length_buf.numpy() > 1
        return np.square(self.actions - self.last_actions).mean(axis=1) * valid

    def _reward_prior(self):
        ids = self.reference.indices
        aligned = getattr(self.cfg.reference, "prior_reward_time_aligned", False)
        q_ref = self.q_ref_aligned if aligned else self.q_ref
        return np.exp(-np.square(self.dof_pos[:, ids] - q_ref[:, ids]) / self.cfg.rewards.prior_sigma).sum(1)

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
        airborne = ~self._feet_contact
        if getattr(self.cfg.rewards, "air_height_swing_cap", False):
            airborne &= self.feet_air_time <= (self.cfg.rewards.swing_duty / self.frequency)[:, None]
        reward = (
            airborne
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
