"""Bounded residual position and rate at the physics timestep."""

import numpy as np


class ResidualDynamics:
    def __init__(self, env):
        cfg = env.cfg.control
        self.mode = cfg.residual_mode
        self.leak = cfg.residual_leak
        self.mix = cfg.residual_mix_direct
        self.ff = cfg.residual_feedforward
        self.dt = env.cfg.sim.dt
        self.policy_dt = env.dt
        self.indices = env.action_dof_indices
        self.qids = env.sim.qpos_ids[self.indices]
        self.vids = env.sim.qvel_ids[self.indices]
        self.lower, self.upper = env.sim.model.jnt_range[env.sim.joint_ids[self.indices]].T
        self.kp = env.p_gains[self.indices]
        self.kd = env.d_gains[self.indices]
        self.effort = env.torque_limits[self.indices]
        self.limit = env.residual_limits
        self.state = np.zeros((env.num_envs, len(self.indices)))
        self.applied = np.zeros_like(self.state)
        self.rate = np.zeros_like(self.state)
        self.blocked = np.zeros_like(self.state, dtype=bool)

    def reset(self, ids):
        self.state[ids] = self.applied[ids] = self.rate[ids] = 0
        self.blocked[ids] = False

    def advance(self, command, reference, velocity, sim, *, first):
        if self.mode == "position":
            if first:
                self.rate[:] = (command - self.applied) / self.policy_dt
                self.applied[:] = command
            return

        gain = -np.expm1(-self.leak * self.dt) / self.leak if self.leak else self.dt
        drive = self.leak * command if self.mode == "lowpass" else command
        proposed = self.state + gain * (drive - self.leak * self.state)
        bounded = np.clip(proposed, -self.limit, self.limit)
        increment = bounded - self.state
        mix_weight = 1 - self.mix
        base_offset = mix_weight * self.state + self.mix * command
        target = reference[:, self.indices] + base_offset
        base_torque = (self.kp * (target - sim.qpos[:, self.qids])
                       + self.kd * (velocity[:, self.indices] - sim.qvel[:, self.vids]))

        # Each radian of state change moves both the P target and the residual D target.
        torque_slope = mix_weight * (self.kp + self.kd * self.ff / self.dt)
        with np.errstate(divide="ignore"):
            tau_lo = np.where(torque_slope > 0, (-self.effort - base_torque) / torque_slope, -np.inf)
            tau_hi = np.where(torque_slope > 0, (self.effort - base_torque) / torque_slope, np.inf)
        if mix_weight > 0:
            lower = np.minimum(0, np.maximum((self.lower - target) / mix_weight, tau_lo))
            upper = np.maximum(0, np.minimum((self.upper - target) / mix_weight, tau_hi))
            accepted = np.clip(increment, lower, upper)
        else:
            accepted = increment
        self.blocked[:] = (bounded != proposed) | (accepted != increment)
        self.state += accepted
        self.applied[:] = mix_weight * self.state + self.mix * command
        self.rate[:] = mix_weight * accepted / self.dt
