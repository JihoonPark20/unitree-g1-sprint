"""Bounded residual position and rate at the physics timestep."""

import numpy as np


class ResidualDynamics:
    def __init__(self, env):
        cfg = env.cfg.control
        self.mode = cfg.residual_mode
        self.leak = cfg.residual_leak
        self.mix = cfg.residual_mix_direct
        self.ff = cfg.residual_feedforward
        self.limits_only = getattr(cfg, "frozen_limits_only", False)
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

    def advance(self, command, reference, velocity, sim, *, first, leak=None):
        if self.mode == "position":
            if first:
                self.rate[:] = (command - self.applied) / self.policy_dt
                self.applied[:] = command
            return

        if leak is None:
            gain = -np.expm1(-self.leak * self.dt) / self.leak if self.leak else self.dt
            drive = self.leak * command if self.mode == "lowpass" else command
            proposed = self.state + gain * (drive - self.leak * self.state)
        else:
            blend = -np.expm1(-leak * self.dt)
            if self.mode == "lowpass":
                proposed = self.state + blend * (command - self.state)
            else:
                gain = np.divide(blend, leak, out=np.full_like(leak, self.dt), where=leak > 0)
                proposed = (1 - blend) * self.state + gain * command
        bounded = np.clip(proposed, -self.limit, self.limit)
        increment = bounded - self.state
        mix_weight = 1 - self.mix
        accepted = increment if self.limits_only else self._project(increment, command, reference, velocity, sim)
        self.blocked[:] = (bounded != proposed) | (accepted != increment)
        self.state += accepted
        self.applied[:] = mix_weight * self.state + self.mix * command
        self.rate[:] = mix_weight * accepted / self.dt

    def _project(self, increment, command, reference, velocity, sim):
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
            return np.clip(increment, lower, upper)
        return increment


class HeldLowpass(ResidualDynamics):
    """Use one exact low-pass update and one anti-windup decision per policy tick."""

    def __init__(self, env):
        super().__init__(env)
        if self.mode != "lowpass" or self.ff:
            raise ValueError("The policy-rate comparator requires lowpass and zero feedforward")
        self.dt = self.policy_dt

    def advance(self, command, reference, velocity, sim, *, first, leak=None):
        if first:
            super().advance(command, reference, velocity, sim, first=True, leak=leak)
        else:
            self.blocked.fill(False)
            self.rate.fill(0)


class ResidualRamp(ResidualDynamics):
    """Integrate a constant rate from the accepted offset to each absolute command."""

    def __init__(self, env):
        super().__init__(env)
        if self.mode != "integrator" or self.leak or self.ff or self.mix:
            raise ValueError("The ramp comparator requires pure integration and zero feedforward")
        self.command_rate = np.zeros_like(self.state)

    def reset(self, ids):
        super().reset(ids)
        self.command_rate[ids] = 0

    def advance(self, command, reference, velocity, sim, *, first, leak=None):
        if first:
            self.command_rate[:] = (command - self.state) / self.policy_dt
        super().advance(self.command_rate, reference, velocity, sim, first=first)
