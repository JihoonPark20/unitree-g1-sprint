"""Frozen batched reference generator; independent of the simulator."""

import numpy as np
import torch

from spectral_prior import JOINT_NAMES
from spectral_prior.model import load_prior


class ReferenceGenerator:
    """Frozen deterministic batched oscillator with continuous standing blend."""

    def __init__(self, checkpoint, joint_names, default_pose, num_envs, blend_speed=0.4, speed_calibration=None):
        self.prior, self.mapping, _ = load_prior(checkpoint)
        self.indices = np.array([list(joint_names).index(n) for n in JOINT_NAMES])
        self.default = torch.as_tensor(default_pose, dtype=torch.float32)
        self.phase = torch.zeros(num_envs)
        self.blend_speed = blend_speed
        if blend_speed <= 0:
            raise ValueError("blend_speed must be positive")
        self.speed_calibration = None
        if speed_calibration is not None:
            # (robot_speeds, prior_speeds): commanded speed -> prior speed label producing it on the robot.
            robot, prior = (np.asarray(x, dtype=np.float64) for x in speed_calibration)
            if robot.shape != prior.shape or robot.ndim != 1 or len(robot) < 2 or np.any(np.diff(robot) <= 0):
                raise ValueError("speed_calibration needs two equal-length, strictly increasing knot arrays")
            self.speed_calibration = (robot, prior)

    def frequency(self, speed):
        speed = torch.as_tensor(speed, dtype=torch.float32)
        if self.speed_calibration is not None:
            speed = torch.as_tensor(np.interp(speed.numpy(), *self.speed_calibration), dtype=torch.float32)
        return self.mapping(speed)

    def reset(self, ids):
        self.phase[torch.as_tensor(ids)] = 0

    def advance(self, speed, dt):
        if dt < 0:
            raise ValueError("dt must be nonnegative")
        self.phase = (self.phase + self.frequency(speed) * dt) % 1

    @torch.no_grad()
    def evaluate(self, speed, *, time_offset=0.0):
        speed = torch.as_tensor(speed, dtype=torch.float32)
        f = self.frequency(speed)
        phase = self.phase if time_offset == 0 else (self.phase + f * time_offset) % 1
        generated = self.prior(phase, f, sample=False)[0]
        weight = (speed / self.blend_speed).clamp(0, 1)
        weight = weight.square() * (3 - 2 * weight)
        reference = self.default.expand(len(speed), -1).clone()
        reference[:, self.indices] = torch.lerp(self.default[self.indices], generated, weight[:, None])
        return reference.numpy(), f.numpy()

    def evaluate_with_velocity(self, speed, dt):
        """Phase velocity at the current command; residual actions remain position offsets."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Velocity differentiation requires a positive finite timestep")
        position, frequency = self.evaluate(speed)
        before, _ = self.evaluate(speed, time_offset=-0.5 * dt)
        after, _ = self.evaluate(speed, time_offset=0.5 * dt)
        return position, (after - before) / dt, frequency

    def evaluate_substeps(self, speed, dt, count):
        """Positions and central-difference velocities at dt, 2dt, ... from the issued phase."""
        positions, velocities = [], []
        before = self.evaluate(speed, time_offset=0.5 * dt)[0]
        for k in range(1, count + 1):
            positions.append(self.evaluate(speed, time_offset=k * dt)[0])
            after = self.evaluate(speed, time_offset=(k + 0.5) * dt)[0]
            velocities.append((after - before) / dt)
            before = after
        return np.stack(positions, axis=1), np.stack(velocities, axis=1)
