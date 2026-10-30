"""Frozen batched reference generator; independent of the simulator."""

import numpy as np
import torch

from spectral_prior import JOINT_NAMES
from spectral_prior.model import load_prior


class ReferenceGenerator:
    """Frozen deterministic batched oscillator with continuous standing blend."""

    def __init__(self, checkpoint, joint_names, default_pose, num_envs, blend_speed=0.4):
        self.prior, self.mapping, _ = load_prior(checkpoint)
        self.indices = np.array([list(joint_names).index(n) for n in JOINT_NAMES])
        self.default = torch.as_tensor(default_pose, dtype=torch.float32)
        self.phase = torch.zeros(num_envs)
        self.blend_speed = blend_speed
        if blend_speed <= 0:
            raise ValueError("blend_speed must be positive")

    def reset(self, ids):
        self.phase[torch.as_tensor(ids)] = 0

    def advance(self, speed, dt):
        if dt < 0:
            raise ValueError("dt must be nonnegative")
        speed = torch.as_tensor(speed, dtype=torch.float32)
        self.phase = (self.phase + self.mapping(speed) * dt) % 1

    @torch.no_grad()
    def evaluate(self, speed):
        speed = torch.as_tensor(speed, dtype=torch.float32)
        f = self.mapping(speed)
        generated = self.prior(self.phase, f, sample=False)[0]
        weight = (speed / self.blend_speed).clamp(0, 1)
        weight = weight.square() * (3 - 2 * weight)
        reference = self.default.expand(len(speed), -1).clone()
        reference[:, self.indices] = torch.lerp(self.default[self.indices], generated, weight[:, None])
        return reference.numpy(), f.numpy()
