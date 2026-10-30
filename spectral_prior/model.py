"""Frequency VAE, multi-harmonic FiLM and checkpoint loading."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from spectral_prior import JOINT_NAMES


@dataclass
class PriorConfig:
    harmonics: int = 6
    latent_dim: int = 16
    hidden_dim: int = 128
    layers: int = 3
    encoder_dim: int = 64


class SpectralPrior(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or PriorConfig()
        c = self.config
        self.encoder = nn.Sequential(
            nn.Linear(1, c.encoder_dim), nn.SiLU(), nn.Linear(c.encoder_dim, c.encoder_dim), nn.SiLU()
        )
        self.mu = nn.Linear(c.encoder_dim, c.latent_dim)
        self.logvar = nn.Linear(c.encoder_dim, c.latent_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(2 * c.harmonics if i == 0 else c.hidden_dim, c.hidden_dim) for i in range(c.layers)]
        )
        self.film = nn.ModuleList([nn.Linear(c.latent_dim + 1, 2 * c.hidden_dim) for _ in self.layers])
        self.output = nn.Linear(c.hidden_dim, len(JOINT_NAMES))
        self.register_buffer("orders", torch.arange(1, c.harmonics + 1, dtype=torch.float32))
        self.register_buffer("offset", torch.zeros(10))

    def forward(self, phase, frequency, sample=None):
        # phase is integrated cycles, not f*t: command changes cannot jump phase.
        if sample is None:
            sample = self.training
        frequency = frequency.unsqueeze(-1)
        encoded = self.encoder(frequency)
        mu, logvar = self.mu(encoded), self.logvar(encoded).clamp(-12, 8)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu) if sample else mu
        context = torch.cat((z, frequency), dim=-1)
        angle = 2 * torch.pi * phase.unsqueeze(-1) * self.orders
        hidden = torch.stack((angle.sin(), angle.cos()), dim=-1).flatten(-2)
        for layer, film in zip(self.layers, self.film):
            gamma, beta = film(context).chunk(2, dim=-1)
            hidden = F.silu((1 + 0.1 * gamma.tanh()) * layer(hidden) + 0.1 * beta.tanh())
        return self.output(hidden) + self.offset, mu, logvar


class SpeedFrequencyMap(nn.Module):
    def __init__(self, speeds, frequencies):
        super().__init__()
        v, f = torch.tensor(speeds, dtype=torch.float32), torch.tensor(frequencies, dtype=torch.float32)
        if (
            v.ndim != 1
            or v.shape != f.shape
            or len(v) < 2
            or not torch.isfinite(v).all()
            or not torch.isfinite(f).all()
        ):
            raise ValueError("Invalid speed-frequency anchors")
        if not (torch.diff(v) > 0).all() or not (torch.diff(f) > 0).all() or f[0] <= 0:
            raise ValueError("Speed and cadence knots must be strictly increasing and cadence positive")
        self.register_buffer("speeds", v)
        self.register_buffer("frequencies", f)

    def forward(self, speed):
        speed = speed.clamp(self.speeds[0], self.speeds[-1])
        i = torch.searchsorted(self.speeds, speed.contiguous(), right=True).clamp(1, len(self.speeds) - 1)
        weight = (speed - self.speeds[i - 1]) / (self.speeds[i] - self.speeds[i - 1])
        return torch.lerp(self.frequencies[i - 1], self.frequencies[i], weight)


def load_prior(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("version") != 1 or tuple(checkpoint["joint_names"]) != JOINT_NAMES:
        raise ValueError("Unsupported spectral prior / joint contract")
    model = SpectralPrior(PriorConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    mapping = SpeedFrequencyMap(checkpoint["speed_knots"], checkpoint["frequency_knots"]).to(device)
    return model, mapping, checkpoint
