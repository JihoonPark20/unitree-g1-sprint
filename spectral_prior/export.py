"""Reference sweeps without importing the physics or PPO packages."""

from pathlib import Path

import numpy as np
import torch


def export_reference(prior, output):
    from spectral_prior import JOINT_NAMES
    from spectral_prior.model import load_prior
    from spectral_prior.reference import ReferenceGenerator

    model, mapping, _ = load_prior(prior)
    speeds = torch.linspace(0, 6, 61)
    phase = torch.arange(240) / 240
    with torch.no_grad():
        q = model(phase[None].expand(61, -1), mapping(speeds)[:, None].expand(-1, 240))[0]
    selected_default = np.array([-0.1, 0.3, -0.2, 0.3, 0.97] * 2)
    generator = ReferenceGenerator(prior, JOINT_NAMES, selected_default, 1)
    ramp = np.r_[np.linspace(0, 6, 1500), np.linspace(6, 0, 1500)]
    positions, phases = [], []
    for v in ramp:
        positions.append(generator.evaluate([v])[0][0])
        phases.append(generator.phase.item())
        generator.advance([v], 0.02)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        speeds=speeds.numpy(),
        frequencies=mapping(speeds).numpy(),
        phase=phase.numpy(),
        joint_pos=q.numpy(),
        joint_names=np.array(JOINT_NAMES),
        ramp_speed=ramp,
        ramp_joint_pos=np.array(positions),
        ramp_phase=phases,
        dt=0.02,
    )
    print(path)
