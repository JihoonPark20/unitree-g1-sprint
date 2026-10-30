"""Supervised spectral prior training and evaluation."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from spectral_prior import JOINT_NAMES
from spectral_prior.data import sha256
from spectral_prior.model import SpectralPrior


def train_prior(library, output, steps=5000, batch_size=512, lr=1e-3, beta_kl=1e-4, device="cpu", seed=42, config=None):
    if steps < 1 or batch_size < 1 or lr <= 0 or beta_kl < 0:
        raise ValueError("Invalid prior training settings")
    torch.manual_seed(seed)
    directory = Path(library).parent
    report = json.loads((directory / "curation.json").read_text())
    with np.load(library, allow_pickle=False) as data:
        if tuple(data["joint_names"]) != JOINT_NAMES:
            raise ValueError("Library joint order mismatch")
        phase = torch.tensor(data["phase"], device=device)
        target = torch.tensor(data["joint_pos"], device=device)
        freq = torch.tensor(data["frequencies"], dtype=torch.float32, device=device)[:, None].expand_as(phase)
    # Split by phase bins, not repeated timestamps of the same periodic target.
    validation = ((phase * 60).long() % 5 == 0).flatten()
    phase, freq, target = phase.flatten(), freq.flatten(), target.reshape(-1, 10)
    train_ids = (~validation).nonzero().flatten()
    model = SpectralPrior(config).to(device)
    model.offset.copy_(target.mean(0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, history = float("inf"), []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, steps + 1):
        model.train()
        ids = train_ids[torch.randint(len(train_ids), (batch_size,), device=device)]
        pred, mu, logvar = model(phase[ids], freq[ids])
        rec = F.mse_loss(pred, target[ids])
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
        loss = rec + beta_kl * kl
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite spectral prior loss")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(phase[validation], freq[validation])[0], target[validation]).item()
            row = dict(step=step, train_mse=rec.item(), kl=kl.item(), phase_holdout_mse=val)
            history.append(row)
            print(json.dumps(row), flush=True)
            if val < best:
                best = val
                torch.save(
                    dict(
                        version=1,
                        config=asdict(model.config),
                        model=model.state_dict(),
                        joint_names=list(JOINT_NAMES),
                        speed_knots=report["speed_knots"],
                        frequency_knots=report["frequency_knots"],
                        library_sha256=sha256(library),
                        seed=seed,
                        step=step,
                        beta_kl=beta_kl,
                        validation_mse=val,
                    ),
                    output,
                )
    from spectral_prior.metrics import evaluate_prior

    metrics = evaluate_prior(output, library)
    metrics.update(
        phase_holdout_mse=best,
        validation_scope="Phase interpolation of curated cycles; not unseen-motion generalization",
        history=history,
    )
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
