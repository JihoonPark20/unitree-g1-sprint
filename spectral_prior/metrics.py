"""Paper equations 15–17, with explicit joint-space features for Fréchet distance."""

import json
from pathlib import Path

import numpy as np
import torch

from spectral_prior import JOINT_NAMES
from spectral_prior.data import PAPER_FREQUENCIES, PAPER_SPEEDS, sha256
from spectral_prior.model import load_prior


def frechet_distance(reference, generated):
    """Gaussian Fréchet distance on rows of unnormalized joint angles (radians).

    Symmetric PSD square roots avoid complex sqrtm results for singular gait
    covariances. No diagonal jitter is added to the reported distance.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in (reference, generated)]
    if any(x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Fréchet distance requires at least two finite feature rows per distribution")
    a, b = arrays
    if a.shape[1] != b.shape[1]:
        raise ValueError("Feature dimensions differ")
    ca, cb = np.atleast_2d(np.cov(a, rowvar=False)), np.atleast_2d(np.cov(b, rowvar=False))
    values, vectors = np.linalg.eigh(ca)
    root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    product = root @ cb @ root
    cross = np.sqrt(np.maximum(np.linalg.eigvalsh((product + product.T) / 2), 0)).sum()
    return float(max(0, np.square(a.mean(0) - b.mean(0)).sum() + np.trace(ca) + np.trace(cb) - 2 * cross))


@torch.no_grad()
def evaluate_prior(prior_path, library_path):
    model, mapping, checkpoint = load_prior(prior_path)
    if checkpoint.get("library_sha256") != sha256(library_path):
        raise ValueError("Prior/library hash mismatch: evaluate against the library used to train this prior")
    with np.load(library_path, allow_pickle=False) as data:
        if tuple(data["joint_names"]) != JOINT_NAMES or data["joint_pos"].shape != (5, 600, 10):
            raise ValueError("Expected five 10-second, 60 Hz reference sequences in canonical joint order")
        phase = torch.tensor(data["phase"], dtype=torch.float32)
        frequencies = data["frequencies"].copy()
        speeds = data["speeds"].copy()
        reference = data["joint_pos"].astype(np.float64)
    freq = torch.tensor(frequencies, dtype=torch.float32)[:, None].expand_as(phase)
    generated = model(phase, freq, sample=False)[0].numpy()
    # Dense, endpoint-exclusive phase samples avoid frequency-dependent maxima.
    grid = torch.arange(4096, dtype=torch.float32) / 4096
    boundaries = [float(frequencies[0]), float(frequencies[-1])]
    extremes = [0.6, 2.3]
    amplitude_errors = []
    for extreme, boundary in zip(extremes, boundaries):
        qext = model(grid, torch.full_like(grid, extreme), sample=False)[0].numpy()
        qbound = model(grid, torch.full_like(grid, boundary), sample=False)[0].numpy()
        amplitude_errors.append(np.abs(np.ptp(qext, axis=0) - np.ptp(qbound, axis=0)))
    return {
        "schema_version": 1,
        "paper": "https://arxiv.org/abs/2605.28549v1",
        "prior_sha256": sha256(prior_path),
        "library_sha256": sha256(library_path),
        "frequencies_hz": frequencies.tolist(),
        "speeds_m_s": speeds.tolist(),
        "paper_anchor_match_2dp": bool(
            np.array_equal(np.round(frequencies, 2), PAPER_FREQUENCIES)
            and np.array_equal(np.round(speeds, 2), PAPER_SPEEDS)
        ),
        "author_reference_identity_verified": False,
        "L_rec": float(np.square(generated - reference).mean()),
        "L_rec_per_frequency": np.square(generated - reference).mean(axis=(1, 2)).tolist(),
        "E_BA": float(np.mean(amplitude_errors)),
        "E_BA_per_extreme": np.mean(amplitude_errors, axis=1).tolist(),
        "FID_joint": frechet_distance(reference.reshape(-1, 10), generated.reshape(-1, 10)),
        "protocol": {
            "generation": "deterministic posterior mean; five measured library frequencies; 600 timestamps each",
            "amplitude": "peak-to-peak; 4096 uniform phase samples; generated boundary vs generated extreme",
            "extreme_frequencies_hz": extremes,
            "boundary_frequencies_hz": boundaries,
            "fid_features": "10 raw joint angles in radians per frame; pooled frequencies; unbiased covariance",
            "fid_comparability": "Paper does not specify feature extraction; FID_joint is not table-equivalent FID",
            "scope": "reference reconstruction, not held-out motion generalization",
        },
    }


def write_report(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
