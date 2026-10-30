"""Curate steady GMR motion windows into phase-aligned single-cycle references.

Physical time is preserved during spectral analysis. A cycle is repeated for
10 seconds, never stretched to 10 seconds (which would destroy cadence).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, periodogram, resample, savgol_filter

from spectral_prior import JOINT_NAMES

PAPER_SPEEDS = (0.66, 1.10, 2.29, 2.87, 3.40)
PAPER_FREQUENCIES = (0.68, 0.86, 1.25, 1.36, 1.58)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_motion(path):
    with np.load(path, allow_pickle=False) as data:
        names = data["joint_names"].tolist()
        bodies = data["body_names"].tolist()
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate joint names: {path}")
        ids = [names.index(name) for name in JOINT_NAMES]
        pelvis = bodies.index("pelvis")
        fps = float(data["fps"].item())
        q = np.asarray(data["joint_pos"][:, ids], dtype=float)
        pos = np.asarray(data["body_pos_w"][:, pelvis], dtype=float)
        quat = np.asarray(data["body_quat_w"][:, pelvis], dtype=float)
    if fps <= 0 or len(q) < 8 or not all(np.isfinite(x).all() for x in (q, pos, quat)):
        raise ValueError(f"Invalid motion: {path}")
    w, x, y, z = quat.T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    vel = np.gradient(pos, 1 / fps, axis=0)
    forward = np.cos(yaw) * vel[:, 0] + np.sin(yaw) * vel[:, 1]
    lateral = -np.sin(yaw) * vel[:, 0] + np.cos(yaw) * vel[:, 1]
    return fps, q, forward, lateral, yaw


def spectral_frequency(q, fps, bounds=(0.45, 2.5)):
    """Use both hip pitches for stride frequency, retaining all 10 joint PSDs."""
    frequency, psd = periodogram(q, fs=fps, window="hann", detrend="linear", axis=0, nfft=8 * len(q))
    mask = (frequency >= bounds[0]) & (frequency <= bounds[1])
    power = psd[:, [0, 5]].sum(axis=1)
    index = np.flatnonzero(mask)[np.argmax(power[mask])]
    f = float(frequency[index])
    # Sub-bin quadratic peak estimate; the label remains measured from the PSD.
    if 0 < index < len(power) - 1:
        y = np.log(np.maximum(power[index - 1 : index + 2], 1e-30))
        denominator = y[0] - 2 * y[1] + y[2]
        if denominator < 0:
            f += float(0.5 * (y[0] - y[2]) / denominator * (frequency[1] - frequency[0]))
    # Broad fundamental band allows normal within-window cadence variation.
    concentration = power[np.abs(frequency - f) < 0.2].sum() / max(power[mask].sum(), 1e-12)
    return f, float(concentration), frequency, psd


def extract_cycle(q, fps, frequency, samples=120):
    smooth = savgol_filter(q, max(5, int(fps * 0.14) // 2 * 2 + 1), 3, axis=0)
    peaks, _ = find_peaks(smooth[:, 0], distance=max(1, int(0.7 * fps / frequency)), prominence=0.1)
    periods = np.diff(peaks) / fps
    valid = np.flatnonzero(np.abs(periods * frequency - 1) < 0.2)
    if len(valid) < 2:
        raise ValueError("Fewer than two consistent gait cycles")
    cycles = np.stack([resample(q[peaks[i] : peaks[i + 1]], samples, axis=0) for i in valid])
    median = np.median(cycles, axis=0)
    errors = np.square(cycles - median).mean(axis=(1, 2))
    best = int(np.argmin(errors))
    i = valid[best]
    cycle = savgol_filter(cycles[best], 9, 3, axis=0, mode="wrap")
    return cycle, float(1 / periods[i]), int(peaks[i]), int(peaks[i + 1]), float(errors.mean())


def normalize_reference(q, fps, *, arm_correction=None):
    """Stage I: one measured cycle -> 10 s at 30 Hz -> Fourier 60 Hz -> SG -> PSD.

    Repetition preserves physical cadence. Peak spacing selects the cycle but
    never replaces the final FFT frequency label. Filter/peak settings are
    reproducible implementation choices, not parameters published by the authors.
    """
    if not np.isclose(fps, 30):
        raise ValueError("Stage I requires native 30 Hz GMR data; re-export the original GMR motion at 30 Hz")
    preliminary, _, _, _ = spectral_frequency(q, fps)
    _, peak_frequency, a, b, error = extract_cycle(q, fps, preliminary)
    raw_cycle = q[a:b].copy()
    corrections = []
    if arm_correction is not None:
        if arm_correction != "left_from_right_half_cycle":
            raise ValueError(f"Unknown arm correction: {arm_correction}")
        left = [JOINT_NAMES.index(f"left_{joint}_joint") for joint in ("shoulder_pitch", "elbow")]
        right = [JOINT_NAMES.index(f"right_{joint}_joint") for joint in ("shoulder_pitch", "elbow")]
        # Shift the selected physical cycle, not the 10 s window (which need
        # not contain an integer number of cycles). 2N samples also handles
        # odd-length native cycles without rounding the half-cycle offset.
        n = len(raw_cycle)
        right_upsampled = resample(raw_cycle[:, right], 2 * n, axis=0)
        raw_cycle[:, left] = right_upsampled[(2 * np.arange(n) + n) % (2 * n)]
        corrections.append({
            "method": arm_correction,
            "source_joints": [JOINT_NAMES[i] for i in right],
            "target_joints": [JOINT_NAMES[i] for i in left],
            "phase_shift_cycles": 0.5,
            "native_cycle_frames": n,
            "angle_sign": 1,
            "stage": "after original cycle selection, before repetition/resampling/filtering",
            "reason": "User-requested left-arm correction from the right arm; not original GMR data",
        })
    repeated = np.tile(raw_cycle, (int(np.ceil(300 / len(raw_cycle))), 1))[:300]
    trajectory = savgol_filter(resample(repeated, 600, axis=0), 9, 3, axis=0, mode="wrap")
    frequency, concentration, bins, psd = spectral_frequency(trajectory, 60)
    cycle = savgol_filter(resample(raw_cycle, 120, axis=0), 9, 3, axis=0, mode="wrap")
    return dict(
        cycle=cycle,
        trajectory=trajectory,
        frequency=frequency,
        fft_frequency=frequency,
        peak_frequency=peak_frequency,
        spectral_concentration=concentration,
        cycle_mse=error,
        cycle_start=a,
        cycle_end=b,
        psd_bins=bins,
        psd=psd,
        corrections=corrections,
    )


def candidates_from_file(path, window_s=8.0, stride_s=2.0):
    fps, q, forward, lateral, yaw = load_motion(path)
    if not np.isclose(fps, 30):
        raise ValueError(f"Stage I requires native 30 Hz GMR data: {path}")
    width = round(window_s * fps)
    result = []
    for start in range(0, len(q) - width + 1, max(1, round(stride_s * fps))):
        end = start + width
        speed = float(np.mean(forward[start:end]))
        cv = float(np.std(forward[start:end]) / max(speed, 0.1))
        turn = float(np.ptp(yaw[start:end]))
        side = float(np.mean(np.abs(lateral[start:end])))
        if speed < 0.4 or cv > 0.3 or turn > 0.6 or side > 0.3:
            continue
        freq, concentration, _, _ = spectral_frequency(q[start:end], fps)
        if concentration < 0.55:
            continue
        try:
            cycle, measured, a, b, error = extract_cycle(q[start:end], fps, freq)
        except ValueError:
            continue
        result.append(
            {
                "source": str(Path(path).resolve()),
                "start_frame": start,
                "end_frame": end,
                "cycle_start_frame": start + a,
                "cycle_end_frame": start + b,
                "source_fps": fps,
                "speed": speed,
                "frequency": measured,
                "fft_frequency": freq,
                "spectral_concentration": concentration,
                "speed_cv": cv,
                "yaw_range": turn,
                "cycle_mse": error,
                "quality": cv + turn + error * 5 + (1 - concentration),
                "cycle": cycle,
            }
        )
    return result


def build_library(input_dir, output_dir, targets=PAPER_SPEEDS, selections=None):
    """Automatic curation or explicit JSON list of source/start_frame/end_frame.

    All selected values are measured; paper speeds are selection targets only.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    if selections:
        for item in json.loads(Path(selections).read_text()):
            path = Path(item["source"])
            if not path.is_absolute():
                path = Path(input_dir) / path
            fps, q, forward, _, _ = load_motion(path)
            if not np.isclose(item.get("source_fps", fps), fps):
                raise ValueError("Selection frame rate differs from the source motion")
            start, end = int(item["start_frame"]), int(item["end_frame"])
            if not 0 <= start < end <= len(q):
                raise ValueError("Invalid manual frame range")
            reference = normalize_reference(q[start:end], fps, arm_correction=item.get("arm_correction"))
            candidates.append(
                dict(
                    source=str(path.resolve()),
                    start_frame=start,
                    end_frame=end,
                    cycle_start_frame=start + reference.pop("cycle_start"),
                    cycle_end_frame=start + reference.pop("cycle_end"),
                    source_fps=fps,
                    speed=float(forward[start:end].mean()),
                    quality=reference["cycle_mse"],
                    **reference,
                )
            )
        selected = sorted(candidates, key=lambda c: c["speed"])
    else:
        paths = sorted(Path(input_dir).glob("*.npz"))
        if not paths:
            raise ValueError(f"No NPZ motions in {input_dir}")
        for path in paths:
            found = []
            for window in (2.5, 4.0, 6.0):
                found.extend(candidates_from_file(path, window_s=window, stride_s=0.5))
            candidates.extend(found)
            print(f"{path.name}: {len(found)} steady windows", flush=True)
        selected = []
        for target in sorted(targets):
            pool = [
                c
                for c in candidates
                if 0.6 < c["frequency"] < 2.3
                and (
                    not selected
                    or (c["speed"] > selected[-1]["speed"] + 0.1 and c["frequency"] > selected[-1]["frequency"] + 0.025)
                )
            ]
            if not pool:
                raise ValueError("Insufficient monotonic speed/cadence windows; supply --selections")
            best = min(pool, key=lambda c: abs(c["speed"] - target) + 0.3 * c["quality"])
            selected.append(best)
        for c in selected:
            fps, q, *_ = load_motion(c["source"])
            reference = normalize_reference(q[c["start_frame"] : c["end_frame"]], fps)
            c["cycle_start_frame"] = c["start_frame"] + reference.pop("cycle_start")
            c["cycle_end_frame"] = c["start_frame"] + reference.pop("cycle_end")
            c.update(reference)
    speeds = np.array([c["speed"] for c in selected])
    frequencies = np.array([c["frequency"] for c in selected])
    if len(selected) != 5 or np.any(np.diff(speeds) <= 0) or np.any(np.diff(frequencies) <= 0):
        raise ValueError("Library needs exactly five increasing distinct speed and frequency anchors")
    # Endpoint guards are explicit deployment assumptions from the paper's range.
    if speeds[0] <= 0 or speeds[-1] >= 6 or frequencies[0] <= 0.6 or frequencies[-1] >= 2.3:
        raise ValueError("Measured anchors must lie inside (0,6) m/s and (0.6,2.3) Hz")
    times = np.arange(600) / 60
    phases = times[None, :] * frequencies[:, None] % 1
    trajectories = [c["trajectory"] for c in selected]
    np.savez_compressed(
        output / "library.npz",
        joint_names=np.array(JOINT_NAMES),
        fps=60,
        source_fps=30,
        preprocessing_version=3 if any(c.get("corrections") for c in selected) else 2,
        phase=phases.astype(np.float32),
        joint_pos=np.array(trajectories, dtype=np.float32),
        cycles=np.stack([c["cycle"] for c in selected]).astype(np.float32),
        speeds=speeds,
        frequencies=frequencies,
    )

    def clean(c):
        return {k: v for k, v in c.items() if k not in ("cycle", "trajectory", "psd_bins", "psd")}

    def source_metadata(c):
        with np.load(c["source"], allow_pickle=False) as source:
            return json.loads(source["metadata_json"].item()) if "metadata_json" in source else {}

    report = {
        "version": 2,
        "method": "Native 30 Hz cycle repetition to 10 s; Fourier resample to 60 Hz; Savitzky-Golay; FFT PSD labels",
        "parameters": {
            "duration_s": 10,
            "source_fps": 30,
            "output_fps": 60,
            "savgol_window": 9,
            "savgol_polyorder": 3,
            "savgol_mode": "wrap",
            "psd_window": "hann",
            "psd_detrend": "linear",
            "fft_padding_factor": 8,
            "frequency_estimator": "bilateral hip pitch PSD sum; quadratic log-power peak",
        },
        "paper_speeds": list(PAPER_SPEEDS),
        "paper_frequencies": list(PAPER_FREQUENCIES),
        "paper_anchor_match_2dp": bool(
            np.array_equal(np.round(speeds, 2), PAPER_SPEEDS)
            and np.array_equal(np.round(frequencies, 2), PAPER_FREQUENCIES)
        ),
        "author_reference_identity_verified": False,
        "selected": [
            {**clean(c), "sha256": sha256(c["source"]), "source_metadata": source_metadata(c)} for c in selected
        ],
        "candidate_count": len(candidates),
        "selection_targets": list(targets),
        "speed_knots": [0.0, *speeds.tolist(), 6.0],
        "frequency_knots": [0.6, *frequencies.tolist(), 2.3],
    }
    (output / "curation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "candidates.json").write_text(json.dumps([clean(c) for c in candidates], indent=2) + "\n")
    # Preserve per-joint PSD evidence for selected windows.
    psds = {}
    for i, c in enumerate(selected):
        psds[f"frequency_{i}"] = c["psd_bins"]
        psds[f"psd_{i}"] = c["psd"]
    np.savez_compressed(output / "spectra.npz", **psds)
    return report
