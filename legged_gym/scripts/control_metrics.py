"""Measured controller statistics for first-episode trials."""

import numpy as np
from scipy.signal import welch

PITCH_JOINTS = tuple(f"{side}_{joint}_joint" for joint in ("hip_pitch", "knee", "ankle_pitch") for side in ("left", "right"))
SMOOTH = ("action_rate", "action_acc", "joint_jerk")
SAMPLE_MEANS = ("prior_reward", "action_clip_fraction", "torque_saturation", "anti_windup_fraction", "residual_square")


def spectral_energy(torque, dt):
    """Sum Welch bin powers across complete one-second Hann windows, per joint."""
    width = round(1 / dt)
    overlap = width // 2
    if len(torque) < width:
        return None
    frequencies, psd = welch(torque, fs=1 / dt, window="hann", nperseg=width, noverlap=overlap, axis=0)
    windows = 1 + (len(torque) - width) // (width - overlap)
    high = (frequencies >= 20) & (frequencies <= 100)
    return {"windows": windows, "high": (windows * psd[high].sum(0)).tolist(), "total": (windows * psd.sum(0)).tolist()}


class ControlMetrics:
    """One evaluation seed/command batch. Each environment contributes one measured prefix."""

    def __init__(self, env):
        self.ids = [env.dof_names.index(joint) for joint in PITCH_JOINTS]
        self.action_ids = env.action_dof_indices
        self.limits = env.torque_limits
        self.dt = env.dt
        self.physics_dt = env.cfg.sim.dt
        self.torques = []
        self.counts = np.zeros(env.num_envs, dtype=int)
        self.sums = {key: np.zeros(env.num_envs) for key in SAMPLE_MEANS + SMOOTH}
        shape = (env.num_envs, len(env.dof_names))
        self.jump_squares = {key: np.zeros(shape) for key in ("p_jump_reference", "p_jump_residual")}
        self.jump_counts = np.zeros(env.num_envs, dtype=int)
        self.prev_action = self.prev_velocity = self.prev_action_diff = self.prev_velocity_diff = None

    def add(self, sample, active):
        torques = sample["applied_torque"]
        self.torques.append(torques[:, :, self.ids].copy())
        values = {
            "residual_square": np.square(sample["residual"]).mean(axis=(0, 2)),
            "action_clip_fraction": sample["action_clipped"].mean(axis=1),
            "anti_windup_fraction": sample["anti_windup"].mean(axis=(0, 2)),
            "torque_saturation": (np.abs(torques) >= 0.99 * self.limits).mean(axis=(0, 2)),
            "prior_reward": sample["prior_reward"],
        }
        for key, value in values.items():
            self.sums[key][active] += value[active]

        ref = sample["p_jump_reference"]
        residual = sample["p_jump_residual"]
        valid = active & (self.counts > 0) & np.isfinite(ref).all(1) & np.isfinite(residual).all(1)
        self.jump_counts += valid
        for key in self.jump_squares:
            self.jump_squares[key][valid] += np.square(sample[key][valid])

        action = sample["action"]
        velocity = sample["joint_velocity_end"][:, self.action_ids]
        if self.prev_action is not None:
            da, dv = action - self.prev_action, velocity - self.prev_velocity
            self.sums["action_rate"][active] += np.square(da[active]).mean(1)
            if self.prev_action_diff is not None:
                self.sums["action_acc"][active] += np.square(da[active] - self.prev_action_diff[active]).mean(1)
                jerk = np.abs(dv[active] - self.prev_velocity_diff[active]) / self.dt**2
                self.sums["joint_jerk"][active] += jerk.mean(1)
            self.prev_action_diff, self.prev_velocity_diff = da, dv
        self.prev_action, self.prev_velocity = action.copy(), velocity.copy()
        self.counts += active

    def finish(self):
        data = np.concatenate(self.torques, axis=0) if self.torques else None
        results = []
        for i, ticks in enumerate(self.counts):
            samples = int(ticks)
            substeps = samples * round(self.dt / self.physics_dt)
            results.append({
                "samples": samples,
                "sums": {key: float(values[i]) for key, values in self.sums.items()},
                "spectrum": spectral_energy(data[:substeps, i], self.physics_dt) if substeps else None,
                "jump_count": int(self.jump_counts[i]),
                "jump_squares": {key: values[i].tolist() for key, values in self.jump_squares.items()},
            })
        return results


def summarize_trials(trials):
    """Pool valid controller samples; survival includes every requested trial."""
    stats = [trial["controller"] for trial in trials]
    count = sum(s["samples"] for s in stats)
    result = {"trials": len(trials), "samples": count,
              "survival": float(np.mean([t["complete"] for t in trials])) if trials else None}
    for key in SAMPLE_MEANS + SMOOTH:
        lost = {"action_rate": 1, "action_acc": 2, "joint_jerk": 2}.get(key, 0)
        n = sum(max(0, s["samples"] - lost) for s in stats)
        result[key] = sum(s["sums"][key] for s in stats) / n if n else None
        if key in SMOOTH:
            result[f"{key}_samples"] = n
    square = result.pop("residual_square")
    result["residual_rms"] = float(np.sqrt(square)) if square is not None else None
    spectra = [s["spectrum"] for s in stats if s["spectrum"] is not None]
    result["welch_windows"] = sum(s["windows"] for s in spectra)
    result["E_hi"] = None
    if spectra:
        total = np.sum([s["total"] for s in spectra], axis=0)
        if (total > 0).all():
            result["E_hi"] = float(np.mean(np.sum([s["high"] for s in spectra], axis=0) / total))
    jumps = sum(s["jump_count"] for s in stats)
    for key in ("p_jump_reference", "p_jump_residual"):
        result[f"{key}_rms_per_joint"] = (
            np.sqrt(np.sum([s["jump_squares"][key] for s in stats], axis=0) / jumps).tolist() if jumps else None
        )
    result["jump_samples"] = jumps
    exposed = [t for t in trials if t["pushed"]]
    result["pushed_trials"] = len(exposed)
    result["push_recovery"] = float(np.mean([t["complete"] for t in exposed])) if exposed else None
    return result


def speed_summary(rows):
    trials = [trial for row in rows for trial in row["trials"]]
    summary = summarize_trials(trials)
    for key in ("E_vel", "E_qpos"):
        scores = [row[f"{key}_observed"] for row in rows if row[f"{key}_observed"] is not None]
        summary[key] = float(np.mean(scores)) if scores else None
        summary[f"{key}_covered_speeds"] = len(scores)
    summary["commands"] = [r["command_speed"] for r in rows]
    return summary


def summarize_bins(rows):
    groups = {label: [] for label in ("0-2", "2-4", "4-6")}
    for row in rows:
        speed = row["command_speed"]
        groups["0-2" if speed < 2 else "2-4" if speed < 4 else "4-6"].append(row)
    return {label: speed_summary(group) for label, group in groups.items()}
