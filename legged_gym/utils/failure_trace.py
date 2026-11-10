"""Bounded, pre-reset histories for first-episode failure analysis."""

import numpy as np

FAILURE_NAMES = ("tilt", "low_height", "base_contact", "feet_crossing")


class FailureTrace:
    def __init__(self, num_envs, dt, seconds):
        self.dt = dt
        self.capacity = max(1, int(np.ceil(seconds / dt)))
        self.buffers = {}
        self.step = 0
        self.summaries = [None] * num_envs
        self.rows = []

    def record(self, sample, active, dones):
        if not self.buffers:
            self.buffers = {k: np.empty((self.capacity, *v.shape), dtype=v.dtype)
                            for k, v in sample.items()}
        for key, value in sample.items():
            self.buffers[key][self.step % self.capacity] = value
        self.step += 1
        for index in np.flatnonzero(active & dones):
            self._finish(index, failed=True)

    def _finish(self, index, *, failed):
        if self.summaries[index] is not None:
            raise RuntimeError("A reset episode must not replace a failure trace")
        steps = np.arange(max(0, self.step - self.capacity), self.step)
        row = {k: v[steps % self.capacity, index].copy() for k, v in self.buffers.items()}
        n = len(steps)
        row.update(env_index=np.full(n, index), time_s=(steps + 1) * self.dt,
                   time_to_end_s=(steps + 1 - self.step) * self.dt, failed=np.full(n, failed))
        self.rows.append(row)
        contacts = row["feet_contact"]
        self.summaries[index] = {
            "failed": bool(failed), "end_time_s": self.step * self.dt,
            "window_samples": n,
            "failure_reasons": [name for name, flag in zip(FAILURE_NAMES, row["failure_flags"][-1]) if flag],
            "terminal_roll_rad": float(row["roll_rad"][-1]),
            "terminal_pitch_rad": float(row["pitch_rad"][-1]),
            "terminal_tilt_rad": float(row["tilt_rad"][-1]),
            "terminal_phase": float(row["phase"][-1]),
            "terminal_feet_contact": contacts[-1].tolist(),
            "window_support_changes": int(np.any(contacts[1:] != contacts[:-1], axis=1).sum()),
            "window_single_support_fraction": float((contacts.sum(1) == 1).mean()),
            "window_no_support_fraction": float((contacts.sum(1) == 0).mean()),
        }

    def save(self, path, active, joint_names, foot_names, *, control=None):
        for index in np.flatnonzero(active):
            self._finish(index, failed=False)
        arrays = {key: np.concatenate([r[key] for r in self.rows]) for key in self.rows[0]}
        np.savez_compressed(path, **arrays, joint_names=np.asarray(joint_names),
                            foot_names=np.asarray(foot_names), failure_names=np.asarray(FAILURE_NAMES), **(control or {}))
