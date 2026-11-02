"""NumPy implementation of RLvRL's reward-threshold command curriculum.

References: https://arxiv.org/abs/2205.02824, section III-D;
https://github.com/Improbable-AI/rapid-locomotion-rl (RewardThresholdCurriculum).
This implementation uses bounded cells and the public implementation's local
box neighborhood / incremental weights. Successful cells are deduplicated per
update, so parallel copies cannot instantly saturate a neighbor's weight.
The environment owns segment duration, acceleration and termination handling.
"""

import numpy as np


class GridAdaptiveCurriculum:
    def __init__(self, limits, bin_widths, initial_ranges, thresholds, increment=0.2, local_range=None, progress=None):
        self.limits = np.asarray(limits, dtype=float)
        widths = np.asarray(bin_widths, dtype=float)
        initial = np.asarray(initial_ranges, dtype=float)
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.increment = float(increment)
        self.progress = None if progress is None else dict(progress)
        if self.progress is not None and (
            set(self.progress) != {"min_command", "min_speed_ratio", "max_forward_mae", "max_yaw_mae"}
            or not all(
                (key == "max_yaw_mae" and value is None) or (value is not None and np.isfinite(value) and value > 0)
                for key, value in self.progress.items()
            )
            or self.progress["min_speed_ratio"] > 1
        ):
            raise ValueError("Invalid physical command-progress criteria")
        if (
            self.limits.shape != (2, 2) or widths.shape != (2,) or initial.shape != (2, 2)
            or self.thresholds.shape != (2,)
            or not all(np.isfinite(x).all() for x in (self.limits, widths, initial, self.thresholds))
            or np.any(widths <= 0) or np.any(np.diff(self.limits, axis=1) < 0)
            or not np.any(np.diff(self.limits, axis=1) > 0)
            or np.any(initial[:, 0] > initial[:, 1])
            or np.any(initial[:, 0] < self.limits[:, 0]) or np.any(initial[:, 1] > self.limits[:, 1])
            or np.any((self.thresholds < 0) | (self.thresholds > 1))
            or not 0 < self.increment <= 1
        ):
            raise ValueError("Invalid grid curriculum ranges, widths, thresholds or increment")
        # Fixed axes have one zero-width cell, so yaw=[0, 0] samples exactly
        # zero even after expansion. Other axes partition their limits exactly.
        counts = np.maximum(1, np.ceil(np.round(np.diff(self.limits, axis=1).ravel() / widths, 12)).astype(int))
        self.shape = tuple(counts.tolist())
        edges = [np.linspace(lo, hi, n + 1) for (lo, hi), n in zip(self.limits, counts)]
        self.low = np.stack(np.meshgrid(*(x[:-1] for x in edges), indexing="ij"), axis=-1).reshape(-1, 2)
        self.high = np.stack(np.meshgrid(*(x[1:] for x in edges), indexing="ij"), axis=-1).reshape(-1, 2)
        self.centers = (self.low + self.high) / 2
        self.local_range = np.asarray(widths if local_range is None else local_range, dtype=float)
        if self.local_range.shape != (2,) or not np.isfinite(self.local_range).all() or np.any(self.local_range <= 0):
            raise ValueError("Grid local_range must contain two positive finite radii")
        active = ((self.centers >= initial[:, 0] - 1e-12) & (self.centers <= initial[:, 1] + 1e-12)).all(1)
        if not active.any():
            raise ValueError("Initial command range contains no grid cell centers")
        self.weights = active.astype(float)
        self.trials = np.zeros(len(active), dtype=np.int64)
        self.successes = np.zeros(len(active), dtype=np.int64)
        self.score_sums = np.zeros((len(active), 2))

    def sample(self, rng, size):
        bins = rng.choice(len(self.weights), size=size, p=self.weights / self.weights.sum())
        return rng.uniform(self.low[bins], self.high[bins]), bins

    def update(self, bins, scores, eligible, motion=None):
        """Credit completed command segments; scores are unweighted mean kernels.

        Ineligible segments (falls / too little settled time) count as attempts,
        but cannot expand the grid. Bin -1 denotes dedicated standing samples.
        """
        bins = np.asarray(bins, dtype=int)
        scores = np.asarray(scores, dtype=float)
        eligible = np.asarray(eligible, dtype=bool)
        if scores.shape != (len(bins), 2) or eligible.shape != bins.shape or not np.isfinite(scores).all():
            raise ValueError("Invalid command-segment scores")
        if np.any((bins < -1) | (bins >= len(self.weights))):
            raise ValueError("Command bin outside curriculum grid")
        if self.progress is not None:
            # Settled segment means: command vx, actual signed vx, vx MAE,
            # yaw MAE. A high kernel score alone can reward standing still.
            motion = np.asarray(motion, dtype=float)
            if motion.shape != (len(bins), 4) or not np.isfinite(motion).all():
                raise ValueError("Progress curriculum requires finite command / motion measurements")
            command, forward, vx_error, yaw_error = motion.T
            p = self.progress
            eligible = eligible & (command >= p["min_command"]) & (
                forward >= p["min_speed_ratio"] * command
            ) & (vx_error <= p["max_forward_mae"])
            if p["max_yaw_mae"] is not None:
                eligible &= yaw_error <= p["max_yaw_mae"]
        valid = bins >= 0
        bins, scores, eligible = bins[valid], scores[valid], eligible[valid]
        np.add.at(self.trials, bins, 1)
        np.add.at(self.score_sums, bins, scores)
        successful = eligible & (scores >= self.thresholds).all(1)
        np.add.at(self.successes, bins[successful], 1)
        passed = np.unique(bins[successful])
        if len(passed):
            neighbors = (np.abs(self.centers[:, None] - self.centers[passed]) <= self.local_range + 1e-12).all(2).any(1)
            self.weights[neighbors] = np.minimum(1.0, self.weights[neighbors] + self.increment)

    @property
    def active_ranges(self):
        active = self.weights > 0
        return np.stack((self.low[active].min(0), self.high[active].max(0)), axis=1)

    def metrics(self):
        active = self.weights > 0
        attempts = self.trials.sum()
        mean = self.score_sums.sum(0) / max(1, attempts)
        return {
            "Curriculum/active_fraction": float(active.mean()),
            "Curriculum/active_bins": int(active.sum()),
            "Curriculum/segment_success_fraction": float(self.successes.sum() / max(1, attempts)),
            "Curriculum/mean_linear_score": float(mean[0]),
            "Curriculum/mean_angular_score": float(mean[1]),
            "Curriculum/completed_segments": int(attempts),
            "Curriculum/max_abs_yaw": float(np.abs(self.active_ranges[1]).max()),
        }

    def _contract(self):
        contract = {
            "version": 1, "kind": "rlvrl_grid", "limits": self.limits.tolist(), "shape": list(self.shape),
            "thresholds": self.thresholds.tolist(), "increment": self.increment, "local_range": self.local_range.tolist(),
        }
        if self.progress is not None:
            contract.update(version=2, progress=self.progress)
        return contract

    def state_dict(self):
        return {
            **self._contract(), "weights": self.weights.tolist(), "trials": self.trials.tolist(),
            "successes": self.successes.tolist(), "score_sums": self.score_sums.tolist(),
        }

    def load_state_dict(self, state):
        if any(state.get(k) != v for k, v in self._contract().items()):
            raise ValueError("Checkpoint grid curriculum configuration mismatch; start a new run")
        arrays = {key: np.asarray(state[key], dtype=float) for key in ("weights", "trials", "successes", "score_sums")}
        if any(value.shape != getattr(self, key).shape or not np.isfinite(value).all() for key, value in arrays.items()):
            raise ValueError("Invalid grid curriculum checkpoint arrays")
        w, trials, successes = arrays["weights"], arrays["trials"], arrays["successes"]
        if (
            np.any((w < 0) | (w > 1)) or w.sum() <= 0 or np.any(trials < 0)
            or np.any(successes < 0) or np.any(successes > trials)
            or np.any(trials != np.floor(trials)) or np.any(successes != np.floor(successes))
        ):
            raise ValueError("Invalid grid curriculum weights or counts")
        for key, value in arrays.items():
            getattr(self, key)[:] = value
