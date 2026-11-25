"""Load independent training-seed replicates and retain their experimental provenance."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from legged_gym.scripts.control_metrics import PITCH_JOINTS, speed_summary
from spectral_prior.data import sha256

GRID = tuple(i / 10 for i in range(61))
BANDS = {"0-2": GRID[:20], "2-4": GRID[20:40], "4-6": GRID[40:]}
ARMS = ("A0", "A1", "A1-R", "F", "B-obs", "B-ff", "B", "Hyb", "D-ctrl", "D", "C1-env", "C1-mean", "C2")
TRIALS = {(seed, env) for seed in (42, 43, 44) for env in range(8)}


def read(path):
    return json.loads(Path(path).read_text())


def variant_name(arm, options):
    parts = [arm]
    for key, value in sorted(options.items()):
        suffix = f"{value:g}" if isinstance(value, float) else str(value)
        parts.append(f"{key.replace('_', '-')}-{suffix}")
    return "_".join(parts)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def effective_latency(fixed, maximum):
    return int(fixed) if fixed is not None else (0 if maximum == 0 else None)


def summarize(rows):
    result = speed_summary(rows)
    spectra = [t["controller"]["spectrum"] for row in rows for t in row["trials"] if t["controller"]["spectrum"]]
    covered = sum(any(t["controller"]["spectrum"] for t in row["trials"]) for row in rows)
    result["spectral_covered_speeds"] = covered
    joints = None
    if spectra:
        total = np.sum([s["total"] for s in spectra], axis=0)
        if np.all(total > 0):
            joints = (np.sum([s["high"] for s in spectra], axis=0) / total).tolist()
    result["E_hi_per_joint"] = joints
    return result


def protocol(report, row):
    p = report["protocol"]
    require((p["physics_dt"], p["control_dt"], p["warmup_seconds"], p["measurement_seconds"]) ==
            (.005, .02, 8., 20.), "timesteps or trial duration differ")
    require(p["seeds"] == [42, 43, 44] and p["trials_per_seed_speed"] == 8, "evaluation seed or environment count differs")
    require(tuple(p["spectral_joints"]) == PITCH_JOINTS, "spectral joint order differs")
    delay = effective_latency(p["latency_steps_training"], p["latency_max_training"])
    require(delay == row["train_latency"] and
            (delay if p["latency_steps"] is None else p["latency_steps"]) == row["eval_latency"], "latency mismatch")
    require(p["push_impulse_Ns"] == row["impulse"], "impulse mismatch")
    speeds = tuple(sorted(r["command_speed"] for r in report["results"]))
    require(speeds == GRID or (row["impulse"] > 0 and speeds == (3.,)), "incomplete command grid")
    for speed in report["results"]:
        trials = speed["trials"]
        require(len(trials) == len(TRIALS) and {(t["seed"], t["env_index"]) for t in trials} == TRIALS,
                "missing or duplicate evaluation trial")
        for trial in trials:
            n = trial["samples"]
            require(trial["controller"]["samples"] == n and 0 <= n <= 1000 and
                    (not trial["complete"] or n == 1000), "invalid trial coverage")
            require(trial["complete"] == (trial["failure_time_s"] is None), "inconsistent survival metadata")


def training_signature(cfg, report):
    env, train = cfg["env"], cfg["train"]
    policy = {k: v for k, v in train["policy"].items() if k != "integrate_mean"}
    rewards = dict(env["rewards"])
    scales = dict(rewards["scales"])
    scales.pop("action_rate", None)
    rewards["scales"] = scales
    values = (cfg["task"], report["prior_sha256"], train["algorithm"], policy, train["runner"]["num_steps_per_env"],
              rewards, env["commands"], env["domain_rand"], env["termination"], env["noise"], env["sim"],
              env["env"]["num_envs"], env["asset"]["action_joint_names"], env["control"]["stiffness"],
              env["control"]["damping"], env["control"]["residual_limits"])
    return json.dumps(values, sort_keys=True)


@dataclass
class Replicate:
    row: dict
    path: Path | None = None
    error: str | None = None
    signature: str = ""
    alpha: float = 0.
    kappa: float | None = None
    checkpoint_hash: str = ""
    action_penalty: float = 0.
    controller_signature: str = ""
    options: dict = field(default_factory=dict)
    phase_context: str = ""
    phase_modulation: float = 0.
    phase_obs: bool = False
    bands: dict = field(default_factory=dict)
    survival: dict = field(default_factory=dict)

    def band(self, commands):
        key = tuple(commands)
        if key not in self.bands and key and self.path and not self.error:
            results = read(self.path)["results"]
            chosen = [s for s in results if s["command_speed"] in key]
            if len(chosen) == len(key):
                self.bands[key] = summarize(chosen)
        return self.bands.get(key)


def load_run(root, row):
    r = Replicate(row)
    if row.get("status") != "complete":
        r.error = row.get("reason") or row.get("status", "pending")
        return r
    try:
        r.path = root / row["evaluation"]
        report = read(r.path)
        require(report.get("kind") != "frozen_policy_diagnostic" and "frozen_policy" not in report,
                "frozen-policy diagnostic does not represent an independently trained arm")
        require(report["schema_version"] >= 3, "evaluation lacks controller samples")
        cfg_path = root / row["config"]
        cfg = read(cfg_path)
        checkpoint = Path(report["checkpoint"]).resolve()
        require(cfg_path.resolve() == checkpoint.parent / "run_config.json", "checkpoint and snapshot differ")
        require(checkpoint.name == f"model_{row['iterations'] - 1}.pt", "completed iteration differs")
        if row.get("checkpoint"):
            require(Path(row["checkpoint"]).resolve() == checkpoint, "manifest checkpoint differs")
        r.checkpoint_hash = report["checkpoint_sha256"]
        require(sha256(checkpoint) == r.checkpoint_hash and
                row.get("checkpoint_sha256", r.checkpoint_hash) == r.checkpoint_hash, "checkpoint SHA mismatch")
        require(cfg["train"]["seed"] == row["seed"] and not cfg.get("warm_start_checkpoint"),
                "seed mismatch or warm start")
        control = cfg["env"]["control"]
        require(cfg["task"] == "g1_sprint", "unexpected training task")
        require(cfg["env"]["env"]["num_envs"] == 4096 and cfg["env"]["sim"]["nthread"] == 10 and
                cfg["train"]["runner"]["num_steps_per_env"] == 96, "training rollout dimensions differ")
        arm = control.get("arm", "A0")
        options = control.get("arm_options", {})
        require(arm == row["arm"] and row["variant"] == variant_name(arm, options), "snapshot arm or variant differs")
        r.options = options
        reference = cfg["env"]["reference"]
        r.phase_modulation = reference.get("phase_modulation", 0.)
        r.phase_obs = reference.get("phase_obs", False)
        r.phase_context = json.dumps((
            {k: v for k, v in control.items() if k not in ("arm", "arm_options", "action_scale")},
            {k: v for k, v in reference.items() if k not in ("checkpoint", "phase_modulation")},
        ), sort_keys=True)
        r.controller_signature = json.dumps((control, cfg["env"]["reference"].get("phase_obs"),
                                             cfg["env"]["reference"].get("phase_modulation")), sort_keys=True)
        if (checkpoint.parent / "spectral_prior.pt").is_file():
            require(sha256(checkpoint.parent / "spectral_prior.pt") == report["prior_sha256"], "bundled prior SHA mismatch")
        require(effective_latency(control.get("fixed_latency_steps"), control["max_latency_steps"]) ==
                row["train_latency"], "snapshot training delay differs")
        meta = report.get("training")
        if meta:
            require(meta["arm"] == row["arm"] and meta["seed"] == row["seed"] and meta["iterations"] == row["iterations"]
                    and meta.get("arm_options", {}) == options and meta["config_sha256"] == sha256(cfg_path), "evaluation training metadata mismatch")
        protocol(report, row)
        r.signature = training_signature(cfg, report)
        r.alpha = control["action_scale"]
        r.kappa = control.get("increment_kappa")
        r.action_penalty = cfg["env"]["rewards"]["scales"].get("action_rate", 0)
        r.survival = {s["command_speed"]: np.mean([t["complete"] for t in s["trials"]]).item()
                      for s in report["results"]}
        for commands in (*BANDS.values(), (3.,)):
            chosen = [s for s in report["results"] if s["command_speed"] in commands]
            if len(chosen) == len(commands):
                r.bands[commands] = summarize(chosen)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        r.error = str(exc)
    return r


def discover(root):
    runs = []
    for path in sorted(root.rglob("*.json")):
        try:
            report = read(path)
            if not isinstance(report, dict) or not {"results", "speed_bins", "checkpoint"} <= report.keys():
                continue
            if report.get("kind") == "frozen_policy_diagnostic" or "frozen_policy" in report:
                continue
            checkpoint = Path(report["checkpoint"])
            snapshot = checkpoint.parent / "run_config.json"
            cfg = read(snapshot)
            meta = report.get("training") or {}
            ctrl = cfg["env"]["control"]
            arm = meta.get("arm", ctrl.get("arm", "A0"))
            options = meta.get("arm_options") or ctrl.get("arm_options") or {}
            variant = variant_name(arm, options)
            p = report["protocol"]
            train_lag = effective_latency(p["latency_steps_training"], p["latency_max_training"])
            runs.append(dict(arm=arm, variant=variant, seed=cfg["train"]["seed"], status="complete",
                             iterations=int(checkpoint.stem.split("_")[-1]) + 1, impulse=p["push_impulse_Ns"],
                             train_latency=train_lag, eval_latency=train_lag if p["latency_steps"] is None else p["latency_steps"],
                             config=str(snapshot), checkpoint=str(checkpoint), evaluation=str(path)))
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            continue
    return runs
