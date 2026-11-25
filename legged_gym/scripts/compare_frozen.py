"""Compare the 4 m/s frozen-policy screen within one checkpoint."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from legged_gym.scripts.control_metrics import PITCH_JOINTS, summarize_trials

HELDOUT = tuple(range(141, 147))
MODES = ("position", "ramp", "lowpass", "lowpass-hold")
CONTROLLER_TREATMENT = {
    "residual_mode", "residual_leak", "frozen_residual_ramp", "frozen_lowpass_hold",
    "effective_residual_mode", "cutoff_hz", "residual_update_hz", "residual_feedforward", "frozen_limits_only",
}


def _read(report):
    return json.loads(Path(report).read_text()) if isinstance(report, (str, Path)) else report


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_controller(frozen, protocol):
    ctrl = frozen["controller"]
    mode = frozen["requested_mode"]
    _require(mode in MODES and ctrl["effective_residual_mode"] == mode, "requested and effective residual modes differ")
    _require(not ctrl["learned_leak"] and ctrl["residual_mix_direct"] == 0 and ctrl["increment_kappa"] is None,
             "unexpected learned, mixed or increment controller")
    hold, ramp = bool(ctrl.get("frozen_lowpass_hold")), bool(ctrl.get("frozen_residual_ramp"))
    _require((hold, ramp) == (mode == "lowpass-hold", mode == "ramp"), "controller flags differ from the requested mode")
    _require(ctrl["residual_feedforward"] in (0, 1) and (mode not in ("ramp", "lowpass-hold") or
             ctrl["residual_feedforward"] == 0), "unsupported velocity feedforward for this controller")
    physical = {"ramp":"integrator", "lowpass-hold":"lowpass"}.get(mode, mode)
    _require(ctrl["residual_mode"] == physical, "wrong physical residual controller")
    dt = protocol["control_dt"] if mode in ("position", "lowpass-hold") else protocol["physics_dt"]
    _require(math.isclose(ctrl["residual_update_hz"], 1/dt), "wrong controller update frequency")
    if mode in ("lowpass", "lowpass-hold"):
        cutoff = ctrl["cutoff_hz"]
        _require(cutoff is not None and cutoff > 0 and math.isclose(ctrl["residual_leak"], math.tau*cutoff),
                 "cutoff and physical residual leak differ")
    else:
        _require(ctrl["cutoff_hz"] is None and (not ramp or ctrl["residual_leak"] == 0), "unexpected filter on direct/ramp")
    projection = frozen.get("projection", "directional")
    _require(projection in ("directional", "limits-only") and
             bool(ctrl.get("frozen_limits_only", False)) == (projection == "limits-only"),
             "requested projection disagrees with configured controller")
    allowed = CONTROLLER_TREATMENT | {"target_interp", "effective_target", "residual_feedforward", "frozen_limits_only"}
    source = frozen["source_controller"]
    _require({k: v for k, v in ctrl.items() if k not in allowed} ==
             {k: v for k, v in source.items() if k not in allowed}, "controller changes unrelated checkpoint settings")
    if frozen["requested_target"] != "inherited":
        _require(ctrl["target_interp"] == ctrl["effective_target"] == frozen["requested_target"],
                 "actual and requested target interpolation differ")


def _block(trials, protocol):
    stats = summarize_trials(trials)
    nwin = stats["welch_windows"]
    spectra = [t["controller"]["spectrum"] for t in trials if t["controller"]["spectrum"]]
    dt = protocol["physics_dt"]
    # scipy Welch reports a density. The evaluator stores sums of bin densities multiplied by window count.
    df = 1 / (round(1/dt) * dt)
    power = {}
    for key in ("high", "total"):
        value = np.sum([s[key] for s in spectra], axis=0) * df / nwin if nwin else None
        power[key] = float(np.mean(value)) if value is not None else None

    def observed(key):
        values = [t[key] for t in trials if t["samples"] > 0]
        return mean(values) if values else None

    return dict(trials=len(trials), survivors=sum(t["complete"] for t in trials), survival=stats["survival"],
                samples=stats["samples"], coverage=stats["samples"] / (len(trials)*round(
                    protocol["measurement_seconds"] / protocol["control_dt"])),
                E_vel=observed("E_vel_observed"), mean_forward_speed=observed("mean_forward_speed"),
                E_hi=stats["E_hi"], hf_power_Nm2=power["high"], total_power_Nm2=power["total"],
                **{k:stats[k] for k in ("welch_windows", "residual_rms", "action_clip_fraction",
                                         "anti_windup_fraction", "joint_jerk", "torque_saturation")})


def summary(report):
    """Summarize all trial prefixes in a single-command frozen-policy report."""
    report = _read(report)
    _require(report.get("kind") == "frozen_policy_diagnostic", "requires a frozen-policy diagnostic")
    frozen, protocol = report["frozen_policy"], report["protocol"]
    _require(report["checkpoint_sha256"] == frozen["checkpoint_sha256"], "conflicting checkpoint identities")
    _require(tuple(protocol["spectral_joints"]) == PITCH_JOINTS, "unexpected torque spectrum joint order")
    _require([r["command_speed"] for r in report["results"]] == [4.], "expected a single 4.0 m/s command")
    _validate_controller(frozen, protocol)
    seeds = protocol["seeds"]
    _require(seeds and len(set(seeds)) == len(seeds), "duplicate or missing reset seeds")
    trials = report["results"][0]["trials"]
    expected = {(s, i) for s in seeds for i in range(protocol["trials_per_seed_speed"])}
    trial_ids = [(t["seed"], t["env_index"]) for t in trials]
    _require(len(trial_ids) == len(expected) and set(trial_ids) == expected, "duplicate, absent or unexpected trial IDs")
    steps = round(protocol["measurement_seconds"] / protocol["control_dt"])
    _require(steps > 0, "measurement must contain policy steps")
    for t in trials:
        n = t["samples"]
        _require(n == t["controller"]["samples"] and 0 <= n <= steps and
                 t["complete"] == (t["failure_time_s"] is None) and (not t["complete"] or n == steps),
                 "inconsistent trial coverage")
        if n:
            _require(all(t[k] is not None and math.isfinite(t[k]) for k in ("E_vel_observed", "mean_forward_speed")),
                     "observed trial lacks finite tracking data")
        spectrum = t["controller"]["spectrum"]
        if spectrum is not None:
            hi, total = (np.asarray(spectrum[k], dtype=float) for k in ("high", "total"))
            _require(hi.shape == total.shape == (len(PITCH_JOINTS),) and spectrum["windows"] > 0 and
                     np.isfinite(total).all() and np.isfinite(hi).all() and (hi>=0).all() and
                     (hi <= total + 1e-12 * np.maximum(total, 1)).all(), "invalid Welch power")
    return dict(checkpoint_sha256=report["checkpoint_sha256"], prior_sha256=report["prior_sha256"],
                mode=frozen["requested_mode"], target=frozen["requested_target"], cutoff_hz=frozen["controller"]["cutoff_hz"],
                start=frozen.get("start", "reset"), projection=frozen.get("projection", "directional"),
                feedforward=frozen["controller"]["residual_feedforward"],
                projection_applies_to_residual_state=frozen["requested_mode"] != "position",
                seeds=sorted(seeds), pooled=_block(trials, protocol), per_seed={
                    str(seed): _block([t for t in trials if t["seed"] == seed], protocol) for seed in sorted(seeds)})


def _fingerprints(report):
    frozen, p = report["frozen_policy"], report["protocol"]
    blocks = {}
    for row in frozen.get("switches", []):
        seed, ids = row["seed"], row["active_env_indices"]
        _require(seed not in blocks and row["command_speed"] == 4.0, "unexpected or duplicate switch block")
        _require(len(ids) == len(set(ids)) == row["active_count"] and
                 all(0 <= i < p["trials_per_seed_speed"] for i in ids), "invalid switch environment indices")
        reached = bool(row["reached"])
        _require(reached == bool(ids), "switch coverage and reached flag disagree")
        hashes = tuple(row.get(k) for k in ("qpos_sha256", "qvel_sha256", "residual_sha256", "actor_hidden_sha256"))
        if reached:
            for h in hashes[:3] + ((hashes[3],) if hashes[3] is not None else ()):
                _require(isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h),
                         "incomplete physical or recurrent warmup fingerprint")
        else:
            _require(all(h is None for h in hashes), "unreached switch has a fingerprint")
        blocks[seed] = (reached, sorted(ids), hashes)
    _require(set(blocks) == set(p["seeds"]), "missing switch record for a reset seed")
    return blocks


def compare(baseline, candidate):
    """Assess the prespecified retention/smoothing rules for one matched pair of controllers."""
    baseline, candidate = _read(baseline), _read(candidate)
    a, b = summary(baseline), summary(candidate)
    _require(a["seeds"] == b["seeds"], "baseline and candidate use different reset seeds")
    _require(a["start"] == b["start"] and a["start"] in ("measurement", "reset"), "different or invalid intervention start")
    if a["start"] == "measurement":
        _require(_fingerprints(baseline) == _fingerprints(candidate), "mismatched warmup fingerprints")
    for key in ("checkpoint_sha256", "prior_sha256", "control_contract", "training", "versions"):
        _require(baseline[key] == candidate[key], f"mismatched {key}")
    p, q = baseline["protocol"], candidate["protocol"]
    for k in ("physics_dt", "control_dt", "warmup_seconds", "measurement_seconds", "trials_per_seed_speed",
              "rl_device", "backend", "seeds", "commands", "initial_state", "randomization", "sampling", "welch",
              "spectral_joints", "control_joint_names", "latency_steps", "latency_steps_training", "latency_max_training",
              "push_impulse_Ns"):
        _require(p[k] == q[k], f"mismatched evaluation protocol field {k}")
    _require((p["physics_dt"],p["control_dt"],p["warmup_seconds"],p["measurement_seconds"],
              p["trials_per_seed_speed"],p["push_impulse_Ns"]) == (.005,.02,8.,20.,8,0.), "wrong 4 m/s screen protocol")
    lag = p["latency_steps"] if p["latency_steps"] is not None else p["latency_steps_training"]
    _require(lag == 0 or (lag is None and p["latency_max_training"] == 0), "4 m/s screen requires zero latency")
    fa, fb = baseline["frozen_policy"], candidate["frozen_policy"]
    _require(fa["source_controller"] == fb["source_controller"], "original checkpoint controllers differ")
    _require(a["target"] == b["target"] == "linear", "both screen controllers require the explicit linear target")
    ca, cb = fa["controller"], fb["controller"]
    _require({k:v for k,v in ca.items() if k not in CONTROLLER_TREATMENT} ==
             {k:v for k,v in cb.items() if k not in CONTROLLER_TREATMENT}, "controller differs beyond the residual treatment")
    projection_changed = a["projection"] != b["projection"]
    ff_changed = a["feedforward"] != b["feedforward"]
    if a["mode"] == "position" and b["mode"] != "position":
        label = "residual controller vs direct position"
        factors = ["residual controller"]
        if ff_changed:
            factors.append("residual velocity feedforward")
        if b["projection"] == "limits-only":
            factors.append("limits-only residual projection")
        if len(factors) > 1:
            label += " (combined: " + ", ".join(factors) + ")"
    elif a["mode"] == "lowpass-hold" and b["mode"] == "lowpass":
        _require(a["cutoff_hz"] == b["cutoff_hz"] and not projection_changed and
                 a["feedforward"] == b["feedforward"] == 0,
                 "substep-rate contrast requires equal-cutoff filters, identical projection and zero feedforward")
        factors, label = ["update cadence"], "substep vs matched held filter"
    elif a["mode"] == b["mode"] and a["cutoff_hz"] == b["cutoff_hz"]:
        active_projection_change = projection_changed and a["projection_applies_to_residual_state"]
        _require(ff_changed != active_projection_change,
                 "at fixed mode/cutoff the comparison must isolate feedforward or physical projection")
        factors = ["residual velocity feedforward" if ff_changed else "residual projection"]
        label = factors[0] + " at fixed residual dynamics"
    else:
        raise ValueError("expected a direct control, an equal-cutoff cadence comparison, or one isolated controller change")
    for result in (a,b):
        if result["cutoff_hz"] is not None:
            _require(result["cutoff_hz"] in (10.,20.), "cutoff is outside the registered screen")
    x,y = a["pooled"], b["pooled"]
    rms_ratio = (y["residual_rms"] / x["residual_rms"] if y["residual_rms"] is not None and
                 x["residual_rms"] is not None and x["residual_rms"] > 0 else None)
    spectral = all(d["welch_windows"] and d["E_hi"] is not None for result in (a,b) for d in result["per_seed"].values())
    criteria = {
        "both_survive_44_of_48": (x["survivors"] >= 44 and y["survivors"] >= 44) if x["trials"] == 48 else None,
        "at_most_one_extra_fall": y["survivors"] >= x["survivors"] - 1,
        "both_cover_90_percent": min(x["coverage"],y["coverage"]) >= .90,
        "tracking_within_0.05_m_s": (y["E_vel"] <= x["E_vel"] + .05 if None not in (x["E_vel"], y["E_vel"]) else None),
        "ten_percent_lower_E_hi": (y["E_hi"] <= .90 * x["E_hi"] if spectral and x["E_hi"] > 0 else None),
        "lower_E_hi_every_block": all(b["per_seed"][s]["E_hi"] < d["E_hi"] for s,d in a["per_seed"].items()) if spectral else None,
        "rms_ratio_in_open_interval": .5 < rms_ratio < 2. if rms_ratio is not None else None,
        "welch_in_every_block": spectral,
    }
    confirm = (tuple(a["seeds"]) == HELDOUT and b["mode"] in ("lowpass","ramp")
               and factors in (["residual controller"], ["update cadence"]) and a["start"] == "reset"
               and a["projection"] == b["projection"] == "directional" and a["feedforward"] == b["feedforward"] == 0)
    verdict = "INCONCLUSIVE"
    if confirm:
        measured = [v for k, v in criteria.items() if k != "welch_in_every_block"]
        if any(v is False for v in measured):
            verdict = "NOT VERIFIED"
        elif all(v is True for v in measured) and spectral:
            verdict = "VERIFIED"
    return dict(verdict=verdict, confirmation_eligible=confirm, criteria=criteria, rms_ratio=rms_ratio,
                comparison=label, comparison_factors=factors,
                independent_training_policies=1, baseline=a, candidate=b)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, help="Save numerical endpoints, criteria and reset-seed blocks as JSON")
    args = parser.parse_args(argv)
    try:
        result = compare(args.baseline, args.candidate)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    result.update(baseline_report=str(args.baseline.resolve()), candidate_report=str(args.candidate.resolve()))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['verdict']} · {result['comparison']} · one fixed training policy")
    for key, passed in result["criteria"].items():
        print(f"  {key:<28} { {True:'pass', False:'fail', None:'unavailable'}[passed] }")
    print("Endpoint                      Baseline          Candidate")
    def fmt(x):
        return "unavailable" if x is None else f"{x:.6g}"

    for k, value in result["baseline"]["pooled"].items():
        other = result["candidate"]["pooled"][k]
        print(f"{k:<28} {fmt(value):>14} {fmt(other):>17}")


if __name__ == "__main__":
    main()
