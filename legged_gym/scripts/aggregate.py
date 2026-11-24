"""Summarize saved speed sweeps and adjudicate the G1 residual-controller experiments."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from legged_gym.scripts.study_decisions import Decisions
from legged_gym.scripts.study_results import ARMS, BANDS, PITCH_JOINTS, discover, load_run, read

METRICS = ("E_hi", "E_vel", "survival", "joint_jerk", "E_qpos", "torque_saturation", "residual_rms",
           "action_rate", "action_acc", "action_clip_fraction", "anti_windup_fraction", "prior_reward",
           "push_recovery", "jump_samples", "pushed_trials", "welch_windows")
PRIMARY = ("E_hi", "E_vel", "survival", "residual_rms", "joint_jerk", "push_recovery")


def tables(runs, commands):
    groups = defaultdict(list)
    for r in runs:
        row = r.row
        groups[row["arm"], row["variant"], row["train_latency"], row["eval_latency"], row["impulse"]].append(r)
    for arm in ARMS:
        if not any(k[0] == arm for k in groups):
            groups[arm, arm, 0, 0, 0] = []
    per_seed, table = [], []
    for (arm, variant, train, test, impulse), rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        bands = dict(BANDS)
        if commands and tuple(commands) not in BANDS.values():
            bands["G0"] = tuple(commands)
        if impulse:
            bands["push_3"] = (3.,)
        for label, speeds in bands.items():
            identifier = dict(arm=arm, variant=variant, train_delay=train, eval_delay=test, impulse_Ns=impulse, band=label)
            values = defaultdict(list)
            reasons = set()
            seen = set()
            valid = []
            counts = defaultdict(int)
            for r in rows:
                if r.row.get("status") == "complete":
                    counts[r.row["seed"]] += 1
            for r in rows:
                seed = r.row["seed"]
                error = r.error
                if counts[seed] > 1:
                    error = "duplicate completed seed"
                s = None if error else r.band(speeds)
                if not s:
                    reasons.add(f"seed {seed}: {error or 'no commands in band'}")
                    continue
                if seed in seen:
                    continue
                seen.add(seed)
                valid.append(r)
                per_seed.append(dict(**identifier, seed=seed, iterations=r.row["iterations"],
                                     evaluation=str(r.path), summary=s))
                for key in METRICS:
                    if s[key] is not None:
                        values[key].append(s[key])
            summary = dict(**identifier, seeds=",".join(str(seed) for seed in sorted(seen)), n=len(seen),
                           coverage="; ".join(sorted(reasons)) or ("measured" if valid else "no runs"),
                           evidence=" | ".join(str(r.path) for r in valid))
            for metric in METRICS:
                v = values[metric]
                summary.update({metric+"_n": len(v), metric+"_mean": mean(v) if v else None,
                                metric+"_sd": stdev(v) if len(v) > 1 else None})
            table.append(summary)
    return table, per_seed


def pretty(row, metric):
    value, sd = row[metric+"_mean"], row[metric+"_sd"]
    if value is None:
        return "—"
    return f"{value:.5g} ± {sd:.3g}" if sd is not None else f"{value:.5g} (n=1)"


def aggregate(root):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    rows = read(manifest) if manifest.exists() else discover(root)
    settings = read(root / "decisions.json") if (root / "decisions.json").exists() else {}
    runs = [load_run(root, r) for r in (rows["runs"] if isinstance(rows, dict) else rows)]
    study = Decisions(root, runs, settings)
    verdicts = study.evaluate()
    table, per_seed = tables(runs, study.commands)
    verdicts.update(independent_unit="training seed", iterations=study.n,
                    spectral_joint_order=PITCH_JOINTS, coverage=[dict(arm=r.row["arm"], variant=r.row["variant"],
                    seed=r.row["seed"], status=r.row.get("status"), evaluation=str(r.path) if r.path else None,
                    error=r.error) for r in runs], skipped=settings.get("skipped", {}))
    (root / "verdicts.json").write_text(json.dumps(verdicts, indent=2, allow_nan=False) + "\n")
    (root / "seed_metrics.json").write_text(json.dumps(per_seed, allow_nan=False) + "\n")
    with (root / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    lines = ["# Controller experiment results", "",
             "Values are means and sample standard deviations across independent training seeds. "
             "Each CSV metric has its own available seed count. Missing estimates are shown as a dash.", "",
             "| Arm / variant | Train / eval delay | Push (N s) | Band (m/s) | Seeds | " +
             " | ".join(PRIMARY) + " | Coverage |", "|" + "---|"*(6+len(PRIMARY))]
    for t in table:
        cells = [f"{t['arm']} / {t['variant']}", f"{t['train_delay']} / {t['eval_delay']}", str(t["impulse_Ns"]),
                 t["band"], str(t["n"]), *(pretty(t, k) for k in PRIMARY), t["coverage"].replace("|", "/")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Hypotheses", "", "| Hypothesis | Verdict | Criterion |", "|---|---|---|"]
    for h, result in verdicts["hypotheses"].items():
        lines.append(f"| {h} | {result['verdict']} | {result['reason']} |")
    lines += ["", "Per-seed metrics, spectral joint powers as ratios and per-joint RMS P jumps are in "
              "`seed_metrics.json`. The order of the P jumps follows each evaluation's `protocol.control_joint_names`.",
              "The conditions, missing evidence, RMS corrections and paths for each decision are in `verdicts.json`."]
    (root / "results.md").write_text("\n".join(lines) + "\n")
    return verdicts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "sprint-ode-runs")
    args = parser.parse_args(argv)
    report = aggregate(args.root)
    print(" ".join(f"{h}={v['verdict']}" for h, v in report["hypotheses"].items()))


if __name__ == "__main__":
    main()
