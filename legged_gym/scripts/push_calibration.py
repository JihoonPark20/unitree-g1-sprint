"""Bracket the A0 3 m/s end-of-trial survival target on identical initial conditions."""

from copy import copy
from pathlib import Path

from spectral_prior.metrics import write_report


def calibrate_push(env, runner, args, evaluate):
    output = Path(args.output)
    sweeps = output.parent / (output.stem + "_probes")
    probes = []

    def probe(impulse):
        options = copy(args)
        options.push_impulse = impulse
        options.output = sweeps / f"impulse_{impulse:g}.json"
        report = evaluate(env, runner, options)
        row = report["results"][0]
        probes.append({"impulse_Ns": impulse, "survival": row["survival_fraction"],
                       "trials": len(row["trials"]), "pushed_trials": sum(t["pushed"] for t in row["trials"]),
                       "report": str(options.output.resolve())})
        return row["survival_fraction"]

    zero = probe(0.)
    resolution = 1 / probes[0]["trials"]
    chosen = None
    reason = "unpushed survival is at or below the 50% target"
    if zero > 0.5:
        low, high = 0., 1.
        while high <= 256:
            if probe(high) <= 0.5:
                for _ in range(8):
                    best = min(probes[1:], key=lambda p: abs(p["survival"] - 0.5))
                    if abs(best["survival"] - 0.5) <= resolution / 2 + 1e-12:
                        break
                    middle = (low + high) / 2
                    if probe(middle) > 0.5:
                        low = middle
                    else:
                        high = middle
                best = min(probes[1:], key=lambda p: abs(p["survival"] - 0.5))
                if abs(best["survival"] - 0.5) <= resolution + 1e-12:
                    chosen, reason = best["impulse_Ns"], "observed survival within one trial of 50%"
                else:
                    reason = "bracketed, but no observed survival within one trial of 50%"
                break
            low, high = high, high * 2
        else:
            reason = "no crossing of 50% through 256 N s"
    summary = {"kind": "push_calibration", "speed_m_s": 3.0, "checkpoint": str(args.checkpoint_path),
               "target_survival": 0.5, "impulse_Ns": chosen, "reason": reason, "probes": probes}
    write_report(summary, output)
    print(f"Push calibration: {reason}. Saved {output}")
    return summary
