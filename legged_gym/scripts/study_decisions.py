"""Preregistered comparisons across independent policy training seeds."""

import math
from collections import defaultdict
from statistics import mean, stdev

from legged_gym.scripts.study_results import BANDS, GRID, read

PASS, FAIL, UNKNOWN = "VERIFIED", "NOT VERIFIED", "INCONCLUSIVE"


def outcome(verdict, reason, evidence=(), **details):
    return dict(verdict=verdict, reason=reason, evidence=sorted(set(evidence)), **details)


def combine(criteria, *, any_of=False):
    values = [c["verdict"] for c in criteria.values()]
    if any_of:
        verdict = PASS if PASS in values else UNKNOWN if UNKNOWN in values else FAIL
    else:
        verdict = UNKNOWN if UNKNOWN in values else FAIL if FAIL in values else PASS
    return outcome(verdict, "any registered endpoint" if any_of else "all registered conditions",
                   (p for c in criteria.values() for p in c["evidence"]), criteria=criteria)


def strict_better(x, y, higher):
    signed = 1 if higher else -1
    shift = signed * (mean(x) - mean(y))
    spread = max(stdev(x), stdev(y))
    return (shift > spread and not math.isclose(shift, spread, rel_tol=1e-12)
            and all(signed * (a - b) > 0 for a in x for b in y))


def compare_values(x, y, paths, *, higher=False, retain=False):
    details = dict(candidate=x, reference=y, mean_difference=mean(x)-mean(y),
                   larger_seed_sd=max(stdev(x), stdev(y)))
    if retain:
        sign = 1 if higher else -1
        if all(sign*(a-b) >= 0 for a, b in zip(x, y)):
            return outcome(PASS, "every matched training seed retained performance", paths, **details)
        if strict_better(y, x, higher):
            return outcome(FAIL, "consistent adverse shift exceeding seed spread", paths, **details)
        return outcome(UNKNOWN, "retention has mixed or adverse paired observations", paths, **details)
    good = strict_better(x, y, higher)
    return outcome(PASS if good else FAIL, "all cross-seed signs and mean shift exceeding both seed SDs"
                   if good else "registered directional difference criterion failed", paths, **details)


def choose_band(survival):
    common = [v for v in GRID if all(seed.get(v, 0) >= .9 for seed in survival)]
    top = max(common, default=None)
    commands = [] if top is None else list(BANDS["4-6"] if top >= 4 else
                                           [v for v in GRID if v <= top and v >= round(top-2, 1)])
    return dict(available=True, highest_speed=top, commands=commands,
                sprint_not_reached=top is None or top < 4)


class Decisions:
    def __init__(self, root, runs, settings):
        self.root, self.runs, self.settings = root, runs, settings
        self.selection = settings.get("selected", {})
        self.index = defaultdict(list)
        self.corrections = []
        for run in runs:
            r = run.row
            if r.get("status") == "complete":
                self.index[r["variant"], r["seed"], r["train_latency"], r["eval_latency"], r["impulse"]].append(run)
        self.n = settings.get("iterations")
        if self.n is None:
            counts = {r.row["iterations"] for r in runs if not r.error}
            self.n = counts.pop() if len(counts) == 1 else None
        self.g0 = self.baseline_band()
        self.commands = self.g0["commands"]

    def selected(self, name):
        if name in self.selection:
            selected = self.selection[name]
            family = {r.row["arm"] for r in self.runs if r.row["variant"] == selected}
            allowed = ({"B", "B-obs", "B-ff"} if name in ("best_B", "H5_B") else
                       {"A0", "A1", "F", "B", "B-obs", "B-ff", "Hyb"} if name == "phase_parent" else {name})
            return selected if family <= allowed else None
        choices = {r.row["variant"] for r in self.runs if r.row["arm"] == name}
        return choices.pop() if len(choices) == 1 else None

    def cohort(self, variant, nseed, lag, impulse):
        found = []
        if not variant:
            return [], "condition is unselected or ambiguous"
        for seed in range(nseed):
            items = self.index[variant, seed, lag, lag, impulse]
            if len(items) != 1:
                return found, f"{variant} seed {seed}, delay {lag}, impulse {impulse}: {len(items)} completed rows"
            r = items[0]
            if r.error or r.row["iterations"] != self.n:
                return found + [r], r.error or "training iteration differs"
            found.append(r)
        if len({r.checkpoint_hash for r in found}) != nseed:
            return found, "one checkpoint used for multiple independent training seeds"
        if len({(r.signature, r.controller_signature, r.phase_context, r.action_penalty) for r in found}) != 1:
            return found, "PPO, prior, rewards or environment differ within condition"
        return found, None

    def baseline_band(self):
        runs, err = self.cohort(self.selected("A0"), 3, 0, 0)
        result = (dict(available=False, highest_speed=None, commands=[], sprint_not_reached=None, reason=err)
                  if err else choose_band([r.survival for r in runs]))
        result["evidence"] = [str(r.path) for r in runs if r.path]
        frozen = self.settings.get("g0")
        if frozen is not None and (frozen.get("commands"), frozen.get("highest_speed")) != (
                result["commands"], result["highest_speed"]):
            return dict(available=False, highest_speed=None, commands=[], sprint_not_reached=None,
                        reason="frozen G0 disagrees with A0 evidence", evidence=result["evidence"])
        return result

    def recorded_phase_correction(self, trial, control):
        for item in self.settings.get("rms_corrections", []):
            if (item.get("variant"), item.get("reference"), item.get("scale")) != (
                    trial.row["variant"], control.row["variant"], trial.alpha):
                continue
            source = item.get("source_variant")
            if not source or source == trial.row["variant"]:
                continue
            original, err = self.cohort(source, 3, 0, 0)
            comparators, ref_err = self.cohort(control.row["variant"], 3, 0, 0)
            if err or ref_err or original[0].row["arm"] != "D" or original[0].phase_context != control.phase_context:
                continue
            if original[0].alpha != control.alpha:
                continue
            bands = [r.band(self.commands) for r in original + comparators]
            if any(not b or not b.get("residual_rms") for b in bands):
                continue
            ratio = mean(b["residual_rms"] for b in bands[:3]) / mean(b["residual_rms"] for b in bands[3:])
            if (ratio <= .5 or ratio >= 2) and math.isclose(original[0].alpha / ratio, trial.alpha, rel_tol=1e-12):
                return math.isclose(item.get("ratio", 0), ratio, rel_tol=1e-12)
        return False

    def lineage_error(self, trial, control):
        arm, ref = trial.row["arm"], control.row["arm"]
        opts, prior = trial.options, control.options
        if arm in ("B", "B-obs", "B-ff", "Hyb", "C1-env", "C1-mean", "C2") and ref in (
                "F", "B", "B-obs", "B-ff"):
            if opts.get("cutoff_hz", 10) != prior.get("cutoff_hz", 10):
                return "filter cutoff differs from selected comparator"
            if arm in ("Hyb", "C1-env", "C1-mean", "C2") and opts.get("base_arm", "B") != ref:
                return "controller was trained on a different B parent"
        if arm == "D" and ref == "D-ctrl":
            if {k:v for k,v in opts.items() if k != "residual_scale"} != {
                    k:v for k,v in prior.items() if k != "residual_scale"}:
                return "phase arms differ in their parent controller"
            if (trial.phase_context != control.phase_context or not trial.phase_obs or not control.phase_obs
                    or trial.phase_modulation != .1 or control.phase_modulation != 0):
                return "resolved D/D-ctrl controller or phase settings differ from the registered contrast"
            if trial.alpha != control.alpha and not self.recorded_phase_correction(trial, control):
                return "D/D-ctrl physical action scales differ without a matching measured RMS correction"
        return None

    def paired(self, candidate, reference, nseed, lag=0, impulse=0., commands=None):
        x, errx = self.cohort(candidate, nseed, lag, impulse)
        y, erry = self.cohort(reference, nseed, lag, impulse)
        paths = [str(r.path) for r in x + y if r.path]
        error = errx or erry
        if not error:
            error = self.lineage_error(x[0], y[0])
            if x[0].signature != y[0].signature:
                error = "PPO, prior, rewards or common environment differ between arms"
            if x[0].row["arm"] != "A1-R" and y[0].row["arm"] != "A1-R" and x[0].action_penalty != y[0].action_penalty:
                error = "unregistered reward-weight difference"
        band = self.commands if commands is None else commands
        pair = [] if error else [(a.band(band), b.band(band)) for a, b in zip(x, y)]
        if not pair or any(not a or not b for a, b in pair):
            error = error or "decision band has no coverage"
        return pair, paths, error

    def metric(self, hypothesis, candidate, reference, key, nseed=3, *, lag=0, impulse=0.,
               commands=None, retain=False, higher=False):
        pair, paths, error = self.paired(candidate, reference, nseed, lag, impulse, commands)
        if error:
            return outcome(UNKNOWN, error, paths)
        rms = [[d[i]["residual_rms"] for d in pair] for i in (0, 1)]
        if any(v is None or not math.isfinite(v) for series in rms for v in series):
            return outcome(UNKNOWN, "residual RMS is missing", paths)
        a, b = map(mean, rms)
        if min(a, b) <= 0 or a >= 2*b or b >= 2*a:
            ratio = a/b if b else None
            scale = self.index[candidate, 0, lag, lag, impulse][0].alpha*b/a if min(a, b) > 0 else None
            correction = dict(hypothesis=hypothesis, variant=candidate, reference=reference, train_latency=lag,
                              scale=scale, ratio=ratio)
            if correction not in self.corrections:
                self.corrections.append(correction)
            return outcome(UNKNOWN, "residual RMS confound or undefined physical scale", paths, correction=correction)
        values = []
        for i in (0, 1):
            series = []
            for entry in pair:
                s = entry[i]
                if key in ("E_hi", "E_vel") and s["survival"] < .9:
                    return outcome(UNKNOWN, "speed-band survival below 0.9", paths)
                coverage = {"E_hi": "spectral_covered_speeds", "E_vel": "E_vel_covered_speeds"}.get(key)
                if coverage and s[coverage] != len(s["commands"]):
                    return outcome(UNKNOWN, f"incomplete {key} command coverage", paths)
                v = s.get(key)
                if v is None or not math.isfinite(v):
                    return outcome(UNKNOWN, f"{key} unavailable", paths)
                series.append(v)
            values.append(series)
        return compare_values(*values, paths, higher=higher, retain=retain)

    def primary(self, label, trial, ref, nseed=3):
        return combine({k: self.metric(label, trial, ref, k, nseed, higher=k == "survival")
                        for k in ("E_hi", "E_vel", "survival")}, any_of=True)

    def calibrated_impulse(self):
        path = self.settings.get("calibration")
        if not path:
            return None
        a0, err = self.cohort(self.selected("A0"), 3, 0, 0)
        if err:
            return None
        try:
            cal = read(self.root / path)
            j = cal["impulse_Ns"]
            if (cal["kind"] == "push_calibration" and cal["speed_m_s"] == 3 and cal["target_survival"] == .5
                    and j is not None and j > 0 and cal["checkpoint"] == a0[0].row.get("checkpoint")
                    and any(p["impulse_Ns"] == j and p["trials"] == 24 and p["pushed_trials"] > 0
                            and abs(p["survival"] - .5) <= 1/24 + 1e-12 for p in cal["probes"])):
                return j
        except (OSError, KeyError, TypeError, ValueError):
            pass
        return None

    def latency(self, candidate, reference):
        clauses = {}
        for lag in (0, 1, 2):
            gate = self.metric("H5", candidate, reference, "survival", lag=lag, higher=True)
            if gate["verdict"] == UNKNOWN:
                clauses[f"delay_{lag}"] = gate
                continue
            if lag == 0:
                base = gate
            elif "delay_0" not in clauses:
                dx = [v-w for v, w in zip(base["candidate"], gate["candidate"])]
                dy = [v-w for v, w in zip(base["reference"], gate["reference"])]
                clauses[f"delay_{lag}"] = compare_values(dx, dy, base["evidence"] + gate["evidence"])
        return combine(clauses)

    def evaluate(self):
        select = self.selected
        a0, a1, f = map(select, ("A0", "A1", "F"))
        b, obs, ff, best, hyb = map(select, ("B", "B-obs", "B-ff", "best_B", "Hyb"))
        h = {}
        spectrum = self.metric("H0", a1, a0, "E_hi")
        if spectrum["verdict"] == PASS:
            old, new = mean(spectrum["reference"]), mean(spectrum["candidate"])
            if old - new < .1 * old and not math.isclose(new, .9*old, rel_tol=1e-12):
                spectrum.update(verdict=FAIL, reason="E_hi reduction below 10%")
        h["H0"] = combine(dict(E_hi=spectrum, survival=self.metric("H0", a1, a0, "survival", retain=True, higher=True),
                               E_vel=self.metric("H0", a1, a0, "E_vel", retain=True)))
        h["H0"]["contrast"] = "4b424f1 substep velocity lookahead vs exact spectral targets"
        h["H1a"] = self.metric("H1a", f, a1, "E_hi")
        h["H1b"] = self.metric("H1b", obs, f, "E_vel", 5)
        h["H1c"] = combine({k: self.metric("H1c", ff, f, k, 5) for k in ("E_vel", "E_hi")}, any_of=True)
        components = combine({"B-obs": self.primary("G2", obs, f, 5), "B-ff": self.primary("G2", ff, f, 5),
                              "B": self.primary("G2", b, f)}, any_of=True)
        g2 = {PASS: True, FAIL: False, UNKNOWN: None}[components["verdict"]]
        impulse = self.calibrated_impulse()
        push = (self.metric("H2", hyb, a1, "push_recovery", commands=[3.], impulse=impulse,
                            higher=True, retain=True) if impulse else outcome(UNKNOWN, "positive calibrated impulse unavailable"))
        h["H2"] = combine(dict(push_recovery=push, E_hi=self.metric("H2", hyb, best, "E_hi", retain=True)))
        d, ctrl = select("D"), select("D-ctrl")
        prior = self.metric("H3", d, ctrl, "prior_reward")
        if prior["verdict"] != UNKNOWN:
            new, floor = mean(prior["candidate"]), .9*mean(prior["reference"])
            prior.update(verdict=PASS if new >= floor or math.isclose(new, floor, rel_tol=1e-12) else FAIL,
                         reason="registered 10% prior-reward floor")
        h["H3"] = combine(dict(E_vel=self.metric("H3", d, ctrl, "E_vel"), prior_reward=prior))
        h["H4"] = self.c_family(best) if g2 is True else outcome(UNKNOWN, "G2 did not pass")
        h["H5"] = self.latency(select("H5_B") or best, a1)
        for label, ref in (("A1-R", a1), ("D-ctrl", select("phase_parent"))):
            if select(label) and ref:
                self.metric("RMS", select(label), ref, "survival")
        return dict(hypotheses=h, g0=self.g0, g2=g2, g2_components=components, corrections=self.corrections)

    def c_family(self, best):
        clauses = {}
        for arm in ("C1-env", "C1-mean", "C2"):
            for k in ((None,) if arm == "C2" else (.1, .2, .4)):
                variants = {r.row["variant"] for r in self.runs if r.row["arm"] == arm and r.kappa == k}
                chosen = self.selected(arm)
                v = chosen if chosen in variants else next(iter(variants)) if len(variants) == 1 else None
                clauses[f"{arm}/{k}"] = self.primary("H4", v, best)
        return combine(clauses, any_of=True)
