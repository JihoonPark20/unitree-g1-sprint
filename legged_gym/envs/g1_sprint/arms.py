"""Resolve the controller ablations from their experimental parents."""

import math

B_ARMS = ("B-obs", "B-ff", "B")
PHASE_PARENTS = ("A0", "A1", "F", *B_ARMS, "Hyb")
PARENTS = {
    "A0": None, "A1": "A0", "A1-R": "A1", "F": "A1",
    "B-obs": "F", "B-ff": "F", "B": "F", "Hyb": "B",
    "D-ctrl": "B", "D": "D-ctrl", "C1-env": "B", "C1-mean": "B", "C2": "B",
}
OPTIONS = {
    "cutoff_hz": 10, "beta": 0.2, "kappa": 0.2, "action_rate_weight": 0.0001,
    "base_arm": "B", "hyb_base_arm": "B",
}


def add_arm_arguments(parser):
    parser.add_argument("--arm", choices=PARENTS, help="Select a SPRINT residual controller experiment")
    parser.add_argument("--cutoff-hz", type=int, choices=(5, 10, 20), help="Lowpass cutoff of F or its parent")
    parser.add_argument("--beta", type=float, choices=(0.2, 0.4), help="Direct position fraction for Hyb")
    parser.add_argument("--kappa", type=float, choices=(0.1, 0.2, 0.4), help="C1 increment bound / residual bound")
    parser.add_argument("--action-rate-weight", type=float, help="Positive magnitude of the A1-R action-rate penalty")
    parser.add_argument("--residual-scale", type=float, help="Physical action scale for an RMS-matched retraining")
    parser.add_argument("--base-arm", choices=PHASE_PARENTS, help="Parent controller for Hyb, phase and C arms")
    parser.add_argument("--hyb-base-arm", choices=B_ARMS, help="B parent of a Hyb-based phase comparison")


def _lineage(arm, options):
    parent = options["base_arm"]
    if arm in ("D", "D-ctrl"):
        allowed = PHASE_PARENTS
    else:
        allowed = B_ARMS
    if parent not in allowed:
        raise ValueError(f"{arm} base arm must be one of {allowed}")
    chain = []
    current = arm
    while current is not None:
        chain.append(current)
        if current == "Hyb" and arm != "Hyb":
            current = options["hyb_base_arm"]
        elif current in ("Hyb", "D-ctrl", "C1-env", "C1-mean", "C2"):
            current = parent
        else:
            current = PARENTS[current]
    return chain


def select_arm(env, train, args, *, restoring):
    arm = getattr(args, "arm", None)
    supplied = {key: getattr(args, key) for key in (*OPTIONS, "residual_scale") if getattr(args, key, None) is not None}
    if arm is None:
        if supplied:
            raise ValueError("Arm options require --arm")
        return
    if args.task != "g1_sprint":
        raise ValueError("--arm belongs to the g1_sprint task")
    experiment = f"g1_sprint_{arm}"
    if getattr(args, "experiment_name", None) not in (None, experiment):
        raise ValueError(f"--arm {arm} uses experiment_name {experiment}")
    if getattr(args, "warm_start", None) and (
        arm != "A0" or "residual_scale" in supplied or getattr(args, "latency_steps", None) in (1, 2)
    ):
        raise ValueError("Changed controller arms require fresh training and reject --warm_start")
    control = env.control
    if restoring and control.arm != arm:
        raise ValueError(f"Checkpoint arm {control.arm} cannot resume as {arm}")
    saved_options = control.arm_options if restoring else {}
    defaults = {**OPTIONS, "residual_scale": control.action_scale}
    options = {**defaults, **saved_options, **supplied}
    chain = _lineage(arm, options)
    supported = {"residual_scale"}
    if "F" in chain:
        supported.add("cutoff_hz")
    if "Hyb" in chain:
        supported.add("beta")
    if arm.startswith("C1-"):
        supported.add("kappa")
    if arm == "A1-R":
        supported.add("action_rate_weight")
    if arm in ("Hyb", "D-ctrl", "D", "C1-env", "C1-mean", "C2"):
        supported.add("base_arm")
    if options["base_arm"] == "Hyb" and arm in ("D-ctrl", "D"):
        supported.add("hyb_base_arm")
    unused = supplied.keys() - supported
    if unused:
        raise ValueError(f"Unsupported options for {arm}: {', '.join(sorted(unused))}")
    for key in ("action_rate_weight", "residual_scale"):
        if not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if restoring:
        if any(value != saved_options.get(key, defaults[key]) for key, value in supplied.items()):
            raise ValueError("Cannot change arm parameters while resuming a checkpoint")
    else:
        control.arm = arm
        control.arm_options = supplied
        control.action_scale = options["residual_scale"]
        if "A1" in chain:
            control.target_interp = "exact"
        if "F" in chain:
            control.residual_mode = "lowpass"
            control.residual_leak = math.tau * options["cutoff_hz"]
            control.residual_obs = "B-obs" in chain or "B" in chain
            control.residual_feedforward = float("B-ff" in chain or "B" in chain)
        if "Hyb" in chain:
            control.residual_mix_direct = options["beta"]
        if "D-ctrl" in chain:
            env.reference.phase_obs = True
            env.reference.phase_modulation = 0.1 if arm == "D" else 0.0
        if arm.startswith("C1-"):
            control.residual_mode = "integrator" if arm == "C1-env" else "position"
            control.residual_leak = 0.0
            control.increment_kappa = options["kappa"]
            train.policy.integrate_mean = arm == "C1-mean"
        if arm == "C2":
            control.learned_leak = True
        if arm == "A1-R":
            env.rewards.scales["action_rate"] = -options["action_rate_weight"]
    train.runner.experiment_name = experiment
