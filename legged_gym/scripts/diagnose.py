"""Training diagnostics: zero-residual probes and cross-run TensorBoard comparison.

probe   -- no training. Rolls out residual = 0 (the untrained policy's mean) per task.
compare -- reads logs/<task>/<run>/ and reports sample cost to reach milestones.
speed   -- no training. Prior frequency and speed label used per commanded speed (speed calibration).
"""

import argparse
import json
from pathlib import Path

import numpy as np

# (tag, threshold): first sample count where the 10-iteration rolling mean reaches it.
MILESTONES = (("Episode/length_s", 15.0), ("Curriculum/max_command_x", 2.0), ("Curriculum/max_command_x", 4.0))
FINAL_TAGS = (
    "Episode/length_s", "Curriculum/max_command_x", "Tracking/forward_mae_m_s", "Tracking/prior_mae_rad",
    "Gait/single_support_fraction", "Posture/roll_abs_rad",
)


def first_reach(steps, values, threshold, window=10):
    """First step at which the trailing `window`-point mean is >= threshold, else None."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return None
    rolling = np.convolve(values, np.ones(window) / window, mode="valid")
    hit = np.flatnonzero(rolling >= threshold)
    return None if not hit.size else steps[hit[0] + window - 1]


def probe(args):
    import torch

    from legged_gym.envs import task_registry

    for task in args.tasks:
        for speed in args.speeds:
            cfg, _ = task_registry.get_cfgs(task)
            cfg.env.num_envs, cfg.sim.nthread = args.num_envs, args.num_threads
            cfg.commands.curriculum, cfg.commands.standing_fraction = False, 0.0
            cfg.commands.ranges = [[speed, speed], [0.0, 0.0], [0.0, 0.0]]
            cfg.commands.acceleration_limit = 1e3  # Probe the prior at this speed, not the ramp.
            env = task_registry.get_task_class(task)(cfg=cfg)
            steps = round(args.seconds / env.dt)
            alive, life, single, legacy, aligned = np.ones(env.num_envs, bool), np.zeros(env.num_envs), [], [], []
            ids, sigma = env.reference.indices, cfg.rewards.prior_sigma
            zero = torch.zeros(env.num_envs, env.num_actions)
            for _ in range(steps):
                q_ref = env.q_ref.copy()  # Reference that drives this transition.
                _, _, done, _ = env.step(zero)
                live = alive & ~done.numpy()  # Pre-reset state only; reset rows were overwritten.
                if live.any():
                    q = env.dof_pos[live][:, ids]
                    legacy.append(np.exp(-np.square(q - q_ref[live][:, ids]) / sigma).mean())
                    aligned.append(np.exp(-np.square(q - env.q_ref_aligned[live][:, ids]) / sigma).mean())
                    single.append((env._feet_contact[live].sum(1) == 1).mean())
                life[alive] += env.dt
                alive = live
            env.close()
            mean = lambda x: float(np.mean(x)) if x else float("nan")  # noqa: E731
            print(json.dumps({
                "task": task, "speed": speed, "mean_life_s": round(float(life.mean()), 3),
                "survived_fraction": round(float((life >= steps * env.dt - 1e-9).mean()), 3),
                "single_support": round(mean(single), 3),
                "prior_kernel_legacy": round(mean(legacy), 3), "prior_kernel_aligned": round(mean(aligned), 3),
            }), flush=True)


def speed(args):
    import torch

    from legged_gym.envs import task_registry

    for task in args.tasks:
        cfg, _ = task_registry.get_cfgs(task)
        cfg.env.num_envs, cfg.sim.nthread = 1, 1
        env = task_registry.get_task_class(task)(cfg=cfg)
        reference = env.reference
        for v in args.speeds:
            label = v if reference.speed_calibration is None else float(np.interp(v, *reference.speed_calibration))
            print(json.dumps({
                "task": task, "command": v, "prior_speed_label": round(label, 3),
                "frequency_hz": round(float(reference.frequency(torch.tensor([v], dtype=torch.float32))[0]), 3),
            }), flush=True)
        env.close()


def load_run(path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    config = json.loads((path / "run_config.json").read_text())
    per_iteration = config["train"]["runner"]["num_steps_per_env"] * config["env"]["env"]["num_envs"]
    events = EventAccumulator(str(path), size_guidance={"scalars": 0})
    events.Reload()
    tags = set(events.Tags()["scalars"])
    if not tags:
        return None
    start = min(e.wall_time for tag in tags for e in events.Scalars(tag))
    run = {}
    for tag in set(FINAL_TAGS) | {tag for tag, _ in MILESTONES}:
        if tag in tags:
            scalars = events.Scalars(tag)
            run[tag] = (
                np.array([(e.step + 1) * per_iteration for e in scalars]),
                np.array([e.value for e in scalars]),
                np.array([(e.wall_time - start) / 3600 for e in scalars]),
            )
    return run


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    return f"{np.mean(values):.3g}±{np.std(values):.2g} (n={len(values)})"


def compare(args):
    root = Path(args.logdir)
    tasks = args.tasks or sorted(p.name for p in root.iterdir() if p.is_dir())
    report = {}
    for task in tasks:
        runs = [load_run(p) for p in sorted((root / task).glob("*")) if (p / "run_config.json").is_file()]
        runs = [r for r in runs if r]
        if not runs:
            continue
        row = {}
        for tag, threshold in MILESTONES:
            reach = [(first_reach(*r[tag][:2], threshold), first_reach(r[tag][2], r[tag][1], threshold))
                     for r in runs if tag in r]
            row[f"{tag}>={threshold} [M steps]"] = summarize([None if s is None else s / 1e6 for s, _ in reach])
            row[f"{tag}>={threshold} [h]"] = summarize([h for _, h in reach])
            row[f"{tag}>={threshold} reached"] = f"{sum(s is not None for s, _ in reach)}/{len(reach)}"
        for tag in FINAL_TAGS:
            row[f"final {tag}"] = summarize([float(r[tag][1][-args.final_window:].mean()) for r in runs if tag in r])
        report[task] = row
    if args.json:
        print(json.dumps(report, indent=2))
        return
    for task, row in report.items():
        print(f"\n== {task}")
        for key, value in row.items():
            print(f"  {key:52s} {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe", help="Zero-residual rollout per task (no training)")
    p.add_argument("--tasks", nargs="+", default=["g1_sprint_legacy", "g1_sprint_paper_map", "g1_sprint"])
    p.add_argument("--speeds", nargs="+", type=float, default=[0.0, 0.6, 1.5, 3.0])
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--seconds", type=float, default=5.0)
    p.set_defaults(func=probe)
    s = sub.add_parser("speed", help="Prior speed label and frequency per command (no training)")
    s.add_argument("--tasks", nargs="+", default=["g1_sprint_paper_map", "g1_sprint"])
    s.add_argument("--speeds", nargs="+", type=float, default=[0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    s.set_defaults(func=speed)
    c = sub.add_parser("compare", help="Milestone sample cost per task across seeds")
    c.add_argument("--logdir", default="logs")
    c.add_argument("--tasks", nargs="+")
    c.add_argument("--final_window", type=int, default=50, help="Iterations averaged for final values")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=compare)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
