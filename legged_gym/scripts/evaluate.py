"""Paper speed-grid evaluation with explicit trial coverage and aligned samples."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from legged_gym.scripts.control_metrics import PITCH_JOINTS, ControlMetrics, speed_summary, summarize_bins
from legged_gym.utils.failure_trace import FailureTrace

from spectral_prior import JOINT_NAMES
from spectral_prior.data import sha256
from spectral_prior.metrics import frechet_distance, write_report

PAPER_SPEED_GRID = [i / 10 for i in range(61)]


def aggregate_results(results):
    """Equally weight commands; incomplete sweeps never become full metrics."""
    complete = all(r["complete"] for r in results)
    output = {
        "complete": complete,
        "covered_speeds": sum(r["samples"] > 0 for r in results),
        "requested_speeds": len(results),
    }
    for metric in ("E_qpos", "E_vel"):
        observed = [r[metric + "_observed"] for r in results if r[metric + "_observed"] is not None]
        mean = float(np.mean(observed)) if observed else None
        output[metric] = mean if complete else None
        output[metric + "_observed_only"] = mean
    return output


def evaluate_policy(env, runner, args):
    failure_window = getattr(args, "failure_window", 0.0)
    if not np.isfinite(failure_window) or failure_window < 0:
        raise ValueError("failure_window must be finite and nonnegative")
    if args.seconds < env.dt or args.warmup_seconds < 0:
        raise ValueError("Measurement must last at least one control tick and warmup must be nonnegative")
    warmup = int(np.ceil(args.warmup_seconds / env.dt))
    steps = int(np.ceil(args.seconds / env.dt))
    total = warmup + steps
    # Also protect callers that construct an environment without make_playback.
    env.max_episode_length = total + 1
    env.cfg.commands.resampling_time = (total + 1) * env.dt
    env.cfg.commands.curriculum = False
    env.cfg.commands.standing_fraction = 0
    env.evaluation_latency_steps = getattr(args, "latency_steps", None)
    impulse = getattr(args, "push_impulse", 0.)
    env.record_evaluation = True
    env.record_failure_trace = failure_window > 0
    policy = runner.get_inference_policy(device=runner.device)
    results, pooled_q, pooled_ref = [], [], []
    output = Path(args.output)
    trace_dir = output.with_suffix("")
    if args.save_trajectories or failure_window:
        trace_dir.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            for speed in sorted(args.speeds):
                trials, speed_q, speed_ref = [], [], []
                for seed in args.seeds:
                    # Common initial-condition random stream across speeds.
                    env.rng = np.random.default_rng(seed)
                    torch.manual_seed(seed)
                    env.command_ranges[:] = [[speed, speed], [0, 0], [0, 0]]
                    obs = env.reset()
                    runner.alg.actor.reset()
                    runner.alg.critic.reset()
                    active = np.ones(env.num_envs, dtype=bool)
                    counts = np.zeros(env.num_envs, dtype=int)
                    sums_q = np.zeros(env.num_envs)
                    sums_v = np.zeros(env.num_envs)
                    sums_speed = np.zeros(env.num_envs)
                    reached = np.zeros(env.num_envs, dtype=bool)
                    failure_times = [None] * env.num_envs
                    failure_trace = FailureTrace(env.num_envs, env.dt, failure_window) if failure_window else None
                    metrics = ControlMetrics(env)
                    pushed = np.zeros(env.num_envs, dtype=bool)
                    physical_rows = {key: [] for key in ("applied_torque", "requested_torque", "joint_velocity")}
                    jump_rows = {key: [] for key in ("p_jump_reference", "p_jump_residual")}
                    jump_masks = []
                    q_rows, ref_rows, trace_rows = [], [], []
                    for step in range(total):
                        if step == warmup and impulse:
                            pushed[:] = active
                            direction = np.where((np.arange(env.num_envs) + seed) % 2, -1., 1.)
                            env.evaluation_impulse = impulse * direction * pushed
                        obs, _, dones, extras = env.step(policy(obs.to(runner.device)))
                        sample = extras["evaluation"]  # t+dt, before automatic reset
                        done = dones.cpu().numpy().astype(bool)
                        if failure_trace is not None:
                            failure_trace.record(sample["failure_trace"], active, done)
                        if step >= warmup:
                            metrics.add(sample, active)
                            ids = np.flatnonzero(active)
                            q = sample["joint_pos"][ids]
                            reference = sample["reference_joint_pos"][ids]
                            velocity = sample["forward_velocity"][ids]
                            counts[ids] += 1
                            sums_q[ids] += np.abs(q - reference).mean(axis=1)
                            sums_v[ids] += np.abs(velocity - speed)
                            sums_speed[ids] += velocity
                            reached[ids] |= np.abs(velocity - speed) <= 0.05
                            q_rows.append(q.astype(np.float32))
                            ref_rows.append(reference.astype(np.float32))
                            if args.save_trajectories:
                                for key, rows in physical_rows.items():
                                    rows.append(sample[key][:, ids].swapaxes(0, 1))
                                for key, rows in jump_rows.items():
                                    rows.append(sample[key][ids].copy())
                                jump_masks.append(np.isfinite(sample["p_jump_reference"][ids]).all(1))
                                trace_rows.append(
                                    np.column_stack((ids, np.full(len(ids), (step + 1) * env.dt), velocity))
                                )
                        if np.any(active & extras["time_outs"].cpu().numpy().astype(bool)):
                            raise RuntimeError("Unexpected timeout inside the evaluation horizon")
                        for index in np.flatnonzero(active & done):
                            failure_times[index] = (step + 1) * env.dt
                        active &= ~done  # A reset never starts a replacement trial.
                        runner.alg.actor.reset(dones.to(runner.device))
                        runner.alg.critic.reset(dones.to(runner.device))
                        if not active.any():
                            break
                    q = np.concatenate(q_rows) if q_rows else np.empty((0, 10), dtype=np.float32)
                    reference = np.concatenate(ref_rows) if ref_rows else np.empty((0, 10), dtype=np.float32)
                    speed_q.append(q)
                    speed_ref.append(reference)
                    if failure_trace is not None:
                        failure_trace.save(trace_dir / f"failure_v{speed:g}_seed{seed}.npz", active,
                                           env.action_dof_names, env.cfg.asset.feet,
                                           control={"p_gains": env.p_gains[env.action_dof_indices],
                                                    "d_gains": env.d_gains[env.action_dof_indices],
                                                    "torque_limits_Nm": env.torque_limits[env.action_dof_indices],
                                                    "physics_dt_s": env.cfg.sim.dt, "control_dt_s": env.dt})
                    control_stats = metrics.finish()
                    for i in range(env.num_envs):
                        n = int(counts[i])
                        trials.append(
                            {
                                "seed": seed,
                                "env_index": i,
                                "controller": control_stats[i],
                                "pushed": bool(pushed[i]),
                                "samples": n,
                                "complete": bool(active[i] and n == steps),
                                "failure_time_s": failure_times[i],
                                "reached_actual_speed": bool(reached[i]),
                                "E_qpos_observed": float(sums_q[i] / n) if n else None,
                                "E_vel_observed": float(sums_v[i] / n) if n else None,
                                "mean_forward_speed": float(sums_speed[i] / n) if n else None,
                                **({"failure_diagnostics": failure_trace.summaries[i]} if failure_trace else {}),
                            }
                        )
                    if args.save_trajectories:
                        trace = np.concatenate(trace_rows) if trace_rows else np.empty((0, 3))
                        np.savez_compressed(
                            trace_dir / f"v{speed:g}_seed{seed}.npz",
                            joint_pos=q,
                            reference_joint_pos=reference,
                            env_index=trace[:, 0].astype(int),
                            time_s=trace[:, 1],
                            forward_velocity=trace[:, 2],
                            command_speed=speed,
                            joint_names=np.array(JOINT_NAMES),
                            control_joint_names=np.array(env.dof_names),
                            p_jump_valid=np.concatenate(jump_masks) if jump_masks else np.empty(0, dtype=bool),
                            **{key: np.concatenate(rows) if rows else np.empty((0, len(env.dof_names)))
                               for key, rows in jump_rows.items()},
                            **{key: np.concatenate(rows) if rows else
                               np.empty((0, env.cfg.control.decimation, len(env.dof_names)))
                               for key, rows in physical_rows.items()},
                        )
                q, reference = np.concatenate(speed_q), np.concatenate(speed_ref)
                pooled_q.append(q)
                pooled_ref.append(reference)
                row = {
                    "command_speed": speed,
                    "samples": len(q),
                    "trials": trials,
                    "complete": all(t["complete"] for t in trials),
                    "survival_fraction": float(np.mean([t["failure_time_s"] is None for t in trials])),
                    "reached_actual_speed_fraction": float(np.mean([t["reached_actual_speed"] for t in trials])),
                    "FID_joint_observed": frechet_distance(reference, q) if len(q) >= 2 else None,
                }
                for name in ("E_qpos_observed", "E_vel_observed", "mean_forward_speed"):
                    values = [t[name] for t in trials if t[name] is not None]
                    row[name] = float(np.mean(values)) if values else None
                row["controller_summary"] = speed_summary([row])
                results.append(row)
                print(
                    f"v={speed:.1f}: survival={row['survival_fraction']:.3f}, samples={len(q)}, complete={row['complete']}",
                    flush=True,
                )
    finally:
        env.record_evaluation = False
        env.record_failure_trace = False
        env.evaluation_latency_steps = None
    aggregate = aggregate_results(results)
    q, reference = np.concatenate(pooled_q), np.concatenate(pooled_ref)
    fid = frechet_distance(reference, q) if len(q) >= 2 else None
    aggregate["FID_joint"] = fid if aggregate["complete"] else None
    aggregate["FID_joint_observed_only"] = fid
    grid_matches = sorted(args.speeds) == PAPER_SPEED_GRID
    ramp_complete = warmup * env.dt >= max(args.speeds) / env.cfg.commands.acceleration_limit
    checkpoint = Path(args.checkpoint_path).resolve()
    snapshot = checkpoint.parent / "run_config.json"
    training = json.loads(snapshot.read_text())
    control = training["env"]["control"]
    report = {
        "schema_version": 3,
        "paper": "https://arxiv.org/abs/2605.28549v1",
        "checkpoint": str(checkpoint),
        "training": {
            "arm": control.get("arm", "A0"),
            "arm_options": control.get("arm_options", {}),
            "seed": training["train"]["seed"],
            "iterations": runner.current_learning_iteration + 1,
            "config": str(snapshot),
            "config_sha256": sha256(snapshot),
            "experiment_name": training["train"]["runner"]["experiment_name"],
        },
        "checkpoint_sha256": sha256(args.checkpoint_path),
        "prior_sha256": sha256(env.cfg.reference.checkpoint),
        "control_contract": runner.sprint_contract(),
        "speed_bins": summarize_bins(results),
        "versions": {name: version(name) for name in ("mujoco", "rsl-rl-lib", "torch", "numpy")},
        "protocol": {
            "speed_grid_matches_paper": grid_matches,
            "paper_grid_complete": grid_matches and aggregate["complete"],
            "command_ramp_complete_before_measurement": ramp_complete,
            "default_protocol_settings": (
                grid_matches
                and impulse == 0
                and getattr(args, "latency_steps", None) in (None, 0)
                and args.seeds == [42, 43, 44]
                and env.num_envs == 8
                and steps * env.dt == 20
                and warmup * env.dt >= 6 / env.cfg.commands.acceleration_limit + 2
            ),
            "backend": "MuJoCo (paper: Isaac Gym)",
            "rl_device": str(runner.device),
            "physics_dt": env.cfg.sim.dt,
            "control_dt": env.dt,
            "seeds": args.seeds,
            "trials_per_seed_speed": env.num_envs,
            "warmup_seconds": warmup * env.dt,
            "measurement_seconds": steps * env.dt,
            "initial_state": "environment reset distribution; fixed seed stream per command; LSTM and phase reset",
            "commands": "constant target [vx,0,0]; existing acceleration ramp from rest",
            "randomization": "noise, spontaneous pushes, friction and mass variation disabled",
            "latency_steps": getattr(args, "latency_steps", None),
            "latency_steps_training": getattr(env.cfg.control, "fixed_latency_steps", None),
            "latency_max_training": env.cfg.control.max_latency_steps,
            "push_impulse_Ns": impulse,
            "push_protocol": "heading-frame lateral pelvis COM impulse on first measured tick, opposite signs by environment parity and seed",
            "spectral_joints": PITCH_JOINTS,
            "control_joint_names": env.dof_names,
            "welch": "200 Hz applied torque, one-second Hann windows, half overlap, detrended. Per-joint summed 20-100 Hz / all bins, then equal joint mean.",
            "smoothness": "Policy-rate clipped residual actions and pre-reset endpoint joint velocities. Joint mean squared first/second action difference and absolute second velocity difference / dt^2, pooled by valid count.",
            "p_jump": "RMS Nm at the first substep relative to the previous last substep. Reference is the delayed zero-residual target; residual is the remaining applied target offset after clipping.",
            "speed_bins": "command bins [0,2), [2,4), [4,6]. Tracking errors equally weight trials then observed commands; controller sums pool valid samples.",
            "sampling": "post-step state and same-time spectral reference; includes first failure; excludes all later resets",
            "aggregation": "time mean per trial, equal trial mean per speed, equal speed mean; full metrics require every trial complete",
            "failure_handling": "no replacement trials or success filtering; observed-only diagnostics separated; missing full metrics are null",
            "fid_features": "10 raw joint angles (radians) per frame; unbiased covariance; pooled samples",
            "fid_comparability": "Paper feature extraction unspecified; FID_joint is not table-equivalent FID",
            "reached_speed": "at least one measured actual-velocity sample within 0.05 m/s; diagnostic only, never a sample filter",
            "trajectory_directory": str(trace_dir.resolve()) if args.save_trajectories else None,
            "failure_trace_directory": str(trace_dir.resolve()) if failure_window else None,
            "failure_window_seconds": failure_window,
            "failure_trace_sampling": "Last window including terminal state before reset; survivors end at horizon. Phase is t+dt; PD target and issued residual produced that transition. Feet follow foot_names. Reasons are nonexclusive.",
        },
        "aggregate": aggregate,
        "results": results,
    }
    write_report(report, output)
    print(f"Saved {output}; complete paper grid: {report['protocol']['paper_grid_complete']}")
    return report


def main(argv=None):
    from legged_gym.scripts.play import make_playback
    from legged_gym.utils.helpers import get_args

    args = get_args(argv, mode="evaluate")
    if args.calibrate_push:
        args.speeds = [3.0]
        if args.warmup_seconds is None:
            args.warmup_seconds = 8.0
    env, runner = make_playback(args, evaluation=True)
    try:
        if args.calibrate_push:
            from legged_gym.scripts.push_calibration import calibrate_push

            calibrate_push(env, runner, args, evaluate_policy)
        else:
            evaluate_policy(env, runner, args)
    finally:
        env.close()


if __name__ == "__main__":
    main()
