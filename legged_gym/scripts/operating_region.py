"""Place saved evaluation trajectories on each joint's actuator torque-speed envelope.

    python -m legged_gym.scripts.operating_region run.json [--envelope-velocity-scale S] [--envelope-torque-scale S]

The report must come from sprint-evaluate --save-trajectories. The envelope defaults to the one the run
applied, then to the checkpoint's run_config.json and SprintCfg. The scale options multiply it.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from legged_gym.envs.g1_sprint.config import SprintCfg
from legged_gym.envs.g1_sprint.envelope import actuator_table, envelope_bounds
from legged_gym.sim.mujoco_batch import joint_parameters

DEFAULT_JOINTS = [f"{side}_{joint}_joint" for joint in ("hip_pitch", "knee", "ankle_pitch", "shoulder_pitch")
                  for side in ("left", "right")]


def load(report_path):
    report = json.loads(Path(report_path).read_text())
    folder = report["protocol"]["trajectory_directory"]
    if not folder:
        raise ValueError("Report has no trajectories; rerun sprint-evaluate with --save-trajectories")
    files = sorted(Path(folder).glob("v*_seed*.npz"))
    if not files:
        raise ValueError(f"No trajectory files in {folder}")
    names = list(np.load(files[0])["control_joint_names"])
    data = {key: np.concatenate([np.load(f)[key].reshape(-1, len(names)) for f in files])
            for key in ("applied_torque", "requested_torque", "joint_velocity")}
    config = Path(report["checkpoint"]).parent / "run_config.json"
    control = json.loads(config.read_text())["env"]["control"]
    return report, names, data, control


def region(names, data, control, used=None, velocity_scale=None, torque_scale=None):
    """`used` is the envelope recorded in the report; the scales multiply it."""
    if used:
        models = used["model"]
        y1, y2, x1, x2 = (np.asarray(used[k], dtype=float) for k in ("y1", "y2", "x1", "x2"))
    else:
        table = actuator_table(names, control.get("joint_actuators", SprintCfg.control.joint_actuators),
                               control.get("actuator_models", SprintCfg.control.actuator_models))
        models = table["model"]
        v, t = control.get("envelope_velocity_scale", 1.0), control.get("envelope_torque_scale", 1.0)
        y1, y2, x1, x2 = table["y1"] * t, table["y2"] * t, table["x1"] * v, table["x2"] * v
    v, t = velocity_scale or 1.0, torque_scale or 1.0
    y1, y2, x1, x2 = y1 * t, y2 * t, x1 * v, x2 * v
    effort = joint_parameters(names, control["effort_limits"])
    tau, req, vel = data["applied_torque"], data["requested_torque"], data["joint_velocity"]
    lo, hi = envelope_bounds(y1, y2, x1, x2, vel)
    lo, hi = np.maximum(lo, -effort), np.minimum(hi, effort)
    speed = np.abs(vel)
    rows = {}
    for j, name in enumerate(names):
        rows[name] = dict(
            actuator=models[j], effort_Nm=float(effort[j]), y1=float(y1[j]), y2=float(y2[j]),
            x1=float(x1[j]), x2=float(x2[j]),
            speed_p50=float(np.median(speed[:, j])), speed_p99=float(np.percentile(speed[:, j], 99)),
            speed_max=float(speed[:, j].max()),
            above_x1=float(np.mean(speed[:, j] > x1[j])), above_x2=float(np.mean(speed[:, j] > x2[j])),
            driving=float(np.mean(tau[:, j] * vel[:, j] > 0)),
            applied_outside_envelope=float(np.mean((tau[:, j] > hi[:, j] + 1e-6) | (tau[:, j] < lo[:, j] - 1e-6))),
            request_beyond_envelope=float(np.mean((req[:, j] > hi[:, j]) | (req[:, j] < lo[:, j]))),
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--envelope-velocity-scale", type=float, help="Multiply x1 and x2")
    parser.add_argument("--envelope-torque-scale", type=float, help="Multiply y1 and y2")
    parser.add_argument("--joints", nargs="+", default=DEFAULT_JOINTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report, names, data, control = load(args.report)
    used = report["protocol"].get("torque_speed_envelope")
    rows = region(names, data, control, used, args.envelope_velocity_scale, args.envelope_torque_scale)
    result = dict(report=str(args.report.resolve()), samples=len(data["joint_velocity"]),
                  envelope_applied_in_run=used is not None, joints=rows)
    if args.output:
        args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{result['samples']} substep samples; envelope applied in run: {used is not None}")
    print(f"{'joint':28s} {'actuator':>10} {'x1':>5} {'x2':>5} {'|w| p99':>8} {'>x1':>6} {'>x2':>6} {'outside':>8} {'req>env':>8}")
    for name in args.joints:
        r = rows[name]
        print(f"{name:28s} {r['actuator']:>10} {r['x1']:5.1f} {r['x2']:5.1f} {r['speed_p99']:8.1f} {r['above_x1']*100:5.1f}% "
              f"{r['above_x2']*100:5.1f}% {r['applied_outside_envelope']*100:7.1f}% {r['request_beyond_envelope']*100:7.1f}%")


if __name__ == "__main__":
    main()
