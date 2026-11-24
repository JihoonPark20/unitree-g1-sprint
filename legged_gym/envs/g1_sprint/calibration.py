"""Calibrate the prior's speed input to the robot's own stride.

The prior's speed labels are human speeds; retargeted onto G1 the same cycle moves the pelvis 15-35% slower.
"""

import mujoco
import numpy as np
import torch


def cycle_speed(model, data, qpos_adr, cycle, feet, frequency, stance_fraction):
    """No-slip pelvis speed of one joint cycle: median backward speed of each foot over its stance.

    Kinematic estimate with a level, fixed pelvis; stance = the lowest `stance_fraction` of the cycle.
    """
    positions = []
    for q in cycle:
        data.qpos[:] = 0
        data.qpos[3] = 1  # Identity free-joint quaternion.
        data.qpos[qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos[feet].copy())
    x = np.asarray(positions)
    dt = 1.0 / (frequency * len(cycle))
    speeds = []
    for k in range(len(feet)):
        vx = -(np.roll(x[:, k, 0], -1) - np.roll(x[:, k, 0], 1)) / (2 * dt)
        height = x[:, k, 2]
        speeds.append(np.median(vx[height <= np.quantile(height, stance_fraction)]))
    return float(np.mean(speeds))


def kinematic_speed_calibration(model, joint_names, default, reference, feet, stance_fraction, step=0.05, samples=120):
    """(robot_speeds, prior_speeds) knots: command speed -> prior speed label with that no-slip robot speed."""
    data = mujoco.MjData(model)
    qpos_adr = [model.joint(name).qposadr[0] for name in joint_names]
    feet = [model.body(name).id for name in feet]
    knots = reference.mapping.speeds.numpy()
    labels = np.arange(knots[1], knots[-1] + 1e-9, step)  # Below the first clip the prior only blends to standing.
    phase = torch.linspace(0, 1, samples + 1)[:-1]
    robot = []
    with torch.no_grad():
        for label in labels:
            f = float(reference.mapping(torch.tensor([label], dtype=torch.float32))[0])
            cycle = np.tile(default, (samples, 1))
            cycle[:, reference.indices] = reference.prior(phase, torch.full((samples,), f), sample=False)[0].numpy()
            robot.append(cycle_speed(model, data, qpos_adr, cycle, feet, f, stance_fraction))
    robot = np.asarray(robot)
    keep = robot > np.maximum.accumulate(np.r_[0.0, robot[:-1]])  # Strictly increasing for interpolation.
    if keep.sum() < 2:
        raise ValueError("Prior cycles do not produce increasing robot speed; cannot calibrate")
    return np.r_[0.0, robot[keep]], np.r_[0.0, labels[keep]]
