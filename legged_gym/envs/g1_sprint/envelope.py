"""Unitree actuator model: speed-dependent torque limits, friction and reflected inertia."""

import re

import numpy as np

FIELDS = ("y1", "y2", "x1", "x2", "fs", "fd", "armature")
FRICTION_VELOCITY = 0.01  # Va in unitree_actuators.py, rad/s


def actuator_table(joint_names, joint_actuators, models):
    """Per-joint actuator names and parameter arrays; later regex entries win."""
    chosen = [None] * len(joint_names)
    for pattern, model in joint_actuators.items():
        if model not in models:
            raise ValueError(f"Unknown actuator model: {model}")
        mask = [re.fullmatch(pattern, name) is not None for name in joint_names]
        if not any(mask):
            raise ValueError(f"Joint pattern matches no joints: {pattern}")
        chosen = [model if hit else old for hit, old in zip(mask, chosen)]
    missing = [name for name, model in zip(joint_names, chosen) if model is None]
    if missing:
        raise ValueError(f"Joints without an actuator model: {missing}")
    table = {key: np.array([float(models[m][key]) for m in chosen]) for key in FIELDS}
    if (table["x1"] < 0).any() or (table["x2"] <= table["x1"]).any() or (table["y1"] < 0).any() or (table["y2"] < 0).any():
        raise ValueError("Actuator models need 0 <= x1 < x2 and nonnegative peak torques")
    table["model"] = chosen
    return table


def envelope_bounds(y1, y2, x1, x2, velocity):
    """Torque bounds (lo, hi): y1 drives with the velocity, y2 brakes against it; both fall from x1 to zero at x2.

    At zero velocity both directions count as braking, as in unitree_actuators.py.
    """
    speed = np.abs(velocity)
    fade = np.clip((x2 - speed) / (x2 - x1), 0, 1)
    drive, brake = y1 * fade, y2 * fade
    hi = np.where(velocity > 0, drive, brake)
    lo = -np.where(velocity < 0, drive, brake)
    return lo, hi


def friction(fs, fd, velocity):
    return fs * np.tanh(velocity / FRICTION_VELOCITY) + fd * velocity
