"""Spectral prior data, model, pretraining and runtime reference generation."""

JOINT_NAMES = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "knee", "ankle_pitch", "shoulder_pitch", "elbow")
)
