"""Unitree G1 with 23 policy actions and six wrists held by PD."""

from legged_gym.envs.base.legged_robot_config import MujocoPPOCfg, MujocoRobotCfg


class G1FlatCfg(MujocoRobotCfg):
    class env(MujocoRobotCfg.env):
        num_actions = 23

    class asset(MujocoRobotCfg.asset):
        file = "g1/g1_29dof.xml"
        root_body = "pelvis"
        mass_body = "torso_link"
        joint_names = (
            [
                f"{side}_{joint}_joint"
                for side in ("left", "right")
                for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
            ]
            + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
            + [
                f"{side}_{joint}_joint"
                for side in ("left", "right")
                for joint in (
                    "shoulder_pitch",
                    "shoulder_roll",
                    "shoulder_yaw",
                    "elbow",
                    "wrist_roll",
                    "wrist_pitch",
                    "wrist_yaw",
                )
            ]
        )
        # Implementation choice: the paper does not specify wrist control.
        action_joint_names = [name for name in joint_names if "_wrist_" not in name]
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]

    class init_state(MujocoRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8]
        default_joint_angles = {
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        }

    class control(MujocoRobotCfg.control):
        stiffness = {".*": 40.0, ".*_hip_.*": 100.0, ".*_knee_joint": 150.0, "waist_yaw_joint": 200.0}
        damping = {".*": 1.0, ".*_hip_.*": 2.0, ".*_knee_joint": 4.0, ".*_ankle_.*": 2.0, "waist_.*": 5.0}
        effort_limits = {
            ".*": 25.0,
            ".*_hip_pitch_joint": 88.0,
            ".*_hip_yaw_joint": 88.0,
            "waist_yaw_joint": 88.0,
            ".*_hip_roll_joint": 139.0,
            ".*_knee_joint": 139.0,
            ".*_wrist_(pitch|yaw)_joint": 5.0,
        }
        frictionloss = 0.0

    class domain_rand(MujocoRobotCfg.domain_rand):
        friction_range = [0.3, 1.0]
        push_interval_s = [5.0, 5.0]


class G1FlatCfgPPO(MujocoPPOCfg):
    class runner(MujocoPPOCfg.runner):
        experiment_name = "g1_29dof_flat"
