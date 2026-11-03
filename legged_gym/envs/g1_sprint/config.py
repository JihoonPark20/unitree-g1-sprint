"""Paper-specified settings and explicit implementation choices (see README)."""

from legged_gym.envs.g1_sprint.robot_config import G1FlatCfg, G1FlatCfgPPO
from legged_gym.paths import DEFAULT_PRIOR


class SprintCfg(G1FlatCfg):
    class env(G1FlatCfg.env):
        num_envs = 4096
        history_length = 1  # LSTM owns temporal state.

    class reference:
        checkpoint = str(DEFAULT_PRIOR)
        blend_speed = 0.4

    class control(G1FlatCfg.control):
        decimation = 4  # 200 Hz physics, 50 Hz policy
        action_scale = 0.25
        clip_actions = 2.0
        # Physical residual bounds, in radians, around the prior/default pose.
        # Shoulder roll/yaw are outside the ten-joint motion prior.
        residual_limits = {".*": 0.5, ".*_shoulder_(roll|yaw)_joint": 0.15}
        max_latency_steps = 0  # Enable latency randomization after basic gait validation.

    class domain_rand(G1FlatCfg.domain_rand):
        push_robots = False  # Baseline gait learning; mass/friction/noise stay enabled.

    class termination(G1FlatCfg.termination):
        terminate_on_feet_crossing = True
        feet_crossing_margin = 0.02  # Left foot at least 2 cm right of right foot.
        feet_crossing_duration_s = 0.1  # Consecutive control steps; allow brief recovery.

    class commands(G1FlatCfg.commands):
        # Moving samples activate the full prior; separate samples train stops.
        ranges = [[0.4, 0.8], [0.0, 0.0], [0.0, 0.0]]
        limit_ranges = [[0.0, 6.0], [0.0, 0.0], [0.0, 0.0]]
        standing_fraction = 0.05
        acceleration_limit = 1.0
        curriculum_kind = "rlvrl_grid"
        curriculum_bin_widths = [0.2, 0.2]  # vx (m/s), yaw rate (rad/s)
        curriculum_local_range = [0.2, 0.2]
        curriculum_thresholds = [0.8, 0.0]  # Forward-only promotion; yaw reward remains active.
        curriculum_increment = 0.2
        curriculum_min_settled_s = 2.0
        curriculum_settle_tolerance = 0.02  # Filtered vx must reach the sampled command.
        curriculum_progress = {
            "min_command": 0.2, "min_speed_ratio": 0.5,
            "max_forward_mae": 0.1, "max_yaw_mae": None,
        }  # m/s, dimensionless, m/s, rad/s; evaluated after acceleration.

    class rewards(G1FlatCfg.rewards):
        soft_dof_pos_limit = 1.0
        # Do not reward foot lifting while commanded to stand. Reuse the
        # prior's smooth speed blend; lateral/turn commands still allow steps.
        air_height_command_gate = True
        # Unpublished kernel widths are tuned for actual motion tracking.
        # These are not the authors' settings.
        tracking_sigma = 0.1
        tracking_ang_sigma = 0.25
        # Same ten-joint sum / maximum 3, with a narrower tracking tolerance
        # to distinguish following the prior from cancelling it to stand still.
        prior_sigma = 0.03
        min_feet_distance = 0.12
        signed_feet_distance = True  # Adaptation: penalize reversed left/right foot ordering.
        swing_duty = 0.4
        scales = {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 1.0,
            "prior": 0.3,
            "close_feet": -100.0,
            "air_height": 5.0,
            "low_speed_air": -6.0,
            "high_speed_ground": -3.0,
            "feet_parallel": 1.0,
            "feet_slide": -10.0,
            # No RuN equivalent: unit penalty budget (RuN alive scale = 1),
            # normalized by max two-foot deficit 2*0.4/0.6. A heuristic.
            "swing_deficit": -0.75,
            "alive": 1.0,
            "torques": -5e-6,
            "dof_acc": -2e-8,
            "dof_pos_limits": -10.0,
            "dof_vel": -5e-4,
            "ang_vel_xy": -0.5,
            "base_height": -30.0,
            # RuN's -2 projected-gravity penalty scale, with SPRINT Eq.14
            # speed-dependent dead zone retained. Not an identical reward.
            "torso_pitch": -2.0,
        }


class SprintPPOCfg(G1FlatCfgPPO):
    class viewer:
        enabled = True
        history_points = 500
        host = "127.0.0.1"
        max_fps = 15
        port = 8080

    class runner(G1FlatCfgPPO.runner):
        experiment_name = "g1_sprint"
        num_steps_per_env = 32
        log_interval = 1  # Console only; TensorBoard and Viser update every iteration.

    class policy(G1FlatCfgPPO.policy):
        actor_hidden_dims = []
        critic_hidden_dims = []
        activation = "elu"
        init_noise_std = 0.3
        noise_std_range = [0.1, 0.5]  # Applied to the actual rsl-rl distribution.

    class algorithm(G1FlatCfgPPO.algorithm):
        learning_rate = 3e-4
        entropy_coef = 0.001
