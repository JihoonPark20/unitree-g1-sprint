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
        clip_actions = 10.0
        max_latency_steps = 2  # Control ticks; sampled per episode.

    class commands(G1FlatCfg.commands):
        # Bootstrap with standing + slow walking so the policy sees changing
        # commands and an active gait prior before any curriculum promotion.
        # 0.8 m/s covers the first measured walking anchor (0.661 m/s).
        ranges = [[0.0, 0.8], [0.0, 0.0], [-0.1, 0.1]]
        limit_ranges = [[0.0, 6.0], [0.0, 0.0], [-0.7, 0.7]]
        standing_fraction = 0.05
        acceleration_limit = 1.0
        curriculum_kind = "rlvrl_grid"
        curriculum_bin_widths = [0.2, 0.2]  # vx (m/s), yaw rate (rad/s)
        curriculum_local_range = [0.2, 0.2]
        curriculum_thresholds = [0.8, 0.8]  # Unweighted linear/angular kernels.
        curriculum_increment = 0.2
        curriculum_min_settled_s = 2.0
        curriculum_settle_tolerance = 0.02  # Filtered vx must reach the sampled command.

    class rewards(G1FlatCfg.rewards):
        soft_dof_pos_limit = 1.0
        preset = "sprint_run_scales_v1"
        # RuN Eqs. 9/10: 2 exp(-2 e_v^2), 0.5 exp(-e_yaw^2).
        tracking_sigma = 0.5
        tracking_ang_sigma = 1.0
        # SPRINT Eq.11 stays a sum of ten kernels. Weight 1/10 and sigma
        # 1/(10*0.6) match RuN Eq.7's peak and local quadratic coefficient
        # on this subset, not its full-joint aggregate exponential globally.
        prior_sigma = 1.0 / (10 * 0.6)
        min_feet_distance = 0.12
        swing_duty = 0.4
        scales = {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 0.5,
            "prior": 0.1,
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

    class policy(G1FlatCfgPPO.policy):
        actor_hidden_dims = []
        critic_hidden_dims = []
        activation = "elu"
        init_noise_std = 0.5

    class algorithm(G1FlatCfgPPO.algorithm):
        learning_rate = 3e-4
