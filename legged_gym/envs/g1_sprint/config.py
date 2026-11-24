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
        # Eq.11: compare q and q_ref at the same instant (post-step state vs reference at t+dt).
        prior_reward_time_aligned = True
        # Feed the prior the speed label whose cycle moves G1 at the commanded speed (no-slip kinematics).
        # Human speed labels overstate G1 speed by 15-35%. Deviation from the paper's speed map; see docs.
        speed_calibration = "g1_kinematic"

    class control(G1FlatCfg.control):
        decimation = 4  # 200 Hz physics, 50 Hz policy
        stiffness = {
            **G1FlatCfg.control.stiffness, ".*_hip_pitch_joint": 200.0,
            ".*_knee_joint": 300.0, ".*_(shoulder_pitch|elbow)_joint": 80.0,
            ".*_ankle_pitch_joint": 40.0,
        }
        damping = {
            **G1FlatCfg.control.damping, ".*_hip_pitch_joint": 4.0, ".*_knee_joint": 8.0,
            ".*_ankle_pitch_joint": 2.0, ".*_(shoulder_pitch|elbow)_joint": 2.0,
        }
        reference_velocity = True
        # Interpolate the held target along dq_ref over the 200 Hz substeps, so zero residual
        # tracks q_ref(t+dt) instead of lagging it. SPRINT does not publish a PD target rule.
        reference_lookahead = True
        residual_mode = "position"
        residual_leak = 0.0
        residual_feedforward = 0.0
        residual_obs = False
        residual_mix_direct = 0.0
        target_interp = None  # Inherit the checkpoint's reference_lookahead rule.
        # MJCF actuator ctrlrange; the inherited 25 Nm default under-limited ankles and waist roll/pitch.
        effort_limits = {**G1FlatCfg.control.effort_limits, ".*_ankle_.*": 50.0, "waist_(roll|pitch)_joint": 50.0}
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
        # SPRINT: expand "from static standing". With this controller a 0.4-0.8 start did not learn to
        # survive within 190M steps; 0-0.4 survived by 50M (briefly shuffling before stepping).
        ranges = [[0.0, 0.4], [0.0, 0.0], [0.0, 0.0]]
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
        # Anti-standing guard on top of RLvRL's score threshold. No forward-MAE gate: 0.1 m/s blocked a policy
        # tracking 1.0 m/s at 0.90 (score 0.87 passes); the score already bounds error to ~0.15 m/s.
        curriculum_progress = {
            "min_command": 0.2, "min_speed_ratio": 0.5,
            "max_forward_mae": None, "max_yaw_mae": None,
        }  # m/s, dimensionless, m/s, rad/s; evaluated after acceleration.
        # Optional extra sampling inside an active vx band, e.g. {"range": [1.25, 1.30], "fraction": 0.25}
        # (CLI --curriculum-focus). Not in the paper or RLvRL; off by default.
        curriculum_focus = None
        # Optional (CLI --curriculum-cycle-average): score the linear kernel on each reference gait cycle's mean
        # speed. Intra-stride surge grows with speed and caps the per-step score below 0.8 near 3.9 m/s at
        # tracking_sigma 0.1 even with zero mean error. Not in RLvRL (per-step kernels); off by default.
        curriculum_cycle_average = False

    class rewards(G1FlatCfg.rewards):
        soft_dof_pos_limit = 1.0
        torso_roll_deadzone = 0.1  # Absolute lateral projected gravity, not radians.
        # Do not reward foot lifting while commanded to stand. Reuse the
        # prior's smooth speed blend; lateral/turn commands still allow steps.
        air_height_command_gate = True
        # Credit foot clearance only within the Eq.12 swing duration (swing_duty / f), so holding
        # a foot up cannot outscore velocity tracking. Adaptation; the paper's kernel is uncapped.
        air_height_swing_cap = True
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
            "torso_roll": 0.0,  # Optional stabilization experiment; disabled for the baseline.
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
        num_steps_per_env = 96  # 1.92 s BPTT window: the LSTM infers phase over a full 0.61 Hz stride.
        log_interval = 1  # Console only; TensorBoard and Viser update every iteration.

    class policy(G1FlatCfgPPO.policy):
        recurrent = True
        actor_hidden_dims = []
        critic_hidden_dims = [256, 128]  # SPRINT specifies the actor only.
        obs_normalization = True
        activation = "elu"
        init_noise_std = 0.3
        noise_std_range = [0.1, 0.5]  # Applied to the actual rsl-rl distribution.

    class algorithm(G1FlatCfgPPO.algorithm):
        learning_rate = 3e-4
        entropy_coef = 0.001


class SprintLegacyCfg(SprintCfg):
    """Pre-diagnosis settings, kept only to measure the improvement (sprint-diagnose)."""

    class reference(SprintCfg.reference):
        prior_reward_time_aligned = False
        speed_calibration = None

    class control(SprintCfg.control):
        reference_lookahead = False
        effort_limits = G1FlatCfg.control.effort_limits

    class commands(SprintCfg.commands):
        ranges = [[0.4, 0.8], [0.0, 0.0], [0.0, 0.0]]
        curriculum_progress = {**SprintCfg.commands.curriculum_progress, "max_forward_mae": 0.1}

    class rewards(SprintCfg.rewards):
        air_height_swing_cap = False


class SprintLegacyPPOCfg(SprintPPOCfg):
    class runner(SprintPPOCfg.runner):
        num_steps_per_env = 32

    class policy(SprintPPOCfg.policy):
        critic_hidden_dims = []
        obs_normalization = False


class SprintPaperMapCfg(SprintCfg):
    """g1_sprint with the paper's human speed->frequency map; isolates the speed calibration."""

    class reference(SprintCfg.reference):
        speed_calibration = None


SPRINT_TASKS = {
    "g1_sprint": (SprintCfg, SprintPPOCfg),
    "g1_sprint_paper_map": (SprintPaperMapCfg, SprintPPOCfg),
    "g1_sprint_legacy": (SprintLegacyCfg, SprintLegacyPPOCfg),
}
