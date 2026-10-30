"""Class-style settings for CPU MuJoCo and device-independent rsl-rl PPO.

Locomotion settings follow unitree_rl_lab @ 4960b84732b0c2ec593dccbfe963fda1bcd7b1e3.
Physics/contact parameters are MuJoCo adaptations, not PhysX equivalents.
"""

from .base_config import BaseConfig


class MujocoRobotCfg(BaseConfig):
    seed = 42

    class env:
        num_envs = 4096
        num_actions = 0
        history_length = 1
        episode_length_s = 20.0

    class sim:
        dt = 0.005
        nthread = 10  # Set to 0 to let MuJoCo choose the CPU count.
        persistent = False  # Memory scales with worker count, not environment count.
        iterations = 50

    class asset:
        file = ""
        root_body = ""
        mass_body = ""
        joint_names = []
        action_joint_names = []  # Other joints retain their default PD targets.
        feet = []

    class init_state:
        pos = [0.0, 0.0, 0.4]
        default_joint_angles = {}
        joint_velocity_range = [-1.0, 1.0]

    class control:
        decimation = 4
        action_scale = 0.25
        clip_actions = 100.0
        stiffness = {".*": 25.0}
        damping = {".*": 0.5}
        effort_limits = {".*": 23.4}
        armature = 0.01
        frictionloss = 0.01

    class commands:
        resampling_time = 10.0
        standing_fraction = 0.1
        ranges = [[-0.1, 0.1], [-0.1, 0.1], [-1.0, 1.0]]
        limit_ranges = [[-1.0, 1.0], [-0.4, 0.4], [-1.0, 1.0]]
        curriculum = True

    class noise:
        add_noise = True
        ang_vel = 0.2
        gravity = 0.05
        joint_pos = 0.01
        joint_vel = 1.5

    class domain_rand:
        randomize_friction = True
        friction_range = [0.3, 1.2]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        push_robots = True
        push_interval_s = [5.0, 10.0]
        max_push_vel_xy = 0.5

    class rewards:
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        contact_threshold = 1.0
        contact_history_length = 3
        base_height_target = 0.78
        feet_clearance_target = 0.1
        feet_clearance_std = 0.05
        scales = {}

    class termination:
        min_height = 0.2
        max_tilt = 0.8
        base_contact = False


class MujocoPPOCfg(BaseConfig):
    seed = 42

    class runner:
        num_steps_per_env = 24
        max_iterations = 50000
        save_interval = 100
        experiment_name = "locomotion"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
