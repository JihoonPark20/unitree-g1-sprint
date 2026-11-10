"""Policy checkpoint playback; use mjpython for the macOS viewer."""

import time

import torch

from legged_gym.envs import task_registry
from legged_gym.utils.helpers import get_args


def make_playback(args, *, evaluation=False):
    cfg, train_cfg = task_registry.get_cfgs(args.task, args)
    cfg.env.num_envs = args.num_envs or (8 if evaluation else 1)
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    cfg.commands.curriculum = False
    if evaluation:
        if cfg.commands.acceleration_limit <= 0:
            raise ValueError("Evaluation requires a positive command acceleration limit")
        if args.warmup_seconds is None:
            args.warmup_seconds = max(args.speeds) / cfg.commands.acceleration_limit + 2
        duration = args.warmup_seconds + args.seconds + 1
        cfg.commands.resampling_time = duration
        cfg.env.episode_length_s = duration
        cfg.seed = args.seeds[0]
        train_cfg.seed = args.seeds[0]
    else:
        cfg.commands.ranges = [list(r) for r in cfg.commands.limit_ranges]
        if args.velocity:
            cfg.commands.ranges = [[v, v] for v in args.velocity]
            cfg.commands.standing_fraction = 0.0
    env, _ = task_registry.make_env(args.task, args, cfg)
    try:
        runner, _ = task_registry.make_alg_runner(env, args.task, args, train_cfg, log_root=None)
    except Exception:
        env.close()
        raise
    return env, runner


def play(args):
    env, runner = make_playback(args)
    try:
        policy = runner.get_inference_policy(device=runner.device)
        env.headless = args.headless
        env.follow_camera = args.camera == "follow"
        env.camera_azimuth = args.camera_azimuth
        env.camera_elevation = args.camera_elevation
        env.camera_distance = args.camera_distance
        env.camera_height = args.camera_height
        print(f"Checkpoint: {args.checkpoint_path}")
        print(f"Camera: {args.camera}; commands [vx, vy, yaw rate]: {env.commands[0].tolist()}")
        if not args.headless:
            print("Camera: drag to orbit/pan, scroll to zoom; F centers on robot, R resets view.")
        obs = env.get_observations()
        total_reward, done_count = 0.0, 0
        with torch.inference_mode():
            for step in range(args.steps):
                start = time.perf_counter()
                obs, rewards, dones, _ = env.step(policy(obs.to(runner.device)))
                runner.alg.actor.reset(dones.to(runner.device))
                runner.alg.critic.reset(dones.to(runner.device))
                total_reward += rewards.mean().item()
                done_count += dones.sum().item()
                if (step + 1) % max(1, round(1 / env.dt)) == 0:
                    print(
                        f"t={(step + 1) * env.dt:.1f}s env0: command_vx={env.commands[0, 0]:.2f} "
                        f"filtered_vx={env.filtered_speed[0]:.2f} body_vx={env.base_lin_vel[0, 0]:.2f} m/s "
                        f"height={env.base_height[0]:.2f}m completed_episodes={done_count}",
                        flush=True,
                    )
                if not args.headless:
                    if env._viewer is not None and not env._viewer.is_running():
                        break
                    time.sleep(max(0, env.dt - (time.perf_counter() - start)))
        print(f"Playback: accumulated mean reward={total_reward:.3f}, completed episodes={done_count}")
    finally:
        env.close()


def main(argv=None):
    play(get_args(argv, mode="play"))


if __name__ == "__main__":
    main()
