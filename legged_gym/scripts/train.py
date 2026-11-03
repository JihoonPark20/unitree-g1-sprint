# SPDX-License-Identifier: BSD-3-Clause
# Based on legged_gym, Copyright (c) 2021 ETH Zurich, Nikita Rudin.
"""Train SPRINT with the Viser monitor; --headless disables the browser viewer."""

import shlex
import sys

from legged_gym.envs import task_registry
from legged_gym.utils import get_args


def train(args):
    env, _ = task_registry.make_env(name=args.task, args=args)
    monitor = None
    try:
        runner, cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
        if getattr(args, "warm_start", None):
            print(f"Actor initialized from {args.warm_start}; critic, optimizer, exploration and curriculum start fresh")
        print(f"MuJoCo CPU: {env.num_envs} envs / {env.sim.batch.nthread} workers; PPO: {runner.device}")
        print(f"Commands: vx={env.command_ranges[0].tolist()}, yaw={env.command_ranges[2].tolist()}")
        print(f"Logs: {runner.logger.log_dir}")
        if cfg.viewer.enabled and not args.headless:
            try:
                from legged_gym.utils.training_monitor import TrainingMonitor

                monitor = TrainingMonitor(env, runner, cfg.viewer, task=args.task)
                print(f"Training monitor: {monitor.url}")
            except Exception as error:
                install = f"{shlex.quote(sys.executable)} -m pip install -e ."
                raise RuntimeError(
                    f"Training monitor failed to start with {sys.executable}: {error}. "
                    f"For missing dependencies, run `{install}` from the project directory. "
                    "Use --headless to train without the web monitor."
                ) from error
        runner.learn(
            num_learning_iterations=cfg.runner.max_iterations,
            init_at_random_ep_len=False,
        )
        print(f"Checkpoint: {runner.logger.log_dir}/model_{runner.current_learning_iteration}.pt")
    finally:
        if monitor is not None:
            monitor.close()
        env.close()


def main(argv=None):
    train(get_args(argv))


if __name__ == "__main__":
    main()
