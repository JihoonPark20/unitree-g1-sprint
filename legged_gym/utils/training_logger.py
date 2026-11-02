"""Full TensorBoard metrics with standard rsl_rl-style terminal progress."""

import datetime
import json
import statistics
from pathlib import Path

import torch
from rsl_rl.utils.logger import Logger, LogWriter

from .helpers import class_to_dict


class TrainingLogger(Logger):
    def __init__(self, *args, control_dt, setup_output="", **kwargs):
        super().__init__(*args, **kwargs)
        self.control_dt = control_dt
        self.setup_output = setup_output
        self.console_interval = self.cfg.get("console_log_interval", 1)
        if not isinstance(self.console_interval, int) or self.console_interval < 1:
            raise ValueError("console_log_interval must be a positive integer")

    def init_logging_writer(self):
        super().init_logging_writer()
        if self.writer is not None and hasattr(self.writer, "add_text"):
            if self.setup_output:
                self.writer.add_text("Setup/models", f"```text\n{self.setup_output.rstrip()}\n```", 0)
            settings = json.dumps({"env": class_to_dict(self.env_cfg), "train": self.cfg}, indent=2, default=str)
            self.writer.add_text("Setup/config", f"```json\n{settings}\n```", 0)

    def log(self, it, start_it, total_it, collect_time, learn_time, loss_dict,
            learning_rate, action_std, rnd_weight, **kwargs):
        # Keep writer updates at EVERY iteration, independently of console cadence.
        if self.writer is None:
            return
        collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
        iteration_time = collect_time + learn_time
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time
        fps = int(collection_size / max(iteration_time, 1e-9))

        episode_metrics = {}
        for key in dict.fromkeys(k for info in self.ep_extras for k in info):
            values = [torch.as_tensor(info[key], device=self.device).reshape(-1)
                      for info in self.ep_extras if key in info]
            value = torch.cat(values).float().mean()
            tag = key if "/" in key else "Episode/" + key
            self.writer.add_scalar(tag, value, it)
            episode_metrics[tag] = value.item()

        for key, value in loss_dict.items():
            self.writer.add_scalar(f"Loss/{key}", value, it)
        self.writer.add_scalar("Loss/learning_rate", learning_rate, it)
        self.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collect_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        reward = statistics.mean(self.rewbuffer) if self.rewbuffer else None
        length = statistics.mean(self.lenbuffer) if self.lenbuffer else None
        if reward is not None:
            self.writer.add_scalar("Train/mean_reward", reward, it)
            self.writer.add_scalar("Train/mean_episode_length", length, it)
            if not isinstance(self.writer, LogWriter):
                self.writer.add_scalar("Train/mean_reward/time", reward, int(self.tot_time))
                self.writer.add_scalar("Train/mean_episode_length/time", length, int(self.tot_time))
            if self.cfg["algorithm"].get("rnd_cfg"):
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer), it)
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer), it)
                self.writer.add_scalar("Rnd/weight", rnd_weight, it)

        if it == start_it or (it + 1) % self.console_interval == 0 or it == total_it - 1:
            done = it + 1 - start_it
            eta = datetime.timedelta(seconds=int(self.tot_time / done * (total_it - it - 1)))
            self._print_progress(
                it, total_it, fps, collect_time, learn_time, loss_dict,
                learning_rate, action_std, reward, length, episode_metrics, eta,
            )

        if isinstance(self.writer, LogWriter):
            for video in Path(self.log_dir).rglob("*.mp4"):
                if video not in self.uploaded_videos:
                    self.writer.save_video(video, it)
                    self.uploaded_videos.add(video)
        self.ep_extras.clear()

    def _print_progress(self, it, total_it, fps, collect_time, learn_time, losses,
                        learning_rate, action_std, reward, length, metrics, eta):
        width, pad = 80, 40
        # Match the upstream iteration index (and checkpoint/TensorBoard step).
        lines = ["#" * width, f" Learning iteration {it}/{total_it} ".center(width), ""]

        def row(label, value):
            lines.append(f"{label + ':':>{pad}} {value}")

        if self.cfg.get("run_name"):
            row("Run name", self.cfg["run_name"])
        row("Total steps", str(self.tot_timesteps))
        row("Steps per second", str(fps))
        row("Collection time", f"{collect_time:.3f}s")
        row("Learning time", f"{learn_time:.3f}s")
        for name, value in losses.items():
            row(f"Mean {name} loss", f"{float(value):.4f}")
        row("Learning rate", f"{learning_rate:.2e}")
        if reward is not None:
            if self.cfg["algorithm"].get("rnd_cfg"):
                row("Mean extrinsic reward", f"{statistics.mean(self.erewbuffer):.2f}")
                row("Mean intrinsic reward", f"{statistics.mean(self.irewbuffer):.2f}")
            row("Mean reward", f"{reward:.2f}")
            row("Mean episode length", f"{length:.2f}")
        row("Mean action std", f"{action_std.mean().item():.2f}")
        # All reward terms belong in the standard episode log. Detailed custom
        # diagnostics remain in TensorBoard without overwhelming the terminal.
        for key, value in metrics.items():
            if key.startswith("Reward/") or key == "Curriculum/max_command_x":
                row(key, f"{value:.4f}")
        lines.append("-" * width)
        row("Iteration time", f"{collect_time + learn_time:.2f}s")
        row("Time elapsed", str(datetime.timedelta(seconds=int(self.tot_time))))
        row("ETA", str(eta))
        print("\n".join(lines) + "\n", flush=True)

    def stop_logging_writer(self):
        super().stop_logging_writer()
        if self.writer is not None and not isinstance(self.writer, LogWriter):
            self.writer.flush()
            self.writer.close()

    def abort_logging_writer(self):
        super().abort_logging_writer()
        if self.writer is not None and not isinstance(self.writer, LogWriter):
            self.writer.flush()
            self.writer.close()
