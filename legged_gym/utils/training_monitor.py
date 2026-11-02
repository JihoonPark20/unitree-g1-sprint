"""Read-only Viser monitoring of a selected training environment and PPO progress.

The trainer copies at most one state per display interval. Rendering uses a
private MuJoCo model/data pair on a worker thread, with a single pending frame.
"""

import copy
import math
import threading
import time
from collections import deque

import mujoco
import numpy as np

from legged_gym.utils.camera import ViserCameraControls


class TrainingMonitor:
    def __init__(self, env, runner, cfg, *, task):
        import viser
        from mjviser import ViserMujocoScene

        if not math.isfinite(cfg.max_fps) or cfg.max_fps <= 0 or cfg.history_points < 2:
            raise ValueError("viewer.max_fps must be positive and history_points must be at least 2")
        self.env = env
        self.runner = runner
        self._interval = 1.0 / cfg.max_fps
        self._start = time.monotonic()
        self._last_capture = -math.inf
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pending_frame = None
        self._stats = {}
        self._stats_version = 0
        self._shown_stats_version = -1
        self._reward_history = deque(maxlen=cfg.history_points)
        self._velocity_history = deque(maxlen=cfg.history_points)
        self._selected_env = None
        self._thread = None
        self.server = None
        self._attached = False
        self._closed = False
        self._original_step = env.step
        self._original_log = runner.logger.log
        try:
            self.server = viser.ViserServer(host=cfg.host, port=cfg.port, label=f"Legged Gym · {task}", verbose=False)
            self.server.scene.set_up_direction("+z")
            # mjviser changes visualization properties; never pass the training model.
            self.model = copy.copy(env.sim.model)
            visual_bodies = self.model.geom_bodyid[self.model.geom_group == 1]
            collisions = (self.model.geom_contype != 0) & np.isin(self.model.geom_bodyid, visual_bodies)
            self.model.geom_group[collisions] = 3
            self.data = mujoco.MjData(self.model)
            self.scene = ViserMujocoScene(self.server, self.model, num_envs=1)
            self.scene.camera_tracking_enabled = False
            self.scene.site_groups_visible = [False] * 6
            self._make_gui(task, cfg.max_fps)
            self._publish_frame(self._snapshot(0))

            env.step = self._step
            runner.logger.log = self._log
            self._attached = True
            self._thread = threading.Thread(target=self._publish_loop, name="legged-gym-viser", daemon=True)
            self._thread.start()
        except Exception:
            self.close()
            raise

    @property
    def url(self):
        host = self.server.get_host()
        if host == "0.0.0.0":
            host = "127.0.0.1"
        return f"http://{host}:{self.server.get_port()}"

    def _make_gui(self, task, max_fps):
        gui = self.server.gui
        gui.add_markdown(
            f"## {task.upper()} · Live training\nObserving sampled training actions, including exploration."
        )
        self.environment = gui.add_number("Environment", initial_value=0, min=0, max=self.env.num_envs - 1, step=1)
        self.camera = ViserCameraControls(self.server, self.env.sim.qpos[0, :3])
        self.freeze = gui.add_checkbox(
            "Freeze view", initial_value=False, hint="Training continues while the view is frozen."
        )
        self.progress = gui.add_markdown("Waiting for the first PPO iteration…")
        self.motion = gui.add_markdown("Waiting for a training frame…")
        empty = np.array([0.0], dtype=np.float64)
        self.velocity_plot = gui.add_uplot(
            (empty, empty, empty, empty),
            (
                {"label": "Elapsed s"},
                {"label": "Command", "stroke": "#f59e0b"},
                {"label": "Ramped command", "stroke": "#60a5fa"},
                {"label": "Actual", "stroke": "#22c55e"},
            ),
            title="X · Forward velocity · m/s (heading frame)",
            scales={"x": {"time": False}},
            height=180,
        )
        self.lateral_velocity_plot = gui.add_uplot(
            (empty, empty, empty),
            (
                {"label": "Elapsed s"},
                {"label": "Command", "stroke": "#f59e0b"},
                {"label": "Actual", "stroke": "#22c55e"},
            ),
            title="Y · Lateral velocity · m/s (heading frame)",
            scales={"x": {"time": False}},
            height=180,
        )
        self.angular_velocity_plot = gui.add_uplot(
            (empty, empty, empty),
            (
                {"label": "Elapsed s"},
                {"label": "Command", "stroke": "#f59e0b"},
                {"label": "Actual", "stroke": "#22c55e"},
            ),
            title="Yaw angular velocity · rad/s (body Z)",
            scales={"x": {"time": False}},
            height=180,
        )
        self.reward_plot = gui.add_uplot(
            (empty, empty),
            ({"label": "Iteration"}, {"label": "Mean episode return", "stroke": "#60a5fa"}),
            title="Training reward",
            scales={"x": {"time": False}},
            height=180,
        )
        gui.add_markdown(
            f"{self.env.num_envs:,} environments · {self.runner.device} learner\n\nDisplay limit: {max_fps:g} FPS"
        )

    def _snapshot(self, index):
        """Called only on the trainer thread; all mutable arrays are copied."""
        env = self.env
        qpos = env.sim.qpos[index].copy()
        qvel = env.sim.qvel[index].copy()
        # Match SPRINT's forward tracking reward; torso pitch must not distort vx.
        w, x, y, z = qpos[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        velocity = np.array(
            [
                np.cos(yaw) * qvel[0] + np.sin(yaw) * qvel[1],
                -np.sin(yaw) * qvel[0] + np.cos(yaw) * qvel[1],
                env.base_ang_vel[index, 2],
            ]
        )
        return {
            "env": index,
            "time": time.monotonic() - self._start,
            "qpos": qpos,
            "qvel": qvel,
            "velocity": velocity,
            "command": env.commands[index].copy(),
            "filtered_speed": float(env.filtered_speed[index]),
            "frequency": float(env.frequency[index]),
            "contacts": env._feet_contact[index].copy(),
            "max_command_x": float(env.command_ranges[0, 1]),
        }

    def _step(self, actions):
        result = self._original_step(actions)
        now = time.monotonic()
        if now - self._last_capture >= self._interval and not self._stop.is_set():
            self._last_capture = now
            if self.server.get_clients() and not self.freeze.value:
                index = int(np.clip(self.environment.value, 0, self.env.num_envs - 1))
                frame = self._snapshot(index)
                with self._lock:
                    self._pending_frame = frame  # Drop stale frames instead of slowing training.
        return result

    def _log(self, **kwargs):
        result = self._original_log(**kwargs)
        logger = self.runner.logger
        reward = float(np.mean(logger.rewbuffer)) if logger.rewbuffer else None
        length = float(np.mean(logger.lenbuffer)) * self.env.dt if logger.lenbuffer else None
        elapsed = kwargs["collect_time"] + kwargs["learn_time"]
        with self._lock:
            self._stats = {
                "iteration": kwargs["it"],
                "total": kwargs["total_it"],
                "reward": reward,
                "length": length,
                "lr": kwargs["learning_rate"],
                "fps": self.env.num_envs * self.runner.cfg["num_steps_per_env"] / max(elapsed, 1e-9),
                "collect": kwargs["collect_time"],
                "learn": kwargs["learn_time"],
            }
            if reward is not None:
                self._reward_history.append((kwargs["it"] + 1, reward))
            self._stats_version += 1
        return result

    def _publish_loop(self):
        try:
            while not self._stop.wait(self._interval):
                with self._lock:
                    frame, self._pending_frame = self._pending_frame, None
                    stats = self._stats.copy()
                    version = self._stats_version
                    history = list(self._reward_history)
                if frame is not None and not self.freeze.value:
                    self._publish_frame(frame)
                if stats and version != self._shown_stats_version:
                    self._publish_stats(stats, history)
                    self._shown_stats_version = version
        except Exception as error:
            self._stop.set()
            print(f"Training monitor stopped updating: {error}. PPO training continues.")

    def _publish_frame(self, frame):
        self.data.qpos[:] = frame["qpos"]
        self.data.qvel[:] = frame["qvel"]
        # Kinematics is sufficient for geometry and avoids another physics step.
        mujoco.mj_kinematics(self.model, self.data)
        root = frame["qpos"][:3].copy()
        if self._selected_env != frame["env"]:
            self._velocity_history.clear()
            self._selected_env = frame["env"]
        self._velocity_history.append(
            (
                frame["time"],
                frame["command"][0],
                frame["filtered_speed"],
                frame["velocity"][0],
                frame["command"][1],
                frame["velocity"][1],
                frame["command"][2],
                frame["velocity"][2],
            )
        )
        self.camera.update_root(root)
        with self.server.atomic():
            self.scene.update_from_mjdata(self.data)
            command, velocity = frame["command"], frame["velocity"]
            contacts = " / ".join("contact" if value else "air" for value in frame["contacts"])
            self.motion.content = (
                f"**Environment {frame['env']}**\n\n"
                "| | Command | Actual |\n|---|---:|---:|\n"
                f"| X / Forward m/s | {command[0]:.2f} | {velocity[0]:.2f} |\n"
                f"| Y / Lateral m/s | {command[1]:.2f} | {velocity[1]:.2f} |\n"
                f"| Yaw rad/s | {command[2]:.2f} | {velocity[2]:.2f} |\n\n"
                f"Ramped forward command: **{frame['filtered_speed']:.2f} m/s** · "
                f"Reference frequency: **{frame['frequency']:.2f} Hz**\n\n"
                f"Feet: {contacts}\n\nCommand curriculum: max forward **{frame['max_command_x']:.2f} m/s**"
            )
            history = np.asarray(self._velocity_history, dtype=float).T
            self.velocity_plot.data = tuple(history[[0, 1, 2, 3]])
            self.lateral_velocity_plot.data = tuple(history[[0, 4, 5]])
            self.angular_velocity_plot.data = tuple(history[[0, 6, 7]])

    def _publish_stats(self, stats, history):
        reward = "—" if stats["reward"] is None else f"{stats['reward']:.2f}"
        length = "—" if stats["length"] is None else f"{stats['length']:.1f} s"
        self.progress.content = (
            f"**Iteration {stats['iteration'] + 1:,} / {stats['total']:,}**\n\n"
            f"Episode return: **{reward}** · Length: **{length}**\n\n"
            f"{stats['fps']:,.0f} samples/s · LR {stats['lr']:.2g}\n\n"
            f"Collection {stats['collect']:.2f}s · PPO {stats['learn']:.2f}s"
        )
        if history:
            self.reward_plot.data = tuple(np.asarray(history, dtype=float).T)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._attached:
            self.env.step = self._original_step
            self.runner.logger.log = self._original_log
            self._attached = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.server is not None:
            self.server.stop()
