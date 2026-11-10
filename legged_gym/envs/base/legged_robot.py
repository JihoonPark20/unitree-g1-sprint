"""Class-based legged_gym task on MuJoCo CPU batches.

Simulation, rewards and history stay on CPU. rsl-rl transfers the compact
TensorDict to the learner device (MPS, CUDA or CPU) once per control step.
"""

import math

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from legged_gym.sim.mujoco_batch import MujocoBatch, joint_parameters


def rotate_inverse(q, v):
    """Rotate world vectors into a body's frame; MuJoCo quaternions are wxyz."""
    xyz = q[:, 1:4]
    uv = np.cross(xyz, v)
    return v - 2 * q[:, :1] * uv + 2 * np.cross(xyz, uv)


class SprintRobot(VecEnv):
    def __init__(self, cfg, sim_params=None, physics_engine="mujoco", sim_device="cpu", headless=True):
        if physics_engine != "mujoco" or str(sim_device) != "cpu":
            raise ValueError("MuJoCo batch physics requires physics_engine='mujoco', sim_device='cpu'")
        if cfg.env.num_envs <= 0 or cfg.sim.dt <= 0 or cfg.control.decimation < 1:
            raise ValueError("num_envs, sim.dt and decimation must be positive")
        if cfg.env.history_length < 1 or cfg.rewards.contact_history_length < 1 or cfg.env.episode_length_s <= 0:
            raise ValueError("Observation/contact history lengths and episode length must be positive")
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.headless = headless
        self.num_envs = cfg.env.num_envs
        self.num_actions = cfg.env.num_actions
        self.dt = cfg.sim.dt * cfg.control.decimation
        self.max_episode_length = math.ceil(cfg.env.episode_length_s / self.dt)
        self.rng = np.random.default_rng(cfg.seed)
        self.dof_names = list(cfg.asset.joint_names)
        self.num_dof = len(self.dof_names)
        self.action_dof_names = list(cfg.asset.action_joint_names)
        if not self.num_dof or len(set(self.dof_names)) != self.num_dof:
            raise ValueError("asset.joint_names must contain unique physical joint names")
        if (
            self.num_actions <= 0
            or len(self.action_dof_names) != self.num_actions
            or len(set(self.action_dof_names)) != self.num_actions
            or not set(self.action_dof_names).issubset(self.dof_names)
        ):
            raise ValueError("asset.action_joint_names must contain one unique physical joint per action")
        self.action_dof_indices = np.array([self.dof_names.index(name) for name in self.action_dof_names])
        self.held_dof_indices = np.array(
            [i for i, name in enumerate(self.dof_names) if name not in self.action_dof_names], dtype=int
        )
        self.sim = MujocoBatch(cfg, self.rng)
        self.default_dof_pos = joint_parameters(self.dof_names, cfg.init_state.default_joint_angles, 0.0)
        self.p_gains = joint_parameters(self.dof_names, cfg.control.stiffness)
        self.d_gains = joint_parameters(self.dof_names, cfg.control.damping)
        self.torque_limits = joint_parameters(self.dof_names, cfg.control.effort_limits)
        limits = self.sim.model.jnt_range[self.sim.joint_ids]
        centers = limits.mean(axis=1)
        half_ranges = (limits[:, 1] - limits[:, 0]) * cfg.rewards.soft_dof_pos_limit / 2
        self.dof_pos_limits = np.stack((centers - half_ranges, centers + half_ranges), axis=1)
        self.feet_indices = self.sim.feet_ids - 1  # Contact buffers omit the world body.
        self._root_contact_index = self.sim.model.body(cfg.asset.root_body).id - 1
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.common_step_counter = 0
        shape = (self.num_envs, self.num_actions)
        self.actions = np.zeros(shape)
        self.last_actions = np.zeros(shape)
        self.last_dof_vel = np.zeros((self.num_envs, self.num_dof))
        self.torques = np.zeros((self.num_envs, self.num_dof))
        self.commands = np.zeros((self.num_envs, 3))
        self.command_ranges = np.array(cfg.commands.ranges, dtype=float, copy=True)
        self._next_push = np.zeros(self.num_envs)
        feet_shape = (self.num_envs, len(self.feet_indices))
        self.feet_air_time = np.zeros(feet_shape)
        self.feet_contact_time = np.zeros(feet_shape)
        self.last_air_time = np.zeros(feet_shape)
        self.last_contact_time = np.zeros(feet_shape)
        self._feet_contact = np.zeros(feet_shape, dtype=bool)
        self._first_contact = np.zeros(feet_shape, dtype=bool)
        self._touchdown_air_time = np.zeros(feet_shape)
        self._contact_history = np.zeros(
            (cfg.rewards.contact_history_length, self.num_envs, self.sim.model.nbody - 1, 3)
        )
        self._contact_cursor = 0
        self._prepare_reward_function()
        self.episode_sums = {name: np.zeros(self.num_envs) for name in self.reward_scales}
        self.rew_buf = np.zeros(self.num_envs)
        self._history = {}
        self.obs_buf = None
        self._viewer = None
        self._viewer_model = None
        self._viewer_data = None
        self._closed = False
        self.reset()
        self.num_obs = self.obs_buf["policy"].shape[-1]
        self.num_privileged_obs = self.obs_buf["critic"].shape[-1]

    def _prepare_reward_function(self):
        self.reward_scales = {name: value for name, value in self.cfg.rewards.scales.items() if value != 0}
        self.reward_functions = {name: getattr(self, f"_reward_{name}") for name in self.reward_scales}

    def _update_contacts(self):
        forces = self.sim.contact_forces
        self._contact_history[self._contact_cursor] = forces
        self._contact_cursor = (self._contact_cursor + 1) % self.cfg.rewards.contact_history_length
        contacts = np.linalg.norm(forces[:, self.feet_indices], axis=-1) > self.cfg.rewards.contact_threshold
        landing = contacts & ~self._feet_contact
        liftoff = ~contacts & self._feet_contact
        self.last_air_time[landing] = self.feet_air_time[landing]
        self.last_contact_time[liftoff] = self.feet_contact_time[liftoff]
        self._touchdown_air_time[landing] = self.feet_air_time[landing]
        self._first_contact |= landing
        self.feet_air_time[:] = np.where(contacts, 0.0, self.feet_air_time + self.cfg.sim.dt)
        self.feet_contact_time[:] = np.where(contacts, self.feet_contact_time + self.cfg.sim.dt, 0.0)
        self._feet_contact[:] = contacts

    def _refresh_state(self):
        self.base_pos = self.sim.qpos[:, :3].copy()
        self.base_quat = self.sim.qpos[:, 3:7].copy()
        self.base_lin_vel = rotate_inverse(self.base_quat, self.sim.qvel[:, :3])
        self.base_ang_vel = self.sim.qvel[:, 3:6].copy()
        self.projected_gravity = rotate_inverse(self.base_quat, np.broadcast_to([0.0, 0.0, -1.0], (self.num_envs, 3)))
        self.dof_pos = self.sim.qpos[:, self.sim.qpos_ids]
        self.dof_vel = self.sim.qvel[:, self.sim.qvel_ids]
        self.feet_pos = self.sim.xpos[:, self.sim.feet_ids]
        self.feet_vel = self.sim.feet_velocities
        self.contact_forces = self.sim.contact_forces
        self._contact_peak = np.linalg.norm(self._contact_history, axis=-1).max(axis=0)
        self.base_height = self.base_pos[:, 2]
        self.feet_ground_height = 0.0

    def check_termination(self):
        tilt = np.arccos(np.clip(-self.projected_gravity[:, 2], -1.0, 1.0))
        self.terminated_buf = (tilt > self.cfg.termination.max_tilt) | (
            self.base_height < self.cfg.termination.min_height
        )
        if self.cfg.termination.base_contact:
            self.terminated_buf |= self._contact_peak[:, self._root_contact_index] > self.cfg.rewards.contact_threshold
        finite = np.isfinite(self.sim.qpos).all(axis=1) & np.isfinite(self.sim.qvel).all(axis=1)
        if not finite.all():
            raise FloatingPointError(f"Nonfinite MuJoCo state in environments {np.flatnonzero(~finite).tolist()}")
        # Failure and time limit occurring together are true terminations.
        self.time_out_buf = (self.episode_length_buf.numpy() >= self.max_episode_length) & ~self.terminated_buf
        self.reset_buf = self.terminated_buf | self.time_out_buf

    def compute_reward(self):
        self.rew_buf.fill(0)
        for name, function in self.reward_functions.items():
            value = function() * self.reward_scales[name] * self.dt
            self.rew_buf += value
            self.episode_sums[name] += value
        if not np.isfinite(self.rew_buf).all():
            raise FloatingPointError("Nonfinite reward")

    def _observation_terms(self, critic):
        terms = {}
        if critic:
            terms["lin_vel"] = self.base_lin_vel
        terms.update(
            ang_vel=self.base_ang_vel * 0.2,
            gravity=self.projected_gravity,
            commands=self.commands,
            joint_pos=self.dof_pos - self.default_dof_pos,
            joint_vel=self.dof_vel * 0.05,
        )
        terms["actions"] = self.actions
        return terms

    def compute_observations(self, ids=None):
        # The usual full-batch step can use views instead of advanced-index
        # copies of every observation term and history. Partial resets still
        # update only the selected rows, without advancing surviving episodes.
        rows = slice(None) if ids is None else ids
        noise_scales = {
            "ang_vel": self.cfg.noise.ang_vel * 0.2,
            "gravity": self.cfg.noise.gravity,
            "joint_pos": self.cfg.noise.joint_pos,
            "joint_vel": self.cfg.noise.joint_vel * 0.05,
        } if self.cfg.noise.add_noise else {}
        data = {}
        for group in ("policy", "critic"):
            terms = self._observation_terms(critic=group == "critic")
            flattened = []
            for name, value in terms.items():
                value = value[rows]
                if group == "policy":
                    noise = noise_scales.get(name, 0.0)
                    if noise:
                        # A full-batch value may alias simulator/task state.
                        value = value + self.rng.uniform(-noise, noise, value.shape)
                key = (group, name)
                if key not in self._history:
                    self._history[key] = np.zeros(
                        (self.num_envs, self.cfg.env.history_length, value.shape[-1]), dtype=np.float32
                    )
                history = self._history[key]
                history[rows, :-1] = history[rows, 1:]
                if ids is None:
                    np.clip(value, -100.0, 100.0, out=history[:, -1])
                else:
                    history[ids, -1] = np.clip(value, -100.0, 100.0)
                # Isaac Lab flattens history within each term, oldest first.
                flattened.append(history.reshape(self.num_envs, -1))
            # Keep the export as an independent float32 snapshot. CPU replay
            # retains the previous observation across step(), and .to("cpu")
            # need not copy; returning a reusable history buffer corrupts it.
            data[group] = torch.from_numpy(np.concatenate(flattened, axis=-1))
        self.obs_buf = TensorDict(data, batch_size=[self.num_envs])

    def get_observations(self):
        return self.obs_buf

    def reset(self):
        self.reset_idx(np.arange(self.num_envs))
        self._refresh_state()
        self.compute_observations()
        return self.get_observations()

    def reset_idx(self, ids):
        ids = np.asarray(ids, dtype=int)
        if not ids.size:
            return
        self.sim.reset(ids)
        self.sim.qpos[ids, :3] = self.cfg.init_state.pos
        self.sim.qpos[ids, :2] += self.rng.uniform(-0.5, 0.5, (len(ids), 2))
        yaw = self.rng.uniform(-np.pi, np.pi, len(ids))
        self.sim.qpos[ids, 3:7] = np.stack(
            (np.cos(yaw / 2), np.zeros(len(ids)), np.zeros(len(ids)), np.sin(yaw / 2)), axis=-1
        )
        self.sim.qpos[np.ix_(ids, self.sim.qpos_ids)] = self.default_dof_pos
        self.sim.qvel[np.ix_(ids, self.sim.qvel_ids)] = self.rng.uniform(
            *self.cfg.init_state.joint_velocity_range, (len(ids), self.num_dof)
        )
        self.sim.qvel[np.ix_(ids, self.sim.qvel_ids[self.held_dof_indices])] = 0
        self.actions[ids] = 0
        self.last_actions[ids] = 0
        self.torques[ids] = 0
        self.last_dof_vel[ids] = self.sim.qvel[np.ix_(ids, self.sim.qvel_ids)]
        self.episode_length_buf[ids] = 0
        self._resample_commands(ids)
        self._next_push[ids] = self.rng.uniform(*self.cfg.domain_rand.push_interval_s, len(ids))
        self._contact_history[:, ids] = 0
        for buf in (
            self.feet_air_time,
            self.feet_contact_time,
            self.last_air_time,
            self.last_contact_time,
            self._feet_contact,
            self._first_contact,
            self._touchdown_air_time,
        ):
            buf[ids] = 0
        for history in self._history.values():
            history[ids] = 0
        for values in self.episode_sums.values():
            values[ids] = 0
        self.sim.forward(ids)

    def _resample_commands(self, ids):
        if len(ids):
            self.commands[ids] = self.rng.uniform(self.command_ranges[:, 0], self.command_ranges[:, 1], (len(ids), 3))
            standing = self.rng.random(len(ids)) < self.cfg.commands.standing_fraction
            self.commands[ids[standing]] = 0.0

    def _push_robots(self):
        if not self.cfg.domain_rand.push_robots:
            return
        self._next_push -= self.dt
        ids = np.flatnonzero(self._next_push <= 0)
        if ids.size:
            limit = self.cfg.domain_rand.max_push_vel_xy
            self.sim.qvel[ids, :2] = self.rng.uniform(-limit, limit, (len(ids), 2))
            self._next_push[ids] = self.rng.uniform(*self.cfg.domain_rand.push_interval_s, len(ids))

    def _episode_metrics(self, ids):
        lengths = self.episode_length_buf.numpy()[ids] * self.dt
        tilt = np.arccos(np.clip(-self.projected_gravity[ids, 2], -1.0, 1.0))
        return {
            **{
                f"Reward/{name}": float(values[ids].mean() / self.cfg.env.episode_length_s)
                for name, values in self.episode_sums.items()
            },
            "Episode/length_s": float(lengths.mean()),
            # Completed-episode reach rates, including the terminal tick.
            "Episode/reached_2s_fraction": float((lengths >= 2.0 - 1e-12).mean()),
            "Episode/reached_5s_fraction": float((lengths >= 5.0 - 1e-12).mean()),
            # Nonexclusive causes, measured before automatic reset.
            "Episode/tilt_fraction": float((tilt > self.cfg.termination.max_tilt).mean()),
            "Episode/low_height_fraction": float((self.base_height[ids] < self.cfg.termination.min_height).mean()),
            "Episode/base_contact_fraction": float((
                (self._contact_peak[ids, self._root_contact_index] > self.cfg.rewards.contact_threshold)
                & self.cfg.termination.base_contact
            ).mean()),
            "Episode/reward_per_second": float(
                sum(values[ids].sum() for values in self.episode_sums.values())
                / max(self.episode_length_buf.numpy()[ids].sum() * self.dt, self.dt)
            ),
            "Episode/time_out_fraction": float(self.time_out_buf[ids].mean()),
            "Curriculum/max_command_x": float(self.command_ranges[0, 1]),
        }

    def training_state_dict(self):
        state = {"command_ranges": self.command_ranges.tolist(), "common_step_counter": self.common_step_counter}
        return state

    def load_training_state_dict(self, state):
        ranges = np.asarray(state["command_ranges"], dtype=float)
        if ranges.shape != (3, 2) or not np.isfinite(ranges).all() or np.any(ranges[:, 0] > ranges[:, 1]):
            raise ValueError("Invalid command curriculum state")
        self.command_ranges[:] = ranges
        self.common_step_counter = int(state["common_step_counter"])
        self._resample_commands(np.arange(self.num_envs))
        self.compute_observations()

    def close(self):
        if not self._closed:
            if self._viewer is not None:
                self._viewer.close()
            self.sim.close()
            self._closed = True

    # Reward functions return unweighted per-environment values; dt is applied once.

    def _reward_tracking_ang_vel(self):
        return np.exp(-np.square(self.commands[:, 2] - self.base_ang_vel[:, 2]) / self.cfg.rewards.tracking_sigma)

    def _reward_alive(self):
        return (~self.terminated_buf).astype(float)

    def _reward_ang_vel_xy(self):
        return np.square(self.base_ang_vel[:, :2]).sum(axis=1)

    def _reward_dof_vel(self):
        return np.square(self.dof_vel).sum(axis=1)

    def _reward_dof_acc(self):
        return np.square((self.dof_vel - self.last_dof_vel) / self.dt).sum(axis=1)

    def _reward_torques(self):
        return np.square(self.torques).sum(axis=1)

    def _reward_base_height(self):
        return np.square(self.base_height - self.cfg.rewards.base_height_target)

    def _reward_feet_slide(self):
        contact = self._contact_peak[:, self.feet_indices] > self.cfg.rewards.contact_threshold
        return (np.linalg.norm(self.feet_vel[:, :, :2], axis=-1) * contact).sum(axis=1)
