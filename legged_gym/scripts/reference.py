"""Inspect dataset/prior trajectories on a suspended G1, without a policy."""

import argparse
import math
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch

from legged_gym.envs.g1_sprint.config import SprintCfg
from legged_gym.paths import DEFAULT_PRIOR, ROOT_DIR
from legged_gym.sim.mujoco_batch import joint_parameters, load_model
from legged_gym.utils.camera import ViserCameraControls
from spectral_prior.reference import ReferenceGenerator


class ReferencePlayback:
    """Kinematic inspection or torque-limited PD with a physically fixed pelvis."""

    def __init__(self, library, prior, *, motion=None, height=1.5):
        self.cfg = SprintCfg()
        self.names = list(self.cfg.asset.joint_names)
        self.default = joint_parameters(self.names, self.cfg.init_state.default_joint_angles, 0)
        self.wrists = np.array([i for i, name in enumerate(self.names) if "_wrist_" in name])
        self.reference = ReferenceGenerator(prior, self.names, self.default, 1, self.cfg.reference.blend_speed)
        self.clips = {}
        with np.load(library, allow_pickle=False) as data:
            joints = data["joint_names"].tolist()
            for i, q in enumerate(data["joint_pos"]):
                label = f"{i + 1}: {data['speeds'][i]:.3f} m/s, {data['frequencies'][i]:.3f} Hz"
                self.clips[label] = self._clip(q, joints, float(data["fps"]))
        if not self.clips:
            raise ValueError("Reference library is empty")
        self.motion = None
        if motion is not None:
            with np.load(motion, allow_pickle=False) as data:
                self.motion = self._clip(data["joint_pos"], data["joint_names"].tolist(), float(data["fps"]))
        self.model = load_model(self.cfg, fixed_base_height=height)
        self.data = mujoco.MjData(self.model)
        self.gravity_data = mujoco.MjData(self.model)
        joints = np.array([self.model.joint(name).id for name in self.names])
        self.qids = self.model.jnt_qposadr[joints]
        self.vids = self.model.jnt_dofadr[joints]
        self.limits = self.model.jnt_range[joints]
        self.kp = joint_parameters(self.names, self.cfg.control.stiffness)
        self.kd = joint_parameters(self.names, self.cfg.control.damping)
        self.effort = joint_parameters(self.names, self.cfg.control.effort_limits)
        self.time = 0.0
        self.saturation = 0.0
        self.reset(self.default)

    def _clip(self, q, names, fps):
        q = np.asarray(q, dtype=float)
        if (
            not math.isfinite(fps) or fps <= 0 or q.ndim != 2 or len(q) < 2
            or q.shape[1] != len(names) or len(set(names)) != len(names)
            or not set(names).issubset(self.names) or not np.isfinite(q).all()
        ):
            raise ValueError("Motion requires finite joint_pos[T,J], fps > 0, and unique known joint_names")
        indices = np.array([self.names.index(name) for name in names])
        return q, indices, fps

    def target(self, source, clip, speed, hold_wrists, at=None):
        t = self.time if at is None else at
        if source == "prior":
            frequency = self.reference.mapping(torch.tensor([speed], dtype=torch.float32)).item()
            self.reference.phase.fill_((t * frequency) % 1)
            q = self.reference.evaluate([speed])[0][0].astype(float)
        else:
            frames, indices, fps = self.motion if source == "motion" else self.clips[clip]
            frame = (t * fps) % len(frames)
            i = int(frame)
            q = self.default.copy()
            q[indices] = frames[i] + (frame - i) * (frames[(i + 1) % len(frames)] - frames[i])
        if hold_wrists:
            q[self.wrists] = self.default[self.wrists]
        return q

    def reset(self, target):
        self.time = 0.0
        self.saturation = 0.0
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qids] = target
        mujoco.mj_forward(self.model, self.data)

    def advance(self, source, clip, speed, hold_wrists, mode, rate, steps, *, gravity_comp=False):
        for _ in range(steps):
            q = self.target(source, clip, speed, hold_wrists)
            if mode == "pd":
                target = np.clip(q, self.limits[:, 0], self.limits[:, 1])
                torque = self.kp * (target - self.data.qpos[self.qids]) - self.kd * self.data.qvel[self.vids]
                if gravity_comp:
                    # Bias at zero velocity contains gravity only, without
                    # Coriolis/centrifugal terms. Compensation shares motor limits.
                    self.gravity_data.qpos[:] = self.data.qpos
                    self.gravity_data.qvel[:] = 0
                    self.gravity_data.mocap_pos[:] = self.data.mocap_pos
                    self.gravity_data.mocap_quat[:] = self.data.mocap_quat
                    mujoco.mj_forward(self.model, self.gravity_data)
                    torque += self.gravity_data.qfrc_bias[self.vids]
                self.saturation = float(np.mean(np.abs(torque) >= self.effort))
                self.data.ctrl[:] = np.clip(torque, -self.effort, self.effort)
                mujoco.mj_step(self.model, self.data)
                if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                    raise FloatingPointError("Nonfinite suspended PD state")
                if self.data.warning.number.any():
                    raise FloatingPointError("MuJoCo reported a suspended PD dynamics warning")
            self.time += self.model.opt.timestep * rate
        q = self.target(source, clip, speed, hold_wrists)
        if mode == "kinematic":
            self.data.qpos[self.qids] = q
            self.data.qvel[:] = 0
            self.saturation = 0.0
        mujoco.mj_forward(self.model, self.data)
        return q


class ReferenceViewer:
    def __init__(self, playback, args):
        import viser
        from mjviser import ViserMujocoScene

        self.playback = playback
        self.server = viser.ViserServer(host=args.host, port=args.port, label="SPRINT · Reference", verbose=False)
        self.server.scene.set_up_direction("+z")
        visual_bodies = playback.model.geom_bodyid[playback.model.geom_group == 1]
        collisions = (playback.model.geom_contype != 0) & np.isin(playback.model.geom_bodyid, visual_bodies)
        playback.model.geom_group[collisions] = 3
        self.scene = ViserMujocoScene(self.server, playback.model, num_envs=1)
        self.scene.camera_tracking_enabled = False
        self.scene.site_groups_visible = [False] * 6
        self.camera = ViserCameraControls(self.server, (0, 0, args.height - 0.3))
        gui = self.server.gui
        gui.add_markdown(
            "## Suspended reference playback\nNo policy or residual actions. Pelvis position and orientation are fixed."
        )
        sources = ["library", "prior"] + (["motion"] if playback.motion is not None else [])
        self.source = gui.add_dropdown("Source", sources, initial_value=args.source)
        self.clip = gui.add_dropdown("Library clip", list(playback.clips), initial_value=list(playback.clips)[args.clip - 1])
        self.speed = gui.add_slider("Prior speed (m/s)", min=0, max=6, step=0.1, initial_value=args.speed)
        self.mode = gui.add_dropdown("Playback", ["kinematic", "pd"], initial_value=args.mode)
        self.gravity_comp = gui.add_checkbox(
            "Gravity compensation (diagnostic)", initial_value=getattr(args, "gravity_comp", False),
            hint="Adds gravity torque before motor clipping. Viewer only; training is unchanged.",
        )
        self.hold = gui.add_checkbox("Hold wrists", initial_value=True)
        self.rate = gui.add_slider("Playback rate", min=0.1, max=2.0, step=0.1, initial_value=1.0)
        self.paused = gui.add_checkbox("Pause", initial_value=False)
        self.restart = gui.add_button("Restart")
        self.joint = gui.add_dropdown("Inspect joint", playback.names, initial_value="left_elbow_joint")
        self.status = gui.add_markdown("")
        empty = np.zeros(1)
        self.plot = gui.add_uplot(
            (empty, empty, empty),
            ({"label": "Motion time (s)"}, {"label": "Reference", "stroke": "#60a5fa"},
             {"label": "Actual", "stroke": "#f59e0b"}),
            title="Joint angle (degrees)", scales={"x": {"time": False}}, height=180,
        )
        gui.add_markdown(
            "**library:** curated 10-joint training data; other joints use the default pose.\n\n"
            "**prior:** generated training reference, including the standing blend.\n\n"
            "**motion:** optional GMR NPZ joint data; root motion is ignored. Turn off Hold wrists to inspect source wrists.\n\n"
            "**kinematic:** exact poses, without dynamics or joint-limit clipping. "
            "**pd:** gravity, torque limits and contacts remain active; targets are clipped to joint limits. "
            "Optional gravity compensation is a viewer diagnostic, not the training controller. "
            "This is a tracking check, not a locomotion test. Clips loop and may jump at the boundary."
        )
        self._restart_requested = False
        self._selection = None
        self._joint = None
        self.history = deque(maxlen=300)

        @self.restart.on_click
        def restart(_):
            self._restart_requested = True

    @property
    def url(self):
        host = self.server.get_host()
        return f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{self.server.get_port()}"

    def tick(self, steps=6):
        p = self.playback
        selection = (
            self.source.value, self.clip.value, self.speed.value, self.hold.value,
            self.mode.value, self.gravity_comp.value,
        )
        source, clip, speed, hold, mode, gravity_comp = selection
        self.clip.disabled = source != "library"
        self.speed.disabled = source != "prior"
        self.gravity_comp.disabled = mode != "pd"
        if selection != self._selection or self._restart_requested:
            q = p.target(source, clip, speed, hold, at=0)
            p.reset(np.clip(q, p.limits[:, 0], p.limits[:, 1]) if mode == "pd" else q)
            self.history.clear()
            self._selection = selection
            self._restart_requested = False
        q = p.advance(
            source, clip, speed, hold, mode, self.rate.value, 0 if self.paused.value else steps,
            gravity_comp=gravity_comp,
        )
        self.scene.update_from_mjdata(p.data)
        index = p.names.index(self.joint.value)
        if self._joint != index:
            self.history.clear()
            self._joint = index
        actual = p.data.qpos[p.qids]
        tracked = (
            p.reference.indices if source == "prior"
            else (p.motion[1] if source == "motion" else p.clips[clip][1])
        )
        if not self.history or self.history[-1][0] != p.time:
            self.history.append((p.time, np.rad2deg(q[index]), np.rad2deg(actual[index])))
        self.plot.data = tuple(np.array(self.history).T)
        outside = np.flatnonzero((q < p.limits[:, 0]) | (q > p.limits[:, 1]))
        violations = ", ".join(p.names[i] for i in outside) or "none"
        self.status.content = (
            f"Time: **{p.time:.2f} s** · {source} / {mode}\n\n"
            f"Gravity compensation: **{'on' if gravity_comp and mode == 'pd' else 'off'}**\n\n"
            f"Selected joint: reference **{np.rad2deg(q[index]):.1f}°**, actual **{np.rad2deg(actual[index]):.1f}°**\n\n"
            f"Reference-joint mean error ({len(tracked)} joints): "
            f"**{np.rad2deg(np.abs(q[tracked] - actual[tracked])).mean():.2f}°**\n\n"
            f"All-joint mean error: **{np.rad2deg(np.abs(q - actual)).mean():.2f}°** · "
            f"Motor saturation: **{p.saturation:.0%}**\n\nReference outside joint limits: **{violations}**"
        )

    def close(self):
        self.server.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default=str(ROOT_DIR / "outputs/reference/library.npz"))
    parser.add_argument("--prior", default=str(DEFAULT_PRIOR))
    parser.add_argument("--motion", type=Path, help="Optional GMR NPZ with joint_pos, joint_names and fps")
    parser.add_argument("--source", choices=["library", "prior", "motion"], default=None)
    parser.add_argument("--clip", type=int, default=1, help="Library clip, numbered from 1")
    parser.add_argument("--speed", type=float, default=1.0, help="Prior command speed, 0–6 m/s")
    parser.add_argument("--mode", choices=["kinematic", "pd"], default="kinematic")
    parser.add_argument("--gravity-comp", action="store_true", help="Viewer-only gravity torque compensation in PD mode")
    parser.add_argument("--height", type=float, default=1.5, help="Fixed pelvis height in meters")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    args.source = args.source or ("motion" if args.motion else "library")
    if args.source == "motion" and args.motion is None:
        parser.error("--source motion requires --motion /path/to/motion.npz")
    if not math.isfinite(args.speed) or not 0 <= args.speed <= 6:
        parser.error("--speed must be between 0 and 6")
    torch.set_num_threads(1)
    playback = ReferencePlayback(args.library, args.prior, motion=args.motion, height=args.height)
    if not 1 <= args.clip <= len(playback.clips):
        parser.error(f"--clip must be between 1 and {len(playback.clips)}")
    viewer = ReferenceViewer(playback, args)
    print(f"Reference viewer: {viewer.url}", flush=True)
    print("Pelvis fixed in midair. No policy checkpoint required. Stop with Ctrl+C.", flush=True)
    try:
        while True:
            start = time.monotonic()
            viewer.tick()
            time.sleep(max(0, 6 * playback.model.opt.timestep - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
