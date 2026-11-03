"""Official MuJoCo 3.17 batch API; no per-environment Python stepping."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from mujoco import batch

from legged_gym import LEGGED_GYM_RESOURCES_DIR


def joint_parameters(names, patterns, default=None):
    """Resolve regex settings in insertion order; more specific entries come last."""
    values = np.full(len(names), np.nan if default is None else default, dtype=np.float64)
    for pattern, value in patterns.items():
        mask = [re.fullmatch(pattern, name) is not None for name in names]
        if not any(mask):
            raise ValueError(f"Joint pattern matches no joints: {pattern}")
        values[mask] = value
    if not np.isfinite(values).all():
        raise ValueError("Every joint must have a finite configured value")
    return values


def load_model(cfg, *, fixed_base_height=None):
    """Add task terrain and native fixed-size sensors to the official Unitree MJCF."""
    path = Path(LEGGED_GYM_RESOURCES_DIR) / "robots" / cfg.asset.file
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str(path.parent / compiler.get("meshdir", "")))
    compiler.set("fusestatic", "false")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib.update(
        timestep=str(cfg.sim.dt), integrator="implicitfast", iterations=str(cfg.sim.iterations), cone="elliptic"
    )
    # Remove the vendor's DDS/noisy sensors and poses; the task owns policy I/O.
    for tag in ("sensor", "actuator", "keyframe"):
        for element in root.findall(tag):
            root.remove(element)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset, "texture", name="floor_checker", type="2d", builtin="checker",
        rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", width="512", height="512",
    )
    ET.SubElement(
        asset, "material", name="floor_material", texture="floor_checker",
        texuniform="true", texrepeat="1 1", reflectance="0.1",
    )
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.1",
        friction="1 0.005 0.0001",
        material="floor_material",
        condim="3",
    )
    ET.SubElement(world, "light", name="task_light", pos="0 0 4", dir="0 0 -1", directional="true")
    bodies = {b.get("name"): b for b in world.iter("body")}
    bodies[cfg.asset.root_body].set("pos", " ".join(map(str, cfg.init_state.pos)))
    if fixed_base_height is not None:
        if not np.isfinite(fixed_base_height) or fixed_base_height <= 0:
            raise ValueError("Fixed base height must be positive and finite")
        pelvis = bodies[cfg.asset.root_body]
        pelvis.set("pos", f"0 0 {fixed_base_height}")
        # MuJoCo 3.17 keeps mocap roots in their own weld group. A plain
        # static root joins the world group, re-enabling pelvis/hip collisions
        # normally filtered between adjacent links in the floating robot.
        pelvis.set("mocap", "true")
        for joint in list(pelvis):
            if joint.tag == "freejoint" or (joint.tag == "joint" and joint.get("type") == "free"):
                pelvis.remove(joint)
    for joint in world.iter("joint"):
        joint.set("damping", "0")  # Damping is applied by task PD, exactly once.
        joint.set("armature", str(cfg.control.armature))
        joint.set("frictionloss", str(cfg.control.frictionloss))
        # Vendor G1 joint force ranges differ from the training actuator limits.
        # The task's motor ctrlrange is the single source of torque saturation.
        joint.set("actuatorfrclimited", "false")
    limits = joint_parameters(cfg.asset.joint_names, cfg.control.effort_limits)
    actuators = ET.SubElement(root, "actuator")
    for name, limit in zip(cfg.asset.joint_names, limits):
        ET.SubElement(actuators, "motor", name=name, joint=name, ctrllimited="true", ctrlrange=f"{-limit} {limit}")
    sensors = ET.SubElement(root, "sensor")
    for name in bodies:
        ET.SubElement(sensors, "contact", name=f"contact_{name}", body2=name, data="force", reduce="netforce")
    for name in cfg.asset.feet:
        ET.SubElement(sensors, "framelinvel", name=f"velocity_{name}", objtype="body", objname=name)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    num_joints = len(cfg.asset.joint_names)
    base_q, base_v = (7, 6) if fixed_base_height is None else (0, 0)
    if model.nq != base_q + num_joints or model.nv != base_v + num_joints or model.nu != num_joints:
        raise ValueError("Unexpected base or hinge joint count")
    return model


class MujocoBatch:
    def __init__(self, cfg, rng):
        self.model = load_model(cfg)
        self.batch = batch.Batch(
            self.model, nsim=cfg.env.num_envs, nthread=cfg.sim.nthread, persistent=cfg.sim.persistent
        )
        self.qpos = self.batch.bind("qpos")
        self.qvel = self.batch.bind("qvel")
        self.ctrl = self.batch.bind("ctrl")
        self.xpos = self.batch.bind("xpos")
        self.sensordata = self.batch.bind("sensordata")
        self.joint_ids = np.array([self.model.joint(n).id for n in cfg.asset.joint_names])
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids]
        self.body_names = [self.model.body(i).name for i in range(1, self.model.nbody)]
        self.feet_ids = np.array([self.model.body(n).id for n in cfg.asset.feet])
        self.contact_columns = np.array(
            [
                np.arange(self.model.sensor(f"contact_{n}").adr[0], self.model.sensor(f"contact_{n}").adr[0] + 3)
                for n in self.body_names
            ]
        )
        self.velocity_columns = np.array(
            [
                np.arange(self.model.sensor(f"velocity_{n}").adr[0], self.model.sensor(f"velocity_{n}").adr[0] + 3)
                for n in cfg.asset.feet
            ]
        )
        self._randomize_model(cfg, rng)
        self.batch.reset()

    def _randomize_model(self, cfg, rng):
        if cfg.domain_rand.randomize_friction:
            friction = self.batch.expand("geom_friction")
            friction[:, :, 0] = rng.uniform(*cfg.domain_rand.friction_range, (cfg.env.num_envs, 1))
        if cfg.domain_rand.randomize_base_mass:
            body_id = self.model.body(cfg.asset.mass_body).id
            masses = self.batch.expand("body_mass")
            inertias = self.batch.expand("body_inertia")
            original = self.model.body_mass[body_id]
            randomized = original + rng.uniform(*cfg.domain_rand.added_mass_range, cfg.env.num_envs)
            if np.any(randomized <= 0):
                raise ValueError("Base mass randomization must leave strictly positive masses")
            masses[:, body_id] = randomized
            inertias[:, body_id] *= (randomized / original)[:, None]
            self.batch.set_const()

    def step(self, torque, *, observe=False):
        self.ctrl[:] = torque
        self.batch.step(stage=mujoco.mjtStage.mjSTAGE_VEL if observe else mujoco.mjtStage.mjSTAGE_NONE)
        bad_state_warnings = [
            int(mujoco.mjtWarning.mjWARN_BADQPOS),
            int(mujoco.mjtWarning.mjWARN_BADQVEL),
            int(mujoco.mjtWarning.mjWARN_BADQACC),
        ]
        failed = np.any(self.batch.warning[:, bad_state_warnings, 1] > 0, axis=1)
        if failed.any():
            raise FloatingPointError(
                f"MuJoCo reported invalid dynamics in environments {np.flatnonzero(failed).tolist()}"
            )

    def reset(self, ids):
        self.batch.reset(ids, stage=mujoco.mjtStage.mjSTAGE_NONE)

    def forward(self, ids=None):
        self.batch.forward(ids)

    @property
    def contact_forces(self):
        return self.sensordata[:, self.contact_columns]

    @property
    def feet_velocities(self):
        return self.sensordata[:, self.velocity_columns]

    def close(self):
        # The public Batch API owns its pool through object lifetime. Release
        # exported views too: their base objects keep the native batch alive.
        self.qpos = self.qvel = self.ctrl = self.xpos = self.sensordata = None
        self.batch = None
