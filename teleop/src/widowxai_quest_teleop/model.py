from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .config import PROJECT_ROOT, resolve_project_path
from .math3d import matrix_to_quat
from .types import Pose


DEFAULT_MODEL_XML = PROJECT_ROOT / "third_party" / "trossen_arm_mujoco" / "trossen_arm_mujoco" / "assets" / "wxai" / "wxai_follower.xml"
ARM_JOINT_NAMES = tuple(f"joint_{index}" for index in range(6))
JOINT3_ORIGIN_IN_LINK3 = np.array([0.245, 0.0, 0.06], dtype=float)


class WidowXAIModel:
    """Official WidowXAI MJCF decorated for the reference teleop viewer."""

    def __init__(self, xml_path: str | Path = DEFAULT_MODEL_XML) -> None:
        self.xml_path = resolve_project_path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(
                f"WidowXAI model not found: {self.xml_path}. Run git submodule update --init --recursive."
            )

        spec = mujoco.MjSpec.from_file(str(self.xml_path))
        link3 = spec.body("link_3")
        if link3 is None:
            raise RuntimeError("official model is missing link_3")
        link3.add_site(
            name="wrist_anchor",
            pos=JOINT3_ORIGIN_IN_LINK3.tolist(),
            size=[0.008, 0.0, 0.0],
            rgba=[1.0, 0.45, 0.0, 1.0],
        )
        spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
        spec.visual.headlight.specular = [0.0, 0.0, 0.0]
        spec.worldbody.add_light(
            name="viewer_light",
            pos=[0.0, 0.0, 1.5],
            dir=[0.0, 0.0, -1.0],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0.0, 0.0, 0.05],
            rgba=[0.16, 0.20, 0.24, 1.0],
            contype=0,
            conaffinity=0,
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.ee_site_id = self._site_id("ee_site")
        self.wrist_site_id = self._site_id("wrist_anchor")
        self.joint_ids = np.array([self._joint_id(name) for name in ARM_JOINT_NAMES], dtype=int)
        self.qpos_indices = np.array([self.model.jnt_qposadr[index] for index in self.joint_ids], dtype=int)
        self.dof_indices = np.array([self.model.jnt_dofadr[index] for index in self.joint_ids], dtype=int)
        self.joint_limits = self.model.jnt_range[self.joint_ids].copy()
        self.gripper_qpos_indices = np.array(
            [
                self.model.jnt_qposadr[self._joint_id("right_carriage_joint")],
                self.model.jnt_qposadr[self._joint_id("left_carriage_joint")],
            ],
            dtype=int,
        )

    def _site_id(self, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if identifier < 0:
            raise RuntimeError(f"model site not found: {name}")
        return identifier

    def _joint_id(self, name: str) -> int:
        identifier = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if identifier < 0:
            raise RuntimeError(f"model joint not found: {name}")
        return identifier

    def forward(self, q_arm: np.ndarray) -> None:
        q = np.asarray(q_arm, dtype=float).reshape(6)
        self.data.qpos[:] = 0.0
        self.data.qpos[self.qpos_indices] = q
        mujoco.mj_fwdPosition(self.model, self.data)

    def pose(self, site: str = "ee") -> Pose:
        site_id = self.ee_site_id if site == "ee" else self.wrist_site_id
        rotation = self.data.site_xmat[site_id].reshape(3, 3).copy()
        return Pose(self.data.site_xpos[site_id].copy(), matrix_to_quat(rotation))

    def fk(self, q_arm: np.ndarray) -> tuple[Pose, Pose]:
        self.forward(q_arm)
        return self.pose("ee"), self.pose("wrist")

    def jacobian(self, site: str = "ee") -> tuple[np.ndarray, np.ndarray]:
        site_id = self.ee_site_id if site == "ee" else self.wrist_site_id
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacobian_position, jacobian_rotation, site_id)
        return jacobian_position[:, self.dof_indices].copy(), jacobian_rotation[:, self.dof_indices].copy()

    def clamp_joints(self, q_arm: np.ndarray) -> np.ndarray:
        q = np.asarray(q_arm, dtype=float).reshape(6)
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def in_self_collision(self, q_arm: np.ndarray, gripper_q: float = 0.044) -> bool:
        """Check the official model's collision geometry at one commanded pose."""

        self.data.qpos[:] = 0.0
        self.data.qpos[self.qpos_indices] = np.asarray(q_arm, dtype=float).reshape(6)
        self.data.qpos[self.gripper_qpos_indices] = float(gripper_q)
        mujoco.mj_forward(self.model, self.data)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body_1 = self.model.geom_bodyid[contact.geom1]
            body_2 = self.model.geom_bodyid[contact.geom2]
            names = {
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_1),
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_2),
            }
            # The two carriage collision boxes intentionally meet when the
            # valid gripper closes; this is not an arm self-collision.
            if names == {"carriage_right", "carriage_left"}:
                continue
            return True
        return False

    def first_self_collision_on_path(
        self,
        start_q: np.ndarray,
        end_q: np.ndarray,
        *,
        start_gripper_q: float = 0.044,
        end_gripper_q: float = 0.044,
        samples: int = 251,
    ) -> float | None:
        """Return the first colliding interpolation fraction, if any."""

        start = np.asarray(start_q, dtype=float).reshape(6)
        end = np.asarray(end_q, dtype=float).reshape(6)
        for alpha in np.linspace(0.0, 1.0, max(2, int(samples))):
            q = start + alpha * (end - start)
            gripper = float(start_gripper_q) + alpha * (
                float(end_gripper_q) - float(start_gripper_q)
            )
            if self.in_self_collision(q, gripper):
                return float(alpha)
        return None

    def set_viewer_qpos(self, data: mujoco.MjData, q_arm: np.ndarray, gripper_q: float) -> None:
        """Apply commands exactly as the reference kit's passive viewer does."""
        data.qpos[self.qpos_indices] = np.asarray(q_arm, dtype=float).reshape(6)
        data.qpos[self.gripper_qpos_indices] = float(gripper_q)
        mujoco.mj_forward(self.model, data)
