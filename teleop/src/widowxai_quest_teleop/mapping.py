from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import resolve_project_path
from .math3d import (
    clip_norm,
    matrix_to_quat,
    matrix_to_rotation_vector,
    quat_to_matrix,
    rotation_vector_to_matrix,
)
from .types import Pose


class ClutchPoseMapper:
    """Incremental, clutch-relative Quest-to-WidowXAI pose mapping.

    Vectors are columns. Translation uses
    ``p_arm = task_position_signs * (M_position @ p_quest)``.
    Controller world-frame rotation deltas are mapped by matrix conjugation:
    ``dR_arm = M_rotation @ dR_quest @ M_rotation.T``. The calibrated task
    rotation signs are applied to the resulting arm-frame rotation vector.
    """

    def __init__(
        self,
        calibration_path: str | Path,
        *,
        translation_scale: float = 1.0,
        rotation_scale: float = 1.0,
        position_reach_limit_m: float | None = 0.10,
        rotation_reach_limit_rad: float | None = 0.45,
    ) -> None:
        with resolve_project_path(calibration_path).open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        self.name = str(calibration["name"])
        self.calibrated_position_matrix = np.asarray(calibration["position_matrix"], dtype=float).reshape(3, 3)
        self.orientation_enabled = bool(calibration.get("orientation_enabled", True))
        rotation_key = "rotation_matrix" if "rotation_matrix" in calibration else "rotation_matrix_unconfirmed"
        signs_key = "task_rotation_signs" if "task_rotation_signs" in calibration else "task_rotation_signs_unconfirmed"
        self.calibrated_rotation_matrix = np.asarray(calibration[rotation_key], dtype=float).reshape(3, 3)
        self.position_matrix = self.calibrated_position_matrix.copy()
        self.rotation_matrix = self.calibrated_rotation_matrix.copy()
        self.position_signs = np.asarray(
            calibration.get("task_position_signs", [1.0, 1.0, 1.0]),
            dtype=float,
        ).reshape(3)
        self.rotation_signs = np.asarray(calibration[signs_key], dtype=float).reshape(3)
        if (
            np.any(~np.isfinite(self.position_signs))
            or not np.allclose(np.abs(self.position_signs), 1.0, atol=1e-12)
        ):
            raise ValueError("task_position_signs must contain three +1/-1 values")
        self.translation_scale = float(translation_scale)
        self.rotation_scale = float(rotation_scale)
        self.position_reach_limit_m = position_reach_limit_m
        self.rotation_reach_limit_rad = rotation_reach_limit_rad
        self.engaged = False
        self.reanchor_generation = 0
        self._previous_controller: Pose | None = None
        self._target: Pose | None = None
        self._pivot: np.ndarray | None = None
        self.engage_head_yaw_rad: float | None = None

    @property
    def held_target(self) -> Pose | None:
        return self._target

    @staticmethod
    def _rotation_y(angle: float) -> np.ndarray:
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])

    def engage(
        self,
        controller: Pose,
        robot_target: Pose,
        wrist_pivot: np.ndarray,
        *,
        head_quaternion_wxyz: np.ndarray | None = None,
    ) -> Pose:
        if head_quaternion_wxyz is None:
            self.engage_head_yaw_rad = None
            heading_correction = np.eye(3)
        else:
            w, x, y, z = np.asarray(head_quaternion_wxyz, dtype=float).reshape(4)
            self.engage_head_yaw_rad = float(
                np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))
            )
            heading_correction = self._rotation_y(-self.engage_head_yaw_rad)
        self.position_matrix = self.calibrated_position_matrix @ heading_correction
        self.rotation_matrix = self.calibrated_rotation_matrix @ heading_correction
        self._previous_controller = controller
        self._target = robot_target
        self._pivot = np.asarray(wrist_pivot, dtype=float).reshape(3).copy()
        self.engaged = True
        self.reanchor_generation += 1
        return robot_target

    def release(self) -> None:
        self.engaged = False
        self._previous_controller = None

    def update(self, controller: Pose, current_robot: Pose) -> Pose | None:
        if not self.engaged:
            return None
        if self._previous_controller is None or self._target is None or self._pivot is None:
            raise RuntimeError("mapper is engaged without anchor state")

        translation_increment = self.position_signs * (
            self.position_matrix
            @ (
                self.translation_scale
                * (controller.position - self._previous_controller.position)
            )
        )
        previous_target_rotation = quat_to_matrix(self._target.quaternion_wxyz)
        controller_now_rotation = quat_to_matrix(controller.quaternion_wxyz)
        controller_previous_rotation = quat_to_matrix(self._previous_controller.quaternion_wxyz)
        controller_increment = controller_now_rotation @ controller_previous_rotation.T

        if self.orientation_enabled:
            arm_increment = self.rotation_matrix @ controller_increment @ self.rotation_matrix.T
            arm_rotvec = matrix_to_rotation_vector(arm_increment)
            arm_rotvec = arm_rotvec * self.rotation_signs * self.rotation_scale
            arm_increment = rotation_vector_to_matrix(arm_rotvec)
        else:
            arm_increment = np.eye(3)

        proposed_rotation = arm_increment @ previous_target_rotation
        current_rotation = quat_to_matrix(current_robot.quaternion_wxyz)
        if self.rotation_reach_limit_rad:
            rotation_error = matrix_to_rotation_vector(proposed_rotation @ current_rotation.T)
            limited_error = clip_norm(rotation_error, float(self.rotation_reach_limit_rad))
            proposed_rotation = rotation_vector_to_matrix(limited_error) @ current_rotation
        effective_increment = proposed_rotation @ previous_target_rotation.T

        proposed_position = self._pivot + effective_increment @ (self._target.position - self._pivot)
        proposed_position += translation_increment
        proposed_pivot = self._pivot + translation_increment
        if self.position_reach_limit_m:
            offset = proposed_position - current_robot.position
            limited_offset = clip_norm(offset, float(self.position_reach_limit_m))
            correction = current_robot.position + limited_offset - proposed_position
            proposed_position += correction
            proposed_pivot += correction

        self._target = Pose(proposed_position, matrix_to_quat(proposed_rotation))
        self._pivot = proposed_pivot
        self._previous_controller = controller
        return self._target
