from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math3d import orientation_error_world, quat_to_matrix
from .model import WidowXAIModel
from .types import Pose


@dataclass(frozen=True)
class IKDiagnostics:
    position_residual_m: float
    orientation_residual_rad: float
    position_manipulability: float
    rotation_manipulability: float
    position_damping: float
    rotation_damping: float
    step_norm_rad: float
    iterations: int
    minimum_joint_limit_margin_rad: float
    joint_limit_clipped: bool
    wrist_parked: bool
    status: str


class DecoupledIK:
    """Sequential 3+3 differential IK for the WidowXAI.

    Joints 0-2 move the joint-3 wrist anchor. Joints 3-5 then track tool
    orientation. Each subproblem has independent manipulability-adaptive
    damping. The combined step receives the reference kit's per-joint cap.
    """

    def __init__(
        self,
        model: WidowXAIModel,
        *,
        rest_q: np.ndarray,
        lambda_position: float = 0.05,
        lambda_position_extra: float = 0.15,
        position_manipulability_threshold: float = 0.05,
        rest_bias: float = 0.02,
        lambda_rotation: float = 0.05,
        lambda_rotation_extra: float = 0.40,
        rotation_manipulability_threshold: float = 0.50,
        orientation_hold_error_rad: float = 2.20,
        max_dq_per_joint: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.rest_q = np.asarray(rest_q, dtype=float).reshape(6).copy()
        self.lambda_position = float(lambda_position)
        self.lambda_position_extra = float(lambda_position_extra)
        self.position_manipulability_threshold = float(position_manipulability_threshold)
        self.rest_bias = float(rest_bias)
        self.lambda_rotation = float(lambda_rotation)
        self.lambda_rotation_extra = float(lambda_rotation_extra)
        self.rotation_manipulability_threshold = float(rotation_manipulability_threshold)
        self.orientation_hold_error_rad = float(orientation_hold_error_rad)
        self.max_dq_per_joint = (
            np.full(6, np.inf)
            if max_dq_per_joint is None
            else np.asarray(max_dq_per_joint, dtype=float).reshape(6).copy()
        )
        self.joint_limits = self.model.joint_limits.copy()

    @classmethod
    def from_config(cls, model: WidowXAIModel, config: dict) -> "DecoupledIK":
        ik = config["ik"]
        rest_q = config["model"]["simulation_start_q_rad"]
        return cls(
            model,
            rest_q=np.asarray(rest_q, dtype=float),
            lambda_position=ik["lambda_position"],
            lambda_position_extra=ik["lambda_position_extra"],
            position_manipulability_threshold=ik["position_manipulability_threshold"],
            rest_bias=ik["rest_bias"],
            lambda_rotation=ik["lambda_rotation"],
            lambda_rotation_extra=ik["lambda_rotation_extra"],
            rotation_manipulability_threshold=ik["rotation_manipulability_threshold"],
            orientation_hold_error_rad=ik["orientation_hold_error_rad"],
            max_dq_per_joint=np.asarray(ik["max_dq_per_joint_rad"], dtype=float),
        )

    def _clamp_joints(self, q_arm: np.ndarray) -> np.ndarray:
        q = np.asarray(q_arm, dtype=float).reshape(6)
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def set_joint_limits(self, joint_limits: np.ndarray) -> None:
        """Use deployment limits, including any hardware safety margin."""

        limits = np.asarray(joint_limits, dtype=float).reshape(6, 2).copy()
        if not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("invalid IK joint limits")
        self.joint_limits = limits

    @staticmethod
    def _adaptive_damping(base: float, extra: float, manipulability: float, threshold: float) -> float:
        ramp = max(0.0, 1.0 - manipulability / max(threshold, 1e-12))
        return float(np.sqrt(base * base + extra * extra * ramp * ramp))

    @staticmethod
    def _dls(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
        task_matrix = jacobian @ jacobian.T + damping * damping * np.eye(jacobian.shape[0])
        return jacobian.T @ np.linalg.solve(task_matrix, error)

    def solve(
        self,
        target: Pose,
        q_seed: np.ndarray,
    ) -> tuple[np.ndarray, IKDiagnostics]:
        q_initial = self._clamp_joints(q_seed)
        target_rotation = quat_to_matrix(target.quaternion_wxyz)
        limit_clipped = False
        wrist_parked = False
        status = "ok"
        position_manipulability = 0.0
        rotation_manipulability = 0.0
        position_damping = self.lambda_position
        rotation_damping = self.lambda_rotation

        try:
            current_ee, current_wrist = self.model.fk(q_initial)
            current_rotation = quat_to_matrix(current_ee.quaternion_wxyz)
            ee_to_wrist_local = current_rotation.T @ (current_wrist.position - current_ee.position)
            target_wrist = target.position + target_rotation @ ee_to_wrist_local
            position_error = target_wrist - current_wrist.position

            jacobian_position, _ = self.model.jacobian("wrist")
            arm_jacobian = jacobian_position[:, :3]
            position_manipulability = abs(float(np.linalg.det(arm_jacobian)))
            position_damping = self._adaptive_damping(
                self.lambda_position,
                self.lambda_position_extra,
                position_manipulability,
                self.position_manipulability_threshold,
            )
            mu2 = self.rest_bias * self.rest_bias
            normal = arm_jacobian.T @ arm_jacobian
            normal += (position_damping * position_damping + mu2) * np.eye(3)
            rhs = arm_jacobian.T @ position_error + mu2 * (self.rest_q[:3] - q_initial[:3])
            q_after_arm = q_initial.copy()
            q_after_arm[:3] += np.linalg.solve(normal, rhs)

            current_ee, _ = self.model.fk(q_after_arm)
            rotation_error = orientation_error_world(target.quaternion_wxyz, current_ee.quaternion_wxyz)
            wrist_parked = float(np.linalg.norm(rotation_error)) > self.orientation_hold_error_rad
            _, jacobian_rotation = self.model.jacobian("ee")
            wrist_jacobian = jacobian_rotation[:, 3:6]
            rotation_manipulability = abs(float(np.linalg.det(wrist_jacobian)))
            rotation_damping = self._adaptive_damping(
                self.lambda_rotation,
                self.lambda_rotation_extra,
                rotation_manipulability,
                self.rotation_manipulability_threshold,
            )
            wrist_step = np.zeros(3)
            if wrist_parked:
                status = "orientation_parked"
            else:
                wrist_step = self._dls(wrist_jacobian, rotation_error, rotation_damping)

            q_unclamped = q_after_arm.copy()
            q_unclamped[3:6] += wrist_step
            q_reachable = self._clamp_joints(q_unclamped)
            limit_clipped = not np.allclose(q_reachable, q_unclamped)
            dq = np.clip(
                q_reachable - q_initial,
                -self.max_dq_per_joint,
                self.max_dq_per_joint,
            )
            q = self._clamp_joints(q_initial + dq)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            q = q_initial.copy()
            status = "numerical_failure"

        if not np.all(np.isfinite(q)):
            q = q_initial.copy()
            status = "nonfinite_rejected"

        current_ee, current_wrist = self.model.fk(q)
        current_rotation = quat_to_matrix(current_ee.quaternion_wxyz)
        ee_to_wrist_local = current_rotation.T @ (current_wrist.position - current_ee.position)
        final_target_wrist = target.position + target_rotation @ ee_to_wrist_local
        position_residual = float(np.linalg.norm(final_target_wrist - current_wrist.position))
        orientation_residual = float(
            np.linalg.norm(orientation_error_world(target.quaternion_wxyz, current_ee.quaternion_wxyz))
        )
        margins = np.minimum(q - self.joint_limits[:, 0], self.joint_limits[:, 1] - q)
        diagnostics = IKDiagnostics(
            position_residual_m=position_residual,
            orientation_residual_rad=orientation_residual,
            position_manipulability=position_manipulability,
            rotation_manipulability=rotation_manipulability,
            position_damping=position_damping,
            rotation_damping=rotation_damping,
            step_norm_rad=float(np.linalg.norm(q - q_initial)),
            iterations=1,
            minimum_joint_limit_margin_rad=float(np.min(margins)),
            joint_limit_clipped=limit_clipped,
            wrist_parked=wrist_parked,
            status=status,
        )
        return q, diagnostics
