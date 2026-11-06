from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .math3d import normalize_quat
from .math3d import matrix_to_rotation_vector, quat_to_matrix
from .types import Pose


def pose_filter_alphas(control: Mapping[str, object]) -> tuple[float, float]:
    """Return translation/rotation weights, falling back to the legacy shared weight."""
    shared = float(control.get("pose_filter_alpha", 1.0))
    translation = float(control.get("pose_filter_translation_alpha", shared))
    rotation = float(control.get("pose_filter_rotation_alpha", shared))
    return translation, rotation


def pose_ema(
    previous: Pose | None,
    current: Pose,
    translation_alpha: float,
    rotation_alpha: float | None = None,
) -> Pose:
    """Reference-kit EMA with independently tunable position and quaternion weights."""
    if previous is None:
        return current
    translation_weight = float(np.clip(translation_alpha, 0.0, 1.0))
    rotation_weight = float(
        np.clip(translation_alpha if rotation_alpha is None else rotation_alpha, 0.0, 1.0)
    )
    position = (
        (1.0 - translation_weight) * previous.position
        + translation_weight * current.position
    )
    quaternion = current.quaternion_wxyz
    if float(np.dot(previous.quaternion_wxyz, quaternion)) < 0.0:
        quaternion = -quaternion
    quaternion = normalize_quat(
        (1.0 - rotation_weight) * previous.quaternion_wxyz
        + rotation_weight * quaternion
    )
    return Pose(position, quaternion)


def _low_pass_alpha(cutoff_hz: float, dt_s: float) -> float:
    return float(1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff_hz * dt_s)))


class ControllerPoseFilter:
    """Reference EMA with optional One-Euro-style adaptive rotation bandwidth."""

    def __init__(self, control: Mapping[str, object]) -> None:
        self.translation_alpha, self.fixed_rotation_alpha = pose_filter_alphas(control)
        adaptive_raw = control.get("adaptive_rotation_filter", {})
        if adaptive_raw is None:
            adaptive_raw = {}
        if not isinstance(adaptive_raw, Mapping):
            raise ValueError("control.adaptive_rotation_filter must be a mapping")
        self.adaptive_enabled = bool(adaptive_raw.get("enabled", False))
        self.minimum_cutoff_hz = float(adaptive_raw.get("minimum_cutoff_hz", 1.5))
        self.speed_coefficient = float(adaptive_raw.get("speed_coefficient", 20.0))
        self.speed_exponent = float(adaptive_raw.get("speed_exponent", 1.0))
        self.derivative_cutoff_hz = float(adaptive_raw.get("derivative_cutoff_hz", 2.0))
        self.maximum_cutoff_hz = float(adaptive_raw.get("maximum_cutoff_hz", 35.0))
        for name, value in (
            ("pose_filter_translation_alpha", self.translation_alpha),
            ("pose_filter_rotation_alpha", self.fixed_rotation_alpha),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"control.{name} must be finite and within [0, 1]")
        for name, value in (
            ("minimum_cutoff_hz", self.minimum_cutoff_hz),
            ("derivative_cutoff_hz", self.derivative_cutoff_hz),
            ("maximum_cutoff_hz", self.maximum_cutoff_hz),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"control.adaptive_rotation_filter.{name} must be positive")
        if not np.isfinite(self.speed_coefficient) or self.speed_coefficient < 0.0:
            raise ValueError(
                "control.adaptive_rotation_filter.speed_coefficient must be nonnegative"
            )
        if not np.isfinite(self.speed_exponent) or self.speed_exponent <= 0.0:
            raise ValueError(
                "control.adaptive_rotation_filter.speed_exponent must be positive"
            )
        if self.maximum_cutoff_hz < self.minimum_cutoff_hz:
            raise ValueError(
                "control.adaptive_rotation_filter.maximum_cutoff_hz must not be below minimum"
            )
        self._filtered: Pose | None = None
        self._previous_raw_quaternion: np.ndarray | None = None
        self._previous_timestamp_s: float | None = None
        self._filtered_angular_speed = 0.0
        self.rotation_alpha = 1.0
        self.rotation_cutoff_hz = self.maximum_cutoff_hz

    def reset(self) -> None:
        self._filtered = None
        self._previous_raw_quaternion = None
        self._previous_timestamp_s = None
        self._filtered_angular_speed = 0.0
        self.rotation_alpha = 1.0
        self.rotation_cutoff_hz = self.maximum_cutoff_hz

    def update(self, current: Pose, timestamp_s: float) -> Pose:
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("controller pose timestamp must be finite")
        if self._filtered is None or self._previous_raw_quaternion is None:
            self._filtered = current
            self._previous_raw_quaternion = current.quaternion_wxyz.copy()
            self._previous_timestamp_s = timestamp
            return current

        previous_timestamp = self._previous_timestamp_s
        dt_s = 1.0 / 90.0 if previous_timestamp is None else timestamp - previous_timestamp
        dt_s = float(np.clip(dt_s, 0.001, 0.050))
        rotation_alpha = self.fixed_rotation_alpha
        if self.adaptive_enabled:
            raw_increment = (
                quat_to_matrix(current.quaternion_wxyz)
                @ quat_to_matrix(self._previous_raw_quaternion).T
            )
            raw_speed = float(np.linalg.norm(matrix_to_rotation_vector(raw_increment)) / dt_s)
            derivative_alpha = _low_pass_alpha(self.derivative_cutoff_hz, dt_s)
            self._filtered_angular_speed = (
                (1.0 - derivative_alpha) * self._filtered_angular_speed
                + derivative_alpha * raw_speed
            )
            self.rotation_cutoff_hz = float(
                np.clip(
                    self.minimum_cutoff_hz
                    + self.speed_coefficient
                    * self._filtered_angular_speed**self.speed_exponent,
                    self.minimum_cutoff_hz,
                    self.maximum_cutoff_hz,
                )
            )
            rotation_alpha = _low_pass_alpha(self.rotation_cutoff_hz, dt_s)

        self.rotation_alpha = rotation_alpha
        self._filtered = pose_ema(
            self._filtered,
            current,
            self.translation_alpha,
            rotation_alpha,
        )
        self._previous_raw_quaternion = current.quaternion_wxyz.copy()
        self._previous_timestamp_s = timestamp
        return self._filtered
