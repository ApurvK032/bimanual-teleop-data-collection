from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decoupled_ik import DecoupledIK, IKDiagnostics
from .math3d import matrix_to_quat, quat_to_matrix, rotation_vector_to_matrix
from .model import WidowXAIModel
from .types import Pose


@dataclass(frozen=True)
class DiagnosticMotion:
    key: str
    label: str
    kind: str
    axis_world: np.ndarray
    positive_label: str
    negative_label: str

    def __post_init__(self) -> None:
        axis = np.asarray(self.axis_world, dtype=float).reshape(3).copy()
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError("diagnostic motion axis must be finite and nonzero")
        if self.kind not in ("translation", "rotation"):
            raise ValueError("diagnostic motion kind must be translation or rotation")
        object.__setattr__(self, "axis_world", axis / norm)


# These axes intentionally match the guided six-axis Quest calibration.
DIAGNOSTIC_MOTIONS = (
    DiagnosticMotion(
        "left_right",
        "left / right",
        "translation",
        np.array([0.0, -1.0, 0.0]),
        "right",
        "left",
    ),
    DiagnosticMotion(
        "up_down",
        "up / down",
        "translation",
        np.array([0.0, 0.0, 1.0]),
        "up",
        "down",
    ),
    DiagnosticMotion(
        "to_and_fro",
        "to / fro",
        "translation",
        np.array([1.0, 0.0, 0.0]),
        "forward",
        "backward",
    ),
    DiagnosticMotion(
        "screw",
        "screw",
        "rotation",
        np.array([1.0, 0.0, 0.0]),
        "clockwise calibration direction",
        "counter-clockwise calibration direction",
    ),
    DiagnosticMotion(
        "nod_yes",
        "nod yes",
        "rotation",
        np.array([0.0, 1.0, 0.0]),
        "front down",
        "front up",
    ),
    DiagnosticMotion(
        "nod_no",
        "nod no",
        "rotation",
        np.array([0.0, 0.0, 1.0]),
        "front left",
        "front right",
    ),
)


@dataclass(frozen=True)
class EndpointReport:
    motion_key: str
    direction: str
    q_goal: np.ndarray
    diagnostics: IKDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "q_goal",
            np.asarray(self.q_goal, dtype=float).reshape(6).copy(),
        )


@dataclass(frozen=True)
class DiagnosticPoint:
    motion_key: str
    motion_label: str
    phase: str
    q_command: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "q_command",
            np.asarray(self.q_command, dtype=float).reshape(6).copy(),
        )


@dataclass(frozen=True)
class DiagnosticPlan:
    home_q: np.ndarray
    points: tuple[DiagnosticPoint, ...]
    endpoints: tuple[EndpointReport, ...]
    rate_hz: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "home_q",
            np.asarray(self.home_q, dtype=float).reshape(6).copy(),
        )

    @property
    def duration_s(self) -> float:
        return len(self.points) / self.rate_hz

    def command_array(self) -> np.ndarray:
        return np.vstack([point.q_command for point in self.points])


@dataclass(frozen=True)
class PlanDynamics:
    max_step_rad: np.ndarray
    max_velocity_rad_s: np.ndarray
    max_acceleration_rad_s2: np.ndarray
    max_jerk_rad_s3: np.ndarray


def minimum_snap_fraction(value: float | np.ndarray) -> float | np.ndarray:
    """C3-smooth 0-to-1 interpolation with zero endpoint velocity/accel/jerk."""

    u = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(u)) or np.any((u < 0.0) | (u > 1.0)):
        raise ValueError("minimum-snap interpolation input must be within [0, 1]")
    result = 35.0 * u**4 - 84.0 * u**5 + 70.0 * u**6 - 20.0 * u**7
    return float(result) if u.ndim == 0 else result


def _target_for_direction(
    home_pose: Pose,
    motion: DiagnosticMotion,
    sign: float,
    translation_amplitude_m: float,
    rotation_amplitude_rad: float,
) -> Pose:
    if motion.kind == "translation":
        return Pose(
            home_pose.position + sign * translation_amplitude_m * motion.axis_world,
            home_pose.quaternion_wxyz,
        )
    home_rotation = quat_to_matrix(home_pose.quaternion_wxyz)
    rotation_increment = rotation_vector_to_matrix(
        sign * rotation_amplitude_rad * motion.axis_world
    )
    return Pose(
        home_pose.position,
        matrix_to_quat(rotation_increment @ home_rotation),
    )


def _solve_endpoint(
    solver: DecoupledIK,
    target: Pose,
    home_q: np.ndarray,
    *,
    maximum_iterations: int,
    maximum_position_error_m: float,
    maximum_orientation_error_rad: float,
) -> tuple[np.ndarray, IKDiagnostics]:
    q = np.asarray(home_q, dtype=float).reshape(6).copy()
    diagnostics: IKDiagnostics | None = None
    for _ in range(maximum_iterations):
        q, diagnostics = solver.solve(target, q)
        if (
            diagnostics.status == "ok"
            and diagnostics.position_residual_m <= maximum_position_error_m
            and diagnostics.orientation_residual_rad <= maximum_orientation_error_rad
        ):
            return q, diagnostics
    assert diagnostics is not None
    raise ValueError(
        "diagnostic endpoint IK did not converge: "
        f"status={diagnostics.status}, "
        f"position_error={diagnostics.position_residual_m:.6f} m, "
        f"orientation_error={diagnostics.orientation_residual_rad:.6f} rad"
    )


def _leg_points(
    motion: DiagnosticMotion,
    phase: str,
    q_start: np.ndarray,
    q_end: np.ndarray,
    samples: int,
) -> list[DiagnosticPoint]:
    points: list[DiagnosticPoint] = []
    for index in range(1, samples + 1):
        fraction = minimum_snap_fraction(index / samples)
        q = q_start + fraction * (q_end - q_start)
        points.append(DiagnosticPoint(motion.key, motion.label, phase, q))
    return points


def build_six_axis_plan(
    model: WidowXAIModel,
    solver: DecoupledIK,
    home_q: np.ndarray,
    *,
    gripper_position_m: float,
    rate_hz: float = 90.0,
    segment_duration_s: float = 1.25,
    pause_duration_s: float = 0.35,
    translation_amplitude_m: float = 0.012,
    rotation_amplitude_rad: float = np.deg2rad(3.0),
    maximum_endpoint_position_error_m: float = 0.001,
    maximum_endpoint_orientation_error_rad: float = np.deg2rad(0.25),
    maximum_ik_iterations: int = 300,
) -> DiagnosticPlan:
    """Precompute six small Cartesian tests as deterministic joint trajectories.

    IK is used only before execution to find each endpoint. Runtime commands are
    minimum-snap joint interpolations, so Quest/network/filter jitter cannot enter
    this diagnostic path.
    """

    finite_positive = {
        "rate_hz": rate_hz,
        "segment_duration_s": segment_duration_s,
        "translation_amplitude_m": translation_amplitude_m,
        "rotation_amplitude_rad": rotation_amplitude_rad,
        "maximum_endpoint_position_error_m": maximum_endpoint_position_error_m,
        "maximum_endpoint_orientation_error_rad": maximum_endpoint_orientation_error_rad,
    }
    for name, raw in finite_positive.items():
        if not np.isfinite(raw) or float(raw) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(pause_duration_s) or pause_duration_s < 0.0:
        raise ValueError("pause_duration_s must be finite and nonnegative")
    if maximum_ik_iterations < 1:
        raise ValueError("maximum_ik_iterations must be positive")

    home = np.asarray(home_q, dtype=float).reshape(6).copy()
    home_pose, _ = model.fk(home)
    segment_samples = max(2, round(rate_hz * segment_duration_s))
    pause_samples = round(rate_hz * pause_duration_s)
    points: list[DiagnosticPoint] = []
    endpoint_reports: list[EndpointReport] = []

    for motion in DIAGNOSTIC_MOTIONS:
        target_positive = _target_for_direction(
            home_pose,
            motion,
            1.0,
            translation_amplitude_m,
            rotation_amplitude_rad,
        )
        target_negative = _target_for_direction(
            home_pose,
            motion,
            -1.0,
            translation_amplitude_m,
            rotation_amplitude_rad,
        )
        q_positive, positive_diagnostics = _solve_endpoint(
            solver,
            target_positive,
            home,
            maximum_iterations=maximum_ik_iterations,
            maximum_position_error_m=maximum_endpoint_position_error_m,
            maximum_orientation_error_rad=maximum_endpoint_orientation_error_rad,
        )
        q_negative, negative_diagnostics = _solve_endpoint(
            solver,
            target_negative,
            home,
            maximum_iterations=maximum_ik_iterations,
            maximum_position_error_m=maximum_endpoint_position_error_m,
            maximum_orientation_error_rad=maximum_endpoint_orientation_error_rad,
        )
        endpoint_reports.extend(
            (
                EndpointReport(motion.key, motion.positive_label, q_positive, positive_diagnostics),
                EndpointReport(motion.key, motion.negative_label, q_negative, negative_diagnostics),
            )
        )

        points.extend(
            _leg_points(motion, f"home to {motion.positive_label}", home, q_positive, segment_samples)
        )
        points.extend(
            _leg_points(motion, f"{motion.positive_label} to home", q_positive, home, segment_samples)
        )
        points.extend(
            _leg_points(motion, f"home to {motion.negative_label}", home, q_negative, segment_samples)
        )
        points.extend(
            _leg_points(motion, f"{motion.negative_label} to home", q_negative, home, segment_samples)
        )
        points.extend(
            DiagnosticPoint(motion.key, motion.label, "pause at home", home)
            for _ in range(pause_samples)
        )

    for point in points:
        if model.in_self_collision(point.q_command, gripper_position_m):
            raise ValueError(
                f"MuJoCo predicts a self-collision during {point.motion_label}: {point.phase}"
            )

    return DiagnosticPlan(home, tuple(points), tuple(endpoint_reports), float(rate_hz))


def plan_dynamics(plan: DiagnosticPlan) -> PlanDynamics:
    commands = np.vstack([plan.home_q, plan.command_array()])
    delta = np.diff(commands, axis=0)
    velocity = delta * plan.rate_hz
    acceleration = np.diff(velocity, axis=0) * plan.rate_hz
    jerk = np.diff(acceleration, axis=0) * plan.rate_hz

    def maximum(values: np.ndarray) -> np.ndarray:
        if values.shape[0] == 0:
            return np.zeros(6)
        return np.max(np.abs(values), axis=0)

    return PlanDynamics(
        max_step_rad=maximum(delta),
        max_velocity_rad_s=maximum(velocity),
        max_acceleration_rad_s2=maximum(acceleration),
        max_jerk_rad_s3=maximum(jerk),
    )


def validate_plan_dynamics(
    plan: DiagnosticPlan,
    *,
    maximum_step_rad: np.ndarray,
    maximum_velocity_rad_s: np.ndarray,
    maximum_acceleration_rad_s2: np.ndarray,
    maximum_jerk_rad_s3: np.ndarray,
) -> PlanDynamics:
    dynamics = plan_dynamics(plan)
    limits = {
        "step": (dynamics.max_step_rad, maximum_step_rad),
        "velocity": (dynamics.max_velocity_rad_s, maximum_velocity_rad_s),
        "acceleration": (dynamics.max_acceleration_rad_s2, maximum_acceleration_rad_s2),
        "jerk": (dynamics.max_jerk_rad_s3, maximum_jerk_rad_s3),
    }
    for name, (observed_raw, allowed_raw) in limits.items():
        observed = np.asarray(observed_raw, dtype=float).reshape(6)
        allowed = np.asarray(allowed_raw, dtype=float).reshape(6)
        if np.any(~np.isfinite(allowed)) or np.any(allowed <= 0.0):
            raise ValueError(f"diagnostic {name} limits must be finite and positive")
        exceeded = np.flatnonzero(observed > allowed + 1e-12)
        if exceeded.size:
            joint = int(exceeded[0])
            raise ValueError(
                f"diagnostic joint {joint} {name} {observed[joint]:.6f} "
                f"exceeds {allowed[joint]:.6f}"
            )
    return dynamics
