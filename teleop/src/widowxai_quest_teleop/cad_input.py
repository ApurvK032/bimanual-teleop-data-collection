from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .sample_buffer import LatestValueMailbox
from .hold_to_run import HoldToRunError, validate_evdev_key_code


DEFAULT_CAD_JOINT_NAMES = (
    "link1_link",
    "link2_link",
    "link3_link",
    "link4_link",
    "link5_link",
)


class CadInputError(ValueError):
    pass


class CadSafetyError(RuntimeError):
    pass


CAD_MAPPING_PENDING = "candidate_pending_physical_validation"
CAD_MAPPING_ACCEPTED = "accepted_by_operator"


def cad_commission_joint_index(value: Any) -> int | None:
    """Validate an optional isolated follower joint selection."""

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise CadSafetyError("CAD commission joint must be an integer from 0 through 4")
    joint = int(value)
    if not 0 <= joint < 5:
        raise CadSafetyError("CAD commission joint must be in the range [0, 4]")
    return joint


def validate_cad_commissioning_selection(
    *,
    live: bool,
    mapping_status: str,
    accept_unvalidated_mapping: bool,
    commission_joint: int | None,
) -> int | None:
    """Require isolated-axis operation for every pending physical mapping."""

    joint = cad_commission_joint_index(commission_joint)
    if not live or mapping_status == CAD_MAPPING_ACCEPTED:
        return joint
    if mapping_status != CAD_MAPPING_PENDING:
        raise CadSafetyError(f"unknown CAD mapping status: {mapping_status}")
    if not accept_unvalidated_mapping:
        raise CadSafetyError(
            "CAD signs are still candidate evidence; live output requires "
            "--accept-unvalidated-mapping"
        )
    if joint is None:
        raise CadSafetyError(
            "pending CAD mapping can move only one follower joint; pass "
            "--commission-joint 0, 1, 2, 3, or 4"
        )
    return joint


def cad_live_confirmation_token(robot_ip: str) -> str:
    """Return a CAD-specific token that cannot enable the Quest launcher."""

    address = str(robot_ip).strip()
    if not address:
        raise CadSafetyError("CAD hardware robot IP is required")
    return f"LIVE-WIDOWXAI-CAD-{address}"


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float).reshape(size)
    except (TypeError, ValueError) as exc:
        raise CadInputError(f"{name} must contain exactly {size} numeric values") from exc
    if not np.all(np.isfinite(vector)):
        raise CadInputError(f"{name} must contain only finite values")
    return vector


def wrapped_angle_delta(current: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Return the shortest signed angular displacement for each joint."""

    delta = np.asarray(current, dtype=float) - np.asarray(anchor, dtype=float)
    return np.arctan2(np.sin(delta), np.cos(delta))


class CadJointFilter:
    """Time-based, speed-adaptive low-pass filter for five tracked joints.

    The filter is updated only for a new M3T packet.  It uses monotonic packet
    arrival time rather than the tracker's wall-clock timestamp, and all angle
    differences take the shortest path across +/-pi.  The raw stream still
    goes through :class:`CadFreshnessWatchdog`; filtering is never used to hide
    a jump, restart, or stale source.
    """

    def __init__(
        self,
        *,
        minimum_cutoff_hz: float,
        speed_coefficient: float,
        derivative_cutoff_hz: float,
        maximum_cutoff_hz: float,
    ) -> None:
        values = {
            "minimum_cutoff_hz": minimum_cutoff_hz,
            "speed_coefficient": speed_coefficient,
            "derivative_cutoff_hz": derivative_cutoff_hz,
            "maximum_cutoff_hz": maximum_cutoff_hz,
        }
        parsed: dict[str, float] = {}
        for name, raw in values.items():
            try:
                parsed[name] = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"CAD source filter {name} must be finite") from None
            if not np.isfinite(parsed[name]):
                raise ValueError(f"CAD source filter {name} must be finite")
        if parsed["minimum_cutoff_hz"] <= 0.0:
            raise ValueError("CAD source filter minimum cutoff must be positive")
        if parsed["speed_coefficient"] < 0.0:
            raise ValueError("CAD source filter speed coefficient must be nonnegative")
        if parsed["derivative_cutoff_hz"] <= 0.0:
            raise ValueError("CAD source filter derivative cutoff must be positive")
        if parsed["maximum_cutoff_hz"] < parsed["minimum_cutoff_hz"]:
            raise ValueError("CAD source filter maximum cutoff must exceed its minimum")

        self.minimum_cutoff_hz = parsed["minimum_cutoff_hz"]
        self.speed_coefficient = parsed["speed_coefficient"]
        self.derivative_cutoff_hz = parsed["derivative_cutoff_hz"]
        self.maximum_cutoff_hz = parsed["maximum_cutoff_hz"]
        self.last_alpha = np.zeros(5, dtype=float)
        self.last_cutoff_hz = np.full(5, self.minimum_cutoff_hz, dtype=float)
        self._last_arrival_ns: int | None = None
        self._last_raw: np.ndarray | None = None
        self._filtered: np.ndarray | None = None
        self._filtered_velocity = np.zeros(5, dtype=float)

    @staticmethod
    def _alpha(cutoff_hz: np.ndarray | float, dt_s: float) -> np.ndarray:
        cutoff = np.asarray(cutoff_hz, dtype=float)
        return 1.0 - np.exp(-2.0 * np.pi * cutoff * float(dt_s))

    def reset(
        self,
        q: np.ndarray | None = None,
        arrival_monotonic_ns: int | None = None,
    ) -> np.ndarray | None:
        self._last_arrival_ns = None
        self._last_raw = None
        self._filtered = None
        self._filtered_velocity.fill(0.0)
        self.last_alpha.fill(0.0)
        self.last_cutoff_hz.fill(self.minimum_cutoff_hz)
        if q is None:
            return None
        return self.update(q, arrival_monotonic_ns)

    def update(
        self,
        q: np.ndarray,
        arrival_monotonic_ns: int | None,
    ) -> np.ndarray:
        raw = _finite_vector(q, 5, "CAD filter input")
        if arrival_monotonic_ns is None or int(arrival_monotonic_ns) <= 0:
            raise ValueError("CAD source filter requires a positive monotonic timestamp")
        arrival_ns = int(arrival_monotonic_ns)
        if self._filtered is None or self._last_raw is None or self._last_arrival_ns is None:
            self._last_arrival_ns = arrival_ns
            self._last_raw = raw.copy()
            self._filtered = raw.copy()
            self.last_alpha.fill(1.0)
            return self._filtered.copy()

        dt_s = (arrival_ns - self._last_arrival_ns) / 1e9
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("CAD source filter timestamps must increase")
        raw_velocity = wrapped_angle_delta(raw, self._last_raw) / dt_s
        derivative_alpha = float(self._alpha(self.derivative_cutoff_hz, dt_s))
        self._filtered_velocity += derivative_alpha * (
            raw_velocity - self._filtered_velocity
        )
        self.last_cutoff_hz = np.minimum(
            self.maximum_cutoff_hz,
            self.minimum_cutoff_hz
            + self.speed_coefficient * np.abs(self._filtered_velocity),
        )
        self.last_alpha = self._alpha(self.last_cutoff_hz, dt_s)
        self._filtered += self.last_alpha * wrapped_angle_delta(raw, self._filtered)
        self._filtered = np.arctan2(np.sin(self._filtered), np.cos(self._filtered))
        self._last_arrival_ns = arrival_ns
        self._last_raw = raw.copy()
        return self._filtered.copy()


@dataclass(frozen=True)
class CadJointSample:
    sequence: int
    source_time_ns: int
    arrival_monotonic_ns: int
    arrival_epoch_ns: int
    q: np.ndarray
    root_locked: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or int(self.sequence) < 0:
            raise CadInputError("CAD sequence must be a nonnegative integer")
        if isinstance(self.source_time_ns, bool) or int(self.source_time_ns) <= 0:
            raise CadInputError("CAD source timestamp must be a positive integer")
        if int(self.arrival_monotonic_ns) <= 0 or int(self.arrival_epoch_ns) <= 0:
            raise CadInputError("CAD arrival timestamps must be positive")
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "source_time_ns", int(self.source_time_ns))
        object.__setattr__(self, "arrival_monotonic_ns", int(self.arrival_monotonic_ns))
        object.__setattr__(self, "arrival_epoch_ns", int(self.arrival_epoch_ns))
        object.__setattr__(self, "q", _finite_vector(self.q, 5, "CAD q").copy())
        if not isinstance(self.root_locked, (bool, np.bool_)):
            raise CadInputError("CAD root_locked must be boolean")
        object.__setattr__(self, "root_locked", bool(self.root_locked))


def parse_cad_joint_packet(
    raw: str | bytes | dict[str, Any],
    *,
    expected_names: Sequence[str] = DEFAULT_CAD_JOINT_NAMES,
    arrival_monotonic_ns: int | None = None,
    arrival_epoch_ns: int | None = None,
    max_packet_age_s: float = 0.100,
    max_future_skew_s: float = 0.050,
    require_root_locked: bool = False,
) -> CadJointSample:
    """Parse one M3T joint packet and reject incomplete or delayed state."""

    try:
        if isinstance(raw, bytes):
            payload = json.loads(raw.decode("utf-8"))
        elif isinstance(raw, str):
            payload = json.loads(raw)
        else:
            payload = raw
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CadInputError("CAD packet is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise CadInputError("CAD packet must be a JSON object")

    root_locked = payload.get("root_locked", False)
    if not isinstance(root_locked, bool):
        raise CadInputError("CAD packet root_locked must be boolean")
    if require_root_locked and not root_locked:
        raise CadInputError("CAD packet rejected because the M3T root is not locked")

    expected = tuple(str(name) for name in expected_names)
    names_raw = payload.get("names")
    if not isinstance(names_raw, list) or not all(isinstance(name, str) for name in names_raw):
        raise CadInputError("CAD packet names must be a list of strings")
    names = tuple(names_raw)
    if len(names) != len(set(names)):
        raise CadInputError("CAD packet contains duplicate joint names")
    if len(names) != len(expected) or set(names) != set(expected):
        missing = sorted(set(expected) - set(names))
        unexpected = sorted(set(names) - set(expected))
        raise CadInputError(
            f"CAD packet joint set mismatch: missing={missing}, unexpected={unexpected}"
        )

    values = _finite_vector(payload.get("q"), len(expected), "CAD q")
    by_name = dict(zip(names, values, strict=True))
    ordered = np.array([by_name[name] for name in expected], dtype=float)

    sequence = payload.get("seq")
    source_time = payload.get("time_ns")
    if isinstance(sequence, bool) or not isinstance(sequence, (int, np.integer)):
        raise CadInputError("CAD sequence must be an integer")
    if isinstance(source_time, bool) or not isinstance(source_time, (int, np.integer)):
        raise CadInputError("CAD source timestamp must be an integer")

    monotonic_ns = time.perf_counter_ns() if arrival_monotonic_ns is None else int(arrival_monotonic_ns)
    epoch_ns = time.time_ns() if arrival_epoch_ns is None else int(arrival_epoch_ns)
    max_age_ns = int(float(max_packet_age_s) * 1e9)
    max_future_ns = int(float(max_future_skew_s) * 1e9)
    if max_age_ns <= 0 or max_future_ns < 0:
        raise ValueError("CAD packet age limits must be positive/nonnegative")
    source_age_ns = epoch_ns - int(source_time)
    if source_age_ns > max_age_ns:
        raise CadInputError(
            f"CAD packet is stale at receipt ({source_age_ns / 1e6:.1f} ms)"
        )
    if source_age_ns < -max_future_ns:
        raise CadInputError(
            f"CAD packet timestamp is {-source_age_ns / 1e6:.1f} ms in the future"
        )

    return CadJointSample(
        sequence=int(sequence),
        source_time_ns=int(source_time),
        arrival_monotonic_ns=monotonic_ns,
        arrival_epoch_ns=epoch_ns,
        q=ordered,
        root_locked=root_locked,
    )


@dataclass(frozen=True)
class CadFreshnessStatus:
    fresh: bool
    age_s: float
    recovery_streak: int
    unique_samples: int
    repeated_samples: int
    discontinuity_generation: int
    discontinuities: int
    last_discontinuity: str


class CadFreshnessWatchdog:
    """Require continuous, plausible M3T packets before enabling motion."""

    def __init__(
        self,
        timeout_s: float,
        fresh_samples_to_recover: int,
        max_source_step_rad: np.ndarray,
    ) -> None:
        timeout = float(timeout_s)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("CAD stale timeout must be finite and positive")
        self.timeout_ns = int(timeout * 1e9)
        self.fresh_samples_to_recover = int(fresh_samples_to_recover)
        if self.fresh_samples_to_recover < 1:
            raise ValueError("CAD recovery count must be at least one")
        self.max_source_step_rad = _finite_vector(
            max_source_step_rad, 5, "CAD source step limits"
        )
        if np.any(self.max_source_step_rad <= 0.0):
            raise ValueError("CAD source step limits must be positive")

        self.last_sequence: int | None = None
        self.last_q: np.ndarray | None = None
        self.last_unique_arrival_ns: int | None = None
        self.unique_samples = 0
        self.repeated_samples = 0
        self.discontinuity_generation = 0
        self.discontinuities = 0
        self.last_discontinuity = ""
        self._recovery_streak = 0
        self._fresh = False
        self._timed_out = False

    def _mark_discontinuity(self, reason: str) -> None:
        self._fresh = False
        self._recovery_streak = 0
        self.discontinuity_generation += 1
        self.discontinuities += 1
        self.last_discontinuity = reason

    def observe(
        self,
        sample: CadJointSample,
        now_ns: int | None = None,
    ) -> CadFreshnessStatus:
        if self.last_sequence is not None and sample.sequence == self.last_sequence:
            self.repeated_samples += 1
            return self.poll(now_ns)

        restarted = self.last_sequence is not None and sample.sequence < self.last_sequence
        jump_joint: int | None = None
        if not restarted and self.last_q is not None:
            step = np.abs(wrapped_angle_delta(sample.q, self.last_q))
            outside = np.flatnonzero(step > self.max_source_step_rad + 1e-12)
            if outside.size:
                jump_joint = int(outside[0])

        if restarted:
            self._mark_discontinuity("sequence_restart")
        elif jump_joint is not None:
            self._mark_discontinuity(f"source_jump_joint_{jump_joint}")

        self.last_sequence = sample.sequence
        self.last_q = sample.q.copy()
        self.last_unique_arrival_ns = sample.arrival_monotonic_ns
        self.unique_samples += 1
        self._timed_out = False
        if not restarted and jump_joint is None:
            self._recovery_streak += 1
            if self._recovery_streak >= self.fresh_samples_to_recover:
                self._fresh = True
        return self.poll(now_ns)

    def poll(self, now_ns: int | None = None) -> CadFreshnessStatus:
        now = time.perf_counter_ns() if now_ns is None else int(now_ns)
        age_ns = (
            self.timeout_ns + 1
            if self.last_unique_arrival_ns is None
            else now - self.last_unique_arrival_ns
        )
        if age_ns > self.timeout_ns:
            if not self._timed_out and self.last_unique_arrival_ns is not None:
                self._mark_discontinuity("timeout")
            self._timed_out = True
            self._fresh = False
            self._recovery_streak = 0
        return CadFreshnessStatus(
            fresh=self._fresh,
            age_s=max(0.0, age_ns / 1e9),
            recovery_streak=self._recovery_streak,
            unique_samples=self.unique_samples,
            repeated_samples=self.repeated_samples,
            discontinuity_generation=self.discontinuity_generation,
            discontinuities=self.discontinuities,
            last_discontinuity=self.last_discontinuity,
        )


def rest_command_limits(
    model_joint_limits: np.ndarray,
    rest_q: np.ndarray,
    minimum_delta_rad: np.ndarray,
    maximum_delta_rad: np.ndarray,
) -> np.ndarray:
    """Intersect the commissioning travel envelope with official model limits."""

    model_limits = np.asarray(model_joint_limits, dtype=float).reshape(6, 2)
    rest = _finite_vector(rest_q, 6, "rest position")
    minimum = _finite_vector(minimum_delta_rad, 6, "minimum rest-relative travel")
    maximum = _finite_vector(maximum_delta_rad, 6, "maximum rest-relative travel")
    if np.any(minimum > 0.0) or np.any(maximum < 0.0) or np.any(minimum >= maximum):
        raise CadSafetyError("rest-relative travel must straddle zero with ordered bounds")
    lower = np.maximum(model_limits[:, 0], rest + minimum)
    upper = np.minimum(model_limits[:, 1], rest + maximum)
    if np.any(lower >= upper):
        raise CadSafetyError("commissioning envelope leaves an empty joint range")
    if np.any(rest < lower) or np.any(rest > upper):
        raise CadSafetyError("rest position is outside the commissioning envelope")
    return np.column_stack([lower, upper])


def validate_cad_hardware_config(config: dict[str, Any]) -> None:
    """Validate the deliberately narrow first physical CAD profile.

    This validator is independent of the Quest/IK configuration because CAD
    packets already contain joint estimates.  It keeps the initial physical
    experiment bounded to a short, low-speed, five-joint commissioning run.
    """

    try:
        model = config["model"]
        cad = config["cad"]
        commissioning = config["commissioning"]
        control = config["control"]
        hardware = config["hardware"]
    except (KeyError, TypeError) as exc:
        raise CadSafetyError(f"CAD hardware configuration is missing {exc}") from None

    required_true = (
        "enabled",
        "live_output_implemented",
        "require_explicit_enable",
        "start_at_rest",
        "return_to_rest_on_exit",
    )
    for name in required_true:
        if hardware.get(name) is not True:
            raise CadSafetyError(f"hardware.{name} must be true")
    if hardware.get("control_gripper") is not False:
        raise CadSafetyError(
            "CAD commissioning must leave the untracked gripper disabled"
        )
    hold_to_run = hardware.get("hold_to_run")
    if not isinstance(hold_to_run, dict):
        raise CadSafetyError("hardware.hold_to_run configuration is required")
    if hold_to_run.get("enabled") is not True:
        raise CadSafetyError("hardware.hold_to_run.enabled must be true")
    if hold_to_run.get("backend") != "linux_evdev_key":
        raise CadSafetyError(
            "hardware.hold_to_run.backend must be linux_evdev_key"
        )
    if hold_to_run.get("require_explicit_device") is not True:
        raise CadSafetyError(
            "hardware.hold_to_run.require_explicit_device must be true"
        )
    if hold_to_run.get("require_initial_release") is not True:
        raise CadSafetyError(
            "hardware.hold_to_run.require_initial_release must be true"
        )
    try:
        validate_evdev_key_code(hold_to_run.get("key_code"))
    except HoldToRunError as exc:
        raise CadSafetyError(str(exc)) from None

    names = cad.get("expected_names")
    if not isinstance(names, list) or tuple(names) != DEFAULT_CAD_JOINT_NAMES:
        raise CadSafetyError(
            "cad.expected_names must match the five ordered M3T leader joints"
        )
    if cad.get("require_root_locked") is not True:
        raise CadSafetyError("physical CAD input must require root_locked=true")

    finite_vectors = {
        "model.rest_q_rad": (model.get("rest_q_rad"), 6),
        "model.command_anchor_q_rad": (model.get("command_anchor_q_rad"), 6),
        "cad.max_source_step_rad": (cad.get("max_source_step_rad"), 5),
        "cad.signs": (cad.get("signs"), 5),
        "cad.scales": (cad.get("scales"), 5),
        "cad.source_deadband_rad": (cad.get("source_deadband_rad"), 5),
        "commissioning.minimum_delta_rad": (
            commissioning.get("minimum_delta_rad"),
            6,
        ),
        "commissioning.maximum_delta_rad": (
            commissioning.get("maximum_delta_rad"),
            6,
        ),
        "commissioning.max_command_step_rad": (
            commissioning.get("max_command_step_rad"),
            6,
        ),
        "hardware.startup_max_joint_delta_rad": (
            hardware.get("startup_max_joint_delta_rad"),
            6,
        ),
    }
    parsed: dict[str, np.ndarray] = {}
    for name, (raw, size) in finite_vectors.items():
        try:
            parsed[name] = _finite_vector(raw, size, name)
        except CadInputError as exc:
            raise CadSafetyError(str(exc)) from None

    canonical_rest = np.zeros(6)
    canonical_home = np.array(
        [0.0, np.pi / 3.0, 5.0 * np.pi / 12.0, -np.pi / 3.0, 0.0, 0.0]
    )
    if not np.allclose(
        parsed["model.rest_q_rad"], canonical_rest, atol=1e-9, rtol=0.0
    ):
        raise CadSafetyError("first CAD physical profile must start at all-zero rest")
    if not np.allclose(
        parsed["model.command_anchor_q_rad"], canonical_home, atol=1e-9, rtol=0.0
    ):
        raise CadSafetyError("first CAD physical profile must anchor at normal home")
    try:
        hardware_rest = _finite_vector(
            hardware.get("rest_q_rad"), 6, "hardware.rest_q_rad"
        )
    except CadInputError as exc:
        raise CadSafetyError(str(exc)) from None
    if not np.allclose(hardware_rest, canonical_rest, atol=1e-9, rtol=0.0):
        raise CadSafetyError("CAD shutdown pose must be all-zero rest")

    signs = parsed["cad.signs"]
    scales = parsed["cad.scales"]
    deadbands = parsed["cad.source_deadband_rad"]
    source_steps = parsed["cad.max_source_step_rad"]
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise CadSafetyError("CAD physical signs must be exactly -1 or +1")
    if np.any(scales <= 0.0) or np.any(scales > 0.10 + 1e-12):
        raise CadSafetyError("first CAD physical scales must be within (0, 0.10]")
    if np.any(deadbands < 0.0):
        raise CadSafetyError("CAD source deadbands must be nonnegative")
    if np.any(source_steps <= 0.0) or np.any(source_steps > 0.12 + 1e-12):
        raise CadSafetyError("CAD source-step gates must be within (0, 0.12] rad")

    minimum = parsed["commissioning.minimum_delta_rad"]
    maximum = parsed["commissioning.maximum_delta_rad"]
    if np.any(minimum > 0.0) or np.any(maximum < 0.0) or np.any(minimum >= maximum):
        raise CadSafetyError("CAD commissioning travel must straddle zero")
    if np.any(np.maximum(np.abs(minimum[:5]), np.abs(maximum[:5])) > np.deg2rad(2.0) + 1e-9):
        raise CadSafetyError("first CAD physical envelope is limited to two degrees")
    if max(abs(minimum[5]), abs(maximum[5])) > 1e-5:
        raise CadSafetyError("untracked WidowX joint 5 must remain fixed")
    if np.any(parsed["commissioning.max_command_step_rad"] <= 0.0):
        raise CadSafetyError("CAD command-step caps must be positive")
    maximum_command_step = parsed["commissioning.max_command_step_rad"]
    if np.any(maximum_command_step[:5] > 0.003 + 1e-12) or maximum_command_step[5] > 2e-6 + 1e-12:
        raise CadSafetyError("CAD per-tick command caps exceed the first-run limits")
    startup_step = parsed["hardware.startup_max_joint_delta_rad"]
    if np.any(startup_step <= 0.0) or np.any(startup_step > 0.006 + 1e-12):
        raise CadSafetyError("CAD startup step caps must be within (0, 0.006] rad")

    positive_values = {
        "cad.stale_timeout_s": cad.get("stale_timeout_s"),
        "cad.max_packet_age_s": cad.get("max_packet_age_s"),
        "control.loop_rate_hz": control.get("loop_rate_hz"),
        "hardware.command_goal_time_s": hardware.get("command_goal_time_s"),
        "hardware.startup_ramp_duration_s": hardware.get("startup_ramp_duration_s"),
        "hardware.startup_ramp_rate_hz": hardware.get("startup_ramp_rate_hz"),
        "hardware.max_feedback_error_rad": hardware.get("max_feedback_error_rad"),
        "hardware.feedback_check_rate_hz": hardware.get("feedback_check_rate_hz"),
        "hardware.shutdown_move_duration_s": hardware.get("shutdown_move_duration_s"),
        "hardware.startup_rest_tolerance_rad": hardware.get(
            "startup_rest_tolerance_rad"
        ),
    }
    for name, raw in positive_values.items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise CadSafetyError(f"{name} must be finite and positive") from None
        if not np.isfinite(value) or value <= 0.0:
            raise CadSafetyError(f"{name} must be finite and positive")
    if float(control["loop_rate_hz"]) > 100.0:
        raise CadSafetyError("first CAD physical loop must not exceed 100 Hz")
    if not 0.025 <= float(hardware["command_goal_time_s"]) <= 0.030:
        raise CadSafetyError("first CAD physical driver horizon must be 25-30 ms")
    if float(hardware["max_feedback_error_rad"]) > 0.08:
        raise CadSafetyError("CAD feedback stop must not exceed 0.08 rad")
    if float(hardware["startup_rest_tolerance_rad"]) > 0.08:
        raise CadSafetyError("CAD startup must be within 0.08 rad of all-zero rest")
    if float(cad["max_packet_age_s"]) > float(cad["stale_timeout_s"]):
        raise CadSafetyError("CAD packet-age gate must not exceed the stale timeout")
    if float(cad["stale_timeout_s"]) > 0.100:
        raise CadSafetyError("first CAD physical stale timeout must not exceed 100 ms")
    source_filter = cad.get("source_filter")
    if not isinstance(source_filter, dict) or source_filter.get("enabled") is not True:
        raise CadSafetyError("physical CAD input requires cad.source_filter.enabled=true")
    try:
        validated_filter = CadJointFilter(
            minimum_cutoff_hz=source_filter.get("minimum_cutoff_hz"),
            speed_coefficient=source_filter.get("speed_coefficient"),
            derivative_cutoff_hz=source_filter.get("derivative_cutoff_hz"),
            maximum_cutoff_hz=source_filter.get("maximum_cutoff_hz"),
        )
    except ValueError as exc:
        raise CadSafetyError(str(exc)) from None
    if not 1.0 <= validated_filter.minimum_cutoff_hz <= 5.0:
        raise CadSafetyError("CAD source filter minimum cutoff must be within [1, 5] Hz")
    if validated_filter.maximum_cutoff_hz > 35.0:
        raise CadSafetyError("CAD source filter maximum cutoff must not exceed 35 Hz")
    try:
        future_skew = float(cad.get("max_future_skew_s"))
    except (TypeError, ValueError):
        raise CadSafetyError("CAD future-skew gate must be finite") from None
    if not np.isfinite(future_skew) or not 0.0 <= future_skew <= 0.050:
        raise CadSafetyError("CAD future-skew gate must be within [0, 50] ms")
    if float(hardware["startup_ramp_duration_s"]) < 2.0:
        raise CadSafetyError("CAD startup ramp must be at least two seconds")
    if float(hardware["startup_ramp_rate_hz"]) < 120.0:
        raise CadSafetyError("CAD startup ramp must retain the proven 120 Hz rate")
    if float(hardware["feedback_check_rate_hz"]) < 50.0:
        raise CadSafetyError("CAD physical feedback must be checked at least at 50 Hz")
    try:
        feedback_delay = float(hardware.get("feedback_tracking_delay_s"))
        max_gripper_delta = float(hardware.get("max_gripper_delta_m"))
    except (TypeError, ValueError):
        raise CadSafetyError("CAD feedback/gripper gate values must be finite") from None
    if not np.isfinite(feedback_delay) or not 0.0 <= feedback_delay <= 0.030:
        raise CadSafetyError("CAD feedback alignment delay must be within [0, 30] ms")
    if not np.isfinite(max_gripper_delta) or max_gripper_delta <= 0.0:
        raise CadSafetyError("CAD unchanged-gripper startup cap must be positive")

    collision_samples = hardware.get("startup_collision_samples")
    if (
        isinstance(collision_samples, bool)
        or not isinstance(collision_samples, (int, np.integer))
        or int(collision_samples) < 501
    ):
        raise CadSafetyError("CAD startup collision screen requires at least 501 samples")
    if hardware.get("end_effector_profile") != "legacy_1_8":
        raise CadSafetyError("CAD commissioning requires the proven legacy_1_8 follower profile")
    # Both lab controllers run firmware 1.9.4 since 2026-10-20; the driver
    # major.minor must match, so the proven 1.8.6 driver can no longer connect.
    if str(hardware.get("driver_version_tested")) != "1.9.3":
        raise CadSafetyError("CAD commissioning requires trossen-arm 1.9.3")

    recovery = cad.get("fresh_samples_to_recover")
    if isinstance(recovery, bool) or not isinstance(recovery, (int, np.integer)):
        raise CadSafetyError("CAD recovery count must be an integer")
    if int(recovery) < 3:
        raise CadSafetyError("physical CAD input requires at least three recovery samples")

    duration = hardware.get("max_demo_duration_s")
    try:
        duration_s = float(duration)
    except (TypeError, ValueError):
        raise CadSafetyError("CAD maximum demo duration must be finite") from None
    if not np.isfinite(duration_s) or not 0.0 < duration_s <= 15.0:
        raise CadSafetyError("first CAD physical run is capped at 15 seconds")

    if commissioning.get("mapping_status") not in (
        CAD_MAPPING_PENDING,
        CAD_MAPPING_ACCEPTED,
    ):
        raise CadSafetyError("CAD mapping status is missing or unknown")
    if commissioning.get("require_isolated_joint_while_pending") is not True:
        raise CadSafetyError(
            "commissioning.require_isolated_joint_while_pending must be true"
        )
    if hardware.get("arm_velocity_feedforward", {}).get("enabled", False):
        raise CadSafetyError("CAD commissioning must not use velocity feedforward")

    joint_limits = control.get("joint_command_limits", {})
    if joint_limits.get("enabled") is not True:
        raise CadSafetyError("CAD hardware output requires joint motion limits")
    try:
        max_velocity = _finite_vector(
            joint_limits.get("max_velocity"), 6, "CAD maximum velocity"
        )
        max_acceleration = _finite_vector(
            joint_limits.get("max_acceleration"), 6, "CAD maximum acceleration"
        )
    except CadInputError as exc:
        raise CadSafetyError(str(exc)) from None
    if np.any(max_velocity <= 0.0) or np.any(max_acceleration <= 0.0):
        raise CadSafetyError("CAD velocity and acceleration limits must be positive")
    if np.any(max_velocity[:5] > 0.10 + 1e-12) or np.any(
        max_acceleration[:5] > 0.50 + 1e-12
    ):
        raise CadSafetyError("CAD velocity/acceleration exceed first-run limits")
    per_tick = max_velocity / float(control["loop_rate_hz"])
    if np.any(per_tick > parsed["commissioning.max_command_step_rad"] + 1e-12):
        raise CadSafetyError("CAD velocity limits exceed the per-tick command caps")


class CadRestMapper:
    """Map five tracked leader joints relative to a rest-anchored robot pose."""

    def __init__(
        self,
        *,
        rest_q: np.ndarray,
        signs: np.ndarray,
        scales: np.ndarray,
        source_deadband_rad: np.ndarray | None = None,
        commission_joint: int | None = None,
        command_limits: np.ndarray,
    ) -> None:
        self.rest_q = _finite_vector(rest_q, 6, "rest position")
        self.signs = _finite_vector(signs, 5, "CAD joint signs")
        self.scales = _finite_vector(scales, 5, "CAD joint scales")
        if np.any(self.signs == 0.0) or np.any(self.scales <= 0.0):
            raise CadSafetyError("CAD signs must be nonzero and scales must be positive")
        self.source_deadband_rad = (
            np.zeros(5, dtype=float)
            if source_deadband_rad is None
            else _finite_vector(source_deadband_rad, 5, "CAD source deadband")
        )
        if np.any(self.source_deadband_rad < 0.0):
            raise CadSafetyError("CAD source deadband must be nonnegative")
        self.commission_joint = cad_commission_joint_index(commission_joint)
        self.command_limits = np.asarray(command_limits, dtype=float).reshape(6, 2)
        if not np.all(np.isfinite(self.command_limits)) or np.any(
            self.command_limits[:, 0] >= self.command_limits[:, 1]
        ):
            raise CadSafetyError("CAD command limits must be finite and ordered")
        self.engaged = False
        self.reanchor_generation = 0
        self._source_anchor: np.ndarray | None = None
        self._robot_anchor: np.ndarray | None = None

    def release(self) -> None:
        self.engaged = False
        self._source_anchor = None
        self._robot_anchor = None

    def engage(self, source_q: np.ndarray, robot_q: np.ndarray) -> np.ndarray:
        source = _finite_vector(source_q, 5, "CAD source anchor")
        robot = _finite_vector(robot_q, 6, "robot anchor")
        if np.any(robot < self.command_limits[:, 0] - 1e-12) or np.any(
            robot > self.command_limits[:, 1] + 1e-12
        ):
            raise CadSafetyError("robot anchor is outside the commissioning envelope")
        self._source_anchor = source.copy()
        self._robot_anchor = robot.copy()
        self.engaged = True
        self.reanchor_generation += 1
        if self.commission_joint is None:
            return robot.copy()
        anchored = self.rest_q.copy()
        anchored[self.commission_joint] = robot[self.commission_joint]
        return anchored

    def map(self, source_q: np.ndarray) -> np.ndarray:
        if not self.engaged or self._source_anchor is None or self._robot_anchor is None:
            raise CadSafetyError("CAD mapper is not anchored")
        source = _finite_vector(source_q, 5, "CAD source position")
        source_delta = wrapped_angle_delta(source, self._source_anchor)
        source_delta = np.sign(source_delta) * np.maximum(
            np.abs(source_delta) - self.source_deadband_rad,
            0.0,
        )
        mapped_delta = self.signs * self.scales * source_delta
        target = self._robot_anchor.copy()
        if self.commission_joint is None:
            target[:5] += mapped_delta
        else:
            # Pending physical sign checks are deliberately one-axis-only.
            # Every unselected tracked joint is commanded to the configured
            # home pose even if its visual estimate drifts.
            selected = self.commission_joint
            target[:5] = self.rest_q[:5]
            target[selected] = self._robot_anchor[selected] + mapped_delta[selected]
        # The leader currently publishes five joints. Keep J5 at rest rather
        # than allowing repeated re-anchors to accumulate an unobserved offset.
        target[5] = self.rest_q[5]
        outside = np.flatnonzero(
            (target < self.command_limits[:, 0] - 1e-12)
            | (target > self.command_limits[:, 1] + 1e-12)
        )
        if outside.size:
            joint = int(outside[0])
            raise CadSafetyError(
                f"CAD target for joint {joint} left the commissioning envelope"
            )
        return target


class CadDeadmanController:
    """Require a release/repress after every stale or discontinuous stream."""

    def __init__(self, mapper: CadRestMapper) -> None:
        self.mapper = mapper
        self._was_pressed = False
        self._needs_release = False

    @property
    def engaged(self) -> bool:
        return self.mapper.engaged

    @property
    def needs_release(self) -> bool:
        return self._needs_release

    def fault(self, deadman_pressed: bool) -> None:
        self.mapper.release()
        self._needs_release = self._needs_release or bool(deadman_pressed)

    def update(
        self,
        *,
        deadman_pressed: bool,
        stream_fresh: bool,
        sample: CadJointSample | None,
        robot_q: np.ndarray,
    ) -> np.ndarray | None:
        pressed = bool(deadman_pressed)
        if not stream_fresh or sample is None:
            self.fault(pressed)
            if not pressed:
                self._needs_release = False
                self._was_pressed = False
            else:
                self._was_pressed = True
            return None
        if not pressed:
            self.mapper.release()
            self._needs_release = False
            self._was_pressed = False
            return None
        if self._needs_release:
            self._was_pressed = True
            return None
        try:
            if not self._was_pressed or not self.mapper.engaged:
                self.mapper.engage(sample.q, robot_q)
            target = self.mapper.map(sample.q)
        except CadSafetyError:
            self.fault(pressed)
            raise
        self._was_pressed = True
        return target


class CadUdpReceiver:
    """Receive validated M3T packets into a capacity-one latest-state mailbox."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        expected_names: Sequence[str] = DEFAULT_CAD_JOINT_NAMES,
        max_packet_age_s: float = 0.100,
        max_future_skew_s: float = 0.050,
        require_root_locked: bool = False,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.expected_names = tuple(expected_names)
        self.max_packet_age_s = float(max_packet_age_s)
        self.max_future_skew_s = float(max_future_skew_s)
        self.require_root_locked = bool(require_root_locked)
        self.mailbox: LatestValueMailbox[CadJointSample] = LatestValueMailbox()
        self.received_packets = 0
        self.valid_packets = 0
        self.invalid_packets = 0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(0.1)
        self._socket = sock
        self.port = int(sock.getsockname()[1])
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="cad-joint-udp", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        sock = self._socket
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._socket = None

    def _run(self) -> None:
        sock = self._socket
        if sock is None:
            return
        while not self._stop.is_set():
            try:
                payload, _address = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop.is_set():
                    self.last_error = f"{type(exc).__name__}: {exc}"
                return
            self.received_packets += 1
            arrival_monotonic_ns = time.perf_counter_ns()
            arrival_epoch_ns = time.time_ns()
            try:
                sample = parse_cad_joint_packet(
                    payload,
                    expected_names=self.expected_names,
                    arrival_monotonic_ns=arrival_monotonic_ns,
                    arrival_epoch_ns=arrival_epoch_ns,
                    max_packet_age_s=self.max_packet_age_s,
                    max_future_skew_s=self.max_future_skew_s,
                    require_root_locked=self.require_root_locked,
                )
            except (CadInputError, ValueError) as exc:
                self.invalid_packets += 1
                self.last_error = str(exc)
                continue
            self.valid_packets += 1
            self.last_error = None
            self.mailbox.publish(sample)
