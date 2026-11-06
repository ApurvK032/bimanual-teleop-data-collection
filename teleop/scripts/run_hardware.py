from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np

from widowxai_quest_teleop.clutch import ClutchController
from widowxai_quest_teleop.config import (
    DUAL_ARM_SIDES,
    TASK_PROFILE_NAMES,
    apply_task_profile,
    load_config,
    parse_dual_arm_config,
    task_profile_for_quest_selection,
)
from widowxai_quest_teleop.decoupled_ik import DecoupledIK
from widowxai_quest_teleop.dual_arm_model import DualArmCollisionModel
from widowxai_quest_teleop.gripper import trigger_to_gripper_position
from widowxai_quest_teleop.hardware import (
    CommandGate,
    DryRunBackend,
    END_EFFECTOR_PROFILE_TO_VARIANT,
    HardwareSafetyError,
    HardwareUnavailableError,
    TrossenArmBackend,
)
from widowxai_quest_teleop.mapping import ClutchPoseMapper
from widowxai_quest_teleop.model import WidowXAIModel
from widowxai_quest_teleop.one_arm_safety import (
    MARGINAL_START_CONTACT_M,
    cross_arm_separation,
    screen_gripper_path,
    screen_selected_path,
    screen_self_collision_path,
)
from widowxai_quest_teleop.motion_limiter import (
    VelocityFeedforwardFilter,
    bounded_command_period,
    configured_command_spacing_stage,
    configured_minimum_command_interval,
    limiter_from_config,
    minimum_command_spacing_wait,
)
from widowxai_quest_teleop.pose_filter import ControllerPoseFilter, pose_filter_alphas
from widowxai_quest_teleop.quest_power import prepare_tabletop_tracking
from widowxai_quest_teleop.safety import FreshSequenceWatchdog, TimeAlignedCommandHistory
from widowxai_quest_teleop.telemetry import TelemetryLogger
from widowxai_quest_teleop.transport import QuestReceiver
from widowxai_quest_teleop.workstation import (
    WorkstationSafetyError,
    run_workstation_preflight,
)


def validate_hardware_config(config: dict) -> None:
    """Reject malformed live safety settings before any network or arm activity."""

    hardware = config["hardware"]
    quest = config["quest"]
    # A bimanual profile carries its hand and mapping per arm under `arms`,
    # validated by config.parse_dual_arm_config. Everything below this point is
    # shared control/hardware safety and applies to both profiles.
    if str(quest.get("mode", "")).lower() != "bimanual":
        if quest.get("hand", "left") not in ("left", "right"):
            raise HardwareSafetyError("quest.hand must be 'left' or 'right'")
        if quest.get("mapping_mode", "real") not in ("real", "mirror"):
            raise HardwareSafetyError("quest.mapping_mode must be 'real' or 'mirror'")
    control = config["control"]
    update_mode = control.get("update_mode", "fixed_rate")
    if update_mode not in ("fixed_rate", "quest_synchronized"):
        raise HardwareSafetyError(f"unknown control.update_mode: {update_mode}")
    tabletop_tracking = quest.get("prepare_tabletop_tracking_via_adb", False)
    if not isinstance(tabletop_tracking, bool):
        raise HardwareSafetyError(
            "quest.prepare_tabletop_tracking_via_adb must be a boolean"
        )

    positive_values = {
        "control.loop_rate_hz": control["loop_rate_hz"],
        "quest.stale_timeout_s": quest["stale_timeout_s"],
        "hardware.startup_ramp_duration_s": hardware["startup_ramp_duration_s"],
        "hardware.startup_ramp_rate_hz": hardware["startup_ramp_rate_hz"],
        "hardware.max_feedback_error_rad": hardware["max_feedback_error_rad"],
        "hardware.max_gripper_feedback_error_m": hardware["max_gripper_feedback_error_m"],
        "hardware.feedback_check_rate_hz": hardware["feedback_check_rate_hz"],
        "hardware.max_gripper_delta_m": hardware["max_gripper_delta_m"],
        "hardware.shutdown_move_duration_s": hardware.get("shutdown_move_duration_s", 2.0),
        "hardware.startup_gripper_move_duration_s": hardware.get(
            "startup_gripper_move_duration_s", 2.0
        ),
    }
    for name, raw in positive_values.items():
        value = float(raw)
        if not np.isfinite(value) or value <= 0.0:
            raise HardwareSafetyError(f"{name} must be finite and positive")

    if "max_demo_duration_s" not in hardware:
        raise HardwareSafetyError(
            "hardware.max_demo_duration_s must be a positive number or explicit null"
        )
    max_demo_duration = hardware["max_demo_duration_s"]
    if max_demo_duration is not None:
        max_demo_duration_value = float(max_demo_duration)
        if (
            not np.isfinite(max_demo_duration_value)
            or max_demo_duration_value <= 0.0
        ):
            raise HardwareSafetyError(
                "hardware.max_demo_duration_s must be positive or explicit null"
            )

    nonnegative_values = {
        "control.minimum_command_interval_s": control.get("minimum_command_interval_s", 0.0),
        "hardware.command_goal_time_s": hardware["command_goal_time_s"],
        "hardware.feedback_tracking_delay_s": hardware.get("feedback_tracking_delay_s", 0.0),
        "hardware.joint_limit_margin_rad": hardware["joint_limit_margin_rad"],
    }
    for name, raw in nonnegative_values.items():
        value = float(raw)
        if not np.isfinite(value) or value < 0.0:
            raise HardwareSafetyError(f"{name} must be finite and nonnegative")
    minimum_command_interval = float(control.get("minimum_command_interval_s", 0.0))
    try:
        command_spacing_stage = configured_command_spacing_stage(control)
    except ValueError as exc:
        raise HardwareSafetyError(str(exc)) from None
    if minimum_command_interval > 0.0 and update_mode != "quest_synchronized":
        raise HardwareSafetyError(
            "control.minimum_command_interval_s requires quest_synchronized updates"
        )
    if minimum_command_interval >= float(quest["stale_timeout_s"]):
        raise HardwareSafetyError(
            "control.minimum_command_interval_s must be shorter than the stale timeout"
        )
    if minimum_command_interval <= 0.0 and "command_spacing_stage" in control:
        raise HardwareSafetyError(
            f"control.command_spacing_stage={command_spacing_stage!r} requires a positive "
            "minimum command interval"
        )

    max_delta = np.asarray(config["ik"]["max_dq_per_joint_rad"], dtype=float).reshape(6)
    if not np.all(np.isfinite(max_delta)) or np.any(max_delta <= 0.0):
        raise HardwareSafetyError("ik.max_dq_per_joint_rad must contain six finite positive caps")

    joint_limits = control.get("joint_command_limits")
    if joint_limits and joint_limits.get("enabled", False):
        max_velocity = np.asarray(joint_limits["max_velocity"], dtype=float).reshape(6)
        max_acceleration = np.asarray(joint_limits["max_acceleration"], dtype=float).reshape(6)
        max_jerk_raw = joint_limits.get("max_jerk")
        if not np.all(np.isfinite(max_velocity)) or np.any(max_velocity <= 0.0):
            raise HardwareSafetyError("joint command velocities must contain six finite positive limits")
        if not np.all(np.isfinite(max_acceleration)) or np.any(max_acceleration <= 0.0):
            raise HardwareSafetyError("joint command accelerations must contain six finite positive limits")
        if max_jerk_raw is not None:
            max_jerk = np.asarray(max_jerk_raw, dtype=float).reshape(6)
            if not np.all(np.isfinite(max_jerk)) or np.any(max_jerk <= 0.0):
                raise HardwareSafetyError("joint command jerks must contain six finite positive limits")
        if np.any(max_velocity / float(control["loop_rate_hz"]) > max_delta + 1e-12):
            raise HardwareSafetyError("joint velocity limits exceed the configured per-tick command caps")

    gripper_limits = control.get("gripper_command_limits")
    if gripper_limits and gripper_limits.get("enabled", False):
        gripper_velocity = np.asarray(gripper_limits["max_velocity"], dtype=float).reshape(1)
        gripper_acceleration = np.asarray(gripper_limits["max_acceleration"], dtype=float).reshape(1)
        gripper_jerk_raw = gripper_limits.get("max_jerk")
        if not np.all(np.isfinite(gripper_velocity)) or np.any(gripper_velocity <= 0.0):
            raise HardwareSafetyError("gripper command velocity must be finite and positive")
        if not np.all(np.isfinite(gripper_acceleration)) or np.any(gripper_acceleration <= 0.0):
            raise HardwareSafetyError("gripper command acceleration must be finite and positive")
        if gripper_jerk_raw is not None:
            gripper_jerk = np.asarray(gripper_jerk_raw, dtype=float).reshape(1)
            if not np.all(np.isfinite(gripper_jerk)) or np.any(gripper_jerk <= 0.0):
                raise HardwareSafetyError("gripper command jerk must be finite and positive")
        if gripper_velocity[0] / float(control["loop_rate_hz"]) > float(hardware["max_gripper_delta_m"]) + 1e-12:
            raise HardwareSafetyError("gripper velocity limit exceeds the configured per-tick command cap")

    feedforward = hardware.get("arm_velocity_feedforward", {})
    if feedforward is None:
        feedforward = {}
    if not isinstance(feedforward, dict):
        raise HardwareSafetyError("hardware.arm_velocity_feedforward must be a mapping")
    feedforward_enabled = feedforward.get("enabled", False)
    if not isinstance(feedforward_enabled, bool):
        raise HardwareSafetyError("hardware.arm_velocity_feedforward.enabled must be a boolean")
    if feedforward_enabled:
        try:
            feedforward_filter = VelocityFeedforwardFilter(
                np.zeros(6),
                filter_alpha=feedforward["filter_alpha"],
                gain=feedforward["gain"],
                max_velocity=np.asarray(feedforward["max_velocity_rad_s"], dtype=float),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HardwareSafetyError(f"invalid arm velocity feedforward: {exc}") from None
        joint_limits = control.get("joint_command_limits", {}) or {}
        if joint_limits.get("enabled", False):
            joint_velocity_caps = np.asarray(
                joint_limits["max_velocity"],
                dtype=float,
            ).reshape(6)
            if np.any(feedforward_filter.max_velocity > joint_velocity_caps):
                raise HardwareSafetyError(
                    "arm velocity feedforward caps must not exceed the joint command velocity caps"
                )

    gripper_min = float(hardware["gripper_min_demo_m"])
    gripper_open = float(hardware["gripper_open_m"])
    if not np.isfinite(gripper_min) or not np.isfinite(gripper_open) or gripper_min >= gripper_open:
        raise HardwareSafetyError("hardware gripper limits must be finite and ordered")

    return_to_rest = hardware.get("return_to_rest_on_exit", True)
    if not isinstance(return_to_rest, bool):
        raise HardwareSafetyError("hardware.return_to_rest_on_exit must be a boolean")
    rest_q = np.asarray(hardware.get("rest_q_rad", np.zeros(6)), dtype=float).reshape(6)
    if not np.all(np.isfinite(rest_q)):
        raise HardwareSafetyError("hardware.rest_q_rad must contain six finite values")
    rest_gripper = float(hardware.get("rest_gripper_m", 0.0))
    if not np.isfinite(rest_gripper):
        raise HardwareSafetyError("hardware.rest_gripper_m must be finite")
    if hardware.get("control_gripper", True) and not gripper_min <= rest_gripper <= gripper_open:
        raise HardwareSafetyError("hardware.rest_gripper_m is outside the configured gripper range")

    translation_alpha, rotation_alpha = pose_filter_alphas(control)
    for name, pose_alpha in (
        ("control.pose_filter_translation_alpha", translation_alpha),
        ("control.pose_filter_rotation_alpha", rotation_alpha),
    ):
        if not np.isfinite(pose_alpha) or not 0.0 <= pose_alpha <= 1.0:
            raise HardwareSafetyError(f"{name} must be finite and within [0, 1]")
    try:
        ControllerPoseFilter(control)
    except ValueError as exc:
        raise HardwareSafetyError(str(exc)) from None

    collision_samples = hardware["startup_collision_samples"]
    if isinstance(collision_samples, bool) or int(collision_samples) != collision_samples:
        raise HardwareSafetyError("hardware.startup_collision_samples must be an integer")
    if int(collision_samples) < 2:
        raise HardwareSafetyError("hardware.startup_collision_samples must be at least 2")

    fresh_samples = quest["fresh_samples_to_recover"]
    if isinstance(fresh_samples, bool) or int(fresh_samples) != fresh_samples:
        raise HardwareSafetyError("quest.fresh_samples_to_recover must be an integer")
    if int(fresh_samples) < 1:
        raise HardwareSafetyError("quest.fresh_samples_to_recover must be at least 1")


def validate_live_hardware_timing(config: dict) -> None:
    """Reject command timing that is inappropriate for an unproven physical demo."""

    control = config["control"]
    hardware = config["hardware"]
    loop_hz = float(control["loop_rate_hz"])
    goal_time = float(hardware["command_goal_time_s"])
    if goal_time <= 0.001 and loop_hz < 300.0:
        raise HardwareSafetyError(
            "goal_time disables driver interpolation below Trossen's documented 300 Hz minimum"
        )

    command_path = control.get("command_path", "limited_interpolated")
    if command_path == "article_unshaped":
        if control.get("joint_command_limits", {}).get("enabled", False):
            raise HardwareSafetyError(
                "article_unshaped must not add a joint velocity/acceleration limiter"
            )
        if (
            hardware.get("control_gripper", True)
            and control.get("gripper_command_limits", {}).get("enabled", False)
        ):
            raise HardwareSafetyError(
                "article_unshaped must not add a gripper velocity/acceleration limiter"
            )
        return
    if command_path != "limited_interpolated":
        raise HardwareSafetyError(f"unknown control.command_path: {command_path}")

    if not control.get("joint_command_limits", {}).get("enabled", False):
        raise HardwareSafetyError("live demo requires explicit joint velocity and acceleration limits")
    if (
        hardware.get("control_gripper", True)
        and not control.get("gripper_command_limits", {}).get("enabled", False)
    ):
        raise HardwareSafetyError("live demo requires explicit gripper velocity and acceleration limits")


def require_quest_selection(
    sample,
    expected_hand: str,
    expected_mapping_mode: str,
) -> None:
    """Fail closed if the page no longer reports the preflight-locked input."""

    if sample.hand != expected_hand or sample.mapping_mode != expected_mapping_mode:
        raise HardwareSafetyError(
            "Quest page selection changed after preflight: "
            f"received {sample.hand}/{sample.mapping_mode}, "
            f"locked {expected_hand}/{expected_mapping_mode}; "
            "keep grip released and restart the run"
        )


def wait_for_released_quest(
    receiver: QuestReceiver,
    timeout_s: float,
    stale_timeout_s: float,
    *,
    expected_hand: str | None = None,
    expected_mapping_mode: str | None = None,
):
    """Require live tracking and a released deadman before any position mode is enabled."""

    if (expected_hand is None) != (expected_mapping_mode is None):
        raise ValueError("expected hand and mapping mode must be provided together")
    deadline = time.perf_counter() + float(timeout_s)
    watchdog = FreshSequenceWatchdog(stale_timeout_s, fresh_samples_to_recover=3)
    while time.perf_counter() < deadline:
        sample, _ = receiver.mailbox.take_latest()
        if sample is not None:
            if expected_hand is not None and expected_mapping_mode is not None:
                require_quest_selection(
                    sample,
                    expected_hand,
                    expected_mapping_mode,
                )
            freshness = watchdog.observe(sample)
            if freshness.fresh and sample.grip < 0.65:
                return sample
        time.sleep(0.01)
    raise HardwareSafetyError(
        "Quest preflight failed: enter WebXR, keep the grip released, and provide fresh tracking"
    )


def resolve_demo_duration(hardware: dict, requested_duration_s: float) -> float:
    """Resolve an optional operator duration against an optional config cap."""

    requested = float(requested_duration_s)
    if not np.isfinite(requested) or requested < 0.0:
        raise ValueError("requested duration must be finite and nonnegative")
    configured = hardware["max_demo_duration_s"]
    duration_limit = np.inf if configured is None else float(configured)
    if requested == 0.0:
        return duration_limit
    return min(requested, duration_limit)


def ramp_to_home(backend, gate: CommandGate, start, home_q: np.ndarray, config: dict) -> None:
    """Move J0-J5 to home while leaving the physical gripper untouched."""

    hardware = config["hardware"]
    duration = float(hardware["startup_ramp_duration_s"])
    rate = float(hardware["startup_ramp_rate_hz"])
    steps = max(1, round(duration * rate))
    period = 1.0 / rate
    feedback_stride = max(1, round(rate / float(hardware["feedback_check_rate_hz"])))
    start_q = start.q_arm.copy()
    start_gripper = float(start.gripper_position_m)
    next_tick = time.perf_counter()
    for index in range(1, steps + 1):
        alpha = index / steps
        q = start_q + alpha * (home_q - start_q)
        gate.validate(q, start_gripper)
        backend.send_positions(q, start_gripper, include_gripper=False)
        if index % feedback_stride == 0 or index == steps:
            feedback = backend.read_state()
            if np.max(np.abs(feedback.q_arm - q)) > hardware["max_feedback_error_rad"]:
                raise HardwareSafetyError(
                    "measured joint tracking error exceeded the demo limit during startup ramp"
                )
        next_tick += period
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0.0:
            time.sleep(sleep_for)
        else:
            next_tick = time.perf_counter()


def validate_time_aligned_feedback(
    q_feedback: np.ndarray,
    feedback_time_s: float,
    command_history: TimeAlignedCommandHistory,
    max_error_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Check measured joints against the delayed interpolated command."""

    measured = np.asarray(q_feedback, dtype=float).reshape(6)
    reference = command_history.reference_at(feedback_time_s)
    error = measured - reference
    absolute_error = np.abs(error)
    joint = int(np.argmax(absolute_error))
    if absolute_error[joint] > float(max_error_rad):
        # The faulting sample is the one sample that never reaches telemetry:
        # every caller validates before writing its record, so raising bare
        # discards the only evidence of what went wrong. Carry it in the
        # message instead.
        reference_age_s = float(feedback_time_s) - command_history.newest_time_s
        raise HardwareSafetyError(
            f"measured joint {joint} time-aligned tracking error "
            f"{absolute_error[joint]:.6f} rad exceeded the demo limit "
            f"{float(max_error_rad):.6f} rad"
            f" | measured={measured[joint]:.6f} reference={reference[joint]:.6f}"
            f" | reference {command_history.clamp_state(feedback_time_s)},"
            f" newest command {reference_age_s * 1000.0:.1f} ms old,"
            f" history spans {(command_history.newest_time_s - command_history.oldest_time_s) * 1000.0:.1f} ms"
            f" | all-joint |error| = {np.round(absolute_error, 6).tolist()}"
        )
    return reference, error


def make_startup_command_gate(
    state,
    startup_max_delta: np.ndarray,
    hardware: dict,
) -> CommandGate:
    """Permit measured encoder offsets during a one-way ramp into model limits."""

    startup_limits = state.joint_limits.copy()
    # Trossen accepts feedback within each joint's reported position tolerance,
    # including tiny negative sleep-pose readings for nominally zero-limited
    # joints. Expand only this startup gate by that reported tolerance. The
    # teleoperation gate below still uses the strict configured margin.
    tolerances = np.asarray(state.position_tolerances, dtype=float).reshape(7)
    startup_limits[:, 0] -= tolerances
    startup_limits[:, 1] += tolerances
    return CommandGate(
        state.q_arm,
        state.gripper_position_m,
        startup_limits,
        startup_max_delta,
        # The measured sleep pose can be a few milliradians outside MuJoCo's
        # ideal zero. Physical limits govern only this inward startup ramp.
        joint_limit_margin_rad=0.0,
        gripper_limits_m=(startup_limits[6, 0], startup_limits[6, 1]),
        max_gripper_delta_m=hardware["max_gripper_delta_m"],
    )


def teleop_gripper_limits(
    state,
    hardware: dict,
    *,
    control_gripper: bool,
) -> tuple[float, float]:
    """Return commanded limits, or tolerated feedback bounds for arm-only runs."""

    if control_gripper:
        return (
            float(hardware["gripper_min_demo_m"]),
            float(hardware["gripper_open_m"]),
        )
    tolerance = float(np.asarray(state.position_tolerances, dtype=float).reshape(7)[6])
    return (
        float(state.joint_limits[6, 0] - tolerance),
        float(state.joint_limits[6, 1] + tolerance),
    )


def gripper_feedback_tolerance(state, hardware: dict) -> float:
    """Never demand tighter gripper tracking than the controller specifies."""

    controller_tolerance = float(
        np.asarray(state.position_tolerances, dtype=float).reshape(7)[6]
    )
    return max(float(hardware["max_gripper_feedback_error_m"]), controller_tolerance)


def open_gripper_at_home(backend, model: WidowXAIModel, state, home_q: np.ndarray, config: dict):
    """Reproduce the previous stack's separate blocking gripper-open step."""

    hardware = config["hardware"]
    gripper_open = float(hardware["gripper_open_m"])
    tolerance = float(np.asarray(state.position_tolerances, dtype=float).reshape(7)[6])
    if not (
        state.joint_limits[6, 0] - tolerance
        <= gripper_open
        <= state.joint_limits[6, 1] + tolerance
    ):
        raise HardwareSafetyError("configured open gripper position is outside controller limits")
    collision_alpha = model.first_self_collision_on_path(
        home_q,
        home_q,
        start_gripper_q=state.gripper_position_m,
        end_gripper_q=gripper_open,
        samples=hardware["startup_collision_samples"],
    )
    if collision_alpha is not None:
        raise HardwareSafetyError(
            "pinned 2025 MuJoCo collision geometry rejects the gripper-open path "
            f"near {collision_alpha * 100:.1f}%"
        )
    duration = float(hardware.get("startup_gripper_move_duration_s", 2.0))
    print(f"startup: opening gripper to {gripper_open:.3f} m with a {duration:g} s blocking move")
    backend.move_gripper_blocking(gripper_open, duration_s=duration)
    settled = backend.read_state()
    if (
        abs(settled.gripper_position_m - gripper_open)
        > gripper_feedback_tolerance(settled, hardware)
    ):
        raise HardwareSafetyError("gripper did not reach open within controller tolerance")
    return settled


def return_to_rest(
    backend,
    model: WidowXAIModel,
    config: dict,
    *,
    control_gripper: bool,
) -> None:
    """Validate and execute the previous stack's blocking sleep transition."""

    hardware = config["hardware"]
    state = backend.read_state()
    rest_q = np.asarray(hardware.get("rest_q_rad", np.zeros(6)), dtype=float).reshape(6)
    rest_gripper = float(hardware.get("rest_gripper_m", 0.0))
    tolerances = np.asarray(state.position_tolerances, dtype=float).reshape(7)
    lower = state.joint_limits[:, 0] - tolerances
    upper = state.joint_limits[:, 1] + tolerances
    rest_positions = np.concatenate([rest_q, [rest_gripper]])
    checked_positions = rest_positions if control_gripper else rest_positions[:6]
    checked_lower = lower if control_gripper else lower[:6]
    checked_upper = upper if control_gripper else upper[:6]
    if np.any(checked_positions < checked_lower) or np.any(checked_positions > checked_upper):
        raise HardwareSafetyError("configured rest pose is outside controller limits")

    arm_collision_alpha = model.first_self_collision_on_path(
        state.q_arm,
        rest_q,
        start_gripper_q=state.gripper_position_m,
        end_gripper_q=state.gripper_position_m,
        samples=hardware["startup_collision_samples"],
    )
    if arm_collision_alpha is not None:
        raise HardwareSafetyError(
            "pinned 2025 MuJoCo collision geometry rejects the return-to-rest path "
            f"near {arm_collision_alpha * 100:.1f}%"
        )
    if control_gripper:
        gripper_collision_alpha = model.first_self_collision_on_path(
            rest_q,
            rest_q,
            start_gripper_q=state.gripper_position_m,
            end_gripper_q=rest_gripper,
            samples=hardware["startup_collision_samples"],
        )
        if gripper_collision_alpha is not None:
            raise HardwareSafetyError(
                "pinned 2025 MuJoCo collision geometry rejects gripper close at rest "
                f"near {gripper_collision_alpha * 100:.1f}%"
            )

    duration = float(hardware.get("shutdown_move_duration_s", 2.0))
    print(
        f"shutdown: returning arm to rest [0, 0, 0, 0, 0, 0] deg "
        f"with a {duration:g} s blocking move"
    )
    backend.move_to_rest(
        rest_q,
        rest_gripper,
        duration_s=duration,
        include_gripper=control_gripper,
    )
    settled = backend.read_state()
    if np.max(np.abs(settled.q_arm - rest_q)) > hardware["max_feedback_error_rad"]:
        raise HardwareSafetyError("arm did not reach rest within the feedback limit")
    if control_gripper and (
        abs(settled.gripper_position_m - rest_gripper)
        > gripper_feedback_tolerance(settled, hardware)
    ):
        raise HardwareSafetyError("gripper did not reach its rest position within the feedback limit")
    print("rest reached: driver cleanup can proceed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed WidowXAI physical demo")
    parser.add_argument("--config", default="configs/safe_demo_30pct.yaml")
    parser.add_argument("--live", action="store_true", help="use the official Trossen hardware driver")
    parser.add_argument(
        "--robot-ip",
        help="override hardware.robot_ip from the selected config",
    )
    parser.add_argument(
        "--end-effector-profile",
        choices=tuple(END_EFFECTOR_PROFILE_TO_VARIANT),
        help=(
            "override the selected config; legacy_1_8 reproduces the "
            "previously working arm setup"
        ),
    )
    parser.add_argument("--confirm-live", default="", help="must equal LIVE-WIDOWXAI-<robot-ip>")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help=(
            "optional positive run duration; 0 uses the configured limit, "
            "or runs until Ctrl+C when that limit is null"
        ),
    )
    parser.add_argument("--label", default="widowxai-hardware-demo")
    parser.add_argument(
        "--task-profile",
        choices=TASK_PROFILE_NAMES,
        help="strictly select one fixed hand and Behind/Mirror candidate",
    )
    parser.add_argument(
        "--dual-guard-config",
        help=(
            "dual profile supplying measured base transforms and the stationary arm; "
            "requires --moving-side and --live"
        ),
    )
    parser.add_argument(
        "--moving-side",
        choices=DUAL_ARM_SIDES,
        help="physical side moved while the other dual-profile arm stays read-only",
    )
    args = parser.parse_args()

    if bool(args.dual_guard_config) != bool(args.moving_side):
        raise SystemExit("--dual-guard-config and --moving-side must be supplied together")
    if args.dual_guard_config and not args.live:
        raise SystemExit("the stationary-arm guard is available only for an explicit live run")

    config = load_config(args.config)
    page_locked_task_profile = bool(
        config["project"].get("page_locked_task_profile", False)
    )
    if (
        config["project"].get("require_task_profile", False)
        and not args.task_profile
        and not page_locked_task_profile
    ):
        raise SystemExit("this configuration requires --task-profile")
    if args.task_profile:
        try:
            apply_task_profile(config, args.task_profile)
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"invalid task profile: {exc}") from None
    hardware = config["hardware"]
    control = config["control"]
    if not hardware.get("enabled") or not hardware.get("require_explicit_enable"):
        raise SystemExit("hardware demo configuration is not explicitly gated")
    try:
        validate_hardware_config(config)
    except (KeyError, TypeError, ValueError, HardwareSafetyError) as exc:
        raise SystemExit(f"invalid hardware safety configuration: {exc}") from None
    controller_filter = ControllerPoseFilter(control)
    if not np.isfinite(args.duration) or args.duration < 0.0:
        raise SystemExit("--duration must be finite and nonnegative")

    model = WidowXAIModel(config["model"]["xml_path"])
    solver = DecoupledIK.from_config(model, config)
    home_q = model.clamp_joints(np.asarray(config["model"]["simulation_start_q_rad"], dtype=float))
    max_delta = np.asarray(config["ik"]["max_dq_per_joint_rad"], dtype=float)

    robot_ip = args.robot_ip or hardware.get("robot_ip")
    end_effector_profile = args.end_effector_profile or hardware.get("end_effector_profile")
    guard_config = None
    guard_arms = None
    guard_model = None
    guard_side = args.moving_side
    guard_other = None
    guard_clearance_m = 0.0
    if args.dual_guard_config:
        guard_config = load_config(args.dual_guard_config)
        try:
            guard_arms = parse_dual_arm_config(guard_config)
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"invalid stationary-arm guard configuration: {exc}") from None
        guard_other = next(side for side in DUAL_ARM_SIDES if side != guard_side)
        if robot_ip != guard_arms[guard_side].robot_ip:
            raise SystemExit(
                f"moving robot IP {robot_ip} does not match "
                f"arms.{guard_side}.robot_ip {guard_arms[guard_side].robot_ip}"
            )
        if not all(guard_arms[side].placement.measured for side in DUAL_ARM_SIDES):
            raise SystemExit("stationary-arm guard requires measured transforms for both bases")
        guard_clearance_m = float(guard_config["safety"]["cross_arm_clearance_m"])
        guard_model = DualArmCollisionModel(
            {side: guard_arms[side].placement for side in DUAL_ARM_SIDES},
            xml_path=guard_config["model"]["xml_path"],
            clearance_m=guard_clearance_m,
        )
        config["_stationary_arm_guard"] = {
            "dual_config": args.dual_guard_config,
            "moving_side": guard_side,
            "moving_ip": robot_ip,
            "stationary_side": guard_other,
            "stationary_ip": guard_arms[guard_other].robot_ip,
            "cross_arm_clearance_m": guard_clearance_m,
        }
    if args.live:
        try:
            validate_live_hardware_timing(config)
        except HardwareSafetyError as exc:
            raise SystemExit(f"unsafe live command timing: {exc}") from None
        if platform.system() not in ("Linux", "Darwin"):
            raise SystemExit(
                "live hardware is unavailable on native Windows: Trossen's official driver "
                "supports Ubuntu and macOS only; no arm connection was attempted"
            )
        if not robot_ip:
            raise SystemExit("--robot-ip is required for a live demo")
        if not end_effector_profile:
            raise SystemExit("--end-effector-profile is required for a live demo")
        if str(end_effector_profile) not in END_EFFECTOR_PROFILE_TO_VARIANT:
            raise SystemExit("unsupported end-effector profile")
        expected = f"LIVE-WIDOWXAI-{robot_ip}"
        if args.confirm_live != expected:
            raise SystemExit(f"live output remains disabled; pass --confirm-live {expected}")
        live_robot_ips = [robot_ip]
        if guard_arms is not None:
            live_robot_ips.append(guard_arms[guard_other].robot_ip)
        try:
            workstation = run_workstation_preflight(hardware, live_robot_ips)
        except WorkstationSafetyError as exc:
            raise SystemExit(f"live workstation preflight failed: {exc}") from None
        print(
            "preflight: workstation "
            f"AC={workstation.power_supply} profile={workstation.power_profile} "
            f"robot_source={workstation.route_source_ip} "
            f"devices={list(workstation.route_devices)}"
        )
        backend = TrossenArmBackend(
            robot_ip,
            command_goal_time_s=hardware["command_goal_time_s"],
            end_effector_variant=END_EFFECTOR_PROFILE_TO_VARIANT[str(end_effector_profile)],
            required_driver_version=str(hardware["driver_version_tested"]),
        )
        stationary_backend = (
            TrossenArmBackend(
                guard_arms[guard_other].robot_ip,
                end_effector_variant=END_EFFECTOR_PROFILE_TO_VARIANT[
                    str(end_effector_profile)
                ],
                required_driver_version=str(hardware["driver_version_tested"]),
            )
            if guard_arms is not None
            else None
        )
    else:
        backend = DryRunBackend(
            home_q,
            model.joint_limits,
            gripper_position_m=float(hardware["gripper_open_m"]),
        )
        stationary_backend = None

    quest = config["quest"]
    control_gripper = bool(hardware.get("control_gripper", True))
    select_profile_from_page = page_locked_task_profile and not args.task_profile
    receiver = QuestReceiver(
        quest["websocket_url"],
        hand=None if select_profile_from_page else quest.get("hand", "left"),
        mapping_mode=None if select_profile_from_page else quest.get("mapping_mode", "real"),
    )

    connected = False
    motion_started = False
    stationary_connected = False
    stationary_state = None
    stationary_next_read_s = 0.0
    try:
        if quest.get("prepare_tabletop_tracking_via_adb", False):
            print("preflight: waking Quest tabletop tracking through ADB")
            try:
                prepare_tabletop_tracking()
            except RuntimeError as exc:
                raise HardwareSafetyError(str(exc)) from None
        receiver.start()
        print("preflight: waiting for fresh Quest tracking with grip released")
        preflight_sample = wait_for_released_quest(
            receiver,
            10.0,
            quest["stale_timeout_s"],
        )
        if select_profile_from_page:
            try:
                selected_profile = task_profile_for_quest_selection(
                    preflight_sample.hand,
                    preflight_sample.mapping_mode,
                )
                apply_task_profile(config, selected_profile)
                validate_hardware_config(config)
                if args.live:
                    validate_live_hardware_timing(config)
            except (KeyError, TypeError, ValueError, HardwareSafetyError) as exc:
                raise HardwareSafetyError(
                    f"could not lock the page-selected task profile: {exc}"
                ) from None
            quest = config["quest"]
            print(
                "preflight: page selected "
                f"{preflight_sample.hand}/{preflight_sample.mapping_mode}; "
                f"locked task profile {selected_profile}"
            )
        selected_hand = quest.get("hand", "left")
        selected_mapping_mode = quest.get("mapping_mode", "real")
        require_quest_selection(
            preflight_sample,
            selected_hand,
            selected_mapping_mode,
        )
        mapper = ClutchPoseMapper(
            quest["calibration"],
            translation_scale=quest["translation_scale"],
            rotation_scale=quest["rotation_scale"],
            position_reach_limit_m=quest["position_reach_limit_m"],
            rotation_reach_limit_rad=quest["rotation_reach_limit_rad"],
        )
        clutch = ClutchController(mapper)
        watchdog = FreshSequenceWatchdog(
            quest["stale_timeout_s"],
            quest["fresh_samples_to_recover"],
        )
        state = backend.connect()
        connected = True
        print(
            f"preflight: driver={state.driver_version} firmware={state.firmware_version} "
            f"q={np.round(state.q_arm, 3).tolist()}"
        )
        if stationary_backend is not None:
            stationary_state = stationary_backend.connect()
            stationary_connected = True
            stationary_next_read_s = time.perf_counter() + 1.0 / float(
                hardware["feedback_check_rate_hz"]
            )
            print(
                f"preflight: stationary {guard_other} arm @ "
                f"{guard_arms[guard_other].robot_ip} is READ-ONLY; "
                f"q={np.round(stationary_state.q_arm, 3).tolist()}"
            )

        combined_limits = state.joint_limits.copy()
        combined_limits[:6, 0] = np.maximum(combined_limits[:6, 0], model.joint_limits[:, 0])
        combined_limits[:6, 1] = np.minimum(combined_limits[:6, 1], model.joint_limits[:, 1])
        margin = float(hardware["joint_limit_margin_rad"])
        solver_limits = combined_limits[:6].copy()
        solver_limits[:, 0] += margin
        solver_limits[:, 1] -= margin
        try:
            solver.set_joint_limits(solver_limits)
        except ValueError as exc:
            raise HardwareSafetyError("physical limits and margin leave invalid IK limits") from exc
        if guard_model is not None:
            startup_separation = screen_selected_path(
                model,
                guard_model,
                moving_side=guard_side,
                moving_start=state.q_arm,
                moving_end=home_q,
                holding_q=stationary_state.q_arm,
                moving_gripper_m=state.gripper_position_m,
                holding_gripper_m=stationary_state.gripper_position_m,
                samples=int(hardware["startup_collision_samples"]),
                clearance_m=guard_clearance_m,
                allow_marginal_start_m=MARGINAL_START_CONTACT_M,
            )
            return_separation = screen_selected_path(
                model,
                guard_model,
                moving_side=guard_side,
                moving_start=home_q,
                moving_end=np.asarray(hardware["rest_q_rad"], dtype=float),
                holding_q=stationary_state.q_arm,
                moving_gripper_m=state.gripper_position_m,
                holding_gripper_m=stationary_state.gripper_position_m,
                samples=int(hardware["startup_collision_samples"]),
                clearance_m=guard_clearance_m,
            )
            print(
                "preflight: stationary-arm guarded startup/shutdown paths clear; "
                f"minimum separation {min(startup_separation, return_separation):.3f} m"
            )
        else:
            # Same owner-accepted marginal-start rule as the guarded path: a
            # measured rest contact up to 2 mm must never deepen, must clear,
            # and must never return; every other contact still fails closed.
            try:
                start_depth, cleared_at = screen_self_collision_path(
                    model,
                    start_q=state.q_arm,
                    end_q=home_q,
                    start_gripper_m=state.gripper_position_m,
                    samples=int(hardware["startup_collision_samples"]),
                    label=f"arm {robot_ip}",
                    allow_marginal_start_m=MARGINAL_START_CONTACT_M,
                )
            except HardwareSafetyError as exc:
                raise HardwareSafetyError(
                    f"pinned 2025 MuJoCo collision geometry rejects the startup ramp: {exc}"
                ) from None
            if cleared_at is not None:
                print(
                    f"preflight: measured {start_depth * 1000:.3f} mm rest contact never "
                    f"deepens and clears by {cleared_at * 100:.1f}% toward home"
                )
        startup_max_delta = np.asarray(
            hardware.get("startup_max_joint_delta_rad", max_delta),
            dtype=float,
        ).reshape(6)
        ramp_gate = make_startup_command_gate(state, startup_max_delta, hardware)

        backend.enable_position_control(include_gripper=False)
        motion_started = True
        backend.send_positions(
            state.q_arm,
            state.gripper_position_m,
            include_gripper=False,
        )
        print("startup: ramping from measured pose to home [0, 60, 75, -60, 0, 0] deg")
        ramp_to_home(backend, ramp_gate, state, home_q, config)
        settled = backend.read_state()
        if np.max(np.abs(settled.q_arm - home_q)) > hardware["max_feedback_error_rad"]:
            raise HardwareSafetyError("arm did not reach home within the feedback limit")
        if control_gripper:
            if stationary_backend is not None:
                stationary_state = stationary_backend.read_state()
                gripper_separation = screen_gripper_path(
                    guard_model,
                    moving_side=guard_side,
                    moving_q=home_q,
                    holding_q=stationary_state.q_arm,
                    moving_start_gripper_m=settled.gripper_position_m,
                    moving_end_gripper_m=float(hardware["gripper_open_m"]),
                    holding_gripper_m=stationary_state.gripper_position_m,
                    clearance_m=guard_clearance_m,
                    samples=int(hardware["startup_collision_samples"]),
                )
                print(
                    "preflight: stationary-arm guarded gripper-open path clear; "
                    f"minimum separation {gripper_separation:.3f} m"
                )
            settled = open_gripper_at_home(backend, model, settled, home_q, config)
        print(
            "home reached: keep grip released; "
            f"hold {selected_hand} grip when ready to teleoperate"
        )
        wait_for_released_quest(
            receiver,
            10.0,
            quest["stale_timeout_s"],
            expected_hand=selected_hand,
            expected_mapping_mode=selected_mapping_mode,
        )

        gripper_command = (
            float(hardware["gripper_open_m"])
            if control_gripper
            else float(settled.gripper_position_m)
        )
        command_gate = CommandGate(
            home_q,
            gripper_command,
            combined_limits,
            max_delta,
            joint_limit_margin_rad=hardware["joint_limit_margin_rad"],
            gripper_limits_m=teleop_gripper_limits(
                settled,
                hardware,
                control_gripper=control_gripper,
            ),
            max_gripper_delta_m=hardware["max_gripper_delta_m"],
        )
        q_command = home_q.copy()
        q_des = q_command.copy()
        gripper_des = gripper_command
        joint_limiter = limiter_from_config(
            control.get("joint_command_limits"),
            q_command,
        )
        gripper_limiter = (
            limiter_from_config(
                control.get("gripper_command_limits"),
                np.array([gripper_command]),
            )
            if control_gripper
            else None
        )
        feedforward_config = hardware.get("arm_velocity_feedforward", {}) or {}
        arm_feedforward_filter = (
            VelocityFeedforwardFilter(
                q_command,
                filter_alpha=feedforward_config["filter_alpha"],
                gain=feedforward_config["gain"],
                max_velocity=np.asarray(
                    feedforward_config["max_velocity_rad_s"],
                    dtype=float,
                ),
            )
            if feedforward_config.get("enabled", False)
            else None
        )
        q_feedforward_velocity = np.zeros(6)
        q_feedback = settled.q_arm.copy()
        gripper_feedback = float(settled.gripper_position_m)
        last_sample = None
        filtered_controller = None
        target_pose = None
        diagnostics = None

        loop_hz = float(control["loop_rate_hz"])
        period = 1.0 / loop_hz
        update_mode = control.get("update_mode", "fixed_rate")
        quest_synchronized = update_mode == "quest_synchronized"
        minimum_command_interval_s = configured_minimum_command_interval(control)
        command_spacing_stage = configured_command_spacing_stage(control)
        feedback_stride = max(1, round(loop_hz / float(hardware["feedback_check_rate_hz"])))
        feedback_period = 1.0 / float(hardware["feedback_check_rate_hz"])
        duration = resolve_demo_duration(hardware, args.duration)
        started = time.perf_counter()
        next_tick_s = started
        next_feedback_s = started
        last_command_send_s = started
        command_history = TimeAlignedCommandHistory(
            q_command,
            started,
            float(hardware.get("feedback_tracking_delay_s", 0.0)),
        )
        q_feedback_reference = q_command.copy()
        q_feedback_error = q_feedback - q_feedback_reference
        mailbox_generation = 0
        tick = 0

        mode = "LIVE" if args.live else "DRY RUN"
        output_scope = (
            "J0-J5 plus gripper"
            if control_gripper
            else "J0-J5 only; physical gripper untouched"
        )
        command_cadence = (
            f"Quest-synchronized commands (nominal {loop_hz:g} Hz)"
            if quest_synchronized
            else f"{loop_hz:g} Hz commands"
        )
        runtime_scope = (
            "no duration deadline; Ctrl+C stops"
            if np.isinf(duration)
            else f"{duration:g} s maximum runtime"
        )
        print(
            f"{mode}: {command_cadence}, {hardware['command_goal_time_s'] * 1000:g} ms "
            f"driver horizon, {output_scope}, configured pose scale and motion limits, "
            f"velocity feedforward "
            f"{'enabled' if arm_feedforward_filter is not None else 'disabled'}, "
            f"{selected_hand} grip is the deadman, "
            f"{selected_mapping_mode} mapping, {runtime_scope}"
        )
        with TelemetryLogger(args.label, config, config["telemetry"]["output_dir"]) as telemetry:
            while time.perf_counter() - started < duration:
                pre_consume_wait_s = 0.0
                pre_send_wait_s = 0.0
                if quest_synchronized:
                    if command_spacing_stage == "before_consume":
                        pre_consume_wait_s = minimum_command_spacing_wait(
                            last_command_send_s,
                            minimum_command_interval_s,
                            time.perf_counter(),
                        )
                        remaining_s = duration - (time.perf_counter() - started)
                        if pre_consume_wait_s >= remaining_s:
                            if remaining_s > 0.0:
                                time.sleep(remaining_s)
                            break
                        if pre_consume_wait_s > 0.0:
                            # Milestone-2 behavior: wait for the slot before
                            # consuming, so queued frames collapse to latest.
                            time.sleep(pre_consume_wait_s)
                    remaining_s = duration - (time.perf_counter() - started)
                    if remaining_s <= 0.0:
                        break
                    sample, mailbox_generation = receiver.mailbox.wait_take_latest(
                        mailbox_generation,
                        min(feedback_period, remaining_s),
                    )
                else:
                    sample, _ = receiver.mailbox.take_latest()
                control_consume_ns = time.perf_counter_ns()
                if sample is not None:
                    require_quest_selection(
                        sample,
                        selected_hand,
                        selected_mapping_mode,
                    )
                    last_sample = sample
                    freshness = watchdog.observe(sample)
                else:
                    freshness = watchdog.poll()

                if quest_synchronized and sample is None:
                    if last_sample is not None and not freshness.fresh:
                        robot_pose, wrist_pose = model.fk(q_feedback)
                        target_pose = clutch.update(
                            grip=last_sample.grip,
                            stream_fresh=False,
                            controller_pose=filtered_controller or last_sample.controller_pose,
                            robot_pose=robot_pose,
                            wrist_pivot=wrist_pose.position,
                            head_quaternion_wxyz=last_sample.head_quaternion_wxyz,
                        )
                        if joint_limiter is not None:
                            joint_limiter.reset(q_command)
                        if gripper_limiter is not None:
                            gripper_limiter.reset(np.array([gripper_command]))
                        if arm_feedforward_filter is not None:
                            arm_feedforward_filter.reset(q_command)
                            q_feedforward_velocity = np.zeros(6)
                        if not mapper.engaged:
                            filtered_controller = None
                            controller_filter.reset()
                    if time.perf_counter() >= next_feedback_s:
                        feedback = backend.read_state()
                        feedback_read_ns = time.perf_counter_ns()
                        q_feedback = feedback.q_arm
                        gripper_feedback = float(feedback.gripper_position_m)
                        q_feedback_reference, q_feedback_error = validate_time_aligned_feedback(
                            q_feedback,
                            feedback_read_ns / 1e9,
                            command_history,
                            hardware["max_feedback_error_rad"],
                        )
                        next_feedback_s = feedback_read_ns / 1e9 + feedback_period
                    continue

                # The clutch anchor represents where the physical arm is, while
                # the IK remains warm-started from the last safe command.
                robot_pose, wrist_pose = model.fk(q_feedback)
                pose_sample = sample if quest_synchronized else last_sample
                if pose_sample is not None:
                    filtered_controller = controller_filter.update(
                        pose_sample.controller_pose,
                        pose_sample.capture_monotonic_ms / 1000.0,
                    )
                    target_pose = clutch.update(
                        grip=pose_sample.grip,
                        stream_fresh=freshness.fresh,
                        controller_pose=filtered_controller,
                        robot_pose=robot_pose,
                        wrist_pivot=wrist_pose.position,
                        head_quaternion_wxyz=pose_sample.head_quaternion_wxyz,
                    )
                else:
                    target_pose = None

                target_active = target_pose is not None and freshness.fresh and mapper.engaged
                limiter_flags: list[str] = []
                if target_active:
                    ik_start = time.perf_counter_ns()
                    q_des, diagnostics = solver.solve(target_pose, q_command)
                    ik_end = time.perf_counter_ns()
                    if control_gripper:
                        gripper_des = trigger_to_gripper_position(
                            last_sample.trigger,
                            hardware["gripper_open_m"],
                            hardware["gripper_min_demo_m"],
                        )
                else:
                    ik_start = ik_end = 0
                    diagnostics = None
                    q_des = q_command.copy()
                    gripper_des = gripper_command
                    if joint_limiter is not None:
                        joint_limiter.reset(q_command)
                    if gripper_limiter is not None:
                        gripper_limiter.reset(np.array([gripper_command]))
                if not mapper.engaged:
                    filtered_controller = None
                    controller_filter.reset()

                if target_active:
                    limiter_elapsed_s = time.perf_counter() - last_command_send_s
                    if quest_synchronized and command_spacing_stage == "before_send":
                        # The final barrier guarantees this much outgoing time
                        # even though the limiter runs shortly before it.
                        limiter_elapsed_s = max(
                            limiter_elapsed_s,
                            minimum_command_interval_s,
                        )
                    limiter_dt = bounded_command_period(limiter_elapsed_s, loop_hz)
                    if joint_limiter is None:
                        q_command = q_des.copy()
                    else:
                        joint_result = joint_limiter.step(q_des, limiter_dt)
                        q_command = joint_result.command
                        limiter_flags.extend(joint_result.flags("joint"))
                    if gripper_limiter is None:
                        gripper_command = gripper_des
                    else:
                        gripper_result = gripper_limiter.step(np.array([gripper_des]), limiter_dt)
                        gripper_command = float(gripper_result.command[0])
                        limiter_flags.extend(gripper_result.flags("gripper"))
                    if arm_feedforward_filter is not None:
                        q_feedforward_velocity = arm_feedforward_filter.update(
                            q_command,
                            limiter_dt,
                        )
                elif arm_feedforward_filter is not None:
                    arm_feedforward_filter.reset(q_command)
                    q_feedforward_velocity = np.zeros(6)
                command_gate.validate(q_command, gripper_command)
                if model.in_self_collision(q_command, gripper_command):
                    raise HardwareSafetyError("pinned 2025 MuJoCo model predicts a self-collision")
                if stationary_backend is not None:
                    now_s = time.perf_counter()
                    if now_s >= stationary_next_read_s:
                        stationary_state = stationary_backend.read_state()
                        stationary_next_read_s = now_s + feedback_period
                    cross_arm_separation(
                        guard_model,
                        moving_side=guard_side,
                        moving_q=q_command,
                        holding_q=stationary_state.q_arm,
                        moving_gripper_m=gripper_command,
                        holding_gripper_m=stationary_state.gripper_position_m,
                        clearance_m=guard_clearance_m,
                    )
                if quest_synchronized and command_spacing_stage == "before_send":
                    pre_send_wait_s = minimum_command_spacing_wait(
                        last_command_send_s,
                        minimum_command_interval_s,
                        time.perf_counter(),
                    )
                    remaining_s = duration - (time.perf_counter() - started)
                    if pre_send_wait_s >= remaining_s:
                        if remaining_s > 0.0:
                            time.sleep(remaining_s)
                        break
                    if pre_send_wait_s > 0.0:
                        # Low-latency behavior: consume and solve immediately,
                        # then enforce the no-burst interval at the final send
                        # boundary. An overrun is never repaid with catch-up.
                        time.sleep(pre_send_wait_s)
                if arm_feedforward_filter is None:
                    backend.send_positions(
                        q_command,
                        gripper_command,
                        include_gripper=control_gripper,
                    )
                else:
                    backend.send_positions(
                        q_command,
                        gripper_command,
                        include_gripper=control_gripper,
                        arm_feedforward_velocity=q_feedforward_velocity,
                    )
                command_send_ns = time.perf_counter_ns()
                command_send_epoch_ns = time.time_ns()
                last_command_send_s = command_send_ns / 1e9
                command_history.append(last_command_send_s, q_command)

                feedback_sample_fresh = False
                feedback_read_ns = 0
                feedback_due = (
                    time.perf_counter() >= next_feedback_s
                    if quest_synchronized
                    else tick % feedback_stride == 0
                )
                if feedback_due:
                    feedback = backend.read_state()
                    feedback_read_ns = time.perf_counter_ns()
                    feedback_sample_fresh = True
                    q_feedback = feedback.q_arm
                    gripper_feedback = float(feedback.gripper_position_m)
                    q_feedback_reference, q_feedback_error = validate_time_aligned_feedback(
                        q_feedback,
                        feedback_read_ns / 1e9,
                        command_history,
                        hardware["max_feedback_error_rad"],
                    )
                    if quest_synchronized:
                        next_feedback_s = feedback_read_ns / 1e9 + feedback_period

                telemetry.log(
                    pc_epoch_ns=time.time_ns(),
                    pc_monotonic_ns=time.perf_counter_ns(),
                    quest_sequence="" if last_sample is None else last_sample.sequence,
                    quest_capture_monotonic_ms="" if last_sample is None else last_sample.capture_monotonic_ms,
                    quest_capture_epoch_ms="" if last_sample is None else last_sample.capture_epoch_ms,
                    quest_send_monotonic_ms="" if last_sample is None else last_sample.send_monotonic_ms,
                    pc_socket_arrival_monotonic_ns="" if last_sample is None else last_sample.pc_arrival_monotonic_ns,
                    pc_socket_arrival_epoch_ns="" if last_sample is None else last_sample.pc_arrival_epoch_ns,
                    control_consume_monotonic_ns=control_consume_ns,
                    ik_start_monotonic_ns=ik_start,
                    ik_end_monotonic_ns=ik_end,
                    command_send_monotonic_ns=command_send_ns,
                    command_send_epoch_ns=command_send_epoch_ns,
                    feedback_read_monotonic_ns=feedback_read_ns,
                    feedback_sample_fresh=feedback_sample_fresh,
                    command_spacing_wait_ms=(pre_consume_wait_s + pre_send_wait_s) * 1000.0,
                    command_pre_consume_wait_ms=pre_consume_wait_s * 1000.0,
                    command_pre_send_wait_ms=pre_send_wait_s * 1000.0,
                    mailbox_overwrite_count=receiver.mailbox.overwrite_count,
                    reconnect_generation="" if last_sample is None else last_sample.reconnect_generation,
                    quest_hand="" if last_sample is None else last_sample.hand,
                    quest_mapping_mode="" if last_sample is None else last_sample.mapping_mode,
                    quest_grip="" if last_sample is None else last_sample.grip,
                    quest_trigger="" if last_sample is None else last_sample.trigger,
                    stream_fresh=freshness.fresh,
                    clutch_engaged=mapper.engaged,
                    reanchor_generation=mapper.reanchor_generation,
                    raw_controller_position="" if last_sample is None else last_sample.controller_pose.position,
                    raw_controller_quaternion_wxyz="" if last_sample is None else last_sample.controller_pose.quaternion_wxyz,
                    pose_filter_rotation_alpha=controller_filter.rotation_alpha,
                    pose_filter_rotation_cutoff_hz=controller_filter.rotation_cutoff_hz,
                    head_quaternion_wxyz="" if last_sample is None else last_sample.head_quaternion_wxyz,
                    engage_head_yaw_rad="" if mapper.engage_head_yaw_rad is None else mapper.engage_head_yaw_rad,
                    mapped_target_position="" if target_pose is None else target_pose.position,
                    mapped_target_quaternion_wxyz="" if target_pose is None else target_pose.quaternion_wxyz,
                    q_des=q_des,
                    q_cmd=q_command,
                    q_feedforward_velocity=q_feedforward_velocity,
                    q_feedback=q_feedback,
                    q_feedback_reference=q_feedback_reference,
                    q_feedback_error=q_feedback_error,
                    gripper_des_m=gripper_des,
                    gripper_cmd_m=gripper_command,
                    gripper_feedback_m=gripper_feedback,
                    position_residual_m="" if diagnostics is None else diagnostics.position_residual_m,
                    orientation_residual_rad="" if diagnostics is None else diagnostics.orientation_residual_rad,
                    ik_step_norm_rad="" if diagnostics is None else diagnostics.step_norm_rad,
                    minimum_joint_limit_margin_rad="" if diagnostics is None else diagnostics.minimum_joint_limit_margin_rad,
                    ik_status="" if diagnostics is None else diagnostics.status,
                    limiter_flags="|".join(limiter_flags),
                )

                tick += 1
                if quest_synchronized:
                    continue
                # Pace only after the command has been sent. Waiting between IK
                # and send added almost one full 120 Hz period to fresh poses.
                next_tick_s += period
                sleep_for = next_tick_s - time.perf_counter()
                if sleep_for > 0.0:
                    time.sleep(sleep_for)
                else:
                    # Do not issue catch-up bursts after an overrun.
                    next_tick_s = time.perf_counter()
            print(f"telemetry: {telemetry.run_dir}")
    except KeyboardInterrupt:
        print("operator stop: returning to rest before driver cleanup")
    except (HardwareSafetyError, HardwareUnavailableError) as exc:
        raise SystemExit(f"SAFETY STOP: {exc}") from None
    finally:
        shutdown_warnings: list[str] = []
        if connected:
            returned_to_rest = False
            if motion_started and hardware.get("return_to_rest_on_exit", True):
                try:
                    if stationary_backend is not None:
                        moving_state = backend.read_state()
                        stationary_state = stationary_backend.read_state()
                        shutdown_separation = screen_selected_path(
                            model,
                            guard_model,
                            moving_side=guard_side,
                            moving_start=moving_state.q_arm,
                            moving_end=np.asarray(hardware["rest_q_rad"], dtype=float),
                            holding_q=stationary_state.q_arm,
                            moving_gripper_m=moving_state.gripper_position_m,
                            holding_gripper_m=stationary_state.gripper_position_m,
                            samples=int(hardware["startup_collision_samples"]),
                            clearance_m=guard_clearance_m,
                            moving_end_gripper_m=(
                                float(hardware["rest_gripper_m"])
                                if control_gripper
                                else moving_state.gripper_position_m
                            ),
                            allow_marginal_start_m=MARGINAL_START_CONTACT_M,
                        )
                        print(
                            "shutdown: stationary-arm guarded return path clear; "
                            f"minimum separation {shutdown_separation:.3f} m"
                        )
                        if control_gripper:
                            gripper_shutdown_separation = screen_gripper_path(
                                guard_model,
                                moving_side=guard_side,
                                moving_q=np.asarray(hardware["rest_q_rad"], dtype=float),
                                holding_q=stationary_state.q_arm,
                                moving_start_gripper_m=moving_state.gripper_position_m,
                                moving_end_gripper_m=float(hardware["rest_gripper_m"]),
                                holding_gripper_m=stationary_state.gripper_position_m,
                                clearance_m=guard_clearance_m,
                                samples=int(hardware["startup_collision_samples"]),
                            )
                            print(
                                "shutdown: stationary-arm guarded gripper-close path clear; "
                                f"minimum separation {gripper_shutdown_separation:.3f} m"
                            )
                    return_to_rest(
                        backend,
                        model,
                        config,
                        control_gripper=control_gripper,
                    )
                    returned_to_rest = True
                except Exception as exc:
                    shutdown_warnings.append(f"return to rest failed: {exc}")
            if not returned_to_rest:
                try:
                    backend.safe_hold()
                except Exception as exc:
                    shutdown_warnings.append(f"measured hold was not confirmed: {exc}")
        try:
            backend.close()
        except Exception as exc:
            shutdown_warnings.append(f"driver cleanup failed: {exc}")
        if stationary_backend is not None:
            try:
                stationary_backend.close()
            except Exception as exc:
                shutdown_warnings.append(f"stationary driver cleanup failed: {exc}")
        receiver.stop()
        for warning in shutdown_warnings:
            print(
                f"EMERGENCY SHUTDOWN WARNING: {warning}; cut controller power now",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
