from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass, field

import numpy as np

if __package__:
    from scripts.preflight_dual_hardware import offline_checks
    from scripts.run_dual_sim import dual_arm_telemetry_record
    from scripts.run_hardware import (
        gripper_feedback_tolerance,
        make_startup_command_gate,
        resolve_demo_duration,
        teleop_gripper_limits,
        validate_hardware_config,
        validate_live_hardware_timing,
        validate_time_aligned_feedback,
    )
else:
    # `python scripts/run_dual_hardware.py` puts scripts/, not the repository
    # root, on sys.path. Import sibling launchers directly in that supported
    # execution mode; package imports remain stable for tests and `-m` usage.
    from preflight_dual_hardware import offline_checks
    from run_dual_sim import dual_arm_telemetry_record
    from run_hardware import (
        gripper_feedback_tolerance,
        make_startup_command_gate,
        resolve_demo_duration,
        teleop_gripper_limits,
        validate_hardware_config,
        validate_live_hardware_timing,
        validate_time_aligned_feedback,
    )
from widowxai_quest_teleop.config import (
    DUAL_ARM_SIDES,
    DualArmConfigError,
    dual_live_confirmation_token,
    load_config,
    parse_dual_arm_config,
    require_live_dual_arm_config,
)
from widowxai_quest_teleop.dual_arm_coordinator import build_dual_arm_system
from widowxai_quest_teleop.hardware import (
    CommandGate,
    DryRunBackend,
    END_EFFECTOR_PROFILE_TO_VARIANT,
    HardwareSafetyError,
    HardwareUnavailableError,
    TrossenArmBackend,
)
from widowxai_quest_teleop.motion_limiter import (
    bounded_command_period,
    configured_command_spacing_stage,
    configured_minimum_command_interval,
    minimum_command_spacing_wait,
)
from widowxai_quest_teleop.quest_power import prepare_tabletop_tracking
from widowxai_quest_teleop.safety import TimeAlignedCommandHistory
from widowxai_quest_teleop.telemetry import DUAL_ARM_TELEMETRY_COLUMNS, TelemetryLogger
from widowxai_quest_teleop.transport import BimanualQuestReceiver
from widowxai_quest_teleop.workstation import (
    WorkstationSafetyError,
    run_workstation_preflight,
)

GRIP_RELEASED = 0.65
# A contact this shallow, at a pose the arm is physically resting in, is model
# conservatism near the zero fold rather than a real fold. For reference the
# closed gripper's own carriage pair reports ~0.18 mm and is already
# whitelisted in model.py.
DEFAULT_MARGINAL_CONTACT_M = 0.002


@dataclass
class ArmChannel:
    """One arm's physical side-channel: driver, gate, and feedback history."""

    side: str
    backend: object
    connected: bool = False
    motion_started: bool = False
    gate: CommandGate | None = None
    command_history: TimeAlignedCommandHistory | None = None
    q_feedback_reference: np.ndarray | None = None
    q_feedback_error: np.ndarray | None = None
    next_feedback_s: float = 0.0
    send_monotonic_ns: int = 0
    send_epoch_ns: int = 0
    send_duration_ms: float = 0.0
    feedback_read_monotonic_ns: int = 0
    feedback_sample_fresh: bool = False
    feedback_reference_state: str = ""
    feedback_newest_command_age_ms: float | None = None
    feedback_history_span_ms: float | None = None
    reached_rest: bool = False
    notes: list[str] = field(default_factory=list)


def reset_feedback_telemetry(channels: dict[str, ArmChannel]) -> None:
    """Start one control row with no claimed encoder read."""

    for channel in channels.values():
        channel.feedback_read_monotonic_ns = 0
        channel.feedback_sample_fresh = False
        channel.feedback_reference_state = ""
        channel.feedback_newest_command_age_ms = None
        channel.feedback_history_span_ms = None


def read_and_validate_channel_feedback(
    channel: ArmChannel,
    arm,
    hardware: dict,
    feedback_period: float,
) -> HardwareSafetyError | None:
    """Read one arm and retain its evidence even when tracking validation fails."""

    feedback = channel.backend.read_state()
    read_ns = time.perf_counter_ns()
    read_time_s = read_ns / 1e9
    channel.feedback_read_monotonic_ns = read_ns
    channel.feedback_sample_fresh = True
    arm.q_feedback = feedback.q_arm
    arm.gripper_feedback_m = float(feedback.gripper_position_m)

    # validate_time_aligned_feedback raises on the one sample that matters most.
    # Populate the channel first so the caller can write that sample before
    # re-raising the fail-closed exception.
    history = channel.command_history
    channel.q_feedback_reference = history.reference_at(read_time_s)
    channel.q_feedback_error = feedback.q_arm - channel.q_feedback_reference
    channel.feedback_reference_state = history.clamp_state(read_time_s)
    channel.feedback_newest_command_age_ms = (
        read_time_s - history.newest_time_s
    ) * 1000.0
    channel.feedback_history_span_ms = (
        history.newest_time_s - history.oldest_time_s
    ) * 1000.0
    channel.next_feedback_s = read_time_s + feedback_period
    try:
        validate_time_aligned_feedback(
            feedback.q_arm,
            read_time_s,
            history,
            hardware["max_feedback_error_rad"],
        )
    except HardwareSafetyError as exc:
        return exc
    return None


def read_due_feedback(
    channels: dict[str, ArmChannel],
    arms: dict[str, object],
    hardware: dict,
    feedback_period: float,
) -> HardwareSafetyError | None:
    """Read due channels, stopping at the first fail-closed tracking fault."""

    for side in DUAL_ARM_SIDES:
        channel = channels[side]
        if time.perf_counter() < channel.next_feedback_s:
            continue
        fault = read_and_validate_channel_feedback(
            channel, arms[side], hardware, feedback_period
        )
        if fault is not None:
            return fault
    return None


def add_feedback_telemetry(
    record: dict[str, object],
    channels: dict[str, ArmChannel],
    *,
    fault: HardwareSafetyError | None = None,
) -> dict[str, object]:
    """Attach truthful per-arm feedback fields to a coordinated tick."""

    for side in DUAL_ARM_SIDES:
        channel = channels[side]
        record[f"{side}_q_feedback_reference"] = channel.q_feedback_reference
        record[f"{side}_q_feedback_error"] = channel.q_feedback_error
        record[f"{side}_feedback_sample_fresh"] = channel.feedback_sample_fresh
        record[f"{side}_feedback_read_monotonic_ns"] = channel.feedback_read_monotonic_ns
        record[f"{side}_feedback_reference_state"] = channel.feedback_reference_state
        record[f"{side}_feedback_newest_command_age_ms"] = (
            channel.feedback_newest_command_age_ms
        )
        record[f"{side}_feedback_history_span_ms"] = channel.feedback_history_span_ms
    if fault is not None:
        record["fault_reason"] = str(fault)
    return record


def command_send_telemetry(
    channels: dict[str, ArmChannel], *, command_sent: bool
) -> tuple[dict[str, int], dict[str, int]]:
    """Return this row's sends without reusing timestamps on idle rows."""

    monotonic = {
        side: channels[side].send_monotonic_ns if command_sent else 0
        for side in DUAL_ARM_SIDES
    }
    epoch = {
        side: channels[side].send_epoch_ns if command_sent else 0
        for side in DUAL_ARM_SIDES
    }
    return monotonic, epoch


def command_send_duration_telemetry(
    channels: dict[str, ArmChannel], *, command_sent: bool
) -> dict[str, float | str]:
    """Return each driver's blocking call duration for this row only."""

    return {
        side: channels[side].send_duration_ms if command_sent else ""
        for side in DUAL_ARM_SIDES
    }


def send_dual_commands(
    channels: dict[str, ArmChannel],
    arms: dict[str, object],
    *,
    control_gripper: bool,
) -> float:
    """Send both sides, retaining each call duration and the resulting skew."""

    for side in DUAL_ARM_SIDES:
        channel = channels[side]
        arm = arms[side]
        send_started_ns = time.perf_counter_ns()
        if arm.feedforward_filter is None:
            channel.backend.send_positions(
                arm.q_command,
                arm.gripper_command_m,
                include_gripper=control_gripper,
            )
        else:
            channel.backend.send_positions(
                arm.q_command,
                arm.gripper_command_m,
                include_gripper=control_gripper,
                arm_feedforward_velocity=arm.feedforward_velocity,
            )
        channel.send_monotonic_ns = time.perf_counter_ns()
        channel.send_epoch_ns = time.time_ns()
        channel.send_duration_ms = (
            channel.send_monotonic_ns - send_started_ns
        ) / 1e6
        channel.command_history.append(
            channel.send_monotonic_ns / 1e9, arm.q_command
        )
    sends = [channels[side].send_monotonic_ns for side in DUAL_ARM_SIDES]
    return (max(sends) - min(sends)) / 1e9


def wait_for_released_bimanual(
    receiver: BimanualQuestReceiver,
    timeout_s: float,
    stale_timeout_s: float,
    *,
    require_both: bool = True,
):
    """Require live tracking on both controllers with both grips released.

    Both are demanded before any position mode is enabled. Starting with one
    controller tracked would arm an arm whose operator input is not yet proven.
    """

    from widowxai_quest_teleop.safety import FreshSequenceWatchdog

    deadline = time.perf_counter() + float(timeout_s)
    watchdogs = {hand: FreshSequenceWatchdog(stale_timeout_s, 3) for hand in DUAL_ARM_SIDES}
    while time.perf_counter() < deadline:
        sample, _ = receiver.mailbox.take_latest()
        if sample is not None:
            ready = []
            for hand in DUAL_ARM_SIDES:
                projected = sample.for_hand(hand)
                if projected is None:
                    watchdogs[hand].poll()
                    continue
                freshness = watchdogs[hand].observe(projected)
                if freshness.fresh and projected.grip < GRIP_RELEASED:
                    ready.append(hand)
            if (len(ready) == len(DUAL_ARM_SIDES)) or (not require_both and ready):
                return sample
        time.sleep(0.01)
    raise HardwareSafetyError(
        "dual Quest preflight failed: enter WebXR in bimanual mode, keep BOTH grips "
        "released, and provide fresh tracking on both controllers"
    )


def ramp_both_to_home(
    channels: dict[str, ArmChannel],
    states: dict[str, object],
    home_q: np.ndarray,
    collision_model,
    config: dict,
) -> None:
    """Ramp both arms to home in lockstep, screening the combined state.

    Both arms advance on the same step index so the combined scene is checked
    at every intermediate pose that will actually exist. Ramping one arm to
    completion first would leave the other at an unscreened intermediate pose.
    """

    hardware = config["hardware"]
    duration = float(hardware["startup_ramp_duration_s"])
    rate = float(hardware["startup_ramp_rate_hz"])
    steps = max(1, round(duration * rate))
    period = 1.0 / rate
    feedback_stride = max(1, round(rate / float(hardware["feedback_check_rate_hz"])))
    start_q = {side: states[side].q_arm.copy() for side in DUAL_ARM_SIDES}
    start_gripper = {side: float(states[side].gripper_position_m) for side in DUAL_ARM_SIDES}

    next_tick = time.perf_counter()
    for index in range(1, steps + 1):
        alpha = index / steps
        commanded = {
            side: start_q[side] + alpha * (home_q - start_q[side]) for side in DUAL_ARM_SIDES
        }
        report = collision_model.check(commanded, start_gripper)
        if report.colliding:
            raise HardwareSafetyError(
                f"combined scene rejects the startup ramp at {alpha * 100:.1f}%: "
                f"{report.describe()}"
            )
        for side in DUAL_ARM_SIDES:
            channel = channels[side]
            channel.gate.validate(commanded[side], start_gripper[side])
            channel.backend.send_positions(
                commanded[side], start_gripper[side], include_gripper=False
            )
        if index % feedback_stride == 0 or index == steps:
            for side in DUAL_ARM_SIDES:
                feedback = channels[side].backend.read_state()
                if (
                    np.max(np.abs(feedback.q_arm - commanded[side]))
                    > hardware["max_feedback_error_rad"]
                ):
                    raise HardwareSafetyError(
                        f"{side} arm joint tracking error exceeded the demo limit "
                        "during the startup ramp"
                    )
        next_tick += period
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0.0:
            time.sleep(sleep_for)
        else:
            next_tick = time.perf_counter()


def return_both_to_rest(
    channels: dict[str, ArmChannel],
    collision_model,
    config: dict,
    *,
    control_gripper: bool,
) -> None:
    """Return both arms to rest one at a time, screening each move.

    Sequential by choice. A simultaneous shutdown path is not proven safe for
    an arbitrary intermediate state, and shutdown is the one phase where the
    slower option costs nothing.
    """

    hardware = config["hardware"]
    rest_q = np.asarray(hardware["rest_q_rad"], dtype=float).reshape(6)
    rest_gripper = float(hardware["rest_gripper_m"])
    duration = float(hardware.get("shutdown_move_duration_s", 2.0))
    samples = int(hardware["startup_collision_samples"])

    for side in DUAL_ARM_SIDES:
        channel = channels[side]
        if not channel.connected or not channel.motion_started:
            continue
        other = [name for name in DUAL_ARM_SIDES if name != side][0]
        state = channel.backend.read_state()
        other_state = channels[other].backend.read_state() if channels[other].connected else None
        holding_q = rest_q if other_state is None else other_state.q_arm
        holding_gripper = (
            rest_gripper if other_state is None else float(other_state.gripper_position_m)
        )

        found = collision_model.first_collision_on_path(
            {side: state.q_arm, other: holding_q},
            {side: rest_q, other: holding_q},
            start_gripper={side: state.gripper_position_m, other: holding_gripper},
            end_gripper={
                side: rest_gripper if control_gripper else state.gripper_position_m,
                other: holding_gripper,
            },
            samples=samples,
        )
        if found is not None:
            alpha, report = found
            raise HardwareSafetyError(
                f"combined scene rejects the {side} return-to-rest near "
                f"{alpha * 100:.1f}%: {report.describe()}"
            )
        print(f"shutdown: returning {side} arm to rest with a {duration:g} s blocking move")
        channel.backend.move_to_rest(
            rest_q, rest_gripper, duration_s=duration, include_gripper=control_gripper
        )
        settled = channel.backend.read_state()
        if np.max(np.abs(settled.q_arm - rest_q)) > hardware["max_feedback_error_rad"]:
            raise HardwareSafetyError(f"{side} arm did not reach rest within the feedback limit")
        if control_gripper and (
            abs(settled.gripper_position_m - rest_gripper)
            > gripper_feedback_tolerance(settled, hardware)
        ):
            raise HardwareSafetyError(f"{side} gripper did not reach its rest position")
        channel.reached_rest = True
        print(f"rest reached: {side} arm")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed bimanual WidowXAI demo")
    parser.add_argument("--config", default="configs/dual_widowxai.yaml")
    parser.add_argument("--live", action="store_true", help="use the official Trossen driver on BOTH arms")
    parser.add_argument(
        "--confirm-live",
        default="",
        help="must equal LIVE-WIDOWXAI-DUAL-<left-ip>-<right-ip>",
    )
    parser.add_argument(
        "--end-effector-profile",
        choices=tuple(END_EFFECTOR_PROFILE_TO_VARIANT),
        help="overrides the selected config for both arms",
    )
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--label", default="dual-widowxai-hardware")
    parser.add_argument(
        "--accept-unvalidated-calibrations",
        action="store_true",
        help=(
            "explicit operator override for the calibration-acceptance gate ONLY; "
            "requires --live and the dual token, records the override in the run "
            "evidence, and relaxes no other gate"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        arms_config = parse_dual_arm_config(config)
    except DualArmConfigError as exc:
        raise SystemExit(f"invalid dual-arm configuration: {exc}") from None

    hardware = config["hardware"]
    control = config["control"]
    if not hardware.get("enabled") or not hardware.get("require_explicit_enable"):
        raise SystemExit("dual-arm hardware configuration is not explicitly gated")
    try:
        validate_hardware_config(config)
    except (KeyError, TypeError, ValueError, HardwareSafetyError) as exc:
        raise SystemExit(f"invalid hardware safety configuration: {exc}") from None
    if not np.isfinite(args.duration) or args.duration < 0.0:
        raise SystemExit("--duration must be finite and nonnegative")

    control_gripper = bool(hardware.get("control_gripper", True))
    home_q = np.asarray(config["model"]["simulation_start_q_rad"], dtype=float).reshape(6)
    gripper_open = float(hardware["gripper_open_m"])

    try:
        offline_checks(config, arms_config)
    except (HardwareSafetyError, ValueError) as exc:
        raise SystemExit(f"SAFETY STOP: offline dual-arm checks failed: {exc}") from None

    arms, collision_model, coordinator = build_dual_arm_system(
        config,
        arms_config,
        initial_q=home_q,
        initial_gripper_m=gripper_open,
        control_gripper=control_gripper,
    )
    home_q = arms["left"].model.clamp_joints(home_q)

    end_effector_profile = args.end_effector_profile or hardware.get("end_effector_profile")
    if args.live:
        try:
            validate_live_hardware_timing(config)
            pending = require_live_dual_arm_config(
                config,
                arms_config,
                allow_unvalidated_calibrations=args.accept_unvalidated_calibrations,
            )
        except (HardwareSafetyError, DualArmConfigError) as exc:
            raise SystemExit(f"live dual-arm output remains disabled: {exc}") from None
        if pending:
            print(
                "OPERATOR OVERRIDE: driving live on UNVALIDATED calibrations "
                f"{pending}. Every other gate stays active. The twelve signed "
                "motions have not been validated on hardware; a controller axis "
                "may map to an unexpected arm direction.",
                file=sys.stderr,
                flush=True,
            )
        if platform.system() not in ("Linux", "Darwin"):
            raise SystemExit(
                "live hardware is unavailable on native Windows; no arm connection was attempted"
            )
        if not end_effector_profile:
            raise SystemExit("--end-effector-profile is required for a live dual-arm demo")
        expected = dual_live_confirmation_token(arms_config)
        if args.confirm_live != expected:
            raise SystemExit(
                f"live dual-arm output remains disabled; pass --confirm-live {expected}. "
                "A single-arm LIVE-WIDOWXAI-<ip> token never enables two arms."
            )
        try:
            workstation = run_workstation_preflight(
                hardware,
                [arms_config[side].robot_ip for side in DUAL_ARM_SIDES],
            )
        except WorkstationSafetyError as exc:
            raise SystemExit(f"live workstation preflight failed: {exc}") from None
        print(
            "preflight: workstation "
            f"AC={workstation.power_supply} profile={workstation.power_profile} "
            f"robot_source={workstation.route_source_ip} "
            f"devices={list(workstation.route_devices)}"
        )
        variant = END_EFFECTOR_PROFILE_TO_VARIANT[str(end_effector_profile)]
        channels = {
            side: ArmChannel(
                side,
                TrossenArmBackend(
                    arms_config[side].robot_ip,
                    command_goal_time_s=hardware["command_goal_time_s"],
                    end_effector_variant=variant,
                    required_driver_version=str(hardware["driver_version_tested"]),
                ),
            )
            for side in DUAL_ARM_SIDES
        }
    else:
        channels = {
            side: ArmChannel(
                side,
                DryRunBackend(home_q, arms[side].model.joint_limits, gripper_position_m=gripper_open),
            )
            for side in DUAL_ARM_SIDES
        }

    receiver = BimanualQuestReceiver(
        config["quest"]["websocket_url"],
        # Keyed by controller hand, not arm side: the transport checks each
        # hand's page selection, and the two differ whenever a profile swaps
        # the hand-to-arm assignment.
        mapping_modes={
            arms_config[side].controller_hand: arms_config[side].mapping_mode
            for side in DUAL_ARM_SIDES
        },
    )

    quest = config["quest"]
    max_skew_s = float(config["safety"]["max_command_skew_s"])
    fault_reason = ""

    try:
        if quest.get("prepare_tabletop_tracking_via_adb", False):
            print("preflight: waking Quest tabletop tracking through ADB")
            try:
                prepare_tabletop_tracking()
            except RuntimeError as exc:
                raise HardwareSafetyError(str(exc)) from None
        receiver.start()
        print("preflight: waiting for fresh tracking on BOTH controllers with grips released")
        wait_for_released_bimanual(receiver, 15.0, quest["stale_timeout_s"])

        # Connect both controllers before either enters position mode. A
        # partial connection must never leave one arm armed.
        states = {}
        for side in DUAL_ARM_SIDES:
            channel = channels[side]
            states[side] = channel.backend.connect()
            channel.connected = True
            print(
                f"preflight: {side} arm @ {arms_config[side].robot_ip} "
                f"driver={states[side].driver_version} firmware={states[side].firmware_version} "
                f"q={np.round(states[side].q_arm, 3).tolist()}"
            )

        startup_max_delta_bootstrap = np.asarray(
            hardware.get("startup_max_joint_delta_rad", config["ik"]["max_dq_per_joint_rad"]),
            dtype=float,
        ).reshape(6)
        measured = {side: states[side].q_arm.copy() for side in DUAL_ARM_SIDES}
        grippers = {side: float(states[side].gripper_position_m) for side in DUAL_ARM_SIDES}

        # An unpowered arm sags under gravity, so the pose measured at connect
        # is where it drooped to, not where it was parked. That droop can put a
        # joint marginally inside the model's collision geometry near zero. The
        # arm is physically resting there, so lift it off the droop first, then
        # screen the real ramp from the held pose with no exemption at all.
        marginal_limit_m = float(
            hardware.get("startup_marginal_contact_m", DEFAULT_MARGINAL_CONTACT_M)
        )
        try:
            alpha, settle_q = collision_model.relieving_step(
                measured,
                {side: home_q for side in DUAL_ARM_SIDES},
                grippers,
                marginal_limit_m=marginal_limit_m,
                samples=int(hardware["startup_collision_samples"]),
            )
        except ValueError as exc:
            raise HardwareSafetyError(
                f"combined scene rejects the measured startup pose: {exc}"
            ) from None

        if alpha > 0.0:
            print(
                f"startup: measured pose is {alpha * 100:.1f}% inside the model's "
                "near-zero geometry (gravity droop); energising and lifting off it first"
            )
            for side in DUAL_ARM_SIDES:
                channels[side].gate = make_startup_command_gate(
                    states[side], startup_max_delta_bootstrap, hardware
                )
            for side in DUAL_ARM_SIDES:
                channels[side].backend.enable_position_control(include_gripper=False)
                channels[side].motion_started = True
                channels[side].backend.send_positions(
                    measured[side], grippers[side], include_gripper=False
                )
            for side in DUAL_ARM_SIDES:
                channels[side].backend.move_to_rest(
                    settle_q[side],
                    grippers[side],
                    duration_s=float(hardware.get("startup_settle_duration_s", 2.0)),
                    include_gripper=False,
                )
            # Re-measure from the actively held pose. This, not the drooped
            # reading, is what the startup ramp is screened against.
            for side in DUAL_ARM_SIDES:
                states[side] = channels[side].backend.read_state()
                measured[side] = states[side].q_arm.copy()
                grippers[side] = float(states[side].gripper_position_m)
                print(
                    f"        {side}: held q = {np.round(states[side].q_arm, 6).tolist()}"
                )
            held_depth = collision_model.deepest_contact_m(measured, grippers)
            if held_depth > 0.0:
                raise HardwareSafetyError(
                    f"arms still report a {held_depth * 1000:.3f} mm model contact after "
                    "the settle move; refusing to ramp"
                )

        found = collision_model.first_collision_on_path(
            measured,
            {side: home_q for side in DUAL_ARM_SIDES},
            start_gripper=grippers,
            end_gripper=grippers,
            samples=int(hardware["startup_collision_samples"]),
        )
        if found is not None:
            alpha, report = found
            raise HardwareSafetyError(
                f"combined scene rejects the measured startup ramp near {alpha * 100:.1f}%: "
                f"{report.describe()}"
            )

        margin = float(hardware["joint_limit_margin_rad"])
        startup_max_delta = np.asarray(
            hardware.get("startup_max_joint_delta_rad", config["ik"]["max_dq_per_joint_rad"]),
            dtype=float,
        ).reshape(6)
        combined_limits = {}
        for side in DUAL_ARM_SIDES:
            limits = states[side].joint_limits.copy()
            model_limits = arms[side].model.joint_limits
            limits[:6, 0] = np.maximum(limits[:6, 0], model_limits[:, 0])
            limits[:6, 1] = np.minimum(limits[:6, 1], model_limits[:, 1])
            combined_limits[side] = limits
            solver_limits = limits[:6].copy()
            solver_limits[:, 0] += margin
            solver_limits[:, 1] -= margin
            try:
                arms[side].solver.set_joint_limits(solver_limits)
            except ValueError as exc:
                raise HardwareSafetyError(
                    f"{side} physical limits and margin leave invalid IK limits"
                ) from exc
            channels[side].gate = make_startup_command_gate(
                states[side], startup_max_delta, hardware
            )

        for side in DUAL_ARM_SIDES:
            channels[side].backend.enable_position_control(include_gripper=False)
            channels[side].motion_started = True
            channels[side].backend.send_positions(
                states[side].q_arm, states[side].gripper_position_m, include_gripper=False
            )
        print("startup: ramping BOTH arms from measured pose to home in lockstep")
        ramp_both_to_home(channels, states, home_q, collision_model, config)

        settled = {}
        for side in DUAL_ARM_SIDES:
            state = channels[side].backend.read_state()
            if np.max(np.abs(state.q_arm - home_q)) > hardware["max_feedback_error_rad"]:
                raise HardwareSafetyError(f"{side} arm did not reach home within the feedback limit")
            settled[side] = state

        if control_gripper:
            for side in DUAL_ARM_SIDES:
                duration = float(hardware.get("startup_gripper_move_duration_s", 2.0))
                print(f"startup: opening {side} gripper to {gripper_open:.3f} m")
                channels[side].backend.move_gripper_blocking(gripper_open, duration_s=duration)
                state = channels[side].backend.read_state()
                if (
                    abs(state.gripper_position_m - gripper_open)
                    > gripper_feedback_tolerance(state, hardware)
                ):
                    raise HardwareSafetyError(f"{side} gripper did not reach open")
                settled[side] = state

        print("home reached on both arms: keep BOTH grips released")
        wait_for_released_bimanual(receiver, 15.0, quest["stale_timeout_s"])

        started = time.perf_counter()
        for side in DUAL_ARM_SIDES:
            channel = channels[side]
            gripper_command = (
                gripper_open if control_gripper else float(settled[side].gripper_position_m)
            )
            channel.gate = CommandGate(
                home_q,
                gripper_command,
                combined_limits[side],
                np.asarray(config["ik"]["max_dq_per_joint_rad"], dtype=float),
                joint_limit_margin_rad=margin,
                gripper_limits_m=teleop_gripper_limits(
                    settled[side], hardware, control_gripper=control_gripper
                ),
                max_gripper_delta_m=hardware["max_gripper_delta_m"],
            )
            channel.command_history = TimeAlignedCommandHistory(
                home_q, started, float(hardware.get("feedback_tracking_delay_s", 0.0))
            )
            channel.q_feedback_reference = home_q.copy()
            channel.q_feedback_error = np.zeros(6)
            channel.next_feedback_s = started
            arms[side].q_command = home_q.copy()
            arms[side].q_des = home_q.copy()
            arms[side].q_feedback = settled[side].q_arm.copy()
            arms[side].gripper_command_m = gripper_command
            arms[side].gripper_feedback_m = float(settled[side].gripper_position_m)

        loop_hz = float(control["loop_rate_hz"])
        period = 1.0 / loop_hz
        quest_synchronized = control.get("update_mode", "fixed_rate") == "quest_synchronized"
        minimum_command_interval_s = configured_minimum_command_interval(control)
        command_spacing_stage = configured_command_spacing_stage(control)
        feedback_period = 1.0 / float(hardware["feedback_check_rate_hz"])
        duration = resolve_demo_duration(hardware, args.duration)
        next_tick_s = started
        last_command_send_s = started
        mailbox_generation = 0

        mode = "LIVE" if args.live else "DRY RUN"
        print(
            f"{mode} BIMANUAL: {loop_hz:g} Hz, "
            f"{hardware['command_goal_time_s'] * 1000:g} ms driver horizon, "
            f"left<-left controller @ {arms_config['left'].robot_ip}, "
            f"right<-right controller @ {arms_config['right'].robot_ip}, "
            f"each grip is that arm's deadman, "
            f"{'no deadline' if np.isinf(duration) else f'{duration:g} s maximum'}"
        )

        with TelemetryLogger(
            args.label,
            config,
            config["telemetry"]["output_dir"],
            columns=DUAL_ARM_TELEMETRY_COLUMNS,
            ik_status_columns=tuple(f"{side}_ik_status" for side in DUAL_ARM_SIDES),
            strict_columns=True,
        ) as telemetry:
            while time.perf_counter() - started < duration:
                pre_consume_wait_s = 0.0
                pre_send_wait_s = 0.0
                if quest_synchronized:
                    if command_spacing_stage == "before_consume":
                        pre_consume_wait_s = minimum_command_spacing_wait(
                            last_command_send_s, minimum_command_interval_s, time.perf_counter()
                        )
                        if pre_consume_wait_s > 0.0:
                            time.sleep(pre_consume_wait_s)
                    remaining_s = duration - (time.perf_counter() - started)
                    if remaining_s <= 0.0:
                        break
                    sample, mailbox_generation = receiver.mailbox.wait_take_latest(
                        mailbox_generation, min(feedback_period, remaining_s)
                    )
                else:
                    sample, _ = receiver.mailbox.take_latest()
                consume_ns = time.perf_counter_ns()
                reset_feedback_telemetry(channels)

                limiter_elapsed_s = time.perf_counter() - last_command_send_s
                if quest_synchronized and command_spacing_stage == "before_send":
                    limiter_elapsed_s = max(limiter_elapsed_s, minimum_command_interval_s)
                limiter_dt = bounded_command_period(limiter_elapsed_s, loop_hz)

                tick = coordinator.step(
                    sample,
                    limiter_dt=limiter_dt,
                    robot_q_source={
                        side: arms[side].q_feedback for side in DUAL_ARM_SIDES
                    },
                )

                idle_hold = (
                    quest_synchronized
                    and sample is None
                    and not any(proposal.active for proposal in tick.proposals.values())
                )
                loop_fault: HardwareSafetyError | None = None

                if not idle_hold:
                    # Gate both arms before either is sent. A gate rejection on
                    # one arm must not leave the other already commanded.
                    for side in DUAL_ARM_SIDES:
                        channels[side].gate.validate(
                            arms[side].q_command,
                            arms[side].gripper_command_m,
                            update=False,
                        )
                    for side in DUAL_ARM_SIDES:
                        channels[side].gate.validate(
                            arms[side].q_command,
                            arms[side].gripper_command_m,
                        )

                    if quest_synchronized and command_spacing_stage == "before_send":
                        pre_send_wait_s = minimum_command_spacing_wait(
                            last_command_send_s,
                            minimum_command_interval_s,
                            time.perf_counter(),
                        )
                        if pre_send_wait_s > 0.0:
                            time.sleep(pre_send_wait_s)

                    skew_s = send_dual_commands(
                        channels,
                        arms,
                        control_gripper=control_gripper,
                    )
                    if loop_fault is None and skew_s > max_skew_s:
                        loop_fault = HardwareSafetyError(
                            f"measured dual-arm command skew {skew_s * 1000:.3f} ms exceeded "
                            f"the configured limit {max_skew_s * 1000:.3f} ms"
                        )
                    last_command_send_s = max(
                        channels[side].send_monotonic_ns for side in DUAL_ARM_SIDES
                    ) / 1e9

                feedback_fault = None
                if loop_fault is None:
                    feedback_fault = read_due_feedback(
                        channels, arms, hardware, feedback_period
                    )
                fault = loop_fault or feedback_fault

                send_ns_by_side, send_epoch_ns_by_side = command_send_telemetry(
                    channels, command_sent=not idle_hold
                )
                send_duration_ms_by_side = command_send_duration_telemetry(
                    channels, command_sent=not idle_hold
                )

                record = dual_arm_telemetry_record(
                    tick,
                    consume_ns=consume_ns,
                    send_ns_by_side=send_ns_by_side,
                    send_epoch_ns_by_side=send_epoch_ns_by_side,
                    sample=sample,
                    overwrite_count=receiver.mailbox.overwrite_count,
                    pre_consume_wait_s=pre_consume_wait_s,
                    pre_send_wait_s=pre_send_wait_s,
                    arms=arms,
                    send_duration_ms_by_side=send_duration_ms_by_side,
                )
                telemetry.log(
                    **add_feedback_telemetry(record, channels, fault=fault)
                )
                if fault is not None:
                    raise fault

                if quest_synchronized:
                    continue
                next_tick_s += period
                sleep_for = next_tick_s - time.perf_counter()
                if sleep_for > 0.0:
                    time.sleep(sleep_for)
                else:
                    next_tick_s = time.perf_counter()
            print(f"telemetry: {telemetry.run_dir}")
    except KeyboardInterrupt:
        print("operator stop: returning both arms to rest before driver cleanup")
    except (HardwareSafetyError, HardwareUnavailableError) as exc:
        fault_reason = str(exc)
        print(f"SAFETY STOP: {exc}", file=sys.stderr, flush=True)
    finally:
        shutdown_warnings: list[str] = []
        any_connected = any(channel.connected for channel in channels.values())
        if any_connected and hardware.get("return_to_rest_on_exit", True):
            try:
                return_both_to_rest(
                    channels, collision_model, config, control_gripper=control_gripper
                )
            except Exception as exc:  # noqa: BLE001 - shutdown must not mask
                shutdown_warnings.append(f"return to rest failed: {exc}")
        for side in DUAL_ARM_SIDES:
            channel = channels[side]
            if channel.connected and not channel.reached_rest:
                try:
                    channel.backend.safe_hold()
                except Exception as exc:  # noqa: BLE001
                    shutdown_warnings.append(f"{side} measured hold was not confirmed: {exc}")
        for side in DUAL_ARM_SIDES:
            try:
                channels[side].backend.close()
            except Exception as exc:  # noqa: BLE001
                shutdown_warnings.append(f"{side} driver cleanup failed: {exc}")
        receiver.stop()
        print(
            "shutdown summary: "
            + ", ".join(
                f"{side}[connected={channels[side].connected} "
                f"moved={channels[side].motion_started} rest={channels[side].reached_rest}]"
                for side in DUAL_ARM_SIDES
            )
        )
        for warning in shutdown_warnings:
            print(
                f"EMERGENCY SHUTDOWN WARNING: {warning}; cut controller power now",
                file=sys.stderr,
                flush=True,
            )
        if fault_reason:
            raise SystemExit(f"SAFETY STOP: {fault_reason}")


if __name__ == "__main__":
    main()
