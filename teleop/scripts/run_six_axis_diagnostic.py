from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

if __package__:
    from scripts.run_hardware import (
        make_startup_command_gate,
        ramp_to_home,
        return_to_rest,
        teleop_gripper_limits,
    )
else:
    from run_hardware import (
        make_startup_command_gate,
        ramp_to_home,
        return_to_rest,
        teleop_gripper_limits,
    )
from widowxai_quest_teleop.config import load_config, resolve_project_path
from widowxai_quest_teleop.decoupled_ik import DecoupledIK
from widowxai_quest_teleop.hardware import (
    CommandGate,
    DryRunBackend,
    END_EFFECTOR_PROFILE_TO_VARIANT,
    HardwareSafetyError,
    HardwareUnavailableError,
    TrossenArmBackend,
)
from widowxai_quest_teleop.model import WidowXAIModel
from widowxai_quest_teleop.motion_diagnostic import (
    DIAGNOSTIC_MOTIONS,
    DiagnosticPlan,
    PlanDynamics,
    build_six_axis_plan,
    validate_plan_dynamics,
)


def _finite_positive(name: str, raw: object) -> float:
    value = float(raw)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def validate_diagnostic_config(config: dict) -> None:
    diagnostic = config["diagnostic"]
    hardware = config["hardware"]
    positive_names = (
        "command_rate_hz",
        "command_goal_time_s",
        "translation_amplitude_m",
        "rotation_amplitude_rad",
        "segment_duration_s",
        "feedback_rate_hz",
        "maximum_endpoint_position_error_m",
        "maximum_endpoint_orientation_error_rad",
        "maximum_tracking_error_rad",
        "maximum_loop_overrun_s",
    )
    for name in positive_names:
        _finite_positive(f"diagnostic.{name}", diagnostic[name])
    pause = float(diagnostic["pause_duration_s"])
    if not np.isfinite(pause) or pause < 0.0:
        raise ValueError("diagnostic.pause_duration_s must be finite and nonnegative")
    if int(diagnostic["maximum_ik_iterations"]) < 1:
        raise ValueError("diagnostic.maximum_ik_iterations must be positive")

    rate = float(diagnostic["command_rate_hz"])
    goal_time = float(diagnostic["command_goal_time_s"])
    if goal_time <= 0.001 and rate < 300.0:
        raise ValueError("driver interpolation cannot be disabled below 300 Hz")
    if goal_time > 0.2:
        raise ValueError("diagnostic command goal time must not exceed 200 ms")
    if float(diagnostic["translation_amplitude_m"]) > 0.050:
        raise ValueError("diagnostic translation amplitude is capped at 50 mm")
    if float(diagnostic["rotation_amplitude_rad"]) > np.deg2rad(12.0):
        raise ValueError("diagnostic rotation amplitude is capped at 12 degrees")
    if float(diagnostic["segment_duration_s"]) < 0.75:
        raise ValueError("diagnostic segments must last at least 0.75 s")
    if float(diagnostic["feedback_rate_hz"]) > rate:
        raise ValueError("diagnostic feedback rate cannot exceed the command rate")

    for name in (
        "maximum_joint_step_rad",
        "maximum_joint_velocity_rad_s",
        "maximum_joint_acceleration_rad_s2",
        "maximum_joint_jerk_rad_s3",
    ):
        values = np.asarray(diagnostic[name], dtype=float).reshape(6)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"diagnostic.{name} must contain six finite positive values")

    if not hardware.get("enabled") or not hardware.get("require_explicit_enable"):
        raise ValueError("hardware diagnostic configuration is not explicitly gated")
    if hardware.get("control_gripper", True):
        raise ValueError("six-axis diagnostic must leave the gripper disabled")


def build_plan_for_state(
    config: dict,
    model: WidowXAIModel,
    solver: DecoupledIK,
    state,
) -> tuple[DiagnosticPlan, PlanDynamics, np.ndarray]:
    diagnostic = config["diagnostic"]
    hardware = config["hardware"]
    home_q = np.asarray(config["model"]["simulation_start_q_rad"], dtype=float).reshape(6)

    combined_limits = state.joint_limits.copy()
    combined_limits[:6, 0] = np.maximum(combined_limits[:6, 0], model.joint_limits[:, 0])
    combined_limits[:6, 1] = np.minimum(combined_limits[:6, 1], model.joint_limits[:, 1])
    solver_limits = combined_limits[:6].copy()
    solver_limits[:, 0] += float(hardware["joint_limit_margin_rad"])
    solver_limits[:, 1] -= float(hardware["joint_limit_margin_rad"])
    solver.set_joint_limits(solver_limits)
    if np.any(home_q < solver_limits[:, 0]) or np.any(home_q > solver_limits[:, 1]):
        raise HardwareSafetyError("home pose is outside the diagnostic joint-limit margin")

    startup_collision = model.first_self_collision_on_path(
        state.q_arm,
        home_q,
        start_gripper_q=state.gripper_position_m,
        end_gripper_q=state.gripper_position_m,
        samples=int(hardware["startup_collision_samples"]),
    )
    if startup_collision is not None:
        raise HardwareSafetyError(
            "MuJoCo rejects the measured-to-home path near "
            f"{startup_collision * 100.0:.1f}%"
        )

    plan = build_six_axis_plan(
        model,
        solver,
        home_q,
        gripper_position_m=state.gripper_position_m,
        rate_hz=float(diagnostic["command_rate_hz"]),
        segment_duration_s=float(diagnostic["segment_duration_s"]),
        pause_duration_s=float(diagnostic["pause_duration_s"]),
        translation_amplitude_m=float(diagnostic["translation_amplitude_m"]),
        rotation_amplitude_rad=float(diagnostic["rotation_amplitude_rad"]),
        maximum_endpoint_position_error_m=float(
            diagnostic["maximum_endpoint_position_error_m"]
        ),
        maximum_endpoint_orientation_error_rad=float(
            diagnostic["maximum_endpoint_orientation_error_rad"]
        ),
        maximum_ik_iterations=int(diagnostic["maximum_ik_iterations"]),
    )
    dynamics = validate_plan_dynamics(
        plan,
        maximum_step_rad=np.asarray(diagnostic["maximum_joint_step_rad"], dtype=float),
        maximum_velocity_rad_s=np.asarray(
            diagnostic["maximum_joint_velocity_rad_s"], dtype=float
        ),
        maximum_acceleration_rad_s2=np.asarray(
            diagnostic["maximum_joint_acceleration_rad_s2"], dtype=float
        ),
        maximum_jerk_rad_s3=np.asarray(
            diagnostic["maximum_joint_jerk_rad_s3"], dtype=float
        ),
    )
    return plan, dynamics, combined_limits


def create_run_dir(config: dict, label: str) -> Path:
    output_root = resolve_project_path(config["telemetry"]["output_dir"])
    now = datetime.now()
    run_dir = (
        output_root
        / now.strftime("%Y-%m-%d")
        / f"{now.strftime('%Y%m%d-%H%M%S')}_{label}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_results(
    run_dir: Path,
    rows: list[dict[str, object]],
    config: dict,
    dynamics: PlanDynamics,
) -> None:
    metadata = {
        "config": config,
        "plan_dynamics": {
            "max_step_rad": dynamics.max_step_rad.tolist(),
            "max_velocity_rad_s": dynamics.max_velocity_rad_s.tolist(),
            "max_acceleration_rad_s2": dynamics.max_acceleration_rad_s2.tolist(),
            "max_jerk_rad_s3": dynamics.max_jerk_rad_s3.tolist(),
        },
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    if rows:
        with (run_dir / "diagnostic.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("diagnostic summary: no completed command rows")
        return

    def timing_summary(label: str, values: np.ndarray) -> None:
        if values.size:
            print(
                f"{label}: p50={np.percentile(values, 50):.3f} ms, "
                f"p95={np.percentile(values, 95):.3f} ms, "
                f"max={np.max(values):.3f} ms"
            )

    send_times = np.asarray([float(row["command_send_monotonic_ns"]) for row in rows])
    realtime_paced = bool(rows[0]["realtime_paced"])
    if realtime_paced and send_times.size >= 2:
        intervals_ms = np.diff(send_times) / 1e6
        timing_summary("command intervals", intervals_ms)
    timing_summary(
        "driver send calls",
        np.asarray([float(row["send_call_duration_ms"]) for row in rows]),
    )
    if realtime_paced:
        timing_summary(
            "schedule lateness",
            np.asarray([float(row["schedule_lateness_ms"]) for row in rows]),
        )
    timing_summary(
        "feedback calls",
        np.asarray(
            [
                float(row["feedback_call_duration_ms"])
                for row in rows
                if row["feedback_call_duration_ms"] != ""
            ]
        ),
    )
    errors = np.asarray(
        [
            float(row["feedback_error_max_rad"])
            for row in rows
            if row["feedback_error_max_rad"] != ""
        ]
    )
    if errors.size:
        print(
            "sampled tracking error: "
            f"p50={np.percentile(errors, 50):.4f} rad, "
            f"p95={np.percentile(errors, 95):.4f} rad, "
            f"max={np.max(errors):.4f} rad"
        )


def execute_plan(
    backend,
    gate: CommandGate,
    plan: DiagnosticPlan,
    config: dict,
    gripper_position_m: float,
    rows: list[dict[str, object]],
    *,
    pace_realtime: bool,
) -> list[dict[str, object]]:
    diagnostic = config["diagnostic"]
    period = 1.0 / plan.rate_hz
    feedback_stride = max(
        1,
        round(plan.rate_hz / float(diagnostic["feedback_rate_hz"])),
    )
    feedback_q = plan.home_q.copy()
    started = time.perf_counter()
    next_send_not_before_s = started
    current_motion = ""

    for tick, point in enumerate(plan.points):
        if point.motion_key != current_motion:
            current_motion = point.motion_key
            motion = next(item for item in DIAGNOSTIC_MOTIONS if item.key == current_motion)
            print(
                f"test {len({row['motion_key'] for row in rows}) + 1}/6: "
                f"{motion.label} ({motion.positive_label}, then {motion.negative_label})"
            )

        if pace_realtime:
            sleep_for = next_send_not_before_s - time.perf_counter()
            if sleep_for > 0.0:
                time.sleep(sleep_for)
        loop_start_ns = time.perf_counter_ns()
        late_by = max(0.0, loop_start_ns / 1e9 - next_send_not_before_s)
        if pace_realtime:
            if late_by > float(diagnostic["maximum_loop_overrun_s"]):
                raise HardwareSafetyError(
                    f"diagnostic command loop overran by {late_by * 1000.0:.1f} ms"
                )

        gate.validate(point.q_command, gripper_position_m)
        send_start_ns = time.perf_counter_ns()
        backend.send_positions(
            point.q_command,
            gripper_position_m,
            include_gripper=False,
        )
        send_ns = time.perf_counter_ns()
        send_duration_ms = (send_ns - send_start_ns) / 1e6
        if pace_realtime:
            # Pace from the last successfully delivered command, not from an
            # absolute timeline. A blocking driver/feedback call can therefore
            # create one long interval, but never a catch-up burst afterward.
            next_send_not_before_s = send_ns / 1e9 + period
        feedback_fresh = False
        feedback_error: float | str = ""
        feedback_duration_ms: float | str = ""
        if tick % feedback_stride == 0:
            feedback_start_ns = time.perf_counter_ns()
            state = backend.read_state()
            feedback_end_ns = time.perf_counter_ns()
            feedback_duration_ms = (feedback_end_ns - feedback_start_ns) / 1e6
            feedback_q = state.q_arm.copy()
            feedback_fresh = True
            feedback_error = float(np.max(np.abs(feedback_q - point.q_command)))
            if feedback_error > float(diagnostic["maximum_tracking_error_rad"]):
                raise HardwareSafetyError(
                    "measured tracking error exceeded the diagnostic limit: "
                    f"{feedback_error:.6f} rad"
                )

        row: dict[str, object] = {
            "tick": tick,
            "motion_key": point.motion_key,
            "motion_label": point.motion_label,
            "phase": point.phase,
            "planned_elapsed_s": tick * period,
            "actual_elapsed_s": send_ns / 1e9 - started,
            "realtime_paced": pace_realtime,
            "schedule_lateness_ms": late_by * 1000.0,
            "command_send_monotonic_ns": send_ns,
            "send_call_duration_ms": send_duration_ms,
            "feedback_fresh": feedback_fresh,
            "feedback_call_duration_ms": feedback_duration_ms,
            "feedback_error_max_rad": feedback_error,
        }
        row.update({f"q_cmd_{joint}": point.q_command[joint] for joint in range(6)})
        row.update({f"q_feedback_{joint}": feedback_q[joint] for joint in range(6)})
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic, Quest-free six-axis WidowXAI motion diagnostic"
    )
    parser.add_argument("--config", default="configs/six_axis_arm_diagnostic.yaml")
    parser.add_argument("--live", action="store_true", help="use the physical WidowXAI")
    parser.add_argument("--robot-ip", help="override the explicitly configured controller IP")
    parser.add_argument(
        "--end-effector-profile",
        choices=tuple(END_EFFECTOR_PROFILE_TO_VARIANT),
        help="override the explicitly configured follower profile",
    )
    parser.add_argument("--confirm-live", default="", help="must equal LIVE-WIDOWXAI-<robot-ip>")
    parser.add_argument("--label", default="six-axis-arm-diagnostic")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        validate_diagnostic_config(config)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid diagnostic configuration: {exc}") from None

    hardware = config["hardware"]
    diagnostic = config["diagnostic"]
    model = WidowXAIModel(config["model"]["xml_path"])
    solver = DecoupledIK.from_config(model, config)
    home_q = np.asarray(config["model"]["simulation_start_q_rad"], dtype=float).reshape(6)
    robot_ip = str(args.robot_ip or hardware["robot_ip"])
    end_effector_profile = str(
        args.end_effector_profile or hardware["end_effector_profile"]
    )

    if args.live:
        if platform.system() not in ("Linux", "Darwin"):
            raise SystemExit("physical WidowXAI diagnostic requires Ubuntu or macOS")
        expected = f"LIVE-WIDOWXAI-{robot_ip}"
        if args.confirm_live != expected:
            raise SystemExit(f"live output remains disabled; pass --confirm-live {expected}")
        backend = TrossenArmBackend(
            robot_ip,
            command_goal_time_s=float(diagnostic["command_goal_time_s"]),
            end_effector_variant=END_EFFECTOR_PROFILE_TO_VARIANT[end_effector_profile],
            required_driver_version=str(hardware["driver_version_tested"]),
        )
        print(
            "LIVE diagnostic: clear the workspace and remain ready to cut controller power. "
            "The gripper will not be commanded."
        )
    else:
        rest_q = np.asarray(hardware["rest_q_rad"], dtype=float).reshape(6)
        backend = DryRunBackend(
            rest_q,
            model.joint_limits,
            gripper_position_m=float(hardware["gripper_open_m"]),
        )
        print("DRY RUN: no physical arm output")

    connected = False
    motion_started = False
    run_dir: Path | None = None
    plan: DiagnosticPlan | None = None
    dynamics: PlanDynamics | None = None
    rows: list[dict[str, object]] = []
    failure: Exception | None = None
    try:
        state = backend.connect()
        connected = True
        print(
            f"preflight: driver={state.driver_version} firmware={state.firmware_version} "
            f"q={np.round(state.q_arm, 3).tolist()}"
        )
        plan, dynamics, combined_limits = build_plan_for_state(
            config,
            model,
            solver,
            state,
        )
        print(
            f"preflight plan: {plan.duration_s:.1f} s at {plan.rate_hz:g} Hz, "
            f"{float(diagnostic['command_goal_time_s']) * 1000.0:g} ms driver horizon, "
            f"{float(diagnostic['translation_amplitude_m']) * 1000.0:g} mm translations and "
            f"{np.degrees(float(diagnostic['rotation_amplitude_rad'])):g} deg rotations, "
            f"maximum planned joint speed={np.max(dynamics.max_velocity_rad_s):.3f} rad/s"
        )

        startup_max_delta = np.asarray(
            hardware["startup_max_joint_delta_rad"], dtype=float
        ).reshape(6)
        ramp_gate = make_startup_command_gate(state, startup_max_delta, hardware)
        backend.enable_position_control(include_gripper=False)
        motion_started = True
        backend.send_positions(state.q_arm, state.gripper_position_m, include_gripper=False)
        print("startup: moving to home [0, 60, 75, -60, 0, 0] deg")
        ramp_to_home(backend, ramp_gate, state, home_q, config)
        settled = backend.read_state()
        if np.max(np.abs(settled.q_arm - home_q)) > float(hardware["max_feedback_error_rad"]):
            raise HardwareSafetyError("arm did not reach home within the feedback limit")

        command_gate = CommandGate(
            home_q,
            settled.gripper_position_m,
            combined_limits,
            np.asarray(diagnostic["maximum_joint_step_rad"], dtype=float),
            joint_limit_margin_rad=float(hardware["joint_limit_margin_rad"]),
            gripper_limits_m=teleop_gripper_limits(
                settled,
                hardware,
                control_gripper=False,
            ),
            max_gripper_delta_m=float(hardware["max_gripper_delta_m"]),
        )
        run_dir = create_run_dir(config, args.label)
        print("home reached: beginning deterministic six-axis sequence")
        execute_plan(
            backend,
            command_gate,
            plan,
            config,
            settled.gripper_position_m,
            rows,
            pace_realtime=args.live,
        )
        if args.live:
            time.sleep(0.2)
        final_state = backend.read_state()
        if np.max(np.abs(final_state.q_arm - home_q)) > float(
            diagnostic["maximum_tracking_error_rad"]
        ):
            raise HardwareSafetyError("arm did not finish the diagnostic at home")
        print("six-axis sequence complete at home")
    except KeyboardInterrupt:
        print("operator stop: returning to rest before driver cleanup")
    except (HardwareSafetyError, HardwareUnavailableError, RuntimeError, ValueError) as exc:
        failure = exc
    finally:
        if run_dir is not None and plan is not None and dynamics is not None:
            try:
                save_results(run_dir, rows, config, dynamics)
                print_summary(rows)
                print(f"diagnostic telemetry: {run_dir}")
            except Exception as exc:
                print(f"telemetry warning: {exc}", file=sys.stderr)

        shutdown_warnings: list[str] = []
        if connected:
            returned_to_rest = False
            if motion_started and hardware.get("return_to_rest_on_exit", True):
                try:
                    return_to_rest(
                        backend,
                        model,
                        config,
                        control_gripper=False,
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
        for warning in shutdown_warnings:
            print(
                f"EMERGENCY SHUTDOWN WARNING: {warning}; cut controller power now",
                file=sys.stderr,
                flush=True,
            )

    if failure is not None:
        raise SystemExit(f"SAFETY STOP: {failure}") from None


if __name__ == "__main__":
    main()
