from __future__ import annotations

import argparse
import select
import sys
import time
from dataclasses import replace

import mujoco
import numpy as np

from widowxai_quest_teleop.cad_input import (
    CadDeadmanController,
    CadFreshnessWatchdog,
    CadJointFilter,
    CadRestMapper,
    CadSafetyError,
    CadUdpReceiver,
    cad_commission_joint_index,
    rest_command_limits,
)
from widowxai_quest_teleop.config import load_config
from widowxai_quest_teleop.hardware import CommandGate, HardwareSafetyError
from widowxai_quest_teleop.model import WidowXAIModel
from widowxai_quest_teleop.motion_limiter import limiter_from_config
from widowxai_quest_teleop.telemetry import CAD_TELEMETRY_COLUMNS, TelemetryLogger
from widowxai_quest_teleop.viewer import close_passive_viewer


class SimulationDeadmanConsole:
    """Simulation-only latch controlled from the terminal or MuJoCo window."""

    def __init__(self) -> None:
        self._pressed = False
        self._quit = False
        self._interactive = False
        self.press_generation = 0
        self.release_generation = 0

    @property
    def pressed(self) -> bool:
        return self._pressed

    @property
    def quit_requested(self) -> bool:
        return self._quit

    def set_pressed(self, value: bool) -> None:
        pressed = bool(value)
        if pressed != self._pressed:
            if pressed:
                self.press_generation += 1
            else:
                self.release_generation += 1
        self._pressed = pressed

    def start(self) -> None:
        if not sys.stdin.isatty():
            print("stdin is not interactive; simulation deadman remains released")
            return
        self._interactive = True

    def handle_viewer_key(self, keycode: int) -> None:
        """Handle GLFW letter keycodes delivered by MuJoCo's passive viewer."""

        if keycode == ord("E"):
            self.set_pressed(True)
            print("SIM deadman ENGAGED from MuJoCo window; press R to release")
        elif keycode == ord("R"):
            self.set_pressed(False)
            print("SIM deadman released from MuJoCo window")
        elif keycode == ord("Q"):
            self.set_pressed(False)
            self._quit = True

    def poll(self) -> None:
        if not self._interactive:
            return
        readable, _writable, _exceptional = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return
        command = sys.stdin.readline().strip().lower()
        if command in ("e", "engage"):
            self.set_pressed(True)
            print("SIM deadman ENGAGED; type 'r' + Enter to release")
        elif command in ("r", "release"):
            self.set_pressed(False)
            print("SIM deadman released")
        elif command in ("q", "quit", ""):
            self.set_pressed(False)
            self._quit = True
        else:
            print("simulation commands: e=engage, r=release, q=quit")


def make_command_gate(
    rest_q: np.ndarray,
    command_limits: np.ndarray,
    model: WidowXAIModel,
    config: dict,
    gripper_q: float,
) -> CommandGate:
    combined = np.vstack([command_limits, np.array([0.0, 0.044])])
    return CommandGate(
        rest_q,
        gripper_q,
        combined,
        np.asarray(config["commissioning"]["max_command_step_rad"], dtype=float),
        # J1/J2 intentionally start at their official lower limits. The much
        # tighter rest-relative commissioning envelope supplies the guard.
        joint_limit_margin_rad=0.0,
        gripper_limits_m=(0.0, 0.044),
        max_gripper_delta_m=0.044,
    )


def return_simulation_to_rest(
    *,
    model: WidowXAIModel,
    sim_data: mujoco.MjData,
    viewer,
    limiter,
    gate: CommandGate,
    q_command: np.ndarray,
    rest_q: np.ndarray,
    gripper_q: float,
    loop_rate_hz: float,
    timeout_s: float,
) -> np.ndarray:
    if np.max(np.abs(q_command - rest_q)) <= 1e-8:
        model.set_viewer_qpos(sim_data, rest_q, gripper_q)
        if viewer is not None:
            viewer.sync()
        return rest_q.copy()

    print("returning MuJoCo command to all-zero rest")
    limiter.reset(q_command)
    period = 1.0 / loop_rate_hz
    deadline = time.perf_counter() + timeout_s
    next_tick = time.perf_counter()
    while time.perf_counter() < deadline:
        result = limiter.step(rest_q, period)
        candidate = result.command
        if model.in_self_collision(candidate, gripper_q):
            raise CadSafetyError("self-collision predicted while returning simulation to rest")
        gate.validate(candidate, gripper_q)
        q_command = candidate
        model.set_viewer_qpos(sim_data, q_command, gripper_q)
        if viewer is not None:
            viewer.sync()
        if np.max(np.abs(q_command - rest_q)) <= 1e-6:
            model.set_viewer_qpos(sim_data, rest_q, gripper_q)
            return rest_q.copy()
        next_tick += period
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0.0:
            time.sleep(sleep_for)
        else:
            next_tick = time.perf_counter()
    raise CadSafetyError("simulation did not return to rest before the configured timeout")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview validated M3T leader joints from all-zero rest in MuJoCo"
    )
    parser.add_argument("--config", default="configs/cad_rest_commissioning.yaml")
    parser.add_argument("--udp-host")
    parser.add_argument("--udp-port", type=int)
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs until q/Ctrl+C")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--label", default="cad-rest-mujoco")
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument(
        "--commission-joint",
        type=int,
        choices=range(5),
        help="preview only one follower joint (0-4), matching physical sign commissioning",
    )
    parser.add_argument(
        "--simulation-auto-deadman",
        action="store_true",
        help=(
            "simulation smoke tests only: engage once after the first fresh packet; "
            "it remains fail-closed after any stale/discontinuity"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config.get("hardware", {}).get("enabled", False):
        raise SystemExit("CAD MuJoCo launcher refuses configurations with hardware.enabled=true")
    if config.get("hardware", {}).get("live_output_implemented", False):
        raise SystemExit("CAD MuJoCo launcher cannot enable physical output")
    if not np.isfinite(args.duration) or args.duration < 0.0:
        raise SystemExit("--duration must be finite and nonnegative")
    try:
        commission_joint = cad_commission_joint_index(args.commission_joint)
    except CadSafetyError as exc:
        raise SystemExit(f"CAD COMMISSIONING SELECTION REJECTED: {exc}") from None

    model = WidowXAIModel(config["model"]["xml_path"])
    rest_q = np.asarray(config["model"]["rest_q_rad"], dtype=float).reshape(6)
    if not np.all(np.isfinite(rest_q)):
        raise SystemExit("rest position must contain six finite values")
    if model.in_self_collision(rest_q, float(config["model"]["gripper_preview_m"])):
        raise SystemExit("official MuJoCo model reports a self-collision at rest")

    commissioning = config["commissioning"]
    command_limits = rest_command_limits(
        model.joint_limits,
        rest_q,
        np.asarray(commissioning["minimum_delta_rad"], dtype=float),
        np.asarray(commissioning["maximum_delta_rad"], dtype=float),
    )
    mapper = CadRestMapper(
        rest_q=rest_q,
        signs=np.asarray(config["cad"]["signs"], dtype=float),
        scales=np.asarray(config["cad"]["scales"], dtype=float),
        source_deadband_rad=np.asarray(config["cad"]["source_deadband_rad"], dtype=float),
        commission_joint=commission_joint,
        command_limits=command_limits,
    )
    controller = CadDeadmanController(mapper)
    watchdog = CadFreshnessWatchdog(
        config["cad"]["stale_timeout_s"],
        config["cad"]["fresh_samples_to_recover"],
        np.asarray(config["cad"]["max_source_step_rad"], dtype=float),
    )
    source_filter_config = config["cad"]["source_filter"]
    if source_filter_config.get("enabled") is not True:
        raise SystemExit("CAD MuJoCo commissioning requires the source joint filter")
    source_filter = CadJointFilter(
        minimum_cutoff_hz=source_filter_config["minimum_cutoff_hz"],
        speed_coefficient=source_filter_config["speed_coefficient"],
        derivative_cutoff_hz=source_filter_config["derivative_cutoff_hz"],
        maximum_cutoff_hz=source_filter_config["maximum_cutoff_hz"],
    )

    udp_host = args.udp_host or config["cad"]["udp_host"]
    udp_port = config["cad"]["udp_port"] if args.udp_port is None else args.udp_port
    receiver = CadUdpReceiver(
        udp_host,
        udp_port,
        expected_names=config["cad"]["expected_names"],
        max_packet_age_s=config["cad"]["max_packet_age_s"],
        max_future_skew_s=config["cad"]["max_future_skew_s"],
        require_root_locked=config["cad"].get("require_root_locked", True),
    )

    gripper_q = float(config["model"]["gripper_preview_m"])
    gate = make_command_gate(rest_q, command_limits, model, config, gripper_q)
    limiter = limiter_from_config(config["control"]["joint_command_limits"], rest_q)
    if limiter is None:
        raise SystemExit("CAD commissioning requires the joint command limiter")

    sim_data = mujoco.MjData(model.model)
    model.set_viewer_qpos(sim_data, rest_q, gripper_q)
    console = SimulationDeadmanConsole()
    viewer = None
    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(
            model.model,
            sim_data,
            key_callback=console.handle_viewer_key,
        )
        viewer.cam.lookat[:] = [0.25, 0.0, 0.18]
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

    if not args.simulation_auto_deadman:
        console.start()
    print("PHYSICAL OUTPUT DISABLED: this process never imports or opens the Trossen driver")
    print(f"starting and ending at rest q={rest_q.tolist()}")
    print(f"listening for M3T joints on udp://{udp_host}:{udp_port}")
    print(
        "mapping selection: "
        + (
            "all five mapped joints"
            if commission_joint is None
            else (
                f"ONLY follower joint {commission_joint} from "
                f"{config['cad']['expected_names'][commission_joint]}"
            )
        )
    )
    if args.simulation_auto_deadman:
        print("simulation auto-deadman will engage once after three fresh packets")
    else:
        print(
            "simulation commands: click MuJoCo and press E/R/Q, or use "
            "e/r/q + Enter in this terminal"
        )

    telemetry = None
    if not args.no_telemetry:
        telemetry = TelemetryLogger(
            args.label,
            config,
            config["telemetry"]["output_dir"],
            columns=CAD_TELEMETRY_COLUMNS,
            ik_status_columns=(),
            strict_columns=True,
        )

    q_command = rest_q.copy()
    q_des = rest_q.copy()
    last_sample = None
    last_filtered_sample = None
    filter_generation = watchdog.discontinuity_generation
    loop_hz = float(config["control"]["loop_rate_hz"])
    period = 1.0 / loop_hz
    started = time.perf_counter()
    previous_tick = started
    next_tick = started
    last_log = started
    auto_deadman_engaged = False
    last_event = "startup_at_rest"

    try:
        receiver.start()
        while args.duration <= 0.0 or time.perf_counter() - started < args.duration:
            console.poll()
            if console.quit_requested:
                break
            if viewer is not None and not viewer.is_running():
                break
            sample, _generation = receiver.mailbox.take_latest()
            now_ns = time.perf_counter_ns()
            if sample is not None:
                last_sample = sample
                freshness = watchdog.observe(sample, now_ns)
            else:
                freshness = watchdog.poll(now_ns)
            if freshness.discontinuity_generation != filter_generation:
                source_filter.reset()
                last_filtered_sample = None
                filter_generation = freshness.discontinuity_generation
            if sample is not None:
                filtered_q = source_filter.update(
                    sample.q, sample.arrival_monotonic_ns
                )
                last_filtered_sample = replace(sample, q=filtered_q)

            if args.simulation_auto_deadman:
                if freshness.fresh and not auto_deadman_engaged:
                    console.set_pressed(True)
                    auto_deadman_engaged = True
                deadman_pressed = console.pressed
            else:
                deadman_pressed = console.pressed

            try:
                target = controller.update(
                    deadman_pressed=deadman_pressed,
                    stream_fresh=freshness.fresh,
                    sample=last_filtered_sample,
                    robot_q=q_command,
                )
                last_event = ""
            except CadSafetyError as exc:
                target = None
                last_event = f"mapping_fault:{exc}"
                print(f"CAD SAFETY HOLD: {exc}; release and re-engage the simulation deadman")

            now = time.perf_counter()
            dt = max(1e-6, now - previous_tick)
            previous_tick = now
            limiter_flags: list[str] = []
            if target is None:
                q_des = q_command.copy()
                limiter.reset(q_command)
                candidate = q_command.copy()
            else:
                q_des = target
                result = limiter.step(q_des, dt)
                candidate = result.command
                limiter_flags = result.flags("joint")

            try:
                if model.in_self_collision(candidate, gripper_q):
                    raise CadSafetyError("official model predicts a self-collision")
                gate.validate(candidate, gripper_q)
                q_command = candidate
            except (CadSafetyError, HardwareSafetyError) as exc:
                controller.fault(deadman_pressed)
                limiter.reset(q_command)
                q_des = q_command.copy()
                last_event = f"command_fault:{exc}"
                print(f"CAD SAFETY HOLD: {exc}; release and re-engage the simulation deadman")

            model.set_viewer_qpos(sim_data, q_command, gripper_q)
            if viewer is not None:
                viewer.sync()

            if telemetry is not None:
                source_age_ms = (
                    ""
                    if last_sample is None
                    else (time.time_ns() - last_sample.source_time_ns) / 1e6
                )
                arrival_age_ms = (
                    ""
                    if last_sample is None
                    else (now_ns - last_sample.arrival_monotonic_ns) / 1e6
                )
                telemetry.log(
                    pc_epoch_ns=time.time_ns(),
                    pc_monotonic_ns=time.perf_counter_ns(),
                    cad_sequence="" if last_sample is None else last_sample.sequence,
                    cad_source_time_ns="" if last_sample is None else last_sample.source_time_ns,
                    cad_source_age_ms=source_age_ms,
                    cad_arrival_age_ms=arrival_age_ms,
                    cad_root_locked=False if last_sample is None else last_sample.root_locked,
                    receiver_packets=receiver.received_packets,
                    receiver_valid_packets=receiver.valid_packets,
                    receiver_invalid_packets=receiver.invalid_packets,
                    receiver_last_error=receiver.last_error or "",
                    cad_fresh=freshness.fresh,
                    recovery_streak=freshness.recovery_streak,
                    discontinuity_generation=freshness.discontinuity_generation,
                    last_discontinuity=freshness.last_discontinuity,
                    deadman_pressed=deadman_pressed,
                    deadman_engaged=controller.engaged,
                    deadman_needs_release=controller.needs_release,
                    deadman_source="simulation_terminal_latch",
                    deadman_device="",
                    deadman_key_code="",
                    deadman_state_age_ms=0.0,
                    deadman_press_generation=console.press_generation,
                    deadman_release_generation=console.release_generation,
                    reanchor_generation=mapper.reanchor_generation,
                    commission_joint=""
                    if commission_joint is None
                    else commission_joint,
                    hardware_live=False,
                    q_source="" if last_sample is None else last_sample.q,
                    q_source_filtered=""
                    if last_filtered_sample is None
                    else last_filtered_sample.q,
                    source_filter_alpha=source_filter.last_alpha,
                    source_filter_cutoff_hz=source_filter.last_cutoff_hz,
                    q_des=q_des,
                    q_cmd=q_command,
                    q_feedback=sim_data.qpos[model.qpos_indices].copy(),
                    limiter_flags="|".join(limiter_flags),
                    event=last_event,
                )

            if now - last_log >= 1.0:
                print(
                    "CAD_SIM "
                    f"commission_joint={commission_joint if commission_joint is not None else 'all'} "
                    f"recv/valid/invalid={receiver.received_packets}/"
                    f"{receiver.valid_packets}/{receiver.invalid_packets} "
                    f"fresh={int(freshness.fresh)} age_ms={freshness.age_s * 1000:.1f} "
                    f"deadman={int(deadman_pressed)} engaged={int(controller.engaged)} "
                    f"needs_release={int(controller.needs_release)} "
                    f"discontinuity={freshness.last_discontinuity or 'none'} "
                    f"q_source={np.array2string(last_sample.q if last_sample is not None else np.zeros(5), precision=4, suppress_small=True)} "
                    f"q_cmd={np.array2string(q_command, precision=4, suppress_small=True)}"
                )
                last_log = now

            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0.0:
                time.sleep(sleep_for)
            else:
                # Never generate catch-up command bursts after an overrun.
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("operator stopped CAD MuJoCo commissioning")
    finally:
        console.set_pressed(False)
        controller.update(
            deadman_pressed=False,
            stream_fresh=False,
            sample=last_filtered_sample,
            robot_q=q_command,
        )
        receiver.stop()
        try:
            q_command = return_simulation_to_rest(
                model=model,
                sim_data=sim_data,
                viewer=viewer,
                limiter=limiter,
                gate=gate,
                q_command=q_command,
                rest_q=rest_q,
                gripper_q=gripper_q,
                loop_rate_hz=loop_hz,
                timeout_s=float(commissioning["return_to_rest_timeout_s"]),
            )
            print(f"rest reached: q={np.round(q_command, 8).tolist()}")
        finally:
            if telemetry is not None:
                telemetry.close()
                print(f"telemetry: {telemetry.run_dir}")
            if viewer is not None:
                close_passive_viewer(viewer)


if __name__ == "__main__":
    main()
