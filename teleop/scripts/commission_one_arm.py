from __future__ import annotations

"""Run one measured WidowXAI rest -> home -> rest commissioning cycle.

The selected arm is the only controller placed in position mode. The other arm
is connected read-only so its fresh measured pose participates in cross-arm
path screening. The gripper is never placed in position mode or commanded.
"""

import argparse
import sys
import time

import numpy as np

if __package__:
    from scripts.run_hardware import make_startup_command_gate, ramp_to_home, return_to_rest
else:
    from run_hardware import make_startup_command_gate, ramp_to_home, return_to_rest

from widowxai_quest_teleop.config import (
    DUAL_ARM_SIDES,
    DualArmConfigError,
    load_config,
    parse_dual_arm_config,
)
from widowxai_quest_teleop.dual_arm_model import DualArmCollisionModel
from widowxai_quest_teleop.hardware import (
    END_EFFECTOR_PROFILE_TO_VARIANT,
    HardwareSafetyError,
    HardwareUnavailableError,
    TrossenArmBackend,
)
from widowxai_quest_teleop.model import WidowXAIModel
from widowxai_quest_teleop.one_arm_safety import (
    MARGINAL_START_CONTACT_M,
    screen_selected_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Commission one WidowXAI arm through rest -> home -> rest"
    )
    parser.add_argument("--config", default="configs/dual_widowxai.yaml")
    parser.add_argument("--side", choices=DUAL_ARM_SIDES, required=True)
    parser.add_argument(
        "--confirm-live",
        default="",
        help="must equal COMMISSION-WIDOWXAI-<side>-<ip>-HOME-CYCLE",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        arms = parse_dual_arm_config(config)
    except DualArmConfigError as exc:
        raise SystemExit(f"invalid dual-arm configuration: {exc}") from None

    side = args.side
    other = next(name for name in DUAL_ARM_SIDES if name != side)
    arm = arms[side]
    expected = f"COMMISSION-WIDOWXAI-{side}-{arm.robot_ip}-HOME-CYCLE"
    if args.confirm_live != expected:
        raise SystemExit(f"motion disabled; pass --confirm-live {expected}")
    if not arm.placement.measured or not arms[other].placement.measured:
        raise SystemExit("motion disabled; both base transforms must be measured")

    hardware = config["hardware"]
    if not hardware.get("enabled") or not hardware.get("require_explicit_enable"):
        raise SystemExit("motion disabled; hardware gates are not explicitly enabled")
    variant = END_EFFECTOR_PROFILE_TO_VARIANT[str(hardware["end_effector_profile"])]
    home_q = np.asarray(config["model"]["simulation_start_q_rad"], dtype=float).reshape(6)
    rest_q = np.asarray(hardware["rest_q_rad"], dtype=float).reshape(6)
    samples = int(hardware["startup_collision_samples"])
    clearance_m = float(config["safety"]["cross_arm_clearance_m"])

    single_model = WidowXAIModel(config["model"]["xml_path"])
    dual_model = DualArmCollisionModel(
        {name: arms[name].placement for name in DUAL_ARM_SIDES},
        xml_path=config["model"]["xml_path"],
        clearance_m=clearance_m,
    )
    moving = TrossenArmBackend(
        arm.robot_ip,
        command_goal_time_s=float(hardware["command_goal_time_s"]),
        end_effector_variant=variant,
        required_driver_version=str(hardware["driver_version_tested"]),
    )
    holding = TrossenArmBackend(
        arms[other].robot_ip,
        end_effector_variant=variant,
        required_driver_version=str(hardware["driver_version_tested"]),
    )

    moving_connected = False
    holding_connected = False
    motion_started = False
    returned_to_rest = False
    try:
        moving_state = moving.connect()
        moving_connected = True
        holding_state = holding.connect()
        holding_connected = True
        print("=== fresh measured state ===")
        print(f"  moving {side} @ {arm.robot_ip}: {np.round(moving_state.q_arm, 6).tolist()}")
        print(
            f"  holding {other} @ {arms[other].robot_ip}: "
            f"{np.round(holding_state.q_arm, 6).tolist()} (read-only)"
        )

        outward_min = screen_selected_path(
            single_model,
            dual_model,
            moving_side=side,
            moving_start=moving_state.q_arm,
            moving_end=home_q,
            holding_q=holding_state.q_arm,
            moving_gripper_m=moving_state.gripper_position_m,
            holding_gripper_m=holding_state.gripper_position_m,
            samples=samples,
            clearance_m=clearance_m,
            allow_marginal_start_m=MARGINAL_START_CONTACT_M,
        )
        return_min = screen_selected_path(
            single_model,
            dual_model,
            moving_side=side,
            moving_start=home_q,
            moving_end=rest_q,
            holding_q=holding_state.q_arm,
            moving_gripper_m=moving_state.gripper_position_m,
            holding_gripper_m=holding_state.gripper_position_m,
            samples=samples,
            clearance_m=clearance_m,
        )
        print(
            f"path screen passed: minimum separation {min(outward_min, return_min):.3f} m "
            f"(required {clearance_m:.3f} m)"
        )

        startup_delta = np.asarray(hardware["startup_max_joint_delta_rad"], dtype=float)
        gate = make_startup_command_gate(moving_state, startup_delta, hardware)
        moving.enable_position_control(include_gripper=False)
        motion_started = True
        moving.send_positions(
            moving_state.q_arm,
            moving_state.gripper_position_m,
            include_gripper=False,
        )
        print(f"startup: moving ONLY {side} arm to home; gripper untouched")
        ramp_to_home(moving, gate, moving_state, home_q, config)
        home_state = moving.read_state()
        home_error = float(np.max(np.abs(home_state.q_arm - home_q)))
        if home_error > float(hardware["max_feedback_error_rad"]):
            raise HardwareSafetyError(
                f"{side} arm home error {home_error:.6f} rad exceeds the feedback limit"
            )
        print(f"home reached: maximum joint error {home_error:.6f} rad")
        time.sleep(0.5)

        # Re-read the stationary arm before screening the actual return path.
        holding_state = holding.read_state()
        return_min = screen_selected_path(
            single_model,
            dual_model,
            moving_side=side,
            moving_start=moving.read_state().q_arm,
            moving_end=rest_q,
            holding_q=holding_state.q_arm,
            moving_gripper_m=home_state.gripper_position_m,
            holding_gripper_m=holding_state.gripper_position_m,
            samples=samples,
            clearance_m=clearance_m,
        )
        print(f"return path re-screened: minimum separation {return_min:.3f} m")
        return_to_rest(moving, single_model, config, control_gripper=False)
        returned_to_rest = True
        rest_state = moving.read_state()
        rest_error = float(np.max(np.abs(rest_state.q_arm - rest_q)))
        print(f"HOME-CYCLE PASSED for {side}: final rest error {rest_error:.6f} rad")
        print(f"{other} arm remained read-only; neither gripper was commanded")
    except (HardwareSafetyError, HardwareUnavailableError, RuntimeError, ValueError) as exc:
        print(f"SAFETY STOP: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
    finally:
        if moving_connected and motion_started and not returned_to_rest:
            try:
                return_to_rest(moving, single_model, config, control_gripper=False)
                print("recovery: moving arm returned to rest")
            except Exception as exc:  # noqa: BLE001 - emergency path must report everything
                print(
                    f"EMERGENCY SHUTDOWN WARNING: return to rest failed: {exc}; "
                    "cut controller power now",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    moving.safe_hold()
                except Exception:
                    pass
        for backend in (moving, holding):
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 - report cleanup without masking result
                print(f"WARNING: driver cleanup failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
