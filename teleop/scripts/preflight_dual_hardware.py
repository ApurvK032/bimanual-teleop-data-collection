from __future__ import annotations

import argparse

import numpy as np

from widowxai_quest_teleop.config import (
    DUAL_ARM_SIDES,
    DualArmConfigError,
    dual_live_confirmation_token,
    load_config,
    parse_dual_arm_config,
    require_live_dual_arm_config,
)
from widowxai_quest_teleop.dual_arm_preflight import offline_checks
from widowxai_quest_teleop.hardware import (
    END_EFFECTOR_PROFILE_TO_VARIANT,
    HardwareSafetyError,
    HardwareUnavailableError,
    TrossenArmBackend,
)
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a dual-arm profile and read both WidowXAI controllers "
            "without enabling position mode or sending any command"
        )
    )
    parser.add_argument("--config", default="configs/dual_widowxai.yaml")
    parser.add_argument(
        "--end-effector-profile",
        choices=tuple(END_EFFECTOR_PROFILE_TO_VARIANT),
        help="defaults to the selected config; legacy_1_8 matches the proven local setup",
    )
    parser.add_argument(
        "--contact-arms",
        action="store_true",
        help=(
            "additionally open a read-only driver session to each controller; "
            "omit to run the offline checks only"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        arms_config = parse_dual_arm_config(config)
    except DualArmConfigError as exc:
        raise SystemExit(f"PREFLIGHT FAILED: invalid dual-arm configuration: {exc}") from None

    try:
        offline_checks(config, arms_config)
    except (HardwareSafetyError, ValueError) as exc:
        raise SystemExit(f"PREFLIGHT FAILED: {exc}") from None

    print()
    print(f"dual live confirmation token: {dual_live_confirmation_token(arms_config)}")
    try:
        require_live_dual_arm_config(config, arms_config)
        print("live dual-arm gates: SATISFIED")
    except DualArmConfigError as exc:
        print(f"live dual-arm gates: BLOCKED — {exc}")

    if not args.contact_arms:
        print()
        print("no controller was contacted; pass --contact-arms for the read-only state preflight")
        return

    hardware = config["hardware"]
    profile = args.end_effector_profile or hardware.get("end_effector_profile")
    if not profile:
        raise SystemExit("PREFLIGHT FAILED: an end-effector profile is required")
    variant = END_EFFECTOR_PROFILE_TO_VARIANT[str(profile)]

    print()
    print("=== read-only controller state (no position mode, no commands) ===")
    backends = {}
    try:
        for side in DUAL_ARM_SIDES:
            arm = arms_config[side]
            backend = TrossenArmBackend(
                arm.robot_ip,
                end_effector_variant=variant,
                required_driver_version=str(hardware["driver_version_tested"]),
            )
            backends[side] = backend
            state = backend.connect()
            print(f"{side:>5} arm @ {arm.robot_ip}")
            print(f"        driver={state.driver_version} firmware={state.firmware_version}")
            print(f"        q_rad={np.round(state.q_arm, 6).tolist()}")
            print(f"        gripper_m={state.gripper_position_m:.6f}")
        print("NO-MOTION DUAL PREFLIGHT PASSED for both controllers")
        print("position mode was not enabled on either arm; no command was sent")
    except (HardwareSafetyError, HardwareUnavailableError) as exc:
        connected = sorted(backends)
        raise SystemExit(
            f"PREFLIGHT FAILED: {exc}; sessions opened for {connected} were closed"
        ) from None
    finally:
        for backend in backends.values():
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 - report, never mask cleanup
                print(f"WARNING: driver cleanup failed: {exc}")


if __name__ == "__main__":
    main()
