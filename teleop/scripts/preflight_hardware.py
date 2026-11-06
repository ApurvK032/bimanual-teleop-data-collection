from __future__ import annotations

import argparse

import numpy as np

from widowxai_quest_teleop.config import load_config
from widowxai_quest_teleop.hardware import (
    END_EFFECTOR_PROFILE_TO_VARIANT,
    HardwareSafetyError,
    HardwareUnavailableError,
    TrossenArmBackend,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover and read a WidowXAI without enabling position mode or sending commands"
    )
    parser.add_argument("--config", default="configs/safe_demo_30pct.yaml")
    parser.add_argument("--robot-ip", help="exact controller IP; defaults to the selected config")
    parser.add_argument(
        "--end-effector-profile",
        choices=tuple(END_EFFECTOR_PROFILE_TO_VARIANT),
        help="defaults to the selected config; legacy_1_8 matches the proven local setup",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    hardware = config["hardware"]
    robot_ip = args.robot_ip or hardware.get("robot_ip")
    profile = args.end_effector_profile or hardware.get("end_effector_profile")
    if not robot_ip:
        raise SystemExit("PREFLIGHT FAILED: robot IP is required")
    try:
        variant = None if profile is None else END_EFFECTOR_PROFILE_TO_VARIANT[str(profile)]
    except KeyError:
        choices = ", ".join(END_EFFECTOR_PROFILE_TO_VARIANT)
        raise SystemExit(f"PREFLIGHT FAILED: end-effector profile must be one of: {choices}") from None
    backend = TrossenArmBackend(
        robot_ip,
        end_effector_variant=variant,
        required_driver_version=str(hardware["driver_version_tested"]),
    )
    try:
        if profile is None:
            discovery = backend.discover()
            print("DISCOVERY-ONLY PREFLIGHT PASSED")
            print(f"robot_ip: {discovery.robot_ip}")
            print(f"driver_version: {discovery.driver_version}")
            print(f"firmware_version: {discovery.firmware_version}")
            print("the driver was not configured; no position command was sent")
            print("select the physical follower profile, then rerun the no-motion state preflight")
            return

        state = backend.connect()
        print("NO-MOTION STATE PREFLIGHT PASSED")
        print(f"robot_ip: {robot_ip}")
        print(f"driver_version: {state.driver_version}")
        print(f"firmware_version: {state.firmware_version}")
        print(f"end_effector_profile: {profile}")
        print(f"arm_q_rad: {np.round(state.q_arm, 6).tolist()}")
        print(f"gripper_m: {state.gripper_position_m:.6f}")
        print(f"joint_limits: {np.round(state.joint_limits, 6).tolist()}")
        print(f"position_tolerances: {np.round(state.position_tolerances, 6).tolist()}")
        print("position mode was not enabled; no position command was sent")
    except (HardwareSafetyError, HardwareUnavailableError) as exc:
        raise SystemExit(f"PREFLIGHT FAILED: {exc}") from None
    finally:
        backend.close()


if __name__ == "__main__":
    main()
