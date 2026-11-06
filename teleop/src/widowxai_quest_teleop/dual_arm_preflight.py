from __future__ import annotations

import numpy as np

from .config import DUAL_ARM_SIDES
from .dual_arm_coordinator import build_dual_arm_system
from .hardware import HardwareSafetyError


def offline_checks(config: dict, arms_config: dict) -> None:
    """Screen standard dual-arm paths without opening either backend."""

    print("=== offline dual-arm checks ===")
    for side in DUAL_ARM_SIDES:
        arm = arms_config[side]
        print(
            f"{side:>5} arm: controller={arm.controller_hand} mapping={arm.mapping_mode} "
            f"ip={arm.robot_ip} calibration={arm.calibration_status} "
            f"base={arm.placement.measurement_status}"
        )
    print(f"base separation: {config['_dual_arm']['base_separation_m']:.3f} m")

    home_q = np.asarray(config["model"]["simulation_start_q_rad"], dtype=float).reshape(6)
    hardware = config["hardware"]
    gripper_open = float(hardware["gripper_open_m"])
    rest_q = np.asarray(hardware["rest_q_rad"], dtype=float).reshape(6)
    rest_gripper = float(hardware["rest_gripper_m"])

    arms, collision_model, _ = build_dual_arm_system(
        config,
        arms_config,
        initial_q=home_q,
        initial_gripper_m=gripper_open,
        control_gripper=bool(hardware.get("control_gripper", True)),
    )
    home_q = arms["left"].model.clamp_joints(home_q)
    samples = int(hardware["startup_collision_samples"])

    def both(value):
        return {side: value for side in DUAL_ARM_SIDES}

    for name, start_q, end_q, start_g, end_g in (
        ("rest -> home (simultaneous)", rest_q, home_q, rest_gripper, rest_gripper),
        ("gripper open at home", home_q, home_q, rest_gripper, gripper_open),
        ("home -> rest (simultaneous)", home_q, rest_q, gripper_open, gripper_open),
        ("gripper close at rest", rest_q, rest_q, gripper_open, rest_gripper),
    ):
        found = collision_model.first_collision_on_path(
            both(start_q),
            both(end_q),
            start_gripper=both(start_g),
            end_gripper=both(end_g),
            samples=samples,
        )
        if found is not None:
            alpha, report = found
            raise HardwareSafetyError(
                f"combined scene rejects {name} near {alpha * 100:.1f}%: "
                f"{report.describe()}"
            )
        print(f"  combined path clear: {name}")

    for moving in DUAL_ARM_SIDES:
        holding = next(side for side in DUAL_ARM_SIDES if side != moving)
        found = collision_model.first_collision_on_path(
            {moving: rest_q, holding: home_q},
            {moving: home_q, holding: home_q},
            start_gripper=both(gripper_open),
            end_gripper=both(gripper_open),
            samples=samples,
        )
        if found is not None:
            alpha, report = found
            raise HardwareSafetyError(
                f"combined scene rejects {moving} rest->home while {holding} holds home "
                f"near {alpha * 100:.1f}%: {report.describe()}"
            )
        print(f"  combined path clear: {moving} moves while {holding} holds")

    report = collision_model.check(both(home_q), both(gripper_open))
    print(
        f"  home/home clearance: {report.separation_m:.3f} m "
        f"(configured margin {config['safety']['cross_arm_clearance_m']:.3f} m)"
    )
    print("offline dual-arm checks PASSED")
