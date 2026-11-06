"""Offline check of a teleop arm profile with the teleop repo's own validators (no arm, no headset contacted).

    python -m bimanual_collect.profile_check <profile.yaml>      prints one JSON object

Two-arm profiles: parse_dual_arm_config, the combined-scene path checks (offline_checks), the live confirmation
token and the live gates (accepted calibrations, measured bases). One-arm profiles: the hardware launcher's own
config and timing validators, and the calibration's acceptance status. Runs in the main environment (.venv), which
has the teleop package and MuJoCo.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

from . import rig


def check(profile: Path) -> dict:
    teleop = rig.load().teleop
    os.chdir(teleop)  # the profiles' relative paths (model, calibrations) are relative to the teleop repo
    from widowxai_quest_teleop.config import (
        DualArmConfigError,
        calibration_acceptance_status,
        dual_live_confirmation_token,
        load_config,
        parse_dual_arm_config,
        require_live_dual_arm_config,
    )

    result: dict = {"profile": str(profile), "ok": False, "gates": "blocked", "errors": [], "notes": []}
    try:
        config = load_config(profile)
    except Exception as exc:  # noqa: BLE001 - report any load problem
        result["errors"].append(f"the profile does not load: {exc}")
        return result
    if "arms" in config:
        result["kind"] = "dual"
        try:
            arms = parse_dual_arm_config(config)
        except DualArmConfigError as exc:
            result["errors"].append(str(exc))
            return result
        result["arms"] = {side: {"ip": arm.robot_ip, "calibration": arm.calibration_status,
                                 "base": arm.placement.measurement_status} for side, arm in arms.items()}
        result["token"] = dual_live_confirmation_token(arms)
        from widowxai_quest_teleop.dual_arm_preflight import offline_checks
        from widowxai_quest_teleop.hardware import HardwareSafetyError

        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                offline_checks(config, arms)
        except (HardwareSafetyError, ValueError) as exc:
            result["errors"].append(f"combined-scene check failed: {exc}")
            return result
        result["checks"] = [line.strip() for line in printed.getvalue().splitlines() if line.strip()]
        try:
            require_live_dual_arm_config(config, arms)
            result["gates"] = "satisfied"
        except DualArmConfigError as exc:
            result["errors"].append(f"live gates blocked: {exc}")
            return result
    else:
        result["kind"] = "single"
        hardware, quest = config.get("hardware", {}), config.get("quest", {})
        ip = str(hardware.get("robot_ip", ""))
        result["arms"] = {quest.get("hand", "left"): {"ip": ip}}
        result["token"] = f"LIVE-WIDOWXAI-{ip}"
        spec = importlib.util.spec_from_file_location("_run_hardware", teleop / "scripts" / "run_hardware.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            module.validate_hardware_config(config)
            module.validate_live_hardware_timing(config)
        except Exception as exc:  # noqa: BLE001 - the validators raise their own safety errors
            result["errors"].append(f"hardware settings rejected: {exc}")
            return result
        calibration = quest.get("calibration")
        status = calibration_acceptance_status(calibration) if calibration else "none"
        result["arms"][quest.get("hand", "left")]["calibration"] = status
        if not status.startswith("accepted"):
            result["errors"].append(f"the calibration is {status!r}, not accepted")
            return result
        result["gates"] = "satisfied"
    result["ok"] = True
    return result


def main() -> None:
    profile = Path(sys.argv[1]).resolve()
    print(json.dumps(check(profile)))


if __name__ == "__main__":
    main()
