from __future__ import annotations

import argparse
import json

from widowxai_quest_teleop.config import load_config
from widowxai_quest_teleop.workstation import (
    WorkstationSafetyError,
    run_workstation_preflight,
)


def robot_ips_from_config(config: dict) -> list[str]:
    arms = config.get("arms")
    if isinstance(arms, dict):
        addresses = [
            str(arms[side].get("robot_ip", "")).strip()
            for side in ("left", "right")
            if isinstance(arms.get(side), dict)
        ]
    else:
        addresses = [str(config.get("hardware", {}).get("robot_ip", "")).strip()]
    if not addresses or any(not address for address in addresses):
        raise WorkstationSafetyError("selected config does not define every robot IP")
    return addresses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only AC, power-profile, and robot-route preflight"
    )
    parser.add_argument("--config", default="configs/quest_50pct_hardware.yaml")
    parser.add_argument(
        "--robot-ip",
        action="append",
        help="override config addresses; repeat for a two-arm check",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    addresses = args.robot_ip or robot_ips_from_config(config)
    try:
        report = run_workstation_preflight(config["hardware"], addresses)
    except WorkstationSafetyError as exc:
        raise SystemExit(f"WORKSTATION PREFLIGHT FAILED: {exc}") from None
    print(
        json.dumps(
            {
                "passed": True,
                "robot_ips": addresses,
                "power_supply": report.power_supply,
                "power_profile": report.power_profile,
                "route_source_ip": report.route_source_ip,
                "route_devices": report.route_devices,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
