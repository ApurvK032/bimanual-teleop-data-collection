from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


class WorkstationSafetyError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkstationPreflightReport:
    power_supply: str
    power_profile: str
    route_devices: tuple[str, ...]
    route_source_ip: str


DEFAULT_WORKSTATION_PREFLIGHT: dict[str, Any] = {
    "enabled": True,
    "require_external_power": True,
    "required_power_profile": "performance",
    "expected_robot_source_ip": "192.168.1.10",
    "require_direct_route": True,
}


def _settings(hardware: dict[str, Any]) -> dict[str, Any]:
    configured = hardware.get("workstation_preflight", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, dict):
        raise WorkstationSafetyError("hardware.workstation_preflight must be a mapping")
    merged = dict(DEFAULT_WORKSTATION_PREFLIGHT)
    merged.update(configured)
    if merged.get("enabled") is not True:
        raise WorkstationSafetyError("live output cannot disable workstation preflight")
    for key in ("require_external_power", "require_direct_route"):
        if not isinstance(merged.get(key), bool):
            raise WorkstationSafetyError(f"workstation preflight {key} must be boolean")
    for key in ("required_power_profile", "expected_robot_source_ip"):
        if not str(merged.get(key, "")).strip():
            raise WorkstationSafetyError(f"workstation preflight {key} is required")
    return merged


def _external_power_supply(power_supply_root: Path) -> str:
    try:
        supplies = sorted(power_supply_root.iterdir())
    except OSError as exc:
        raise WorkstationSafetyError(
            f"could not inspect external power under {power_supply_root}: {exc}"
        ) from None
    system_battery = False
    for supply in supplies:
        try:
            supply_type = (supply / "type").read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if supply_type == "battery":
            # Wireless mice and controllers report scope "Device"; only a
            # system battery can power the workstation itself.
            try:
                scope = (supply / "scope").read_text(encoding="utf-8").strip().lower()
            except OSError:
                scope = "system"
            if scope != "device":
                system_battery = True
            continue
        try:
            online = (supply / "online").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if online == "1":
            return supply.name
    if not system_battery:
        # A desktop without a system battery can only be running on mains.
        return "mains-no-battery"
    raise WorkstationSafetyError(
        "external AC power could not be verified; connect the laptop charger"
    )


def _run_text(
    command: list[str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> str:
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkstationSafetyError(
            f"workstation preflight could not run {' '.join(command)}: {exc}"
        ) from None
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise WorkstationSafetyError(
            f"workstation preflight command failed ({' '.join(command)}): {detail}"
        )
    return result.stdout.strip()


def _robot_route(
    robot_ip: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    raw = _run_text(["ip", "-j", "route", "get", robot_ip], runner)
    try:
        routes = json.loads(raw)
    except json.JSONDecodeError:
        raise WorkstationSafetyError(
            f"route lookup for {robot_ip} did not return valid JSON"
        ) from None
    if not isinstance(routes, list) or len(routes) != 1 or not isinstance(routes[0], dict):
        raise WorkstationSafetyError(
            f"route lookup for {robot_ip} did not return exactly one route"
        )
    return routes[0]


def run_workstation_preflight(
    hardware: dict[str, Any],
    robot_ips: Sequence[str],
    *,
    power_supply_root: Path = Path("/sys/class/power_supply"),
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> WorkstationPreflightReport:
    """Fail closed on the workstation conditions that broke prior live runs."""

    settings = _settings(hardware)
    addresses = tuple(str(ip).strip() for ip in robot_ips)
    if not addresses or any(not address for address in addresses):
        raise WorkstationSafetyError("workstation preflight requires every robot IP")

    power_supply = "not-required"
    if settings["require_external_power"]:
        power_supply = _external_power_supply(power_supply_root)

    required_profile = str(settings["required_power_profile"]).strip()
    power_profile = _run_text(["powerprofilesctl", "get"], runner)
    if power_profile != required_profile:
        raise WorkstationSafetyError(
            f"power profile is {power_profile!r}; select {required_profile!r} before live output"
        )

    expected_source = str(settings["expected_robot_source_ip"]).strip()
    route_devices: list[str] = []
    for robot_ip in addresses:
        route = _robot_route(robot_ip, runner)
        source = str(route.get("prefsrc", "")).strip()
        device = str(route.get("dev", "")).strip()
        gateway = str(route.get("gateway", "")).strip()
        if source != expected_source:
            raise WorkstationSafetyError(
                f"route to {robot_ip} uses source {source or '<none>'}, expected "
                f"dedicated robot Ethernet source {expected_source}"
            )
        if settings["require_direct_route"] and gateway:
            raise WorkstationSafetyError(
                f"route to {robot_ip} goes through gateway {gateway}; a direct robot "
                "Ethernet route is required"
            )
        if not device:
            raise WorkstationSafetyError(f"route to {robot_ip} has no network device")
        route_devices.append(device)

    return WorkstationPreflightReport(
        power_supply=power_supply,
        power_profile=power_profile,
        route_devices=tuple(route_devices),
        route_source_ip=expected_source,
    )
