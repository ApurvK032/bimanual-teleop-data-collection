"""The bench: which arms, where, how they are driven. Backs the app's Bench tab.

What the wizard saves lives in configs/local/bench.json (git-ignored). From it the app writes the teleop profiles
the launchers read, all in configs/local/:
  bench.yaml         the run profile: two arms (from configs/examples/dual_widowxai_bench.yaml) or one arm
                     (from configs/examples/left_arm_40pct_full_gripper.yaml)
  bench_left.yaml    two-arm benches only: one-arm profiles at the validated 0.40 response, for the first-run
  bench_right.yaml   check (one arm moves, the other stays connected read-only and screens every command)
and points configs/rig.yaml at bench.yaml (`arms`, `paths.arm_profile`).

v1 layout: arms side by side and parallel at the table edge, operator standing behind them. Each controller keeps
its hand's accepted Behind calibration; those map hand motion into the arm's own base frame, so they carry over to
any bench with this layout. Only the arm addresses, sides, base separation and the driving settings change.

Network rules kept here: arms are found and watched with ping only (never a connection to the arm's control
port, teleop AGENTS.md rule 7); the PC's arm-side address is 192.168.1.10 (what the teleop workstation check
expects).
"""
from __future__ import annotations

import ipaddress
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from . import rig

LOCAL = rig.ROOT / "configs" / "local"
STATE_FILE = LOCAL / "bench.json"
PROFILE = LOCAL / "bench.yaml"
SIDE_PROFILES = {side: LOCAL / f"bench_{side}.yaml" for side in ("left", "right")}
DUAL_EXAMPLE = rig.ROOT / "configs" / "examples" / "dual_widowxai_bench.yaml"
SINGLE_EXAMPLE = rig.ROOT / "configs" / "examples" / "left_arm_40pct_full_gripper.yaml"
SOURCE_IP = "192.168.1.10"  # the PC's address on the arm network, as the teleop workstation check expects
SUBNET = f"{SOURCE_IP.rsplit('.', 1)[0]}.0/24"
CONNECTION_NAME = "widowxai-arms"
CALIBRATIONS = {side: f"configs/calibrations/{side}_behind_all_motions_20260812_accepted.json" for side in ("left", "right")}

# Response presets: (translation scale, rotation scale). 0.40 is the scope the Behind calibrations were accepted at.
RESPONSES = {
    "validated": (0.40, 0.40, "Gentle: 0.40 (the validated scope; use on a new bench)"),
    "standard": (0.60, 0.60, "Standard: 0.60"),
    "fast": (0.70, 0.60, "Fast: 0.70 move / 0.60 turn"),
}
TRACKING_STOPS = (0.08, 0.12)
CAPS_S = (45, 60, 90, 120, 300, 600)
# Two-arm benches: the combined collision check refuses any command that brings the arms closer than this.
CLEARANCES_MM = (15, 20, 25, 30)
# Optional table floor in the same check: any arm part (except the bases) touching the table is refused. The floor
# can sit a little below the real surface so the gripper may touch it gently but never press into it.
FLOOR_OFFSETS_MM = (0, 1, 2, 3, 4, 5)
TABLE_THICKNESS_M = 0.04
# The lists above are the offered choices; the Bench tab also takes a custom value inside these ranges.
RANGES = {
    "custom_move": (0.20, 1.00, "custom move response"),
    "custom_turn": (0.20, 1.00, "custom turn response"),
    "tracking_stop_rad": (0.05, 0.15, "tracking stop (rad)"),
    "cap_s": (30, 1800, "run time limit (s)"),
    "clearance_mm": (10, 50, "arm-to-arm clearance (mm)"),
    "table_floor_offset_mm": (0, 10, "table floor offset (mm)"),
}
DEFAULTS = {
    "arms": 2, "left_ip": None, "right_ip": None, "single_hand": "right", "single_ip": None, "identified": False,
    "spacing_mm": 590, "spacing_measured": False, "response": "validated", "custom_move": 0.60, "custom_turn": 0.60,
    "tracking_stop_rad": 0.08, "cap_s": 60,
    "gripper": True, "clearance_mm": 30, "table_floor": False, "table_width_mm": 1400, "table_depth_mm": 720,
    "table_rear_inset_mm": 51, "table_floor_offset_mm": 0, "validated": None, "updated": None,
}


def scales(state: dict, response: str | None = None) -> tuple[float, float]:
    """(translation scale, rotation scale) of a preset, or the custom values."""
    key = response or state["response"]
    if key == "custom":
        return float(state["custom_move"]), float(state["custom_turn"])
    move, turn, _ = RESPONSES[key]
    return move, turn


def _in_range(out: dict, key: str, kind=float) -> None:
    low, high, name = RANGES[key]
    out[key] = kind(round(float(out[key]), 2) if kind is float else round(float(out[key])))
    if not low <= out[key] <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}")


# ----------------------------------------------------------------------------- state
def load_state() -> dict:
    try:
        return DEFAULTS | json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        imported = import_profile(rig.load().arm_profile)
        return DEFAULTS | imported


def import_profile(path: Path) -> dict:
    """Prefill the wizard from an existing hand-made profile in configs/local/. The committed examples
    (placeholder addresses) and the teleop repo's own profiles are not a bench, so they are never imported."""
    path = Path(path)
    if LOCAL not in path.resolve().parents:
        return {}
    try:
        config = yaml.safe_load(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(config, dict) or not config.get("hardware"):
        return {}
    hardware = config["hardware"]
    found: dict = {"tracking_stop_rad": float(hardware.get("max_feedback_error_rad", 0.08)),
                   "cap_s": int(float(hardware.get("max_demo_duration_s") or 60)),
                   "gripper": bool(hardware.get("control_gripper", True)), "imported_from": str(path)}
    if "arms" in config:
        arms = config["arms"]
        left, right = arms.get("left", {}), arms.get("right", {})
        found |= {"arms": 2, "left_ip": left.get("robot_ip"), "right_ip": right.get("robot_ip"), "identified": True}
        try:
            y = [float(arm["base_transform"]["position_m"][1]) for arm in (left, right)]
            found |= {"spacing_mm": round(abs(y[0] - y[1]) * 1000), "spacing_measured": True}
        except (KeyError, TypeError, IndexError, ValueError):
            pass
        scales = (float(left.get("translation_scale", 0.4)), float(left.get("rotation_scale", 0.4)))
        try:
            found["clearance_mm"] = int(round(float(config["safety"]["cross_arm_clearance_m"]) * 1000))
        except (KeyError, TypeError, ValueError):
            pass
        table = (config.get("simulation_environment") or {}).get("tabletop")
        if isinstance(table, dict):
            try:
                found |= {"table_floor": True, "table_width_mm": round(float(table["width_m"]) * 1000),
                          "table_depth_mm": round(float(table["depth_m"]) * 1000),
                          "table_rear_inset_mm": round(-float(table["rear_edge_x_m"]) * 1000),
                          "table_floor_offset_mm": round(-float(table.get("top_z_m", 0.0)) * 1000)}
            except (KeyError, TypeError, ValueError):
                pass
    else:
        quest = config.get("quest", {})
        found |= {"arms": 1, "single_hand": quest.get("hand", "right"), "single_ip": hardware.get("robot_ip"),
                  "identified": True}
        scales = (float(quest.get("translation_scale", 0.4)), float(quest.get("rotation_scale", 0.4)))
    nearest = min(RESPONSES, key=lambda k: abs(RESPONSES[k][0] - scales[0]) + abs(RESPONSES[k][1] - scales[1]))
    if abs(RESPONSES[nearest][0] - scales[0]) + abs(RESPONSES[nearest][1] - scales[1]) < 0.005:
        found["response"] = nearest
    else:
        found |= {"response": "custom", "custom_move": round(scales[0], 2), "custom_turn": round(scales[1], 2)}
    return found


def _ip(value, name: str) -> str:
    try:
        address = ipaddress.ip_address(str(value).strip())
    except ValueError:
        raise ValueError(f"{name}: {value!r} is not an IP address") from None
    if address not in ipaddress.ip_network(SUBNET) or str(address) == SOURCE_IP:
        raise ValueError(f"{name}: {address} must be on {SUBNET} and not the PC's own {SOURCE_IP}")
    return str(address)


def validate(state: dict) -> dict:
    out = DEFAULTS | state
    out["arms"] = int(out["arms"])
    if out["arms"] not in (1, 2):
        raise ValueError("arms must be 1 or 2")
    if out["arms"] == 2:
        for side in ("left", "right"):
            if out[f"{side}_ip"]:
                out[f"{side}_ip"] = _ip(out[f"{side}_ip"], f"{side} arm")
        if out["left_ip"] and out["left_ip"] == out["right_ip"]:
            raise ValueError("the two arms need different addresses")
    else:
        if out["single_hand"] not in ("left", "right"):
            raise ValueError("the hand must be left or right")
        if out["single_ip"]:
            out["single_ip"] = _ip(out["single_ip"], "arm")
    out["spacing_mm"] = int(round(float(out["spacing_mm"])))
    if not 350 <= out["spacing_mm"] <= 1500:
        raise ValueError("base separation must be between 350 and 1500 mm (centre to centre)")
    if out["response"] not in (*RESPONSES, "custom"):
        raise ValueError(f"response must be one of {', '.join(RESPONSES)} or custom")
    _in_range(out, "custom_move")
    _in_range(out, "custom_turn")
    _in_range(out, "tracking_stop_rad")
    _in_range(out, "cap_s", int)
    out["gripper"] = bool(out["gripper"])
    _in_range(out, "clearance_mm", int)
    out["table_floor"] = bool(out["table_floor"])
    for key, low, high, name in (("table_width_mm", 400, 3000, "table width"), ("table_depth_mm", 300, 2000, "table depth"),
                                 ("table_rear_inset_mm", 0, 300, "base distance from the table edge")):
        out[key] = int(round(float(out[key])))
        if not low <= out[key] <= high:
            raise ValueError(f"{name} must be between {low} and {high} mm")
    _in_range(out, "table_floor_offset_mm", int)
    return out


def save_state(changes: dict) -> dict:
    state = load_state()
    before = {k: state.get(k) for k in ("arms", "left_ip", "right_ip", "single_ip", "single_hand", "spacing_mm")}
    state = validate(state | {k: v for k, v in changes.items() if k in DEFAULTS})
    after = {k: state.get(k) for k in before}
    if after != before and "validated" not in changes:
        state["validated"] = None  # a different bench needs its first-run check again
    if any(before[k] != after[k] for k in ("left_ip", "right_ip", "single_ip", "arms")) and "identified" not in changes:
        state["identified"] = False
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    LOCAL.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=1))
    return state


def complete(state: dict) -> bool:
    if state["arms"] == 2:
        return bool(state["left_ip"] and state["right_ip"] and state["identified"] and state["spacing_measured"])
    return bool(state["single_ip"])


# ----------------------------------------------------------------------------- profiles
def set_yaml_values(text: str, values: dict[tuple[str, ...], str]) -> str:
    """Replace scalar values at YAML key paths, keeping the file's layout and comments (inline comments that
    mention 'example' are dropped with the old value). Every path must exist exactly once."""
    lines = text.split("\n")
    stack: list[tuple[int, str]] = []
    used: dict[tuple[str, ...], int] = {}
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)([A-Za-z_][\w-]*):(\s*)(.*)$", line)
        if not match or line.lstrip().startswith("#"):
            continue
        indent, key, rest = len(match.group(1)), match.group(2), match.group(4)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = (*[k for _, k in stack], key)
        if path in values:
            comment = ""
            if "#" in rest:
                kept = rest[rest.index("#"):]
                if "example" not in kept.lower():
                    comment = "  " + kept
            lines[index] = f"{match.group(1)}{key}: {values[path]}{comment}"
            used[path] = used.get(path, 0) + 1
        value = rest.split("#", 1)[0].strip()
        if not value:
            stack.append((indent, key))
    missing = [".".join(p) for p in values if used.get(p) != 1]
    if missing:
        raise ValueError(f"profile template changed: {', '.join(missing)} not found exactly once")
    return "\n".join(lines)


def _header(title: str, state: dict) -> str:
    return (f"# {title}\n# Written by the collection app (Bench tab) on {time.strftime('%Y-%m-%d %H:%M')}. Change it there:\n"
            f"# edits made here are overwritten. Source settings: configs/local/bench.json.\n")


def _strip_header(text: str) -> str:
    lines = text.split("\n")
    while lines and (lines[0].startswith("#") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines)


def dual_profile(state: dict) -> str:
    move, turn = scales(state)
    half = state["spacing_mm"] / 2000
    values = {
        ("arms", "left", "robot_ip"): state["left_ip"], ("arms", "right", "robot_ip"): state["right_ip"],
        ("arms", "left", "base_transform", "position_m"): f"[0.0, {half:.4f}, 0.0]",
        ("arms", "right", "base_transform", "position_m"): f"[0.0, {-half:.4f}, 0.0]",
        ("arms", "left", "translation_scale"): f"{move:.2f}", ("arms", "right", "translation_scale"): f"{move:.2f}",
        ("arms", "left", "rotation_scale"): f"{turn:.2f}", ("arms", "right", "rotation_scale"): f"{turn:.2f}",
        ("hardware", "workstation_preflight", "expected_robot_source_ip"): SOURCE_IP,
        ("hardware", "max_demo_duration_s"): f"{float(state['cap_s']):.1f}",
        ("hardware", "max_feedback_error_rad"): f"{state['tracking_stop_rad']:.2f}",
        ("hardware", "control_gripper"): "true" if state["gripper"] else "false",
        ("safety", "cross_arm_clearance_m"): f"{state['clearance_mm'] / 1000:.3f}",
    }
    body = set_yaml_values(_strip_header(DUAL_EXAMPLE.read_text()), values)
    if state["table_floor"]:
        body = body.rstrip("\n") + "\n" + table_block(state)
    title = (f"Two-arm bench: left {state['left_ip']}, right {state['right_ip']}, bases {state['spacing_mm']} mm apart, "
             f"operator behind. Live token LIVE-WIDOWXAI-DUAL-{state['left_ip']}-{state['right_ip']}.")
    return _header(title, state) + "\n" + body


def table_block(state: dict) -> str:
    """The table floor for the combined collision check. The teleop reads it from `simulation_environment.tabletop`
    in every launcher (simulation, preflight and hardware share one construction path), despite the key's name."""
    return (
        "\n# Table floor (Bench tab): the combined collision check refuses any command that makes an arm part other\n"
        "# than the bases touch this box. World frame: x forward from the bases, y left, z up from the base mounts.\n"
        "simulation_environment:\n"
        "  status: measured_by_operator\n"
        "  tabletop:\n"
        f"    width_m: {state['table_width_mm'] / 1000:.3f}\n"
        f"    depth_m: {state['table_depth_mm'] / 1000:.3f}\n"
        f"    thickness_m: {TABLE_THICKNESS_M:.3f}\n"
        f"    rear_edge_x_m: {-state['table_rear_inset_mm'] / 1000:.4f}\n"
        f"    top_z_m: {-state['table_floor_offset_mm'] / 1000:.4f}\n")


def single_profile(hand: str, ip: str, state: dict, response: str | None = None) -> str:
    move, turn = scales(state, response)
    values = {
        ("project", "mode"): f"{hand}_bench_{int(move * 100)}pct_hardware",
        ("quest", "hand"): hand, ("quest", "calibration"): CALIBRATIONS[hand],
        ("quest", "translation_scale"): f"{move:.2f}", ("quest", "rotation_scale"): f"{turn:.2f}",
        ("hardware", "robot_ip"): ip,
        ("hardware", "max_demo_duration_s"): f"{float(state['cap_s']):.1f}",
        ("hardware", "max_feedback_error_rad"): f"{state['tracking_stop_rad']:.2f}",
        ("hardware", "control_gripper"): "true" if state["gripper"] else "false",
    }
    body = set_yaml_values(_strip_header(SINGLE_EXAMPLE.read_text()), values)
    body = body.replace("# Left/Behind calibration, accepted at a 0.40 response.",
                        f"# {hand.title()}/Behind calibration, accepted at a 0.40 response.")
    return _header(f"One arm, {hand} controller, {ip}, operator behind. Live token LIVE-WIDOWXAI-{ip}.", state) + "\n" + body


def write_profiles(state: dict) -> dict[str, str]:
    if not complete(state):
        raise ValueError("finish the arm steps first (addresses, sides and base separation)")
    LOCAL.mkdir(parents=True, exist_ok=True)
    written = {}
    if state["arms"] == 2:
        PROFILE.write_text(dual_profile(state))
        for side in ("left", "right"):
            SIDE_PROFILES[side].write_text(single_profile(side, state[f"{side}_ip"], state, response="validated"))
            written[side] = str(SIDE_PROFILES[side])
    else:
        PROFILE.write_text(single_profile(state["single_hand"], state["single_ip"], state))
    written["profile"] = str(PROFILE)
    point_rig_at_profile(state["arms"])
    return written


def point_rig_at_profile(arms: int) -> None:
    """configs/rig.yaml: arms + paths.arm_profile -> configs/local/bench.yaml (comments kept)."""
    target = rig.CANDIDATES[0]
    if not target.exists():
        target.write_text(rig.CANDIDATES[1].read_text())
    text = target.read_text()
    text, n1 = re.subn(r"^(arms: )\S+", lambda m: f"{m.group(1)}{arms}", text, count=1, flags=re.M)
    profile = "configs/local/bench.yaml"
    text, n2 = re.subn(r"^(  arm_profile: )\S+", lambda m: f"{m.group(1)}{profile}", text, count=1, flags=re.M)
    if n2 == 0:
        text, n2 = re.subn(r"^(paths:\n)", lambda m: f"{m.group(1)}  arm_profile: {profile}\n", text, count=1, flags=re.M)
    if n1 == 0:
        text = f"arms: {arms}\n" + text
    target.write_text(text)
    rig.load.cache_clear()


# ----------------------------------------------------------------------------- PC and network
def _run(command: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def interfaces() -> list[dict]:
    try:
        data = json.loads(_run(["ip", "-j", "addr"]).stdout or "[]")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []
    found = []
    for item in data:
        name = item.get("ifname", "")
        sys_dir = Path("/sys/class/net") / name
        if name == "lo" or (sys_dir / "wireless").exists() or not (sys_dir / "device").exists():
            continue  # wired, physical interfaces only (no Wi-Fi, VPN, bridges)
        addresses = [f"{a['local']}/{a['prefixlen']}" for a in item.get("addr_info", []) if a.get("family") == "inet"]
        usb = "usb" in str((sys_dir / "device").resolve())
        found.append({"name": name, "up": item.get("operstate") == "UP", "carrier": "LOWER_UP" in item.get("flags", []),
                      "addresses": addresses, "usb": usb, "arm_address": any(a.startswith(SOURCE_IP + "/") for a in addresses)})
    return found


def route_to(ip: str) -> dict:
    try:
        routes = json.loads(_run(["ip", "-j", "route", "get", ip]).stdout or "[]")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {"ok": False, "detail": "no route"}
    if len(routes) != 1:
        return {"ok": False, "detail": "no route"}
    route = routes[0]
    ok = route.get("prefsrc") == SOURCE_IP and not route.get("gateway")
    gateway = f" gateway {route['gateway']}" if route.get("gateway") else ""
    detail = f"via {route.get('dev')} from {route.get('prefsrc')}{gateway}"
    return {"ok": ok, "dev": route.get("dev"), "source": route.get("prefsrc"), "detail": detail}


def power_profile() -> dict:
    if shutil.which("powerprofilesctl") is None:
        return {"ok": False, "value": None, "detail": "power-profiles-daemon is not installed"}
    try:
        value = _run(["powerprofilesctl", "get"]).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "value": None, "detail": str(exc)}
    return {"ok": value == "performance", "value": value, "detail": value}


def ac_power() -> dict:
    supplies = list(Path("/sys/class/power_supply").glob("*"))
    batteries = [p for p in supplies if (p / "type").exists() and (p / "type").read_text().strip() == "Battery"
                 and (p / "scope").exists() is False]
    mains = [p for p in supplies if (p / "type").exists() and (p / "type").read_text().strip() == "Mains"]
    online = any((p / "online").exists() and (p / "online").read_text().strip() == "1" for p in mains)
    if not batteries:
        return {"ok": True, "detail": "desktop (no battery)"}
    return {"ok": online, "detail": "on mains power" if online else "on battery: plug in the charger"}


def network_status(state: dict | None = None) -> dict:
    state = state or load_state()
    ips = [state["left_ip"], state["right_ip"]] if state["arms"] == 2 else [state["single_ip"]]
    ips = [ip for ip in ips if ip]
    nets = interfaces()
    return {
        "interfaces": nets, "source_ip": SOURCE_IP, "has_source_ip": any(n["arm_address"] for n in nets),
        "routes": {ip: route_to(ip) for ip in ips}, "power_profile": power_profile(), "ac": ac_power(),
        "nmcli": shutil.which("nmcli") is not None,
    }


def fix_network(ifname: str) -> dict:
    """Give a wired interface the arm-side address with NetworkManager (no gateway, never the default route)."""
    if not any(n["name"] == ifname for n in interfaces()):
        raise ValueError(f"{ifname} is not a wired interface on this PC")
    command = ["nmcli", "connection", "add", "type", "ethernet", "ifname", ifname, "con-name", CONNECTION_NAME,
               "ipv4.method", "manual", "ipv4.addresses", f"{SOURCE_IP}/24", "ipv4.never-default", "yes",
               "ipv6.method", "disabled", "connection.autoconnect", "yes"]
    shown = " ".join(command) + f" && nmcli connection up {CONNECTION_NAME}"
    if shutil.which("nmcli") is None:
        return {"ok": False, "detail": "NetworkManager (nmcli) is not available", "command": shown}
    existing = _run(["nmcli", "-t", "-f", "NAME", "connection", "show"]).stdout.split()
    if CONNECTION_NAME in existing:
        _run(["nmcli", "connection", "delete", CONNECTION_NAME], timeout=15)
    added = _run(command, timeout=20)
    if added.returncode != 0:
        return {"ok": False, "detail": (added.stderr or added.stdout).strip(), "command": shown}
    up = _run(["nmcli", "connection", "up", CONNECTION_NAME], timeout=30)
    if up.returncode != 0:
        return {"ok": False, "detail": (up.stderr or up.stdout).strip(), "command": shown}
    return {"ok": True, "detail": f"{ifname} now has {SOURCE_IP}/24 (connection '{CONNECTION_NAME}')"}


def set_performance() -> dict:
    if shutil.which("powerprofilesctl") is None:
        return {"ok": False, "detail": "power-profiles-daemon is not installed: run System setup"}
    result = _run(["powerprofilesctl", "set", "performance"])
    return {"ok": result.returncode == 0, "detail": (result.stderr or result.stdout).strip() or "performance mode on"}


# ----------------------------------------------------------------------------- arms on the network (ping only)
def ping(ip: str, timeout_s: float = 1.0) -> bool:
    try:
        command = ["ping", "-n", "-q", "-c", "1", "-W", str(max(1, int(timeout_s))), ip]
        return _run(command, timeout=timeout_s + 2).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def ping_many(ips: list[str]) -> dict[str, bool]:
    ips = [ip for ip in ips if ip]
    if not ips:
        return {}
    with ThreadPoolExecutor(max_workers=min(64, len(ips))) as pool:
        return dict(zip(ips, pool.map(ping, ips)))


def neighbours() -> dict[str, str]:
    try:
        rows = json.loads(_run(["ip", "-j", "neigh"]).stdout or "[]")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}
    return {row["dst"]: row.get("lladdr", "") for row in rows if row.get("dst")}


def scan() -> list[dict]:
    """Every address on the arm network that answers ping (the PC itself excluded)."""
    hosts = [str(h) for h in ipaddress.ip_network(SUBNET).hosts() if str(h) != SOURCE_IP]
    answered = [ip for ip, ok in ping_many(hosts).items() if ok]
    macs = neighbours()
    return [{"ip": ip, "mac": macs.get(ip, "")} for ip in sorted(answered, key=ipaddress.ip_address)]


# ----------------------------------------------------------------------------- checks with the teleop tools
def teleop_python(arm_env: bool) -> Path:
    settings = rig.load()
    return settings.teleop / ".venv-arm19" / "bin" / "python" if arm_env else rig.ROOT / ".venv" / "bin" / "python"


def _env() -> dict:
    import os

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONUNBUFFERED"] = "1"
    return env


def profile_check(profile: Path = PROFILE) -> dict:
    if not Path(profile).exists():
        return {"ok": False, "errors": ["no bench profile yet: save the bench first"]}
    try:
        result = subprocess.run([str(teleop_python(False)), "-m", "bimanual_collect.profile_check", str(profile)],
                                cwd=rig.ROOT, capture_output=True, text=True, timeout=120, env=_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "errors": [f"the check did not run: {exc}"]}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "errors": [(result.stderr or result.stdout).strip()[-500:] or "the check failed"]}


def workstation_check(profile: Path = PROFILE) -> dict:
    """The teleop repo's own workstation check (AC power, performance mode, a direct route to each arm)."""
    if not Path(profile).exists():
        return {"ok": False, "detail": "no bench profile yet"}
    settings = rig.load()
    try:
        result = subprocess.run([str(teleop_python(False)), "scripts/check_workstation.py", "--config", str(profile)],
                                cwd=settings.teleop, capture_output=True, text=True, timeout=30, env=_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "detail": str(exc)}
    text = (result.stdout + result.stderr).strip()
    failed = re.search(r"WORKSTATION PREFLIGHT FAILED: (.*)", text)
    ok = result.returncode == 0
    return {"ok": ok, "detail": failed.group(1) if failed else ("passed" if ok else text[-300:])}


def sim_picture(profile: Path = PROFILE) -> dict:
    """Render the two-arm bench in MuJoCo (state/sim_scene.png) for a visual check of sides and spacing."""
    out = rig.load().state / "bench_scene.png"
    try:
        result = subprocess.run([str(teleop_python(False)), "-m", "bimanual_collect", "sim-scene", "--config", str(profile),
                                 "--out", str(out)], cwd=rig.ROOT, capture_output=True, text=True, timeout=180, env=_env())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "detail": str(exc)}
    return {"ok": result.returncode == 0 and out.exists(), "path": str(out),
            "detail": (result.stdout + result.stderr).strip()[-400:]}


# ----------------------------------------------------------------------------- launcher commands
def run_spec(state: dict, kind: str, side: str | None = None) -> dict:
    """The command for one arm process, plus what the UI shows and confirms. kind: live | dry | sim | contact |
    guarded (two-arm benches: `side` moves at 0.40, the other arm is read-only)."""
    settings = rig.load()
    teleop = settings.teleop
    arm_python, sim_python = str(teleop_python(True)), str(teleop_python(False))
    dual = state["arms"] == 2
    profile = str(PROFILE)
    cap = float(state["cap_s"])
    arms = {"left": state["left_ip"], "right": state["right_ip"]} if dual else {state["single_hand"]: state["single_ip"]}
    stamp_label = {"live": "app-live", "dry": "app-dry", "sim": "app-sim", "contact": "app-contact",
                   "guarded": f"app-first-run-{side}"}[kind]
    if kind == "contact":
        if dual:
            command = [arm_python, "-u", "scripts/preflight_dual_hardware.py", "--config", profile, "--contact-arms"]
        else:
            command = [arm_python, "-u", "scripts/preflight_hardware.py", "--config", profile, "--robot-ip", state["single_ip"]]
        title = "Read-only arm check"
        token = None
    elif kind == "sim":
        script = "scripts/run_dual_sim.py" if dual else "scripts/run_live_sim.py"
        command = [sim_python, "-u", script, "--config", profile, "--duration", f"{cap:g}", "--label", stamp_label]
        if dual:
            command.append("--tabletop")
        title, token = "Simulation (MuJoCo model of the bench)", None
    elif kind == "dry":
        script = "scripts/run_dual_hardware.py" if dual else "scripts/run_hardware.py"
        command = [arm_python, "-u", script, "--config", profile, "--duration", f"{cap:g}", "--label", stamp_label]
        title, token = "Dry run (arm launcher with simulated arms, no arm contacted)", None
    elif kind == "live":
        if dual:
            token = f"LIVE-WIDOWXAI-DUAL-{state['left_ip']}-{state['right_ip']}"
            command = [arm_python, "-u", "scripts/run_dual_hardware.py", "--config", profile, "--live", "--confirm-live",
                       token, "--duration", f"{cap:g}", "--label", stamp_label]
        else:
            token = f"LIVE-WIDOWXAI-{state['single_ip']}"
            command = [arm_python, "-u", "scripts/run_hardware.py", "--config", profile, "--robot-ip", state["single_ip"],
                       "--live", "--confirm-live", token, "--duration", f"{cap:g}", "--label", stamp_label]
        title = "Live run: " + ("both arms" if dual else f"{state['single_hand']} arm")
    elif kind == "guarded":
        if not dual or side not in ("left", "right"):
            raise ValueError("the one-arm-at-a-time check is for two-arm benches: say which side moves")
        ip = state[f"{side}_ip"]
        token = f"LIVE-WIDOWXAI-{ip}"
        command = [arm_python, "-u", "scripts/run_hardware.py", "--config", str(SIDE_PROFILES[side]), "--robot-ip", ip,
                   "--dual-guard-config", profile, "--moving-side", side, "--live", "--confirm-live", token,
                   "--duration", f"{cap:g}", "--label", stamp_label]
        title = f"First-run check: {side} arm moves at 0.40, the other is read-only"
        arms = {side: ip}
    else:
        raise ValueError(f"unknown run kind {kind!r}")
    headset = "bimanual" if dual and kind != "guarded" else (side if kind == "guarded" else state["single_hand"])
    return {"kind": kind, "command": command, "cwd": str(teleop), "title": title, "label": stamp_label,
            "profile": profile, "arms": arms, "token": token, "cap_s": None if kind == "contact" else cap,
            "error_limit_rad": state["tracking_stop_rad"], "headset_mode": headset}
