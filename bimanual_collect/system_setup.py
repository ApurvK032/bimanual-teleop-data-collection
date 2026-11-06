"""What this PC still needs from the one-time system setup, and running it with the user's password.

The root part is scripts/system_setup_root.sh, started through pkexec: Ubuntu shows its own password dialog
(polkit), the same one the Software app uses. The app never sees the password. Without a desktop session (for
example over SSH) there is no dialog, so the app shows the command to run instead.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from . import rig

SCRIPT = rig.ROOT / "scripts" / "system_setup_root.sh"
# Our own file: the ZED SDK installer rewrites 99-slabs.rules without the video device line.
UDEV_RULE = Path("/etc/udev/rules.d/98-zed-video.rules")
TOOLS = {"ffmpeg": "ffmpeg", "v4l2-ctl": "v4l-utils", "adb": "adb", "powerprofilesctl": "power-profiles-daemon"}


def needs() -> list[dict]:
    found = []
    try:
        rule_ok = UDEV_RULE.exists() and any("video4linux" in line and 'idVendor}=="2b03"' in line
                                             for line in UDEV_RULE.read_text().splitlines())
    except OSError:
        rule_ok = False
    if not rule_ok:
        found.append({"id": "udev", "label": "ZED camera access for normal users (udev rule)"})
    missing = [package for tool, package in TOOLS.items() if shutil.which(tool) is None]
    if missing:
        found.append({"id": "packages", "label": f"System packages: {', '.join(missing)}"})
    return found


def command(network_if: str | None = None, dry_run: bool = False) -> list[str]:
    args = [str(SCRIPT), *(["--network", network_if] if network_if else []), *(["--dry-run"] if dry_run else [])]
    return args if dry_run else ["pkexec", *args]


def can_ask_password() -> bool:
    """A graphical session where polkit can show its dialog."""
    return shutil.which("pkexec") is not None and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def run(network_if: str | None = None, dry_run: bool = False) -> dict:
    cmd = command(network_if, dry_run)
    shown = "sudo " + " ".join(cmd[1:]) if not dry_run else " ".join(cmd)
    if not dry_run and not can_ask_password():
        return {"ok": False, "output": "", "command": shown,
                "detail": "No desktop session here to ask for the password: run this command in a terminal on the PC."}
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "output": "", "command": shown, "detail": str(exc)}
    output = (result.stdout + result.stderr).strip()
    if result.returncode in (126, 127) and not dry_run:  # pkexec: dialog dismissed / not authorised
        return {"ok": False, "output": output, "command": shown, "detail": "The password dialog was cancelled."}
    return {"ok": result.returncode == 0, "output": output[-4000:], "command": shown,
            "detail": "done" if result.returncode == 0 else f"exit code {result.returncode}"}
