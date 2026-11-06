"""Is a teleop run active right now? (launcher process running, or its telemetry.csv still growing)

Used by the recorder (warn when recording without an arm run, or an arm run goes unrecorded) and by the offline
tools (pause heavy work while the arms run).
"""
from __future__ import annotations

import time
from pathlib import Path

HARDWARE_LAUNCHERS = (b"run_dual_hardware.py", b"run_hardware.py")
SIM_LAUNCHERS = (b"run_dual_sim.py", b"run_live_sim.py")
FRESH_S = 3.0  # telemetry written within this many seconds counts as live


def launcher_of(argv: list[bytes], names: tuple[bytes, ...]) -> str | None:
    """The launcher a process runs, if it is `<python> [options] <path>/<launcher>.py ...`; else None.

    Only the program and its script argument count: a shell, editor or grep that merely mentions the
    file name is not a launcher.
    """
    if not argv or not Path(argv[0].decode(errors="ignore")).name.startswith("python"):
        return None
    for arg in argv[1:4]:
        name = Path(arg.decode(errors="ignore")).name.encode()
        if name in names:
            return name.decode()
        if not arg.startswith(b"-"):
            return None
    return None


def running_launchers(names: tuple[bytes, ...] = HARDWARE_LAUNCHERS + SIM_LAUNCHERS) -> list[str]:
    found = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        launcher = launcher_of(argv, names)
        if launcher:
            found.add(launcher)
    return sorted(found)


def latest_telemetry(roots) -> tuple[Path | None, float | None]:
    """The most recently written telemetry.csv under the runs roots (only the two newest date folders)."""
    newest, newest_mtime = None, None
    for root in map(Path, roots):
        if not root.is_dir():
            continue
        days = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)[:2]
        for day in days:
            for telemetry in day.glob("*/telemetry.csv"):
                try:
                    mtime = telemetry.stat().st_mtime
                except OSError:
                    continue
                if newest_mtime is None or mtime > newest_mtime:
                    newest, newest_mtime = telemetry, mtime
    return newest, newest_mtime


def arm_activity(roots) -> dict:
    launchers = running_launchers()
    telemetry, mtime = latest_telemetry(roots)
    age = None if mtime is None else time.time() - mtime
    return {
        "active": bool(launchers) or (age is not None and age < FRESH_S),
        "launchers": launchers,
        "hardware": any(name.encode() in HARDWARE_LAUNCHERS for name in launchers),
        "run": None if telemetry is None else telemetry.parent.name,
        "telemetry_age_s": None if age is None else round(age, 1),
    }
