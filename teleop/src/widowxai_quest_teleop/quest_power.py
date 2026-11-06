from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence


ADB_COMMAND_TIMEOUT_S = 3.0


def prepare_tabletop_tracking(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Wake a paused Quest WebXR session without fabricating tracking poses."""

    commands: Sequence[Sequence[str]] = (
        ("adb", "get-state"),
        (
            "adb",
            "shell",
            "am",
            "broadcast",
            "-a",
            "com.oculus.vrpowermanager.automation_disable",
        ),
        (
            "adb",
            "shell",
            "am",
            "broadcast",
            "-a",
            "com.oculus.vrpowermanager.prox_close",
        ),
        (
            "adb",
            "shell",
            "am",
            "broadcast",
            "-a",
            "com.oculus.vrpowermanager.wake",
        ),
        ("adb", "shell", "svc", "power", "stayon", "true"),
    )
    try:
        for command in commands:
            completed = runner(
                list(command),
                capture_output=True,
                text=True,
                timeout=ADB_COMMAND_TIMEOUT_S,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "unknown ADB error").strip()
                raise RuntimeError(f"{' '.join(command)} failed: {detail}")
        state = runner(
            ["adb", "shell", "dumpsys", "vrpowermanager"],
            capture_output=True,
            text=True,
            timeout=ADB_COMMAND_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not prepare Quest tabletop tracking: {exc}") from None

    if state.returncode != 0:
        detail = (state.stderr or state.stdout or "unknown ADB error").strip()
        raise RuntimeError(f"could not verify Quest power state: {detail}")
    power_state = state.stdout.replace("\r", "")
    required = ("Virtual proximity state: CLOSE", "State: HEADSET_MOUNTED")
    if not all(item in power_state.splitlines() for item in required):
        raise RuntimeError("Quest did not enter the verified virtual mounted state")
