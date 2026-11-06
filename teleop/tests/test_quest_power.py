from types import SimpleNamespace

import pytest

from widowxai_quest_teleop.quest_power import prepare_tabletop_tracking


def test_tabletop_tracking_pulses_mount_and_verifies_state() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        stdout = ""
        if command[-2:] == ["dumpsys", "vrpowermanager"]:
            stdout = "Virtual proximity state: CLOSE\nState: HEADSET_MOUNTED\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    prepare_tabletop_tracking(runner)

    actions = [command[-1] for command, _ in calls if "broadcast" in command]
    assert actions == [
        "com.oculus.vrpowermanager.automation_disable",
        "com.oculus.vrpowermanager.prox_close",
        "com.oculus.vrpowermanager.wake",
    ]
    assert calls[-1][0] == ["adb", "shell", "dumpsys", "vrpowermanager"]


def test_tabletop_tracking_rejects_unverified_mount_state() -> None:
    def runner(command, **_kwargs):
        stdout = "State: HEADSET_UNMOUNTED\n" if "dumpsys" in command else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    with pytest.raises(RuntimeError, match="verified virtual mounted state"):
        prepare_tabletop_tracking(runner)
