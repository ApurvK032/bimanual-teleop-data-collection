from __future__ import annotations

import stat
from types import SimpleNamespace

import pytest

from widowxai_quest_teleop import hold_to_run
from widowxai_quest_teleop.hold_to_run import (
    HoldToRunError,
    LinuxEvdevHoldToRun,
    validate_evdev_key_code,
)


def fake_evdev(monkeypatch, *, supported: bool = True, pressed: bool = False):
    state = {"pressed": pressed, "fail": False, "closed": []}
    code = 57

    monkeypatch.setattr(hold_to_run.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hold_to_run.os, "open", lambda _path, _flags: 42)
    monkeypatch.setattr(
        hold_to_run.os,
        "fstat",
        lambda _fd: SimpleNamespace(st_mode=stat.S_IFCHR),
    )
    monkeypatch.setattr(
        hold_to_run.os,
        "close",
        lambda fd: state["closed"].append(fd),
    )

    def ioctl(_fd, request, bits, _mutate):
        if state["fail"]:
            raise OSError("device disconnected")
        bits[:] = b"\x00" * len(bits)
        if request == hold_to_run._eviocgbit(hold_to_run.EV_KEY, len(bits)):
            if supported:
                bits[code // 8] |= 1 << (code % 8)
        elif request == hold_to_run._eviocgkey(len(bits)):
            if state["pressed"]:
                bits[code // 8] |= 1 << (code % 8)
        else:  # pragma: no cover - catches an unexpected ioctl in this unit test
            raise AssertionError(f"unexpected ioctl {request}")
        return 0

    monkeypatch.setattr(hold_to_run.fcntl, "ioctl", ioctl)
    return state


def test_evdev_hold_to_run_polls_real_state_and_counts_edges(monkeypatch) -> None:
    state = fake_evdev(monkeypatch)
    control = LinuxEvdevHoldToRun("/dev/input/by-id/test-event-kbd", 57)

    initial = control.open(require_released=True)
    assert not initial.pressed
    assert initial.press_generation == 0
    assert initial.release_generation == 0

    state["pressed"] = True
    pressed = control.sample(monotonic_ns=100)
    assert pressed.pressed
    assert pressed.press_generation == 1
    assert pressed.release_generation == 0

    state["pressed"] = False
    released = control.sample(monotonic_ns=200)
    assert not released.pressed
    assert released.press_generation == 1
    assert released.release_generation == 1

    control.close()
    assert state["closed"] == [42]


def test_evdev_hold_to_run_rejects_held_start_unsupported_code_and_disconnect(
    monkeypatch,
) -> None:
    held = fake_evdev(monkeypatch, pressed=True)
    control = LinuxEvdevHoldToRun("/dev/input/event1", 57)
    with pytest.raises(HoldToRunError, match="released before live preflight"):
        control.open(require_released=True)
    assert held["closed"] == [42]

    unsupported = fake_evdev(monkeypatch, supported=False)
    control = LinuxEvdevHoldToRun("/dev/input/event2", 57)
    with pytest.raises(HoldToRunError, match="does not support key code"):
        control.open()
    assert unsupported["closed"] == [42]

    disconnected = fake_evdev(monkeypatch)
    control = LinuxEvdevHoldToRun("/dev/input/event3", 57)
    control.open()
    disconnected["fail"] = True
    with pytest.raises(HoldToRunError, match="lost hold-to-run device"):
        control.sample()
    assert not control.opened
    assert disconnected["closed"] == [42]


@pytest.mark.parametrize("value", [True, -1, 768, 1.5, "57", None])
def test_evdev_key_code_validation_is_strict(value) -> None:
    with pytest.raises(HoldToRunError):
        validate_evdev_key_code(value)
