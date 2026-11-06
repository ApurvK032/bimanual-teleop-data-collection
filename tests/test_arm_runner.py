"""Arm runner with a fake launcher: phases, exactly one SIGINT, the headset gate, re-attach, refusals."""
from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import pytest

from bimanual_collect import arm_runner
from bimanual_collect.arm_runner import ArmRunner

FAKE = Path(__file__).with_name("fake_launcher.py")


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timed out")


@pytest.fixture
def runner(tmp_path):
    runners = []

    def make(presence=None):
        r = ArmRunner(state_dir=tmp_path / "state", runs_roots=[tmp_path / "runs"], presence=presence, poll_s=0.05)
        runners.append(r)
        return r

    yield make
    for r in runners:
        if r.run is not None and r.run.active and r.run.pid:
            try:
                r.run.process and r.run.process.kill()
            except OSError:
                pass
        r.close()


def start(r, tmp_path, mode, kind="dry", live_s=30.0, label="test-run"):
    command = [sys.executable, str(FAKE), str(tmp_path / "runs"), label, mode, str(live_s)]
    return r.start(kind=kind, title=f"fake {mode}", command=command, cwd=tmp_path, label=label, profile="fake.yaml",
                   arms={"left": "192.0.2.3", "right": "192.0.2.2"}, cap_s=live_s, error_limit_rad=0.12)


def test_phases_live_numbers_and_single_sigint(runner, tmp_path, monkeypatch):
    r = runner()
    start(r, tmp_path, "normal", kind="live")
    wait_for(lambda: r.run.phase == "live")
    assert r.run.motion_started
    live = wait_for(lambda: (r.run.status()["live"] or {}).get("rows", 0) > 30 and r.run.status()["live"])
    assert live["loop_hz"] and 40 < live["loop_hz"] < 120
    assert set(live["arms"]) == {"left", "right"}
    assert live["arms"]["left"]["max_error_rad"] == pytest.approx(0.006, abs=1e-3)
    assert r.brief()["line"].startswith("ARMS LIVE")

    sent = []
    real_kill = arm_runner.os.kill
    monkeypatch.setattr(arm_runner.os, "kill", lambda pid, sig: (sent.append(sig), real_kill(pid, sig)))
    with pytest.raises(RuntimeError, match="run changed"):
        r.stop("previous-run")
    r.stop(r.run.id)
    r.stop()  # a second press must not send a second SIGINT (it would skip the return to rest)
    assert sent == [signal.SIGINT]
    wait_for(lambda: not r.run.active)
    assert r.run.outcome == "done" and r.run.returncode == 0
    assert [line for line in r.run.lines if "rest reached" in line]
    assert not (tmp_path / "state" / "arm_run.json").exists()


def test_delayed_stop_does_not_interrupt_automatic_return(runner, tmp_path, monkeypatch):
    r = runner()
    start(r, tmp_path, "normal", kind="live", live_s=0.5)
    wait_for(lambda: r.run.phase == "live")
    sent = []
    monkeypatch.setattr(arm_runner.os, "kill", lambda pid, sig: sent.append(sig))
    with r._lock:  # simulate a Stop arriving before the monitor sees the launcher's latest output
        wait_for(lambda: "rest reached: left arm" in Path(r.run.log).read_text())
        status = r.stop(r.run.id)
        assert status["phase"] == "returning" and not status["stop_sent"]
    wait_for(lambda: not r.run.active)
    assert not sent and r.run.outcome == "done"


def test_child_gets_sigint_even_if_app_ignores_it(runner, tmp_path):
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        r = runner()
        start(r, tmp_path, "normal")
        wait_for(lambda: r.run.phase == "live")
        r.stop()
        wait_for(lambda: not r.run.active)
        assert r.run.phase == "done" and any("operator stop" in line for line in r.run.lines)
    finally:
        signal.signal(signal.SIGINT, previous)


def test_safety_stop_and_emergency(runner, tmp_path):
    r = runner()
    start(r, tmp_path, "safety", kind="live")
    wait_for(lambda: not r.run.active)
    assert r.run.outcome == "failed" and r.run.error.startswith("Safety stop:")
    r2 = runner()
    start(r2, tmp_path, "emergency", kind="live", label="emergency")
    wait_for(lambda: r2.run.phase == "live")
    r2.stop()
    wait_for(lambda: not r2.run.active)
    assert r2.run.emergency and r2.run.outcome == "failed"  # exit code 0, but the warning wins
    assert r2.brief() == {"line": "CUT ARM POWER NOW", "level": "bad"}


def test_hung_stop_never_escalates_after_motion(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(arm_runner, "STOP_GRACE_S", 0.5)
    r = runner()
    start(r, tmp_path, "hang", kind="live")
    wait_for(lambda: r.run.phase == "live")
    r.stop()
    wait_for(lambda: r.run.status()["stop_overdue"])
    assert r.run.active and not r.run.can_force_quit()
    with pytest.raises(RuntimeError, match="power"):
        r.force_quit()
    r.run.process.kill()  # test cleanup only


def test_crash_reports_the_last_line(runner, tmp_path):
    r = runner()
    start(r, tmp_path, "crash")
    wait_for(lambda: not r.run.active)
    assert r.run.outcome == "failed" and "simulated crash" in r.run.error
    assert not r.run.motion_started


def test_contact_output_is_parsed(runner, tmp_path):
    r = runner()
    start(r, tmp_path, "contact", kind="contact")
    wait_for(lambda: not r.run.active)
    contact = r.run.contact
    assert r.run.outcome == "done"
    assert contact["left"] == {"ip": "192.0.2.3", "driver": "1.9.3", "firmware": "1.9.4",
                               "q_rad": [-0.066, -0.0, 0.003, -0.001, 0.02, 0.012], "gripper_m": 0.0024}
    assert contact["right"]["ip"] == "192.0.2.2"


def test_waits_for_headset_then_launches_and_can_cancel(runner, tmp_path):
    ready = {"ok": False}
    r = runner(presence=lambda run: (ready["ok"], "enter VR in bimanual mode"))
    start(r, tmp_path, "normal")
    time.sleep(0.3)
    assert r.run.phase == "waiting_headset" and r.run.pid is None and r.run.message == "enter VR in bimanual mode"
    ready["ok"] = True
    wait_for(lambda: r.run.phase == "live")
    r.stop()
    wait_for(lambda: not r.run.active)

    ready["ok"] = False
    start(r, tmp_path, "normal", label="cancelled")
    time.sleep(0.2)
    r.stop()
    assert r.run.outcome == "cancelled" and r.run.pid is None


def test_one_at_a_time_and_reattach(runner, tmp_path):
    r = runner()
    start(r, tmp_path, "normal", kind="live")
    wait_for(lambda: r.run.phase == "live")
    with pytest.raises(RuntimeError, match="still running"):
        start(r, tmp_path, "normal")
    pid = r.run.pid
    process = r.run.process
    r.close()  # the app goes away; the launcher keeps running

    r2 = runner()
    assert r2.run is not None and r2.run.adopted and r2.run.pid == pid and r2.run.motion_started
    wait_for(lambda: r2.run.phase == "live")
    r2.stop()
    process.wait(timeout=10)  # reap (in a real restart the old app is gone and init reaps it)
    wait_for(lambda: not r2.run.active)
    assert r2.run.phase == "done"


def test_driver_process_detection():
    argv = [b"/x/.venv-arm19/bin/python", b"-u", b"scripts/preflight_dual_hardware.py", b"--contact-arms"]
    assert arm_runner.launcher_of(argv, arm_runner.DRIVER_SCRIPTS) == "preflight_dual_hardware.py"
    assert arm_runner.launcher_of([b"vim", b"scripts/run_hardware.py"], arm_runner.DRIVER_SCRIPTS) is None
