"""Recorder fault handling with fake cameras: camera loss, encoder death, low disk, crash recovery, labels."""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from bimanual_collect import catalog, episode_io, recorder
from bimanual_collect.recorder import Hub
from bimanual_collect.recovery import recover_episode
from bimanual_collect.zed_format import decode_command
from bimanual_collect.zed_v4l2 import Frame

W, H, FPS = 672, 188, 30  # small side-by-side frames keep the tests fast


class FlakyCamera:
    """Fake source: smooth frames at FPS; `fail_reads` makes the next reads raise (a camera that stopped)."""

    device = "fake"
    mode = "test"

    def __init__(self) -> None:
        self.width, self.height, self.fps = W, H, FPS
        self.dropped_error = self.dropped_short = 0
        self.fail_reads = 0
        self.opens = 0
        self._sequence = 0
        self._lock = threading.Lock()

    def open(self) -> None:
        self.opens += 1
        self._sequence = 0  # a reopened camera restarts its sequence count

    def read(self, timeout_s: float = 2.0) -> Frame:
        time.sleep(1 / self.fps)
        with self._lock:
            if self.fail_reads > 0:
                self.fail_reads -= 1
                raise OSError(19, "No such device")
            value = (self._sequence * 7) % 200 + 20
            data = np.full((H, W, 2), value, np.uint8)
            data[..., 1] = 128
            now = time.perf_counter_ns()
            frame = Frame(data.tobytes(), W, H, self._sequence, now - 20_000_000, now, time.time_ns())
            self._sequence += 1
            return frame

    def close(self) -> None:
        pass


def wait_until(condition, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def decoded_frames(video: Path) -> int:
    data = subprocess.run(decode_command(video), capture_output=True, check=True).stdout
    return len(data) // (W * H * 2)


@pytest.fixture
def hub(tmp_path):
    camera = FlakyCamera()
    hub = Hub(data_root=tmp_path, camera=camera, ffv1_threads=2)
    hub.start()
    yield hub, camera
    hub.stop()


def test_camera_loss_aborts_the_episode_and_the_camera_comes_back(hub) -> None:
    hub, camera = hub
    hub.start_episode("fold")
    time.sleep(0.5)
    camera.fail_reads = 3
    assert wait_until(lambda: not hub.recording, 10), "episode was not closed after the camera failed"
    meta = hub.record_status()["last_episode"]
    assert meta["status"] == "aborted" and "lost" in meta["stop_reason"]
    cam = episode_io.primary(Path(meta["directory"]))
    assert decoded_frames(cam.video) == len(cam.rows()) == meta["frames_recorded"] > 5  # finalised and readable
    # It keeps reopening while the camera stays gone (each failed read costs one reopen), then recovers.
    assert wait_until(lambda: camera.fail_reads == 0 and hub.primary.state == "ok" and hub.primary.reconnects >= 1, 15)
    status = hub.start_episode("again")  # recording works again after the reconnect
    assert status["recording"]
    time.sleep(0.3)
    assert hub.stop_episode()["last_episode"]["status"] == "complete"


def test_encoder_death_aborts_the_episode(hub) -> None:
    hub, _ = hub
    hub.start_episode("fold")
    time.sleep(0.4)
    hub.primary.writer.process.kill()
    assert wait_until(lambda: not hub.recording, 10)
    meta = hub.record_status()["last_episode"]
    assert meta["status"] == "aborted" and "encoder" in meta["stop_reason"]


def test_svo2_boundaries_match_the_csv_across_episodes(tmp_path, monkeypatch) -> None:
    class SvoCamera(FlakyCamera):
        calibration = tmp_path / "SN1.conf"
        recorded = None

        def start_svo(self, directory):
            self.recorded = []

        def read(self):
            frame = super().read()
            if self.recorded is not None:
                self.recorded.append(frame.sequence)
            return frame

        def stop_svo(self):
            sequences.extend(self.recorded)
            self.recorded = None

    sequences = []
    monkeypatch.setattr(recorder, "svo_frame_count", lambda path: len(sequences))
    spec = recorder.fixed_source_spec(SvoCamera())
    spec.svo2 = True
    hub = Hub(cameras=[spec], data_root=tmp_path, ffv1_threads=2)
    hub.start()
    try:
        for index in range(2):
            sequences.clear()
            status = hub.start_episode("fold")
            assert status["episode"]["index"] == index
            assert wait_until(lambda: hub.record_status()["episode"]["frames"] >= 5)
            meta = hub.stop_episode()["last_episode"]
            assert meta["status"] == "complete"
            camera = episode_io.primary(Path(meta["directory"]))
            assert [int(row["camera_sequence"]) for row in camera.rows()] == sequences
            assert meta["cameras"][0]["svo2"]["frames_recorded"] == len(sequences)
    finally:
        hub.stop()


def test_low_disk_stops_a_running_episode(hub, monkeypatch) -> None:
    hub, _ = hub
    hub.start_episode("fold")
    time.sleep(0.3)
    usage = shutil.disk_usage("/")
    monkeypatch.setattr(recorder.shutil, "disk_usage", lambda path: usage._replace(free=int(5e9)))
    hub._monitor_once()
    assert wait_until(lambda: not hub.recording, 10)
    meta = hub.record_status()["last_episode"]
    assert meta["status"] == "complete" and "low disk" in meta["stop_reason"]
    with pytest.raises(RuntimeError, match="GB free"):
        hub.start_episode("again")


def test_crash_leaves_a_recoverable_episode(hub, tmp_path) -> None:
    hub, _ = hub
    status = hub.start_episode("fold")
    directory = Path(status["episode"]["directory"])
    assert json.loads((directory / "episode.json").read_text())["status"] == "recording"  # written at start
    time.sleep(2.5)  # past at least one CSV flush
    crashed = tmp_path / "crash" / "2026-01-01" / "session_000000" / "episode_0000"
    shutil.copytree(directory, crashed)  # what a power cut would leave behind: no stop, unfinalised video
    hub.stop_episode()

    result = recover_episode(crashed)
    meta = json.loads((crashed / "episode.json").read_text())
    assert result is not None and meta["status"] == "recovered" and meta["duration_s"] > 1
    cam = episode_io.primary(crashed)
    rows = cam.rows()
    assert len(rows) == meta["frames_recorded"] > 10
    assert decoded_frames(cam.video) >= len(rows)  # every listed frame decodes
    assert recover_episode(crashed) is None  # only once


def test_labels_and_qc_of_an_unclean_episode(hub) -> None:
    hub, camera = hub
    hub.set_defaults(task="unfold the handkerchief", operator="ana")
    hub.start_episode()
    time.sleep(0.3)
    hub.stop_episode(outcome="failure")
    meta = hub.record_status()["last_episode"]
    episode = Path(meta["directory"])
    assert meta["labels"]["task"] == "unfold the handkerchief" and meta["labels"]["operator"] == "ana"
    hub.label_episode(note="cloth slipped", outcome="unsure")
    assert catalog.label_of(episode, "note") == "cloth slipped"
    assert catalog.label_of(episode, "outcome") == "unsure"
    with pytest.raises(ValueError):
        hub.label_episode(outcome="great")

    unclean = {**json.loads((episode / "episode.json").read_text()), "status": "aborted",
               "stop_reason": "camera zed lost: OSError", "duration_s": 10.0}
    record = catalog.evaluate(episode, unclean, [], {"ok": True, "steps": 300}, {"ok": True, "shift_ms": 1.0, "r": 0.9})
    assert record["verdict"] == "needs_review"
    assert any(reason.startswith("recording: aborted") for reason in record["reasons"])


def test_warns_when_the_arms_run_but_nothing_records(hub, monkeypatch) -> None:
    hub, _ = hub
    monkeypatch.setattr(recorder, "arm_activity", lambda roots: {"active": True, "hardware": True, "launchers": []})
    hub._monitor_once()
    hub._arm_idle_since -= 20  # as if 20 s had passed
    hub._monitor_once()
    assert any("nothing is being recorded" in warning for warning in hub.record_status()["warnings"])


def test_launcher_detection_ignores_mentions() -> None:
    from bimanual_collect.activity import HARDWARE_LAUNCHERS, launcher_of

    def argv(text: str) -> list[bytes]:
        return [part.encode() for part in text.split()]

    assert launcher_of(argv(".venv-arm19/bin/python scripts/run_dual_hardware.py --live"), HARDWARE_LAUNCHERS)
    assert launcher_of(argv("/usr/bin/python3.10 -u scripts/run_hardware.py"), HARDWARE_LAUNCHERS)
    assert launcher_of(argv("vim scripts/run_hardware.py"), HARDWARE_LAUNCHERS) is None
    assert launcher_of(argv("bash -c grep run_dual_hardware.py log"), HARDWARE_LAUNCHERS) is None
    assert launcher_of(argv("python -m pytest tests/test_run_hardware.py"), HARDWARE_LAUNCHERS) is None


class MonoCamera(FlakyCamera):
    """A plain UVC-style camera: one image per frame, its own size and rate."""

    def __init__(self) -> None:
        super().__init__()
        self.width, self.height, self.fps = 320, 240, 15

    def read(self, timeout_s: float = 2.0) -> Frame:
        time.sleep(1 / self.fps)
        data = np.full((self.height, self.width, 2), 90, np.uint8)
        now = time.perf_counter_ns()
        frame = Frame(data.tobytes(), self.width, self.height, self._sequence, now - 10_000_000, now, time.time_ns())
        self._sequence += 1
        return frame


def test_two_cameras_record_into_one_episode(tmp_path) -> None:
    specs = [recorder.fixed_source_spec(FlakyCamera(), name="zed", stereo=True, primary=True),
             recorder.fixed_source_spec(MonoCamera(), name="wrist", stereo=False, primary=False, colour_fix=False)]
    hub = Hub(specs, data_root=tmp_path, ffv1_threads=2)
    hub.start()
    try:
        hub.start_episode("fold")
        time.sleep(1.0)
        meta = hub.stop_episode()["last_episode"]
    finally:
        hub.stop()
    episode = Path(meta["directory"])
    cams = {cam.name: cam for cam in episode_io.cameras(episode)}
    assert set(cams) == {"zed", "wrist"} and cams["zed"].primary and not cams["wrist"].stereo
    assert (cams["wrist"].width, cams["wrist"].height, cams["wrist"].fps) == (320, 240, 15)
    for cam in cams.values():
        data = subprocess.run(decode_command(cam.video), capture_output=True, check=True).stdout
        assert len(data) // (cam.width * cam.height * 2) == len(cam.rows()) > 5
    assert hub.workers[1].view.crop_width == 320  # mono view: the whole frame
