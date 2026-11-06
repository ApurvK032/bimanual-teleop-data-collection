"""Offline tests for the data-collection layer (no camera, no Quest, no arm).

    .venv/bin/python -m pytest -q tests
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# Reference colour path (written independently of the package): BT.601 full-range decode + the two-point fix.
GAIN = np.array([1.072, 0.964, 1.050])
OFFSET = np.array([-17.7, 5.1, -4.3])


def decode(raw):
    """YUYV (H, W, 2) uint8 -> BGR float32, BT.601 full range."""
    y = raw[:, :, 0].astype(np.float32)
    c = raw[:, :, 1].astype(np.float32)
    u = np.repeat(c[:, 0::2], 2, axis=1) - 128.0
    v = np.repeat(c[:, 1::2], 2, axis=1) - 128.0
    return np.stack([y + 1.772 * u, y - 0.344136 * u - 0.714136 * v, y + 1.402 * v], -1)


def to8(img):
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


from bimanual_collect import (  # noqa: E402
    episode_io,  # noqa: E402
    quest_page,
    zed_v4l2,
)
from bimanual_collect.recorder import Hub, OperatorEncoder  # noqa: E402
from bimanual_collect.zed_format import decode_command  # noqa: E402


def test_page_patches_apply_exactly_once_to_the_repo_page(tmp_path) -> None:
    web = quest_page.build_web(tmp_path / "web")
    camera_view = (web / "camera_view.js").read_text()
    client = (web / "client.js").read_text()
    index = (web / "index.html").read_text()
    assert "nativeWidth" in camera_view
    assert "collectHandleRecordButton" in camera_view
    assert "collectDrawRecordBadge" in camera_view
    assert client.count("home_button: window.collectHomeButton?.(inputSources)") == 1
    # Every header/placeholder coordinate moved to the scaled 640-wide grid.
    for leftover in ("this.canvas.width - 124", "this.canvas.width - 108", "this.canvas.width / 2"):
        assert leftover not in camera_view
    assert index.count('src="/collect_hd.js"') == 1
    # The repo's own files are untouched.
    assert "nativeWidth" not in (quest_page.REPO_WEB / "camera_view.js").read_text()


def test_page_patch_fails_loudly_if_the_repo_page_changes() -> None:
    original = (quest_page.REPO_WEB / "camera_view.js").read_text()
    with pytest.raises(quest_page.PatchError, match="native-resolution canvas"):
        quest_page.patch_text("camera_view.js", original.replace("drawFrame(bitmap) {", "drawFrame(image) {"))


def test_v4l2_structs_match_the_x86_64_kernel_abi() -> None:
    assert ctypes.sizeof(zed_v4l2.Buffer) == 88
    assert ctypes.sizeof(zed_v4l2.RequestBuffers) == 20
    assert zed_v4l2.VIDIOC_REQBUFS == 0xC0145608
    assert zed_v4l2.VIDIOC_QUERYBUF == 0xC0585609
    assert zed_v4l2.VIDIOC_QBUF == 0xC058560F
    assert zed_v4l2.VIDIOC_DQBUF == 0xC0585611
    assert zed_v4l2.VIDIOC_STREAMON == 0x40045612


def random_frame(width: int, height: int, sequence: int, rng) -> zed_v4l2.Frame:
    now = time.perf_counter_ns()
    data = rng.integers(0, 256, size=width * height * 2, dtype=np.uint8).tobytes()
    return zed_v4l2.Frame(data, width, height, sequence, now - 50_000_000, now, time.time_ns())


def realistic_frame(width: int, height: int) -> zed_v4l2.Frame:
    """YUYV made from a valid colour image (full-range BT.601), like a real scene: no impossible colours."""
    rng = np.random.default_rng(1)
    small = rng.uniform(0, 255, size=(height // 8 + 1, width // 8 + 1, 3))
    rgb = np.kron(small, np.ones((8, 8, 1)))[:height, :width]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 128 + (b - y) / 1.772
    v = 128 + (r - y) / 1.402
    yuyv = np.empty((height, width, 2), np.uint8)
    yuyv[..., 0] = np.clip(np.round(y), 0, 255)
    yuyv[:, 0::2, 1] = np.clip(np.round((u[:, 0::2] + u[:, 1::2]) / 2), 0, 255)
    yuyv[:, 1::2, 1] = np.clip(np.round((v[:, 0::2] + v[:, 1::2]) / 2), 0, 255)
    now = time.perf_counter_ns()
    return zed_v4l2.Frame(yuyv.tobytes(), width, height, 0, now, now, time.time_ns())


def test_operator_view_is_native_left_eye_and_matches_reference_colour() -> None:
    # Within 4 levels of the reference decode above for real colours. (Impossible colours from random chroma can
    # differ by 5: the headset path clips before the colour fix. Display only; recordings are raw.)
    width, height = 4416, 1242
    frame = realistic_frame(width, height)
    encoder = OperatorEncoder(width, height, jpeg_quality=95, colour_fix=True)
    assert (encoder.crop_width, encoder.height, encoder.x0) == (1656, 1242, 276)
    got = encoder.bgr(frame)
    raw = np.frombuffer(frame.data, np.uint8).reshape(height, width, 2)
    reference = to8(decode(raw[:, 276 : 276 + 1656]) * GAIN + OFFSET)
    difference = np.abs(got.astype(int) - reference.astype(int))
    assert got.shape == (1242, 1656, 3)
    assert difference.max() <= 4
    assert difference.mean() < 1.0


class FakeCamera:
    """Stands in for ZedV4L2: VGA-sized random frames at 30 fps, with sequence 5 lost by the 'camera'."""

    device = "fake"

    def __init__(self) -> None:
        self.width, self.height, self.fps = 1344, 376, 30
        self.dropped_error = 0
        self.dropped_short = 0
        self.sent: dict[int, bytes] = {}
        self._sequence = 0
        self._rng = np.random.default_rng(7)
        self._lock = threading.Lock()

    def open(self) -> None:
        pass

    def read(self, timeout_s: float = 2.0) -> zed_v4l2.Frame:
        time.sleep(1 / self.fps)
        with self._lock:
            if self._sequence == 5:
                self._sequence += 1
                self.dropped_error += 1
            frame = random_frame(self.width, self.height, self._sequence, self._rng)
            self.sent[frame.sequence] = frame.data
            self._sequence += 1
            return frame

    def close(self) -> None:
        pass


def test_episode_recording_is_bit_exact_and_logs_every_frame(tmp_path) -> None:
    camera = FakeCamera()
    hub = Hub(data_root=tmp_path, camera=camera)
    hub.start()
    try:
        hub.start_episode("unit test", operator="tester")
        time.sleep(0.8)
        status = hub.stop_episode(outcome="success", note="fine")
    finally:
        hub.stop()
    summary = status["last_episode"]
    episode = Path(summary["directory"])
    assert summary["status"] == "complete" and summary["stop_reason"] == "operator"
    assert summary["encoder_exit_code"] == 0
    assert summary["recorder_queue_drops"] == 0
    meta = json.loads((episode / "episode.json").read_text())
    assert meta["task"] == "unit test" and meta["format"] == 2
    assert meta["labels"] == {"task": "unit test", "operator": "tester", "outcome": "success", "note": "fine"}
    cam = episode_io.primary(episode)
    assert cam.video == episode / "cameras" / "zed" / "video.mkv"

    rows = cam.rows()
    assert len(rows) == summary["frames_recorded"] > 10
    sequences = [int(row["camera_sequence"]) for row in rows]
    assert sequences == sorted(sequences)
    assert summary["camera_frames_missing"] == (sequences[-1] - sequences[0] + 1) - len(sequences)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,pix_fmt,color_range", "-of", "csv=p=0",
         str(cam.video)], capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert probe == "ffv1,yuv422p,pc"
    decoded = subprocess.run(decode_command(cam.video), capture_output=True, check=True).stdout
    size = camera.width * camera.height * 2
    assert len(decoded) == size * len(rows)
    for index, sequence in enumerate(sequences):
        assert decoded[index * size : (index + 1) * size] == camera.sent[sequence], f"frame {index} differs"
