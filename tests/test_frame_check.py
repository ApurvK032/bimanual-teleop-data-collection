"""frame_check: a sideways-wrapped (torn) frame is found in a lossless recording, and the aligner skips it.

    .venv/bin/python -m pytest -q tests
"""
from __future__ import annotations

import json
import subprocess

import cv2
import numpy as np

from bimanual_collect import frame_check  # noqa: E402
from bimanual_collect.lerobot_align import align_episode  # noqa: E402
from bimanual_collect.zed_format import encode_command  # noqa: E402
from tests.test_lerobot_align import T0, write_episode, write_telemetry  # noqa: E402

WIDTH, HEIGHT = 1344, 376  # side by side, small for the test


def scene(index: int) -> np.ndarray:
    """Static textured background + one moving dark block (an 'arm'), as YUYV (H, W, 2)."""
    rng = np.random.default_rng(1)
    luma = cv2.GaussianBlur(rng.integers(0, 255, (HEIGHT, WIDTH)).astype(np.float32), (0, 0), 3)
    luma = cv2.normalize(luma, None, 30, 220, cv2.NORM_MINMAX)
    x = 200 + 12 * index
    luma[120:220, x:x + 80] = 20
    luma[120:220, WIDTH // 2 + x - 20:WIDTH // 2 + x + 60] = 20
    yuyv = np.empty((HEIGHT, WIDTH, 2), np.uint8)
    yuyv[:, :, 0] = luma.astype(np.uint8)
    yuyv[:, :, 1] = 128
    return yuyv


def test_torn_frame_is_found_and_moving_content_is_not(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(frame_check, "busy", lambda: None)
    episode = tmp_path / "episode_0000"
    episode.mkdir()
    (episode / "episode.json").write_text(json.dumps({"video": {"size": [WIDTH, HEIGHT], "nominal_fps": 15}}))
    frames = [scene(i) for i in range(12)]
    frames[5] = np.roll(frames[5], 300 * 2, axis=1).reshape(HEIGHT, WIDTH, 2)  # data shifted: picture wraps
    encoder = subprocess.Popen(encode_command(episode / "zed_sbs.mkv", WIDTH, HEIGHT, 15, threads=2),
                               stdin=subprocess.PIPE)
    for frame in frames:
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    assert encoder.wait() == 0
    assert frame_check.ensure(episode) == [5]
    result = json.loads((episode / "frame_check.json").read_text())
    assert result["view_moved_at"] == [] and result["frames"] == 12


def test_aligner_replaces_a_torn_frame_with_the_previous_image(tmp_path) -> None:
    write_telemetry(tmp_path / "runs", T0, T0 + 5_000_000_000)
    episode = write_episode(tmp_path / "ep", list(range(10)))
    (episode / "frame_check.json").write_text(json.dumps({"torn_video_frames": [3]}))
    steps = align_episode(episode, roots=(tmp_path / "runs",))
    assert steps.camera_sequence.tolist() == list(range(10))
    assert steps.image_repeated.tolist() == [i == 3 for i in range(10)]
    assert steps.video_frame.tolist() == [0, 1, 2, 2, 4, 5, 6, 7, 8, 9]
    assert steps.report["torn_frames_replaced"] == [3]
