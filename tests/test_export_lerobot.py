"""End-to-end export through LeRobot's writer, then load back with its LeRobotDataset.

Runs only in the LeRobot env (skipped elsewhere):
    .venv-lerobot/bin/python -m pytest -q tests/test_export_lerobot.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")

ROOT = Path(__file__).resolve().parents[1]

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from bimanual_collect.zed_format import EyeDecoder, encode_command  # noqa: E402

W, H, FPS = 1344, 376, 15  # VGA side by side keeps the test small
T0 = 7_000_000_000_000
OFFSET = 1_700_000_000_000_000_000


def make_episode(root: Path, sequences: list[int]) -> tuple[Path, list[np.ndarray]]:
    episode = root / "raw" / "session" / "episode_0000"
    episode.mkdir(parents=True)
    # Smooth, camera-like content (gradients), each frame 12 levels brighter so frames are told apart.
    y, x = np.mgrid[0:H, 0:W].astype(float)
    base = np.empty((H, W, 2))
    base[..., 0] = 50 + 100 * x / W + 15 * np.sin(y / 25)
    base[..., 1] = 128 + 25 * np.sin(x / 60) * np.where(np.arange(W) % 2 == 0, 1, -1)
    frames = []
    for i in range(len(sequences)):
        frame = base.copy()
        frame[..., 0] += 12 * i
        frames.append(np.clip(np.round(frame), 0, 255).astype(np.uint8))
    process = subprocess.Popen(encode_command(episode / "zed_sbs.mkv", W, H, FPS, threads=2), stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    assert process.wait() == 0
    period = 1_000_000_000 // FPS
    with open(episode / "zed_frames.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_frame", "camera_sequence", "exposure_monotonic_ns", "arrival_monotonic_ns",
                         "arrival_epoch_ns"])
        for index, sequence in enumerate(sequences):
            exposure = T0 + (sequence - sequences[0]) * period
            writer.writerow([index, sequence, exposure, exposure + 50_000_000, exposure + OFFSET])
    start, stop = T0 - 50_000_000, T0 + (sequences[-1] - sequences[0] + 2) * period
    (episode / "episode.json").write_text(json.dumps({
        "episode_index": 0, "task": "fold the towel", "started_monotonic_ns": start, "stopped_monotonic_ns": stop,
        "started_epoch_ns": start + OFFSET, "stopped_epoch_ns": stop + OFFSET,
        "video": {"size": [W, H], "nominal_fps": FPS},
    }))
    return episode, frames


def make_telemetry(root: Path) -> None:
    run = root / "runs" / "day" / "run"
    run.mkdir(parents=True)
    columns = ["pc_monotonic_ns", "pc_epoch_ns"] + [
        f"{side}_{name}" for side in ("left", "right")
        for name in ("q_feedback", "gripper_feedback_m", "q_cmd", "gripper_cmd_m")
    ]
    with open(run / "telemetry.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for t in range(T0 - 200_000_000, T0 + 3_000_000_000, 11_111_111):
            row = [t, t + OFFSET]
            for sign in (1.0, -1.0):
                q = [round(sign * 0.1 * (j + 1) * (t - T0) / 1e9, 6) for j in range(6)]
                row += [json.dumps(q), 0.01, json.dumps([v + 0.05 for v in q]), 0.02]
            writer.writerow(row)


def test_export_writes_a_lerobot_dataset_that_loads_back(tmp_path) -> None:
    episode, frames = make_episode(tmp_path, [0, 1, 2, 4, 5, 6, 7, 8])  # camera dropped sequence 3
    make_telemetry(tmp_path)
    subprocess.run(
        [sys.executable, "-m", "bimanual_collect.export_lerobot", str(episode.parent), "--repo-id", "unit",
         "--root", str(tmp_path / "lerobot"), "--runs-root", str(tmp_path / "runs"), "--crf", "10", "--all"],
        check=True, capture_output=True, cwd=ROOT,
    )
    root = tmp_path / "lerobot" / "unit"
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == FPS and info["total_frames"] == 9 and info["robot_type"] == "widowxai_bimanual_quest"
    assert info["features"]["observation.images.zed_left"]["shape"] == [H, W // 2, 3]

    dataset = LeRobotDataset("unit", root=root)
    assert len(dataset) == 9 and dataset[0]["task"] == "fold the towel"
    repeated = [int(dataset[i]["observation.frame_repeated"]) for i in range(9)]
    assert repeated == [0, 0, 0, 1, 0, 0, 0, 0, 0]
    state = np.stack([dataset[i]["observation.state"].numpy() for i in range(9)])
    action = np.stack([dataset[i]["action"].numpy() for i in range(9)])
    np.testing.assert_allclose(action[:, :6] - state[:, :6], 0.05, atol=0.002)  # commands lead by +0.05
    assert state[5, 0] > state[1, 0] and state[5, 7] < state[1, 7]  # left joints rise, right fall

    decoder = EyeDecoder(colour_fix=True)

    def psnr(step: int, frame: int) -> float:
        got = dataset[step]["observation.images.zed_left"].numpy().transpose(1, 2, 0) * 255
        expected = decoder.eyes_rgb(frames[frame])[0].astype(float)
        return 10 * np.log10(255**2 / ((got - expected) ** 2).mean())

    assert psnr(0, 0) > 38  # near-lossless at CRF 10
    assert psnr(3, 2) > 38 and psnr(3, 2) > psnr(3, 3) + 6  # the dropped step shows frame 2 again, not frame 3
    assert psnr(4, 3) > 38  # and the stream picks up again with the next real frame
    manifest = json.loads((tmp_path / "lerobot" / "unit.export.json").read_text())
    assert manifest["episodes"][0]["image_repeated_steps"] == 1 and manifest["video"]["crf"] == 10


def test_export_at_policy_size_left_eye_only(tmp_path) -> None:
    import cv2

    episode, frames = make_episode(tmp_path, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    make_telemetry(tmp_path)
    subprocess.run(
        [sys.executable, "-m", "bimanual_collect.export_lerobot", str(episode.parent), "--repo-id", "small",
         "--root", str(tmp_path / "lerobot"), "--runs-root", str(tmp_path / "runs"), "--crf", "10",
         "--image-size", "336x188", "--eyes", "left", "--all"],
        check=True, capture_output=True, cwd=ROOT,
    )
    info = json.loads((tmp_path / "lerobot" / "small" / "meta" / "info.json").read_text())
    assert info["features"]["observation.images.zed_left"]["shape"] == [188, 336, 3]
    assert "observation.images.zed_right" not in info["features"]
    manifest = json.loads((tmp_path / "lerobot" / "small.export.json").read_text())
    assert manifest["image_size_per_eye"] == [336, 188] and manifest["capture_size_per_eye"] == [W // 2, H]
    dataset = LeRobotDataset("small", root=tmp_path / "lerobot" / "small")
    got = dataset[4]["observation.images.zed_left"].numpy().transpose(1, 2, 0) * 255
    expected = cv2.resize(EyeDecoder(colour_fix=True).eyes_rgb(frames[4])[0], (336, 188), interpolation=cv2.INTER_AREA)
    assert 10 * np.log10(255**2 / ((got - expected.astype(float)) ** 2).mean()) > 36


def test_export_two_episodes_into_one_dataset(tmp_path) -> None:
    """Statistics are merged across episodes; every feature must pass LeRobot's checks there too."""
    import shutil

    episode, _ = make_episode(tmp_path, [0, 1, 2, 4, 5, 6, 7, 8])
    shutil.copytree(episode, episode.parent / "episode_0001")
    make_telemetry(tmp_path)
    subprocess.run(
        [sys.executable, "-m", "bimanual_collect.export_lerobot", str(episode.parent), "--repo-id", "two",
         "--root", str(tmp_path / "lerobot"), "--runs-root", str(tmp_path / "runs"), "--crf", "30",
         "--image-size", "336x188", "--all"],
        check=True, capture_output=True, cwd=ROOT,
    )
    info = json.loads((tmp_path / "lerobot" / "two" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2 and info["total_frames"] == 18
    dataset = LeRobotDataset("two", root=tmp_path / "lerobot" / "two")
    assert len(dataset) == 18 and int(dataset[12]["observation.frame_repeated"]) == 1  # step 3 of episode 1


def test_export_a_single_arm_episode(tmp_path) -> None:
    episode, _ = make_episode(tmp_path, [0, 1, 2, 3, 4, 5])
    run = tmp_path / "runs" / "day" / "single"
    run.mkdir(parents=True)
    with open(run / "telemetry.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pc_monotonic_ns", "pc_epoch_ns", "q_feedback", "gripper_feedback_m", "q_cmd", "gripper_cmd_m"])
        for t in range(T0 - 200_000_000, T0 + 3_000_000_000, 11_111_111):
            q = [round(0.1 * (j + 1) * (t - T0) / 1e9, 6) for j in range(6)]
            writer.writerow([t, t + OFFSET, json.dumps(q), 0.01, json.dumps([v + 0.05 for v in q]), 0.02])
    (run / "config_snapshot.json").write_text(json.dumps({"hardware": {"robot_ip": "192.0.2.3"}}))
    subprocess.run(
        [sys.executable, "-m", "bimanual_collect.export_lerobot", str(episode.parent), "--repo-id", "one",
         "--root", str(tmp_path / "lerobot"), "--runs-root", str(tmp_path / "runs"), "--crf", "30",
         "--image-size", "336x188", "--eyes", "left", "--all"],
        check=True, capture_output=True, cwd=ROOT,
    )
    info = json.loads((tmp_path / "lerobot" / "one" / "meta" / "info.json").read_text())
    assert info["robot_type"] == "widowxai_quest"
    assert info["features"]["observation.state"]["shape"] == [7]
    assert info["features"]["action"]["names"][-1] == "gripper"
    manifest = json.loads((tmp_path / "lerobot" / "one.export.json").read_text())
    assert manifest["episodes"][0]["arms"] == {"arm": "192.0.2.3"}


def encode(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    process = subprocess.Popen(encode_command(path, width, height, fps, threads=2), stdin=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    assert process.wait() == 0


def write_frames_csv(path: Path, sequences: list[int], fps: int, start_ns: int) -> None:
    period = 1_000_000_000 // fps
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_frame", "camera_sequence", "exposure_monotonic_ns", "arrival_monotonic_ns",
                         "arrival_epoch_ns"])
        for index, sequence in enumerate(sequences):
            exposure = start_ns + (sequence - sequences[0]) * period
            writer.writerow([index, sequence, exposure, exposure + 30_000_000, exposure + OFFSET])


def test_export_two_cameras_matched_by_exposure_time(tmp_path) -> None:
    """A stereo ZED at 15 fps (primary) and a mono webcam at 30 fps with one dropped frame."""
    episode = tmp_path / "raw" / "session" / "episode_0000"
    zed_frames = [np.full((H, W, 2), 40 + 12 * i, np.uint8) for i in range(8)]
    for frame in zed_frames:
        frame[..., 1] = 128
    cam_w, cam_h = 320, 240
    web_seq = [s for s in range(16) if s != 7]  # webcam dropped its frame 7
    web_frames = [np.full((cam_h, cam_w, 2), 20 + 10 * s, np.uint8) for s in web_seq]
    for frame in web_frames:
        frame[..., 1] = 128
    encode(episode / "cameras" / "zed" / "video.mkv", zed_frames, 15)
    encode(episode / "cameras" / "wrist" / "video.mkv", web_frames, 30)
    write_frames_csv(episode / "cameras" / "zed" / "frames.csv", list(range(8)), 15, T0)
    write_frames_csv(episode / "cameras" / "wrist" / "frames.csv", web_seq, 30, T0 + 2_000_000)
    start, stop = T0 - 50_000_000, T0 + 9 * 66_666_667
    episode.joinpath("episode.json").write_text(json.dumps({
        "format": 2, "status": "complete", "stop_reason": "operator", "episode_index": 0, "task": "fold",
        "labels": {"task": "fold", "operator": "", "outcome": "success", "note": ""},
        "started_monotonic_ns": start, "stopped_monotonic_ns": stop, "started_epoch_ns": start + OFFSET,
        "stopped_epoch_ns": stop + OFFSET, "duration_s": 0.6,
        "cameras": [
            {"name": "zed", "primary": True, "stereo": True, "video": "cameras/zed/video.mkv",
             "frames": "cameras/zed/frames.csv", "size": [W, H], "nominal_fps": 15, "model": "ZED 2i"},
            {"name": "wrist", "primary": False, "stereo": False, "video": "cameras/wrist/video.mkv",
             "frames": "cameras/wrist/frames.csv", "size": [cam_w, cam_h], "nominal_fps": 30, "model": "Webcam"},
        ],
        "video": {"size": [W, H], "nominal_fps": 15}, "frames_recorded": 8, "camera_frames_missing": 0,
    }))
    make_telemetry(tmp_path)
    subprocess.run(
        [sys.executable, "-m", "bimanual_collect.export_lerobot", str(episode.parent), "--repo-id", "multi",
         "--root", str(tmp_path / "lerobot"), "--runs-root", str(tmp_path / "runs"), "--crf", "10", "--all"],
        check=True, capture_output=True, cwd=ROOT,
    )
    info = json.loads((tmp_path / "lerobot" / "multi" / "meta" / "info.json").read_text())
    features = info["features"]
    assert features["observation.images.wrist"]["shape"] == [cam_h, cam_w, 3]
    assert "observation.images.zed_left" in features and "observation.wrist_frame_repeated" in features
    dataset = LeRobotDataset("multi", root=tmp_path / "lerobot" / "multi")
    assert len(dataset) == 8
    # Step k (zed frame k at T0 + k/15 s) shows webcam sequence 2k (2 ms later); sequence 7 never arrives, but
    # it is not needed (only even sequences are nearest), so no step repeats.
    for k in range(8):
        level = dataset[k]["observation.images.wrist"].numpy().mean() * 255
        assert abs(level - (20 + 10 * 2 * k)) < 4, (k, level)
    assert sum(int(dataset[k]["observation.wrist_frame_repeated"]) for k in range(8)) == 0
