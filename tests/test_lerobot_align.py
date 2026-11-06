"""Alignment rules for the LeRobot export, on synthetic data written with the teleop repo's own TelemetryLogger.

    .venv/bin/python -m pytest -q tests
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from widowxai_quest_teleop.telemetry import DUAL_ARM_TELEMETRY_COLUMNS, TELEMETRY_COLUMNS, TelemetryLogger  # noqa: E402

from bimanual_collect import export_lerobot  # noqa: E402  (imports LeRobot only inside main)
from bimanual_collect.lerobot_align import STATE_NAMES, AlignmentError, align_episode, find_telemetry  # noqa: E402

T0 = 5_000_000_000_000  # an arbitrary CLOCK_MONOTONIC time
EPOCH_OFFSET = 1_700_000_000_000_000_000
TICK = 11_111_111  # 90 Hz


def q_at(t_ns: int, side: str) -> np.ndarray:
    """A known linear motion: joint i of the left arm moves at (i+1) mrad/s, the right arm at -(i+1)."""
    sign = 1.0 if side == "left" else -1.0
    return sign * (np.arange(6) + 1) * 1e-3 * (t_ns - T0) / 1e9


def write_telemetry(root: Path, start_ns: int, stop_ns: int, *, gap=None, epoch_offset=EPOCH_OFFSET,
                    read_every: int = 1) -> Path:
    """read_every > 1: like the hardware launcher, the encoders are read only every Nth tick; the ticks between
    repeat the last read with read time 0 and feedback_sample_fresh = False."""
    logger = TelemetryLogger("synthetic", {"test": True}, root, columns=DUAL_ARM_TELEMETRY_COLUMNS,
                             strict_columns=True)
    with logger:
        last_read = start_ns
        for tick, t in enumerate(range(start_ns, stop_ns, TICK)):
            if gap and gap[0] <= t < gap[1]:
                continue
            fresh = tick % read_every == 0
            last_read = t if fresh else last_read
            record = {"pc_monotonic_ns": t, "pc_epoch_ns": t + epoch_offset}
            for side in ("left", "right"):
                record.update({
                    f"{side}_q_feedback": q_at(last_read, side),
                    f"{side}_gripper_feedback_m": 0.01,
                    f"{side}_feedback_sample_fresh": fresh,
                    f"{side}_feedback_read_monotonic_ns": t if fresh else 0,
                    f"{side}_q_cmd": q_at(t, side) + 0.05,  # commands lead feedback by a visible offset
                    f"{side}_gripper_cmd_m": 0.02,
                    f"{side}_command_send_monotonic_ns": t,
                })
            logger.log(**record)
    return logger.csv_path


def write_episode(directory: Path, sequences: list[int], fps: int = 15) -> Path:
    directory.mkdir(parents=True)
    period = 1_000_000_000 // fps
    with open(directory / "zed_frames.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_frame", "camera_sequence", "exposure_monotonic_ns", "arrival_monotonic_ns",
                         "arrival_epoch_ns"])
        for index, sequence in enumerate(sequences):
            exposure = T0 + 1_000_000_000 + (sequence - sequences[0]) * period
            writer.writerow([index, sequence, exposure, exposure + 53_000_000, exposure + EPOCH_OFFSET])
    start = T0 + 1_000_000_000 - 100_000_000
    stop = start + (sequences[-1] - sequences[0] + 3) * period
    (directory / "episode.json").write_text(json.dumps({
        "episode_index": 0, "task": "fold", "started_monotonic_ns": start, "stopped_monotonic_ns": stop,
        "started_epoch_ns": start + EPOCH_OFFSET, "stopped_epoch_ns": stop + EPOCH_OFFSET,
        "video": {"size": [4416, 1242], "nominal_fps": fps},
    }))
    return directory


def test_steps_follow_camera_sequence_and_mark_dropped_frames(tmp_path) -> None:
    write_telemetry(tmp_path / "runs", T0, T0 + 5_000_000_000)
    episode = write_episode(tmp_path / "ep", [100, 101, 102, 104, 105])  # 103 dropped by the camera
    steps = align_episode(episode, roots=(tmp_path / "runs",))
    assert steps.camera_sequence.tolist() == [100, 101, 102, 103, 104, 105]
    assert steps.image_repeated.tolist() == [False, False, False, True, False, False]
    assert steps.video_frame.tolist() == [0, 1, 2, 2, 3, 4]  # the dropped frame reuses the previous image
    assert steps.state.shape == steps.action.shape == (6, 14)
    assert steps.state.dtype == np.float32
    assert STATE_NAMES[:7] == [f"left_joint_{i}" for i in range(6)] + ["left_gripper"]
    assert STATE_NAMES[7:] == [f"right_joint_{i}" for i in range(6)] + ["right_gripper"]


def test_state_is_interpolated_and_action_is_the_last_command(tmp_path) -> None:
    write_telemetry(tmp_path / "runs", T0, T0 + 5_000_000_000)
    steps = align_episode(write_episode(tmp_path / "ep", list(range(10))), roots=(tmp_path / "runs",))
    for k, t in enumerate(steps.exposure_monotonic_ns):
        np.testing.assert_allclose(steps.state[k, :6], q_at(int(t), "left"), atol=2e-6)
        np.testing.assert_allclose(steps.state[k, 7:13], q_at(int(t), "right"), atol=2e-6)
        last_command_t = T0 + ((int(t) - T0) // TICK) * TICK
        np.testing.assert_allclose(steps.action[k, :6], q_at(last_command_t, "left") + 0.05, atol=2e-6)
    np.testing.assert_allclose(steps.state[:, 6], 0.01)
    np.testing.assert_allclose(steps.action[:, 13], 0.02)
    assert steps.report["max_action_age_ms"] <= TICK / 1e6


def test_held_encoder_values_between_reads_are_not_used_as_new_samples(tmp_path) -> None:
    # The hardware launcher reads the encoders on about every second tick and repeats the last read in between.
    # Stamping a repeated value with its tick time would lag the state by up to one read period.
    # Half a tick of offset puts the camera frames between encoder reads (as on the hardware).
    write_telemetry(tmp_path / "runs", T0 + TICK // 2, T0 + 5_000_000_000, read_every=2)
    steps = align_episode(write_episode(tmp_path / "ep", list(range(10))), roots=(tmp_path / "runs",))
    for k, t in enumerate(steps.exposure_monotonic_ns):
        np.testing.assert_allclose(steps.state[k, :6], q_at(int(t), "left"), atol=2e-6)
        np.testing.assert_allclose(steps.state[k, 7:13], q_at(int(t), "right"), atol=2e-6)


def test_frames_before_the_teleop_started_are_trimmed(tmp_path) -> None:
    # Episode frames start at T0+1 s; telemetry starts 0.5 s later (A pressed before the launcher ran).
    write_telemetry(tmp_path / "runs", T0 + 1_500_000_000, T0 + 5_000_000_000)
    steps = align_episode(write_episode(tmp_path / "ep", list(range(30))), roots=(tmp_path / "runs",))
    # A step needs a command sent at or before its frame (no action otherwise): frames 0..7 (+1.0 .. +1.467 s)
    # come before the first command at +1.5 s; frame 8 (+1.533 s) is the first kept.
    assert steps.report["trimmed_start_steps_without_arm_data"] == 8
    assert steps.camera_sequence[0] == 8


def test_a_telemetry_gap_mid_episode_stops_the_export(tmp_path) -> None:
    write_telemetry(tmp_path / "runs", T0, T0 + 5_000_000_000, gap=(T0 + 1_300_000_000, T0 + 1_700_000_000))
    with pytest.raises(AlignmentError, match="gap"):
        align_episode(write_episode(tmp_path / "ep", list(range(30))), roots=(tmp_path / "runs",))


def test_telemetry_is_found_by_time_and_boot(tmp_path) -> None:
    episode = write_episode(tmp_path / "ep", list(range(30)))
    meta = json.loads((episode / "episode.json").read_text())
    time.sleep(1.1)  # TelemetryLogger names run folders by the second
    other_boot = write_telemetry(tmp_path / "runs", T0, T0 + 5_000_000_000, epoch_offset=EPOCH_OFFSET + 10**12)
    time.sleep(1.1)
    elsewhere = write_telemetry(tmp_path / "runs", T0 + 9_000_000_000, T0 + 12_000_000_000)
    time.sleep(1.1)
    right = write_telemetry(tmp_path / "runs", T0, T0 + 5_000_000_000)
    assert find_telemetry(meta, (tmp_path / "runs",)) == right
    assert other_boot != right and elsewhere != right


def test_dataset_features_match_the_lab_lerobot_conventions() -> None:
    from bimanual_collect.episode_io import CameraRecord

    zed = CameraRecord("zed", Path("v.mkv"), Path("f.csv"), 4416, 1242, 15.0, True, True, info={"model": "ZED 2i"})
    outputs = export_lerobot.camera_outputs([zed], ("left", "right"), None, camera_colour=False)
    features = export_lerobot.dataset_features(outputs, STATE_NAMES)
    assert "observation.frame_repeated" in features and "observation.zed_exposure_monotonic_ns" in features
    assert features["observation.state"]["shape"] == (14,)
    assert features["action"]["names"] == STATE_NAMES
    assert features["observation.images.zed_left"] == {
        "dtype": "video", "shape": (1242, 2208, 3), "names": ["height", "width", "channels"]}
    assert "observation.images.zed_right" in features


def write_single_arm_telemetry(root: Path, start_ns: int, stop_ns: int) -> Path:
    """Like run_hardware.py: one arm, unprefixed columns."""
    logger = TelemetryLogger("single", {"test": True}, root, columns=TELEMETRY_COLUMNS)
    with logger:
        for t in range(start_ns, stop_ns, TICK):
            logger.log(pc_monotonic_ns=t, pc_epoch_ns=t + EPOCH_OFFSET, q_feedback=q_at(t, "left"),
                       gripper_feedback_m=0.03, feedback_sample_fresh=True, feedback_read_monotonic_ns=t,
                       q_cmd=q_at(t, "left") + 0.05, gripper_cmd_m=0.02, command_send_monotonic_ns=t)
    return logger.csv_path


def test_single_arm_runs_give_7_values(tmp_path) -> None:
    write_single_arm_telemetry(tmp_path / "runs", T0, T0 + 5_000_000_000)
    episode = write_episode(tmp_path / "ep", [100, 101, 102, 103])
    steps = align_episode(episode, roots=(tmp_path / "runs",))
    assert steps.names == ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
    assert steps.state.shape == steps.action.shape == (4, 7)
    np.testing.assert_allclose(steps.action[:, :6] - steps.state[:, :6], 0.05, atol=0.002)
    assert steps.report["arms"] == 1
    with pytest.raises(AlignmentError):  # asking for two arms skips the one-arm run
        align_episode(episode, roots=(tmp_path / "runs",), arms=2)


def test_search_prefers_the_matching_arm_count(tmp_path) -> None:
    write_single_arm_telemetry(tmp_path / "runs" / "a", T0, T0 + 5_000_000_000)
    write_telemetry(tmp_path / "runs" / "b", T0, T0 + 5_000_000_000)
    episode = write_episode(tmp_path / "ep", [100, 101, 102, 103])
    assert align_episode(episode, roots=(tmp_path / "runs",), arms=2).state.shape[1] == 14
    assert align_episode(episode, roots=(tmp_path / "runs",), arms=1).state.shape[1] == 7
