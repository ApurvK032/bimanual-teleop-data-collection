from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_dual_hardware as dual_hardware
from widowxai_quest_teleop.safety import TimeAlignedCommandHistory


def test_dual_hardware_launcher_runs_directly_with_clean_pythonpath() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "scripts/run_dual_hardware.py", "--help"],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Fail-closed bimanual WidowXAI demo" in result.stdout


class FeedbackBackend:
    def __init__(self, q_arm: np.ndarray, gripper_position_m: float = 0.03) -> None:
        self.reads = 0
        self.feedback = SimpleNamespace(
            q_arm=np.asarray(q_arm, dtype=float),
            gripper_position_m=gripper_position_m,
        )

    def read_state(self):
        self.reads += 1
        return self.feedback


def channel(side: str, q_feedback: np.ndarray | None = None) -> dual_hardware.ArmChannel:
    q = np.zeros(6) if q_feedback is None else np.asarray(q_feedback, dtype=float)
    result = dual_hardware.ArmChannel(side, FeedbackBackend(q))
    result.command_history = TimeAlignedCommandHistory(np.zeros(6), 9.0, 0.0)
    result.q_feedback_reference = np.zeros(6)
    result.q_feedback_error = np.zeros(6)
    return result


def test_feedback_telemetry_uses_actual_encoder_read_time(monkeypatch) -> None:
    left = channel("left", np.full(6, 0.026))
    left.command_history = TimeAlignedCommandHistory(np.zeros(6), 9.95, 0.025)
    left.command_history.append(9.98, np.full(6, 0.03))
    arm = SimpleNamespace(q_feedback=np.zeros(6), gripper_feedback_m=0.0)
    monkeypatch.setattr(dual_hardware.time, "perf_counter_ns", lambda: 10_000_000_000)

    fault = dual_hardware.read_and_validate_channel_feedback(
        left,
        arm,
        {"max_feedback_error_rad": 0.08},
        feedback_period=0.02,
    )

    assert fault is None
    assert left.feedback_read_monotonic_ns == 10_000_000_000
    assert left.feedback_sample_fresh is True
    assert left.next_feedback_s == pytest.approx(10.02)
    assert left.feedback_reference_state == "interpolated"
    assert left.feedback_newest_command_age_ms == pytest.approx(20.0)
    assert left.feedback_history_span_ms == pytest.approx(30.0)
    np.testing.assert_allclose(left.q_feedback_reference, np.full(6, 0.025))
    np.testing.assert_allclose(left.q_feedback_error, np.full(6, 0.001))
    np.testing.assert_allclose(arm.q_feedback, np.full(6, 0.026))


def test_no_encoder_read_is_not_reported_as_fresh() -> None:
    channels = {side: channel(side) for side in ("left", "right")}
    for arm_channel in channels.values():
        arm_channel.feedback_read_monotonic_ns = 123
        arm_channel.feedback_sample_fresh = True

    dual_hardware.reset_feedback_telemetry(channels)
    record = dual_hardware.add_feedback_telemetry({}, channels)

    for side in ("left", "right"):
        assert record[f"{side}_feedback_read_monotonic_ns"] == 0
        assert record[f"{side}_feedback_sample_fresh"] is False
        assert record[f"{side}_feedback_reference_state"] == ""
        assert record[f"{side}_feedback_newest_command_age_ms"] is None
        assert record[f"{side}_feedback_history_span_ms"] is None


def test_feedback_scheduler_marks_only_the_arm_actually_read(monkeypatch) -> None:
    channels = {side: channel(side) for side in ("left", "right")}
    arms = {
        side: SimpleNamespace(q_feedback=np.zeros(6), gripper_feedback_m=0.0)
        for side in ("left", "right")
    }
    channels["left"].next_feedback_s = 11.0
    channels["right"].next_feedback_s = 9.0
    monkeypatch.setattr(dual_hardware.time, "perf_counter", lambda: 10.0)
    monkeypatch.setattr(dual_hardware.time, "perf_counter_ns", lambda: 10_000_000_000)

    dual_hardware.reset_feedback_telemetry(channels)
    fault = dual_hardware.read_due_feedback(
        channels,
        arms,
        {"max_feedback_error_rad": 0.08},
        feedback_period=0.02,
    )

    assert fault is None
    assert channels["left"].backend.reads == 0
    assert channels["left"].feedback_sample_fresh is False
    assert channels["right"].backend.reads == 1
    assert channels["right"].feedback_sample_fresh is True


def test_idle_row_does_not_reuse_previous_command_send_times() -> None:
    channels = {side: channel(side) for side in ("left", "right")}
    channels["left"].send_monotonic_ns = 101
    channels["left"].send_epoch_ns = 201
    channels["right"].send_monotonic_ns = 102
    channels["right"].send_epoch_ns = 202

    idle_monotonic, idle_epoch = dual_hardware.command_send_telemetry(
        channels, command_sent=False
    )
    sent_monotonic, sent_epoch = dual_hardware.command_send_telemetry(
        channels, command_sent=True
    )

    assert idle_monotonic == {"left": 0, "right": 0}
    assert idle_epoch == {"left": 0, "right": 0}
    assert sent_monotonic == {"left": 101, "right": 102}
    assert sent_epoch == {"left": 201, "right": 202}


class SendingBackend:
    def __init__(self) -> None:
        self.commands = 0

    def send_positions(self, *_args, **_kwargs) -> None:
        self.commands += 1


def test_each_driver_send_duration_and_skew_are_measured(monkeypatch) -> None:
    channels = {
        side: dual_hardware.ArmChannel(side, SendingBackend())
        for side in ("left", "right")
    }
    for arm_channel in channels.values():
        arm_channel.command_history = TimeAlignedCommandHistory(np.zeros(6), 0.5, 0.0)
    arms = {
        side: SimpleNamespace(
            q_command=np.zeros(6),
            gripper_command_m=0.04,
            feedforward_filter=None,
            feedforward_velocity=np.zeros(6),
        )
        for side in ("left", "right")
    }
    clock = iter(
        [
            1_000_000_000,
            1_001_000_000,
            1_002_000_000,
            1_014_000_000,
        ]
    )
    monkeypatch.setattr(dual_hardware.time, "perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(dual_hardware.time, "time_ns", lambda: 2_000_000_000)

    skew_s = dual_hardware.send_dual_commands(
        channels, arms, control_gripper=True
    )
    durations = dual_hardware.command_send_duration_telemetry(
        channels, command_sent=True
    )

    assert skew_s == pytest.approx(0.013)
    assert durations == pytest.approx({"left": 1.0, "right": 12.0})
    assert skew_s > 0.010


def test_tracking_fault_sample_is_retained_for_the_stop_row(monkeypatch) -> None:
    measured = np.zeros(6)
    measured[2] = 0.101761
    channels = {
        "left": channel("left", measured),
        "right": channel("right"),
    }
    arm = SimpleNamespace(q_feedback=np.zeros(6), gripper_feedback_m=0.0)
    monkeypatch.setattr(dual_hardware.time, "perf_counter_ns", lambda: 10_000_000_000)

    fault = dual_hardware.read_and_validate_channel_feedback(
        channels["left"],
        arm,
        {"max_feedback_error_rad": 0.08},
        feedback_period=0.02,
    )
    record = dual_hardware.add_feedback_telemetry({}, channels, fault=fault)

    assert fault is not None
    assert "joint 2 time-aligned tracking error" in record["fault_reason"]
    assert record["left_feedback_sample_fresh"] is True
    assert record["left_feedback_read_monotonic_ns"] == 10_000_000_000
    assert record["left_feedback_reference_state"] == "clamped-to-newest"
    assert record["left_feedback_newest_command_age_ms"] == pytest.approx(1000.0)
    np.testing.assert_allclose(record["left_q_feedback_reference"], np.zeros(6))
    np.testing.assert_allclose(record["left_q_feedback_error"], measured)
    # Validation stops at the first fault, so the untouched arm must not claim
    # that an encoder read happened in the same row.
    assert record["right_feedback_sample_fresh"] is False
    assert record["right_feedback_read_monotonic_ns"] == 0
