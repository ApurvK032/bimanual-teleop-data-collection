"""sync_check: reads one-arm telemetry correctly and recovers a known camera-vs-arm time shift (sign included).

    .venv/bin/python -m pytest -q tests
"""
from __future__ import annotations

import numpy as np
from widowxai_quest_teleop.telemetry import TELEMETRY_COLUMNS, TelemetryLogger  # noqa: E402

from bimanual_collect.sync_check import arm_motion, correlation_curve, load_arms, peak  # noqa: E402

T0 = 5_000_000_000_000
TICK = 11_111_111  # 90 Hz


def wiggle(t_ns: np.ndarray) -> np.ndarray:
    """Smooth, irregular joint motion (rad), 6 joints."""
    s = (np.asarray(t_ns, np.float64) - T0) / 1e9
    return np.stack([0.3 * np.sin(1.3 * s * (j + 1) + j) * np.sin(0.37 * s + 0.5 * j) for j in range(6)], -1)


def test_one_arm_telemetry_keeps_only_fresh_encoder_reads(tmp_path) -> None:
    logger = TelemetryLogger("one-arm", {"test": True}, tmp_path, columns=TELEMETRY_COLUMNS, strict_columns=True)
    with logger:
        last = T0
        for tick, t in enumerate(range(T0, T0 + 2_000_000_000, TICK)):
            fresh = tick % 2 == 0
            last = t if fresh else last
            logger.log(pc_monotonic_ns=t, pc_epoch_ns=t, quest_sequence=tick, ik_status="ok",
                       q_feedback=wiggle(np.array([last]))[0], gripper_feedback_m=0.02,
                       q_feedback_error=[0.001] * 6, feedback_sample_fresh=fresh,
                       feedback_read_monotonic_ns=t if fresh else 0)
    arms, stats = load_arms(logger.csv_path)
    t, values = arms["arm"]
    assert t.size == (stats["rows"] + 1) // 2  # held ticks skipped
    np.testing.assert_allclose(values[:, :6], wiggle(t), atol=1e-6)  # each value at its own read time
    assert stats["ik_failures"] == 0 and stats["quest_sequence_skipped"] == 0
    assert abs(stats["loop_rate_hz"] - 90.0) < 0.1


def test_known_shift_is_recovered_with_its_sign() -> None:
    # Arm read at 45 Hz; the camera (15 fps) shows the arm as it was 30 ms AFTER each exposure timestamp.
    arm_t = np.arange(T0, T0 + 40_000_000_000, 22_222_222, dtype=np.int64)
    arms = {"arm": (arm_t, np.hstack([wiggle(arm_t), np.zeros((arm_t.size, 1))]))}
    exposure = np.arange(T0 + 1_000_000_000, T0 + 39_000_000_000, 66_666_667, dtype=np.int64)
    motion = arm_motion(arms, exposure[:-1] + 30_000_000, exposure[1:] + 30_000_000)
    motion += np.random.default_rng(0).normal(0, 0.05 * motion.std(), motion.size)  # image noise
    pairs = np.arange(10, exposure.size - 10)
    shifts_ms = np.arange(-200, 201, 1.0)
    best, r = peak(shifts_ms, correlation_curve(motion, arms, exposure, pairs, (shifts_ms * 1e6).astype(np.int64)))
    assert abs(best - 30.0) <= 3.0
    assert r > 0.9
