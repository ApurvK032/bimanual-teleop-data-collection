import numpy as np
import pytest

from widowxai_quest_teleop.pose_filter import ControllerPoseFilter, pose_ema, pose_filter_alphas
from widowxai_quest_teleop.sample_buffer import LatestValueMailbox
from widowxai_quest_teleop.safety import FreshSequenceWatchdog
from widowxai_quest_teleop.transport import parse_pose_message
from widowxai_quest_teleop.types import Pose
from widowxai_quest_teleop.wrist_pivot import estimate_wrist_pivot_offset


def test_latest_mailbox_overwrites_old_state() -> None:
    mailbox: LatestValueMailbox[int] = LatestValueMailbox()
    mailbox.publish(1)
    mailbox.publish(2)
    value, generation = mailbox.take_latest()
    assert value == 2
    assert generation == 2
    assert mailbox.overwrite_count == 1


def test_wait_take_latest_consumes_the_new_generation() -> None:
    mailbox: LatestValueMailbox[int] = LatestValueMailbox()
    mailbox.publish(4)
    value, generation = mailbox.wait_take_latest(0, timeout_s=0.01)
    assert (value, generation) == (4, 1)
    assert mailbox.take_latest() == (None, 1)
    assert mailbox.wait_take_latest(generation, timeout_s=0.0) == (None, generation)


def test_transport_parses_webxr_xyzw_as_internal_wxyz() -> None:
    sample = parse_pose_message(
        {
            "type": "pose",
            "sequence": 7,
            "capture_monotonic_ms": 10.0,
            "capture_epoch_ms": 20.0,
            "send_monotonic_ms": 11.0,
            "left": {
                "position": [1, 2, 3],
                "orientation_xyzw": [0, 0, 0, 1],
                "grip": 0.8,
                "trigger": 0.2,
            },
            "head": {"orientation_xyzw": [0, 0, 0, 1]},
            "reconnect_generation": 3,
        }
    )
    np.testing.assert_allclose(sample.controller_pose.quaternion_wxyz, [1, 0, 0, 0])
    assert sample.sequence == 7
    assert sample.reconnect_generation == 3
    assert sample.hand == "left"
    assert sample.mapping_mode == "real"
    np.testing.assert_allclose(sample.head_quaternion_wxyz, [1, 0, 0, 0])


def test_transport_prefers_session_locked_operator_heading() -> None:
    sample = parse_pose_message(
        {
            "type": "pose",
            "sequence": 9,
            "capture_monotonic_ms": 10.0,
            "capture_epoch_ms": 20.0,
            "send_monotonic_ms": 10.1,
            "left": {
                "position": [0, 0, 0],
                "orientation_xyzw": [0, 0, 0, 1],
                "grip": 0,
                "trigger": 0,
            },
            "head": {"orientation_xyzw": [0, 0.70710678, 0, 0.70710678]},
            "operator_head": {"orientation_xyzw": [0, 0, 0, 1]},
        }
    )

    np.testing.assert_allclose(sample.head_quaternion_wxyz, [1, 0, 0, 0])


def test_transport_selects_right_controller() -> None:
    sample = parse_pose_message(
        {
            "type": "pose",
            "sequence": 10,
            "capture_monotonic_ms": 10.0,
            "capture_epoch_ms": 20.0,
            "send_monotonic_ms": 10.1,
            "left": {
                "position": [-1, -2, -3],
                "orientation_xyzw": [0, 0, 0, 1],
                "grip": 0,
                "trigger": 0,
            },
            "right": {
                "position": [1, 2, 3],
                "orientation_xyzw": [0, 0, 0, 1],
                "grip": 0.8,
                "trigger": 0.4,
            },
        },
        hand="right",
    )

    np.testing.assert_allclose(sample.controller_pose.position, [1, 2, 3])
    assert sample.grip == 0.8
    assert sample.trigger == 0.4


def test_transport_auto_selects_and_mirrors_webxr_pose() -> None:
    half_sqrt = np.sqrt(0.5)
    sample = parse_pose_message(
        {
            "type": "pose",
            "selected_hand": "right",
            "mapping_mode": "mirror",
            "sequence": 11,
            "capture_monotonic_ms": 10.0,
            "capture_epoch_ms": 20.0,
            "send_monotonic_ms": 10.1,
            "right": {
                "position": [1, 2, 3],
                "orientation_xyzw": [0, half_sqrt, 0, half_sqrt],
                "grip": 0.7,
                "trigger": 0.3,
            },
        },
        hand=None,
        mapping_mode=None,
    )

    assert sample.hand == "right"
    assert sample.mapping_mode == "mirror"
    np.testing.assert_allclose(sample.controller_pose.position, [-1, 2, 3])
    np.testing.assert_allclose(
        sample.controller_pose.quaternion_wxyz,
        [half_sqrt, 0, -half_sqrt, 0],
    )


def test_transport_rejects_page_selection_mismatch() -> None:
    payload = {
        "type": "pose",
        "selected_hand": "right",
        "mapping_mode": "mirror",
        "right": {
            "position": [0, 0, 0],
            "orientation_xyzw": [0, 0, 0, 1],
        },
    }
    with pytest.raises(ValueError, match="expected left"):
        parse_pose_message(payload, hand="left", mapping_mode="mirror")
    with pytest.raises(ValueError, match="expected real"):
        parse_pose_message(payload, hand="right", mapping_mode="real")


def test_watchdog_requires_fresh_window_and_times_out() -> None:
    base = parse_pose_message(
        {
            "type": "pose",
            "sequence": 1,
            "capture_monotonic_ms": 0,
            "capture_epoch_ms": 0,
            "send_monotonic_ms": 0,
            "left": {"position": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1]},
            "reconnect_generation": 1,
        }
    )
    watchdog = FreshSequenceWatchdog(0.01, fresh_samples_to_recover=2)
    assert not watchdog.observe(base, now_ns=base.pc_arrival_monotonic_ns).fresh
    second = type(base)(**{**base.__dict__, "sequence": 2, "pc_arrival_monotonic_ns": base.pc_arrival_monotonic_ns + 1})
    assert watchdog.observe(second, now_ns=second.pc_arrival_monotonic_ns).fresh
    assert not watchdog.poll(second.pc_arrival_monotonic_ns + 20_000_000).fresh


def test_watchdog_rejects_a_unique_sample_delayed_in_the_mailbox() -> None:
    sample = parse_pose_message(
        {
            "type": "pose",
            "sequence": 1,
            "capture_monotonic_ms": 0,
            "capture_epoch_ms": 0,
            "send_monotonic_ms": 0,
            "left": {"position": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1]},
        }
    )
    watchdog = FreshSequenceWatchdog(0.01, fresh_samples_to_recover=1)
    consumed_20_ms_late = sample.pc_arrival_monotonic_ns + 20_000_000
    status = watchdog.observe(sample, now_ns=consumed_20_ms_late)
    assert not status.fresh
    assert status.age_s == 0.02


def test_pose_ema_matches_reference_position_and_quaternion_nlerp() -> None:
    previous = Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    current = Pose(np.ones(3), np.array([-0.8, -0.6, 0.0, 0.0]))
    filtered = pose_ema(previous, current, 0.8)
    np.testing.assert_allclose(filtered.position, np.full(3, 0.8))
    np.testing.assert_allclose(filtered.quaternion_wxyz, [0.86824314, 0.49613894, 0.0, 0.0])


def test_pose_ema_can_smooth_rotation_more_than_translation() -> None:
    previous = Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    current = Pose(np.ones(3), np.array([0.8, 0.6, 0.0, 0.0]))
    filtered = pose_ema(previous, current, 0.8, 0.5)

    np.testing.assert_allclose(filtered.position, np.full(3, 0.8))
    np.testing.assert_allclose(filtered.quaternion_wxyz, [0.9486833, 0.31622777, 0.0, 0.0])


def test_pose_filter_alphas_fall_back_to_shared_weight() -> None:
    assert pose_filter_alphas({"pose_filter_alpha": 0.8}) == (0.8, 0.8)
    assert pose_filter_alphas(
        {
            "pose_filter_alpha": 0.8,
            "pose_filter_translation_alpha": 0.7,
            "pose_filter_rotation_alpha": 0.4,
        }
    ) == (0.7, 0.4)


def test_adaptive_rotation_filter_opens_bandwidth_for_deliberate_motion() -> None:
    pose_filter = ControllerPoseFilter(
        {
            "pose_filter_alpha": 0.8,
            "adaptive_rotation_filter": {
                "enabled": True,
                "minimum_cutoff_hz": 1.5,
                "speed_coefficient": 20.0,
                "speed_exponent": 2.0,
                "derivative_cutoff_hz": 2.0,
                "maximum_cutoff_hz": 35.0,
            },
        }
    )
    origin = Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    tiny = Pose(np.ones(3), np.array([0.9999995, 0.001, 0.0, 0.0]))
    fast = Pose(np.ones(3) * 2.0, np.array([0.99500417, 0.09983342, 0.0, 0.0]))

    pose_filter.update(origin, 0.0)
    tiny_filtered = pose_filter.update(tiny, 1.0 / 90.0)
    tiny_alpha = pose_filter.rotation_alpha
    fast_filtered = pose_filter.update(fast, 2.0 / 90.0)

    assert tiny_alpha < pose_filter.rotation_alpha
    assert pose_filter.rotation_cutoff_hz <= 35.0
    np.testing.assert_allclose(tiny_filtered.position, np.full(3, 0.8))
    np.testing.assert_allclose(fast_filtered.position, np.full(3, 1.76))


def test_adaptive_rotation_filter_reset_reanchors_without_a_jump() -> None:
    pose_filter = ControllerPoseFilter(
        {
            "pose_filter_alpha": 0.8,
            "adaptive_rotation_filter": {"enabled": True},
        }
    )
    first = Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    second = Pose(np.ones(3), np.array([0.9, 0.1, 0.2, 0.3]))
    pose_filter.update(first, 0.0)
    pose_filter.reset()

    assert pose_filter.update(second, 1.0) is second


def test_wrist_pivot_calibration_recovers_controller_frame_offset() -> None:
    rng = np.random.default_rng(4)
    true_offset = np.array([-0.045, 0.012, -0.018])
    rotations = []
    positions = []
    fixed_pivot = np.array([0.4, 1.0, -0.2])
    for _ in range(100):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-0.8, 0.8)
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
        rotations.append(rotation)
        positions.append(fixed_pivot - rotation @ true_offset)
    estimated = estimate_wrist_pivot_offset(np.asarray(positions), np.asarray(rotations))
    np.testing.assert_allclose(estimated, true_offset, atol=1e-9)
