import numpy as np
import pytest

from widowxai_quest_teleop.motion_limiter import (
    AccelerationLimitedCommand,
    VelocityFeedforwardFilter,
    bounded_command_period,
    configured_command_spacing_stage,
    configured_minimum_command_interval,
    minimum_command_spacing_wait,
)


def test_velocity_feedforward_is_filtered_gained_and_capped() -> None:
    estimator = VelocityFeedforwardFilter(
        np.zeros(2),
        filter_alpha=0.5,
        gain=0.5,
        max_velocity=np.array([0.3, 2.0]),
    )

    np.testing.assert_allclose(estimator.update(np.array([0.01, 0.01]), 0.01), [0.25, 0.25])
    np.testing.assert_allclose(estimator.update(np.array([0.02, 0.02]), 0.01), [0.3, 0.375])


def test_velocity_feedforward_drops_old_direction_on_reversal_and_reset() -> None:
    estimator = VelocityFeedforwardFilter(
        np.zeros(1),
        filter_alpha=0.5,
        gain=0.5,
        max_velocity=np.ones(1),
    )
    assert estimator.update(np.array([0.01]), 0.01)[0] > 0.0
    assert estimator.update(np.array([0.0]), 0.01)[0] < 0.0
    estimator.reset(np.array([0.2]))
    np.testing.assert_array_equal(estimator.update(np.array([0.2]), 0.01), [0.0])


def test_command_period_caps_late_frames_without_stretching_early_frames() -> None:
    assert bounded_command_period(0.006, 90.0) == 0.006
    assert bounded_command_period(0.020, 90.0) == 1.0 / 90.0


def test_minimum_command_spacing_never_repays_a_late_interval() -> None:
    interval = 1.0 / 90.0
    assert minimum_command_spacing_wait(10.0, interval, 10.005) == pytest.approx(
        interval - 0.005
    )
    assert minimum_command_spacing_wait(10.0, interval, 10.050) == 0.0
    assert configured_minimum_command_interval({}) == 0.0
    assert configured_minimum_command_interval(
        {"minimum_command_interval_s": interval}
    ) == pytest.approx(interval)
    assert configured_command_spacing_stage({}) == "before_consume"
    assert configured_command_spacing_stage(
        {"command_spacing_stage": "before_send"}
    ) == "before_send"


def test_minimum_command_spacing_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        configured_minimum_command_interval({"minimum_command_interval_s": -0.1})
    with pytest.raises(ValueError, match="nonnegative"):
        minimum_command_spacing_wait(1.0, np.nan, 1.0)
    with pytest.raises(ValueError, match="spacing stage"):
        configured_command_spacing_stage({"command_spacing_stage": "somewhere"})


def test_final_send_barrier_holds_early_frames_without_catchup() -> None:
    interval = 1.0 / 90.0
    processing_s = 0.0025
    arrivals = [0.0, interval, 2.0 * interval, 0.050, 0.0505]
    sends = [arrivals[0] + processing_s]

    for arrival in arrivals[1:]:
        ready_to_send = arrival + processing_s
        wait = minimum_command_spacing_wait(sends[-1], interval, ready_to_send)
        sends.append(ready_to_send + wait)

    intervals = np.diff(sends)
    assert np.all(intervals >= interval - 1e-12)
    # A late arrival is sent immediately, but the following early arrival is
    # held relative to that actual send rather than an old absolute timeline.
    assert sends[3] == pytest.approx(arrivals[3] + processing_s)
    assert sends[4] == pytest.approx(sends[3] + interval)


def test_limiter_respects_velocity_and_acceleration_bounds() -> None:
    dt = 0.005
    limiter = AccelerationLimitedCommand(
        np.zeros(2),
        max_velocity=np.array([0.30, 0.45]),
        max_acceleration=np.array([0.75, 1.25]),
    )
    previous_velocity = np.zeros(2)
    saw_acceleration_limit = False
    for _ in range(500):
        result = limiter.step(np.ones(2), dt)
        assert np.all(np.abs(result.velocity) <= np.array([0.30, 0.45]) + 1e-12)
        assert np.all(
            np.abs(result.velocity - previous_velocity)
            <= np.array([0.75, 1.25]) * dt + 1e-12
        )
        saw_acceleration_limit = saw_acceleration_limit or bool(np.any(result.acceleration_limited))
        previous_velocity = result.velocity
    assert saw_acceleration_limit


def test_limiter_reverses_without_an_instant_velocity_flip() -> None:
    dt = 0.005
    limiter = AccelerationLimitedCommand(
        np.zeros(1),
        max_velocity=np.array([0.30]),
        max_acceleration=np.array([0.75]),
    )
    for _ in range(100):
        result = limiter.step(np.ones(1), dt)
    before = result.velocity[0]
    reversed_result = limiter.step(np.array([-1.0]), dt)
    assert before > 0.0
    assert reversed_result.velocity[0] >= before - 0.75 * dt - 1e-12
    assert reversed_result.acceleration_limited[0]


def test_reset_holds_position_and_zeroes_velocity() -> None:
    limiter = AccelerationLimitedCommand(
        np.zeros(1),
        max_velocity=np.ones(1),
        max_acceleration=np.ones(1),
    )
    limiter.step(np.ones(1), 0.01)
    limiter.reset(np.array([0.2]))
    result = limiter.step(np.array([0.2]), 0.01)
    np.testing.assert_allclose(result.command, [0.2])
    np.testing.assert_allclose(result.velocity, [0.0])
    np.testing.assert_allclose(result.acceleration, [0.0])


def test_gripper_limiter_snaps_machine_precision_residue_to_closed_endpoint() -> None:
    limiter = AccelerationLimitedCommand(
        np.array([0.040]),
        max_velocity=np.array([0.120]),
        max_acceleration=np.array([1.200]),
    )
    rng = np.random.default_rng(1)
    commands = []
    for _ in range(200):
        result = limiter.step(np.array([0.0]), rng.uniform(0.008, 0.014))
        commands.append(float(result.command[0]))

    assert min(commands) >= 0.0
    assert commands[-1] == 0.0
    np.testing.assert_array_equal(result.velocity, [0.0])
    np.testing.assert_array_equal(result.acceleration, [0.0])


def test_optional_jerk_bound_limits_acceleration_changes() -> None:
    dt = 1.0 / 90.0
    limiter = AccelerationLimitedCommand(
        np.zeros(2),
        max_velocity=np.array([2.0, 3.0]),
        max_acceleration=np.array([15.0, 30.0]),
        max_jerk=np.array([1200.0, 2400.0]),
    )
    previous_acceleration = np.zeros(2)
    saw_jerk_limit = False
    for target in [np.ones(2)] * 20 + [-np.ones(2)] * 20:
        result = limiter.step(target, dt)
        assert np.all(
            np.abs(result.acceleration - previous_acceleration)
            <= np.array([1200.0, 2400.0]) * dt + 1e-10
        )
        saw_jerk_limit = saw_jerk_limit or bool(np.any(result.jerk_limited))
        previous_acceleration = result.acceleration
    assert saw_jerk_limit


def test_jerk_limited_step_settles_without_persistent_oscillation() -> None:
    limiter = AccelerationLimitedCommand(
        np.zeros(1),
        max_velocity=np.array([2.0]),
        max_acceleration=np.array([15.0]),
        max_jerk=np.array([1200.0]),
    )
    for _ in range(500):
        result = limiter.step(np.array([0.2]), 1.0 / 90.0)
    np.testing.assert_allclose(result.command, [0.2], atol=1e-6)
    np.testing.assert_allclose(result.velocity, [0.0], atol=1e-6)
    np.testing.assert_allclose(result.acceleration, [0.0], atol=1e-6)


def test_jerk_and_velocity_bounds_hold_during_retargeting() -> None:
    max_velocity = np.array([2.0])
    max_acceleration = np.array([15.0])
    max_jerk = np.array([1200.0])
    limiter = AccelerationLimitedCommand(
        np.zeros(1),
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        max_jerk=max_jerk,
    )
    previous_acceleration = np.zeros(1)
    rng = np.random.default_rng(7)
    target = np.zeros(1)
    for index in range(10_000):
        dt = rng.uniform(0.008, 0.018)
        if index % 17 == 0:
            target = rng.uniform(-3.0, 3.0, size=1)
        result = limiter.step(target, dt)
        assert np.all(np.abs(result.velocity) <= max_velocity + 1e-10)
        assert np.all(np.abs(result.acceleration) <= max_acceleration + 1e-10)
        assert np.all(
            np.abs(result.acceleration - previous_acceleration)
            <= max_jerk * dt + 1e-8
        )
        previous_acceleration = result.acceleration
