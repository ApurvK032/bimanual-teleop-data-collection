import json

import numpy as np
import pytest

from widowxai_quest_teleop.cad_input import (
    DEFAULT_CAD_JOINT_NAMES,
    CadDeadmanController,
    CadFreshnessWatchdog,
    CadInputError,
    CadJointFilter,
    CadJointSample,
    CadRestMapper,
    CadSafetyError,
    parse_cad_joint_packet,
    rest_command_limits,
    wrapped_angle_delta,
)


def packet(*, sequence=1, source_time_ns=1_000_000_000, names=None, q=None):
    return {
        "seq": sequence,
        "time_ns": source_time_ns,
        "root_locked": True,
        "names": list(DEFAULT_CAD_JOINT_NAMES if names is None else names),
        "q": [0.0, 0.1, 0.2, 0.3, 0.4] if q is None else q,
    }


def sample(sequence: int, arrival_ns: int, q=None) -> CadJointSample:
    return CadJointSample(
        sequence=sequence,
        source_time_ns=1_000_000_000 + arrival_ns,
        arrival_monotonic_ns=arrival_ns,
        arrival_epoch_ns=1_000_000_000 + arrival_ns,
        q=np.zeros(5) if q is None else q,
    )


def commissioning_limits():
    return np.array(
        [
            [-0.04, 0.04],
            [0.0, 0.04],
            [0.0, 0.04],
            [-0.04, 0.04],
            [-0.04, 0.04],
            [-1e-6, 1e-6],
        ]
    )


def test_packet_parser_requires_complete_named_finite_fresh_state() -> None:
    names = list(reversed(DEFAULT_CAD_JOINT_NAMES))
    raw = packet(names=names, q=[4.0, 3.0, 2.0, 1.0, 0.0])
    parsed = parse_cad_joint_packet(
        json.dumps(raw),
        arrival_monotonic_ns=500,
        arrival_epoch_ns=1_050_000_000,
        max_packet_age_s=0.1,
    )
    np.testing.assert_array_equal(parsed.q, np.arange(5.0))
    assert parsed.root_locked

    unlocked = packet()
    unlocked["root_locked"] = False
    with pytest.raises(CadInputError, match="root is not locked"):
        parse_cad_joint_packet(
            unlocked,
            arrival_monotonic_ns=500,
            arrival_epoch_ns=1_050_000_000,
            require_root_locked=True,
        )
    malformed_lock = packet()
    malformed_lock["root_locked"] = 1
    with pytest.raises(CadInputError, match="must be boolean"):
        parse_cad_joint_packet(
            malformed_lock,
            arrival_monotonic_ns=500,
            arrival_epoch_ns=1_050_000_000,
        )

    with pytest.raises(CadInputError, match="joint set mismatch"):
        parse_cad_joint_packet(
            packet(names=list(DEFAULT_CAD_JOINT_NAMES[:-1]), q=[0.0] * 4),
            arrival_monotonic_ns=500,
            arrival_epoch_ns=1_050_000_000,
        )
    duplicated = list(DEFAULT_CAD_JOINT_NAMES)
    duplicated[-1] = duplicated[0]
    with pytest.raises(CadInputError, match="duplicate"):
        parse_cad_joint_packet(
            packet(names=duplicated),
            arrival_monotonic_ns=500,
            arrival_epoch_ns=1_050_000_000,
        )
    with pytest.raises(CadInputError, match="finite"):
        parse_cad_joint_packet(
            packet(q=[0.0, 0.0, np.nan, 0.0, 0.0]),
            arrival_monotonic_ns=500,
            arrival_epoch_ns=1_050_000_000,
        )
    with pytest.raises(CadInputError, match="stale"):
        parse_cad_joint_packet(
            packet(),
            arrival_monotonic_ns=500,
            arrival_epoch_ns=1_200_000_000,
            max_packet_age_s=0.1,
        )
    with pytest.raises(CadInputError, match="future"):
        parse_cad_joint_packet(
            packet(source_time_ns=1_200_000_000),
            arrival_monotonic_ns=500,
            arrival_epoch_ns=1_000_000_000,
            max_future_skew_s=0.05,
        )


def test_watchdog_requires_three_samples_and_recovery_after_timeout() -> None:
    watchdog = CadFreshnessWatchdog(0.1, 3, np.full(5, 0.12))
    assert not watchdog.observe(sample(0, 10_000_000), 10_000_000).fresh
    assert not watchdog.observe(sample(1, 20_000_000), 20_000_000).fresh
    assert watchdog.observe(sample(2, 30_000_000), 30_000_000).fresh

    timed_out = watchdog.poll(131_000_000)
    assert not timed_out.fresh
    assert timed_out.last_discontinuity == "timeout"
    assert timed_out.discontinuity_generation == 1

    assert not watchdog.observe(sample(3, 140_000_000), 140_000_000).fresh
    assert not watchdog.observe(sample(4, 150_000_000), 150_000_000).fresh
    assert watchdog.observe(sample(5, 160_000_000), 160_000_000).fresh


def test_watchdog_rejects_tracker_jump_and_sequence_restart() -> None:
    watchdog = CadFreshnessWatchdog(0.1, 2, np.full(5, 0.12))
    watchdog.observe(sample(10, 10_000_000), 10_000_000)
    assert watchdog.observe(sample(11, 20_000_000), 20_000_000).fresh

    jumped = watchdog.observe(
        sample(12, 30_000_000, [0.0, 0.0, 0.5, 0.0, 0.0]),
        30_000_000,
    )
    assert not jumped.fresh
    assert jumped.last_discontinuity == "source_jump_joint_2"
    assert not watchdog.observe(
        sample(13, 40_000_000, [0.0, 0.0, 0.5, 0.0, 0.0]),
        40_000_000,
    ).fresh
    assert watchdog.observe(
        sample(14, 50_000_000, [0.0, 0.0, 0.5, 0.0, 0.0]),
        50_000_000,
    ).fresh

    restarted = watchdog.observe(
        sample(0, 60_000_000, [0.0, 0.0, 0.5, 0.0, 0.0]),
        60_000_000,
    )
    assert not restarted.fresh
    assert restarted.last_discontinuity == "sequence_restart"


def test_deadman_is_relative_and_requires_release_after_stale() -> None:
    mapper = CadRestMapper(
        rest_q=np.zeros(6),
        signs=np.array([1.0, 1.0, -1.0, -1.0, -1.0]),
        scales=np.full(5, 0.1),
        command_limits=commissioning_limits(),
    )
    deadman = CadDeadmanController(mapper)
    anchor = sample(2, 30_000_000)
    target = deadman.update(
        deadman_pressed=True,
        stream_fresh=True,
        sample=anchor,
        robot_q=np.zeros(6),
    )
    np.testing.assert_array_equal(target, np.zeros(6))

    moved = sample(3, 40_000_000, [0.1, 0.0, 0.0, 0.0, 0.0])
    target = deadman.update(
        deadman_pressed=True,
        stream_fresh=True,
        sample=moved,
        robot_q=np.zeros(6),
    )
    np.testing.assert_allclose(target, [0.01, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert deadman.update(
        deadman_pressed=True,
        stream_fresh=False,
        sample=moved,
        robot_q=target,
    ) is None
    assert deadman.needs_release
    assert deadman.update(
        deadman_pressed=True,
        stream_fresh=True,
        sample=moved,
        robot_q=target,
    ) is None

    deadman.update(
        deadman_pressed=False,
        stream_fresh=True,
        sample=moved,
        robot_q=target,
    )
    reanchored = deadman.update(
        deadman_pressed=True,
        stream_fresh=True,
        sample=moved,
        robot_q=target,
    )
    np.testing.assert_array_equal(reanchored, target)
    assert mapper.reanchor_generation == 2


def test_mapper_fails_closed_outside_rest_envelope_and_keeps_j5_at_rest() -> None:
    mapper = CadRestMapper(
        rest_q=np.zeros(6),
        signs=np.ones(5),
        scales=np.ones(5),
        command_limits=commissioning_limits(),
    )
    mapper.engage(np.zeros(5), np.zeros(6))
    with pytest.raises(CadSafetyError, match="joint 1"):
        mapper.map([0.0, -0.01, 0.0, 0.0, 0.0])
    assert not np.any(mapper.map([0.0, 0.01, 0.0, 0.0, 0.0])[5:])


def test_mapper_deadband_holds_exact_rest_for_visual_jitter() -> None:
    mapper = CadRestMapper(
        rest_q=np.zeros(6),
        signs=np.ones(5),
        scales=np.full(5, 0.1),
        source_deadband_rad=np.full(5, 0.01),
        command_limits=commissioning_limits(),
    )
    mapper.engage(np.zeros(5), np.zeros(6))
    np.testing.assert_array_equal(
        mapper.map([0.009, -0.009, 0.005, -0.005, 0.0]),
        np.zeros(6),
    )
    np.testing.assert_allclose(
        mapper.map([0.02, 0.0, 0.0, 0.0, 0.0]),
        [0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_commissioning_deadbands_preserve_isolated_joint_mapping() -> None:
    deadbands = np.array([0.15, 0.02, 0.01, 0.025, 0.15])
    mapper = CadRestMapper(
        rest_q=np.zeros(6),
        signs=np.array([1.0, 1.0, -1.0, -1.0, -1.0]),
        scales=np.full(5, 0.10),
        source_deadband_rad=deadbands,
        command_limits=np.tile([-1.0, 1.0], (6, 1)),
    )
    mapper.engage(np.zeros(5), np.zeros(6))
    np.testing.assert_array_equal(mapper.map(deadbands * 0.99), np.zeros(6))

    source_directions = np.array([1.0, 1.0, -1.0, -1.0, -1.0])
    for joint in range(5):
        source = np.zeros(5)
        source[joint] = source_directions[joint] * (deadbands[joint] + 0.10)
        expected = np.zeros(6)
        expected[joint] = 0.01
        np.testing.assert_allclose(mapper.map(source), expected, atol=1e-12)


def test_isolated_commissioning_joint_keeps_every_other_joint_at_home() -> None:
    home = np.array([0.0, 1.0, 1.2, -1.0, 0.0, 0.0])
    robot_anchor = home + np.array([0.01, 0.01, 0.005, -0.01, 0.01, 0.0])
    mapper = CadRestMapper(
        rest_q=home,
        signs=np.ones(5),
        scales=np.full(5, 0.10),
        source_deadband_rad=np.zeros(5),
        commission_joint=2,
        command_limits=np.tile([-2.0, 2.0], (6, 1)),
    )
    engaged = mapper.engage(np.zeros(5), robot_anchor)
    expected_anchor = home.copy()
    expected_anchor[2] = robot_anchor[2]
    np.testing.assert_array_equal(engaged, expected_anchor)

    target = mapper.map(np.full(5, 0.20))
    expected = home.copy()
    expected[2] = robot_anchor[2] + 0.02
    np.testing.assert_allclose(target, expected, atol=1e-12)

    for invalid in (-1, 5, True, 1.5):
        with pytest.raises(CadSafetyError, match="commission joint"):
            CadRestMapper(
                rest_q=home,
                signs=np.ones(5),
                scales=np.ones(5),
                commission_joint=invalid,
                command_limits=np.tile([-2.0, 2.0], (6, 1)),
            )


def test_angle_delta_uses_shortest_path_across_pi() -> None:
    result = wrapped_angle_delta(np.array([-np.pi + 0.01]), np.array([np.pi - 0.01]))
    np.testing.assert_allclose(result, [0.02], atol=1e-12)


def test_cad_joint_filter_attenuates_jitter_and_opens_for_fast_motion() -> None:
    joint_filter = CadJointFilter(
        minimum_cutoff_hz=2.0,
        speed_coefficient=4.0,
        derivative_cutoff_hz=2.0,
        maximum_cutoff_hz=20.0,
    )
    joint_filter.update(np.zeros(5), 1_000_000_000)
    jittered = joint_filter.update(
        np.array([0.01, -0.01, 0.0, 0.0, 0.0]), 1_040_000_000
    )
    assert np.max(np.abs(jittered)) < 0.01
    stationary_alpha = joint_filter.last_alpha.copy()

    moved = joint_filter.update(
        np.array([0.13, -0.13, 0.0, 0.0, 0.0]), 1_080_000_000
    )
    assert joint_filter.last_alpha[0] > stationary_alpha[0]
    assert moved[0] > jittered[0]
    assert moved[1] < jittered[1]


def test_cad_joint_filter_wraps_and_reset_does_not_create_a_pi_jump() -> None:
    joint_filter = CadJointFilter(
        minimum_cutoff_hz=2.0,
        speed_coefficient=4.0,
        derivative_cutoff_hz=2.0,
        maximum_cutoff_hz=20.0,
    )
    first = np.full(5, np.pi - 0.01)
    second = np.full(5, -np.pi + 0.01)
    joint_filter.update(first, 1_000_000_000)
    filtered = joint_filter.update(second, 1_040_000_000)
    assert np.max(np.abs(wrapped_angle_delta(filtered, first))) < 0.02

    reset = joint_filter.reset(np.full(5, 0.5), 2_000_000_000)
    np.testing.assert_array_equal(reset, np.full(5, 0.5))
    np.testing.assert_array_equal(joint_filter.last_alpha, np.ones(5))


def test_all_zero_rest_is_valid_noncolliding_and_asymmetric(model) -> None:
    rest = np.zeros(6)
    limits = rest_command_limits(
        model.joint_limits,
        rest,
        np.array([-0.04, 0.0, 0.0, -0.04, -0.04, -1e-6]),
        np.array([0.04, 0.04, 0.04, 0.04, 0.04, 1e-6]),
    )
    np.testing.assert_array_equal(limits[1:3, 0], [0.0, 0.0])
    assert np.all((rest >= limits[:, 0]) & (rest <= limits[:, 1]))
    assert not model.in_self_collision(rest, 0.0)
