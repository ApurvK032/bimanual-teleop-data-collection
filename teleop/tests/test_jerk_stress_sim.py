from __future__ import annotations

import numpy as np

from widowxai_quest_teleop.config import load_config, parse_dual_arm_config
from widowxai_quest_teleop.dual_arm_coordinator import build_dual_arm_system
from widowxai_quest_teleop.jerk_stress_sim import (
    DelayedEncoderPlant,
    ExperimentalSoftLoadGuard,
)
from widowxai_quest_teleop.safety import TimeAlignedCommandHistory
from widowxai_quest_teleop.transport import parse_bimanual_pose_message


HOME_Q = np.array([0.0, 1.0471975512, 1.3089969390, -1.0471975512, 0.0, 0.0])


def bimanual_sample(sequence: int, *, grip: float = 0.9):
    def hand(sign: float) -> dict[str, object]:
        return {
            "tracked": True,
            "mapping_mode": "real",
            "position": [0.30, sign * 0.20, 0.25],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "grip": grip,
            "trigger": 0.0,
        }

    return parse_bimanual_pose_message(
        {
            "type": "bimanual_pose",
            "schema_version": 2,
            "sequence": sequence,
            "capture_monotonic_ms": float(sequence) * 11.0,
            "capture_epoch_ms": 1.7e12 + sequence,
            "send_monotonic_ms": float(sequence) * 11.0,
            "left": hand(1.0),
            "right": hand(-1.0),
        }
    )


def build_experiment():
    config = load_config("configs/dual_widowxai.yaml")
    arms, _, coordinator = build_dual_arm_system(
        config,
        parse_dual_arm_config(config),
        initial_q=HOME_Q,
        initial_gripper_m=0.04,
        control_gripper=True,
    )
    guard = ExperimentalSoftLoadGuard()
    coordinator.enable_simulation_jerk_stress(guard)
    return arms, coordinator, guard


def error(value: float) -> dict[str, np.ndarray]:
    joint_error = np.zeros(6)
    joint_error[3] = value
    return {"left": joint_error.copy(), "right": joint_error.copy()}


def test_guard_reconstructs_three_sample_trigger_and_release_regrip() -> None:
    guard = ExperimentalSoftLoadGuard()
    pressed = {"left": 0.9, "right": 0.9}

    assert not guard.observe(
        error(0.050), feedback_sample_fresh=True, grip_by_side=pressed
    ).active
    assert not guard.observe(
        error(0.050), feedback_sample_fresh=True, grip_by_side=pressed
    ).active
    triggered = guard.observe(
        error(0.050), feedback_sample_fresh=True, grip_by_side=pressed
    )
    assert triggered.active is True
    assert triggered.triggered is True
    assert triggered.phase == "load_yield_wait_release"

    released = guard.observe(
        error(0.0),
        feedback_sample_fresh=False,
        grip_by_side={"left": 0.0, "right": 0.0},
    )
    assert released.active is True
    assert released.phase == "load_yield_wait_regrip"
    rearmed = guard.observe(
        error(0.0), feedback_sample_fresh=False, grip_by_side=pressed
    )
    assert rearmed.active is False
    assert rearmed.rearmed is True
    assert rearmed.phase == "rearmed"


def test_delayed_encoder_plant_does_not_teleport_to_new_command() -> None:
    initial = {"left": HOME_Q.copy(), "right": HOME_Q.copy()}
    plant = DelayedEncoderPlant(
        initial,
        command_delay_s=0.035,
        response_time_constant_s=0.030,
    )
    plant.append_command(0.0, initial)
    stepped = {side: HOME_Q.copy() for side in initial}
    for value in stepped.values():
        value[3] += 0.10
    plant.append_command(0.010, stepped)

    before_delay = plant.advance(0.030, 0.010)
    np.testing.assert_allclose(before_delay["left"], HOME_Q)
    after_delay = plant.advance(0.060, 0.010)
    assert HOME_Q[3] < after_delay["left"][3] < stepped["left"][3]


def test_configured_delayed_loop_produces_meaningful_direction_reversals() -> None:
    """The actual stress plant, not a hand-authored feedback flip, oscillates."""

    sides = ("left", "right")
    initial = {side: np.zeros(6) for side in sides}
    plant = DelayedEncoderPlant(
        initial,
        command_delay_s=0.035,
        response_time_constant_s=0.030,
    )
    plant.append_command(0.0, initial)
    histories = {
        side: TimeAlignedCommandHistory(initial[side], 0.0, 0.025)
        for side in sides
    }
    guard = ExperimentalSoftLoadGuard()
    command = {side: initial[side].copy() for side in sides}
    next_feedback_s = 0.0
    last_sign = 0.0
    meaningful_reversals = 0
    trigger_time_s = None

    for index in range(80):
        timestamp_s = (index + 1) / 90.0
        measured = plant.advance(timestamp_s, 1.0 / 90.0)
        feedback_fresh = timestamp_s >= next_feedback_s
        feedback_error = {
            side: measured[side] - histories[side].reference_at(timestamp_s)
            for side in sides
        }
        if feedback_fresh:
            next_feedback_s = timestamp_s + 1.0 / 50.0
        decision = guard.observe(
            feedback_error,
            feedback_sample_fresh=feedback_fresh,
            grip_by_side={side: 0.9 for side in sides},
        )
        if decision.triggered:
            trigger_time_s = timestamp_s

        previous = command["left"][3]
        if guard.active:
            for side in sides:
                command[side] += np.clip(
                    measured[side] - command[side], -0.006, 0.006
                )
        else:
            # 1.35 rad/s is within the configured joint-4 command limit but
            # fast enough to cross 0.045 rad with the simulated lag.
            for side in sides:
                command[side][3] = min(0.24, command[side][3] + 0.015)

        plant.append_command(timestamp_s, command)
        for side in sides:
            histories[side].append(timestamp_s, command[side])

        delta = command["left"][3] - previous
        if guard.active and abs(delta) >= 1e-4:
            sign = float(np.sign(delta))
            if last_sign and sign != last_sign:
                meaningful_reversals += 1
            last_sign = sign

    assert trigger_time_s is not None
    assert trigger_time_s < 0.20
    assert meaningful_reversals >= 3


def test_dynamic_yield_can_reverse_as_delayed_measurement_moves() -> None:
    arms, coordinator, guard = build_experiment()
    for sequence in range(7):
        coordinator.step(
            bimanual_sample(sequence),
            limiter_dt=0.011,
            robot_q_source={side: HOME_Q.copy() for side in ("left", "right")},
            feedback_error_source=error(0.0),
            feedback_sample_fresh=True,
        )

    # Three consecutive fresh 0.05 rad errors activate the old guard.
    for sequence in (7, 8):
        tick = coordinator.step(
            bimanual_sample(sequence),
            limiter_dt=0.011,
            robot_q_source={side: HOME_Q.copy() for side in ("left", "right")},
            feedback_error_source=error(0.050),
            feedback_sample_fresh=True,
        )
        assert tick.control_state == "normal"

    command_before = {side: arms[side].q_command.copy() for side in ("left", "right")}
    measured_behind = {side: value.copy() for side, value in command_before.items()}
    for value in measured_behind.values():
        value[3] -= 0.060
    trigger_tick = coordinator.step(
        bimanual_sample(9),
        limiter_dt=0.011,
        robot_q_source=measured_behind,
        feedback_error_source=error(0.050),
        feedback_sample_fresh=True,
    )
    assert guard.active is True
    assert trigger_tick.control_state == "load_yield"
    for side in ("left", "right"):
        np.testing.assert_allclose(
            arms[side].q_command[3] - command_before[side][3],
            -0.006,
            atol=1e-12,
        )

    # The destination is not latched. If delayed feedback crosses the command,
    # the old controller immediately reverses direction, reproducing the flaw.
    command_after_first_yield = {
        side: arms[side].q_command.copy() for side in ("left", "right")
    }
    measured_ahead = {
        side: value.copy() for side, value in command_after_first_yield.items()
    }
    for value in measured_ahead.values():
        value[3] += 0.060
    reversed_tick = coordinator.step(
        bimanual_sample(10),
        limiter_dt=0.011,
        robot_q_source=measured_ahead,
        feedback_error_source=error(0.050),
        feedback_sample_fresh=True,
    )
    assert reversed_tick.control_state == "load_yield"
    for side in ("left", "right"):
        np.testing.assert_allclose(arms[side].feedforward_velocity, np.zeros(6))
        assert "experimental_dynamic_load_yield" in reversed_tick.proposals[
            side
        ].limiter_flags
        np.testing.assert_allclose(
            arms[side].q_command[3] - command_after_first_yield[side][3],
            0.006,
            atol=1e-12,
        )


def test_normal_dual_builder_never_enables_rejected_controller() -> None:
    config = load_config("configs/dual_widowxai.yaml")
    _, _, coordinator = build_dual_arm_system(
        config,
        parse_dual_arm_config(config),
        initial_q=HOME_Q,
        initial_gripper_m=0.04,
    )

    assert coordinator._experimental_load_guard is None
    assert coordinator.experimental_load_decision is None
