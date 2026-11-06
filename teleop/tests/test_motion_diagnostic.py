import numpy as np
import pytest

import scripts.run_six_axis_diagnostic as diagnostic_runner
from scripts.run_six_axis_diagnostic import execute_plan
from widowxai_quest_teleop.config import load_config
from widowxai_quest_teleop.decoupled_ik import DecoupledIK
from widowxai_quest_teleop.hardware import CommandGate, HardwareState
from widowxai_quest_teleop.motion_diagnostic import (
    DIAGNOSTIC_MOTIONS,
    DiagnosticPlan,
    DiagnosticPoint,
    build_six_axis_plan,
    minimum_snap_fraction,
    validate_plan_dynamics,
)


def make_plan(model):
    config = load_config("configs/six_axis_arm_diagnostic.yaml")
    solver = DecoupledIK.from_config(model, config)
    limits = model.joint_limits.copy()
    limits[:, 0] += config["hardware"]["joint_limit_margin_rad"]
    limits[:, 1] -= config["hardware"]["joint_limit_margin_rad"]
    solver.set_joint_limits(limits)
    diagnostic = config["diagnostic"]
    plan = build_six_axis_plan(
        model,
        solver,
        np.asarray(config["model"]["simulation_start_q_rad"], dtype=float),
        gripper_position_m=config["hardware"]["gripper_open_m"],
        rate_hz=diagnostic["command_rate_hz"],
        segment_duration_s=diagnostic["segment_duration_s"],
        pause_duration_s=diagnostic["pause_duration_s"],
        translation_amplitude_m=diagnostic["translation_amplitude_m"],
        rotation_amplitude_rad=diagnostic["rotation_amplitude_rad"],
        maximum_endpoint_position_error_m=diagnostic[
            "maximum_endpoint_position_error_m"
        ],
        maximum_endpoint_orientation_error_rad=diagnostic[
            "maximum_endpoint_orientation_error_rad"
        ],
        maximum_ik_iterations=diagnostic["maximum_ik_iterations"],
    )
    return config, plan


def test_minimum_snap_fraction_has_stationary_endpoints() -> None:
    assert minimum_snap_fraction(0.0) == pytest.approx(0.0)
    assert minimum_snap_fraction(1.0) == pytest.approx(1.0)
    epsilon = 1e-4
    assert minimum_snap_fraction(epsilon) < 1e-12
    assert 1.0 - minimum_snap_fraction(1.0 - epsilon) < 1e-12
    with pytest.raises(ValueError, match="within"):
        minimum_snap_fraction(1.01)


def test_six_axis_plan_is_complete_smooth_and_collision_free(model) -> None:
    config, plan = make_plan(model)
    diagnostic = config["diagnostic"]
    assert {point.motion_key for point in plan.points} == {
        motion.key for motion in DIAGNOSTIC_MOTIONS
    }
    assert len(plan.endpoints) == 12
    np.testing.assert_allclose(plan.points[-1].q_command, plan.home_q, atol=1e-12)
    assert all(
        not model.in_self_collision(point.q_command, config["hardware"]["gripper_open_m"])
        for point in plan.points
    )

    dynamics = validate_plan_dynamics(
        plan,
        maximum_step_rad=np.asarray(diagnostic["maximum_joint_step_rad"]),
        maximum_velocity_rad_s=np.asarray(diagnostic["maximum_joint_velocity_rad_s"]),
        maximum_acceleration_rad_s2=np.asarray(
            diagnostic["maximum_joint_acceleration_rad_s2"]
        ),
        maximum_jerk_rad_s3=np.asarray(diagnostic["maximum_joint_jerk_rad_s3"]),
    )
    assert np.max(dynamics.max_velocity_rad_s) < 0.6
    assert np.max(dynamics.max_step_rad) < 0.007


def test_six_axis_plan_rejects_amplitudes_that_do_not_converge(model) -> None:
    config = load_config("configs/six_axis_arm_diagnostic.yaml")
    solver = DecoupledIK.from_config(model, config)
    with pytest.raises(ValueError, match="did not converge"):
        build_six_axis_plan(
            model,
            solver,
            np.asarray(config["model"]["simulation_start_q_rad"], dtype=float),
            gripper_position_m=config["hardware"]["gripper_open_m"],
            translation_amplitude_m=2.0,
            maximum_ik_iterations=2,
        )


def test_partial_diagnostic_rows_survive_a_driver_exception() -> None:
    limits = np.vstack([np.tile([-1.0, 1.0], (6, 1)), [0.0, 0.04]])
    state = HardwareState(np.zeros(6), 0.02, limits, "dry-run")
    motion = DIAGNOSTIC_MOTIONS[0]
    plan = DiagnosticPlan(
        np.zeros(6),
        (
            DiagnosticPoint(motion.key, motion.label, "first", np.zeros(6)),
            DiagnosticPoint(motion.key, motion.label, "second", np.full(6, 0.001)),
        ),
        (),
        90.0,
    )
    gate = CommandGate(
        np.zeros(6),
        0.02,
        limits,
        np.full(6, 0.01),
        joint_limit_margin_rad=0.0,
        gripper_limits_m=(0.0, 0.04),
        max_gripper_delta_m=0.01,
    )

    class FailingBackend:
        def __init__(self):
            self.calls = 0

        def send_positions(self, _q, _gripper, *, include_gripper):
            assert include_gripper is False
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated connection loss")

        def read_state(self):
            return state

    config = {
        "diagnostic": {
            "feedback_rate_hz": 20.0,
            "maximum_loop_overrun_s": 0.05,
            "maximum_tracking_error_rad": 0.08,
        }
    }
    rows: list[dict[str, object]] = []
    with pytest.raises(RuntimeError, match="connection loss"):
        execute_plan(
            FailingBackend(),
            gate,
            plan,
            config,
            0.02,
            rows,
            pace_realtime=False,
        )

    assert len(rows) == 1
    assert rows[0]["tick"] == 0
    assert float(rows[0]["send_call_duration_ms"]) >= 0.0
    assert float(rows[0]["feedback_call_duration_ms"]) >= 0.0


def test_realtime_scheduler_never_catches_up_after_a_slow_send(monkeypatch) -> None:
    class FakeClock:
        def __init__(self):
            self.now = 100.0

        def perf_counter(self):
            return self.now

        def perf_counter_ns(self):
            return round(self.now * 1e9)

        def sleep(self, duration):
            assert duration >= 0.0
            self.now += duration

    clock = FakeClock()
    monkeypatch.setattr(diagnostic_runner.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(diagnostic_runner.time, "perf_counter_ns", clock.perf_counter_ns)
    monkeypatch.setattr(diagnostic_runner.time, "sleep", clock.sleep)

    limits = np.vstack([np.tile([-1.0, 1.0], (6, 1)), [0.0, 0.04]])
    state = HardwareState(np.zeros(6), 0.02, limits, "dry-run")
    motion = DIAGNOSTIC_MOTIONS[0]
    plan = DiagnosticPlan(
        np.zeros(6),
        tuple(
            DiagnosticPoint(
                motion.key,
                motion.label,
                f"point {index}",
                np.full(6, index * 0.001),
            )
            for index in range(3)
        ),
        (),
        90.0,
    )
    gate = CommandGate(
        np.zeros(6),
        0.02,
        limits,
        np.full(6, 0.01),
        joint_limit_margin_rad=0.0,
        gripper_limits_m=(0.0, 0.04),
        max_gripper_delta_m=0.01,
    )

    class SlowBackend:
        def __init__(self):
            self.calls = 0

        def send_positions(self, _q, _gripper, *, include_gripper):
            assert include_gripper is False
            clock.now += (0.001, 0.030, 0.001)[self.calls]
            self.calls += 1

        def read_state(self):
            return state

    rows: list[dict[str, object]] = []
    execute_plan(
        SlowBackend(),
        gate,
        plan,
        {
            "diagnostic": {
                "feedback_rate_hz": 1.0,
                "maximum_loop_overrun_s": 0.05,
                "maximum_tracking_error_rad": 0.08,
            }
        },
        0.02,
        rows,
        pace_realtime=True,
    )

    send_times = np.asarray(
        [float(row["command_send_monotonic_ns"]) / 1e9 for row in rows]
    )
    assert len(rows) == 3
    assert np.all(np.diff(send_times) >= 1.0 / plan.rate_hz - 1e-9)
    assert np.diff(send_times)[1] >= 1.0 / plan.rate_hz
