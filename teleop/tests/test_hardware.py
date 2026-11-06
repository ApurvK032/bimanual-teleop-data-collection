import importlib.metadata
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.run_hardware import (
    make_startup_command_gate,
    open_gripper_at_home,
    ramp_to_home,
    require_quest_selection,
    resolve_demo_duration,
    return_to_rest,
    teleop_gripper_limits,
    validate_hardware_config,
    validate_live_hardware_timing,
    validate_time_aligned_feedback,
)
from widowxai_quest_teleop.config import (
    TASK_PROFILE_NAMES,
    apply_task_profile,
    load_config,
    task_profile_for_quest_selection,
)
from widowxai_quest_teleop.hardware import (
    END_EFFECTOR_PROFILE_TO_VARIANT,
    CommandGate,
    HardwareState,
    HardwareSafetyError,
    HardwareUnavailableError,
    TrossenArmBackend,
    require_compatible_versions,
)
from widowxai_quest_teleop.safety import TimeAlignedCommandHistory


def make_gate() -> CommandGate:
    limits = np.vstack([np.tile([-1.0, 1.0], (6, 1)), [0.0, 0.044]])
    return CommandGate(
        np.zeros(6),
        0.044,
        limits,
        np.full(6, 0.02),
        joint_limit_margin_rad=0.05,
        gripper_limits_m=(0.022, 0.044),
        max_gripper_delta_m=0.005,
    )


def test_command_gate_rejects_instead_of_modifying_commands() -> None:
    gate = make_gate()
    accepted = np.full(6, 0.01)
    gate.validate(accepted, 0.041)
    np.testing.assert_array_equal(gate.previous_q, accepted)
    with pytest.raises(HardwareSafetyError, match="per-tick"):
        gate.validate(np.full(6, 0.04), 0.041)
    with pytest.raises(HardwareSafetyError, match="limit margin"):
        gate.validate(np.full(6, 0.96), 0.041)
    with pytest.raises(HardwareSafetyError, match="non-finite"):
        gate.validate(np.full(6, np.nan), 0.041)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("hardware", "max_feedback_error_rad", np.nan),
        ("hardware", "feedback_tracking_delay_s", np.nan),
        ("hardware", "command_goal_time_s", np.inf),
        ("hardware", "max_gripper_delta_m", 0.0),
        ("hardware", "startup_collision_samples", 1),
        ("control", "loop_rate_hz", np.nan),
        ("quest", "stale_timeout_s", -1.0),
        ("ik", "max_dq_per_joint_rad", [0.005, 0.005, np.nan, 0.01, 0.01, 0.01]),
    ],
)
def test_hardware_config_rejects_nonfinite_or_nonpositive_safety_values(
    section, key, value
) -> None:
    config = load_config("configs/hardware_demo.yaml")
    config[section][key] = value
    with pytest.raises((HardwareSafetyError, ValueError, OverflowError)):
        validate_hardware_config(config)


def test_hardware_config_rejects_unknown_quest_input_selection() -> None:
    config = load_config("configs/hardware_demo.yaml")
    config["quest"]["hand"] = "either"
    with pytest.raises(HardwareSafetyError, match="quest.hand"):
        validate_hardware_config(config)

    config = load_config("configs/hardware_demo.yaml")
    config["quest"]["mapping_mode"] = "automatic"
    with pytest.raises(HardwareSafetyError, match="quest.mapping_mode"):
        validate_hardware_config(config)


def test_right_real_axis_validation_profile_is_conservative_and_pending() -> None:
    baseline = load_config("configs/step3_velocity_feedforward_25ms_hardware.yaml")
    config = load_config("configs/right_real_axis_validation_hardware.yaml")
    validate_hardware_config(config)
    validate_live_hardware_timing(config)

    assert config["quest"]["hand"] == "right"
    assert config["quest"]["mapping_mode"] == "real"
    assert config["quest"]["translation_scale"] < baseline["quest"]["translation_scale"]
    assert config["quest"]["rotation_scale"] < baseline["quest"]["rotation_scale"]
    assert config["hardware"]["control_gripper"] is False
    assert config["hardware"]["max_demo_duration_s"] == pytest.approx(15.0)
    assert config["control"] == baseline["control"]
    assert config["ik"] == baseline["ik"]
    assert config["hardware"]["command_goal_time_s"] == baseline["hardware"][
        "command_goal_time_s"
    ]
    assert config["hardware"]["arm_velocity_feedforward"] == baseline["hardware"][
        "arm_velocity_feedforward"
    ]

    calibration = json.loads(
        Path(config["quest"]["calibration"]).read_text(encoding="utf-8")
    )
    validation = calibration["physical_axis_validation"]
    assert validation["status"] == "pending_not_accepted"
    assert validation["stable_left_baseline_unchanged"] is True
    assert set(validation["motions"]) == {
        "right",
        "up",
        "forward",
        "screw_clockwise",
        "nod_yes_down",
        "nod_no_left",
    }
    assert all(
        motion["status"] == "pending"
        for motion in validation["motions"].values()
    )
    for key in ("position_matrix", "rotation_matrix"):
        matrix = np.asarray(calibration[key], dtype=float)
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-9)
        assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-9)


def test_right_dual_bench_axis_profile_preserves_historical_candidate() -> None:
    config = load_config("configs/right_dual_bench_axis_validation_hardware.yaml")
    dual = load_config("configs/dual_widowxai.yaml")

    validate_hardware_config(config)
    validate_live_hardware_timing(config)
    assert config["quest"]["hand"] == "right"
    assert config["quest"]["mapping_mode"] == "real"
    assert config["quest"]["calibration"].endswith("20260723_candidate.json")
    assert dual["arms"]["right"]["calibration"].endswith("20260812_accepted.json")
    assert config["hardware"]["robot_ip"] == dual["arms"]["right"]["robot_ip"]
    assert config["hardware"]["control_gripper"] is False
    assert config["quest"]["translation_scale"] == pytest.approx(0.20)
    assert config["quest"]["rotation_scale"] == pytest.approx(0.20)
    assert config["quest"]["position_reach_limit_m"] == pytest.approx(0.035)
    assert config["quest"]["rotation_reach_limit_rad"] == pytest.approx(0.08)
    assert config["hardware"]["max_demo_duration_s"] == pytest.approx(15.0)


def test_right_dual_bench_40pct_profile_enables_full_gripper_for_45_seconds() -> None:
    config = load_config("configs/right_dual_bench_40pct_full_gripper_hardware.yaml")
    previous = load_config("configs/right_dual_bench_axis_validation_hardware.yaml")
    dual = load_config("configs/dual_widowxai.yaml")

    validate_hardware_config(config)
    validate_live_hardware_timing(config)
    assert config["quest"]["calibration"] == dual["arms"]["right"]["calibration"]
    assert config["hardware"]["robot_ip"] == dual["arms"]["right"]["robot_ip"]
    assert config["quest"]["translation_scale"] == pytest.approx(0.40)
    assert config["quest"]["rotation_scale"] == pytest.approx(0.40)
    assert config["quest"]["position_reach_limit_m"] == pytest.approx(0.070)
    assert config["quest"]["rotation_reach_limit_rad"] == pytest.approx(0.16)
    assert config["hardware"]["control_gripper"] is True
    assert config["hardware"]["gripper_open_m"] == pytest.approx(0.040)
    assert config["hardware"]["gripper_min_demo_m"] == pytest.approx(0.0)
    assert config["hardware"]["max_demo_duration_s"] == pytest.approx(45.0)
    calibration = json.loads(
        Path(config["quest"]["calibration"]).read_text(encoding="utf-8")
    )
    validation = calibration["physical_validation"]
    assert validation["status"] == "accepted_by_operator_at_40pct"
    assert validation["scope"]["translation_scale"] == pytest.approx(0.40)
    assert validation["scope"]["rotation_scale"] == pytest.approx(0.40)
    assert validation["telemetry"]["ik_failures"] == 0
    assert config["control"] == previous["control"]
    assert config["ik"] == previous["ik"]


def test_right_mirror_axis_validation_profiles_are_isolated_and_pending() -> None:
    baseline = load_config("configs/step3_velocity_feedforward_25ms_hardware.yaml")
    hardware = load_config("configs/right_mirror_axis_validation_hardware.yaml")
    simulation = load_config("configs/right_mirror_axis_validation_mujoco.yaml")
    validate_hardware_config(hardware)
    validate_live_hardware_timing(hardware)
    validate_hardware_config(simulation)

    assert hardware["quest"]["hand"] == "right"
    assert hardware["quest"]["mapping_mode"] == "mirror"
    assert hardware["quest"] == simulation["quest"]
    assert hardware["control"] == simulation["control"] == baseline["control"]
    assert hardware["ik"] == simulation["ik"] == baseline["ik"]
    assert hardware["hardware"]["enabled"] is True
    assert simulation["hardware"]["enabled"] is False
    assert hardware["hardware"]["control_gripper"] is False
    assert simulation["hardware"]["control_gripper"] is False
    assert hardware["quest"]["translation_scale"] < baseline["quest"]["translation_scale"]
    assert hardware["quest"]["rotation_scale"] < baseline["quest"]["rotation_scale"]

    calibration = json.loads(
        Path(hardware["quest"]["calibration"]).read_text(encoding="utf-8")
    )
    acceptance = calibration["physical_axis_acceptance"]
    assert acceptance["status"] == "accepted_by_operator"
    assert calibration["task_mapping_semantics"]["name"] == "behind"
    assert not calibration["task_mapping_semantics"]["mirror_semantics_validated"]
    assert acceptance["derivation"]["task_frame_determinant"] == pytest.approx(-1.0)
    assert set(acceptance["motions"].values()) == {"passed"}
    assert acceptance["telemetry"]["run_count"] == 6
    assert acceptance["telemetry"]["total_rows"] == 8099
    assert acceptance["telemetry"]["total_ik_failures"] == 0
    assert acceptance["stable_left_baseline_unchanged"] is True


def test_quest_50pct_profile_requires_and_applies_fixed_profiles() -> None:
    base = load_config("configs/quest_50pct_hardware.yaml")
    step3 = load_config("configs/step3_velocity_feedforward_25ms_hardware.yaml")
    article = load_config("configs/baseline.yaml")
    assert base["project"]["page_locked_task_profile"] is True
    assert base["hardware"]["control_gripper"] is True
    assert base["hardware"]["max_demo_duration_s"] is None
    for key in ("translation_scale", "rotation_scale"):
        assert base["quest"][key] == pytest.approx(0.50 * article["quest"][key])
    for key in ("position_reach_limit_m", "rotation_reach_limit_rad"):
        assert base["quest"][key] == pytest.approx(0.50 * article["quest"][key])
    assert base["hardware"]["gripper_open_m"] == step3["hardware"]["gripper_open_m"]
    assert base["hardware"]["gripper_min_demo_m"] == step3["hardware"][
        "gripper_min_demo_m"
    ]
    assert base["hardware"]["rest_gripper_m"] == step3["hardware"]["rest_gripper_m"]
    assert base["hardware"]["max_gripper_delta_m"] == step3["hardware"][
        "max_gripper_delta_m"
    ]
    assert base["control"]["gripper_command_limits"] == step3["control"][
        "gripper_command_limits"
    ]

    expected = {
        "left_behind": ("left", "real", "behind"),
        "left_mirror": ("left", "mirror", "mirror"),
        "right_behind": ("right", "real", "behind"),
        "right_mirror": ("right", "mirror", "mirror"),
    }
    assert set(TASK_PROFILE_NAMES) == set(expected)
    for profile_name, (hand, mapping_mode, semantics) in expected.items():
        config = load_config("configs/quest_50pct_hardware.yaml")
        apply_task_profile(config, profile_name)
        validate_hardware_config(config)
        validate_live_hardware_timing(config)
        assert config["quest"]["hand"] == hand
        assert config["quest"]["mapping_mode"] == mapping_mode
        assert config["_task_profile"]["semantics"] == semantics
        assert profile_name in config["project"]["mode"]
        calibration = json.loads(
            Path(config["quest"]["calibration"]).read_text(encoding="utf-8")
        )
        expected_status = {
            "left_behind": "accepted_by_operator_at_40pct",
            "left_mirror": "accepted_by_operator_at_45pct",
            "right_mirror": "accepted_by_operator_at_50pct",
            "right_behind": "accepted_by_operator_at_40pct",
        }.get(profile_name, "candidate_pending_physical_validation")
        assert calibration["task_mapping_semantics"]["status"] == expected_status


def test_demo_duration_supports_explicit_uncapped_profile() -> None:
    uncapped = {"max_demo_duration_s": None}
    assert np.isinf(resolve_demo_duration(uncapped, 0.0))
    assert resolve_demo_duration(uncapped, 30.0) == pytest.approx(30.0)

    capped = {"max_demo_duration_s": 15.0}
    assert resolve_demo_duration(capped, 0.0) == pytest.approx(15.0)
    assert resolve_demo_duration(capped, 30.0) == pytest.approx(15.0)

    with pytest.raises(ValueError, match="nonnegative"):
        resolve_demo_duration(uncapped, -1.0)
    with pytest.raises(ValueError, match="finite"):
        resolve_demo_duration(uncapped, np.inf)


def test_hardware_config_accepts_explicit_uncapped_duration_only() -> None:
    config = load_config("configs/quest_50pct_hardware.yaml")
    validate_hardware_config(config)

    config["hardware"]["max_demo_duration_s"] = 0.0
    with pytest.raises(HardwareSafetyError, match="positive or explicit null"):
        validate_hardware_config(config)

    del config["hardware"]["max_demo_duration_s"]
    with pytest.raises(HardwareSafetyError, match="positive number or explicit null"):
        validate_hardware_config(config)


@pytest.mark.parametrize(
    ("hand", "mapping_mode", "expected"),
    [
        ("left", "real", "left_behind"),
        ("left", "mirror", "left_mirror"),
        ("right", "real", "right_behind"),
        ("right", "mirror", "right_mirror"),
    ],
)
def test_page_selection_resolves_all_four_task_profiles(
    hand, mapping_mode, expected
) -> None:
    assert task_profile_for_quest_selection(hand, mapping_mode) == expected


def test_page_selection_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="hand"):
        task_profile_for_quest_selection("either", "real")
    with pytest.raises(ValueError, match="mapping mode"):
        task_profile_for_quest_selection("left", "automatic")


def test_hardware_rejects_selection_change_after_preflight() -> None:
    locked = SimpleNamespace(hand="right", mapping_mode="mirror")
    require_quest_selection(locked, "right", "mirror")

    changed_hand = SimpleNamespace(hand="left", mapping_mode="mirror")
    with pytest.raises(HardwareSafetyError, match="changed after preflight"):
        require_quest_selection(changed_hand, "right", "mirror")

    changed_mode = SimpleNamespace(hand="right", mapping_mode="real")
    with pytest.raises(HardwareSafetyError, match="changed after preflight"):
        require_quest_selection(changed_mode, "right", "mirror")


def test_command_gate_rejects_nonfinite_caps_and_margin() -> None:
    limits = np.vstack([np.tile([-1.0, 1.0], (6, 1)), [0.0, 0.044]])
    with pytest.raises(HardwareSafetyError, match="margin"):
        CommandGate(
            np.zeros(6),
            0.03,
            limits,
            np.full(6, 0.01),
            joint_limit_margin_rad=np.nan,
            gripper_limits_m=(0.0, 0.044),
            max_gripper_delta_m=0.005,
        )
    with pytest.raises(HardwareSafetyError, match="joint-step caps"):
        CommandGate(
            np.zeros(6),
            0.03,
            limits,
            np.full(6, np.nan),
            joint_limit_margin_rad=0.0,
            gripper_limits_m=(0.0, 0.044),
            max_gripper_delta_m=0.005,
        )


def test_time_aligned_feedback_interpolates_delayed_command_and_detects_stall() -> None:
    history = TimeAlignedCommandHistory(np.zeros(6), 10.0, 0.025)
    history.append(10.01, np.full(6, 0.01))
    history.append(10.02, np.full(6, 0.02))

    # Feedback read at 10.040 s is compared with the interpolated 10.015 s
    # command rather than the newest 10.020 s command.
    reference, error = validate_time_aligned_feedback(
        np.full(6, 0.016),
        10.04,
        history,
        0.01,
    )
    np.testing.assert_allclose(reference, np.full(6, 0.015))
    np.testing.assert_allclose(error, np.full(6, 0.001))
    assert history.clamp_state(10.024) == "clamped-to-oldest"
    assert history.clamp_state(10.040) == "interpolated"
    assert history.clamp_state(10.045) == "clamped-to-newest"

    stalled = np.zeros(6)
    with pytest.raises(
        HardwareSafetyError,
        match="joint 0 time-aligned tracking error.*reference clamped-to-newest",
    ):
        validate_time_aligned_feedback(stalled, 10.045, history, 0.01)


def test_time_aligned_command_history_rejects_nonmonotonic_timestamps() -> None:
    history = TimeAlignedCommandHistory(np.zeros(6), 1.0, 0.025)
    with pytest.raises(ValueError, match="strictly increasing"):
        history.append(1.0, np.ones(6))


def test_hardware_state_rejects_invalid_feedback_and_limits() -> None:
    limits = np.vstack([np.tile([-1.0, 1.0], (6, 1)), [0.0, 0.044]])
    with pytest.raises(HardwareSafetyError, match="non-finite"):
        HardwareState(np.full(6, np.nan), 0.03, limits, "1.11.0")
    with pytest.raises(HardwareSafetyError, match="non-finite"):
        HardwareState(np.zeros(6), np.inf, limits, "1.11.0")
    reversed_limits = limits.copy()
    reversed_limits[2] = [1.0, -1.0]
    with pytest.raises(HardwareSafetyError, match="invalid joint limits"):
        HardwareState(np.zeros(6), 0.03, reversed_limits, "1.11.0")
    with pytest.raises(HardwareSafetyError, match="outside"):
        HardwareState(np.full(6, 1.1), 0.03, limits, "1.11.0")
    tolerated = HardwareState(
        np.full(6, 1.0005),
        0.03,
        limits,
        "1.11.0",
        position_tolerances=np.full(7, 0.001),
    )
    np.testing.assert_array_equal(tolerated.q_arm, np.full(6, 1.0005))


def test_startup_ramp_stops_on_tracking_error() -> None:
    limits = np.vstack([np.tile([-1.0, 1.0], (6, 1)), [0.0, 0.044]])
    start = HardwareState(np.zeros(6), 0.044, limits, "1.11.0")

    class StuckBackend:
        def send_positions(self, _q, _gripper, *, include_gripper=True):
            return

        def read_state(self):
            return start

    gate = CommandGate(
        start.q_arm,
        start.gripper_position_m,
        limits,
        np.full(6, 0.2),
        joint_limit_margin_rad=0.0,
        gripper_limits_m=(0.0, 0.044),
        max_gripper_delta_m=0.01,
    )
    config = {
        "hardware": {
            "startup_ramp_duration_s": 0.1,
            "startup_ramp_rate_hz": 10,
            "feedback_check_rate_hz": 10,
            "max_feedback_error_rad": 0.01,
            "max_gripper_feedback_error_m": 0.005,
            "gripper_open_m": 0.044,
        }
    }
    with pytest.raises(HardwareSafetyError, match="during startup ramp"):
        ramp_to_home(StuckBackend(), gate, start, np.full(6, 0.1), config)


def test_startup_ramp_never_commands_or_tracks_the_gripper() -> None:
    limits = np.vstack([np.tile([-1.0, 1.0], (6, 1)), [0.0, 0.044]])
    start = HardwareState(np.zeros(6), 0.044, limits, "1.11.0")

    class StuckGripperBackend:
        def __init__(self):
            self.include_gripper_values = []

        def send_positions(self, _q, _gripper, *, include_gripper=True):
            self.include_gripper_values.append(include_gripper)

        def read_state(self):
            return start

    gate = CommandGate(
        start.q_arm,
        start.gripper_position_m,
        limits,
        np.full(6, 0.2),
        joint_limit_margin_rad=0.0,
        gripper_limits_m=(0.0, 0.044),
        max_gripper_delta_m=0.02,
    )
    config = {
        "hardware": {
            "startup_ramp_duration_s": 0.1,
            "startup_ramp_rate_hz": 10,
            "feedback_check_rate_hz": 10,
            "max_feedback_error_rad": 0.01,
            "max_gripper_feedback_error_m": 0.005,
            "gripper_open_m": 0.034,
        }
    }
    backend = StuckGripperBackend()
    ramp_to_home(backend, gate, start, np.zeros(6), config)
    assert backend.include_gripper_values == [False]


def test_gripper_opens_after_home_with_previous_blocking_strategy() -> None:
    limits = np.vstack([np.tile([-1.0, 1.0], (6, 1)), [0.0, 0.04]])
    tolerances = np.array([0.2] * 6 + [0.004])
    home = np.zeros(6)
    start = HardwareState(
        home,
        0.002,
        limits,
        "1.8.3",
        position_tolerances=tolerances,
    )

    class FollowingGripperBackend:
        def __init__(self):
            self.state = start
            self.move = None

        def move_gripper_blocking(self, gripper_position_m, *, duration_s):
            self.move = (gripper_position_m, duration_s)
            # A 3.5 mm final error is legal because the controller reports a
            # 4 mm gripper tolerance.
            self.state = HardwareState(
                home,
                gripper_position_m - 0.0035,
                limits,
                "1.8.3",
                position_tolerances=tolerances,
            )

        def read_state(self):
            return self.state

    model = SimpleNamespace(first_self_collision_on_path=lambda *_args, **_kwargs: None)
    config = {
        "hardware": {
            "gripper_open_m": 0.04,
            "startup_gripper_move_duration_s": 2.0,
            "startup_collision_samples": 101,
            "max_gripper_feedback_error_m": 0.003,
        }
    }
    backend = FollowingGripperBackend()
    settled = open_gripper_at_home(backend, model, start, home, config)
    assert backend.move == (0.04, 2.0)
    assert settled.gripper_position_m == pytest.approx(0.0365)


def test_arm_only_startup_ramp_leaves_gripper_untouched() -> None:
    limits = np.vstack([np.tile([-1.0, 1.0], (6, 1)), [0.0, 0.044]])
    start = HardwareState(np.zeros(6), 0.012, limits, "1.8.3")

    class FollowingArmBackend:
        def __init__(self):
            self.q = start.q_arm.copy()
            self.gripper = start.gripper_position_m
            self.include_gripper_values = []

        def send_positions(self, q, gripper, *, include_gripper=True):
            self.q = np.asarray(q, dtype=float).copy()
            self.include_gripper_values.append(include_gripper)
            if include_gripper:
                self.gripper = float(gripper)

        def read_state(self):
            return HardwareState(self.q, self.gripper, limits, "1.8.3")

    backend = FollowingArmBackend()
    gate = CommandGate(
        start.q_arm,
        start.gripper_position_m,
        limits,
        np.full(6, 0.2),
        joint_limit_margin_rad=0.0,
        gripper_limits_m=(0.0, 0.044),
        max_gripper_delta_m=0.01,
    )
    config = {
        "hardware": {
            "control_gripper": False,
            "startup_ramp_duration_s": 0.001,
            "startup_ramp_rate_hz": 1000,
            "feedback_check_rate_hz": 1000,
            "max_feedback_error_rad": 0.01,
            "max_gripper_feedback_error_m": 0.0001,
            "gripper_open_m": 0.04,
        }
    }
    home = np.full(6, 0.1)
    ramp_to_home(backend, gate, start, home, config)
    np.testing.assert_allclose(backend.q, home)
    assert backend.gripper == pytest.approx(start.gripper_position_m)
    assert backend.include_gripper_values == [False]


def test_startup_gate_accepts_measured_zero_offsets_within_physical_limits() -> None:
    physical_limits = np.vstack([np.tile([-2.0, 2.0], (6, 1)), [0.0, 0.044]])
    # Match the real J1 report: nominal lower limit is zero, while the measured
    # sleep encoder is slightly negative but within controller tolerance.
    physical_limits[1] = [0.0, np.pi]
    measured_q = np.array([-0.0004, -0.001, 0.006, 0.002, -0.003, -0.001])
    tolerances = np.full(7, 0.01)
    state = HardwareState(
        measured_q,
        -0.003,
        physical_limits,
        "1.8.3",
        position_tolerances=tolerances,
    )
    gate = make_startup_command_gate(
        state,
        np.full(6, 0.0025),
        {"max_gripper_delta_m": 0.00025},
    )
    gate.validate(measured_q + np.full(6, 0.001), state.gripper_position_m)


def test_arm_only_teleop_gate_accepts_tolerated_gripper_feedback() -> None:
    limits = np.vstack([np.tile([-2.0, 2.0], (6, 1)), [0.0, 0.044]])
    state = HardwareState(
        np.zeros(6),
        -0.003,
        limits,
        "1.8.3",
        position_tolerances=np.full(7, 0.01),
    )
    hardware = {
        "gripper_min_demo_m": 0.028,
        "gripper_open_m": 0.04,
    }
    arm_only_limits = teleop_gripper_limits(
        state,
        hardware,
        control_gripper=False,
    )
    assert arm_only_limits == pytest.approx((-0.01, 0.054))
    gate = CommandGate(
        state.q_arm,
        state.gripper_position_m,
        limits,
        np.full(6, 0.01),
        joint_limit_margin_rad=0.1,
        gripper_limits_m=arm_only_limits,
        max_gripper_delta_m=0.00025,
    )
    gate.validate(state.q_arm, state.gripper_position_m)


def test_hardware_demo_uses_article_reference_control_settings() -> None:
    config = load_config("configs/hardware_demo.yaml")
    validate_hardware_config(config)
    baseline = load_config("configs/baseline.yaml")
    assert config["quest"]["translation_scale"] == baseline["quest"]["translation_scale"] == 1.5
    assert config["quest"]["rotation_scale"] == baseline["quest"]["rotation_scale"] == 1.5
    assert config["control"] == baseline["control"] == {
        "loop_rate_hz": 200,
        "pose_filter_alpha": 0.8,
    }
    assert config["hardware"]["command_goal_time_s"] == 0.0
    assert config["ik"]["max_dq_per_joint_rad"] == baseline["ik"]["max_dq_per_joint_rad"]


def test_article_timing_is_rejected_for_live_hardware_but_safe_profile_passes() -> None:
    with pytest.raises(HardwareSafetyError, match="300 Hz"):
        validate_live_hardware_timing(load_config("configs/hardware_demo.yaml"))
    safe = load_config("configs/safe_demo_30pct.yaml")
    validate_hardware_config(safe)
    validate_live_hardware_timing(safe)
    assert safe["quest"]["translation_scale"] == pytest.approx(1.5 * 0.30)
    assert safe["quest"]["rotation_scale"] == pytest.approx(1.5 * 0.30)
    baseline = load_config("configs/baseline.yaml")
    assert safe["ik"]["max_dq_per_joint_rad"] == baseline["ik"]["max_dq_per_joint_rad"]
    assert safe["control"]["loop_rate_hz"] == 100
    assert safe["hardware"]["command_goal_time_s"] == pytest.approx(0.030)
    assert safe["hardware"]["driver_version_tested"] == "1.9.3"
    assert safe["hardware"]["end_effector_profile"] == "legacy_1_8"
    assert safe["hardware"]["control_gripper"] is False
    assert safe["model"]["simulation_start_q_rad"] == pytest.approx(
        [0.0, np.pi / 3.0, 5.0 * np.pi / 12.0, -np.pi / 3.0, 0.0, 0.0]
    )


def test_step_up_profile_uses_full_gripper_range_and_proven_timing() -> None:
    config = load_config("configs/live_demo_50pct_full_gripper.yaml")
    validate_hardware_config(config)
    validate_live_hardware_timing(config)
    assert config["quest"]["translation_scale"] == pytest.approx(1.5 * 0.50)
    assert config["quest"]["rotation_scale"] == pytest.approx(1.5 * 0.50)
    assert config["control"]["loop_rate_hz"] == 100
    assert config["hardware"]["command_goal_time_s"] == pytest.approx(0.030)
    assert config["hardware"]["control_gripper"] is True
    assert config["hardware"]["gripper_open_m"] == pytest.approx(0.040)
    assert config["hardware"]["gripper_min_demo_m"] == pytest.approx(0.0)
    assert config["hardware"]["return_to_rest_on_exit"] is True
    assert config["hardware"]["rest_q_rad"] == [0.0] * 6


def test_article_30pct_profile_removes_only_post_ik_command_shaping() -> None:
    config = load_config("configs/article_30pct_full_gripper.yaml")
    validate_hardware_config(config)
    validate_live_hardware_timing(config)
    baseline = load_config("configs/baseline.yaml")

    assert config["quest"]["translation_scale"] == pytest.approx(1.5 * 0.30)
    assert config["quest"]["rotation_scale"] == pytest.approx(1.5 * 0.30)
    assert config["quest"]["position_reach_limit_m"] == pytest.approx(0.25 * 0.30)
    assert config["quest"]["rotation_reach_limit_rad"] == pytest.approx(0.60 * 0.30)
    assert config["control"] == {
        "command_path": "article_unshaped",
        "loop_rate_hz": 200,
        "pose_filter_alpha": 0.8,
    }
    assert config["ik"] == baseline["ik"]
    assert config["hardware"]["command_goal_time_s"] == pytest.approx(0.010)
    assert config["hardware"]["startup_ramp_rate_hz"] == 200
    assert config["hardware"]["control_gripper"] is True
    assert config["hardware"]["gripper_min_demo_m"] == 0.0
    assert config["hardware"]["gripper_open_m"] == pytest.approx(0.040)


def test_article_unshaped_rejects_added_shaping() -> None:
    config = load_config("configs/article_30pct_full_gripper.yaml")
    config["control"]["joint_command_limits"] = {
        "enabled": True,
        "max_velocity": [1.0] * 6,
        "max_acceleration": [1.0] * 6,
    }
    with pytest.raises(HardwareSafetyError, match="must not add"):
        validate_live_hardware_timing(config)



def test_zero_horizon_still_requires_trossen_minimum_rate() -> None:
    config = load_config("configs/article_30pct_full_gripper.yaml")
    config["hardware"]["command_goal_time_s"] = 0.0
    with pytest.raises(HardwareSafetyError, match="300 Hz"):
        validate_live_hardware_timing(config)


def test_optimized_30pct_profile_targets_quest_rate_and_light_shaping() -> None:
    config = load_config("configs/optimized_30pct_25ms.yaml")
    validate_hardware_config(config)
    validate_live_hardware_timing(config)
    baseline = load_config("configs/baseline.yaml")

    assert config["quest"]["translation_scale"] == pytest.approx(1.5 * 0.30)
    assert config["quest"]["rotation_scale"] == pytest.approx(1.5 * 0.30)
    assert config["control"]["loop_rate_hz"] == 120
    assert config["control"]["pose_filter_alpha"] == pytest.approx(0.8)
    assert config["control"]["joint_command_limits"]["max_velocity"] == [
        2.0,
        2.0,
        2.0,
        3.0,
        3.0,
        3.0,
    ]
    assert config["control"]["joint_command_limits"]["max_acceleration"] == [
        15.0,
        15.0,
        15.0,
        30.0,
        30.0,
        30.0,
    ]
    assert config["ik"] == baseline["ik"]
    assert config["hardware"]["command_goal_time_s"] == pytest.approx(0.015)
    assert config["hardware"]["startup_ramp_duration_s"] == pytest.approx(2.0)
    assert config["hardware"]["startup_ramp_rate_hz"] == 120
    assert config["hardware"]["startup_max_joint_delta_rad"] == [0.006] * 6
    home = np.asarray(config["model"]["simulation_start_q_rad"])
    per_tick_from_rest = np.abs(home) / (
        config["hardware"]["startup_ramp_duration_s"]
        * config["hardware"]["startup_ramp_rate_hz"]
    )
    assert np.all(per_tick_from_rest <= config["hardware"]["startup_max_joint_delta_rad"])
    assert config["hardware"]["gripper_min_demo_m"] == 0.0
    assert config["hardware"]["gripper_open_m"] == pytest.approx(0.040)


def test_smooth_profile_is_quest_synchronized_jerk_limited_and_full_gripper() -> None:
    config = load_config("configs/smooth_30pct_full_gripper.yaml")
    validate_hardware_config(config)
    validate_live_hardware_timing(config)

    assert config["quest"]["translation_scale"] == pytest.approx(0.45)
    assert config["quest"]["rotation_scale"] == pytest.approx(0.45)
    assert config["control"]["update_mode"] == "quest_synchronized"
    assert config["control"]["loop_rate_hz"] == 90
    assert config["control"]["pose_filter_translation_alpha"] == pytest.approx(0.8)
    assert config["control"]["pose_filter_rotation_alpha"] == pytest.approx(0.5)
    assert config["control"]["joint_command_limits"]["max_velocity"] == [
        3.0,
        3.0,
        3.0,
        4.5,
        4.5,
        4.5,
    ]
    assert config["control"]["joint_command_limits"]["max_acceleration"] == [
        20.0,
        20.0,
        20.0,
        25.0,
        25.0,
        25.0,
    ]
    assert config["control"]["joint_command_limits"]["max_jerk"] == [
        1800.0,
        1800.0,
        1800.0,
        1200.0,
        1200.0,
        1200.0,
    ]
    assert config["hardware"]["command_goal_time_s"] == pytest.approx(0.015)
    assert config["hardware"]["gripper_open_m"] == pytest.approx(0.040)
    assert config["hardware"]["gripper_min_demo_m"] == pytest.approx(0.0)
    assert config["control"]["gripper_command_limits"]["max_velocity"] == [0.120]


def test_calibrated_adaptive_hardware_profile_preserves_fast_timing_and_loose_guards() -> None:
    config = load_config("configs/calibrated_adaptive_30pct_hardware.yaml")
    validate_hardware_config(config)
    validate_live_hardware_timing(config)

    assert config["quest"]["calibration"].endswith("left_guided_6dof.json")
    assert config["quest"]["translation_scale"] == pytest.approx(0.45)
    assert config["quest"]["rotation_scale"] == pytest.approx(0.45)
    assert config["control"]["loop_rate_hz"] == 90
    assert config["control"]["update_mode"] == "quest_synchronized"
    assert config["hardware"]["command_goal_time_s"] == pytest.approx(0.015)
    assert config["control"]["adaptive_rotation_filter"] == {
        "enabled": True,
        "minimum_cutoff_hz": 1.5,
        "speed_coefficient": 20.0,
        "speed_exponent": 2.0,
        "derivative_cutoff_hz": 2.0,
        "maximum_cutoff_hz": 35.0,
    }
    assert config["control"]["joint_command_limits"]["max_acceleration"] == [
        1000.0,
        1000.0,
        1000.0,
        2000.0,
        2000.0,
        2000.0,
    ]
    assert config["control"]["gripper_command_limits"]["max_velocity"] == [0.120]


def test_no_catchup_profiles_change_only_cadence_and_driver_horizon() -> None:
    baseline_hardware = load_config("configs/calibrated_adaptive_30pct_hardware.yaml")
    hardware = load_config("configs/no_catchup_30pct_hardware.yaml")
    baseline_sim = load_config("configs/calibrated_adaptive_30pct_mujoco.yaml")
    simulation = load_config("configs/no_catchup_30pct_mujoco.yaml")

    validate_hardware_config(hardware)
    validate_live_hardware_timing(hardware)
    assert hardware["control"]["minimum_command_interval_s"] == pytest.approx(1.0 / 90.0)
    assert simulation["control"]["minimum_command_interval_s"] == pytest.approx(1.0 / 90.0)
    assert hardware["hardware"]["command_goal_time_s"] == pytest.approx(0.030)

    hardware_control = dict(hardware["control"])
    simulation_control = dict(simulation["control"])
    hardware_control.pop("minimum_command_interval_s")
    simulation_control.pop("minimum_command_interval_s")
    assert hardware_control == baseline_hardware["control"]
    assert simulation_control == baseline_sim["control"]
    assert hardware["quest"] == baseline_hardware["quest"]
    assert simulation["quest"] == baseline_sim["quest"]


def test_send_barrier_profiles_change_only_command_spacing_policy() -> None:
    baseline_hardware = load_config("configs/no_catchup_30pct_hardware.yaml")
    hardware = load_config("configs/send_barrier_30pct_hardware.yaml")
    baseline_sim = load_config("configs/no_catchup_30pct_mujoco.yaml")
    simulation = load_config("configs/send_barrier_30pct_mujoco.yaml")

    validate_hardware_config(hardware)
    validate_live_hardware_timing(hardware)
    assert hardware["control"]["command_spacing_stage"] == "before_send"
    assert simulation["control"]["command_spacing_stage"] == "before_send"
    assert hardware["control"]["minimum_command_interval_s"] == pytest.approx(0.008)
    assert simulation["control"]["minimum_command_interval_s"] == pytest.approx(0.008)

    hardware_control = dict(hardware["control"])
    simulation_control = dict(simulation["control"])
    baseline_hardware_control = dict(baseline_hardware["control"])
    baseline_sim_control = dict(baseline_sim["control"])
    hardware_control.pop("command_spacing_stage")
    simulation_control.pop("command_spacing_stage")
    hardware_control.pop("minimum_command_interval_s")
    simulation_control.pop("minimum_command_interval_s")
    baseline_hardware_control.pop("minimum_command_interval_s")
    baseline_sim_control.pop("minimum_command_interval_s")
    assert hardware_control == baseline_hardware_control
    assert simulation_control == baseline_sim_control
    for section in ("model", "quest", "ik", "hardware", "telemetry"):
        assert hardware[section] == baseline_hardware[section]
        assert simulation[section] == baseline_sim[section]


def test_step2_profile_changes_only_driver_horizon_from_milestone_3_step1() -> None:
    step1 = load_config("configs/send_barrier_30pct_hardware.yaml")
    step2 = load_config("configs/step2_25ms_30pct_hardware.yaml")

    validate_hardware_config(step2)
    validate_live_hardware_timing(step2)
    assert step1["hardware"]["command_goal_time_s"] == pytest.approx(0.030)
    assert step2["hardware"]["command_goal_time_s"] == pytest.approx(0.025)

    step1_hardware = dict(step1["hardware"])
    step2_hardware = dict(step2["hardware"])
    step1_hardware.pop("command_goal_time_s")
    step2_hardware.pop("command_goal_time_s")
    assert step2_hardware == step1_hardware
    for section in ("model", "quest", "control", "ik", "telemetry"):
        assert step2[section] == step1[section]


def test_step3_profile_adds_feedforward_and_tabletop_preflight_to_step2() -> None:
    step2 = load_config("configs/step2_25ms_30pct_hardware.yaml")
    step3 = load_config("configs/step3_velocity_feedforward_25ms_hardware.yaml")

    validate_hardware_config(step3)
    validate_live_hardware_timing(step3)
    feedforward = step3["hardware"]["arm_velocity_feedforward"]
    assert feedforward == {
        "enabled": True,
        "filter_alpha": 0.5,
        "gain": 0.5,
        "max_velocity_rad_s": [1.5, 1.25, 2.75, 2.0, 2.5, 3.0],
    }

    step3_hardware = dict(step3["hardware"])
    step3_hardware.pop("arm_velocity_feedforward")
    assert step3_hardware == step2["hardware"]
    step3_quest = dict(step3["quest"])
    assert step3_quest.pop("prepare_tabletop_tracking_via_adb") is True
    assert step3_quest == step2["quest"]
    for section in ("model", "control", "ik", "telemetry"):
        assert step3[section] == step2[section]


def test_feedforward_config_rejects_caps_above_joint_command_caps() -> None:
    config = load_config("configs/step3_velocity_feedforward_25ms_hardware.yaml")
    config["hardware"]["arm_velocity_feedforward"]["max_velocity_rad_s"][0] = 5.5
    with pytest.raises(HardwareSafetyError, match="must not exceed"):
        validate_hardware_config(config)


def test_hardware_config_rejects_nonboolean_tabletop_preflight() -> None:
    config = load_config("configs/step3_velocity_feedforward_25ms_hardware.yaml")
    config["quest"]["prepare_tabletop_tracking_via_adb"] = "yes"
    with pytest.raises(HardwareSafetyError, match="must be a boolean"):
        validate_hardware_config(config)


def test_hardware_config_rejects_incompatible_minimum_command_spacing() -> None:
    config = load_config("configs/smooth_30pct_full_gripper.yaml")
    config["control"]["minimum_command_interval_s"] = np.nan
    with pytest.raises(HardwareSafetyError, match="minimum_command_interval_s"):
        validate_hardware_config(config)

    config = load_config("configs/smooth_30pct_full_gripper.yaml")
    config["control"]["update_mode"] = "fixed_rate"
    config["control"]["minimum_command_interval_s"] = 1.0 / 90.0
    with pytest.raises(HardwareSafetyError, match="quest_synchronized"):
        validate_hardware_config(config)

    config = load_config("configs/smooth_30pct_full_gripper.yaml")
    config["control"]["minimum_command_interval_s"] = config["quest"]["stale_timeout_s"]
    with pytest.raises(HardwareSafetyError, match="stale timeout"):
        validate_hardware_config(config)

    config = load_config("configs/no_catchup_30pct_hardware.yaml")
    config["control"]["command_spacing_stage"] = "after_everything"
    with pytest.raises(HardwareSafetyError, match="spacing stage"):
        validate_hardware_config(config)

    config = load_config("configs/smooth_30pct_full_gripper.yaml")
    config["control"]["command_spacing_stage"] = "before_send"
    with pytest.raises(HardwareSafetyError, match="positive minimum"):
        validate_hardware_config(config)


def test_hardware_config_rejects_unknown_update_mode_and_bad_jerk() -> None:
    config = load_config("configs/smooth_30pct_full_gripper.yaml")
    config["control"]["update_mode"] = "timer_guess"
    with pytest.raises(HardwareSafetyError, match="update_mode"):
        validate_hardware_config(config)

    config = load_config("configs/smooth_30pct_full_gripper.yaml")
    config["control"]["joint_command_limits"]["max_jerk"][2] = np.nan
    with pytest.raises(HardwareSafetyError, match="jerks"):
        validate_hardware_config(config)

    config = load_config("configs/smooth_30pct_full_gripper.yaml")
    config["control"]["pose_filter_rotation_alpha"] = 1.1
    with pytest.raises(HardwareSafetyError, match="pose_filter_rotation_alpha"):
        validate_hardware_config(config)


def test_return_to_rest_checks_path_and_uses_blocking_backend_move() -> None:
    limits = np.vstack([np.tile([-2.0, 2.0], (6, 1)), [0.0, 0.044]])
    start = HardwareState(np.full(6, 0.2), 0.04, limits, "1.8.3")

    class FollowingBackend:
        def __init__(self):
            self.state = start
            self.move = None

        def read_state(self):
            return self.state

        def move_to_rest(
            self,
            q_rest,
            gripper_position_m,
            *,
            duration_s,
            include_gripper=True,
        ):
            q_rest = np.asarray(q_rest, dtype=float)
            self.move = (q_rest.copy(), gripper_position_m, duration_s, include_gripper)
            self.state = HardwareState(q_rest, gripper_position_m, limits, "1.8.3")

    collision_paths = []

    def collision_check(*args, **kwargs):
        collision_paths.append((args, kwargs))
        return None

    model = SimpleNamespace(first_self_collision_on_path=collision_check)
    config = {
        "hardware": {
            "rest_q_rad": [0.0] * 6,
            "rest_gripper_m": 0.0,
            "shutdown_move_duration_s": 2.0,
            "startup_collision_samples": 101,
            "max_feedback_error_rad": 0.01,
            "max_gripper_feedback_error_m": 0.001,
        }
    }
    backend = FollowingBackend()
    return_to_rest(backend, model, config, control_gripper=True)
    np.testing.assert_array_equal(backend.move[0], np.zeros(6))
    assert backend.move[1:] == (0.0, 2.0, True)
    assert len(collision_paths) == 2
    # The arm moves first with a fixed gripper, then the gripper closes at rest.
    assert collision_paths[0][1]["start_gripper_q"] == collision_paths[0][1]["end_gripper_q"]
    np.testing.assert_array_equal(collision_paths[1][0][0], np.zeros(6))
    np.testing.assert_array_equal(collision_paths[1][0][1], np.zeros(6))


def test_trossen_versions_must_match_major_minor() -> None:
    require_compatible_versions("1.11.0", "v1.11.7")
    with pytest.raises(HardwareSafetyError, match="mismatch"):
        require_compatible_versions("1.11.0", "1.10.9")
    with pytest.raises(HardwareSafetyError, match="could not parse"):
        require_compatible_versions("unknown", "1.11.0")


def test_official_backend_fails_closed_on_windows() -> None:
    backend = TrossenArmBackend("192.168.1.2", driver_module=object(), platform_name="Windows")
    with pytest.raises(HardwareUnavailableError, match="not Windows"):
        backend.connect()


def test_official_backend_uses_configured_nonblocking_driver_calls() -> None:
    class EnumValue:
        def __init__(self, value):
            self.value = value

    class Limit:
        def __init__(self, lower, upper):
            self.position_min = lower
            self.position_max = upper

    class FakeDriver:
        instance = None

        def __init__(self):
            FakeDriver.instance = self
            self.positions = np.array([0.0, 1.0, 0.7, 0.0, 0.0, 0.0, 0.044])
            self.calls = []

        @staticmethod
        def discover(**kwargs):
            assert kwargs == {"subnet": "192.168.1", "ip_start": 2, "ip_end": 2, "timeout": 0.05}
            return [SimpleNamespace(ip="192.168.1.2", model=EnumValue(1), error_state=EnumValue(0), firmware_version="1.11.1")]

        def configure(self, *args):
            self.calls.append(("configure", args))

        def get_all_positions(self):
            return self.positions.copy()

        def get_joint_limits(self):
            return [Limit(-1.0, 1.0) for _ in range(6)] + [Limit(0.0, 0.044)]

        def set_arm_modes(self, mode):
            self.calls.append(("arm_mode", mode))

        def set_gripper_mode(self, mode):
            self.calls.append(("gripper_mode", mode))

        def set_arm_positions(
            self,
            positions,
            goal_time,
            blocking,
            feedforward_velocity=None,
        ):
            self.positions[:6] = np.asarray(positions, dtype=float)
            feedforward = (
                None
                if feedforward_velocity is None
                else np.asarray(feedforward_velocity, dtype=float).copy()
            )
            self.calls.append(
                ("arm_positions", self.positions[:6].copy(), goal_time, blocking, feedforward)
            )

        def set_gripper_position(self, position, goal_time, blocking):
            self.positions[6] = float(position)
            self.calls.append(("gripper_position", float(position), goal_time, blocking))

        def cleanup(self):
            self.calls.append(("cleanup",))

    module = SimpleNamespace(
        TrossenArmDriver=FakeDriver,
        Model=SimpleNamespace(wxai_v0=EnumValue(1)),
        StandardEndEffector=SimpleNamespace(wxai_v0_follower_20250509=EnumValue(2)),
        Mode=SimpleNamespace(position=EnumValue(3)),
    )
    backend = TrossenArmBackend(
        "192.168.1.2",
        driver_module=module,
        platform_name="Linux",
        driver_version="1.11.0",
        command_goal_time_s=0.03,
        end_effector_variant="wxai_v0_follower_20250509",
    )
    state = backend.connect()
    assert state.firmware_version == "1.11.1"
    backend.enable_position_control()
    q = np.array([0.01, 0.99, 0.69, 0.02, -0.01, 0.03])
    backend.send_positions(q, 0.04)
    arm_call, gripper_call = FakeDriver.instance.calls[-2:]
    assert arm_call[0] == "arm_positions"
    np.testing.assert_array_equal(arm_call[1], q)
    assert arm_call[2:4] == (0.03, False)
    assert arm_call[4] is None
    assert gripper_call == ("gripper_position", 0.04, 0.03, False)
    feedforward = np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    backend.send_positions(
        q,
        0.04,
        arm_feedforward_velocity=feedforward,
    )
    arm_call, gripper_call = FakeDriver.instance.calls[-2:]
    np.testing.assert_array_equal(arm_call[4], feedforward)
    assert gripper_call == ("gripper_position", 0.04, 0.03, False)
    backend.move_gripper_blocking(0.04, duration_s=2.0)
    assert FakeDriver.instance.calls[-2][0] == "gripper_mode"
    assert FakeDriver.instance.calls[-1] == ("gripper_position", 0.04, 2.0, True)
    backend.move_to_rest(np.zeros(6), 0.0, duration_s=2.0)
    arm_call, gripper_call = FakeDriver.instance.calls[-3], FakeDriver.instance.calls[-1]
    assert arm_call[0] == "arm_positions"
    np.testing.assert_array_equal(arm_call[1], np.zeros(6))
    assert arm_call[2:4] == (2.0, True)
    assert arm_call[4] is None
    assert gripper_call == ("gripper_position", 0.0, 2.0, True)
    position_call_count = sum(
        item[0] in ("arm_positions", "gripper_position")
        for item in FakeDriver.instance.calls
    )
    FakeDriver.instance.positions[0] = np.nan
    with pytest.raises(HardwareSafetyError, match="shutdown hold"):
        backend.safe_hold()
    assert sum(
        item[0] in ("arm_positions", "gripper_position")
        for item in FakeDriver.instance.calls
    ) == position_call_count
    backend.close()
    assert FakeDriver.instance.calls[-1] == ("cleanup",)


def test_official_backend_rejects_nonfinite_command_horizon() -> None:
    with pytest.raises(ValueError, match="finite"):
        TrossenArmBackend("192.168.1.2", command_goal_time_s=np.nan)


def test_official_backend_rejects_firmware_mismatch_before_configure() -> None:
    class EnumValue:
        def __init__(self, value):
            self.value = value

    class FakeDriver:
        configured = False

        @staticmethod
        def discover(**_kwargs):
            return [
                SimpleNamespace(
                    ip="192.168.1.2",
                    model=EnumValue(1),
                    error_state=EnumValue(0),
                    firmware_version="1.10.4",
                )
            ]

        def configure(self, *_args):
            FakeDriver.configured = True

    module = SimpleNamespace(
        TrossenArmDriver=FakeDriver,
        Model=SimpleNamespace(wxai_v0=EnumValue(1)),
        StandardEndEffector=SimpleNamespace(wxai_v0_follower=object()),
    )
    backend = TrossenArmBackend(
        "192.168.1.2",
        driver_module=module,
        platform_name="Linux",
        driver_version="1.11.0",
        end_effector_variant="wxai_v0_follower_20250509",
    )
    with pytest.raises(HardwareSafetyError, match="mismatch"):
        backend.connect()
    assert not FakeDriver.configured


def test_official_backend_requires_explicit_end_effector_profile() -> None:
    class EnumValue:
        def __init__(self, value):
            self.value = value

    class FakeDriver:
        @staticmethod
        def discover(**_kwargs):
            return [
                SimpleNamespace(
                    ip="192.168.1.2",
                    model=EnumValue(1),
                    error_state=EnumValue(0),
                    firmware_version="1.11.2",
                )
            ]

    module = SimpleNamespace(
        TrossenArmDriver=FakeDriver,
        Model=SimpleNamespace(wxai_v0=EnumValue(1)),
    )
    backend = TrossenArmBackend(
        "192.168.1.2",
        driver_module=module,
        platform_name="Linux",
        driver_version="1.11.0",
    )
    with pytest.raises(HardwareSafetyError, match="profile"):
        backend.connect()


def test_discovery_only_does_not_construct_or_configure_a_driver() -> None:
    class EnumValue:
        def __init__(self, value):
            self.value = value

    class DiscoveryOnlyDriver:
        def __init__(self):
            raise AssertionError("discovery-only preflight constructed a driver")

        @staticmethod
        def discover(**_kwargs):
            return [
                SimpleNamespace(
                    ip="192.168.1.2",
                    model=EnumValue(1),
                    error_state=EnumValue(0),
                    firmware_version="1.11.4",
                )
            ]

    module = SimpleNamespace(
        TrossenArmDriver=DiscoveryOnlyDriver,
        Model=SimpleNamespace(wxai_v0=EnumValue(1)),
    )
    backend = TrossenArmBackend(
        "192.168.1.2",
        driver_module=module,
        platform_name="Linux",
        driver_version="1.11.0",
    )
    discovery = backend.discover()
    assert discovery.robot_ip == "192.168.1.2"
    assert discovery.firmware_version == "1.11.4"


def test_legacy_18_backend_uses_proven_unversioned_follower_without_discovery() -> None:
    class EnumValue:
        def __init__(self, value):
            self.value = value

    class Limit:
        def __init__(self, lower, upper):
            self.position_min = lower
            self.position_max = upper
            self.position_tolerance = 1e-4

    legacy_follower = object()

    class LegacyDriver:
        instance = None

        def __init__(self):
            LegacyDriver.instance = self
            self.calls = []

        def configure(self, *args):
            self.calls.append(("configure", args))

        def get_controller_version(self):
            return "1.8.3"

        def get_all_positions(self):
            return [0.0, 1.0, 0.7, 0.0, 0.0, 0.0, 0.04]

        def get_joint_limits(self):
            return [Limit(-2.0, 2.0) for _ in range(6)] + [Limit(0.0, 0.044)]

        def set_arm_modes(self, mode):
            self.calls.append(("arm_mode", mode))

        def set_arm_positions(self, positions, goal_time, blocking):
            self.calls.append(
                ("arm_positions", np.asarray(positions, dtype=float).copy(), goal_time, blocking)
            )

        def cleanup(self):
            self.calls.append(("cleanup",))

    module = SimpleNamespace(
        TrossenArmDriver=LegacyDriver,
        Model=SimpleNamespace(wxai_v0=EnumValue(1)),
        StandardEndEffector=SimpleNamespace(wxai_v0_follower=legacy_follower),
        Mode=SimpleNamespace(position=EnumValue(3)),
    )
    backend = TrossenArmBackend(
        "192.168.1.2",
        driver_module=module,
        platform_name="Linux",
        driver_version="1.8.6",
        required_driver_version="1.8.6",
        end_effector_variant="wxai_v0_follower",
    )
    state = backend.connect()
    assert state.driver_version == "1.8.6"
    assert state.firmware_version == "1.8.3"
    configure = LegacyDriver.instance.calls[0]
    assert configure[0] == "configure"
    assert configure[1][1] is legacy_follower
    assert configure[1][2:] == ("192.168.1.2", False)
    backend.enable_position_control(include_gripper=False)
    q = np.array([0.0, np.pi / 3.0, 5.0 * np.pi / 12.0, -np.pi / 3.0, 0.0, 0.0])
    backend.send_positions(q, state.gripper_position_m, include_gripper=False)
    assert LegacyDriver.instance.calls[-2][0] == "arm_mode"
    arm_position_call = LegacyDriver.instance.calls[-1]
    assert arm_position_call[0] == "arm_positions"
    np.testing.assert_array_equal(arm_position_call[1], q)
    assert arm_position_call[2:] == (0.0, False)
    backend.close()


def test_required_legacy_driver_version_is_checked_before_arm_connection() -> None:
    class DriverThatMustNotBeConstructed:
        def __init__(self):
            raise AssertionError("wrong driver must fail before construction")

    module = SimpleNamespace(TrossenArmDriver=DriverThatMustNotBeConstructed)
    backend = TrossenArmBackend(
        "192.168.1.2",
        driver_module=module,
        platform_name="Linux",
        driver_version="1.11.0",
        required_driver_version="1.8.6",
        end_effector_variant="wxai_v0_follower",
    )
    with pytest.raises(HardwareUnavailableError, match="no arm connection was attempted"):
        backend.connect()


# Both lab controllers were flashed from firmware 1.8.3 to 1.9.4 on 2026-10-20.
LAB_FIRMWARE_VERSION = "1.9.4"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def pinned_hardware_driver_version() -> str:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'"trossen-arm==([0-9.]+);', pyproject)
    assert match is not None, "pyproject hardware extra must pin trossen-arm exactly"
    return match.group(1)


def test_lab_configs_pin_the_driver_that_matches_the_upgraded_firmware() -> None:
    pinned = pinned_hardware_driver_version()
    require_compatible_versions(pinned, LAB_FIRMWARE_VERSION)
    lab_configs = {}
    for path in sorted((PROJECT_ROOT / "configs").glob("*.yaml")):
        hardware = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("hardware") or {}
        if hardware.get("end_effector_profile") == "legacy_1_8":
            lab_configs[path.name] = str(hardware.get("driver_version_tested"))
    assert {"quest_50pct_hardware.yaml", "dual_widowxai.yaml"} <= set(lab_configs)
    assert {name: version for name, version in lab_configs.items() if version != pinned} == {}


def test_driver_193_connects_to_firmware_194_through_discovery_with_unversioned_follower() -> None:
    class EnumValue:
        def __init__(self, value):
            self.value = value

    class Limit:
        def __init__(self, lower, upper):
            self.position_min = lower
            self.position_max = upper
            self.position_tolerance = 1e-4

    follower = object()

    class Driver193:
        instance = None
        discover_calls = []

        def __init__(self):
            Driver193.instance = self
            self.calls = []

        @staticmethod
        def discover(**kwargs):
            Driver193.discover_calls.append(kwargs)
            return [
                SimpleNamespace(
                    ip="192.168.1.3",
                    model=EnumValue(1),
                    error_state=EnumValue(0),
                    firmware_version=LAB_FIRMWARE_VERSION,
                )
            ]

        def configure(self, *args):
            self.calls.append(("configure", args))

        def get_controller_version(self):
            raise AssertionError("1.9 connects take the firmware version from discovery")

        def get_all_positions(self):
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        def get_joint_limits(self):
            return [Limit(-2.0, 2.0) for _ in range(6)] + [Limit(0.0, 0.044)]

        def set_arm_modes(self, mode):
            self.calls.append(("arm_mode", mode))

        def set_arm_positions(
            self,
            goal_positions,
            goal_time=2.0,
            blocking=True,
            goal_feedforward_velocities=None,
            goal_feedforward_accelerations=None,
        ):
            self.calls.append(
                (
                    "arm_positions",
                    np.asarray(goal_positions, dtype=float).copy(),
                    goal_time,
                    blocking,
                    None
                    if goal_feedforward_velocities is None
                    else np.asarray(goal_feedforward_velocities, dtype=float).copy(),
                )
            )

        def cleanup(self, reboot_controller=False):
            self.calls.append(("cleanup", reboot_controller))

    module = SimpleNamespace(
        TrossenArmDriver=Driver193,
        Model=SimpleNamespace(wxai_v0=EnumValue(1)),
        # Driver 1.9.3 has only the unversioned follower name.
        StandardEndEffector=SimpleNamespace(wxai_v0_follower=follower),
        Mode=SimpleNamespace(position=EnumValue(3)),
    )
    backend = TrossenArmBackend(
        "192.168.1.3",
        driver_module=module,
        platform_name="Linux",
        driver_version="1.9.3",
        required_driver_version=pinned_hardware_driver_version(),
        command_goal_time_s=0.025,
        end_effector_variant=END_EFFECTOR_PROFILE_TO_VARIANT["legacy_1_8"],
    )
    state = backend.connect()
    assert Driver193.discover_calls == [
        {"subnet": "192.168.1", "ip_start": 3, "ip_end": 3, "timeout": 0.05}
    ]
    assert state.driver_version == "1.9.3"
    assert state.firmware_version == LAB_FIRMWARE_VERSION
    configure = Driver193.instance.calls[0]
    assert configure[0] == "configure"
    assert configure[1][1] is follower
    assert configure[1][2:] == ("192.168.1.3", False)
    backend.enable_position_control(include_gripper=False)
    q = np.array([0.0, np.pi / 3.0, 5.0 * np.pi / 12.0, -np.pi / 3.0, 0.0, 0.0])
    feedforward = np.array([0.1, -0.1, 0.0, 0.2, 0.0, -0.2])
    backend.send_positions(q, 0.0, include_gripper=False, arm_feedforward_velocity=feedforward)
    arm_call = Driver193.instance.calls[-1]
    assert arm_call[0] == "arm_positions"
    np.testing.assert_array_equal(arm_call[1], q)
    assert arm_call[2:4] == (0.025, False)
    np.testing.assert_array_equal(arm_call[4], feedforward)
    backend.close()
    assert Driver193.instance.calls[-1] == ("cleanup", False)


def test_retired_186_driver_is_refused_for_upgraded_lab_configs_before_any_contact() -> None:
    class DriverThatMustNotBeTouched:
        def __init__(self):
            raise AssertionError("a retired driver must fail before construction")

        @staticmethod
        def discover(**_kwargs):
            raise AssertionError("a retired driver must fail before discovery")

    config = load_config(PROJECT_ROOT / "configs" / "quest_50pct_hardware.yaml")
    backend = TrossenArmBackend(
        str(config["hardware"]["robot_ip"]),
        driver_module=SimpleNamespace(TrossenArmDriver=DriverThatMustNotBeTouched),
        platform_name="Linux",
        driver_version="1.8.6",
        required_driver_version=str(config["hardware"]["driver_version_tested"]),
        end_effector_variant=END_EFFECTOR_PROFILE_TO_VARIANT["legacy_1_8"],
    )
    with pytest.raises(HardwareUnavailableError, match="no arm connection was attempted"):
        backend.connect()
    with pytest.raises(HardwareSafetyError, match="major.minor must match"):
        require_compatible_versions("1.8.6", LAB_FIRMWARE_VERSION)


def test_installed_193_driver_exposes_every_call_the_backend_makes() -> None:
    """Offline API check against the real driver; importing it opens no connection."""

    trossen_arm = pytest.importorskip("trossen_arm")
    if importlib.metadata.version("trossen-arm") != pinned_hardware_driver_version():
        pytest.skip("runs only in the hardware environment with the pinned driver")
    driver = trossen_arm.TrossenArmDriver
    for name in (
        "discover",
        "configure",
        "get_all_positions",
        "get_joint_limits",
        "set_arm_modes",
        "set_gripper_mode",
        "set_arm_positions",
        "set_gripper_position",
        "cleanup",
    ):
        assert hasattr(driver, name), name
    assert hasattr(trossen_arm.StandardEndEffector, END_EFFECTOR_PROFILE_TO_VARIANT["legacy_1_8"])
    assert hasattr(trossen_arm.Model, "wxai_v0")
    assert hasattr(trossen_arm.Mode, "position")
    for field in ("position_min", "position_max", "position_tolerance"):
        assert hasattr(trossen_arm.JointLimit, field), field
    for field in ("ip", "model", "error_state", "firmware_version"):
        assert hasattr(trossen_arm.DiscoverResult, field), field

    def positional_names(method) -> list[str]:
        signature = method.__doc__.splitlines()[0]
        params = signature[signature.index("(") + 1 : signature.rindex(") ->")]
        return [part.split(":")[0].strip() for part in params.split(", ") if ":" in part]

    assert positional_names(driver.configure)[:5] == [
        "self",
        "model",
        "end_effector",
        "serv_ip",
        "clear_error",
    ]
    assert positional_names(driver.set_arm_positions)[:5] == [
        "self",
        "goal_positions",
        "goal_time",
        "blocking",
        "goal_feedforward_velocities",
    ]
    assert positional_names(driver.set_gripper_position)[:4] == [
        "self",
        "goal_position",
        "goal_time",
        "blocking",
    ]
