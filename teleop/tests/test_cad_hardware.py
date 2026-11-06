from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.run_cad_hardware import (
    make_cad_command_gate,
    require_initial_rest,
    require_released_hold_to_run,
    resolve_cad_duration,
)
from scripts.run_cad_sim import SimulationDeadmanConsole
from scripts.check_cad_stream import summarize_cad_samples
from widowxai_quest_teleop.cad_input import (
    CAD_MAPPING_ACCEPTED,
    CAD_MAPPING_PENDING,
    CadSafetyError,
    CadJointSample,
    cad_live_confirmation_token,
    rest_command_limits,
    validate_cad_commissioning_selection,
    validate_cad_hardware_config,
)
from widowxai_quest_teleop.config import load_config
from widowxai_quest_teleop.hardware import HardwareSafetyError, HardwareState
from widowxai_quest_teleop.hold_to_run import HoldToRunError, HoldToRunSample
from widowxai_quest_teleop.telemetry import CAD_TELEMETRY_COLUMNS


CONFIG_PATH = "configs/cad_hardware_commissioning.yaml"
HOME_Q = np.array([0.0, np.pi / 3.0, 5.0 * np.pi / 12.0, -np.pi / 3.0, 0.0, 0.0])


def hardware_state(q: np.ndarray) -> HardwareState:
    arm_limits = np.array(
        [
            [-3.05433, 3.05433],
            [0.0, np.pi],
            [0.0, 2.35619],
            [-np.pi / 2.0, np.pi / 2.0],
            [-np.pi / 2.0, np.pi / 2.0],
            [-np.pi, np.pi],
        ]
    )
    limits = np.vstack([arm_limits, [0.0, 0.044]])
    return HardwareState(
        q,
        0.0,
        limits,
        "1.9.4",
        "1.9.3",
        np.full(7, 1e-4),
    )


def test_cad_hardware_profile_is_narrow_pending_and_fail_closed() -> None:
    config = load_config(CONFIG_PATH)
    validate_cad_hardware_config(config)

    assert config["commissioning"]["mapping_status"] == CAD_MAPPING_PENDING
    assert config["commissioning"]["require_isolated_joint_while_pending"] is True
    assert config["hardware"]["control_gripper"] is False
    assert config["hardware"]["max_demo_duration_s"] == pytest.approx(15.0)
    assert max(config["cad"]["scales"]) == pytest.approx(0.10)
    assert np.max(np.abs(config["commissioning"]["maximum_delta_rad"][:5])) == pytest.approx(
        np.deg2rad(2.0)
    )
    assert abs(config["commissioning"]["maximum_delta_rad"][5]) <= 1e-5

    simulation = load_config("configs/cad_home_commissioning_mujoco.yaml")
    assert simulation["hardware"]["enabled"] is False
    assert simulation["hardware"]["live_output_implemented"] is False
    assert simulation["hardware"]["start_at_rest"] is True
    assert simulation["hardware"]["return_to_rest_on_exit"] is True
    assert simulation["model"]["rest_q_rad"] == config["model"]["rest_q_rad"]
    assert simulation["model"]["rest_q_rad"] == [0.0] * 6
    for key in (
        "expected_names",
        "max_source_step_rad",
        "signs",
        "scales",
        "source_deadband_rad",
        "source_filter",
    ):
        assert simulation["cad"][key] == config["cad"][key]
    for key in ("minimum_delta_rad", "maximum_delta_rad", "max_command_step_rad"):
        assert simulation["commissioning"][key] == config["commissioning"][key]
    assert simulation["control"] == config["control"]


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("hardware", "control_gripper", True, "gripper"),
        ("hardware", "require_explicit_enable", False, "require_explicit_enable"),
        ("hardware", "max_demo_duration_s", 16.0, "15 seconds"),
        ("hardware", "max_feedback_error_rad", 0.081, "0.08"),
        ("cad", "require_root_locked", False, "root_locked"),
        ("cad", "scales", [0.11] * 5, "scales"),
        ("cad", "source_filter", {"enabled": False}, "source_filter"),
        (
            "commissioning",
            "require_isolated_joint_while_pending",
            False,
            "require_isolated_joint",
        ),
        ("hardware", "hold_to_run", {"enabled": False}, "hold_to_run.enabled"),
        ("control", "loop_rate_hz", 101.0, "100 Hz"),
    ],
)
def test_cad_hardware_profile_rejects_weakened_gates(
    section: str,
    key: str,
    value,
    message: str,
) -> None:
    config = deepcopy(load_config(CONFIG_PATH))
    config[section][key] = value
    with pytest.raises(CadSafetyError, match=message):
        validate_cad_hardware_config(config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("backend", "terminal_latch", "linux_evdev_key"),
        ("require_explicit_device", False, "require_explicit_device"),
        ("require_initial_release", False, "require_initial_release"),
        ("key_code", True, "key code"),
        ("key_code", 768, "key code"),
    ],
)
def test_cad_hardware_profile_rejects_weakened_hold_to_run(
    key: str,
    value,
    message: str,
) -> None:
    config = deepcopy(load_config(CONFIG_PATH))
    config["hardware"]["hold_to_run"][key] = value
    with pytest.raises(CadSafetyError, match=message):
        validate_cad_hardware_config(config)


def test_cad_live_token_is_distinct_and_duration_is_capped() -> None:
    assert cad_live_confirmation_token("192.168.1.2") == (
        "LIVE-WIDOWXAI-CAD-192.168.1.2"
    )
    assert resolve_cad_duration(15.0, 0.0) == pytest.approx(15.0)
    assert resolve_cad_duration(15.0, 5.0) == pytest.approx(5.0)
    assert resolve_cad_duration(15.0, 30.0) == pytest.approx(15.0)


def test_physical_deadman_must_be_readable_and_released_at_every_live_gate() -> None:
    class Control:
        def __init__(self, sample=None, error=None) -> None:
            self._sample = sample
            self._error = error

        def sample(self):
            if self._error is not None:
                raise self._error
            return self._sample

    released = HoldToRunSample(False, 10, 0, 0)
    assert require_released_hold_to_run(Control(released), "before connect") == released
    with pytest.raises(HardwareSafetyError, match="released before connect"):
        require_released_hold_to_run(
            Control(HoldToRunSample(True, 20, 1, 0)),
            "before connect",
        )
    with pytest.raises(HardwareSafetyError, match="disconnected"):
        require_released_hold_to_run(
            Control(error=HoldToRunError("deadman disconnected")),
            "before connect",
        )
    with pytest.raises(CadSafetyError):
        resolve_cad_duration(15.0, -1.0)


def test_pending_live_mapping_requires_exactly_one_commissioning_joint() -> None:
    with pytest.raises(CadSafetyError, match="accept-unvalidated"):
        validate_cad_commissioning_selection(
            live=True,
            mapping_status=CAD_MAPPING_PENDING,
            accept_unvalidated_mapping=False,
            commission_joint=2,
        )
    with pytest.raises(CadSafetyError, match="commission-joint"):
        validate_cad_commissioning_selection(
            live=True,
            mapping_status=CAD_MAPPING_PENDING,
            accept_unvalidated_mapping=True,
            commission_joint=None,
        )
    assert validate_cad_commissioning_selection(
        live=True,
        mapping_status=CAD_MAPPING_PENDING,
        accept_unvalidated_mapping=True,
        commission_joint=2,
    ) == 2
    assert validate_cad_commissioning_selection(
        live=False,
        mapping_status=CAD_MAPPING_PENDING,
        accept_unvalidated_mapping=False,
        commission_joint=None,
    ) is None
    assert validate_cad_commissioning_selection(
        live=True,
        mapping_status=CAD_MAPPING_ACCEPTED,
        accept_unvalidated_mapping=False,
        commission_joint=None,
    ) is None


def test_cad_start_requires_rest_and_command_gate_anchors_to_home() -> None:
    rest = np.zeros(6)
    require_initial_rest(hardware_state(np.full(6, 0.01)), rest, 0.08)
    with pytest.raises(HardwareSafetyError, match="all-zero rest"):
        require_initial_rest(hardware_state(np.array([0.0, 0.081, 0.0, 0.0, 0.0, 0.0])), rest, 0.08)

    state = hardware_state(HOME_Q + 0.005)
    config = load_config(CONFIG_PATH)
    limits = rest_command_limits(
        state.joint_limits[:6],
        HOME_Q,
        np.asarray(config["commissioning"]["minimum_delta_rad"]),
        np.asarray(config["commissioning"]["maximum_delta_rad"]),
    )
    gate = make_cad_command_gate(
        state,
        HOME_Q,
        limits,
        np.asarray(config["commissioning"]["max_command_step_rad"]),
    )
    # Feedback may lag home by more than one per-tick cap; the gate tracks the
    # last commanded home pose, not delayed encoder feedback.
    gate.validate(HOME_Q, state.gripper_position_m)
    with pytest.raises(HardwareSafetyError, match="per-tick"):
        gate.validate(HOME_Q + np.array([0.004, 0, 0, 0, 0, 0]), 0.0)


def test_cad_telemetry_declares_source_command_and_feedback_evidence() -> None:
    required = {
        "cad_sequence",
        "cad_root_locked",
        "receiver_invalid_packets",
        "deadman_needs_release",
        "deadman_source",
        "deadman_device",
        "deadman_key_code",
        "deadman_state_age_ms",
        "deadman_press_generation",
        "deadman_release_generation",
        "commission_joint",
        "command_send_duration_ms",
        "feedback_reference_state",
        "q_source",
        "q_source_filtered",
        "source_filter_cutoff_hz",
        "q_cmd",
        "q_feedback",
        "q_feedback_error",
        "fault_reason",
    }
    assert required <= set(CAD_TELEMETRY_COLUMNS)


def test_cad_hardware_launcher_help_works_with_clean_pythonpath() -> None:
    project = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "scripts/run_cad_hardware.py", "--help"],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--accept-unvalidated-mapping" in result.stdout
    assert "--commission-joint" in result.stdout
    assert "--deadman-device" in result.stdout
    assert "--deadman-key-code" in result.stdout


def test_cad_sim_launcher_exposes_the_same_joint_selector() -> None:
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_cad_sim.py", "--help"],
        cwd=project,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--commission-joint" in result.stdout


def test_cad_sim_viewer_keys_control_the_simulation_deadman() -> None:
    console = SimulationDeadmanConsole()
    console.handle_viewer_key(ord("E"))
    assert console.pressed
    assert console.press_generation == 1
    console.handle_viewer_key(ord("R"))
    assert not console.pressed
    assert console.release_generation == 1
    console.handle_viewer_key(ord("Q"))
    assert console.quit_requested


def test_cad_sim_headless_starts_and_returns_at_all_zero_rest() -> None:
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_cad_sim.py",
            "--config",
            "configs/cad_home_commissioning_mujoco.yaml",
            "--commission-joint",
            "0",
            "--headless",
            "--udp-port",
            "0",
            "--duration",
            "0.05",
            "--no-telemetry",
        ],
        cwd=project,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "starting and ending at rest q=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]" in result.stdout
    assert "rest reached: q=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]" in result.stdout


def test_pending_live_cli_cannot_bypass_isolated_joint_selection() -> None:
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_cad_hardware.py",
            "--live",
            "--confirm-live",
            "LIVE-WIDOWXAI-CAD-192.168.1.2",
            "--accept-unvalidated-mapping",
            "--duration",
            "1",
            "--no-telemetry",
        ],
        cwd=project,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert "pending CAD mapping can move only one follower joint" in (
        result.stdout + result.stderr
    )
    assert "workstation preflight" not in (result.stdout + result.stderr)


def test_pending_live_cli_cannot_use_the_terminal_latch_as_deadman() -> None:
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_cad_hardware.py",
            "--live",
            "--confirm-live",
            "LIVE-WIDOWXAI-CAD-192.168.1.2",
            "--accept-unvalidated-mapping",
            "--commission-joint",
            "0",
            "--duration",
            "1",
            "--no-telemetry",
        ],
        cwd=project,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "requires --deadman-device" in output
    assert "workstation preflight" not in output


def test_hold_to_run_checker_is_read_only_and_has_help() -> None:
    project = Path(__file__).resolve().parents[1]
    script = project / "scripts/check_hold_to_run.py"
    source = script.read_text(encoding="utf-8")
    assert "TrossenArmBackend" not in source
    assert "trossen_arm" not in source
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=project,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--device" in result.stdout
    assert "--key-code" in result.stdout


def test_stationary_cad_stream_check_passes_clean_data_and_rejects_drift() -> None:
    samples = [
        CadJointSample(
            sequence=index,
            source_time_ns=1_000_000_000 + index * 20_000_000,
            arrival_monotonic_ns=2_000_000_000 + index * 20_000_000,
            arrival_epoch_ns=1_002_000_000 + index * 20_000_000,
            q=np.array([0.001 * (-1) ** index, 0.0, 0.0, 0.0, 0.0]),
            root_locked=True,
        )
        for index in range(50)
    ]
    clean = summarize_cad_samples(
        samples,
        duration_s=1.0,
        source_deadband_rad=np.full(5, 0.01),
        discontinuities=0,
        invalid_packets=0,
        final_fresh=True,
    )
    assert clean["passed"]
    assert clean["effective_rate_hz"] == pytest.approx(50.0)
    assert clean["sequence_gaps"] == 0

    drifting = list(samples)
    drifting[-10:] = [
        CadJointSample(
            sequence=40 + index,
            source_time_ns=1_800_000_000 + index * 20_000_000,
            arrival_monotonic_ns=2_800_000_000 + index * 20_000_000,
            arrival_epoch_ns=1_802_000_000 + index * 20_000_000,
            q=np.array([0.05, 0.0, 0.0, 0.0, 0.0]),
            root_locked=True,
        )
        for index in range(10)
    ]
    failed = summarize_cad_samples(
        drifting,
        duration_s=1.0,
        source_deadband_rad=np.full(5, 0.01),
        discontinuities=0,
        invalid_packets=0,
        final_fresh=True,
    )
    assert not failed["passed"]
    assert "configured deadband" in " ".join(failed["failures"])
