from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_ROOT, resolve_project_path


TELEMETRY_COLUMNS = [
    "pc_epoch_ns",
    "pc_monotonic_ns",
    "quest_sequence",
    "quest_capture_monotonic_ms",
    "quest_capture_epoch_ms",
    "quest_send_monotonic_ms",
    "pc_socket_arrival_monotonic_ns",
    "pc_socket_arrival_epoch_ns",
    "control_consume_monotonic_ns",
    "ik_start_monotonic_ns",
    "ik_end_monotonic_ns",
    "command_send_monotonic_ns",
    "command_send_epoch_ns",
    "feedback_read_monotonic_ns",
    "feedback_sample_fresh",
    "command_spacing_wait_ms",
    "command_pre_consume_wait_ms",
    "command_pre_send_wait_ms",
    "mailbox_overwrite_count",
    "reconnect_generation",
    "quest_hand",
    "quest_mapping_mode",
    "quest_grip",
    "quest_trigger",
    "stream_fresh",
    "clutch_engaged",
    "reanchor_generation",
    "raw_controller_position",
    "raw_controller_quaternion_wxyz",
    "pose_filter_rotation_alpha",
    "pose_filter_rotation_cutoff_hz",
    "head_quaternion_wxyz",
    "engage_head_yaw_rad",
    "mapped_target_position",
    "mapped_target_quaternion_wxyz",
    "wrist_target_position",
    "wrist_current_position",
    "q_des",
    "q_cmd",
    "q_feedforward_velocity",
    "q_feedback",
    "q_feedback_reference",
    "q_feedback_error",
    "gripper_des_m",
    "gripper_cmd_m",
    "gripper_feedback_m",
    "position_residual_m",
    "orientation_residual_rad",
    "position_manipulability",
    "rotation_manipulability",
    "position_damping",
    "rotation_damping",
    "ik_step_norm_rad",
    "minimum_joint_limit_margin_rad",
    "ik_status",
    "limiter_flags",
]


# CAD input is joint-space state rather than a Quest pose.  Both CAD launchers
# use this schema with strict column checking so input faults, physical
# feedback, and the command sent to the arm remain on the same timeline.
CAD_TELEMETRY_COLUMNS = [
    "pc_epoch_ns",
    "pc_monotonic_ns",
    "cad_sequence",
    "cad_source_time_ns",
    "cad_source_age_ms",
    "cad_arrival_age_ms",
    "cad_root_locked",
    "receiver_packets",
    "receiver_valid_packets",
    "receiver_invalid_packets",
    "receiver_last_error",
    "cad_fresh",
    "recovery_streak",
    "discontinuity_generation",
    "last_discontinuity",
    "deadman_pressed",
    "deadman_engaged",
    "deadman_needs_release",
    "deadman_source",
    "deadman_device",
    "deadman_key_code",
    "deadman_state_age_ms",
    "deadman_press_generation",
    "deadman_release_generation",
    "reanchor_generation",
    "commission_joint",
    "hardware_live",
    "command_send_monotonic_ns",
    "command_send_epoch_ns",
    "command_send_duration_ms",
    "feedback_read_monotonic_ns",
    "feedback_sample_fresh",
    "feedback_reference_state",
    "feedback_newest_command_age_ms",
    "feedback_history_span_ms",
    "q_source",
    "q_source_filtered",
    "source_filter_alpha",
    "source_filter_cutoff_hz",
    "q_des",
    "q_cmd",
    "q_feedback",
    "q_feedback_reference",
    "q_feedback_error",
    "gripper_feedback_m",
    "limiter_flags",
    "event",
    "fault_reason",
]


DUAL_ARM_SHARED_COLUMNS = [
    "pc_epoch_ns",
    "pc_monotonic_ns",
    "quest_sequence",
    "quest_capture_monotonic_ms",
    "quest_capture_epoch_ms",
    "quest_send_monotonic_ms",
    "pc_socket_arrival_monotonic_ns",
    "pc_socket_arrival_epoch_ns",
    "control_consume_monotonic_ns",
    "command_spacing_wait_ms",
    "command_pre_consume_wait_ms",
    "command_pre_send_wait_ms",
    "mailbox_overwrite_count",
    "reconnect_generation",
    "head_quaternion_wxyz",
    "control_state",
    # Coordinated dual-arm state. Skew is measured between the two driver
    # sends, never assumed from the fact that both happened in one loop.
    "command_skew_ms",
    "cross_arm_collision",
    "cross_arm_collision_kind",
    "cross_arm_collision_bodies",
    "cross_arm_separation_m",
    "coordinated_hold",
    "fault_reason",
]

DUAL_ARM_PER_ARM_COLUMNS = [
    "controller_hand",
    "mapping_mode",
    "tracked",
    "grip",
    "trigger",
    "stream_fresh",
    "stream_age_s",
    "clutch_engaged",
    "reanchor_generation",
    "raw_controller_position",
    "raw_controller_quaternion_wxyz",
    "pose_filter_rotation_alpha",
    "pose_filter_rotation_cutoff_hz",
    "engage_head_yaw_rad",
    "mapped_target_position",
    "mapped_target_quaternion_wxyz",
    "ik_start_monotonic_ns",
    "ik_end_monotonic_ns",
    "command_send_monotonic_ns",
    "command_send_epoch_ns",
    "command_send_duration_ms",
    "feedback_read_monotonic_ns",
    "feedback_sample_fresh",
    "feedback_reference_state",
    "feedback_newest_command_age_ms",
    "feedback_history_span_ms",
    "q_des",
    "q_cmd",
    "q_feedforward_velocity",
    "q_feedback",
    "q_feedback_reference",
    "q_feedback_error",
    "gripper_des_m",
    "gripper_cmd_m",
    "gripper_feedback_m",
    "position_residual_m",
    "orientation_residual_rad",
    "ik_step_norm_rad",
    "minimum_joint_limit_margin_rad",
    "ik_status",
    "limiter_flags",
    "held",
    "rejected",
]

DUAL_ARM_SIDES = ("left", "right")


def dual_arm_columns(sides: tuple[str, ...] = DUAL_ARM_SIDES) -> list[str]:
    """Build the flat dual-arm column list.

    One row per control tick with side-prefixed columns keeps both arms on the
    same timeline, which is what makes skew and one-arm faults readable.
    """

    columns = list(DUAL_ARM_SHARED_COLUMNS)
    for side in sides:
        columns.extend(f"{side}_{name}" for name in DUAL_ARM_PER_ARM_COLUMNS)
    return columns


DUAL_ARM_TELEMETRY_COLUMNS = dual_arm_columns()


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), separators=(",", ":"))
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value


class TelemetryLogger:
    def __init__(
        self,
        label: str,
        config: dict[str, Any],
        output_dir: str | Path = "runs",
        *,
        columns: list[str] | None = None,
        ik_status_columns: tuple[str, ...] = ("ik_status",),
        strict_columns: bool = False,
    ) -> None:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "run"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        day = time.strftime("%Y-%m-%d")
        self.run_dir = resolve_project_path(output_dir) / day / f"{stamp}_{safe_label}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.csv_path = self.run_dir / "telemetry.csv"
        self.columns = list(TELEMETRY_COLUMNS if columns is None else columns)
        # DictWriter drops unknown keys silently, so a caller that logs a
        # column this logger does not declare loses the data without any error.
        # Safety-critical callers opt into failing loudly, because a dropped
        # column can make a fault look like a clean run. The single-arm Quest
        # default remains lenient for compatibility; CAD and dual-arm callers
        # provide their own complete schemas and enable strict checking.
        self._strict_columns = bool(strict_columns)
        self._known_columns = set(self.columns)
        self._ik_status_columns = tuple(ik_status_columns)
        self._handle = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.columns, extrasaction="ignore")
        self._writer.writeheader()
        self.rows = 0
        self.ik_failures = 0
        snapshot = dict(config)
        try:
            snapshot["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            snapshot["git_commit"] = "uncommitted-initial-state"
        (self.run_dir / "config_snapshot.json").write_text(
            json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
        )

    def log(self, **record: Any) -> None:
        if self._strict_columns:
            unknown = sorted(set(record) - self._known_columns)
            if unknown:
                raise ValueError(
                    f"telemetry received undeclared columns {unknown}; add them to the "
                    "column list instead of letting the writer discard them"
                )
        row = {name: _encode(record.get(name, "")) for name in self.columns}
        self._writer.writerow(row)
        self.rows += 1
        for column in self._ik_status_columns:
            if record.get(column) not in (None, "", "ok", "orientation_parked"):
                self.ik_failures += 1

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.flush()
        self._handle.close()
        summary = {
            "rows": self.rows,
            "ik_failures": self.ik_failures,
            "telemetry_csv": str(self.csv_path),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
