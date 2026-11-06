from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from widowxai_quest_teleop.clutch import ClutchController
from widowxai_quest_teleop.config import load_config
from widowxai_quest_teleop.decoupled_ik import DecoupledIK
from widowxai_quest_teleop.gripper import trigger_to_gripper_position
from widowxai_quest_teleop.mapping import ClutchPoseMapper
from widowxai_quest_teleop.model import WidowXAIModel
from widowxai_quest_teleop.motion_limiter import bounded_command_period, limiter_from_config
from widowxai_quest_teleop.pose_filter import ControllerPoseFilter
from widowxai_quest_teleop.telemetry import TelemetryLogger
from widowxai_quest_teleop.types import Pose


def decode_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recorded Quest input through the decoupled backend")
    parser.add_argument("recording", type=Path, help="telemetry.csv from a live Quest run")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--label", default="recording-replay-decoupled")
    args = parser.parse_args()

    config = load_config(args.config)
    quest = config["quest"]
    model = WidowXAIModel(config["model"]["xml_path"])
    solver = DecoupledIK.from_config(model, config)
    q_start = model.clamp_joints(np.asarray(config["model"]["simulation_start_q_rad"], dtype=float))
    mapper = ClutchPoseMapper(
        quest["calibration"],
        translation_scale=quest["translation_scale"],
        rotation_scale=quest["rotation_scale"],
        position_reach_limit_m=quest["position_reach_limit_m"],
        rotation_reach_limit_rad=quest["rotation_reach_limit_rad"],
    )
    clutch = ClutchController(mapper)
    sim_data = mujoco.MjData(model.model)
    hardware = config.get("hardware", {})
    gripper_open = float(hardware.get("gripper_open_m", 0.044))
    gripper_closed = float(hardware.get("gripper_min_demo_m", 0.0))
    gripper_q = gripper_open
    model.set_viewer_qpos(sim_data, q_start, gripper_q)
    q_des = q_start.copy()
    q_command = q_start.copy()
    joint_limiter = limiter_from_config(
        config["control"].get("joint_command_limits"),
        q_command,
    )
    gripper_limiter = limiter_from_config(
        config["control"].get("gripper_command_limits"),
        np.array([gripper_q]),
    )
    filtered_controller = None
    loop_hz = float(config["control"]["loop_rate_hz"])
    quest_synchronized = config["control"].get("update_mode") == "quest_synchronized"
    controller_filter = ControllerPoseFilter(config["control"])
    previous_sequence = None
    previous_capture_ms = None
    replayed = 0

    with args.recording.open("r", newline="", encoding="utf-8") as source, TelemetryLogger(
        args.label, config, config["telemetry"]["output_dir"]
    ) as telemetry:
        for row in csv.DictReader(source):
            if not row.get("quest_sequence") or not row.get("raw_controller_position"):
                continue
            sequence = int(row["quest_sequence"])
            if sequence == previous_sequence:
                continue
            capture_ms = float(row["quest_capture_monotonic_ms"])
            dt = 0.01 if previous_capture_ms is None else np.clip((capture_ms - previous_capture_ms) / 1000.0, 0.001, 0.05)
            previous_capture_ms = capture_ms
            previous_sequence = sequence

            controller = Pose(
                decode_array(row["raw_controller_position"]),
                decode_array(row["raw_controller_quaternion_wxyz"]),
            )
            head_quaternion = (
                decode_array(row["head_quaternion_wxyz"])
                if row.get("head_quaternion_wxyz")
                else None
            )
            diagnostics = None
            target = None
            limiter_flags: list[str] = []
            iterations = 1 if quest_synchronized else max(1, round(float(dt) * loop_hz))
            step_dt = bounded_command_period(float(dt) / iterations, loop_hz)
            for _ in range(iterations):
                filtered_controller = controller_filter.update(
                    controller,
                    capture_ms / 1000.0,
                )
                robot_pose, wrist_pose = model.fk(q_command)
                target = clutch.update(
                    grip=float(row.get("quest_grip") or 0.0),
                    stream_fresh=row.get("stream_fresh", "True").lower() == "true",
                    controller_pose=filtered_controller,
                    robot_pose=robot_pose,
                    wrist_pivot=wrist_pose.position,
                    head_quaternion_wxyz=head_quaternion,
                )
                if target is not None:
                    q_des, diagnostics = solver.solve(target, q_command)
                    if joint_limiter is None:
                        q_command = q_des.copy()
                    else:
                        joint_result = joint_limiter.step(q_des, step_dt)
                        q_command = joint_result.command
                        limiter_flags.extend(joint_result.flags("joint"))
            if mapper.engaged:
                gripper_des = trigger_to_gripper_position(
                    float(row.get("quest_trigger") or 0.0),
                    gripper_open,
                    gripper_closed,
                )
                if gripper_limiter is None:
                    gripper_q = gripper_des
                else:
                    gripper_result = gripper_limiter.step(
                        np.array([gripper_des]),
                        bounded_command_period(float(dt), loop_hz),
                    )
                    gripper_q = float(gripper_result.command[0])
                    limiter_flags.extend(gripper_result.flags("gripper"))
            else:
                filtered_controller = None
                controller_filter.reset()
                gripper_des = gripper_q
                if joint_limiter is not None:
                    joint_limiter.reset(q_command)
                if gripper_limiter is not None:
                    gripper_limiter.reset(np.array([gripper_q]))
            model.set_viewer_qpos(sim_data, q_command, gripper_q)
            telemetry.log(
                pc_epoch_ns=time.time_ns(),
                pc_monotonic_ns=time.perf_counter_ns(),
                quest_sequence=sequence,
                quest_capture_monotonic_ms=capture_ms,
                quest_grip=float(row.get("quest_grip") or 0.0),
                quest_trigger=float(row.get("quest_trigger") or 0.0),
                stream_fresh=True,
                clutch_engaged=mapper.engaged,
                reanchor_generation=mapper.reanchor_generation,
                raw_controller_position=controller.position,
                raw_controller_quaternion_wxyz=controller.quaternion_wxyz,
                pose_filter_rotation_alpha=controller_filter.rotation_alpha,
                pose_filter_rotation_cutoff_hz=controller_filter.rotation_cutoff_hz,
                mapped_target_position="" if target is None else target.position,
                mapped_target_quaternion_wxyz="" if target is None else target.quaternion_wxyz,
                q_des=q_des,
                q_cmd=q_command,
                q_feedback=sim_data.qpos[model.qpos_indices].copy(),
                gripper_des_m=gripper_des,
                gripper_cmd_m=gripper_q,
                gripper_feedback_m=gripper_q,
                position_residual_m="" if diagnostics is None else diagnostics.position_residual_m,
                orientation_residual_rad="" if diagnostics is None else diagnostics.orientation_residual_rad,
                position_manipulability="" if diagnostics is None else diagnostics.position_manipulability,
                rotation_manipulability="" if diagnostics is None else diagnostics.rotation_manipulability,
                position_damping="" if diagnostics is None else diagnostics.position_damping,
                rotation_damping="" if diagnostics is None else diagnostics.rotation_damping,
                ik_step_norm_rad="" if diagnostics is None else diagnostics.step_norm_rad,
                minimum_joint_limit_margin_rad="" if diagnostics is None else diagnostics.minimum_joint_limit_margin_rad,
                ik_status="" if diagnostics is None else diagnostics.status,
                limiter_flags="|".join(limiter_flags),
            )
            replayed += 1
        run_dir = telemetry.run_dir
    print(f"replayed {replayed} unique Quest samples")
    print(f"telemetry: {run_dir}")


if __name__ == "__main__":
    main()
