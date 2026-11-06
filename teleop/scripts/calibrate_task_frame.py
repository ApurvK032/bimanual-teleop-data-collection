from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from widowxai_quest_teleop.config import load_config, resolve_project_path
from widowxai_quest_teleop.math3d import (
    matrix_to_rotation_vector,
    normalize_quat,
    quat_to_matrix,
)
from widowxai_quest_teleop.task_frame_calibration import (
    build_calibration_document,
    direction_map_quality,
    fit_direction_map,
    heading_correction,
)
from widowxai_quest_teleop.transport import BimanualQuestReceiver, QuestReceiver
from widowxai_quest_teleop.types import Pose, QuestSample


@dataclass(frozen=True)
class Gesture:
    key: str
    kind: str
    instruction: str
    desired_robot_axis: np.ndarray


GESTURES = (
    Gesture(
        "right",
        "translation",
        "RIGHT: move 10-15 cm straight toward your right; keep rotation fixed",
        np.array([0.0, -1.0, 0.0]),
    ),
    Gesture(
        "up",
        "translation",
        "UP: move 10-15 cm straight upward; keep controller rotation fixed",
        np.array([0.0, 0.0, 1.0]),
    ),
    Gesture(
        "forward",
        "translation",
        "FORWARD: move 10-15 cm in the direction the robot tool should call forward; keep rotation fixed",
        np.array([1.0, 0.0, 0.0]),
    ),
    Gesture(
        "screw_clockwise",
        "rotation",
        "SCREW: point forward, then roll 30-60 degrees clockwise as viewed from behind",
        np.array([1.0, 0.0, 0.0]),
    ),
    Gesture(
        "nod_yes_down",
        "rotation",
        "NOD YES: point forward, then tilt its front end 30-60 degrees downward",
        np.array([0.0, 1.0, 0.0]),
    ),
    Gesture(
        "nod_no_left",
        "rotation",
        "NOD NO: point forward, then turn its front end 30-60 degrees toward robot-left",
        np.array([0.0, 0.0, 1.0]),
    ),
)


class SampleStream:
    def __init__(self, receiver: QuestReceiver) -> None:
        self.receiver = receiver
        self.generation = 0
        self.expected_hand: str | None = None
        self.expected_mapping_mode: str | None = None

    def lock_selection(self, sample: QuestSample) -> None:
        self.expected_hand = sample.hand
        self.expected_mapping_mode = sample.mapping_mode

    def next(self, timeout_s: float = 0.5) -> QuestSample | None:
        sample, self.generation = self.receiver.mailbox.wait_take_latest(
            self.generation,
            timeout_s,
        )
        if (
            sample is not None
            and self.expected_hand is not None
            and (
                sample.hand != self.expected_hand
                or sample.mapping_mode != self.expected_mapping_mode
            )
        ):
            raise ValueError(
                "Quest hand/mapping selection changed during calibration; "
                "restart with one fixed selection"
            )
        return sample


class BimanualHandSampleStream:
    """Project one fixed hand from schema-v2 bimanual packets for calibration."""

    def __init__(
        self,
        receiver: BimanualQuestReceiver,
        *,
        hand: str,
        mapping_mode: str,
    ) -> None:
        self.receiver = receiver
        self.hand = hand
        self.mapping_mode = mapping_mode
        self.generation = 0

    def lock_selection(self, sample: QuestSample) -> None:
        if sample.hand != self.hand or sample.mapping_mode != self.mapping_mode:
            raise ValueError(
                "bimanual calibration received an unexpected hand or mapping mode"
            )

    def next(self, timeout_s: float = 0.5) -> QuestSample | None:
        sample, self.generation = self.receiver.mailbox.wait_take_latest(
            self.generation,
            timeout_s,
        )
        if sample is None:
            return None
        controller = sample.controller(self.hand)
        if controller is None or not controller.tracked:
            return None
        if controller.mapping_mode != self.mapping_mode:
            raise ValueError(
                f"Quest {self.hand} mapping changed from {self.mapping_mode} "
                f"to {controller.mapping_mode} during calibration"
            )
        return sample.for_hand(self.hand)


def average_pose(samples: list[QuestSample]) -> tuple[Pose, np.ndarray]:
    if not samples:
        raise ValueError("no Quest samples were captured")
    position = np.mean([sample.controller_pose.position for sample in samples], axis=0)
    reference = samples[0].controller_pose.quaternion_wxyz
    quaternions = []
    for sample in samples:
        quaternion = sample.controller_pose.quaternion_wxyz
        quaternions.append(quaternion if np.dot(reference, quaternion) >= 0.0 else -quaternion)
    controller_quaternion = normalize_quat(np.mean(quaternions, axis=0))
    heads = [sample.head_quaternion_wxyz for sample in samples if sample.head_quaternion_wxyz is not None]
    if not heads:
        raise ValueError("Quest headset orientation is missing")
    head_reference = heads[0]
    aligned_heads = [head if np.dot(head_reference, head) >= 0.0 else -head for head in heads]
    head_quaternion = normalize_quat(np.mean(aligned_heads, axis=0))
    return Pose(position, controller_quaternion), head_quaternion


def publish_prompt(
    receiver: QuestReceiver | BimanualQuestReceiver,
    text: str,
    *,
    pulse_ms: int = 0,
    hand: str | None = None,
) -> None:
    receiver.publish({"type": "calibration_prompt", "text": text})
    if pulse_ms:
        payload: dict[str, object] = {
            "type": "haptic",
            "intensity": 0.75,
            "duration_ms": pulse_ms,
        }
        if hand is not None:
            payload["hand"] = hand
        receiver.publish(payload)


def wait_for_release(
    stream: SampleStream | BimanualHandSampleStream,
    timeout_s: float = 20.0,
) -> None:
    deadline = time.perf_counter() + timeout_s
    released = 0
    while time.perf_counter() < deadline:
        sample = stream.next()
        if sample is None:
            continue
        released = released + 1 if sample.grip <= 0.35 else 0
        if released >= 5:
            return
    raise TimeoutError("selected controller grip was not released")


def wait_for_press(
    stream: SampleStream | BimanualHandSampleStream,
    timeout_s: float = 120.0,
) -> QuestSample:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        sample = stream.next()
        if sample is not None and sample.grip >= 0.70:
            return sample
    raise TimeoutError("selected controller grip was not pressed")


def wait_for_tracked_sample(
    stream: SampleStream | BimanualHandSampleStream,
    timeout_s: float,
) -> QuestSample | None:
    """Wait through explicit bimanual tracking-loss frames for a real pose."""

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        sample = stream.next(min(0.5, deadline - time.perf_counter()))
        if sample is not None:
            return sample
    return None


def capture_gesture(
    stream: SampleStream | BimanualHandSampleStream,
    receiver: QuestReceiver | BimanualQuestReceiver,
    gesture: Gesture,
    repeat: int,
    repeats: int,
    hand: str,
) -> tuple[np.ndarray, dict[str, object]]:
    while True:
        wait_for_release(stream)
        prompt = f"{gesture.instruction} — capture {repeat}/{repeats}"
        print(f"\n{prompt}")
        print(
            f"Hold still at the start, squeeze and HOLD {hand} grip, wait for the pulse, "
            "move, hold the endpoint, release grip."
        )
        publish_prompt(receiver, prompt)
        wait_for_press(stream)

        baseline: deque[QuestSample] = deque(maxlen=45)
        settle_deadline = time.perf_counter() + 0.65
        released_early = False
        while time.perf_counter() < settle_deadline:
            sample = stream.next(0.25)
            if sample is None:
                continue
            if sample.grip < 0.55:
                released_early = True
                break
            baseline.append(sample)
        if released_early or len(baseline) < 12:
            print("Capture restarted: hold grip through the initial settling period.")
            continue

        baseline_samples = list(baseline)[-20:]
        start_pose, head_quaternion = average_pose(baseline_samples)
        publish_prompt(
            receiver,
            f"GO — {gesture.instruction}",
            pulse_ms=140,
            hand=hand,
        )
        print("GO")

        endpoint: deque[QuestSample] = deque(maxlen=90)
        motion_started = time.perf_counter()
        timed_out = False
        while True:
            if time.perf_counter() - motion_started > 10.0:
                timed_out = True
                break
            sample = stream.next(0.25)
            if sample is None:
                continue
            if sample.grip <= 0.35:
                break
            endpoint.append(sample)
        if timed_out or len(endpoint) < 20 or time.perf_counter() - motion_started < 0.8:
            print("Capture restarted: move after the pulse, hold the endpoint, then release within 10 seconds.")
            continue

        endpoint_samples = list(endpoint)[-20:]
        end_pose, _ = average_pose(endpoint_samples)
        correction = heading_correction(head_quaternion)
        position_delta_world = end_pose.position - start_pose.position
        controller_increment = (
            quat_to_matrix(end_pose.quaternion_wxyz)
            @ quat_to_matrix(start_pose.quaternion_wxyz).T
        )
        rotation_delta_world = matrix_to_rotation_vector(controller_increment)
        position_delta = correction @ position_delta_world
        rotation_delta = correction @ rotation_delta_world
        translation_m = float(np.linalg.norm(position_delta))
        rotation_rad = float(np.linalg.norm(rotation_delta))

        if gesture.kind == "translation":
            if translation_m < 0.06:
                print(f"Capture restarted: translation was only {translation_m * 100:.1f} cm; use 10-15 cm.")
                continue
            if translation_m > 0.30:
                print("Capture restarted: translation exceeded 30 cm; use only 10-15 cm.")
                continue
            if rotation_rad > 0.45:
                print(f"Capture restarted: unintended rotation was {np.degrees(rotation_rad):.1f} degrees.")
                continue
            observed = position_delta
        else:
            if rotation_rad < np.radians(20.0):
                print(f"Capture restarted: rotation was only {np.degrees(rotation_rad):.1f} degrees; use 30-60 degrees.")
                continue
            if rotation_rad > np.radians(80.0):
                print("Capture restarted: rotation exceeded 80 degrees; use only 30-60 degrees.")
                continue
            if translation_m > 0.12:
                print(f"Capture restarted: unintended translation was {translation_m * 100:.1f} cm.")
                continue
            observed = rotation_delta

        details: dict[str, object] = {
            "gesture": gesture.key,
            "kind": gesture.kind,
            "repeat": repeat,
            "observed_operator_vector": observed.tolist(),
            "desired_robot_axis": gesture.desired_robot_axis.tolist(),
            "translation_distance_m": translation_m,
            "rotation_angle_deg": float(np.degrees(rotation_rad)),
        }
        print(
            f"Accepted: translation={translation_m * 100:.1f} cm, "
            f"rotation={np.degrees(rotation_rad):.1f} deg"
        )
        publish_prompt(
            receiver,
            f"Accepted {gesture.key}",
            pulse_ms=60,
            hand=hand,
        )
        return observed, details


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit Quest-to-WidowXAI translation and rotation axes without connecting to the arm"
    )
    parser.add_argument("--config", default="configs/smooth_30pct_full_gripper.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/calibrations/left_guided_6dof.json"),
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--bimanual-hand",
        choices=("left", "right"),
        help="calibrate one fixed hand from the schema-v2 bimanual Quest stream",
    )
    parser.add_argument(
        "--mapping-mode",
        choices=("real", "mirror"),
        help="required page mapping for --bimanual-hand",
    )
    parser.add_argument(
        "--simulation-only",
        action="store_true",
        help="mark the output as MuJoCo-only and explicitly unaccepted for hardware",
    )
    args = parser.parse_args()
    if args.repeats < 2 or args.repeats > 5:
        raise SystemExit("--repeats must be between 2 and 5")
    if (args.bimanual_hand is None) != (args.mapping_mode is None):
        raise SystemExit("--bimanual-hand and --mapping-mode must be supplied together")
    if args.bimanual_hand is not None and not args.simulation_only:
        raise SystemExit(
            "bimanual guided calibration currently requires --simulation-only"
        )

    output = resolve_project_path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {output}; pass --overwrite to replace it")
    config = load_config(args.config)
    if args.bimanual_hand is None:
        receiver = QuestReceiver(
            config["quest"]["websocket_url"],
            hand=None,
            mapping_mode=None,
        )
        stream: SampleStream | BimanualHandSampleStream = SampleStream(receiver)
    else:
        receiver = BimanualQuestReceiver(
            config["quest"]["websocket_url"],
            mapping_modes={args.bimanual_hand: args.mapping_mode},
        )
        stream = BimanualHandSampleStream(
            receiver,
            hand=args.bimanual_hand,
            mapping_mode=args.mapping_mode,
        )
    receiver.start()
    print("QUEST-ONLY CALIBRATION: this process does not import a robot driver or send arm commands.")
    print("Stop every run_hardware.py process. Face the direction that should be robot-forward.")
    if args.bimanual_hand is None:
        print("Choose the controller and Real/Mirror mode on http://localhost:8443/.")
    else:
        print(
            f"BIMANUAL INPUT: calibrating only the {args.bimanual_hand} controller "
            f"in {args.mapping_mode} mode; the other controller is ignored."
        )
        print("The output is marked MuJoCo-only and is not accepted for hardware.")
    print("Six gestures will be captured twice.")

    hand = "page-selected"
    mapping_mode = "page-selected"
    saved_hand: str | None = None
    saved_mapping_mode: str | None = None

    position_observed: list[np.ndarray] = []
    position_desired: list[np.ndarray] = []
    rotation_observed: list[np.ndarray] = []
    rotation_desired: list[np.ndarray] = []
    captures: list[dict[str, object]] = []
    capture_log = output.with_suffix(".captures.json")

    if capture_log.exists():
        saved = json.loads(capture_log.read_text(encoding="utf-8"))
        saved_hand = saved.get("hand")
        saved_mapping_mode = saved.get("mapping_mode")
        saved_output = resolve_project_path(saved.get("output", output))
        if saved_output != output:
            raise SystemExit(
                f"capture log belongs to a different output: {saved_output}"
            )
        captures = list(saved.get("captures", []))
        position_observed = [
            np.asarray(value, dtype=float) for value in saved.get("position_observed", [])
        ]
        position_desired = [
            np.asarray(value, dtype=float) for value in saved.get("position_desired", [])
        ]
        rotation_observed = [
            np.asarray(value, dtype=float) for value in saved.get("rotation_observed", [])
        ]
        rotation_desired = [
            np.asarray(value, dtype=float) for value in saved.get("rotation_desired", [])
        ]
        print(
            "Resuming saved calibration: "
            f"{len(position_observed)} translation and "
            f"{len(rotation_observed)} rotation captures already accepted."
        )

    def save_capture_log() -> None:
        capture_log.parent.mkdir(parents=True, exist_ok=True)
        capture_log.write_text(
            json.dumps(
                {
                    "output": str(output),
                    "hand": hand,
                    "mapping_mode": mapping_mode,
                    "captures": captures,
                    "position_observed": [value.tolist() for value in position_observed],
                    "position_desired": [value.tolist() for value in position_desired],
                    "rotation_observed": [value.tolist() for value in rotation_observed],
                    "rotation_desired": [value.tolist() for value in rotation_desired],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def repeated_capture(gesture: Gesture) -> list[tuple[np.ndarray, dict[str, object]]]:
        accepted = [
            (np.asarray(item["observed_operator_vector"], dtype=float), item)
            for item in captures
            if item.get("gesture") == gesture.key and item.get("kind") == gesture.kind
        ]
        existing_count = len(accepted)
        if existing_count > args.repeats:
            raise ValueError(f"capture log has too many {gesture.key} samples")
        for repeat in range(existing_count + 1, args.repeats + 1):
            while True:
                observed, details = capture_gesture(
                    stream,
                    receiver,
                    gesture,
                    repeat,
                    args.repeats,
                    hand,
                )
                if accepted:
                    previous = accepted[0][0] / np.linalg.norm(accepted[0][0])
                    current = observed / np.linalg.norm(observed)
                    separation = float(
                        np.degrees(np.arccos(np.clip(np.dot(previous, current), -1.0, 1.0)))
                    )
                    if separation > 25.0:
                        print(
                            f"Capture restarted: repeat direction differs by {separation:.1f} degrees; "
                            "perform the same signed motion as the first capture."
                        )
                        continue
                    details["repeat_axis_separation_deg"] = separation
                accepted.append((observed, details))
                break
        return accepted[existing_count:]

    try:
        first_timeout_s = 120.0 if args.bimanual_hand is not None else 10.0
        first = wait_for_tracked_sample(stream, first_timeout_s)
        if first is None:
            raise SystemExit("no fresh Quest poses arrived; start localhost and enter passthrough first")
        stream.lock_selection(first)
        hand = first.hand
        mapping_mode = first.mapping_mode
        if saved_hand is not None and saved_hand != hand:
            raise ValueError(
                f"capture log uses {saved_hand} controller, page selected {hand}"
            )
        if (
            saved_mapping_mode is not None
            and saved_mapping_mode != mapping_mode
        ):
            raise ValueError(
                "capture log uses mapping mode "
                f"{saved_mapping_mode}, page selected {mapping_mode}"
            )
        print(f"Selected input: {hand} controller, {mapping_mode} mapping.")
        task_frame_determinant = -1 if mapping_mode == "mirror" else 1
        translation_gestures = [gesture for gesture in GESTURES if gesture.kind == "translation"]
        rotation_gestures = [gesture for gesture in GESTURES if gesture.kind == "rotation"]
        for gesture in translation_gestures:
            for observed, details in repeated_capture(gesture):
                captures.append(details)
                position_observed.append(observed)
                position_desired.append(gesture.desired_robot_axis)
                save_capture_log()

        position_matrix, position_quality = fit_direction_map(
            np.asarray(position_observed),
            np.asarray(position_desired),
            mapping_determinant=task_frame_determinant,
        )
        if position_quality["max_axis_error_deg"] > 25.0:
            raise ValueError("translation calibration is inconsistent; capture log was preserved")
        print(
            "\nTranslation frame accepted: "
            f"median error {position_quality['median_axis_error_deg']:.1f} deg, "
            f"max {position_quality['max_axis_error_deg']:.1f} deg"
        )

        for gesture in rotation_gestures:
            accepted = [
                (np.asarray(item["observed_operator_vector"], dtype=float), item)
                for item in captures
                if item.get("gesture") == gesture.key and item.get("kind") == gesture.kind
            ]
            existing_count = len(accepted)
            if existing_count > args.repeats:
                raise ValueError(f"capture log has too many {gesture.key} samples")
            for repeat in range(existing_count + 1, args.repeats + 1):
                while True:
                    observed, details = capture_gesture(
                        stream,
                        receiver,
                        gesture,
                        repeat,
                        args.repeats,
                        hand,
                    )
                    quality = direction_map_quality(
                        position_matrix,
                        observed.reshape(1, 3),
                        gesture.desired_robot_axis.reshape(1, 3),
                        axial_vectors=True,
                    )
                    axis_error = float(quality["max_axis_error_deg"])
                    if axis_error > 30.0:
                        print(
                            f"Capture restarted: {gesture.key} axis is {axis_error:.1f} degrees from "
                            "the calibrated task axis. Start with the controller pointing forward."
                        )
                        continue
                    if accepted:
                        previous = accepted[0][0] / np.linalg.norm(accepted[0][0])
                        current = observed / np.linalg.norm(observed)
                        separation = float(
                            np.degrees(np.arccos(np.clip(np.dot(previous, current), -1.0, 1.0)))
                        )
                        if separation > 25.0:
                            print(
                                f"Capture restarted: repeat direction differs by {separation:.1f} degrees."
                            )
                            continue
                        details["repeat_axis_separation_deg"] = separation
                    details["linked_task_axis_error_deg"] = axis_error
                    accepted.append((observed, details))
                    break
            for observed, details in accepted[existing_count:]:
                captures.append(details)
                rotation_observed.append(observed)
                rotation_desired.append(gesture.desired_robot_axis)
                save_capture_log()

        document = build_calibration_document(
            name=output.stem,
            position_observed=np.asarray(position_observed),
            position_desired=np.asarray(position_desired),
            rotation_observed=np.asarray(rotation_observed),
            rotation_desired=np.asarray(rotation_desired),
            captures=captures,
            link_rotation_to_position=True,
            position_mapping_determinant=task_frame_determinant,
        )
        document["quest_input"] = {
            "hand": hand,
            "mapping_mode": mapping_mode,
        }
        if args.simulation_only:
            document["validation_scope"] = {
                "status": "simulation_only",
                "allowed_runtime": "mujoco",
                "hardware_accepted": False,
                "note": "Captured for MuJoCo mapping; do not use for physical robot output.",
            }
            document["physical_validation"] = {
                "status": "simulation_only",
                "hardware_motion_performed": False,
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        position_quality = document["guided_calibration"]["position_quality"]
        rotation_quality = document["guided_calibration"]["rotation_quality"]
        print(f"\nSaved: {output}")
        print(
            "Translation fit: "
            f"median {position_quality['median_axis_error_deg']:.1f} deg, "
            f"max {position_quality['max_axis_error_deg']:.1f} deg"
        )
        print(
            "Rotation fit: "
            f"median {rotation_quality['median_axis_error_deg']:.1f} deg, "
            f"max {rotation_quality['max_axis_error_deg']:.1f} deg"
        )
        print("Do not use it on hardware yet; inspect and validate all six axes in MuJoCo first.")
        publish_prompt(
            receiver,
            "Six-axis calibration saved; validate in MuJoCo",
            pulse_ms=180,
            hand=hand,
        )
    except (TimeoutError, ValueError) as exc:
        raise SystemExit(f"calibration failed: {exc}") from None
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
