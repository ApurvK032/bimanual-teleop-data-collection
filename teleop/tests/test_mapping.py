import json

import numpy as np

from widowxai_quest_teleop.config import (
    TASK_PROFILE_NAMES,
    apply_task_profile,
    load_config,
    resolve_project_path,
)
from widowxai_quest_teleop.decoupled_ik import DecoupledIK
from widowxai_quest_teleop.mapping import ClutchPoseMapper
from widowxai_quest_teleop.math3d import (
    matrix_to_quat,
    matrix_to_rotation_vector,
    quat_to_matrix,
    rotation_vector_to_matrix,
)
from widowxai_quest_teleop.types import Pose


IDENTITY_POSE = Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))


def test_confirmed_position_matrix_maps_column_basis_vectors() -> None:
    mapper = ClutchPoseMapper(
        "configs/calibrations/left_behind_full_pose_good_20260602.json",
        position_reach_limit_m=None,
        rotation_reach_limit_rad=None,
    )
    for axis in range(3):
        mapper.engage(IDENTITY_POSE, IDENTITY_POSE, np.zeros(3))
        delta = np.zeros(3)
        delta[axis] = 0.01
        target = mapper.update(Pose(delta, IDENTITY_POSE.quaternion_wxyz), IDENTITY_POSE)
        assert target is not None
        np.testing.assert_allclose(target.position, 0.01 * mapper.position_matrix[:, axis], atol=1e-12)
        mapper.release()


def test_reengage_has_no_target_jump() -> None:
    mapper = ClutchPoseMapper(
        "configs/calibrations/left_behind_full_pose_good_20260602.json",
        position_reach_limit_m=None,
        rotation_reach_limit_rad=None,
    )
    controller = Pose(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    robot = Pose(np.array([0.2, -0.1, 0.4]), np.array([1.0, 0.0, 0.0, 0.0]))
    mapper.engage(controller, robot, np.array([0.1, -0.1, 0.4]))
    first = mapper.update(controller, robot)
    assert first is not None
    np.testing.assert_allclose(first.position, robot.position, atol=1e-12)
    mapper.release()
    mapper.engage(Pose(controller.position + 10.0, controller.quaternion_wxyz), robot, np.array([0.1, -0.1, 0.4]))
    second = mapper.update(Pose(controller.position + 10.0, controller.quaternion_wxyz), robot)
    assert second is not None
    np.testing.assert_allclose(second.position, robot.position, atol=1e-12)


def test_head_yaw_is_frozen_at_engage_and_keeps_operator_forward_aligned() -> None:
    mapper = ClutchPoseMapper(
        "configs/calibrations/left_behind_full_pose_good_20260602.json",
        position_reach_limit_m=None,
        rotation_reach_limit_rad=None,
    )
    yaw = 0.7
    heading = rotation_vector_to_matrix(np.array([0.0, yaw, 0.0]))
    head_quaternion = matrix_to_quat(heading)
    mapper.engage(
        IDENTITY_POSE,
        IDENTITY_POSE,
        np.zeros(3),
        head_quaternion_wxyz=head_quaternion,
    )
    operator_forward = heading @ np.array([0.0, 0.0, -0.05])
    target = mapper.update(Pose(operator_forward, IDENTITY_POSE.quaternion_wxyz), IDENTITY_POSE)
    assert target is not None
    expected = mapper.calibrated_position_matrix @ np.array([0.0, 0.0, -0.05])
    np.testing.assert_allclose(target.position, expected, atol=1e-12)
    np.testing.assert_allclose(mapper.rotation_matrix, mapper.calibrated_rotation_matrix @ heading.T, atol=1e-12)
    assert np.isclose(mapper.engage_head_yaw_rad, yaw)


def test_mirrored_position_mode_does_not_apply_unconfirmed_orientation() -> None:
    mapper = ClutchPoseMapper(
        "configs/calibrations/left_mirrored_position_good_20260602.json",
        position_reach_limit_m=None,
        rotation_reach_limit_rad=None,
    )
    mapper.engage(IDENTITY_POSE, IDENTITY_POSE, np.zeros(3))
    controller_rotation = matrix_to_quat(rotation_vector_to_matrix(np.array([0.4, 0.0, 0.0])))
    target = mapper.update(Pose(np.zeros(3), controller_rotation), IDENTITY_POSE)
    assert target is not None
    np.testing.assert_allclose(quat_to_matrix(target.quaternion_wxyz), np.eye(3), atol=1e-12)


def test_right_behind_frame_exactly_cancels_relay_reflection() -> None:
    real = json.loads(
        resolve_project_path(
            "configs/calibrations/right_real_guided_6dof_20260723_candidate.json"
        ).read_text(encoding="utf-8")
    )
    mirror = json.loads(
        resolve_project_path(
            "configs/calibrations/"
            "right_behind_guided_6dof_20260723_accepted.json"
        ).read_text(encoding="utf-8")
    )
    sagittal_reflection = np.diag([-1.0, 1.0, 1.0])
    real_position = np.asarray(real["position_matrix"], dtype=float)
    real_rotation = np.asarray(real["rotation_matrix"], dtype=float)
    mirror_position = np.asarray(mirror["position_matrix"], dtype=float)
    mirror_rotation = np.asarray(mirror["rotation_matrix"], dtype=float)

    np.testing.assert_allclose(
        mirror_position,
        real_position @ sagittal_reflection,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        mirror_rotation,
        real_rotation @ sagittal_reflection,
        atol=1e-12,
    )
    np.testing.assert_allclose(mirror_position.T @ mirror_position, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(mirror_position), -1.0)
    assert mirror["quest_input"] == {"hand": "right", "mapping_mode": "mirror"}
    assert mirror["task_mapping_semantics"] == {
        "name": "behind",
        "behavior": "native_parallel",
        "mirror_semantics_validated": False,
        "explanation": (
            "The page and transport reported Mirror, but this task frame cancels "
            "the transport reflection. The operator observed and accepted "
            "Behind/native-parallel behavior."
        ),
    }

    for delta in (
        np.array([0.03, -0.01, 0.02]),
        np.array([-0.02, 0.04, -0.01]),
    ):
        np.testing.assert_allclose(
            mirror_position @ (sagittal_reflection @ delta),
            real_position @ delta,
            atol=1e-12,
        )

    for axis in np.eye(3):
        controller_increment = rotation_vector_to_matrix(0.2 * axis)
        mirrored_increment = (
            sagittal_reflection
            @ controller_increment
            @ sagittal_reflection
        )
        np.testing.assert_allclose(
            mirror_rotation @ mirrored_increment @ mirror_rotation.T,
            real_rotation @ controller_increment @ real_rotation.T,
            atol=1e-12,
        )


def test_right_mirror_six_captured_gestures_are_ik_and_collision_safe(model) -> None:
    config = load_config("configs/right_mirror_axis_validation_mujoco.yaml")
    calibration = json.loads(
        resolve_project_path(config["quest"]["calibration"]).read_text(encoding="utf-8")
    )
    source = json.loads(
        resolve_project_path(
            calibration["physical_axis_acceptance"]["source_calibration"]
        ).read_text(encoding="utf-8")
    )
    captures = {
        item["gesture"]: item
        for item in source["guided_calibration"]["captures"]
        if item["repeat"] == 1
    }
    expected_axes = {
        "right": np.array([0.0, -1.0, 0.0]),
        "up": np.array([0.0, 0.0, 1.0]),
        "forward": np.array([1.0, 0.0, 0.0]),
        "screw_clockwise": np.array([1.0, 0.0, 0.0]),
        "nod_yes_down": np.array([0.0, 1.0, 0.0]),
        "nod_no_left": np.array([0.0, 0.0, 1.0]),
    }
    solver = DecoupledIK.from_config(model, config)
    limits = model.joint_limits.copy()
    margin = float(config["hardware"]["joint_limit_margin_rad"])
    limits[:, 0] += margin
    limits[:, 1] -= margin
    solver.set_joint_limits(limits)
    home_q = np.asarray(config["model"]["simulation_start_q_rad"], dtype=float)
    robot_pose, wrist_pose = model.fk(home_q)
    reflection = np.diag([-1.0, 1.0, 1.0])
    identity_controller = Pose(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    for gesture, expected_axis in expected_axes.items():
        capture = captures[gesture]
        observed = np.asarray(capture["observed_operator_vector"], dtype=float)
        observed /= np.linalg.norm(observed)
        mapper = ClutchPoseMapper(
            config["quest"]["calibration"],
            translation_scale=config["quest"]["translation_scale"],
            rotation_scale=config["quest"]["rotation_scale"],
            position_reach_limit_m=config["quest"]["position_reach_limit_m"],
            rotation_reach_limit_rad=config["quest"]["rotation_reach_limit_rad"],
        )
        mapper.engage(
            identity_controller,
            robot_pose,
            wrist_pose.position,
        )
        if capture["kind"] == "translation":
            mirrored_delta = reflection @ (0.08 * observed)
            controller = Pose(
                mirrored_delta,
                identity_controller.quaternion_wxyz,
            )
        else:
            real_increment = rotation_vector_to_matrix(0.30 * observed)
            mirrored_increment = reflection @ real_increment @ reflection
            controller = Pose(
                np.zeros(3),
                matrix_to_quat(mirrored_increment),
            )
        target = mapper.update(controller, robot_pose)
        assert target is not None

        if capture["kind"] == "translation":
            mapped_direction = target.position - robot_pose.position
        else:
            mapped_direction = matrix_to_rotation_vector(
                quat_to_matrix(target.quaternion_wxyz)
                @ quat_to_matrix(robot_pose.quaternion_wxyz).T
            )
        mapped_direction /= np.linalg.norm(mapped_direction)
        assert np.dot(mapped_direction, expected_axis) > np.cos(np.deg2rad(20.0))

        q_solution = home_q.copy()
        diagnostics = None
        for _ in range(300):
            q_solution, diagnostics = solver.solve(target, q_solution)
            if (
                diagnostics.status == "ok"
                and diagnostics.position_residual_m <= 0.001
                and diagnostics.orientation_residual_rad <= np.deg2rad(0.25)
            ):
                break
        assert diagnostics is not None
        assert diagnostics.status == "ok"
        assert diagnostics.position_residual_m <= 0.001
        assert diagnostics.orientation_residual_rad <= np.deg2rad(0.25)
        assert not model.in_self_collision(
            q_solution,
            config["hardware"]["gripper_open_m"],
        )


def test_both_hands_mirror_profiles_follow_operator_defined_four_axis_flips() -> None:
    controller_reflection = np.diag([-1.0, 1.0, 1.0])
    task_position_transform = np.diag([-1.0, -1.0, 1.0])
    task_rotation_reflection = np.diag([1.0, -1.0, 1.0])

    for hand in ("left", "right"):
        behind_path = (
            f"configs/calibrations/{hand}_behind_all_motions_20260723_candidate.json"
        )
        mirror_path = (
            "configs/calibrations/right_mirror_20260723_accepted.json"
            if hand == "right"
            else "configs/calibrations/left_mirror_all_motions_20260812_accepted.json"
        )
        behind_document = json.loads(
            resolve_project_path(behind_path).read_text(encoding="utf-8")
        )
        mirror_document = json.loads(
            resolve_project_path(mirror_path).read_text(encoding="utf-8")
        )
        assert behind_document["task_position_signs"] == [1.0, 1.0, 1.0]
        assert behind_document["task_rotation_signs"] == [1.0, 1.0, 1.0]
        assert mirror_document["task_position_signs"] == [-1.0, -1.0, 1.0]
        assert mirror_document["task_rotation_signs"] == [-1.0, 1.0, -1.0]

        for axis in np.eye(3):
            behind = ClutchPoseMapper(
                behind_path,
                position_reach_limit_m=None,
                rotation_reach_limit_rad=None,
            )
            mirror = ClutchPoseMapper(
                mirror_path,
                position_reach_limit_m=None,
                rotation_reach_limit_rad=None,
            )
            behind.engage(IDENTITY_POSE, IDENTITY_POSE, np.zeros(3))
            mirror.engage(IDENTITY_POSE, IDENTITY_POSE, np.zeros(3))
            behind_target = behind.update(
                Pose(0.02 * axis, IDENTITY_POSE.quaternion_wxyz),
                IDENTITY_POSE,
            )
            mirror_target = mirror.update(
                Pose(
                    controller_reflection @ (0.02 * axis),
                    IDENTITY_POSE.quaternion_wxyz,
                ),
                IDENTITY_POSE,
            )
            assert behind_target is not None
            assert mirror_target is not None
            np.testing.assert_allclose(
                mirror_target.position,
                task_position_transform @ behind_target.position,
                atol=1e-12,
            )

            behind = ClutchPoseMapper(
                behind_path,
                position_reach_limit_m=None,
                rotation_reach_limit_rad=None,
            )
            mirror = ClutchPoseMapper(
                mirror_path,
                position_reach_limit_m=None,
                rotation_reach_limit_rad=None,
            )
            behind.engage(IDENTITY_POSE, IDENTITY_POSE, np.zeros(3))
            mirror.engage(IDENTITY_POSE, IDENTITY_POSE, np.zeros(3))
            controller_increment = rotation_vector_to_matrix(0.2 * axis)
            mirrored_increment = (
                controller_reflection
                @ controller_increment
                @ controller_reflection
            )
            behind_target = behind.update(
                Pose(np.zeros(3), matrix_to_quat(controller_increment)),
                IDENTITY_POSE,
            )
            mirror_target = mirror.update(
                Pose(np.zeros(3), matrix_to_quat(mirrored_increment)),
                IDENTITY_POSE,
            )
            assert behind_target is not None
            assert mirror_target is not None
            np.testing.assert_allclose(
                quat_to_matrix(mirror_target.quaternion_wxyz),
                task_rotation_reflection
                @ quat_to_matrix(behind_target.quaternion_wxyz)
                @ task_rotation_reflection,
                atol=1e-12,
            )


def test_all_four_task_profiles_are_ik_and_collision_safe_for_six_motions(model) -> None:
    controller_reflection = np.diag([-1.0, 1.0, 1.0])
    task_position_transform = np.diag([-1.0, -1.0, 1.0])
    task_rotation_reflection = np.diag([1.0, -1.0, 1.0])
    base_expected_axes = {
        "right": np.array([0.0, -1.0, 0.0]),
        "up": np.array([0.0, 0.0, 1.0]),
        "forward": np.array([1.0, 0.0, 0.0]),
        "screw_clockwise": np.array([1.0, 0.0, 0.0]),
        "nod_yes_down": np.array([0.0, 1.0, 0.0]),
        "nod_no_left": np.array([0.0, 0.0, 1.0]),
    }
    identity_controller = Pose(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    for profile_name in TASK_PROFILE_NAMES:
        config = load_config("configs/quest_50pct_hardware.yaml")
        apply_task_profile(config, profile_name)
        calibration = json.loads(
            resolve_project_path(config["quest"]["calibration"]).read_text(
                encoding="utf-8"
            )
        )
        semantics = calibration["task_mapping_semantics"]
        if semantics["name"] == "behind":
            source_path = semantics["source"]
        else:
            behind = json.loads(
                resolve_project_path(semantics["behind_source"]).read_text(
                    encoding="utf-8"
                )
            )
            source_path = behind["task_mapping_semantics"]["source"]
        source = json.loads(
            resolve_project_path(source_path).read_text(encoding="utf-8")
        )
        captures = {
            item["gesture"]: item
            for item in source["guided_calibration"]["captures"]
            if item["repeat"] == 1
        }

        solver = DecoupledIK.from_config(model, config)
        limits = model.joint_limits.copy()
        margin = float(config["hardware"]["joint_limit_margin_rad"])
        limits[:, 0] += margin
        limits[:, 1] -= margin
        solver.set_joint_limits(limits)
        home_q = np.asarray(config["model"]["simulation_start_q_rad"], dtype=float)
        robot_pose, wrist_pose = model.fk(home_q)

        for gesture, base_expected_axis in base_expected_axes.items():
            capture = captures[gesture]
            observed = np.asarray(capture["observed_operator_vector"], dtype=float)
            observed /= np.linalg.norm(observed)
            mapper = ClutchPoseMapper(
                config["quest"]["calibration"],
                translation_scale=config["quest"]["translation_scale"],
                rotation_scale=config["quest"]["rotation_scale"],
                position_reach_limit_m=config["quest"]["position_reach_limit_m"],
                rotation_reach_limit_rad=config["quest"]["rotation_reach_limit_rad"],
            )
            mapper.engage(identity_controller, robot_pose, wrist_pose.position)
            if capture["kind"] == "translation":
                delta = 0.08 * observed
                if config["quest"]["mapping_mode"] == "mirror":
                    delta = controller_reflection @ delta
                controller = Pose(delta, identity_controller.quaternion_wxyz)
                expected_axis = (
                    task_position_transform @ base_expected_axis
                    if semantics["name"] == "mirror"
                    else base_expected_axis
                )
            else:
                increment = rotation_vector_to_matrix(0.30 * observed)
                if config["quest"]["mapping_mode"] == "mirror":
                    increment = (
                        controller_reflection @ increment @ controller_reflection
                    )
                controller = Pose(np.zeros(3), matrix_to_quat(increment))
                expected_axis = (
                    np.linalg.det(task_rotation_reflection)
                    * task_rotation_reflection
                    @ base_expected_axis
                    if semantics["name"] == "mirror"
                    else base_expected_axis
                )
            target = mapper.update(controller, robot_pose)
            assert target is not None
            if capture["kind"] == "translation":
                mapped_direction = target.position - robot_pose.position
            else:
                mapped_direction = matrix_to_rotation_vector(
                    quat_to_matrix(target.quaternion_wxyz)
                    @ quat_to_matrix(robot_pose.quaternion_wxyz).T
                )
            mapped_direction /= np.linalg.norm(mapped_direction)
            assert np.dot(mapped_direction, expected_axis) > np.cos(np.deg2rad(20.0))

            q_solution = home_q.copy()
            diagnostics = None
            for _ in range(300):
                q_solution, diagnostics = solver.solve(target, q_solution)
                if (
                    diagnostics.status == "ok"
                    and diagnostics.position_residual_m <= 0.0015
                    and diagnostics.orientation_residual_rad <= np.deg2rad(0.25)
                ):
                    break
            assert diagnostics is not None
            assert diagnostics.status == "ok"
            # The configured posture bias intentionally leaves a small steady
            # Cartesian residual at the expanded 50% reach envelope.
            assert diagnostics.position_residual_m <= 0.0015
            assert diagnostics.orientation_residual_rad <= np.deg2rad(0.25)
            assert not model.in_self_collision(
                q_solution,
                config["hardware"]["gripper_open_m"],
            )
