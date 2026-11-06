import json

import numpy as np
import pytest

from widowxai_quest_teleop.config import load_config, resolve_project_path
from widowxai_quest_teleop.math3d import matrix_to_quat, rotation_vector_to_matrix
from widowxai_quest_teleop.task_frame_calibration import (
    build_calibration_document,
    direction_map_quality,
    fit_direction_map,
    heading_correction,
)


def test_direction_fit_recovers_a_proper_rotation_from_repeated_axes() -> None:
    expected_mapping = rotation_vector_to_matrix(np.array([0.25, -0.4, 0.15]))
    observed = np.vstack([np.eye(3), np.eye(3)])
    desired = (expected_mapping @ observed.T).T

    fitted, quality = fit_direction_map(observed, desired)

    np.testing.assert_allclose(fitted, expected_mapping, atol=1e-12)
    assert np.isclose(np.linalg.det(fitted), 1.0)
    assert quality["max_axis_error_deg"] < 1e-5


def test_direction_fit_recovers_requested_mirror_handedness() -> None:
    reflection = np.diag([-1.0, 1.0, 1.0])
    task_rotation = rotation_vector_to_matrix(np.array([0.25, -0.4, 0.15]))
    expected_mapping = task_rotation @ reflection
    observed = np.vstack([np.eye(3), np.eye(3)])
    desired = (expected_mapping @ observed.T).T

    fitted, quality = fit_direction_map(
        observed,
        desired,
        mapping_determinant=-1,
    )

    np.testing.assert_allclose(fitted, expected_mapping, atol=1e-12)
    assert np.isclose(np.linalg.det(fitted), -1.0)
    assert quality["mapping_determinant"] == pytest.approx(-1.0)
    assert quality["max_axis_error_deg"] < 1e-5


def test_heading_correction_removes_captured_head_yaw() -> None:
    yaw = 0.6
    heading = rotation_vector_to_matrix(np.array([0.0, yaw, 0.0]))
    quaternion = matrix_to_quat(heading)
    operator_forward = np.array([0.0, 0.0, -1.0])
    world_forward = heading @ operator_forward

    np.testing.assert_allclose(
        heading_correction(quaternion) @ world_forward,
        operator_forward,
        atol=1e-12,
    )


def test_calibration_document_uses_independent_valid_position_and_rotation_maps() -> None:
    position_mapping = rotation_vector_to_matrix(np.array([0.1, 0.2, -0.3]))
    rotation_mapping = rotation_vector_to_matrix(np.array([-0.2, 0.15, 0.35]))
    observed = np.vstack([np.eye(3), np.eye(3)])
    document = build_calibration_document(
        name="guided",
        position_observed=observed,
        position_desired=(position_mapping @ observed.T).T,
        rotation_observed=observed,
        rotation_desired=(rotation_mapping @ observed.T).T,
        captures=[],
    )

    np.testing.assert_allclose(document["position_matrix"], position_mapping, atol=1e-12)
    np.testing.assert_allclose(document["rotation_matrix"], rotation_mapping, atol=1e-12)
    assert document["task_rotation_signs"] == [1.0, 1.0, 1.0]


def test_linked_calibration_uses_one_rigid_frame_for_translation_and_rotation() -> None:
    mapping = rotation_vector_to_matrix(np.array([0.1, -0.3, 0.2]))
    observed = np.vstack([np.eye(3), np.eye(3)])
    desired = (mapping @ observed.T).T
    document = build_calibration_document(
        name="linked",
        position_observed=observed,
        position_desired=desired,
        rotation_observed=observed,
        rotation_desired=desired,
        captures=[],
        link_rotation_to_position=True,
    )

    np.testing.assert_allclose(document["position_matrix"], mapping, atol=1e-12)
    np.testing.assert_allclose(document["rotation_matrix"], mapping, atol=1e-12)
    assert document["guided_calibration"]["rotation_frame_linked_to_position"]
    quality = direction_map_quality(mapping, observed, desired)
    assert quality["max_axis_error_deg"] < 1e-5


def test_linked_mirror_calibration_handles_polar_and_axial_vectors() -> None:
    reflection = np.diag([-1.0, 1.0, 1.0])
    task_rotation = rotation_vector_to_matrix(np.array([0.1, -0.3, 0.2]))
    mapping = task_rotation @ reflection
    observed = np.vstack([np.eye(3), np.eye(3)])
    position_desired = (mapping @ observed.T).T
    rotation_desired = (
        np.linalg.det(mapping) * mapping @ observed.T
    ).T

    document = build_calibration_document(
        name="linked_mirror",
        position_observed=observed,
        position_desired=position_desired,
        rotation_observed=observed,
        rotation_desired=rotation_desired,
        captures=[],
        link_rotation_to_position=True,
        position_mapping_determinant=-1,
    )

    np.testing.assert_allclose(document["position_matrix"], mapping, atol=1e-12)
    np.testing.assert_allclose(document["rotation_matrix"], mapping, atol=1e-12)
    assert document["guided_calibration"]["task_frame_determinant"] == pytest.approx(
        -1.0
    )
    assert document["guided_calibration"]["rotation_quality"][
        "max_axis_error_deg"
    ] < 1e-5


def test_guided_mujoco_profile_is_linked_unshaped_and_hardware_disabled() -> None:
    config = load_config("configs/guided_calibration_30pct_mujoco.yaml")
    calibration = json.loads(
        resolve_project_path(config["quest"]["calibration"]).read_text(encoding="utf-8")
    )
    position = np.asarray(calibration["position_matrix"], dtype=float)
    rotation = np.asarray(calibration["rotation_matrix"], dtype=float)

    np.testing.assert_allclose(position, rotation, atol=1e-12)
    np.testing.assert_allclose(position.T @ position, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(position), 1.0)
    assert calibration["task_rotation_signs"] == [1.0, 1.0, 1.0]
    assert calibration["guided_calibration"]["rotation_frame_linked_to_position"]
    assert config["control"]["update_mode"] == "quest_synchronized"
    assert not config["control"]["joint_command_limits"]["enabled"]
    assert not config["control"]["gripper_command_limits"]["enabled"]
    assert not config["hardware"]["enabled"]
    assert not config["hardware"]["require_explicit_enable"]


def test_dual_mujoco_right_frame_is_an_explicit_identity_hand_transfer() -> None:
    left = json.loads(
        resolve_project_path(
            "configs/calibrations/left_behind_mujoco_20260811.json"
        ).read_text(encoding="utf-8")
    )
    right = json.loads(
        resolve_project_path(
            "configs/calibrations/right_behind_mujoco_20260811_derived.json"
        ).read_text(encoding="utf-8")
    )

    np.testing.assert_allclose(right["position_matrix"], left["position_matrix"])
    np.testing.assert_allclose(right["rotation_matrix"], left["rotation_matrix"])
    assert left["quest_input"] == {"hand": "left", "mapping_mode": "real"}
    assert right["quest_input"] == {"hand": "right", "mapping_mode": "real"}
    assert right["derivation"]["method"].startswith("identity hand transfer")
    assert not right["derivation"]["independent_right_hand_capture_performed"]
    assert right["guided_calibration"]["captures"] == []
    assert left["validation_scope"]["status"] == "simulation_only"
    assert right["validation_scope"]["status"] == "simulation_only"


def test_right_real_independent_rotation_calibration_preserves_translation() -> None:
    candidate = json.loads(
        resolve_project_path(
            "configs/calibrations/"
            "right_real_guided_6dof_20260723_independent_rotation_candidate.json"
        ).read_text(encoding="utf-8")
    )
    source = json.loads(
        resolve_project_path(
            candidate["guided_calibration"]["source_calibration"]
        ).read_text(encoding="utf-8")
    )
    position = np.asarray(candidate["position_matrix"], dtype=float)
    rotation = np.asarray(candidate["rotation_matrix"], dtype=float)

    np.testing.assert_allclose(position, source["position_matrix"], atol=1e-12)
    np.testing.assert_allclose(position.T @ position, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(position), 1.0)
    assert np.isclose(np.linalg.det(rotation), 1.0)
    assert not candidate["guided_calibration"]["rotation_frame_linked_to_position"]
    assert (
        candidate["guided_calibration"]["rotation_quality"]["max_axis_error_deg"]
        < source["guided_calibration"]["rotation_quality"]["max_axis_error_deg"]
    )
    assert candidate["quest_input"] == {"hand": "right", "mapping_mode": "real"}


def test_rejected_right_real_semantic_candidate_is_algebraically_explicit() -> None:
    config = load_config("configs/step3_right_real_candidate_mujoco.yaml")
    candidate = json.loads(
        resolve_project_path(
            "configs/calibrations/"
            "right_real_guided_6dof_20260723_semantic_axes_candidate.json"
        ).read_text(encoding="utf-8")
    )
    source = json.loads(
        resolve_project_path(
            candidate["guided_calibration"]["source_calibration"]
        ).read_text(encoding="utf-8")
    )
    position = np.asarray(candidate["position_matrix"], dtype=float)
    rotation = np.asarray(candidate["rotation_matrix"], dtype=float)
    signs = np.diag(np.asarray(candidate["task_rotation_signs"], dtype=float))
    source_rotation = np.asarray(source["rotation_matrix"], dtype=float)
    swap_xy = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(position, source["position_matrix"], atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(signs @ rotation, swap_xy @ source_rotation, atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0)
    assert candidate["quest_input"] == {"hand": "right", "mapping_mode": "real"}

    active = json.loads(
        resolve_project_path(config["quest"]["calibration"]).read_text(encoding="utf-8")
    )
    assert active["name"] == "right_real_guided_6dof_20260723_candidate"
    assert active["guided_calibration"]["rotation_frame_linked_to_position"]
    assert config["quest"]["hand"] == "right"
    assert config["quest"]["mapping_mode"] == "real"
    assert not config["hardware"]["enabled"]
