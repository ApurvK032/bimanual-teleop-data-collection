from __future__ import annotations

from typing import Any

import numpy as np


def head_yaw_rad(quaternion_wxyz: np.ndarray) -> float:
    """Extract Quest's Y-up heading using the same convention as the live mapper."""
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
    return float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))


def heading_correction(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Rotate a Quest world vector into the operator frame frozen at capture."""
    angle = -head_yaw_rad(quaternion_wxyz)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=float,
    )


def fit_direction_map(
    observed_vectors: np.ndarray,
    desired_vectors: np.ndarray,
    *,
    mapping_determinant: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the closest orthogonal map with the requested handedness."""

    if mapping_determinant not in (-1, 1):
        raise ValueError("mapping_determinant must be -1 or +1")
    observed = np.asarray(observed_vectors, dtype=float).reshape(-1, 3)
    desired = np.asarray(desired_vectors, dtype=float).reshape(-1, 3)
    if observed.shape != desired.shape or len(observed) < 3:
        raise ValueError("at least three paired 3D direction samples are required")
    observed_norms = np.linalg.norm(observed, axis=1)
    desired_norms = np.linalg.norm(desired, axis=1)
    if np.any(observed_norms < 1e-8) or np.any(desired_norms < 1e-8):
        raise ValueError("calibration directions must be nonzero")
    observed_unit = observed / observed_norms[:, None]
    desired_unit = desired / desired_norms[:, None]
    excitation = np.linalg.svd(observed_unit, compute_uv=False)
    if excitation[-1] < 0.35:
        raise ValueError("calibration gestures did not excite three independent axes")

    covariance = desired_unit.T @ observed_unit
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = mapping_determinant * np.sign(
        np.linalg.det(left @ right_t)
    )
    mapping = left @ correction @ right_t

    quality = direction_map_quality(mapping, observed_unit, desired_unit)
    quality["excitation_singular_values"] = excitation.tolist()
    quality["mapping_determinant"] = float(np.linalg.det(mapping))
    return mapping, quality


def direction_map_quality(
    mapping: np.ndarray,
    observed_vectors: np.ndarray,
    desired_vectors: np.ndarray,
    *,
    axial_vectors: bool = False,
) -> dict[str, Any]:
    """Measure angular agreement for a fixed direction mapping."""
    orthogonal_map = np.asarray(mapping, dtype=float).reshape(3, 3)
    observed = np.asarray(observed_vectors, dtype=float).reshape(-1, 3)
    desired = np.asarray(desired_vectors, dtype=float).reshape(-1, 3)
    if observed.shape != desired.shape or not len(observed):
        raise ValueError("paired 3D direction samples are required")
    observed_norms = np.linalg.norm(observed, axis=1)
    desired_norms = np.linalg.norm(desired, axis=1)
    if np.any(observed_norms < 1e-8) or np.any(desired_norms < 1e-8):
        raise ValueError("calibration directions must be nonzero")
    observed_unit = observed / observed_norms[:, None]
    desired_unit = desired / desired_norms[:, None]
    vector_map = orthogonal_map
    if axial_vectors:
        # A reflection F maps polar vectors as Fv, but axial vectors (including
        # rotation vectors) as det(F)Fv. This is the vector form of the live
        # mapper's F @ dR @ F.T conjugation.
        vector_map = np.linalg.det(orthogonal_map) * orthogonal_map
    mapped = (vector_map @ observed_unit.T).T
    alignment = np.sum(mapped * desired_unit, axis=1)
    alignment = np.clip(alignment, -1.0, 1.0)
    errors_deg = np.degrees(np.arccos(alignment))
    return {
        "sample_count": int(len(observed)),
        "median_axis_error_deg": float(np.median(errors_deg)),
        "max_axis_error_deg": float(np.max(errors_deg)),
        "minimum_alignment": float(np.min(alignment)),
        "axis_errors_deg": errors_deg.tolist(),
    }


def build_calibration_document(
    *,
    name: str,
    position_observed: np.ndarray,
    position_desired: np.ndarray,
    rotation_observed: np.ndarray,
    rotation_desired: np.ndarray,
    captures: list[dict[str, Any]],
    link_rotation_to_position: bool = False,
    position_mapping_determinant: int = 1,
) -> dict[str, Any]:
    position_matrix, position_quality = fit_direction_map(
        position_observed,
        position_desired,
        mapping_determinant=position_mapping_determinant,
    )
    if link_rotation_to_position:
        # Both controller translation deltas and dR_world rotation vectors are
        # expressed in the same Quest world frame. One rigid task-frame change
        # must therefore map both; fitting two unrelated frames is unphysical.
        rotation_matrix = position_matrix.copy()
        rotation_quality = direction_map_quality(
            rotation_matrix,
            rotation_observed,
            rotation_desired,
            axial_vectors=True,
        )
    else:
        rotation_matrix, rotation_quality = fit_direction_map(
            rotation_observed,
            rotation_desired,
        )
    if position_quality["max_axis_error_deg"] > 25.0:
        raise ValueError("translation calibration is inconsistent; repeat the guided captures")
    if rotation_quality["max_axis_error_deg"] > 25.0:
        raise ValueError("rotation calibration is inconsistent; repeat the guided captures")
    return {
        "name": name,
        "position_matrix": position_matrix.tolist(),
        "rotation_matrix": rotation_matrix.tolist(),
        "orientation_delta_mode": "matrix_conjugate",
        "task_rotation_signs": [1.0, 1.0, 1.0],
        "vectors_are_columns": True,
        "quaternion_convention": "wxyz",
        "guided_calibration": {
            "rotation_frame_linked_to_position": bool(link_rotation_to_position),
            "task_frame_determinant": float(np.linalg.det(position_matrix)),
            "position_quality": position_quality,
            "rotation_quality": rotation_quality,
            "captures": captures,
        },
    }
