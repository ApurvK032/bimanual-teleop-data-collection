from __future__ import annotations

import numpy as np


def estimate_wrist_pivot_offset(
    grip_positions: np.ndarray,
    grip_rotations: np.ndarray,
    *,
    maximum_offset_m: float = 0.20,
) -> np.ndarray:
    """Estimate the controller-frame grip-to-wrist offset from twist samples."""
    positions = np.asarray(grip_positions, dtype=float)
    rotations = np.asarray(grip_rotations, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("grip_positions must have shape (N, 3)")
    if rotations.shape != (positions.shape[0], 3, 3):
        raise ValueError("grip_rotations must have shape (N, 3, 3)")
    if positions.shape[0] < 12:
        raise ValueError("at least 12 calibration samples are required")
    delta_positions = positions - positions.mean(axis=0)
    delta_rotations = rotations - rotations.mean(axis=0)
    normal = np.einsum("nji,njk->ik", delta_rotations, delta_rotations)
    rhs = np.einsum("nji,nj->i", delta_rotations, delta_positions)
    if np.linalg.cond(normal) > 1e8:
        raise ValueError("calibration motion did not excite enough rotation axes")
    offset = -np.linalg.solve(normal, rhs)
    if not np.all(np.isfinite(offset)) or np.linalg.norm(offset) > maximum_offset_m:
        raise ValueError("estimated wrist offset failed the sanity bound")
    return offset

