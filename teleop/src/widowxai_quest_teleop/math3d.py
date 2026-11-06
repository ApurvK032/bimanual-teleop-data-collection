from __future__ import annotations

import numpy as np


def normalize_quat(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("zero-length quaternion")
    q = q / norm
    return q if q[0] >= 0.0 else -q


def quat_conjugate(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_multiply(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a_wxyz, dtype=float).reshape(4)
    bw, bx, by, bz = np.asarray(b_wxyz, dtype=float).reshape(4)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def quat_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    r = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    return normalize_quat(q)


def rotation_vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    axis = v / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def matrix_to_rotation_vector(matrix: np.ndarray) -> np.ndarray:
    q = matrix_to_quat(matrix)
    vector = q[1:]
    sin_half = float(np.linalg.norm(vector))
    if sin_half < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, float(q[0]))
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return vector / sin_half * angle


def orientation_error_world(target_wxyz: np.ndarray, current_wxyz: np.ndarray) -> np.ndarray:
    return matrix_to_rotation_vector(quat_to_matrix(target_wxyz) @ quat_to_matrix(current_wxyz).T)


def clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if maximum <= 0.0 or norm <= maximum:
        return value.copy()
    return value * (maximum / norm)

