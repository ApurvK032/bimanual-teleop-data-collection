from __future__ import annotations

import numpy as np


def trigger_to_gripper_position(
    trigger: float,
    open_position_m: float,
    closed_position_m: float,
) -> float:
    """Map the complete trigger stroke to the complete configured gripper stroke."""

    open_position = float(open_position_m)
    closed_position = float(closed_position_m)
    if not np.isfinite(open_position) or not np.isfinite(closed_position):
        raise ValueError("gripper endpoints must be finite")
    if closed_position >= open_position:
        raise ValueError("closed gripper position must be below the open position")
    amount = float(np.clip(float(trigger), 0.0, 1.0))
    return open_position - amount * (open_position - closed_position)
