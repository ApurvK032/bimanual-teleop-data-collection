from __future__ import annotations

import json

import numpy as np

from scripts.viewer_client import parse_ik_state


def test_viewer_accepts_latest_finite_state_from_selected_teleop() -> None:
    raw = json.dumps(
        {
            "type": "ik_state",
            "teleop_id": "reference",
            "q_arm": [0, 1, 2, 3, 4, 5],
            "gripper_q": 0.02,
        }
    )
    q_arm, gripper = parse_ik_state(raw, "reference")
    np.testing.assert_array_equal(q_arm, np.arange(6))
    assert gripper == 0.02
    assert parse_ik_state(raw, "different") is None
