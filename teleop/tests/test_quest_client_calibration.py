"""The Quest page's wrist-calibration shortcut must never steal teleop frames.

Grip is the clutch and the trigger drives the gripper, so "grip + trigger" is
ordinary teleoperation. On 2026-10-23 the right-hand page auto-armed wrist
calibration whenever no wrist offset was saved, so closing the gripper while
clutched stopped pose streaming for 5 s and the arm held until re-anchoring.
These tests run the real client.js in Node with fake browser objects.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "src" / "widowxai_quest_teleop" / "web" / "client.js"
HARNESS = ROOT / "tests" / "js" / "quest_client_harness.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js runs the Quest page script")

CLUTCH_ONLY = {"grip": 1.0, "trigger": 0.0}
CLUTCH_AND_CLOSE = {"grip": 1.0, "trigger": 1.0}
RELEASED = {"grip": 0.0, "trigger": 0.0}
# 600 frames at 90 Hz is 6.7 s, longer than the 5 s calibration capture.
LONG_PRESS = 600


def run_page(scenario: dict) -> dict:
    result = subprocess.run(
        [NODE, str(HARNESS), str(CLIENT), json.dumps(scenario)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_closing_the_gripper_while_clutched_never_pauses_the_right_hand_stream() -> None:
    result = run_page(
        {
            "query": "?hand=right&mode=real",
            "hands": ["right"],
            "steps": [
                {"frames": 30, "right": CLUTCH_ONLY},
                {"frames": LONG_PRESS, "right": CLUTCH_AND_CLOSE},
                {"frames": 30, "right": CLUTCH_ONLY},
                {"frames": LONG_PRESS, "right": CLUTCH_AND_CLOSE},
            ],
        }
    )

    assert [step["silent_frames"] for step in result["steps"]] == [0, 0, 0, 0]
    assert result["pose_packets"] == 30 + LONG_PRESS + 30 + LONG_PRESS
    assert result["last_packet"]["right"]["grip"] == 1.0
    assert result["last_packet"]["right"]["trigger"] == 1.0


def test_left_hand_page_streams_through_grip_and_trigger() -> None:
    result = run_page(
        {
            "query": "?hand=left&mode=real",
            "hands": ["left", "right"],
            "steps": [{"frames": LONG_PRESS, "left": CLUTCH_AND_CLOSE}],
        }
    )

    assert result["steps"][0]["silent_frames"] == 0


def test_bimanual_page_streams_while_either_hand_closes_its_gripper() -> None:
    result = run_page(
        {
            "query": "?input=bimanual",
            "hands": ["left", "right"],
            "steps": [
                {"frames": LONG_PRESS, "right": CLUTCH_AND_CLOSE, "left": CLUTCH_ONLY},
                {"frames": LONG_PRESS, "left": CLUTCH_AND_CLOSE, "right": RELEASED},
            ],
        }
    )

    assert [step["silent_frames"] for step in result["steps"]] == [0, 0]
    assert result["packet_types"] == ["bimanual_pose"]


def test_explicit_calibrate_wrist_still_captures_and_a_failure_disarms_it() -> None:
    result = run_page(
        {
            "query": "?hand=right&mode=real",
            "hands": ["right"],
            "arm_calibration": True,
            "steps": [
                # Armed by the button: the chord starts the 5 s capture. A still
                # wrist cannot be solved, so the capture fails and disarms.
                {"frames": LONG_PRESS, "right": CLUTCH_AND_CLOSE},
                {"frames": 30, "right": RELEASED},
                # Back to teleop: the same chord must now stream normally.
                {"frames": LONG_PRESS, "right": CLUTCH_AND_CLOSE},
            ],
        }
    )

    capture, _released, teleop = result["steps"]
    assert 4900.0 <= capture["longest_silent_ms"] <= 5100.0
    assert "Wrist calibration failed" in result["status"]
    assert "Calibrate wrist" in result["status"]
    assert teleop["silent_frames"] == 0


def test_a_saved_wrist_offset_is_still_applied() -> None:
    result = run_page(
        {
            "query": "?hand=right&mode=real",
            "hands": ["right"],
            "local_storage": {"widowxai.rightWristOffset": json.dumps([0.0, 0.0, 0.05])},
            "steps": [{"frames": 5, "right": CLUTCH_ONLY}],
        }
    )

    # Identity orientation: the reported point is the grip point plus the offset.
    assert result["last_packet"]["right"]["position"] == pytest.approx([0.1, 1.0, -0.25])
