from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.calibrate_task_frame import BimanualHandSampleStream
from widowxai_quest_teleop.safety import FreshSequenceWatchdog
from widowxai_quest_teleop.transport import (
    BIMANUAL_SCHEMA_VERSION,
    BimanualQuestReceiver,
    parse_bimanual_pose_message,
    parse_pose_message,
)

WEB = Path(__file__).resolve().parents[1] / "src" / "widowxai_quest_teleop" / "web"


def bimanual_payload(**overrides):
    payload = {
        "type": "bimanual_pose",
        "schema_version": BIMANUAL_SCHEMA_VERSION,
        "sequence": 11,
        "capture_monotonic_ms": 1000.0,
        "capture_epoch_ms": 1700000000000.0,
        "send_monotonic_ms": 1001.0,
        "reconnect_generation": 2,
        "left": {
            "tracked": True,
            "mapping_mode": "real",
            "position": [1.0, 2.0, 3.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "grip": 0.9,
            "trigger": 0.25,
        },
        "right": {
            "tracked": True,
            "mapping_mode": "real",
            "position": [-1.0, 2.0, 3.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "grip": 0.1,
            "trigger": 0.75,
        },
        "head": {"orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
    }
    payload.update(overrides)
    return payload


def test_one_packet_carries_both_controllers_under_one_sequence() -> None:
    sample = parse_bimanual_pose_message(bimanual_payload())

    assert sample.sequence == 11
    assert sample.reconnect_generation == 2
    assert sample.tracked_hands() == ("left", "right")
    # The shared sequence and capture stamps are what align the two arms.
    left = sample.for_hand("left")
    right = sample.for_hand("right")
    assert left.sequence == right.sequence == 11
    assert left.capture_monotonic_ms == right.capture_monotonic_ms
    assert left.pc_arrival_monotonic_ns == right.pc_arrival_monotonic_ns
    assert left.grip == pytest.approx(0.9)
    assert right.trigger == pytest.approx(0.75)
    assert left.hand == "left"
    assert right.hand == "right"


def test_webxr_xyzw_becomes_internal_wxyz_for_both_hands() -> None:
    payload = bimanual_payload()
    payload["left"]["orientation_xyzw"] = [0.1, 0.2, 0.3, 0.9]
    sample = parse_bimanual_pose_message(payload)

    quaternion = sample.for_hand("left").controller_pose.quaternion_wxyz
    expected = np.array([0.9, 0.1, 0.2, 0.3])
    assert np.allclose(quaternion, expected / np.linalg.norm(expected))


def test_mirroring_is_applied_per_hand_not_globally() -> None:
    """A mirror on one arm must not reflect the other arm's controller."""

    payload = bimanual_payload()
    payload["left"]["mapping_mode"] = "mirror"
    sample = parse_bimanual_pose_message(
        payload, mapping_modes={"left": "mirror", "right": "real"}
    )

    assert sample.for_hand("left").controller_pose.position[0] == pytest.approx(-1.0)
    assert sample.for_hand("right").controller_pose.position[0] == pytest.approx(-1.0)
    # The right hand's raw x was already -1.0 and must be untouched, while the
    # left hand's +1.0 was reflected. Confirm the left reflection really ran.
    payload["left"]["position"] = [5.0, 2.0, 3.0]
    mirrored = parse_bimanual_pose_message(
        payload, mapping_modes={"left": "mirror", "right": "real"}
    )
    assert mirrored.for_hand("left").controller_pose.position[0] == pytest.approx(-5.0)


def test_untracked_controller_is_explicit_and_never_a_stale_pose() -> None:
    payload = bimanual_payload()
    payload["left"] = {"tracked": False, "mapping_mode": "real"}
    sample = parse_bimanual_pose_message(payload)

    assert sample.tracked_hands() == ("right",)
    assert sample.for_hand("left") is None
    assert sample.controller("left").tracked is False
    # The other arm keeps a complete, usable sample.
    assert sample.for_hand("right") is not None


def test_absent_controller_block_is_untracked_rather_than_an_error() -> None:
    payload = bimanual_payload()
    del payload["right"]
    sample = parse_bimanual_pose_message(payload)

    assert sample.tracked_hands() == ("left",)
    assert sample.for_hand("right") is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"sequence": -1},
        {"capture_monotonic_ms": float("nan")},
        {"capture_epoch_ms": float("inf")},
        {"send_monotonic_ms": float("nan")},
    ],
)
def test_invalid_shared_timing_rejects_the_whole_packet(overrides) -> None:
    """Shared timing is what aligns the arms; a bad value poisons both."""

    with pytest.raises(ValueError):
        parse_bimanual_pose_message(bimanual_payload(**overrides))


def test_unsupported_schema_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported bimanual schema version"):
        parse_bimanual_pose_message(bimanual_payload(schema_version=1))


def test_single_arm_packet_is_not_accepted_as_bimanual() -> None:
    with pytest.raises(ValueError, match="not a bimanual pose message"):
        parse_bimanual_pose_message({"type": "pose", "schema_version": 1})


def test_page_mapping_mode_mismatch_is_rejected_per_hand() -> None:
    payload = bimanual_payload()
    payload["right"]["mapping_mode"] = "mirror"
    with pytest.raises(ValueError, match="right mapping mode"):
        parse_bimanual_pose_message(payload, mapping_modes={"right": "real"})


def test_untracked_hand_goes_stale_while_the_tracked_hand_stays_fresh() -> None:
    """The packet sequence keeps advancing while one controller is out of view.

    Counting that as a unique sample for the missing hand would report a stale
    controller as fresh and let its arm run on a frozen pose.
    """

    watchdogs = {hand: FreshSequenceWatchdog(0.05, 2) for hand in ("left", "right")}
    now_ns = 0
    # Three tracked frames, then long enough without the left controller for
    # its 50 ms freshness window to expire while the right keeps streaming.
    for step in range(24):
        now_ns += 5_000_000
        left_tracked = step < 3
        payload = bimanual_payload(sequence=100 + step)
        if not left_tracked:
            payload["left"] = {"tracked": False, "mapping_mode": "real"}
        sample = parse_bimanual_pose_message(payload)
        sample = type(sample)(
            **{**sample.__dict__, "pc_arrival_monotonic_ns": now_ns}
        )
        for hand in ("left", "right"):
            projected = sample.for_hand(hand)
            if projected is None:
                watchdogs[hand].poll(now_ns)
            else:
                watchdogs[hand].observe(projected, now_ns)

    assert watchdogs["right"].poll(now_ns).fresh is True
    assert watchdogs["left"].poll(now_ns).fresh is False


def test_bimanual_receiver_validates_its_configured_mapping_modes() -> None:
    receiver = BimanualQuestReceiver(
        "ws://127.0.0.1:8443/ws", mapping_modes={"left": "real", "right": "mirror"}
    )
    assert receiver.mapping_modes == {"left": "real", "right": "mirror"}

    with pytest.raises(ValueError, match="unsupported Quest hand"):
        BimanualQuestReceiver("ws://127.0.0.1:8443/ws", mapping_modes={"third": "real"})
    with pytest.raises(ValueError, match="unsupported Quest mapping mode"):
        BimanualQuestReceiver("ws://127.0.0.1:8443/ws", mapping_modes={"left": "flipped"})


def test_bimanual_receiver_counts_bad_messages_without_dying() -> None:
    receiver = BimanualQuestReceiver("ws://127.0.0.1:8443/ws")
    for raw in ("not json", json.dumps({"type": "pose"}), json.dumps(bimanual_payload())):
        try:
            receiver.mailbox.publish(receiver._parse(raw))
        except Exception:
            receiver.bad_messages += 1
    assert receiver.bad_messages == 2
    assert receiver.mailbox.peek()[0] is not None


def test_guided_calibration_projects_only_its_fixed_bimanual_hand() -> None:
    receiver = BimanualQuestReceiver(
        "ws://127.0.0.1:8443/ws", mapping_modes={"right": "real"}
    )
    stream = BimanualHandSampleStream(
        receiver,
        hand="right",
        mapping_mode="real",
    )
    receiver.mailbox.publish(parse_bimanual_pose_message(bimanual_payload()))

    sample = stream.next(0.0)

    assert sample is not None
    assert sample.hand == "right"
    assert sample.grip == pytest.approx(0.1)
    assert np.allclose(sample.controller_pose.position, [-1.0, 2.0, 3.0])


def test_guided_calibration_does_not_reuse_an_untracked_bimanual_hand() -> None:
    receiver = BimanualQuestReceiver(
        "ws://127.0.0.1:8443/ws", mapping_modes={"left": "real"}
    )
    stream = BimanualHandSampleStream(
        receiver,
        hand="left",
        mapping_mode="real",
    )
    payload = bimanual_payload()
    payload["left"] = {"tracked": False, "mapping_mode": "real"}
    receiver.mailbox.publish(parse_bimanual_pose_message(payload))

    assert stream.next(0.0) is None


# -- single-arm regression -------------------------------------------------


def test_single_arm_packet_parsing_is_unchanged() -> None:
    """The validated one-arm path must keep working exactly as before."""

    payload = {
        "type": "pose",
        "schema_version": 1,
        "selected_hand": "right",
        "mapping_mode": "real",
        "sequence": 7,
        "capture_monotonic_ms": 10.0,
        "capture_epoch_ms": 20.0,
        "send_monotonic_ms": 30.0,
        "right": {
            "position": [1.0, 2.0, 3.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "grip": 0.8,
            "trigger": 0.4,
        },
    }
    sample = parse_pose_message(payload, "right", "real")
    assert sample.hand == "right"
    assert sample.grip == pytest.approx(0.8)
    assert np.allclose(sample.controller_pose.position, [1.0, 2.0, 3.0])


def test_quest_page_keeps_single_arm_controls_and_adds_bimanual_mode() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    client = (WEB / "client.js").read_text(encoding="utf-8")

    # Existing single-arm controls must survive.
    assert 'id="hand-select"' in html
    assert 'id="mapping-mode-select"' in html
    assert "selected_hand: selectedHand" in client
    assert "mapping_mode: selectedMappingMode" in client
    assert 'schema_version: 1' in client

    # New bimanual controls and packet.
    assert 'id="input-mode-select"' in html
    assert 'value="bimanual"' in html
    assert 'id="left-mapping-mode-select"' in html
    assert 'id="right-mapping-mode-select"' in html
    assert 'type: "bimanual_pose"' in client
    assert "schema_version: 2" in client
    # Both hands must be captured in one frame callback.
    assert "onBimanualFrame" in client
    assert "bimanualControllerBlock" in client
    assert "message.hand" in client
    assert "source.handedness === requestedHand" in client
    # Tracking loss is explicit, not an early return that drops the frame.
    assert "tracked: false" in client
