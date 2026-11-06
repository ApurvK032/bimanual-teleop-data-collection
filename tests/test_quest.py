"""Headset side: adb device parsing, the relay observer, and the 'is the headset ready' rule."""
from __future__ import annotations

import json

from bimanual_collect.quest import parse_devices, ready_for, usb_state

ADB = """List of devices attached
2G0YC1ZF8B0123   device usb:3-2 product:eureka model:Quest_3 device:eureka transport_id:4
1WMHH000000000   unauthorized usb:3-1 transport_id:5
3HXXX            no permissions (missing udev rules? user is in the plugdev group); see [...] usb:1-1
"""


def test_parse_devices():
    devices = parse_devices(ADB)
    assert [d["state"] for d in devices] == ["device", "unauthorized", "no_permission"]
    assert devices[0]["model"] == "Quest_3" and devices[0]["product"] == "eureka"
    assert [usb_state(d) for d in devices] == ["connected", "unauthorized", "no_permission"]
    assert usb_state(None) == "none" and usb_state({"state": "recovery"}) == "offline"


def page(mode="bimanual", **hands):
    base = {"left": {"tracked": True, "grip": 0.0, "mapping": "real"},
            "right": {"tracked": True, "grip": 0.0, "mapping": "real"}}
    base.update(hands)
    return {"pages": [{"agent": "OculusBrowser"}], "streaming": True, "mode": mode, "hands": base}


def test_ready_for():
    assert ready_for(page(), "bimanual") == (True, "Headset ready.")
    assert not ready_for(None, "bimanual")[0]
    assert "Open" in ready_for({"pages": [], "streaming": False}, "bimanual")[1]
    assert "Enter" in ready_for({"pages": [{}], "streaming": False}, "bimanual")[1]
    assert "Bimanual" in ready_for(page(mode="single left"), "bimanual")[1]
    assert "left controller is not tracked" in ready_for(page(left={"tracked": False}), "bimanual")[1]
    assert "Release the right grip" in ready_for(page(right={"tracked": True, "grip": 0.9, "mapping": "real"}), "bimanual")[1]
    assert "Behind" in ready_for(page(left={"tracked": True, "grip": 0, "mapping": "mirror"}), "bimanual")[1]
    single = page(mode="single right", left=None)
    assert ready_for(single, "right")[0]


def test_headset_watch_observes_packets():
    from bimanual_collect.quest_page import HeadsetWatch

    watch = HeadsetWatch()
    source = object()
    watch.clients[source] = {"agent": "Mozilla/5.0 (Quest 3) OculusBrowser/40", "since": 0, "page": True}
    packet = {"type": "bimanual_pose", "sequence": 7, "left": {"tracked": True, "grip": 0.0, "mapping_mode": "real"},
              "right": {"tracked": False, "grip": 0.5, "mapping_mode": "real"}}
    watch.observe(json.dumps(packet), source)
    status = watch.status()
    assert status["streaming"] and status["mode"] == "bimanual" and status["from_headset"]
    assert status["hands"]["right"] == {"tracked": False, "grip": 0.5, "mapping": "real"}
    single = {"type": "pose", "selected_hand": "left", "mapping_mode": "real", "left": {"grip": 1.0}}
    watch.observe(json.dumps(single), source)
    status = watch.status()
    assert status["mode"] == "single left" and status["hands"]["right"] is None
    assert status["hands"]["left"]["grip"] == 1.0 and status["hands"]["left"]["tracked"]
