from __future__ import annotations

from typing import cast

import pytest
from fastapi import WebSocket

from scripts.smoke_test_relay import run_smoke_test
from widowxai_quest_teleop.relay import Client, LatestStateHub


def test_latest_state_queue_overwrites_old_bimanual_pose() -> None:
    hub = LatestStateHub()
    source = Client(cast(WebSocket, object()), generation=1)
    subscriber = Client(cast(WebSocket, object()), generation=2)
    hub.clients.update((source, subscriber))

    hub.publish_pose("old", source, message_type="bimanual_pose")
    hub.publish_pose("new", source, message_type="bimanual_pose")

    assert subscriber.pose_queue.qsize() == 1
    assert subscriber.pose_queue.get_nowait() == "new"
    assert subscriber.event_queue.empty()
    assert hub.pose_messages == 2
    assert hub.pose_messages_by_type == {"pose": 0, "bimanual_pose": 2}
    assert hub.overwritten_pose_messages == 1


def test_latest_state_queue_rejects_unknown_message_type() -> None:
    hub = LatestStateHub()
    source = Client(cast(WebSocket, object()), generation=1)

    with pytest.raises(ValueError, match="unsupported latest-state message type"):
        hub.publish_pose("bad", source, message_type="controller_event")


def test_real_websocket_relay_handles_single_bimanual_and_reconnect() -> None:
    health = run_smoke_test()

    assert health["pose_messages"] == 3
    assert health["pose_messages_by_type"] == {"pose": 1, "bimanual_pose": 2}
