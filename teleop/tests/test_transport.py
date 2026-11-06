from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from widowxai_quest_teleop import transport


class BlockingWebSocket:
    def __init__(self) -> None:
        self.closed = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def __iter__(self):
        self.closed.wait(5.0)
        return iter(())

    def close(self) -> None:
        self.closed.set()


def test_receiver_stop_interrupts_an_idle_websocket(monkeypatch) -> None:
    websocket = BlockingWebSocket()
    monkeypatch.setattr(transport, "connect", lambda *_args, **_kwargs: websocket)
    receiver = transport.QuestReceiver("ws://test.invalid/ws")
    receiver.start()

    deadline = time.perf_counter() + 1.0
    while receiver.reconnects == 0 and time.perf_counter() < deadline:
        time.sleep(0.001)
    assert receiver.reconnects == 1

    started = time.perf_counter()
    receiver.stop()
    assert time.perf_counter() - started < 0.25
    assert receiver._thread is not None
    assert not receiver._thread.is_alive()


def test_receiver_publishes_compact_ik_state() -> None:
    sent = []
    receiver = transport.QuestReceiver("ws://test.invalid/ws")
    receiver._websocket = SimpleNamespace(send=sent.append)
    assert receiver.publish({"type": "ik_state", "q_arm": [0] * 6, "gripper_q": 0.044})
    assert sent == ['{"type":"ik_state","q_arm":[0,0,0,0,0,0],"gripper_q":0.044}']


def test_receiver_accepts_right_controller_selection() -> None:
    receiver = transport.QuestReceiver(
        "ws://test.invalid/ws",
        hand="right",
        mapping_mode="mirror",
    )
    assert receiver.hand == "right"
    assert receiver.mapping_mode == "mirror"


def test_receiver_can_follow_page_input_selection() -> None:
    receiver = transport.QuestReceiver(
        "ws://test.invalid/ws",
        hand=None,
        mapping_mode=None,
    )
    assert receiver.hand is None
    assert receiver.mapping_mode is None


def test_quest_page_exposes_hand_and_mapping_controls() -> None:
    web_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "widowxai_quest_teleop"
        / "web"
    )
    html = (web_root / "index.html").read_text(encoding="utf-8")
    javascript = (web_root / "client.js").read_text(encoding="utf-8")

    assert 'id="hand-select"' in html
    assert 'id="mapping-mode-select"' in html
    assert 'value="real"' in html
    assert 'value="mirror"' in html
    assert "Behind / Parallel — matched motion" in html
    assert "Mirrored — flip left/right, front/back, screw, nod-no" in html
    assert "selected_hand: selectedHand" in javascript
    assert "mapping_mode: selectedMappingMode" in javascript
