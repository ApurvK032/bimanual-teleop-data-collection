from __future__ import annotations

import json
import threading
import time
from typing import Any

import numpy as np
from websockets.sync.client import connect

from .sample_buffer import LatestValueMailbox
from .types import BimanualQuestSample, ControllerSample, Pose, QuestSample


QUEST_HANDS = ("left", "right")
QUEST_MAPPING_MODES = ("real", "mirror")
BIMANUAL_SCHEMA_VERSION = 2
BIMANUAL_MESSAGE_TYPE = "bimanual_pose"


def _validated_selection(
    value: str | None,
    choices: tuple[str, ...],
    label: str,
) -> str | None:
    if value is None:
        return None
    selected = str(value).lower()
    if selected not in choices:
        raise ValueError(f"unsupported Quest {label}: {value}")
    return selected


def _reported_hand(payload: dict[str, Any], expected_hand: str | None) -> str:
    reported = payload.get("selected_hand")
    if reported is not None:
        selected = _validated_selection(str(reported), QUEST_HANDS, "hand")
        assert selected is not None
        return selected
    if expected_hand is not None:
        return expected_hand
    available = [hand for hand in QUEST_HANDS if isinstance(payload.get(hand), dict)]
    if len(available) != 1:
        raise ValueError("pose message must identify exactly one selected controller")
    return available[0]


def _mirror_webxr_pose(
    position: np.ndarray,
    orientation_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reflect a pose across WebXR's operator sagittal plane."""

    mirrored_position = position.copy()
    mirrored_position[0] *= -1.0
    mirrored_orientation = orientation_xyzw.copy()
    mirrored_orientation[1:3] *= -1.0
    return mirrored_position, mirrored_orientation


def _head_quaternion_wxyz(payload: dict[str, Any]) -> np.ndarray | None:
    # Prefer the session-locked operator frame from new clients. Older clients
    # and recordings still work through the live-head fallback.
    head = payload.get("operator_head")
    if not isinstance(head, dict) or "orientation_xyzw" not in head:
        head = payload.get("head")
    if not isinstance(head, dict) or "orientation_xyzw" not in head:
        return None
    head_xyzw = np.asarray(head["orientation_xyzw"], dtype=float).reshape(4)
    return np.array([head_xyzw[3], head_xyzw[0], head_xyzw[1], head_xyzw[2]])


def parse_pose_message(
    raw: str | dict[str, Any],
    hand: str | None = "left",
    mapping_mode: str | None = "real",
) -> QuestSample:
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if payload.get("type") != "pose":
        raise ValueError("not a pose message")
    expected_hand = _validated_selection(hand, QUEST_HANDS, "hand")
    expected_mapping_mode = _validated_selection(
        mapping_mode,
        QUEST_MAPPING_MODES,
        "mapping mode",
    )
    selected_hand = _reported_hand(payload, expected_hand)
    if expected_hand is not None and selected_hand != expected_hand:
        raise ValueError(
            f"Quest page selected {selected_hand}, expected {expected_hand}"
        )
    selected_mapping_mode = _validated_selection(
        str(payload.get("mapping_mode", "real")),
        QUEST_MAPPING_MODES,
        "mapping mode",
    )
    assert selected_mapping_mode is not None
    if (
        expected_mapping_mode is not None
        and selected_mapping_mode != expected_mapping_mode
    ):
        raise ValueError(
            "Quest page selected mapping mode "
            f"{selected_mapping_mode}, expected {expected_mapping_mode}"
        )
    controller_payload = payload.get(selected_hand)
    if not isinstance(controller_payload, dict):
        raise ValueError(f"pose message has no {selected_hand} controller")
    xyzw = np.asarray(controller_payload["orientation_xyzw"], dtype=float).reshape(4)
    position = np.asarray(controller_payload["position"], dtype=float).reshape(3)
    if selected_mapping_mode == "mirror":
        position, xyzw = _mirror_webxr_pose(position, xyzw)
    controller = Pose(
        position,
        np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float),
    )
    head_wxyz = _head_quaternion_wxyz(payload)
    return QuestSample(
        sequence=int(payload["sequence"]),
        capture_monotonic_ms=float(payload["capture_monotonic_ms"]),
        capture_epoch_ms=float(payload["capture_epoch_ms"]),
        send_monotonic_ms=float(payload["send_monotonic_ms"]),
        pc_arrival_monotonic_ns=time.perf_counter_ns(),
        pc_arrival_epoch_ns=time.time_ns(),
        reconnect_generation=int(payload.get("reconnect_generation", 0)),
        controller_pose=controller,
        grip=float(controller_payload.get("grip", 0.0)),
        trigger=float(controller_payload.get("trigger", 0.0)),
        hand=selected_hand,
        mapping_mode=selected_mapping_mode,
        head_quaternion_wxyz=head_wxyz,
    )


def _parse_bimanual_controller(
    payload: Any,
    hand: str,
    expected_mapping_mode: str | None,
) -> ControllerSample:
    """Parse one hand of a bimanual frame, representing loss explicitly."""

    if payload is None or not isinstance(payload, dict):
        # An absent controller block is untracked, not an error. The other arm
        # must keep running when one controller sleeps or leaves the volume.
        return ControllerSample(
            hand=hand,
            tracked=False,
            mapping_mode=expected_mapping_mode or "real",
        )
    mapping_mode = _validated_selection(
        str(payload.get("mapping_mode", expected_mapping_mode or "real")),
        QUEST_MAPPING_MODES,
        "mapping mode",
    )
    assert mapping_mode is not None
    if expected_mapping_mode is not None and mapping_mode != expected_mapping_mode:
        raise ValueError(
            f"Quest page selected {hand} mapping mode {mapping_mode}, "
            f"expected {expected_mapping_mode}"
        )
    if not bool(payload.get("tracked", True)):
        return ControllerSample(hand=hand, tracked=False, mapping_mode=mapping_mode)
    position = np.asarray(payload["position"], dtype=float).reshape(3)
    xyzw = np.asarray(payload["orientation_xyzw"], dtype=float).reshape(4)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(xyzw)):
        raise ValueError(f"{hand} controller pose is non-finite")
    if mapping_mode == "mirror":
        position, xyzw = _mirror_webxr_pose(position, xyzw)
    return ControllerSample(
        hand=hand,
        tracked=True,
        mapping_mode=mapping_mode,
        controller_pose=Pose(
            position,
            np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float),
        ),
        grip=float(payload.get("grip", 0.0)),
        trigger=float(payload.get("trigger", 0.0)),
    )


def parse_bimanual_pose_message(
    raw: str | dict[str, Any],
    *,
    mapping_modes: dict[str, str] | None = None,
) -> BimanualQuestSample:
    """Parse one schema-v2 frame carrying both controllers.

    The whole packet is rejected when its shared sequence or timestamps are
    invalid, because those are what align the two arms in time. Per-controller
    tracking loss is represented on the controller instead.
    """

    payload = json.loads(raw) if isinstance(raw, str) else raw
    if payload.get("type") != BIMANUAL_MESSAGE_TYPE:
        raise ValueError("not a bimanual pose message")
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != BIMANUAL_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported bimanual schema version {schema_version}; "
            f"expected {BIMANUAL_SCHEMA_VERSION}"
        )
    sequence = int(payload["sequence"])
    capture_monotonic_ms = float(payload["capture_monotonic_ms"])
    capture_epoch_ms = float(payload["capture_epoch_ms"])
    send_monotonic_ms = float(payload["send_monotonic_ms"])
    shared_timestamps = (capture_monotonic_ms, capture_epoch_ms, send_monotonic_ms)
    if sequence < 0 or not all(np.isfinite(value) for value in shared_timestamps):
        raise ValueError("bimanual packet has an invalid shared sequence or timestamp")

    expected_modes = dict(mapping_modes or {})
    controllers: dict[str, ControllerSample] = {}
    for hand in QUEST_HANDS:
        expected_mode = _validated_selection(
            expected_modes.get(hand),
            QUEST_MAPPING_MODES,
            "mapping mode",
        )
        controllers[hand] = _parse_bimanual_controller(
            payload.get(hand),
            hand,
            expected_mode,
        )
    home_button = payload.get("home_button")
    if not isinstance(home_button, bool) or any(
        controllers[hand].tracked and "grip" not in payload[hand] for hand in QUEST_HANDS
    ):
        home_button = None
    return BimanualQuestSample(
        sequence=sequence,
        capture_monotonic_ms=capture_monotonic_ms,
        capture_epoch_ms=capture_epoch_ms,
        send_monotonic_ms=send_monotonic_ms,
        pc_arrival_monotonic_ns=time.perf_counter_ns(),
        pc_arrival_epoch_ns=time.time_ns(),
        reconnect_generation=int(payload.get("reconnect_generation", 0)),
        controllers=controllers,
        head_quaternion_wxyz=_head_quaternion_wxyz(payload),
        home_button=home_button,
    )


class _MailboxWebSocketReceiver:
    """Reconnecting WebSocket subscriber backed by a capacity-one mailbox.

    Subclasses supply only ``_parse``. The socket, reconnect, and mailbox
    behaviour stays identical between the single-arm and bimanual paths so the
    validated stale/reconnect semantics are not reimplemented twice.
    """

    _THREAD_NAME = "quest-websocket"

    def __init__(self, websocket_url: str) -> None:
        self.websocket_url = websocket_url
        self.mailbox: LatestValueMailbox[Any] = LatestValueMailbox()
        self.bad_messages = 0
        self.reconnects = 0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._websocket_lock = threading.Lock()
        self._websocket = None

    def _parse(self, raw: str) -> Any:
        raise NotImplementedError

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._THREAD_NAME, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._websocket_lock:
            websocket = self._websocket
        if websocket is not None:
            try:
                websocket.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def publish(self, payload: dict[str, Any]) -> bool:
        """Publish a control-state event through the relay connection."""

        with self._websocket_lock:
            websocket = self._websocket
        if websocket is None:
            return False
        try:
            websocket.send(json.dumps(payload, separators=(",", ":")))
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with connect(self.websocket_url, open_timeout=2.0, close_timeout=1.0) as websocket:
                    with self._websocket_lock:
                        self._websocket = websocket
                    try:
                        self.reconnects += 1
                        self.last_error = None
                        for raw in websocket:
                            if self._stop.is_set():
                                break
                            try:
                                sample = self._parse(raw)
                            except (
                                ValueError,
                                KeyError,
                                TypeError,
                                AssertionError,
                                json.JSONDecodeError,
                            ):
                                self.bad_messages += 1
                                continue
                            self.mailbox.publish(sample)
                    finally:
                        with self._websocket_lock:
                            if self._websocket is websocket:
                                self._websocket = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._stop.wait(0.5)


class QuestReceiver(_MailboxWebSocketReceiver):
    """Single-arm subscriber: one selected controller per packet."""

    def __init__(
        self,
        websocket_url: str,
        *,
        hand: str | None = "left",
        mapping_mode: str | None = "real",
    ) -> None:
        super().__init__(websocket_url)
        self.hand = _validated_selection(hand, QUEST_HANDS, "hand")
        self.mapping_mode = _validated_selection(
            mapping_mode,
            QUEST_MAPPING_MODES,
            "mapping mode",
        )
        self.mailbox: LatestValueMailbox[QuestSample] = LatestValueMailbox()

    def _parse(self, raw: str) -> QuestSample:
        return parse_pose_message(raw, self.hand, self.mapping_mode)


class BimanualQuestReceiver(_MailboxWebSocketReceiver):
    """Bimanual subscriber: both controllers arrive in one capacity-one sample.

    One mailbox is deliberate. Two mailboxes would let the arms drift apart in
    time and would make "which controller does this arm's command belong to"
    a timing question instead of a packet question.
    """

    _THREAD_NAME = "quest-bimanual-websocket"

    def __init__(
        self,
        websocket_url: str,
        *,
        mapping_modes: dict[str, str] | None = None,
    ) -> None:
        super().__init__(websocket_url)
        modes = dict(mapping_modes or {})
        for hand, mode in modes.items():
            if hand not in QUEST_HANDS:
                raise ValueError(f"unsupported Quest hand: {hand}")
            _validated_selection(mode, QUEST_MAPPING_MODES, "mapping mode")
        self.mapping_modes = modes
        self.mailbox: LatestValueMailbox[BimanualQuestSample] = LatestValueMailbox()

    def _parse(self, raw: str) -> BimanualQuestSample:
        return parse_bimanual_pose_message(raw, mapping_modes=self.mapping_modes)
