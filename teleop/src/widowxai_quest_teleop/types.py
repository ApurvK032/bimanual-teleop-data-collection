from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math3d import normalize_quat


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    quaternion_wxyz: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", np.asarray(self.position, dtype=float).reshape(3).copy())
        object.__setattr__(self, "quaternion_wxyz", normalize_quat(self.quaternion_wxyz))


@dataclass(frozen=True)
class QuestSample:
    sequence: int
    capture_monotonic_ms: float
    capture_epoch_ms: float
    send_monotonic_ms: float
    pc_arrival_monotonic_ns: int
    pc_arrival_epoch_ns: int
    reconnect_generation: int
    controller_pose: Pose
    grip: float
    trigger: float
    hand: str
    mapping_mode: str
    head_quaternion_wxyz: np.ndarray | None = None


@dataclass(frozen=True)
class ControllerSample:
    """One controller's contribution to a single bimanual WebXR frame.

    ``tracked`` is explicit rather than implied by a reused pose. A controller
    that lost optical tracking arrives with ``tracked=False`` and no pose, so a
    coordinator can hold that arm alone instead of replaying a stale command.
    """

    hand: str
    tracked: bool
    mapping_mode: str
    controller_pose: Pose | None = None
    grip: float = 0.0
    trigger: float = 0.0


@dataclass(frozen=True)
class BimanualQuestSample:
    """Both controllers captured in one WebXR frame under one sequence number.

    The shared sequence and capture timestamps are what make the two arms
    frame-synchronous. Two independently timed single-hand streams cannot
    provide this, which is why the dual-arm runtime never merges two mailboxes.
    """

    sequence: int
    capture_monotonic_ms: float
    capture_epoch_ms: float
    send_monotonic_ms: float
    pc_arrival_monotonic_ns: int
    pc_arrival_epoch_ns: int
    reconnect_generation: int
    controllers: dict[str, ControllerSample]
    head_quaternion_wxyz: np.ndarray | None = None
    home_button: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "controllers", dict(self.controllers))

    def controller(self, hand: str) -> ControllerSample | None:
        return self.controllers.get(str(hand).lower())

    def tracked_hands(self) -> tuple[str, ...]:
        return tuple(
            hand
            for hand, controller in sorted(self.controllers.items())
            if controller.tracked
        )

    def for_hand(self, hand: str) -> QuestSample | None:
        """Project one tracked controller onto the validated single-arm sample.

        Every downstream primitive — clutch, mapper, pose filter, freshness
        watchdog — already consumes a ``QuestSample``. Projecting instead of
        reimplementing keeps the dual-arm path on exactly the control code the
        single-arm profile validated.
        """

        controller = self.controller(hand)
        if controller is None or not controller.tracked or controller.controller_pose is None:
            return None
        return QuestSample(
            sequence=self.sequence,
            capture_monotonic_ms=self.capture_monotonic_ms,
            capture_epoch_ms=self.capture_epoch_ms,
            send_monotonic_ms=self.send_monotonic_ms,
            pc_arrival_monotonic_ns=self.pc_arrival_monotonic_ns,
            pc_arrival_epoch_ns=self.pc_arrival_epoch_ns,
            reconnect_generation=self.reconnect_generation,
            controller_pose=controller.controller_pose,
            grip=controller.grip,
            trigger=controller.trigger,
            hand=controller.hand,
            mapping_mode=controller.mapping_mode,
            head_quaternion_wxyz=self.head_quaternion_wxyz,
        )
