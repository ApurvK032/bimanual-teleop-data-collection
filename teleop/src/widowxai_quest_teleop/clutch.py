from __future__ import annotations

import numpy as np

from .mapping import ClutchPoseMapper
from .types import Pose


class ClutchController:
    """Reference-kit grip-edge state machine with zero-delta stale recovery."""

    def __init__(self, mapper: ClutchPoseMapper, grip_threshold: float = 0.65) -> None:
        self.mapper = mapper
        self.grip_threshold = float(grip_threshold)
        self._was_pressed = False
        self._needs_reanchor = False

    def force_reanchor(self) -> None:
        self._needs_reanchor = self._needs_reanchor or self.mapper.engaged
        self.mapper.release()

    def update(
        self,
        *,
        grip: float,
        stream_fresh: bool,
        controller_pose: Pose,
        robot_pose: Pose,
        wrist_pivot: np.ndarray,
        head_quaternion_wxyz: np.ndarray | None = None,
    ) -> Pose | None:
        pressed = float(grip) >= self.grip_threshold
        if not stream_fresh:
            if pressed:
                self.force_reanchor()
            else:
                self._needs_reanchor = False
                self.mapper.release()
            self._was_pressed = pressed
            return None
        if not pressed:
            self._needs_reanchor = False
            if self.mapper.engaged:
                self.mapper.release()
            self._was_pressed = False
            return None
        if self._needs_reanchor or not self._was_pressed or not self.mapper.engaged:
            self.mapper.engage(
                controller_pose,
                robot_pose,
                wrist_pivot,
                head_quaternion_wxyz=head_quaternion_wxyz,
            )
            self._needs_reanchor = False
        self._was_pressed = True
        return self.mapper.update(controller_pose, robot_pose)
