"""Simulation-only reproduction of the rejected measured-feedback load yield.

Nothing in this module is imported by a hardware launcher.  It intentionally
recreates the controller that oscillated on 2026-08-14 so delayed MuJoCo
feedback can expose the same failure before a replacement is designed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoadYieldDecision:
    active: bool
    triggered: bool
    rearmed: bool
    phase: str
    consecutive_high_samples: int
    max_error_rad: float


class ExperimentalSoftLoadGuard:
    """Exact state shape of the physically rejected dynamic yield.

    Once active, the coordinator continually replaces the recovery destination
    with the newest delayed encoder sample.  This is deliberately *not* the
    fixed-latch remedy proposed after the physical failure.
    """

    def __init__(
        self,
        *,
        threshold_rad: float = 0.045,
        consecutive_samples: int = 3,
        max_yield_delta_rad: float = 0.006,
        grip_threshold: float = 0.65,
    ) -> None:
        self.threshold_rad = float(threshold_rad)
        self.consecutive_samples = int(consecutive_samples)
        self.max_yield_delta_rad = float(max_yield_delta_rad)
        self.grip_threshold = float(grip_threshold)
        if not np.isfinite(self.threshold_rad) or self.threshold_rad <= 0.0:
            raise ValueError("load-yield threshold must be finite and positive")
        if self.consecutive_samples < 1:
            raise ValueError("load-yield persistence must be at least one sample")
        if (
            not np.isfinite(self.max_yield_delta_rad)
            or self.max_yield_delta_rad <= 0.0
        ):
            raise ValueError("load-yield delta must be finite and positive")
        if not np.isfinite(self.grip_threshold):
            raise ValueError("load-yield grip threshold must be finite")
        self.active = False
        self.release_seen = False
        self.consecutive_high = 0
        self.activations = 0
        self.phase = "monitoring"
        self.max_error_rad = 0.0

    def observe(
        self,
        feedback_error_by_side: dict[str, np.ndarray],
        *,
        feedback_sample_fresh: bool,
        grip_by_side: dict[str, float],
    ) -> LoadYieldDecision:
        errors = [
            np.asarray(value, dtype=float).reshape(6)
            for value in feedback_error_by_side.values()
        ]
        if len(errors) != 2 or any(not np.all(np.isfinite(error)) for error in errors):
            raise ValueError("load-yield guard requires two finite six-joint errors")
        if set(grip_by_side) != set(feedback_error_by_side):
            raise ValueError("load-yield guard requires one grip value per arm")
        max_error = max(float(np.max(np.abs(error))) for error in errors)
        self.max_error_rad = max_error
        both_pressed = all(
            float(grip_by_side[side]) >= self.grip_threshold
            for side in feedback_error_by_side
        )
        both_released = all(
            float(grip_by_side[side]) < self.grip_threshold
            for side in feedback_error_by_side
        )
        triggered = False
        rearmed = False

        if not self.active:
            if feedback_sample_fresh and both_pressed and max_error >= self.threshold_rad:
                self.consecutive_high += 1
            elif feedback_sample_fresh:
                self.consecutive_high = 0
            if self.consecutive_high >= self.consecutive_samples:
                self.active = True
                self.release_seen = False
                self.activations += 1
                self.phase = "load_yield_wait_release"
                triggered = True
        else:
            if both_released:
                self.release_seen = True
                self.phase = "load_yield_wait_regrip"
            elif self.release_seen and both_pressed:
                self.active = False
                self.release_seen = False
                self.consecutive_high = 0
                self.phase = "rearmed"
                rearmed = True

        if not self.active and not rearmed:
            self.phase = "monitoring"
        return LoadYieldDecision(
            active=self.active,
            triggered=triggered,
            rearmed=rearmed,
            phase=self.phase,
            consecutive_high_samples=self.consecutive_high,
            max_error_rad=max_error,
        )


class DelayedEncoderPlant:
    """Small delayed first-order plant for Quest-driven MuJoCo stress tests."""

    def __init__(
        self,
        initial_q_by_side: dict[str, np.ndarray],
        *,
        command_delay_s: float,
        response_time_constant_s: float,
        history_s: float = 1.0,
    ) -> None:
        self.command_delay_s = float(command_delay_s)
        self.response_time_constant_s = float(response_time_constant_s)
        self.history_s = float(history_s)
        if self.command_delay_s < 0.0 or not np.isfinite(self.command_delay_s):
            raise ValueError("simulated command delay must be finite and nonnegative")
        if (
            self.response_time_constant_s <= 0.0
            or not np.isfinite(self.response_time_constant_s)
        ):
            raise ValueError("simulated response time constant must be positive")
        if self.history_s <= self.command_delay_s:
            raise ValueError("simulated command history must exceed its delay")
        self.q = {
            side: np.asarray(value, dtype=float).reshape(6).copy()
            for side, value in initial_q_by_side.items()
        }
        if len(self.q) != 2 or any(not np.all(np.isfinite(value)) for value in self.q.values()):
            raise ValueError("simulated encoder plant requires two finite arm states")
        self._history: dict[str, deque[tuple[float, np.ndarray]]] = {
            side: deque() for side in self.q
        }

    def append_command(
        self, timestamp_s: float, q_command_by_side: dict[str, np.ndarray]
    ) -> None:
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("simulated command timestamp must be finite")
        for side in self.q:
            value = np.asarray(q_command_by_side[side], dtype=float).reshape(6)
            history = self._history[side]
            if history and timestamp <= history[-1][0]:
                raise ValueError("simulated command timestamps must increase")
            history.append((timestamp, value.copy()))
            cutoff = timestamp - self.history_s
            while len(history) > 2 and history[1][0] < cutoff:
                history.popleft()

    def _delayed_command(self, side: str, timestamp_s: float) -> np.ndarray:
        history = self._history[side]
        if not history:
            return self.q[side].copy()
        target = float(timestamp_s) - self.command_delay_s
        if target <= history[0][0]:
            return history[0][1].copy()
        if target >= history[-1][0]:
            return history[-1][1].copy()
        for index in range(len(history) - 1):
            before_t, before_q = history[index]
            after_t, after_q = history[index + 1]
            if target <= after_t:
                alpha = (target - before_t) / (after_t - before_t)
                return (1.0 - alpha) * before_q + alpha * after_q
        raise RuntimeError("failed to interpolate delayed simulated command")

    def advance(self, timestamp_s: float, dt_s: float) -> dict[str, np.ndarray]:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("simulated encoder timestep must be finite and positive")
        alpha = 1.0 - np.exp(-dt / self.response_time_constant_s)
        for side in self.q:
            target = self._delayed_command(side, timestamp_s)
            self.q[side] += alpha * (target - self.q[side])
        return {side: value.copy() for side, value in self.q.items()}
