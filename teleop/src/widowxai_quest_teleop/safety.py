from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np

from .types import QuestSample


class TimeAlignedCommandHistory:
    """Interpolate the command that encoder feedback should have reached.

    Non-blocking position commands are intentionally interpolated by the arm
    controller. Comparing feedback with the newest command therefore reports
    ordinary interpolation/transport lag as tracking error. This short history
    provides a delayed command reference while retaining a bounded stall check.
    """

    def __init__(
        self,
        initial_command: np.ndarray,
        initial_time_s: float,
        delay_s: float,
        *,
        retention_s: float = 0.5,
    ) -> None:
        command = np.asarray(initial_command, dtype=float).reshape(-1)
        timestamp = float(initial_time_s)
        delay = float(delay_s)
        retention = float(retention_s)
        if not np.all(np.isfinite(command)):
            raise ValueError("initial command must be finite")
        if not np.isfinite(timestamp):
            raise ValueError("initial command time must be finite")
        if not np.isfinite(delay) or delay < 0.0:
            raise ValueError("tracking delay must be finite and nonnegative")
        if not np.isfinite(retention) or retention <= delay:
            raise ValueError("command-history retention must exceed the tracking delay")
        self.delay_s = delay
        self.retention_s = retention
        self._times: deque[float] = deque([timestamp])
        self._commands: deque[np.ndarray] = deque([command.copy()])

    def append(self, timestamp_s: float, command: np.ndarray) -> None:
        timestamp = float(timestamp_s)
        value = np.asarray(command, dtype=float).reshape(self._commands[0].shape)
        if not np.isfinite(timestamp) or timestamp <= self._times[-1]:
            raise ValueError("command timestamps must be finite and strictly increasing")
        if not np.all(np.isfinite(value)):
            raise ValueError("command must be finite")
        self._times.append(timestamp)
        self._commands.append(value.copy())
        cutoff = timestamp - self.retention_s
        while len(self._times) > 2 and self._times[1] < cutoff:
            self._times.popleft()
            self._commands.popleft()

    @property
    def newest_time_s(self) -> float:
        """Timestamp of the most recent appended command."""

        return self._times[-1]

    @property
    def oldest_time_s(self) -> float:
        """Timestamp of the oldest retained command."""

        return self._times[0]

    def clamp_state(self, feedback_time_s: float) -> str:
        """Describe how ``reference_at`` would resolve this feedback time.

        A tracking-error fault reads very differently depending on whether the
        reference was interpolated or frozen at an end of the history, so the
        fault message reports it rather than leaving it to be guessed.
        """

        target_time = float(feedback_time_s) - self.delay_s
        if target_time <= self._times[0]:
            return "clamped-to-oldest"
        if target_time >= self._times[-1]:
            return "clamped-to-newest"
        return "interpolated"

    def reference_at(self, feedback_time_s: float) -> np.ndarray:
        feedback_time = float(feedback_time_s)
        if not np.isfinite(feedback_time):
            raise ValueError("feedback time must be finite")
        target_time = feedback_time - self.delay_s
        if target_time <= self._times[0]:
            return self._commands[0].copy()
        if target_time >= self._times[-1]:
            return self._commands[-1].copy()
        for index in range(len(self._times) - 1):
            before_time = self._times[index]
            after_time = self._times[index + 1]
            if target_time <= after_time:
                alpha = (target_time - before_time) / (after_time - before_time)
                return (1.0 - alpha) * self._commands[index] + alpha * self._commands[index + 1]
        raise RuntimeError("failed to interpolate command history")


@dataclass(frozen=True)
class FreshnessStatus:
    fresh: bool
    unique_samples: int
    repeated_samples: int
    age_s: float
    reconnect_generation: int


class FreshSequenceWatchdog:
    def __init__(self, timeout_s: float, fresh_samples_to_recover: int = 3) -> None:
        self.timeout_ns = int(float(timeout_s) * 1e9)
        self.fresh_samples_to_recover = max(1, int(fresh_samples_to_recover))
        self.last_sequence: int | None = None
        self.last_unique_arrival_ns: int | None = None
        self.last_reconnect_generation = -1
        self.unique_samples = 0
        self.repeated_samples = 0
        self._recovery_streak = 0
        self._fresh = False

    def observe(self, sample: QuestSample, now_ns: int | None = None) -> FreshnessStatus:
        generation_changed = sample.reconnect_generation != self.last_reconnect_generation
        unique = generation_changed or sample.sequence != self.last_sequence
        if generation_changed:
            self._fresh = False
            self._recovery_streak = 0
            self.last_reconnect_generation = sample.reconnect_generation
        if unique:
            self.unique_samples += 1
            self._recovery_streak += 1
            self.last_sequence = sample.sequence
            self.last_unique_arrival_ns = sample.pc_arrival_monotonic_ns
            if self._recovery_streak >= self.fresh_samples_to_recover:
                self._fresh = True
        else:
            self.repeated_samples += 1
        # Judge freshness when the control loop consumes the sample, not when the
        # receiver thread put it in the mailbox. A delayed mailbox sample must
        # never receive one apparently-fresh control tick.
        return self.poll(now_ns)

    def poll(self, now_ns: int | None = None) -> FreshnessStatus:
        now = perf_counter_ns() if now_ns is None else int(now_ns)
        age_ns = self.timeout_ns + 1 if self.last_unique_arrival_ns is None else now - self.last_unique_arrival_ns
        if age_ns > self.timeout_ns:
            self._fresh = False
            self._recovery_streak = 0
        return FreshnessStatus(
            fresh=self._fresh,
            unique_samples=self.unique_samples,
            repeated_samples=self.repeated_samples,
            age_s=max(0.0, age_ns / 1e9),
            reconnect_generation=self.last_reconnect_generation,
        )
