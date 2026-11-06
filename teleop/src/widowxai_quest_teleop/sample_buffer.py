from __future__ import annotations

from threading import Condition
from time import monotonic
from typing import Generic, TypeVar


T = TypeVar("T")


class LatestValueMailbox(Generic[T]):
    """Capacity-one mailbox: publishing always overwrites stale state."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._value: T | None = None
        self._generation = 0
        self.overwrite_count = 0

    def publish(self, value: T) -> int:
        with self._condition:
            if self._value is not None:
                self.overwrite_count += 1
            self._value = value
            self._generation += 1
            self._condition.notify_all()
            return self._generation

    def take_latest(self) -> tuple[T | None, int]:
        with self._condition:
            value = self._value
            self._value = None
            return value, self._generation

    def peek(self) -> tuple[T | None, int]:
        with self._condition:
            return self._value, self._generation

    def wait_for_newer(self, generation: int, timeout_s: float | None = None) -> tuple[T | None, int]:
        deadline = None if timeout_s is None else monotonic() + timeout_s
        with self._condition:
            while self._generation <= generation:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0.0:
                    return None, self._generation
                self._condition.wait(remaining)
            return self._value, self._generation

    def wait_take_latest(
        self,
        generation: int,
        timeout_s: float | None = None,
    ) -> tuple[T | None, int]:
        """Wait for a newer generation and atomically consume its latest value."""

        deadline = None if timeout_s is None else monotonic() + timeout_s
        with self._condition:
            while self._generation <= generation:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0.0:
                    return None, self._generation
                self._condition.wait(remaining)
            value = self._value
            self._value = None
            return value, self._generation
