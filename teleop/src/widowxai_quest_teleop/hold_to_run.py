from __future__ import annotations

import os
import platform
import stat
import time
from dataclasses import dataclass

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on native Windows
    fcntl = None  # type: ignore[assignment]


EV_KEY = 0x01
KEY_MAX = 0x2FF
KEY_STATE_BYTES = (KEY_MAX + 8) // 8

_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_READ = 2


class HoldToRunError(RuntimeError):
    pass


@dataclass(frozen=True)
class HoldToRunSample:
    pressed: bool
    monotonic_ns: int
    press_generation: int
    release_generation: int


def _ioc(direction: int, type_: int, number: int, size: int) -> int:
    return (
        (direction << _IOC_DIRSHIFT)
        | (type_ << _IOC_TYPESHIFT)
        | (number << _IOC_NRSHIFT)
        | (size << _IOC_SIZESHIFT)
    )


def _eviocgkey(size: int) -> int:
    return _ioc(_IOC_READ, ord("E"), 0x18, size)


def _eviocgbit(event_type: int, size: int) -> int:
    return _ioc(_IOC_READ, ord("E"), 0x20 + event_type, size)


def _bit_is_set(bits: bytes | bytearray, code: int) -> bool:
    return bool(bits[code // 8] & (1 << (code % 8)))


def validate_evdev_key_code(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HoldToRunError("hold-to-run key code must be an integer")
    code = int(value)
    if not 0 <= code <= KEY_MAX:
        raise HoldToRunError(
            f"hold-to-run key code must be in the Linux range [0, {KEY_MAX}]"
        )
    return code


class LinuxEvdevHoldToRun:
    """Poll an authoritative Linux EV_KEY state on every control iteration.

    This intentionally does not infer a held key from auto-repeat events. The
    EVIOCGKEY ioctl asks the kernel for the current physical state, so a key-up
    is visible even when the terminal has not received a line. Any open or
    ioctl failure is fail-closed.
    """

    def __init__(self, device: str, key_code: int) -> None:
        path = str(device).strip()
        if not path:
            raise HoldToRunError("hold-to-run evdev device path is required")
        self.device = path
        self.key_code = validate_evdev_key_code(key_code)
        self._fd: int | None = None
        self._last_pressed: bool | None = None
        self._press_generation = 0
        self._release_generation = 0

    @property
    def opened(self) -> bool:
        return self._fd is not None

    def open(self, *, require_released: bool = True) -> HoldToRunSample:
        if platform.system() != "Linux" or fcntl is None:
            raise HoldToRunError("physical hold-to-run input requires Linux evdev")
        if self._fd is not None:
            raise HoldToRunError("hold-to-run device is already open")
        flags = os.O_RDONLY | os.O_NONBLOCK
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            fd = os.open(self.device, flags)
        except OSError as exc:
            raise HoldToRunError(
                f"cannot open hold-to-run device {self.device}: {exc}"
            ) from None
        self._fd = fd
        try:
            if not stat.S_ISCHR(os.fstat(fd).st_mode):
                raise HoldToRunError(
                    f"hold-to-run path is not an input character device: {self.device}"
                )
            supported = bytearray(KEY_STATE_BYTES)
            try:
                fcntl.ioctl(
                    fd,
                    _eviocgbit(EV_KEY, len(supported)),
                    supported,
                    True,
                )
            except OSError as exc:
                raise HoldToRunError(
                    f"cannot query hold-to-run key support on {self.device}: {exc}"
                ) from None
            if not _bit_is_set(supported, self.key_code):
                raise HoldToRunError(
                    f"evdev device {self.device} does not support key code "
                    f"{self.key_code}"
                )
            sample = self.sample()
            if require_released and sample.pressed:
                raise HoldToRunError(
                    "hold-to-run control must be released before live preflight"
                )
            return sample
        except Exception:
            self.close()
            raise

    def sample(self, monotonic_ns: int | None = None) -> HoldToRunSample:
        if self._fd is None or fcntl is None:
            raise HoldToRunError("hold-to-run device is not open")
        key_state = bytearray(KEY_STATE_BYTES)
        try:
            fcntl.ioctl(
                self._fd,
                _eviocgkey(len(key_state)),
                key_state,
                True,
            )
        except OSError as exc:
            self.close()
            raise HoldToRunError(
                f"lost hold-to-run device {self.device}: {exc}"
            ) from None
        pressed = _bit_is_set(key_state, self.key_code)
        if self._last_pressed is not None and pressed != self._last_pressed:
            if pressed:
                self._press_generation += 1
            else:
                self._release_generation += 1
        self._last_pressed = pressed
        return HoldToRunSample(
            pressed=pressed,
            monotonic_ns=(
                time.perf_counter_ns() if monotonic_ns is None else int(monotonic_ns)
            ),
            press_generation=self._press_generation,
            release_generation=self._release_generation,
        )

    def close(self) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> LinuxEvdevHoldToRun:
        self.open()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
