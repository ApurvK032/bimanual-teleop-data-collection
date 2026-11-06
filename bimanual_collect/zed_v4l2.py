"""Minimal V4L2 mmap capture for ZED cameras and other UVC cameras (no vendor SDK, no OpenCV).

Why not OpenCV or FFmpeg: in 2K mode about 1% of buffers come back short (~230 KB instead of 11 MB) with
V4L2_BUF_FLAG_ERROR set. OpenCV reshapes them to a full frame (stale data below the tear) and FFmpeg stops on
them. Here every buffer's flags and byte count are checked, and each kept frame carries the camera's sequence
number and the kernel's start-of-exposure timestamp (CLOCK_MONOTONIC, the same clock as time.perf_counter_ns()
on Linux, which the teleop telemetry uses).
"""
from __future__ import annotations

import ctypes
import errno
import fcntl
import mmap
import os
import select
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# Side-by-side YUYV (both eyes in one frame) and the frame rates the ZED 2i offers in each mode over USB 3.
# 2k and 1080 are native-scale crops of the sensor (1080 is tighter: same detail, smaller view); 720 and vga
# are binned, wider views. 2k is the maximum resolution but only runs at 15 fps.
SIZES = {"2k": (4416, 1242), "1080": (3840, 1080), "720": (2560, 720), "vga": (1344, 376)}
RATES = {"2k": (15,), "1080": (30, 15), "720": (60, 30, 15), "vga": (100, 60, 30, 15)}


def check_mode(mode: str, fps: int | None = None) -> tuple[int, int, int]:
    """(width, height, fps) for a mode; fps defaults to the mode's fastest rate up to 30."""
    if mode not in SIZES:
        raise ValueError(f"ZED mode must be one of {list(SIZES)}, not {mode!r}")
    rates = RATES[mode]
    fps = int(fps) if fps else max(r for r in rates if r <= 30)
    if fps not in rates:
        raise ValueError(f"ZED mode {mode} runs at {list(rates)} fps, not {fps}")
    return (*SIZES[mode], fps)

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_BUF_FLAG_ERROR = 0x40


class Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class Timecode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_uint8),
        ("seconds", ctypes.c_uint8),
        ("minutes", ctypes.c_uint8),
        ("hours", ctypes.c_uint8),
        ("userbits", ctypes.c_uint8 * 4),
    ]


class BufferM(ctypes.Union):
    _fields_ = [("offset", ctypes.c_uint32), ("userptr", ctypes.c_ulong), ("planes", ctypes.c_void_p), ("fd", ctypes.c_int32)]


class Buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", Timeval),
        ("timecode", Timecode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", BufferM),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("request_fd", ctypes.c_int32),
    ]


class RequestBuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("flags", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


def _iowr(nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord("V") << 8) | nr


def _iow(nr: int, size: int) -> int:
    return (1 << 30) | (size << 16) | (ord("V") << 8) | nr


VIDIOC_REQBUFS = _iowr(8, ctypes.sizeof(RequestBuffers))
VIDIOC_QUERYBUF = _iowr(9, ctypes.sizeof(Buffer))
VIDIOC_QBUF = _iowr(15, ctypes.sizeof(Buffer))
VIDIOC_DQBUF = _iowr(17, ctypes.sizeof(Buffer))
VIDIOC_STREAMON = _iow(18, ctypes.sizeof(ctypes.c_int))
VIDIOC_STREAMOFF = _iow(19, ctypes.sizeof(ctypes.c_int))


@dataclass(frozen=True)
class Frame:
    data: bytes  # YUYV, height x width x 2, both eyes side by side
    width: int
    height: int
    sequence: int  # camera frame counter; a gap means frames were lost
    exposure_monotonic_ns: int  # camera timestamp in CLOCK_MONOTONIC (start of exposure for V4L2)
    dequeue_monotonic_ns: int  # time.perf_counter_ns() when the frame reached us
    dequeue_epoch_ns: int  # time.time_ns() at the same moment


# ---------------------------------------------------------------- finding cameras

# Stereolabs USB product IDs: video interface -> (model, IMU/HID interface that carries the real serial).
ZED_MODELS = {
    "f580": ("ZED", None),
    "f680": ("ZED Mini", "f681"),
    "f780": ("ZED 2", "f781"),
    "f880": ("ZED 2i", "f881"),
}
STEREOLABS = "2b03"


@dataclass(frozen=True)
class VideoDevice:
    node: str  # /dev/videoN (capture node: index 0 of its USB interface)
    name: str
    vendor: str
    product: str
    serial: str  # USB serial (for a ZED: the real camera serial from its HID interface)
    usb_mbps: int | None
    usb_path: str  # e.g. 6-1.1

    @property
    def is_zed(self) -> bool:
        return self.vendor == STEREOLABS and self.product in ZED_MODELS

    @property
    def model(self) -> str:
        return ZED_MODELS[self.product][0] if self.is_zed else self.name


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _usb_device(interface: Path) -> Path | None:
    """The USB device directory (with idVendor) above a video interface."""
    for parent in [interface, *interface.parents]:
        if (parent / "idVendor").exists():
            return parent
    return None


def _zed_serial(usb: Path, hid_product: str | None) -> str:
    """A ZED's video interface reports a placeholder serial (OV0001); the real one is on its HID interface, which
    sits under the same port on the companion USB 2 bus (e.g. video 6-1.1, HID 5-1.1.2)."""
    devpath = _read(usb / "devpath")
    candidates = []
    for device in Path("/sys/bus/usb/devices").iterdir():
        if _read(device / "idVendor") == STEREOLABS and _read(device / "idProduct") == hid_product:
            hid_path = _read(device / "devpath")
            if hid_path == devpath or hid_path.startswith(devpath + "."):
                return _read(device / "serial")
            candidates.append(_read(device / "serial"))
    if len(candidates) == 1:  # one ZED with an IMU on the system
        return candidates[0]
    own = _read(usb / "serial")
    return "" if own.upper().startswith("OV") else own


def video_devices() -> list[VideoDevice]:
    """Capture nodes of every USB camera (one per camera, not the metadata nodes)."""
    devices = []
    for entry in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:] or 0)):
        if _read(entry / "index") not in ("", "0"):
            continue
        usb = _usb_device((entry / "device").resolve())
        if usb is None:
            continue
        vendor, product = _read(usb / "idVendor"), _read(usb / "idProduct")
        speed = _read(usb / "speed")
        serial = _zed_serial(usb, ZED_MODELS[product][1]) if vendor == STEREOLABS and product in ZED_MODELS \
            else _read(usb / "serial")
        devices.append(VideoDevice(f"/dev/{entry.name}", _read(entry / "name"), vendor, product, serial,
                                   int(speed) if speed.isdigit() else None, usb.name))
    return devices


def find_zed(serial: str | None = None) -> VideoDevice:
    """A ZED (any model) by serial, or the only one connected."""
    zeds = [d for d in video_devices() if d.is_zed]
    if serial:
        zeds = [d for d in zeds if d.serial == str(serial)]
    if not zeds:
        wanted = f" with serial {serial}" if serial else ""
        raise RuntimeError(f"no ZED{wanted} found (plugged in? /etc/udev/rules.d/99-slabs.rules installed?)")
    if len(zeds) > 1 and not serial:
        raise RuntimeError(f"{len(zeds)} ZEDs connected ({', '.join(d.serial for d in zeds)}): set camera serials")
    return zeds[0]


def find_uvc(serial: str | None = None, name: str | None = None) -> VideoDevice:
    """A non-ZED UVC camera by USB serial or by a piece of its name (e.g. "RealSense", "C920")."""
    cameras = [d for d in video_devices() if not d.is_zed]
    if serial:
        cameras = [d for d in cameras if d.serial == str(serial)]
    if name:
        cameras = [d for d in cameras if name.lower() in d.name.lower()]
    if not cameras:
        raise RuntimeError(f"no UVC camera matching serial={serial!r} name={name!r}")
    if len(cameras) > 1:
        raise RuntimeError(f"{len(cameras)} cameras match serial={serial!r} name={name!r}: be more specific")
    return cameras[0]


def zed_usb_mbps(serial: str | None = None) -> int | None:
    try:
        return find_zed(serial).usb_mbps
    except RuntimeError:
        return None


def zed_serial(serial: str | None = None) -> str:
    try:
        return find_zed(serial).serial or "unknown"
    except RuntimeError:
        return "unknown"


# ---------------------------------------------------------------- capture

class V4L2Camera:
    """YUYV capture from any UVC camera: every buffer's error flag and size checked, sequence + exposure kept."""

    label = "camera"

    def __init__(self, device: str, width: int, height: int, fps: int, buffers: int = 4,
                 controls: tuple[str, ...] = ()) -> None:
        self.device = device
        self.mode = f"{width}x{height}"
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.controls = controls
        self.frame_bytes = self.width * self.height * 2
        self.buffer_count = buffers
        self.fd: int | None = None
        self.maps: list[mmap.mmap] = []
        self.dropped_error = 0
        self.dropped_short = 0

    def open(self) -> None:
        subprocess.run(
            ["v4l2-ctl", "-d", self.device, f"--set-fmt-video=width={self.width},height={self.height},pixelformat=YUYV",
             f"--set-parm={self.fps}"],
            check=True, capture_output=True, text=True,
        )
        for control in self.controls:  # optional camera controls; a camera without one just skips it
            subprocess.run(["v4l2-ctl", "-d", self.device, "-c", control], capture_output=True, text=True)
        applied = subprocess.run(["v4l2-ctl", "-d", self.device, "--get-parm"], capture_output=True, text=True).stdout
        for line in applied.splitlines():
            if "Frames per second" in line and abs(float(line.split(":")[1].split()[0]) - self.fps) > 0.5:
                raise RuntimeError(f"the {self.label} did not accept {self.fps} fps at {self.mode}: {line.strip()}")
        self.fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK)
        request = RequestBuffers(count=self.buffer_count, type=V4L2_BUF_TYPE_VIDEO_CAPTURE, memory=V4L2_MEMORY_MMAP)
        fcntl.ioctl(self.fd, VIDIOC_REQBUFS, request)
        for index in range(request.count):
            buf = Buffer(index=index, type=V4L2_BUF_TYPE_VIDEO_CAPTURE, memory=V4L2_MEMORY_MMAP)
            fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, buf)
            if buf.length < self.frame_bytes:
                raise RuntimeError(
                    f"V4L2 buffer is {buf.length} bytes, expected {self.frame_bytes} for {self.mode}; "
                    f"the format did not apply (is the {self.label} on USB 3?)"
                )
            self.maps.append(mmap.mmap(self.fd, buf.length, mmap.MAP_SHARED, mmap.PROT_READ, offset=buf.m.offset))
            fcntl.ioctl(self.fd, VIDIOC_QBUF, buf)
        fcntl.ioctl(self.fd, VIDIOC_STREAMON, ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE))

    def read(self, timeout_s: float = 2.0) -> Frame:
        """Return the next complete frame; error-flagged and short buffers are re-queued and counted."""
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no frame from the {self.label}")
            ready, _, _ = select.select([self.fd], [], [], remaining)
            if not ready:
                continue
            buf = Buffer(type=V4L2_BUF_TYPE_VIDEO_CAPTURE, memory=V4L2_MEMORY_MMAP)
            try:
                fcntl.ioctl(self.fd, VIDIOC_DQBUF, buf)
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    continue
                raise
            now_ns = time.perf_counter_ns()
            epoch_ns = time.time_ns()
            # Read everything before QBUF: re-queueing hands the struct back to the kernel, which clears it.
            sequence = int(buf.sequence)
            exposure_ns = int(buf.timestamp.tv_sec) * 1_000_000_000 + int(buf.timestamp.tv_usec) * 1_000
            good = not (buf.flags & V4L2_BUF_FLAG_ERROR) and buf.bytesused == self.frame_bytes
            data = self.maps[buf.index][: self.frame_bytes] if good else b""
            if not good:
                if buf.flags & V4L2_BUF_FLAG_ERROR:
                    self.dropped_error += 1
                else:
                    self.dropped_short += 1
            fcntl.ioctl(self.fd, VIDIOC_QBUF, buf)
            if good:
                return Frame(data, self.width, self.height, sequence, exposure_ns, now_ns, epoch_ns)

    def close(self) -> None:
        if self.fd is None:
            return
        try:
            fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE))
        except OSError:
            pass
        for mapping in self.maps:
            mapping.close()
        self.maps = []
        os.close(self.fd)
        self.fd = None


class ZedV4L2(V4L2Camera):
    """A ZED (any model) in one of its side-by-side modes."""

    label = "ZED"

    def __init__(self, device: str, mode: str = "720", fps: int | None = None, buffers: int = 4) -> None:
        width, height, fps = check_mode(mode, fps)
        # US mains anti-flicker (60 Hz; the camera defaults to 50 Hz and resets it on every replug).
        super().__init__(device, width, height, fps, buffers, controls=("power_line_frequency=2",))
        self.mode = mode
