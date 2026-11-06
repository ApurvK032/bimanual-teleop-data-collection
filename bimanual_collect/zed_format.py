"""ZED recording format helpers shared by the recorder (zed_hub.py) and the exporters. numpy + OpenCV only.

A recorded episode holds zed_sbs.mkv: FFV1, the camera's exact side-by-side YUYV (left eye = columns
[0, W/2), right eye = [W/2, W)), YUV BT.601 FULL range, camera colour.
"""
from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

FULL_RANGE_FILTER = "scale=in_range=pc:out_range=pc"

# Natural-colour fix, BGR: out = raw * GAIN + OFFSET (measured under one room's ceiling
# lights). Applied for viewing and export; recordings keep the camera colour.
COLOUR_GAIN_BGR = np.array([1.072, 0.964, 1.050])
COLOUR_OFFSET_BGR = np.array([-17.7, 5.1, -4.3])


def encode_command(path: Path, width: int, height: int, fps: int, threads: int = 8) -> list[str]:
    """FFmpeg command that stores raw side-by-side YUYV from stdin as lossless FFV1, bit-exact."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuyv422", "-s", f"{width}x{height}", "-framerate", str(fps),
        "-color_range", "pc", "-i", "pipe:0",
        # yuyv422 -> yuv422p must be a pure re-pack so FFV1 stores the camera bytes exactly. FFmpeg 6.1
        # squeezes full range to 16-235 unless the scaler is told in and out are both full range.
        "-vf", FULL_RANGE_FILTER + ",format=yuv422p", "-c:v", "ffv1", "-level", "3", "-slices", "16",
        "-slicecrc", "1", "-threads", str(threads),
        "-color_range", "pc", "-colorspace", "smpte170m", "-color_primaries", "smpte170m", "-color_trc", "smpte170m",
        # Crash safety: a Matroska cluster every second and no output buffering, so a power cut loses at most
        # about a second (default buffering kept the first ~6+ s out of the file entirely).
        "-flush_packets", "1", "-cluster_time_limit", "1000",
        str(path),
    ]


def decode_command(video: Path) -> list[str]:
    """FFmpeg command that turns zed_sbs.mkv back into the camera's exact YUYV bytes on stdout."""
    return ["ffmpeg", "-v", "error", "-i", str(video), "-vf", FULL_RANGE_FILTER + ",format=yuyv422",
            "-color_range", "pc", "-f", "rawvideo", "pipe:1"]


def full_to_limited_lut() -> np.ndarray:
    """Map full-range YUYV into limited range so OpenCV's (limited-range) YUYV->BGR gives the full-range result."""
    v = np.arange(256, dtype=np.float64)
    lut = np.empty((256, 1, 2), np.uint8)
    lut[:, 0, 0] = np.clip(np.round(16 + v * 219 / 255), 0, 255)
    lut[:, 0, 1] = np.clip(np.round(128 + (v - 128) * 224 / 255), 0, 255)
    return lut


def colour_fix_lut() -> np.ndarray:
    v = np.arange(256, dtype=np.float64)
    lut = np.empty((256, 1, 3), np.uint8)
    for channel in range(3):
        lut[:, 0, channel] = np.clip(np.round(v * COLOUR_GAIN_BGR[channel] + COLOUR_OFFSET_BGR[channel]), 0, 255)
    return lut


class EyeDecoder:
    """YUYV side-by-side frame -> per-eye uint8 images (BGR for OpenCV, or RGB for LeRobot)."""

    def __init__(self, colour_fix: bool = True) -> None:
        self._limited = full_to_limited_lut()
        self._fix = colour_fix_lut() if colour_fix else None

    def bgr(self, yuyv: np.ndarray, x0: int = 0, width: int | None = None) -> np.ndarray:
        """yuyv: (H, W, 2) uint8. Columns [x0, x0+width) must start and end on whole YUYV pixel pairs."""
        width = yuyv.shape[1] - x0 if width is None else width
        crop = np.ascontiguousarray(yuyv[:, x0 : x0 + width])
        bgr = cv2.cvtColor(cv2.LUT(crop, self._limited), cv2.COLOR_YUV2BGR_YUYV)
        return cv2.LUT(bgr, self._fix) if self._fix is not None else bgr

    def eyes_rgb(self, yuyv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half = yuyv.shape[1] // 2
        left = cv2.cvtColor(self.bgr(yuyv, 0, half), cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(self.bgr(yuyv, half, half), cv2.COLOR_BGR2RGB)
        return left, right


def iter_frames(video: Path, width: int, height: int) -> Iterator[np.ndarray]:
    """Decode zed_sbs.mkv frame by frame as (H, W, 2) YUYV arrays (exact camera bytes)."""
    size = width * height * 2
    process = subprocess.Popen(decode_command(video), stdout=subprocess.PIPE)
    try:
        while True:
            data = process.stdout.read(size)
            if len(data) < size:
                return
            yield np.frombuffer(data, np.uint8).reshape(height, width, 2)
    finally:
        process.stdout.close()
        process.kill()
        process.wait()
