"""ZED SDK stereo capture and lossless SVO2 recording, with live depth disabled."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from .zed_v4l2 import Frame, check_mode

SVO_NAME = "video.svo2"
SVO_COMPRESSION = "H265_LOSSLESS"
_RESOLUTIONS = {"2k": "HD2K", "1080": "HD1080", "720": "HD720", "vga": "VGA"}


def svo_frame_count(path: Path) -> int:
    """Read the finalized file's frame count, including frames flushed at stop."""
    import pyzed.sl as sl

    init = sl.InitParameters()
    init.set_from_svo_file(str(path))
    init.optional_settings_path = str(path.parent)
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.svo_real_time_mode = False
    camera = sl.Camera()
    result = camera.open(init)
    if result != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"open finalized SVO2: {result}")
    try:
        return camera.get_svo_number_of_frames()
    finally:
        camera.close()


class ZedSdk:
    def __init__(self, device: str, serial: str, mode: str, fps: int | None, calibration: Path) -> None:
        self.width, self.height, self.fps = check_mode(mode, fps)
        self.device, self.serial, self.mode = device, serial, mode
        self.calibration = calibration
        self.dropped_error = self.dropped_short = 0
        self._sequence = 0
        self._camera = None
        # OpenCV converts BGR to limited-range BT.601. Existing episode readers expect full range.
        values = np.arange(256, dtype=float)
        self._full_range = np.column_stack(((values - 16) * 255 / 219, 128 + (values - 128) * 255 / 224))
        self._full_range = np.clip(np.rint(self._full_range), 0, 255).astype(np.uint8).reshape(256, 1, 2)

    def open(self) -> None:
        if os.environ.get("ZED_SDK_SVO_VERSION") == "1":
            raise RuntimeError("unset ZED_SDK_SVO_VERSION to record SVO2")
        try:
            import pyzed.sl as sl
        except ImportError as exc:
            raise RuntimeError(f"install the ZED SDK 5.5+ Python bindings into the recorder environment: {exc}") from exc
        if not hasattr(sl.VIEW, "SIDE_BY_SIDE_UNRECTIFIED_BGR"):
            raise RuntimeError("the recorder needs ZED SDK 5.5+ for unrectified stereo capture")
        self._sl = sl
        self._image = sl.Mat()
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, _RESOLUTIONS[self.mode])
        init.camera_fps = self.fps
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.open_timeout_sec = 5.0
        if self.serial:
            init.set_from_serial_number(int(self.serial))
        if self.calibration.is_file() and self.calibration.name == f"SN{self.serial}.conf":
            init.optional_settings_path = str(self.calibration.parent)
        camera = sl.Camera()
        self._require_ok(camera.open(init), "open ZED")
        self._camera = camera
        try:
            info = camera.get_camera_information()
            config = info.camera_configuration
            actual = (2 * config.resolution.width, config.resolution.height, config.fps)
            if actual != (self.width, self.height, self.fps):
                raise RuntimeError(f"ZED opened at {actual}, expected {(self.width, self.height, self.fps)}")
            if not init.optional_settings_path:
                self.calibration = Path(f"/usr/local/zed/settings/SN{info.serial_number}.conf")
            if not self.calibration.is_file():
                raise RuntimeError(f"ZED SDK calibration file is missing: {self.calibration}")
        except Exception:
            self.close()
            raise

    def _require_ok(self, result, operation: str) -> None:
        if result != self._sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"{operation}: {result}")

    def read(self) -> Frame:
        sl, camera = self._sl, self._camera
        self._require_ok(camera.grab(), "grab ZED image")
        arrival = time.perf_counter_ns()
        epoch = time.time_ns()
        image_epoch = camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        if image_epoch <= 0:
            raise RuntimeError("ZED SDK returned an invalid image timestamp")
        sequence = self._sequence + camera.get_frame_dropped_count()
        self._sequence += 1
        self._require_ok(camera.retrieve_image(self._image, sl.VIEW.SIDE_BY_SIDE_UNRECTIFIED_BGR),
                         "retrieve unrectified stereo images")
        limited = cv2.cvtColor(self._image.get_data(deep_copy=False), cv2.COLOR_BGR2YUV_YUY2)
        raw = cv2.LUT(limited, self._full_range).tobytes()
        return Frame(raw, self.width, self.height, sequence, arrival + (image_epoch - epoch), arrival, epoch)

    def start_svo(self, directory: Path) -> None:
        sl = self._sl
        shutil.copyfile(self.calibration, directory / self.calibration.name)
        mode = getattr(sl.SVO_COMPRESSION_MODE, SVO_COMPRESSION)
        params = sl.RecordingParameters(str(directory / SVO_NAME), mode)
        self._require_ok(self._camera.enable_recording(params), f"start {SVO_COMPRESSION} SVO2 recording")
        try:
            actual = self._camera.get_recording_parameters().compression_mode
            if actual != mode:
                raise RuntimeError(f"requested SVO2 compression {mode}, but the SDK selected {actual}")
        except Exception:
            self._camera.disable_recording()
            raise

    def stop_svo(self) -> None:
        self._camera.disable_recording()

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None
