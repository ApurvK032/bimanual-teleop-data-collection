from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response


JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"
REALSENSE_VENDOR_ID = "8086"
ROLE_NAMES = ("scene", "left_wrist", "right_wrist")
ROLE_LABELS = {
    "scene": "Scene",
    "left_wrist": "Left wrist",
    "right_wrist": "Right wrist",
}


@dataclass(frozen=True)
class CameraDevice:
    role: str
    path: Path
    serial: str
    product: str

    def for_role(self, role: str) -> CameraDevice:
        return CameraDevice(role, self.path, self.serial, self.product)


@dataclass(frozen=True)
class JpegSnapshot:
    sequence: int
    ready_monotonic_ns: int
    jpeg: bytes


class LatestJpegFrame:
    """Thread-safe capacity-one camera mailbox with lightweight rate metrics."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._snapshot: JpegSnapshot | None = None
        self._timestamps_ns: deque[int] = deque(maxlen=90)

    def publish(self, jpeg: bytes, *, ready_monotonic_ns: int | None = None) -> JpegSnapshot:
        if not jpeg.startswith(JPEG_START) or not jpeg.endswith(JPEG_END):
            raise ValueError("camera frame is not a complete JPEG")
        timestamp_ns = ready_monotonic_ns or time.monotonic_ns()
        with self._condition:
            sequence = 1 if self._snapshot is None else self._snapshot.sequence + 1
            self._snapshot = JpegSnapshot(sequence, timestamp_ns, jpeg)
            self._timestamps_ns.append(timestamp_ns)
            self._condition.notify_all()
            return self._snapshot

    def snapshot(self) -> JpegSnapshot | None:
        with self._condition:
            return self._snapshot

    def wait_after(self, sequence: int, timeout_s: float) -> JpegSnapshot | None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._snapshot is None or self._snapshot.sequence <= sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            return self._snapshot

    def measured_fps(self) -> float:
        with self._condition:
            if len(self._timestamps_ns) < 2:
                return 0.0
            elapsed_s = (self._timestamps_ns[-1] - self._timestamps_ns[0]) / 1e9
            return 0.0 if elapsed_s <= 0.0 else (len(self._timestamps_ns) - 1) / elapsed_s


def split_jpeg_stream(buffer: bytearray) -> list[bytes]:
    """Remove and return every complete JPEG currently present in *buffer*."""

    frames: list[bytes] = []
    while True:
        start = buffer.find(JPEG_START)
        if start < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            return frames
        if start:
            del buffer[:start]
        end = buffer.find(JPEG_END, 2)
        if end < 0:
            return frames
        frame_end = end + len(JPEG_END)
        frames.append(bytes(buffer[:frame_end]))
        del buffer[:frame_end]


def _natural_video_key(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else 1_000_000


def _udev_properties(path: Path) -> dict[str, str]:
    completed = subprocess.run(
        ["udevadm", "info", "--query=property", f"--name={path}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=3.0,
    )
    properties: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    return properties


def _video_formats(path: Path) -> str:
    completed = subprocess.run(
        ["v4l2-ctl", "--device", str(path), "--list-formats-ext"],
        check=True,
        capture_output=True,
        text=True,
        timeout=3.0,
    )
    return completed.stdout


def discover_color_devices(video_paths: Iterable[Path] | None = None) -> list[CameraDevice]:
    """Find physical V4L2 cameras with a supported 640x480@30 YUYV stream.

    RealSense D405 and D455 color nodes both satisfy this contract. Generic
    UVC cameras are also shown when they expose the same low-latency format.
    Multiple V4L2 nodes belonging to one physical serial are collapsed to the
    first matching color node.
    """

    paths = sorted(video_paths or Path("/dev").glob("video*"), key=_natural_video_key)
    unique_devices: dict[str, CameraDevice] = {}
    for path in paths:
        try:
            properties = _udev_properties(path)
            formats = _video_formats(path)
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
        if "'YUYV'" not in formats or "640x480" not in formats or "30.000 fps" not in formats:
            continue
        serial = (
            properties.get("ID_SERIAL_SHORT")
            or properties.get("ID_PATH")
            or properties.get("ID_PATH_TAG")
            or str(path)
        )
        product = (
            properties.get("ID_V4L_PRODUCT")
            or properties.get("ID_MODEL_FROM_DATABASE")
            or properties.get("ID_MODEL")
            or "V4L2 camera"
        )
        unique_devices.setdefault(serial, CameraDevice("", path, serial, product))
    return sorted(
        unique_devices.values(),
        key=lambda device: (device.serial, _natural_video_key(device.path)),
    )


def discover_d405_color_devices(video_paths: Iterable[Path] | None = None) -> list[CameraDevice]:
    """Backward-compatible name for the now-general color-camera discovery."""

    return discover_color_devices(video_paths)


def assign_camera_roles(
    devices: Iterable[CameraDevice],
    *,
    scene_serial: str | None = None,
    wrist_serial: str | None = None,
) -> dict[str, CameraDevice]:
    """Return the legacy two-camera assignment used for first-run migration."""

    unique_by_serial: dict[str, CameraDevice] = {}
    for device in devices:
        unique_by_serial.setdefault(device.serial, device)
    ordered = sorted(unique_by_serial.values(), key=lambda item: (item.serial, _natural_video_key(item.path)))
    if len(ordered) < 2:
        raise RuntimeError(f"two D405 color streams are required; found {len(ordered)}")

    by_serial = {device.serial: device for device in ordered}
    if scene_serial is not None and scene_serial not in by_serial:
        raise RuntimeError(f"scene camera serial {scene_serial!r} was not found")
    if wrist_serial is not None and wrist_serial not in by_serial:
        raise RuntimeError(f"wrist camera serial {wrist_serial!r} was not found")

    scene = by_serial[scene_serial] if scene_serial else ordered[0]
    wrist_candidates = [device for device in ordered if device.serial != scene.serial]
    wrist = by_serial[wrist_serial] if wrist_serial else wrist_candidates[0]
    if scene.serial == wrist.serial:
        raise RuntimeError("scene and wrist roles must use different cameras")
    return {
        "scene": scene.for_role("scene"),
        "wrist": wrist.for_role("wrist"),
    }


class FfmpegCamera:
    def __init__(
        self,
        device: CameraDevice,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        jpeg_quality: int = 5,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.frames = LatestJpegFrame()
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._stderr_tail: deque[str] = deque(maxlen=12)

    def command(self) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-thread_queue_size",
            "2",
            "-f",
            "v4l2",
            "-input_format",
            "yuyv422",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
            "-i",
            str(self.device.path),
            "-an",
            "-c:v",
            "mjpeg",
            "-q:v",
            str(self.jpeg_quality),
            "-threads",
            "1",
            "-flush_packets",
            "1",
            "-f",
            "image2pipe",
            "pipe:1",
        ]

    def start(self, timeout_s: float = 6.0) -> None:
        if self._process is not None:
            return
        self._stop.clear()
        self._process = subprocess.Popen(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._reader = threading.Thread(target=self._read_frames, name=f"camera-{self.device.role}", daemon=True)
        self._stderr_reader = threading.Thread(
            target=self._read_stderr,
            name=f"camera-{self.device.role}-stderr",
            daemon=True,
        )
        self._reader.start()
        self._stderr_reader.start()
        if self.frames.wait_after(0, timeout_s) is None:
            detail = " | ".join(self._stderr_tail) or "no FFmpeg diagnostic"
            self.stop()
            raise RuntimeError(f"{self.device.role} camera produced no frame: {detail}")

    def _read_frames(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        buffer = bytearray()
        while not self._stop.is_set():
            chunk = process.stdout.read(64 * 1024)
            if not chunk:
                return
            buffer.extend(chunk)
            for jpeg in split_jpeg_stream(buffer):
                self.frames.publish(jpeg)
            if len(buffer) > 8 * 1024 * 1024:
                buffer.clear()

    def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for raw in iter(process.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                self._stderr_tail.append(line)

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        self._process = None

    def health(self) -> dict[str, object]:
        snapshot = self.frames.snapshot()
        age_ms = None if snapshot is None else (time.monotonic_ns() - snapshot.ready_monotonic_ns) / 1e6
        process_alive = self._process is not None and self._process.poll() is None
        return {
            "role": self.device.role,
            "device": str(self.device.path),
            "serial": self.device.serial,
            "product": self.device.product,
            "resolution": [self.width, self.height],
            "target_fps": self.fps,
            "measured_fps": round(self.frames.measured_fps(), 3),
            "sequence": 0 if snapshot is None else snapshot.sequence,
            "frame_age_ms": None if age_ms is None else round(age_ms, 3),
            "jpeg_bytes": 0 if snapshot is None else len(snapshot.jpeg),
            "process_alive": process_alive,
            "healthy": bool(process_alive and age_ms is not None and age_ms < 1_000.0),
            "stderr_tail": list(self._stderr_tail),
        }


def empty_configuration() -> dict[str, dict[str, object]]:
    return {role: {"serial": None, "enabled": False} for role in ROLE_NAMES}


def normalize_configuration(payload: object) -> dict[str, dict[str, object]]:
    if isinstance(payload, dict) and "roles" in payload:
        payload = payload["roles"]
    if not isinstance(payload, dict):
        raise ValueError("camera configuration must contain a roles object")

    normalized = empty_configuration()
    assigned: dict[str, str] = {}
    for role in ROLE_NAMES:
        raw = payload.get(role, {})
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"{role} configuration must be an object")
        raw_serial = raw.get("serial")
        serial = str(raw_serial).strip() if raw_serial not in (None, "") else None
        enabled = bool(raw.get("enabled", False)) and serial is not None
        if serial is not None:
            previous = assigned.get(serial)
            if previous is not None:
                raise ValueError(f"camera {serial!r} is assigned to both {previous} and {role}")
            assigned[serial] = role
        normalized[role] = {"serial": serial, "enabled": enabled}
    return normalized


def default_config_path() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "widowxai-quest-teleop" / "cameras.json"


class CameraRuntime:
    """Own dynamically selected camera processes without touching robot I/O."""

    def __init__(
        self,
        devices: Iterable[CameraDevice],
        *,
        discoverer: Callable[[], list[CameraDevice]] = discover_color_devices,
        camera_factory: Callable[..., FfmpegCamera] = FfmpegCamera,
        config_path: Path | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        jpeg_quality: int = 5,
    ) -> None:
        self._lock = threading.RLock()
        self._discoverer = discoverer
        self._camera_factory = camera_factory
        self._config_path = config_path
        self._capture_options = {
            "width": width,
            "height": height,
            "fps": fps,
            "jpeg_quality": jpeg_quality,
        }
        self.devices = {device.serial: device for device in devices}
        self.configuration = empty_configuration()
        self.cameras: dict[str, FfmpegCamera] = {}

    def _read_saved_configuration(self) -> dict[str, dict[str, object]] | None:
        if self._config_path is None or not self._config_path.exists():
            return None
        try:
            payload = json.loads(self._config_path.read_text(encoding="utf-8"))
            return normalize_configuration(payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"ignoring invalid camera configuration {self._config_path}: {exc}", flush=True)
            return None

    def _write_configuration(self) -> None:
        if self._config_path is None:
            return
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._config_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"schema_version": 1, "roles": self.configuration}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._config_path)

    def refresh_devices(self) -> list[CameraDevice]:
        discovered = self._discoverer()
        with self._lock:
            refreshed = {device.serial: device for device in discovered}
            # An active V4L2 node may not answer a format probe on every camera
            # driver. Preserve it in the list until its managed process stops.
            for camera in self.cameras.values():
                refreshed.setdefault(camera.device.serial, camera.device.for_role(""))
            self.devices = refreshed
            return list(self.devices.values())

    def first_run_configuration(
        self,
        *,
        scene_serial: str | None = None,
        left_wrist_serial: str | None = None,
        right_wrist_serial: str | None = None,
    ) -> dict[str, dict[str, object]]:
        ordered = sorted(self.devices.values(), key=lambda item: (item.serial, _natural_video_key(item.path)))
        requested = {
            "scene": scene_serial,
            "left_wrist": left_wrist_serial,
            "right_wrist": right_wrist_serial,
        }
        used: set[str] = set()
        result = empty_configuration()
        for role in ROLE_NAMES:
            serial = requested[role]
            if serial is not None:
                if serial not in self.devices:
                    raise RuntimeError(f"{ROLE_LABELS[role]} camera serial {serial!r} was not found")
                if serial in used:
                    raise RuntimeError(f"camera serial {serial!r} was requested for more than one role")
                result[role] = {"serial": serial, "enabled": True}
                used.add(serial)

        # Preserve the original two-camera first-run behavior: the stable first
        # device is scene and the stable second device is the existing wrist
        # feed, now named right_wrist. The UI can reassign either without code.
        for role in ("scene", "right_wrist"):
            if result[role]["serial"] is not None:
                continue
            candidate = next((item for item in ordered if item.serial not in used), None)
            if candidate is None:
                continue
            result[role] = {"serial": candidate.serial, "enabled": True}
            used.add(candidate.serial)
        return result

    def start(
        self,
        *,
        fallback_configuration: dict[str, dict[str, object]] | None = None,
    ) -> None:
        saved = self._read_saved_configuration()
        configuration = saved or fallback_configuration or empty_configuration()
        self.configure(configuration, persist=saved is None, allow_missing=True)

    def configure(
        self,
        payload: object,
        *,
        persist: bool = True,
        allow_missing: bool = False,
    ) -> dict[str, object]:
        desired = normalize_configuration(payload)
        with self._lock:
            missing = [
                str(item["serial"])
                for item in desired.values()
                if item["enabled"] and item["serial"] not in self.devices
            ]
            if missing and not allow_missing:
                raise ValueError(f"selected camera is not connected: {', '.join(missing)}")

            previous_configuration = self.configuration
            previous_cameras = dict(self.cameras)
            retained: dict[str, FfmpegCamera] = {}
            stopped: dict[str, FfmpegCamera] = {}
            for role, camera in previous_cameras.items():
                item = desired[role]
                if item["enabled"] and item["serial"] == camera.device.serial:
                    retained[role] = camera
                else:
                    camera.stop()
                    stopped[role] = camera

            new_cameras = dict(retained)
            started: list[FfmpegCamera] = []
            try:
                for role in ROLE_NAMES:
                    item = desired[role]
                    serial = item["serial"]
                    if not item["enabled"] or serial not in self.devices or role in retained:
                        continue
                    camera = self._camera_factory(
                        self.devices[str(serial)].for_role(role),
                        **self._capture_options,
                    )
                    camera.start()
                    started.append(camera)
                    new_cameras[role] = camera
                    print(
                        f"camera {role}: {camera.device.product} serial={camera.device.serial} "
                        f"device={camera.device.path} {camera.width}x{camera.height}@{camera.fps}",
                        flush=True,
                    )
            except Exception:
                for camera in reversed(started):
                    camera.stop()
                for camera in stopped.values():
                    camera.start()
                self.cameras = previous_cameras
                self.configuration = previous_configuration
                raise

            self.cameras = new_cameras
            self.configuration = desired
            if persist:
                self._write_configuration()
            return self.state()

    def stop(self) -> None:
        with self._lock:
            for camera in self.cameras.values():
                camera.stop()
            self.cameras = {}

    def state(self) -> dict[str, object]:
        with self._lock:
            assigned = {
                str(item["serial"]): role
                for role, item in self.configuration.items()
                if item["serial"] is not None
            }
            devices = [
                {
                    "serial": device.serial,
                    "product": device.product,
                    "device": str(device.path),
                    "assigned_role": assigned.get(device.serial),
                    "active": any(
                        camera.device.serial == device.serial for camera in self.cameras.values()
                    ),
                }
                for device in sorted(
                    self.devices.values(),
                    key=lambda item: (item.product, item.serial, _natural_video_key(item.path)),
                )
            ]
            return {
                "role_names": list(ROLE_NAMES),
                "role_labels": dict(ROLE_LABELS),
                "roles": json.loads(json.dumps(self.configuration)),
                "devices": devices,
            }

    def health(self) -> dict[str, object]:
        with self._lock:
            cameras: dict[str, object] = {}
            healthy = True
            for role in ROLE_NAMES:
                item = self.configuration[role]
                camera = self.cameras.get(role)
                if camera is not None:
                    cameras[role] = camera.health()
                else:
                    enabled = bool(item["enabled"])
                    cameras[role] = {
                        "role": role,
                        "serial": item["serial"],
                        "enabled": enabled,
                        "healthy": not enabled,
                        "process_alive": False,
                        "reason": "disabled" if not enabled else "camera unavailable",
                    }
                    healthy = healthy and not enabled
                if camera is not None:
                    healthy = healthy and bool(cameras[role]["healthy"])
            return {
                "status": "ok" if healthy else "degraded",
                "transport": "separate-latest-frame-websocket",
                "enabled_roles": [
                    role for role, item in self.configuration.items() if item["enabled"]
                ],
                "cameras": cameras,
            }


def create_app(runtime: CameraRuntime) -> FastAPI:
    app = FastAPI(title="WidowXAI Quest camera service")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    async def health() -> dict[str, object]:
        return runtime.health()

    @app.get("/configuration")
    async def configuration() -> dict[str, object]:
        await asyncio.to_thread(runtime.refresh_devices)
        return runtime.state()

    @app.post("/configuration")
    async def configure_cameras(payload: dict[str, object]) -> dict[str, object]:
        try:
            await asyncio.to_thread(runtime.refresh_devices)
            return await asyncio.to_thread(runtime.configure, payload)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/camera/{role}.jpg")
    async def latest_jpeg(role: str) -> Response:
        camera = runtime.cameras.get(role)
        if camera is None:
            raise HTTPException(status_code=404, detail="unknown camera role")
        snapshot = camera.frames.snapshot()
        if snapshot is None:
            raise HTTPException(status_code=503, detail="camera has no frame")
        return Response(
            snapshot.jpeg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Camera-Sequence": str(snapshot.sequence),
                "X-Camera-Frame-Ready-Monotonic-Ns": str(snapshot.ready_monotonic_ns),
            },
        )

    @app.websocket("/ws/{role}")
    async def camera_websocket(websocket: WebSocket, role: str) -> None:
        camera = runtime.cameras.get(role)
        if camera is None:
            await websocket.close(code=1008, reason="unknown camera role")
            return
        await websocket.accept()
        sequence = 0
        try:
            while True:
                snapshot = await asyncio.to_thread(camera.frames.wait_after, sequence, 2.0)
                if snapshot is None:
                    await websocket.close(code=1011, reason="camera frame timeout")
                    return
                await websocket.send_bytes(snapshot.jpeg)
                sequence = snapshot.sequence
        except (RuntimeError, WebSocketDisconnect):
            return

    return app


def _validate_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "udevadm", "v4l2-ctl") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"missing required camera tools: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure and stream operator cameras to the Quest")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8444)
    parser.add_argument("--scene-serial")
    parser.add_argument("--left-wrist-serial")
    parser.add_argument("--right-wrist-serial")
    # Compatibility with the original two-role prototype.
    parser.add_argument("--wrist-serial")
    parser.add_argument("--config-path", type=Path, default=default_config_path())
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=5)
    args = parser.parse_args()

    _validate_tools()
    if args.wrist_serial and args.right_wrist_serial:
        parser.error("--wrist-serial and --right-wrist-serial cannot both be set")
    devices = discover_color_devices()
    runtime = CameraRuntime(
        devices,
        config_path=args.config_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
    )
    fallback = runtime.first_run_configuration(
        scene_serial=args.scene_serial,
        left_wrist_serial=args.left_wrist_serial,
        right_wrist_serial=args.right_wrist_serial or args.wrist_serial,
    )
    runtime.start(fallback_configuration=fallback)
    try:
        import uvicorn

        uvicorn.run(create_app(runtime), host=args.host, port=args.port, log_level="info")
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
