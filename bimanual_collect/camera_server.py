"""Camera + recording service for data collection (port 8444), built on the teleop repo's camera service.

The teleop repo is not modified. Its CameraRuntime takes a device discoverer and a camera factory, so here:
  - every camera the recorder owns (recorder.Hub) is added to the device list and its view is served from the hub;
  - other cameras (e.g. RealSense wrist cameras) keep the repo's own FFmpeg view path.
Camera assignments are saved in <state>/cameras.json (rig config), apart from the teleop repo's own file.

Endpoints added for episodes:
  GET  /record/status                      recording state, warnings, disk, arm activity, last episode
  POST /record/start   {"task", "operator"}
  POST /record/stop    {"outcome": "success|failure|unsure", "note"}
  POST /record/toggle  {"task"}            (the headset's A button)
  POST /record/label   {"directory"?, "task", "operator", "outcome", "note"}   (default: running or last episode)
  POST /record/defaults {"task", "operator"} (used when an episode starts without them)
  GET  /hub/health                         every camera;  GET /zed/health = the primary camera (older pages)
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from fastapi import HTTPException
from widowxai_quest_teleop.camera_service import (
    CameraDevice,
    CameraRuntime,
    FfmpegCamera,
    create_app,
    discover_color_devices,
)

from . import console, rig
from .recorder import CameraWorker, Hub
from .zed_v4l2 import SIZES


class HubCamera:
    """Looks like the repo's FfmpegCamera to its CameraRuntime, but streams one hub camera's view.

    Stopping it (e.g. un-ticking the view in the Quest page) only stops the JPEGs; capture and recording go on.
    """

    def __init__(self, device: CameraDevice, worker: CameraWorker) -> None:
        self.device = device
        self.worker = worker
        self.width = worker.view.crop_width
        self.height = worker.view.height
        self.fps = worker.source.fps
        self.frames = worker.frames
        self._started = False

    def start(self, timeout_s: float = 6.0) -> None:
        if not self._started:
            self.worker.acquire_view()
            self._started = True
        if self.frames.wait_after(0, timeout_s) is None and self.frames.snapshot() is None:
            raise RuntimeError(f"camera {self.worker.name}: no view frame")

    def stop(self) -> None:
        if self._started:
            self.worker.release_view()
            self._started = False

    def health(self) -> dict[str, object]:
        health = self.worker.health()
        age = health["view_age_ms"]
        snapshot = self.frames.snapshot()
        return {
            "role": self.device.role, "device": str(self.device.path), "serial": self.device.serial,
            "product": self.device.product, "resolution": [self.width, self.height], "target_fps": self.fps,
            "measured_fps": round(self.frames.measured_fps(), 3),
            "sequence": 0 if snapshot is None else snapshot.sequence, "frame_age_ms": age,
            "jpeg_bytes": 0 if snapshot is None else len(snapshot.jpeg),
            "process_alive": health["state"] == "ok",
            "healthy": bool(health["state"] == "ok" and age is not None and age < 1000.0),
            "stderr_tail": [] if health["error"] is None else [health["error"]], "hub": health,
        }


def device_serial(worker: CameraWorker) -> str:
    """The id the teleop camera service saves role assignments under. ZEDs keep "ZED-<serial>" (what earlier
    sessions saved, so the Scene panel stays assigned); other cameras get "CAM-<name>-<serial>"."""
    serial = str(worker.info.get("serial", "") or worker.name)
    return f"ZED-{serial}" if str(worker.info.get("model", "")).startswith("ZED") else f"CAM-{worker.name}-{serial}"


def hub_devices(hub: Hub) -> list[CameraDevice]:
    devices = []
    for worker in hub.workers:
        info = worker.info
        product = f"{info.get('model', worker.name)} {info.get('serial', '')} ({info.get('mode', '')} {worker.name})"
        devices.append(CameraDevice("", Path(str(getattr(worker.source, "device", worker.name))),
                                    device_serial(worker), " ".join(product.split())))
    return devices


def build_app(hub: Hub, *, config_path: Path | None = None):
    config_path = config_path or rig.load().state / "cameras.json"
    owned = {device.serial: worker for device, worker in zip(hub_devices(hub), hub.workers)}
    owned_devices = hub_devices(hub)

    def discover() -> list[CameraDevice]:
        paths = {device.path for device in owned_devices}
        others = [device for device in discover_color_devices() if device.path not in paths]
        return [*owned_devices, *others]

    def camera_factory(device: CameraDevice, **capture_options):
        if device.serial in owned:
            return HubCamera(device, owned[device.serial])
        return FfmpegCamera(device, **capture_options)

    runtime = CameraRuntime(discover(), discoverer=discover, camera_factory=camera_factory, config_path=config_path)
    primary_serial = next(serial for serial, worker in owned.items() if worker is hub.primary)
    fallback = runtime.first_run_configuration(scene_serial=primary_serial)
    runtime.start(fallback_configuration=fallback)
    app = create_app(runtime)

    async def _run(action, *args, **kwargs):
        try:
            return await asyncio.to_thread(action, *args, **kwargs)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _text(payload: dict | None, key: str) -> str | None:
        value = (payload or {}).get(key)
        return None if value is None else str(value)

    @app.get("/record/status")
    async def record_status() -> dict:
        return hub.record_status()

    @app.post("/record/start")
    async def record_start(payload: dict | None = None) -> dict:
        return await _run(hub.start_episode, _text(payload, "task") or "", _text(payload, "operator") or "")

    @app.post("/record/stop")
    async def record_stop(payload: dict | None = None) -> dict:
        return await _run(hub.stop_episode, outcome=_text(payload, "outcome"), note=_text(payload, "note"))

    @app.post("/record/toggle")
    async def record_toggle(payload: dict | None = None) -> dict:
        return await _run(hub.toggle_episode, _text(payload, "task") or "")

    @app.post("/record/label")
    async def record_label(payload: dict | None = None) -> dict:
        changes = {key: _text(payload, key) for key in ("task", "operator", "outcome", "note")}
        return await _run(hub.label_episode, _text(payload, "directory"), **changes)

    @app.post("/record/defaults")
    async def record_defaults(payload: dict | None = None) -> dict:
        return hub.set_defaults(_text(payload, "task"), _text(payload, "operator"))

    @app.get("/hub/health")
    async def hub_health() -> dict:
        return hub.health()

    @app.get("/zed/health")
    async def zed_health() -> dict:
        return hub.health()

    console.register(app, hub)
    return app, runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8444)
    parser.add_argument("--mode", choices=list(SIZES), help="ZED mode (default: rig config)")
    parser.add_argument("--fps", type=int, help="frame rate the mode offers (default: rig config, else up to 30)")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="view JPEG quality")
    parser.add_argument("--no-colour-fix", action="store_true", help="show the camera's own colour in the views")
    parser.add_argument("--data-root", type=Path, default=rig.load().data_raw)
    args = parser.parse_args()

    hub = Hub(mode=args.mode, fps=args.fps, jpeg_quality=args.jpeg_quality, colour_fix=not args.no_colour_fix,
              data_root=args.data_root)
    hub.start()
    for worker in hub.workers:
        source = worker.source
        print(f"camera {worker.name}: {worker.info.get('model', '')} {worker.info.get('serial', '')} "
              f"{source.width}x{source.height}@{source.fps}; view {worker.view.crop_width}x{worker.view.height}",
              flush=True)
    print(f"episodes -> {hub.session_dir}", flush=True)
    app, runtime = build_app(hub)
    try:
        import uvicorn

        # Localhost only: the Quest reaches it through `adb reverse tcp:8444 tcp:8444`.
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        runtime.stop()
        hub.stop()


if __name__ == "__main__":
    main()
