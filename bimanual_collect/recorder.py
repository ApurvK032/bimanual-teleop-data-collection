"""Records every configured camera, robustly. One process owns the cameras; per camera:

  capture thread -> every complete frame, with the camera's sequence number and image timestamp
     |-> view thread: JPEG for the headset and console (stereo: middle 4:3 of the left eye, native pixels)
     '-> writer thread (only during an episode): bit-exact lossless video + frames.csv

Episode folder <data_raw>/<date>/session_HHMMSS/episode_NNNN/:
    episode.json                 written at start (status "recording"), rewritten at the end
    cameras/<name>/video.mkv     lossless FFV1
    cameras/<name>/video.svo2    lossless native stereo when the ZED SDK backend is enabled
    cameras/<name>/frames.csv    video_frame, camera_sequence, exposure_monotonic_ns, arrival_monotonic_ns,
                                 arrival_epoch_ns (flushed every 0.2 s)
    cameras/<name>/encoder.log

What can go wrong, and what happens:
  - a camera stops delivering: the episode is closed as "aborted" (reason kept, video finalised), and the camera
    is reopened every 2 s;
  - an encoder dies: the episode is closed as "aborted";
  - the disk fills: warning below recording.warn_free_gb, automatic stop below recording.stop_free_gb;
  - the process dies mid-episode: the next start repairs the episode (recover_episode) and marks it "recovered".
Timestamps are CLOCK_MONOTONIC, the same clock as time.perf_counter_ns() in the teleop telemetry.
"""
from __future__ import annotations

import csv
import json
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from widowxai_quest_teleop.camera_service import LatestJpegFrame

from . import rig
from .activity import arm_activity
from .recovery import recover_unfinished, write_json
from .zed_format import COLOUR_GAIN_BGR, COLOUR_OFFSET_BGR, EyeDecoder, encode_command
from .zed_sdk import SVO_COMPRESSION, SVO_NAME, ZedSdk, svo_frame_count
from .zed_v4l2 import Frame, V4L2Camera, ZedV4L2, find_uvc, find_zed

FORMAT = 2
RECORD_QUEUE_BYTES = 600e6  # encoder slack per camera
OUTCOMES = ("success", "failure", "unsure")
_END = object()  # closes the current episode's writer
_QUIT = object()  # ends the writer thread


class EncoderFailed(RuntimeError):
    pass


# ---------------------------------------------------------------- cameras

@dataclass
class CameraSpec:
    """How to open one camera and describe it in episode.json."""

    name: str
    open_source: Callable[[], object]  # returns an unopened source with open/read/close, width/height/fps
    stereo: bool
    describe: Callable[[object], dict]
    primary: bool = False
    colour_fix: bool = True
    svo2: bool = False


def zed_spec(mode: str, fps: int | None, colour_fix: bool = True, name: str = "zed", serial: str | None = None,
             primary: bool = True, backend: str = "v4l2") -> CameraSpec:
    """Any ZED model (ZED, ZED Mini, ZED 2, ZED 2i), found by serial (or the only one connected)."""
    if backend not in ("v4l2", "zed_sdk"):
        raise ValueError(f"camera {name}: ZED backend must be v4l2 or zed_sdk, got {backend!r}")
    found: dict = {}

    def open_source():
        device = find_zed(serial)
        if device.usb_mbps is not None and device.usb_mbps < 5000:
            raise RuntimeError(f"the {device.model} is on USB {device.usb_mbps} Mbps; it needs USB 3: "
                               "reseat or flip the USB-C plug at the camera")
        found["device"] = device
        if backend == "zed_sdk":
            return ZedSdk(device.node, device.serial, mode, fps, rig.load().calibration(device.serial or None))
        return ZedV4L2(device.node, mode, fps)

    def describe(source) -> dict:
        device = found["device"]
        calibration = source.calibration if backend == "zed_sdk" else rig.load().calibration(device.serial or None)
        try:
            calibration_name = str(calibration.relative_to(rig.ROOT))
        except ValueError:
            calibration_name = str(calibration)
        return {"model": device.model, "serial": device.serial, "mode": source.mode, "device": source.device,
                "usb_mbps": device.usb_mbps, "factory_calibration": calibration_name, "backend": backend}

    return CameraSpec(name, open_source, stereo=True, describe=describe, primary=primary, colour_fix=colour_fix,
                      svo2=backend == "zed_sdk")


def uvc_spec(name: str, entry: dict, primary: bool = False, colour_fix: bool = False) -> CameraSpec:
    """A plain UVC camera (RealSense colour, webcam...) in YUYV, found by USB serial or by part of its name."""
    found: dict = {}
    width, height, fps = int(entry.get("width", 640)), int(entry.get("height", 480)), int(entry.get("fps", 30))

    def open_source():
        device = find_uvc(entry.get("serial"), entry.get("match"))
        found["device"] = device
        return V4L2Camera(device.node, width, height, fps, controls=tuple(entry.get("controls") or ()))

    def describe(source) -> dict:
        device = found["device"]
        return {"model": device.name, "serial": device.serial, "mode": source.mode, "device": source.device,
                "usb_mbps": device.usb_mbps}

    return CameraSpec(name, open_source, stereo=False, describe=describe, primary=primary, colour_fix=colour_fix)


def camera_specs(mode: str | None = None, fps: int | None = None, colour_fix: bool = True) -> list[CameraSpec]:
    """Cameras from the rig config: the `cameras:` list, or one ZED from the `camera:` section.

    mode/fps (command line) override the primary ZED's. The natural-colour view fix was measured for the ZED 2i,
    so other cameras show their own colour unless `colour_fix: true` is set for them.
    """
    settings = rig.load()
    entries = list(settings.cameras) or [{"name": "zed", "type": "zed", "serial": settings.camera_serial,
                                          "mode": settings.camera_mode, "fps": settings.camera_fps}]
    if not any(entry.get("primary") for entry in entries):
        entries[0] = {**entries[0], "primary": True}
    names = [entry.get("name") for entry in entries]
    if len(set(names)) != len(names) or not all(names):
        raise ValueError(f"every camera in the rig config needs its own name (got {names})")
    specs = []
    for entry in entries:
        kind, primary = entry.get("type", "zed"), bool(entry.get("primary"))
        fix = colour_fix and bool(entry.get("colour_fix", kind == "zed"))
        if kind == "zed":
            cam_mode, cam_fps = str(entry.get("mode", settings.camera_mode)), entry.get("fps", settings.camera_fps)
            if primary and mode:
                cam_mode, cam_fps = mode, fps
            specs.append(zed_spec(cam_mode, cam_fps, fix, name=entry["name"], serial=entry.get("serial"),
                                  primary=primary, backend=entry.get("backend", settings.camera_backend)))
        elif kind == "uvc":
            specs.append(uvc_spec(entry["name"], entry, primary=primary, colour_fix=fix))
        else:
            raise ValueError(f"camera {entry['name']}: type must be zed or uvc, not {kind!r}")
    return specs


def fixed_source_spec(source, name: str = "zed", stereo: bool = True, colour_fix: bool = True,
                      primary: bool = True) -> CameraSpec:
    """A spec around an already-made source object (tests, fake cameras)."""
    return CameraSpec(name, lambda: source, stereo=stereo, primary=primary, colour_fix=colour_fix,
                      describe=lambda s: {"model": type(s).__name__, "serial": "TEST",
                                          "mode": getattr(s, "mode", "test"), "device": str(getattr(s, "device", "test"))})


class ViewEncoder:
    """Native-pixel JPEG for the headset/console: the middle 4:3 of the left eye (stereo) or the whole frame."""

    def __init__(self, width: int, height: int, stereo: bool, jpeg_quality: int, colour_fix: bool) -> None:
        eye_width = width // 2 if stereo else width
        self.crop_width = (min(eye_width, height * 4 // 3) if stereo else eye_width) // 2 * 2  # whole YUYV pairs
        self.height = height
        self.x0 = (eye_width - self.crop_width) // 2 // 2 * 2
        self.jpeg_quality = int(jpeg_quality)
        self._decoder = EyeDecoder(colour_fix)

    def bgr(self, frame: Frame) -> np.ndarray:
        raw = np.frombuffer(frame.data, np.uint8).reshape(frame.height, frame.width, 2)
        return self._decoder.bgr(raw, self.x0, self.crop_width)

    def jpeg(self, frame: Frame, *, max_edge: int | None = None, quality: int | None = None) -> bytes:
        image = self.bgr(frame)
        if max_edge is not None and max(image.shape[:2]) > max_edge:
            scale = max_edge / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        jpeg_quality = self.jpeg_quality if quality is None else quality
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return encoded.tobytes()


# Kept for older imports (tests, notebooks): the stereo view encoder.
def OperatorEncoder(width: int, height: int, jpeg_quality: int, colour_fix: bool) -> ViewEncoder:  # noqa: N802
    return ViewEncoder(width, height, True, jpeg_quality, colour_fix)


# ---------------------------------------------------------------- one camera's episode files

class EpisodeWriter:
    def __init__(self, directory: Path, source, ffv1_threads: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._log = open(directory / "encoder.log", "wb")
        self.process = subprocess.Popen(
            encode_command(directory / "video.mkv", source.width, source.height, source.fps, ffv1_threads),
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log,
        )
        self._csv_file = open(directory / "frames.csv", "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["video_frame", "camera_sequence", "exposure_monotonic_ns", "arrival_monotonic_ns",
                            "arrival_epoch_ns"])
        self._csv_file.flush()
        self._flushed = time.monotonic()
        self.frames = 0
        self.first_sequence: int | None = None
        self.last_sequence: int | None = None
        self.queue_drops = 0
        self.drops_at_start = source.dropped_error + source.dropped_short
        self.failed: str | None = None
        self.returncode: int | None = None
        self.closing = False
        self.svo_stopped = False
        self.svo_frames: int | None = None
        self.svo_error: str | None = None

    def write(self, frame: Frame) -> None:
        if self.failed:
            return
        try:
            self.process.stdin.write(frame.data)
        except (BrokenPipeError, OSError, ValueError) as exc:
            self.failed = f"encoder died: {exc}"
            raise EncoderFailed(self.failed) from exc
        if self.first_sequence is None:
            self.first_sequence = frame.sequence
        self.last_sequence = frame.sequence
        self._csv.writerow([self.frames, frame.sequence, frame.exposure_monotonic_ns, frame.dequeue_monotonic_ns,
                            frame.dequeue_epoch_ns])
        self.frames += 1
        now = time.monotonic()
        if now - self._flushed > 0.2:  # a crash then loses at most ~0.2 s of timestamps
            self._csv_file.flush()
            self._flushed = now
            if self.process.poll() is not None:
                self.failed = f"encoder exited with code {self.process.returncode}"
                raise EncoderFailed(self.failed)

    def close(self) -> int:
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            self.returncode = self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.returncode = self.process.wait()
        self._csv_file.close()
        self._log.close()
        return self.returncode

    def summary(self, source) -> dict:
        expected = 0 if self.first_sequence is None else self.last_sequence - self.first_sequence + 1
        return {
            "frames_recorded": self.frames,
            "camera_frames_missing": max(0, expected - self.frames),
            "camera_frames_error_dropped": source.dropped_error + source.dropped_short - self.drops_at_start,
            "recorder_queue_drops": self.queue_drops,
            "encoder_exit_code": self.returncode,
            "encoder_error": self.failed,
        }


# ---------------------------------------------------------------- one camera

class CameraWorker:
    def __init__(self, spec: CameraSpec, hub: Hub, jpeg_quality: int, ffv1_threads: int) -> None:
        self.spec = spec
        self.name = spec.name
        self.hub = hub
        self.jpeg_quality = jpeg_quality
        self.ffv1_threads = ffv1_threads
        self.source = None
        self.view: ViewEncoder | None = None
        self.state = "stopped"  # ok | lost | stopped
        self.error: str | None = None
        self.last_error: str | None = None
        self.reconnects = 0
        self.frames = LatestJpegFrame()  # what the headset / console views stream
        self.view_enabled = threading.Event()
        self._view_users = 0
        self._view_lock = threading.Lock()
        self.latest: Frame | None = None
        self._latest_cond = threading.Condition()
        self.captured = 0
        self.missing_sequences = 0
        self.last_sequence: int | None = None
        self.view_ms: list[float] = []
        self.writer: EpisodeWriter | None = None
        self.finished: EpisodeWriter | None = None
        self._writer_lock = threading.Lock()
        # The same grab belongs to both files. SDK recording boundaries must not split grab from enqueue.
        self._capture_lock = threading.Lock()
        self._capture_gate = threading.Lock()
        self._writer_closed = threading.Event()
        self._writer_closed.set()
        self._queue: queue.Queue | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.info: dict = {}

    # -- lifecycle
    def _open(self) -> None:
        source = self.spec.open_source()
        source.open()
        self.source = source
        self.view = ViewEncoder(source.width, source.height, self.spec.stereo, self.jpeg_quality, self.spec.colour_fix)
        if self._queue is None:
            self._queue = queue.Queue(maxsize=max(30, int(RECORD_QUEUE_BYTES // (source.width * source.height * 2))))
        try:
            self.info = self.spec.describe(source)
        except Exception as exc:  # noqa: BLE001 - metadata only
            self.info = {"describe_error": str(exc)}
        self.state, self.error = "ok", None

    def start(self) -> None:
        self._open()
        for target, label in ((self._capture_loop, "capture"), (self._view_loop, "view"), (self._writer_loop, "writer")):
            thread = threading.Thread(target=target, name=f"{self.name}-{label}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def wait_first_frame(self, timeout_s: float) -> bool:
        with self._latest_cond:
            return self._latest_cond.wait_for(lambda: self.latest is not None or self.state != "ok", timeout_s) \
                and self.latest is not None

    def stop(self) -> None:
        self._stop.set()
        if self._queue is not None:
            self._queue.put(_QUIT)
        with self._latest_cond:
            self._latest_cond.notify_all()
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads = []
        self._close_source()
        self.state = "stopped"

    def _close_source(self) -> None:
        if self.source is not None:
            try:
                self.source.close()
            except Exception:  # noqa: BLE001
                pass

    # -- threads
    @contextmanager
    def _capture_frame_lock(self):
        # A start/stop request holds the gate so it gets the lock before the next grab.
        with self._capture_gate:
            self._capture_lock.acquire()
        try:
            yield
        finally:
            self._capture_lock.release()

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            if self.state != "ok":
                if self._stop.wait(2.0):
                    return
                try:
                    self._open()
                    self.reconnects += 1
                    print(f"camera {self.name}: reconnected", flush=True)
                except Exception as exc:  # noqa: BLE001
                    self.error = f"reopen failed: {exc}"
                continue
            with self._capture_frame_lock() if self.spec.svo2 else nullcontext():
                try:
                    frame = self.source.read()
                except Exception as exc:  # noqa: BLE001 - a camera that stops delivering
                    if self._stop.is_set():
                        return
                    reason = f"{type(exc).__name__}: {exc}"
                    self.state, self.error, self.last_error = "lost", reason, reason
                    with self._writer_lock:
                        if self.writer is not None and not self.writer.closing:
                            self._finish_svo(self.writer)
                    self._close_source()
                else:
                    with self._writer_lock:
                        writer = self.writer
                        if writer is not None and not writer.closing:
                            try:
                                self._queue.put_nowait(frame)
                            except queue.Full:
                                writer.queue_drops += 1
            if self.state == "lost":
                print(f"camera {self.name}: lost ({reason}); reopening every 2 s", flush=True)
                self.hub.camera_lost(self, reason)
                with self._latest_cond:
                    self._latest_cond.notify_all()
                continue
            if self.last_sequence is not None and frame.sequence > self.last_sequence + 1:
                self.missing_sequences += frame.sequence - self.last_sequence - 1
            self.last_sequence = frame.sequence  # a reopened camera restarts its count: no false "missing"
            self.captured += 1
            with self._latest_cond:
                self.latest = frame
                self._latest_cond.notify_all()

    def _view_loop(self) -> None:
        seen = None
        while not self._stop.is_set():
            with self._latest_cond:
                self._latest_cond.wait_for(
                    lambda seen=seen: self._stop.is_set() or (self.latest is not None and self.latest is not seen),
                    timeout=1.0)
                frame = self.latest
            if self._stop.is_set() or frame is None or frame is seen:
                continue
            seen = frame
            if not self.view_enabled.is_set() or self.view is None:
                continue
            started = time.perf_counter()
            try:
                self.frames.publish(self.view.jpeg(frame))
            except Exception as exc:  # noqa: BLE001 - display only
                self.last_error = f"view encode failed: {exc}"
            self.view_ms = (self.view_ms + [(time.perf_counter() - started) * 1000])[-60:]

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get() if self._queue is not None else _QUIT
            if item is _QUIT:
                return
            if item is _END:
                with self._writer_lock:
                    writer, self.writer = self.writer, None
                if writer is not None:
                    writer.close()
                    self.finished = writer
                self._writer_closed.set()
                continue
            writer = self.writer
            if writer is None:
                continue
            try:
                writer.write(item)
            except EncoderFailed as exc:
                self.hub.encoder_failed(self, str(exc))

    # -- episodes
    def begin_episode(self, directory: Path) -> None:
        writer = EpisodeWriter(directory, self.source, self.ffv1_threads)
        with self._capture_gate, self._capture_lock, self._writer_lock:
            try:
                if self.state != "ok":
                    raise RuntimeError(f"camera {self.name} stopped delivering before recording started")
                if self.spec.svo2:
                    self.source.start_svo(directory)
            except Exception:
                writer.close()
                raise
            self._writer_closed.clear()
            self.writer = writer

    def _finish_svo(self, writer: EpisodeWriter) -> None:
        if not self.spec.svo2 or writer.svo_stopped:
            return
        writer.svo_stopped = True
        try:
            self.source.stop_svo()
        except (OSError, RuntimeError, ValueError) as exc:
            writer.svo_error = f"could not finalize SVO2: {exc}"

    def end_episode(self) -> None:
        with self._capture_gate, self._capture_lock, self._writer_lock:
            if self.writer is None:
                self._writer_closed.set()
                return
            self.writer.closing = True
            self._finish_svo(self.writer)
        self._queue.put(_END)  # after every frame already queued

    def wait_closed(self, timeout_s: float) -> bool:
        return self._writer_closed.wait(timeout_s)

    # -- views: JPEGs are made only while someone (headset panel, console) is watching
    def acquire_view(self) -> None:
        with self._view_lock:
            self._view_users += 1
            self.view_enabled.set()

    def release_view(self) -> None:
        with self._view_lock:
            self._view_users = max(0, self._view_users - 1)
            if self._view_users == 0:
                self.view_enabled.clear()

    def health(self) -> dict:
        snapshot = self.frames.snapshot()
        view_age = None if snapshot is None else (time.monotonic_ns() - snapshot.ready_monotonic_ns) / 1e6
        latest = self.latest
        capture_age = None if latest is None else (time.perf_counter_ns() - latest.dequeue_monotonic_ns) / 1e6
        source = self.source
        return {
            "name": self.name, "state": self.state, "error": self.error, "last_error": self.last_error,
            "reconnects": self.reconnects, "primary": self.spec.primary, "stereo": self.spec.stereo,
            "mode": getattr(source, "mode", None), "size": None if source is None else [source.width, source.height],
            "view_size": None if self.view is None else [self.view.crop_width, self.view.height],
            "fps_nominal": getattr(source, "fps", None), "view_fps": round(self.frames.measured_fps(), 2),
            "frames_captured": self.captured, "camera_frames_missing": self.missing_sequences,
            "error_flagged_dropped": getattr(source, "dropped_error", 0),
            "short_dropped": getattr(source, "dropped_short", 0),
            "capture_age_ms": None if capture_age is None else round(capture_age, 1),
            "view_age_ms": None if view_age is None else round(view_age, 1),
            "view_encode_ms_p50": round(float(np.median(self.view_ms)), 2) if self.view_ms else None,
            "queue_depth": 0 if self._queue is None else self._queue.qsize(),
            "queue_capacity": 0 if self._queue is None else self._queue.maxsize,
            "usb_mbps": self.info.get("usb_mbps"),
            "backend": self.info.get("backend"), "svo2_enabled": self.spec.svo2,
        }


# ---------------------------------------------------------------- the hub

@dataclass
class ActiveEpisode:
    index: int
    directory: Path
    labels: dict
    started_epoch_ns: int
    started_monotonic_ns: int
    stopping: bool = False
    status: str = "complete"
    stop_reason: str = "operator"
    done: threading.Event = field(default_factory=threading.Event)


class Hub:
    def __init__(
        self,
        cameras: list[CameraSpec] | None = None,
        *,
        mode: str | None = None,
        fps: int | None = None,
        camera=None,  # tests: one ready-made source object
        data_root: Path | None = None,
        jpeg_quality: int = 95,
        colour_fix: bool = True,
        ffv1_threads: int = 8,
    ) -> None:
        settings = rig.load()
        if cameras is None:
            if camera is not None:
                cameras = [fixed_source_spec(camera, colour_fix=colour_fix)]
            else:
                cameras = camera_specs(mode, fps, colour_fix)
        if not any(spec.primary for spec in cameras):
            cameras[0].primary = True
        self.workers = [CameraWorker(spec, self, jpeg_quality, ffv1_threads) for spec in cameras]
        self.primary = next(worker for worker in self.workers if worker.spec.primary)
        self.data_root = Path(data_root) if data_root is not None else settings.data_raw
        limits = settings.recording
        self.warn_free_bytes = float(limits.get("warn_free_gb", 50)) * 1e9
        self.stop_free_bytes = float(limits.get("stop_free_gb", 10)) * 1e9
        self.start_free_bytes = float(limits.get("min_free_to_start_gb", 20)) * 1e9
        self.task_presets = list(settings.tasks)
        self.defaults = {"task": "", "operator": settings.operator}
        self.session_dir: Path | None = None
        self._episode: ActiveEpisode | None = None
        self._lock = threading.RLock()
        self._last_summary: dict | None = None
        self._stop = threading.Event()
        self._monitor: threading.Thread | None = None
        self.arm: dict = {"active": False}
        self.disk_free: float | None = None
        self.warnings: list[str] = []
        self._arm_idle_since: float | None = None
        self._recording_without_arm_since: float | None = None
        self.recovered: list[dict] = []
        self.runs_roots = settings.teleop_runs

    # -- compatibility with code written for the one-camera hub
    @property
    def camera(self):
        return self.primary.source

    @property
    def mode(self) -> str:
        return getattr(self.primary.source, "mode", "test")

    @property
    def serial(self) -> str:
        return str(self.primary.info.get("serial", "TEST"))

    # -- lifecycle
    def start(self) -> None:
        self.recovered = recover_unfinished(self.data_root)
        for item in self.recovered:
            print(f"recovered unfinished episode {item['directory']}: {item['frames']} frames", flush=True)
        for worker in self.workers:
            worker.start()
        for worker in self.workers:
            if not worker.wait_first_frame(5.0):
                raise RuntimeError(f"camera {worker.name} produced no frame in 5 s ({worker.error or 'no error'})")
        self.session_dir = self.data_root / time.strftime("%Y-%m-%d") / time.strftime("session_%H%M%S")
        self._monitor = threading.Thread(target=self._monitor_loop, name="hub-monitor", daemon=True)
        self._monitor.start()

    def stop(self) -> None:
        if self.recording:
            self.stop_episode(reason="service stopped")
        self._stop.set()
        for worker in self.workers:
            worker.stop()

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._episode is not None

    # -- episodes
    def start_episode(self, task: str = "", operator: str = "") -> dict:
        with self._lock:
            if self._episode is not None:
                return self.record_status()
            down = [f"{w.name} ({w.error})" for w in self.workers if w.state != "ok"]
            if down:
                raise RuntimeError(f"not recording: camera {', '.join(down)} is not delivering")
            assert self.session_dir is not None, "hub not started"
            self.session_dir.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(self.session_dir).free
            if free < self.start_free_bytes:
                raise RuntimeError(f"only {free / 1e9:.0f} GB free; need {self.start_free_bytes / 1e9:.0f} GB to start")
            index = len(sorted(self.session_dir.glob("episode_*")))
            directory = self.session_dir / f"episode_{index:04d}"
            directory.mkdir()
            labels = {"task": str(task or self.defaults["task"] or ""),
                      "operator": str(operator or self.defaults["operator"] or ""), "outcome": None, "note": ""}
            episode = ActiveEpisode(index, directory, labels, time.time_ns(), time.perf_counter_ns())
            write_json(directory / "episode.json", self._episode_json(episode, "recording", None, final=False))
            self._episode = episode
            for worker in self.workers:
                worker.finished = None
            try:
                for worker in self.workers:
                    worker.begin_episode(directory / "cameras" / worker.name)
            except (OSError, RuntimeError, ValueError) as exc:
                self._request_close(episode, "aborted", f"camera {worker.name} could not start recording: {exc}")
                raise
            print(f"REC start episode {index} -> {directory}", flush=True)
            return self.record_status()

    def stop_episode(self, outcome: str | None = None, note: str | None = None, reason: str = "operator",
                     wait_s: float = 60.0) -> dict:
        with self._lock:
            episode = self._episode
            if episode is None:
                return self.record_status()
            if outcome is not None or note is not None:
                self._apply_labels(episode.labels, outcome=outcome, note=note)
        self._request_close(episode, "complete", reason)
        episode.done.wait(wait_s)
        return self.record_status()

    def toggle_episode(self, task: str = "") -> dict:
        return self.stop_episode() if self.recording else self.start_episode(task)

    def camera_lost(self, worker: CameraWorker, reason: str) -> None:
        with self._lock:
            episode = self._episode
        if episode is not None:
            self._request_close(episode, "aborted", f"camera {worker.name} lost: {reason}")

    def encoder_failed(self, worker: CameraWorker, reason: str) -> None:
        with self._lock:
            episode = self._episode
        if episode is not None:
            self._request_close(episode, "aborted", f"camera {worker.name}: {reason}")

    def _request_close(self, episode: ActiveEpisode, status: str, reason: str) -> None:
        with self._lock:
            if episode.stopping:
                return
            episode.stopping, episode.status, episode.stop_reason = True, status, reason
        # Closing waits on the writer threads, so it never runs on a capture or writer thread.
        threading.Thread(target=self._close_episode, args=(episode,), name="episode-close", daemon=True).start()

    def _close_episode(self, episode: ActiveEpisode) -> None:
        stopped_epoch_ns, stopped_monotonic_ns = time.time_ns(), time.perf_counter_ns()
        for worker in self.workers:
            worker.end_episode()
        for worker in self.workers:
            if not worker.wait_closed(90.0):
                episode.status, episode.stop_reason = "aborted", f"camera {worker.name}: writer did not close"
            writer = worker.finished
            if not worker.spec.svo2 or writer is None:
                continue
            if not writer.svo_error and writer.frames:
                try:
                    writer.svo_frames = svo_frame_count(writer.directory / SVO_NAME)
                except (OSError, RuntimeError, ValueError) as exc:
                    writer.svo_error = str(exc)
            if episode.status == "complete":
                if writer.svo_error or writer.svo_frames != writer.frames or not writer.frames:
                    reason = writer.svo_error or f"SVO2 has {writer.svo_frames} frames, MKV has {writer.frames}"
                    episode.status, episode.stop_reason = "aborted", f"camera {worker.name}: {reason}"
                elif writer.returncode != 0:
                    episode.status = "aborted"
                    episode.stop_reason = f"camera {worker.name}: FFV1 encoder exited with code {writer.returncode}"
        summary = self._episode_json(episode, episode.status, episode.stop_reason, final=True,
                                     stopped=(stopped_epoch_ns, stopped_monotonic_ns))
        write_json(episode.directory / "episode.json", summary)
        with self._lock:
            self._episode = None
            self._last_summary = summary
        episode.done.set()
        print(f"REC stop episode {episode.index} ({episode.status}, {episode.stop_reason}): "
              f"{summary['frames_recorded']} frames, {summary['duration_s']} s", flush=True)

    def _episode_json(self, episode: ActiveEpisode, status: str, reason: str | None, final: bool,
                      stopped: tuple[int, int] | None = None) -> dict:
        cameras = []
        for worker in self.workers:
            source = worker.source
            writer = worker.finished if final else worker.writer
            entry = {
                "name": worker.name, "primary": worker.spec.primary, "stereo": worker.spec.stereo,
                "video": f"cameras/{worker.name}/video.mkv", "frames": f"cameras/{worker.name}/frames.csv",
                "size": [source.width, source.height], "nominal_fps": source.fps, "pixel_format": "yuyv",
                "codec": "FFV1 level 3, lossless; decodes to the camera's exact YUYV bytes",
                "colour": "YUV BT.601 full range; camera colour (no colour fix applied)",
                **worker.info,
            }
            if worker.spec.svo2:
                entry["codec"] = "FFV1 level 3, lossless after conversion from SDK BGR to YUYV"
                entry["frame_timestamp"] = "ZED SDK IMAGE (host image timestamp), converted to CLOCK_MONOTONIC"
                entry["svo2"] = {
                    "file": f"cameras/{worker.name}/{SVO_NAME}", "compression": SVO_COMPRESSION,
                    "calibration": f"cameras/{worker.name}/{source.calibration.name}",
                    "status": "recording" if not final else "unverified",
                }
                if final and writer is not None:
                    matched = bool(writer.frames and writer.svo_frames == writer.frames and not writer.svo_error
                                   and writer.returncode == 0)
                    entry["svo2"].update(status="finalized" if matched else "unverified",
                                         frames_recorded=writer.svo_frames, error=writer.svo_error)
            if final and writer is not None:
                entry.update(writer.summary(source))
            cameras.append(entry)
        primary = next(entry for entry in cameras if entry["primary"])
        data = {
            "format": FORMAT,
            "status": status,
            "stop_reason": reason,
            "episode_index": episode.index,
            "task": episode.labels["task"],
            "labels": dict(episode.labels),
            "directory": str(episode.directory),
            "started_epoch_ns": episode.started_epoch_ns,
            "started_monotonic_ns": episode.started_monotonic_ns,
            "cameras": cameras,
            # The primary camera's fields at the top level too, as in format 1 (older tools read these).
            "video": {"file": primary["video"], "size": primary["size"], "nominal_fps": primary["nominal_fps"],
                      "per_eye": [primary["size"][0] // (2 if primary["stereo"] else 1), primary["size"][1]]},
            "camera": {key: primary.get(key) for key in ("model", "serial", "mode", "device", "usb_mbps",
                                                         "factory_calibration")},
            "operator_colour_fix_bgr": {"gain": COLOUR_GAIN_BGR.tolist(), "offset": COLOUR_OFFSET_BGR.tolist()},
            "clock": "monotonic_ns values are CLOCK_MONOTONIC (= time.perf_counter_ns() in the teleop telemetry)",
        }
        if stopped is not None:
            data["stopped_epoch_ns"], data["stopped_monotonic_ns"] = stopped
            data["duration_s"] = round((stopped[1] - episode.started_monotonic_ns) / 1e9, 3)
        if final:
            for key in ("frames_recorded", "camera_frames_missing", "camera_frames_error_dropped",
                        "recorder_queue_drops", "encoder_exit_code"):
                data[key] = primary.get(key)
        return data

    # -- labels
    @staticmethod
    def _apply_labels(labels: dict, **changes) -> None:
        for key, value in changes.items():
            if value is None:
                continue
            if key == "outcome":
                value = str(value).strip().lower() or None
                if value is not None and value not in OUTCOMES:
                    raise ValueError(f"outcome must be one of {OUTCOMES}")
            labels[key] = value if key == "outcome" else str(value)

    def label_episode(self, directory: str | None = None, **changes) -> dict:
        """Set task / operator / outcome / note on the running episode, or on a finished one (default: the last)."""
        with self._lock:
            episode = self._episode
            if episode is not None and (directory is None or Path(directory) == episode.directory):
                self._apply_labels(episode.labels, **changes)
                return dict(episode.labels)
            target = Path(directory) if directory else (
                Path(self._last_summary["directory"]) if self._last_summary else None)
        if target is None or not (target / "episode.json").exists():
            raise ValueError("no episode to label")
        meta = json.loads((target / "episode.json").read_text())
        labels = meta.setdefault("labels", {"task": meta.get("task", ""), "operator": "", "outcome": None, "note": ""})
        self._apply_labels(labels, **changes)
        meta["task"] = labels.get("task", meta.get("task", ""))
        write_json(target / "episode.json", meta)
        with self._lock:
            if self._last_summary and Path(self._last_summary["directory"]) == target:
                self._last_summary = meta
        return labels

    def set_defaults(self, task: str | None = None, operator: str | None = None) -> dict:
        if task is not None:
            self.defaults["task"] = str(task)
        if operator is not None:
            self.defaults["operator"] = str(operator)
        return dict(self.defaults)

    # -- monitoring
    def _monitor_loop(self) -> None:
        while not self._stop.wait(2.0):
            try:
                self._monitor_once()
            except Exception as exc:  # noqa: BLE001 - monitoring must not stop
                self.warnings = [f"monitor error: {exc}"]

    def _monitor_once(self) -> None:
        now = time.monotonic()
        target = self.session_dir if self.session_dir and self.session_dir.exists() else self.data_root
        target.mkdir(parents=True, exist_ok=True)
        self.disk_free = shutil.disk_usage(target).free
        self.arm = arm_activity(self.runs_roots)
        recording = self.recording
        if recording and self.disk_free < self.stop_free_bytes:
            with self._lock:
                episode = self._episode
            if episode is not None:
                self._request_close(episode, "complete", f"low disk space ({self.disk_free / 1e9:.1f} GB free)")
        if recording and not self.arm["active"]:
            self._recording_without_arm_since = self._recording_without_arm_since or now
        else:
            self._recording_without_arm_since = None
        if self.arm["active"] and self.arm.get("hardware") and not recording:
            self._arm_idle_since = self._arm_idle_since or now
        else:
            self._arm_idle_since = None
        warnings = []
        for worker in self.workers:
            if worker.state != "ok":
                warnings.append(f"camera {worker.name} is not delivering ({worker.error}); reopening every 2 s")
            elif worker._queue is not None and worker._queue.qsize() > worker._queue.maxsize // 2:
                warnings.append(f"camera {worker.name}: the recorder is falling behind")
        if self.disk_free < self.warn_free_bytes:
            warnings.append(f"disk: {self.disk_free / 1e9:.0f} GB free (~{self.minutes_left():.0f} min of recording)")
        if self._recording_without_arm_since and now - self._recording_without_arm_since > 5:
            warnings.append("recording, but no arm run is active")
        if self._arm_idle_since and now - self._arm_idle_since > 15:
            warnings.append("an arm run is active but nothing is being recorded (press A to record)")
        self.warnings = warnings

    def bytes_per_minute(self) -> float:
        """Estimate storage for FFV1 plus the native recording when enabled, based on bench captures."""
        return sum(w.source.width * w.source.height * 2 * w.source.fps * 60 * (1 / 6 + (0.15 if w.spec.svo2 else 0))
                   for w in self.workers if w.source)

    def minutes_left(self) -> float:
        if self.disk_free is None:
            return float("nan")
        return max(0.0, (self.disk_free - self.stop_free_bytes) / max(self.bytes_per_minute(), 1.0))

    # -- status
    def record_status(self) -> dict:
        with self._lock:
            episode = self._episode
            status = {
                "recording": episode is not None,
                "session_dir": None if self.session_dir is None else str(self.session_dir),
                "last_episode": self._last_summary,
                "defaults": dict(self.defaults),
                "task_presets": list(self.task_presets),
                "warnings": list(self.warnings),
                "arm": dict(self.arm),
                "disk": {"free_gb": None if self.disk_free is None else round(self.disk_free / 1e9, 1),
                         "minutes_left": None if self.disk_free is None else round(self.minutes_left(), 1)},
                "cameras": {w.name: {"state": w.state, "error": w.error} for w in self.workers},
                "recovered": self.recovered,
            }
            if episode is not None:
                writer = self.primary.writer
                status["episode"] = {
                    "index": episode.index, "directory": str(episode.directory), "task": episode.labels["task"],
                    "labels": dict(episode.labels),
                    "elapsed_s": round((time.perf_counter_ns() - episode.started_monotonic_ns) / 1e9, 1),
                    "frames": 0 if writer is None else writer.frames,
                    "queue_depth": max((w._queue.qsize() for w in self.workers if w._queue is not None), default=0),
                    "stopping": episode.stopping,
                }
            return status

    def health(self) -> dict:
        primary = self.primary.health()
        return {**primary, "zed_mode": primary["mode"], "capture_size": primary["size"],
                "operator_size": primary["view_size"], "camera_fps_nominal": primary["fps_nominal"],
                "recording": self.recording, "cameras": [w.health() for w in self.workers],
                "warnings": list(self.warnings)}
