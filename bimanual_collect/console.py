"""The camera service's part of the collection app's API (episodes, jobs, views, setup), on port 8444.

The page itself is served by the collection app (app.py, http://localhost:8440/), which forwards these endpoints;
/console here redirects there.

Status, camera previews, sessions and episodes (QC, thumbnails, sync plots, previews), review, background jobs
(process / export / depth, one at a time, paused while recording), rig settings and the pixel-box editor.

Everything binds to localhost; from another machine forward the app's port:
    ssh -N -L 8440:localhost:8440 <this pc>      then open http://localhost:8440/
"""
from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse

from . import catalog, rig
from .calibration import intrinsics, map_box_from_2k, map_box_to_2k
from .jobs import JobRunner
from .review import write_review
from .zed_format import EyeDecoder

SERVED_SUFFIXES = {".jpg": "image/jpeg", ".png": "image/png", ".mp4": "video/mp4", ".json": "application/json",
                   ".md": "text/markdown"}
BOX_NAMES = ("sync_roi", "video_crop", "table")


class QuestState:
    """Is a Quest connected over adb? Checked at most every 5 s (adb is slow)."""

    def __init__(self) -> None:
        self._value = {"adb": shutil.which("adb") is not None, "connected": False, "checked": 0.0}
        self._lock = threading.Lock()

    def get(self) -> dict:
        with self._lock:
            if time.time() - self._value["checked"] < 5:
                return dict(self._value)
        connected = False
        if self._value["adb"]:
            try:
                result = subprocess.run(["adb", "get-state"], capture_output=True, text=True, timeout=3)
                connected = result.stdout.strip() == "device"
            except (OSError, subprocess.TimeoutExpired):
                connected = False
        with self._lock:
            self._value.update(connected=connected, checked=time.time())
            return dict(self._value)


def _data_path(path: str) -> Path:
    """A path inside the raw data folder (rejects anything outside it)."""
    root = rig.load().data_raw.resolve()
    resolved = (root / path).resolve()
    if root not in (resolved, *resolved.parents):
        raise HTTPException(status_code=400, detail="path outside the data folder")
    return resolved


def _episode_files(episode: Path) -> dict:
    names = {"thumbs": "thumbs.jpg", "preview": "preview.mp4", "sync_plot": "sync/sync_plot.png",
             "report": "../report.md"}
    return {key: catalog.relative(episode / name) for key, name in names.items() if (episode / name).exists()}


def sessions() -> list[dict]:
    root = rig.load().data_raw
    found = []
    for session in sorted(root.glob("*/session_*"), reverse=True):
        episodes = sorted(p for p in session.glob("episode_*") if (p / "episode.json").exists())
        if not episodes:
            continue
        rows = [catalog.index_row(e) for e in episodes]
        found.append({
            "path": catalog.relative(session), "date": session.parent.name, "name": session.name,
            "episodes": len(rows), "processed": sum(r["qc"] != "not processed" for r in rows),
            "passed": sum(r["qc"] == "pass" for r in rows), "accepted": sum(r["review"] == "accepted" for r in rows),
            "minutes": round(sum(float(r["duration_s"] or 0) for r in rows) / 60, 1),
        })
    return found


def setup_info(hub) -> dict:
    settings = rig.load()
    return {
        "rig_file": str(settings.source.relative_to(rig.ROOT)),
        "data_raw": str(settings.data_raw), "data_lerobot": str(settings.data_lerobot),
        "arms": settings.arms, "tasks": list(settings.tasks), "operator": settings.operator,
        "boxes_2k": {k: list(v) for k, v in settings.boxes_2k.items()},
        "qc": catalog.limits(), "export": settings.export, "recording": settings.recording,
        "cameras": [w.health() | {"info": w.info} for w in hub.workers],
        "lerobot_python": {"path": str(settings.lerobot_python), "exists": settings.lerobot_python.exists()},
        "depth_python": {"path": str(settings.depth_python), "exists": settings.depth_python.exists()},
    }


def save_boxes(boxes_2k: dict[str, list[int]]) -> Path:
    """Write boxes into configs/rig.yaml (created from the example if missing), keeping its comments."""
    target = rig.CANDIDATES[0]
    if not target.exists():
        target.write_text(rig.CANDIDATES[1].read_text())
    text = target.read_text()
    for name, box in boxes_2k.items():
        if name not in BOX_NAMES or len(box) != 4:
            raise ValueError(f"bad box {name}")
        values = ", ".join(str(int(v)) for v in box)
        pattern = re.compile(rf"^(\s+{name}:\s*)\[[^\]]*\]", re.M)
        if pattern.search(text):
            text = pattern.sub(lambda m, values=values: f"{m.group(1)}[{values}]", text, count=1)
        elif "boxes_2k:" in text:
            text = text.replace("boxes_2k:\n", f"boxes_2k:\n  {name}: [{values}]\n", 1)
        else:
            text += f"\nboxes_2k:\n  {name}: [{values}]\n"
    target.write_text(text)
    rig.load.cache_clear()
    return target


def register(app, hub, runner: JobRunner | None = None) -> JobRunner:
    runner = runner or JobRunner()
    quest = QuestState()
    workers = {w.name: w for w in hub.workers}

    def worker(name: str):
        if name not in workers:
            raise HTTPException(status_code=404, detail=f"no camera {name!r}")
        return workers[name]

    async def guarded(action, *args, **kwargs):
        try:
            return await asyncio.to_thread(action, *args, **kwargs)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/console")
    async def console_page() -> RedirectResponse:
        # The console moved to the collection app (python -m bimanual_collect app), which also runs the arms.
        return RedirectResponse("http://localhost:8440/", status_code=307)

    @app.get("/api/status")
    async def status() -> dict:
        health = hub.health()
        return {**hub.record_status(), "cameras_health": health["cameras"], "quest": quest.get(),
                "time": time.time()}

    @app.get("/api/view/{name}.jpg")
    def preview(name: str) -> Response:
        cam = worker(name)
        frame, encoder = cam.latest, cam.view
        if (cam.state != "ok" or frame is None or encoder is None
                or time.perf_counter_ns() - frame.dequeue_monotonic_ns > 500_000_000):
            raise HTTPException(status_code=503, detail="no recent camera frame")
        jpeg = encoder.jpeg(frame, max_edge=480, quality=65)
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/view/{name}.mjpg")
    async def view(name: str, request: Request, fps: float = 10.0) -> StreamingResponse:
        cam = worker(name)
        interval = 1.0 / max(1.0, min(float(fps), 30.0))

        async def frames():
            cam.acquire_view()
            try:
                sequence = 0
                while not await request.is_disconnected():
                    snapshot = await asyncio.to_thread(cam.frames.wait_after, sequence, 2.0)
                    if snapshot is None:
                        continue
                    sequence = snapshot.sequence
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(snapshot.jpeg)).encode()
                           + b"\r\n\r\n" + snapshot.jpeg + b"\r\n")
                    await asyncio.sleep(interval)
            finally:
                cam.release_view()

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/api/sessions")
    async def list_sessions() -> list[dict]:
        return await asyncio.to_thread(sessions)

    @app.get("/api/episodes")
    async def list_episodes(session: str) -> list[dict]:
        folder = _data_path(session)
        episodes = sorted(p for p in folder.glob("episode_*") if (p / "episode.json").exists())
        return [catalog.index_row(e) | {"files": _episode_files(e)} for e in episodes]

    @app.get("/api/episode")
    async def episode_detail(path: str) -> dict:
        folder = _data_path(path)
        if not (folder / "episode.json").exists():
            raise HTTPException(status_code=404, detail="no such episode")
        meta = catalog.read_json(folder / "episode.json")
        return {"row": catalog.index_row(folder), "meta": meta, "qc": catalog.read_json(folder / "episode_qc.json"),
                "review": catalog.review(folder), "files": _episode_files(folder),
                "labels": {k: catalog.label_of(folder, k, meta) for k in ("task", "outcome", "operator", "note")}}

    @app.get("/api/file")
    async def data_file(path: str) -> FileResponse:
        target = _data_path(path)
        if target.suffix not in SERVED_SUFFIXES or not target.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(target, media_type=SERVED_SUFFIXES[target.suffix])

    @app.post("/api/review")
    async def review(payload: dict) -> dict:
        folder = _data_path(str(payload.get("path", "")))
        record = await guarded(write_review, folder, payload.get("status"), payload.get("task"),
                               payload.get("outcome"), payload.get("note"))
        await asyncio.to_thread(catalog.write_index)
        return record

    @app.post("/api/jobs")
    async def submit_job(payload: dict) -> dict:
        kind = str(payload.pop("kind", ""))
        return await guarded(runner.submit, kind, payload)

    @app.get("/api/jobs")
    async def list_jobs() -> list[dict]:
        return runner.list()

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: int) -> dict:
        return await guarded(runner.get, job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: int) -> dict:
        return await guarded(runner.cancel, job_id)

    @app.get("/api/setup")
    async def setup() -> dict:
        return await asyncio.to_thread(setup_info, hub)

    @app.get("/api/doctor")
    async def doctor_checks() -> list[dict]:
        from .doctor import run_checks

        return [check.__dict__ for check in await asyncio.to_thread(run_checks)]

    @app.get("/api/setup/still.jpg")
    async def still() -> Response:
        """The primary camera's full left eye now (for drawing boxes), natural colour."""
        latest = hub.primary.latest
        if latest is None:
            raise HTTPException(status_code=503, detail="no frame yet")
        eye_width = latest.width // 2 if hub.primary.spec.stereo else latest.width
        raw = np.frombuffer(latest.data, np.uint8).reshape(latest.height, latest.width, 2)
        image = EyeDecoder(colour_fix=hub.primary.spec.colour_fix).bgr(raw, 0, eye_width)
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(jpeg.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    def _calib_and_width():
        source = hub.primary.source
        eye_width = source.width // 2 if hub.primary.spec.stereo else source.width
        calib = rig.load().calibration(str(hub.primary.info.get("serial") or "") or None)
        intrinsics(calib, eye_width)  # raises if the file or the mode's section is missing
        return calib, eye_width

    @app.get("/api/setup/boxes")
    async def get_boxes() -> dict:
        calib, eye_width = await guarded(_calib_and_width)
        boxes = rig.load().boxes_2k
        return {"eye_size": [eye_width, hub.primary.source.height],
                "boxes": {name: list(map_box_from_2k(boxes[name], calib, eye_width)) for name in boxes}}

    @app.post("/api/setup/boxes")
    async def set_boxes(payload: dict) -> dict:
        calib, eye_width = await guarded(_calib_and_width)
        boxes = {name: list(map_box_to_2k(tuple(int(v) for v in box), calib, eye_width))
                 for name, box in (payload.get("boxes") or {}).items()}
        target = await guarded(save_boxes, boxes)
        return {"saved": str(target.relative_to(rig.ROOT)), "boxes_2k": boxes}

    return runner
