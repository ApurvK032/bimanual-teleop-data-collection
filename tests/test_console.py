"""Console API with a fake camera: status, live view, labels, sessions/episodes, review, file access, jobs, boxes."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bimanual_collect import console, rig
from bimanual_collect.recorder import Hub
from tests.test_recorder import FlakyCamera, wait_until


@pytest.fixture
def client(tmp_path, monkeypatch):
    # A rig config pointing at a temporary data folder, through the environment, so job subprocesses use it too.
    example = (rig.ROOT / "configs" / "rig.example.yaml").read_text()
    config = tmp_path / "rig.yaml"
    config.write_text(example.replace("  data_raw: data/raw ", f"  data_raw: {tmp_path / 'raw'} ", 1))
    monkeypatch.setenv("BIMANUAL_RIG", str(config))
    rig.load.cache_clear()
    assert rig.load().data_raw == tmp_path / "raw"
    hub = Hub(data_root=tmp_path / "raw", camera=FlakyCamera(), ffv1_threads=2)
    hub.start()
    app = FastAPI()
    runner = console.register(app, hub)
    with TestClient(app) as test_client:
        yield test_client, hub, runner, tmp_path / "raw"
    hub.stop()
    rig.load.cache_clear()


def test_page_and_status(client) -> None:
    http, hub, _, _ = client
    moved = http.get("/console", follow_redirects=False)  # the page lives in the collection app now
    assert moved.status_code == 307 and moved.headers["location"] == "http://localhost:8440/"
    status = http.get("/api/status").json()
    assert status["recording"] is False and status["cameras_health"][0]["state"] == "ok"
    assert "quest" in status and "warnings" in status


def test_live_view_streams_jpegs_and_releases_the_camera(client) -> None:
    http, hub, _, _ = client
    with http.stream("GET", "/api/view/zed.mjpg?fps=30") as response:
        assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
        chunk = b""
        for part in response.iter_bytes():
            chunk += part
            if chunk.count(b"\xff\xd9") >= 2:
                break
    assert b"Content-Type: image/jpeg" in chunk and b"\xff\xd8" in chunk
    assert wait_until(lambda: not hub.primary.view_enabled.is_set(), 5)  # the stream ended: no more JPEGs made
    assert http.get("/api/view/nope.mjpg").status_code == 404


def test_preview_returns_one_small_frame_without_enabling_the_stream(client) -> None:
    http, hub, _, _ = client
    response = http.get("/api/view/zed.jpg")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert max(image.shape[:2]) <= 480
    assert not hub.primary.view_enabled.is_set()


def test_episode_review_files_and_paths(client) -> None:
    http, hub, _, root = client
    hub.start_episode("fold the towel", operator="ana")
    time.sleep(0.4)
    hub.stop_episode(outcome="success")
    sessions = http.get("/api/sessions").json()
    assert len(sessions) == 1 and sessions[0]["episodes"] == 1
    rows = http.get("/api/episodes", params={"session": sessions[0]["path"]}).json()
    episode = rows[0]["episode"]
    assert rows[0]["task"] == "fold the towel" and rows[0]["outcome"] == "success"
    detail = http.get("/api/episode", params={"path": episode}).json()
    assert detail["labels"]["operator"] == "ana" and detail["meta"]["status"] == "complete"
    result = http.post("/api/review", json={"path": episode, "status": "accepted", "note": "good"}).json()
    assert result["status"] == "accepted" and result["note"] == "good"
    assert json.loads((root / episode / "review.json").read_text())["status"] == "accepted"
    assert http.get("/api/file", params={"path": f"{episode}/episode.json"}).status_code == 200
    assert http.get("/api/file", params={"path": "../../etc/passwd"}).status_code == 400
    assert http.get("/api/file", params={"path": f"{episode}/cameras/zed/video.mkv"}).status_code == 404  # not served
    assert http.post("/api/review", json={"path": episode, "status": "maybe"}).status_code == 400


def test_jobs_validate_and_run(client) -> None:
    http, hub, runner, root = client
    assert http.post("/api/jobs", json={"kind": "process", "paths": ["../.."]}).status_code == 400
    assert http.post("/api/jobs", json={"kind": "bogus", "paths": []}).status_code == 400
    hub.start_episode("x")
    time.sleep(0.3)
    hub.stop_episode()
    session = http.get("/api/sessions").json()[0]["path"]
    job = http.post("/api/jobs", json={"kind": "process", "paths": [session]}).json()
    assert job["status"] in ("queued", "running")
    assert wait_until(lambda: http.get(f"/api/jobs/{job['id']}").json()["status"] in ("done", "failed"), 60)
    finished = http.get(f"/api/jobs/{job['id']}").json()
    assert finished["status"] == "done", finished["log"]
    assert (root / session / "episode_0000" / "episode_qc.json").exists()


def test_box_round_trip_through_2k(client, tmp_path, monkeypatch) -> None:
    http, hub, _, _ = client
    calib = tmp_path / "SN1.conf"
    calib.write_text("[LEFT_CAM_2K]\nfx=1000\nfy=1000\ncx=1100\ncy=600\n"
                     "[LEFT_CAM_VGA]\nfx=250\nfy=250\ncx=330\ncy=180\n")
    monkeypatch.setattr(console.rig.load().__class__, "calibration", lambda self, serial=None: calib)
    saved = {}
    monkeypatch.setattr(console, "save_boxes", lambda boxes: saved.update(boxes) or Path(rig.ROOT / "configs" / "rig.yaml"))
    from bimanual_collect import calibration
    monkeypatch.setitem(calibration.SECTION, 336, "VGA")  # the fake camera's eye is 336 px wide
    monkeypatch.setitem(calibration.EYE_HEIGHT, 336, 188)
    response = http.post("/api/setup/boxes", json={"boxes": {"table": [80, 30, 230, 160]}})
    assert response.status_code == 200, response.text
    assert saved["table"] == [100, 0, 700, 520]  # (x - 330) * 4 + 1100, (y - 180) * 4 + 600
    assert http.get("/api/setup/still.jpg").headers["content-type"] == "image/jpeg"
