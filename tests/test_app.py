"""The collection app's API: status, refusals before arm runs, the camera-service proxy, start/stop."""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bimanual_collect import app as app_mod
from bimanual_collect import arm_runner, bench, rig
from bimanual_collect.quest import QuestManager
from bimanual_collect.services import ServiceManager

FAKE = Path(__file__).with_name("fake_launcher.py")
STATE = bench.DEFAULTS | {"arms": 2, "left_ip": "192.0.2.3", "right_ip": "192.0.2.2", "identified": True,
                          "spacing_mm": 590, "spacing_measured": True}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def client(tmp_path, monkeypatch):
    rig_file = tmp_path / "rig.yaml"
    rig_file.write_text(rig.CANDIDATES[1].read_text().replace("state: state", f"state: {tmp_path / 'state'}"))
    monkeypatch.setenv("BIMANUAL_RIG", str(rig_file))
    monkeypatch.setattr(rig, "CANDIDATES", (rig_file, rig.CANDIDATES[1]))
    monkeypatch.setattr(bench, "SOURCE_IP", "192.0.2.10")  # documentation addresses only
    monkeypatch.setattr(bench, "SUBNET", "192.0.2.0/24")
    for name, value in (("LOCAL", tmp_path), ("STATE_FILE", tmp_path / "bench.json"), ("PROFILE", tmp_path / "bench.yaml"),
                        ("SIDE_PROFILES", {s: tmp_path / f"bench_{s}.yaml" for s in ("left", "right")})):
        monkeypatch.setattr(bench, name, value)
    rig.load.cache_clear()
    monkeypatch.setattr(app_mod, "CAMERA", f"http://127.0.0.1:{free_port()}")  # nothing listens there
    services = ServiceManager(start=False)
    quest = QuestManager(start=False, page_status_url=f"http://127.0.0.1:{free_port()}/quest/status")
    benchy = app_mod.Bench(services, quest)
    benchy.runner.runs_roots = [tmp_path / "runs"]
    http = TestClient(app_mod.create_app(services, quest, benchy))
    yield http, benchy, tmp_path, monkeypatch
    run = benchy.runner.run
    if run is not None and run.active and run.process is not None:
        run.process.kill()
    benchy.runner.close()
    rig.load.cache_clear()


def fake_spec(tmp_path, mode="normal"):
    def spec(state, kind, side=None):
        return {"kind": kind, "command": [sys.executable, str(FAKE), str(tmp_path / "runs"), f"app-{kind}", mode, "30"],
                "cwd": str(tmp_path), "title": f"fake {kind}", "label": f"app-{kind}", "profile": "fake", "token":
                "LIVE-WIDOWXAI-DUAL-192.0.2.3-192.0.2.2" if kind in ("live", "guarded") else None,
                "arms": {"left": "192.0.2.3", "right": "192.0.2.2"}, "cap_s": 30, "error_limit_rad": 0.12,
                "headset_mode": "bimanual"}
    return spec


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    raise AssertionError("timed out")


def test_status_and_page(client):
    http, benchy, _, _ = client
    assert "Data collection" in http.get("/").text
    assert http.get("/console/console.js").status_code == 200
    assert http.get("/console/..%2Fapp.py").status_code == 404
    card = http.get("/console/controls.html")
    assert card.status_code == 200 and "operator controls" in card.text and card.headers["content-type"].startswith("text/html")
    pdf = http.get("/console/controls.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF") and pdf.headers["content-type"] == "application/pdf"
    status = http.get("/api/app/status").json()
    assert status["camera"] is None and status["camera_error"]
    assert status["bench"]["complete"] is False and status["arms"]["run"] is None
    assert set(status["services"]) == {"camera", "quest_page"}


def test_proxy_reports_a_missing_camera_service(client):
    http, *_ = client
    response = http.get("/api/sessions")
    assert response.status_code == 503 and "camera service" in response.json()["detail"]
    assert http.get("/not-proxied").status_code == 404


def test_live_start_needs_both_ticks_and_the_right_token(client):
    http, benchy, tmp_path, monkeypatch = client
    monkeypatch.setattr(bench, "run_spec", fake_spec(tmp_path))
    monkeypatch.setattr(app_mod.Bench, "blocking", lambda self, kind, side=None: [])
    token = "LIVE-WIDOWXAI-DUAL-192.0.2.3-192.0.2.2"
    refused = http.post("/api/arms/start", json={"kind": "live", "token": token, "confirm": {"resting": True}})
    assert refused.status_code == 400 and "power switch" in refused.json()["detail"]
    wrong = http.post("/api/arms/start", json={"kind": "live", "token": "LIVE-X", "confirm": {"resting": True, "switch": True}})
    assert wrong.status_code == 400 and "out of date" in wrong.json()["detail"]
    assert benchy.runner.run is None
    expired = http.post("/api/arms/start", json={"kind": "live", "token": token, "expires_at": time.time() - 1,
                                                "confirm": {"resting": True, "switch": True}})
    assert expired.status_code == 408 and benchy.runner.run is None
    # accepted: it waits for the headset (none here), and Stop before launch just cancels
    ok = http.post("/api/arms/start", json={"kind": "live", "token": token, "confirm": {"resting": True, "switch": True}})
    assert ok.status_code == 200 and ok.json()["phase"] == "waiting_headset"
    assert http.get("/api/arms/brief").json()["line"].startswith("ARMS")
    stopped = http.post("/api/arms/stop").json()
    assert stopped["outcome"] == "cancelled" and benchy.runner.run.pid is None


def test_failing_checks_block_the_start(client):
    http, benchy, tmp_path, monkeypatch = client
    monkeypatch.setattr(bench, "run_spec", fake_spec(tmp_path))
    response = http.post("/api/arms/start", json={"kind": "sim"})
    assert response.status_code == 400 and "Bench set up" in response.json()["detail"]
    checks = http.get("/api/arms/checks?kind=live").json()
    assert checks[0]["id"] == "bench" and not checks[0]["ok"]


def test_practice_run_through_the_api(client):
    http, benchy, tmp_path, monkeypatch = client
    monkeypatch.setattr(bench, "run_spec", fake_spec(tmp_path))
    monkeypatch.setattr(app_mod.Bench, "blocking", lambda self, kind, side=None: [])
    benchy.runner.presence = lambda run: (True, "Headset ready.")
    assert http.post("/api/arms/start", json={"kind": "sim"}).status_code == 200
    wait_for(lambda: benchy.runner.run.phase == "live")
    assert http.post("/api/app/quit").status_code == 400  # not while arms run
    assert http.post("/api/services/quest_page/restart").status_code == 400
    http.post("/api/arms/stop")
    wait_for(lambda: not benchy.runner.run.active)
    status = http.get("/api/app/status").json()["arms"]
    assert status["run"]["outcome"] == "done" and status["history"][0]["kind"] == "sim"
    assert "rest reached" in http.get("/api/arms/log").text


def test_bench_endpoints(client):
    http, benchy, tmp_path, _ = client
    saved = http.post("/api/bench", json=STATE).json()
    assert saved["complete"] and (tmp_path / "bench.yaml").exists()
    assert http.get("/api/app/status").json()["bench"]["complete"]
    bad = http.post("/api/bench", json={"left_ip": "192.0.2.2"})
    assert bad.status_code == 400 and "different addresses" in bad.json()["detail"]
    refused = http.post("/api/bench/validated", json={"operator": "x", "checklist": {"left:right": True, "left:up": False}})
    assert refused.status_code == 400
    plan = http.get("/api/arms/plan?kind=live").json()
    assert plan["token"] == "LIVE-WIDOWXAI-DUAL-192.0.2.3-192.0.2.2" and plan["cap_s"] == 60


def test_force_quit_is_refused_after_motion(client):
    http, benchy, tmp_path, monkeypatch = client
    monkeypatch.setattr(arm_runner, "STOP_GRACE_S", 0.3)
    monkeypatch.setattr(bench, "run_spec", fake_spec(tmp_path, mode="hang"))
    monkeypatch.setattr(app_mod.Bench, "blocking", lambda self, kind, side=None: [])
    benchy.runner.presence = lambda run: (True, "")
    token = "LIVE-WIDOWXAI-DUAL-192.0.2.3-192.0.2.2"
    http.post("/api/arms/start", json={"kind": "live", "token": token, "confirm": {"resting": True, "switch": True}})
    wait_for(lambda: benchy.runner.run.phase == "live")
    http.post("/api/arms/stop")
    wait_for(lambda: benchy.runner.run.status()["stop_overdue"])
    refused = http.post("/api/arms/force-quit")
    assert refused.status_code == 400 and "power" in refused.json()["detail"]
