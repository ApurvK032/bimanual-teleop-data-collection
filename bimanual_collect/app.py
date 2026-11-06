"""The collection app: one browser UI for the whole bench, at http://localhost:8440/.

    python -m bimanual_collect app            run in this terminal (Ctrl+C quits; a running arm run is left alone)
    python -m bimanual_collect app --open     what the desktop icon runs: start in the background if needed, open
                                              the browser

The app process runs at normal priority and:
  - keeps the camera/recorder service (:8444, nice 10) and the headset page service (:8443) running
    (services.py), and forwards the UI's camera, episode and job requests to the camera service, so the page has
    one address;
  - watches the headset over USB (quest.py);
  - starts and stops the arm launchers (arm_runner.py) after the per-run checklist; it is their parent, so they
    run at normal priority (the camera service is niced and could not start them at full priority);
  - runs the bench wizard (bench.py).

Localhost only. From another computer: ssh -N -L 8440:localhost:8440 <this pc>, then open http://localhost:8440/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import webbrowser
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from . import bench, rig
from . import quest as quest_mod
from .arm_runner import ArmRunner, driver_processes
from .services import ServiceManager, http_json, port_open, tail

PORT = 8440
URL = f"http://localhost:{PORT}/"
CAMERA = "http://127.0.0.1:8444"
WEB = Path(__file__).resolve().parent / "web" / "console"
STATIC = re.compile(r"^[\w.-]+\.(js|css|svg|png|jpg|webp|html|pdf)$")
MEDIA = {"js": "text/javascript", "css": "text/css", "svg": "image/svg+xml", "png": "image/png", "jpg": "image/jpeg",
         "webp": "image/webp", "html": "text/html; charset=utf-8", "pdf": "application/pdf"}
PROXY_PREFIXES = ("api/", "record/", "hub/", "zed/")
HARDWARE_KINDS = ("live", "guarded", "contact")
CONTACT_VALID_S = 8 * 3600  # a passed read-only arm check counts this long, for an unchanged bench profile


class Cache:
    """Results of slower checks, kept for a few seconds (keyed so a changed bench invalidates them)."""

    def __init__(self) -> None:
        self.values: dict[str, tuple[float, object, object]] = {}
        self.lock = threading.Lock()

    def get(self, name: str, key, max_age_s: float, compute):
        with self.lock:
            hit = self.values.get(name)
            if hit and hit[1] == key and time.time() - hit[0] < max_age_s:
                return hit[2]
        value = compute()
        with self.lock:
            self.values[name] = (time.time(), key, value)
        return value

    def drop(self, *names: str) -> None:
        with self.lock:
            for name in names or list(self.values):
                self.values.pop(name, None)


class Bench:
    """App-side state around the arms: checks, the read-only check of this session, first-run validation."""

    def __init__(self, services: ServiceManager, quest: quest_mod.QuestManager) -> None:
        self.services = services
        self.quest = quest
        self.cache = Cache()
        self.contact: dict | None = self._load_contact()  # {"profile_mtime", "at", "arms"} after a passed check
        self.runner = ArmRunner(presence=self.presence)
        self._contact_watch = threading.Thread(target=self._watch_contact, daemon=True)
        self._contact_watch.start()

    # -- helpers
    @staticmethod
    def profile_key() -> float | None:
        try:
            return bench.PROFILE.stat().st_mtime
        except OSError:
            return None

    def presence(self, run) -> tuple[bool, str]:
        return quest_mod.ready_for(self.quest.page(), run.headset_mode, "real")

    @staticmethod
    def _contact_file() -> Path:
        return rig.load().state / "contact_check.json"

    def _load_contact(self) -> dict | None:
        try:
            return json.loads(self._contact_file().read_text())
        except (OSError, ValueError):
            return None

    def _watch_contact(self) -> None:
        """Remember a passed read-only check (for this profile, this app session)."""
        seen = None
        while True:
            run = self.runner.run
            if run is not None and run.kind == "contact" and not run.active and run.id != seen:
                seen = run.id
                if run.outcome == "done" and run.phase in ("contact_ok", "done") and not run.error:
                    passed = any("PREFLIGHT PASSED" in line for line in run.lines)
                    if passed:
                        self.contact = {"profile_mtime": self.profile_key(), "at": time.time(),
                                        "arms": {k: v for k, v in run.contact.items() if not k.startswith("_")}}
                        try:
                            self._contact_file().parent.mkdir(parents=True, exist_ok=True)
                            self._contact_file().write_text(json.dumps(self.contact))
                        except OSError:
                            pass
            time.sleep(0.5)

    def contact_ok(self) -> bool:
        """A read-only check passed within CONTACT_VALID_S, on the bench profile as it is now."""
        return bool(self.contact and self.contact.get("profile_mtime") == self.profile_key()
                    and time.time() - float(self.contact.get("at", 0)) < CONTACT_VALID_S)

    def profile_check(self) -> dict:
        return self.cache.get("profile", self.profile_key(), 600, bench.profile_check)

    def workstation(self) -> dict:
        return self.cache.get("workstation", self.profile_key(), 15, bench.workstation_check)

    # -- the per-run checklist
    def checks(self, kind: str, side: str | None = None) -> list[dict]:
        state = bench.load_state()
        out: list[dict] = []

        def add(ident: str, label: str, ok: bool, detail: str = "", fix: str = "", level: str = "block") -> None:
            out.append({"id": ident, "label": label, "ok": bool(ok), "detail": detail, "fix": fix,
                        "level": "ok" if ok else level})

        complete = bench.complete(state) and bench.PROFILE.exists()
        add("bench", "Bench set up", complete, "" if complete else "arms, sides and spacing not saved yet",
            "Bench tab: finish the arm steps and save")
        if not complete:
            return out
        dual = state["arms"] == 2
        if kind == "sim":
            add("sim_env", "Simulation environment", (rig.ROOT / ".venv" / "bin" / "python").exists(),
                fix="System tab: install")
            return out
        arm_python = bench.teleop_python(True)
        add("arm_env", "Arm driver environment", arm_python.exists(), str(arm_python),
            "System tab: install (it builds teleop/.venv-arm19)")
        check = self.profile_check()
        add("profile", "Bench settings pass the teleop checks", check.get("ok"),
            "gates satisfied" if check.get("ok") else "; ".join(check.get("errors") or []), "Bench tab: review the settings")
        quest = self.quest.status()
        if kind == "dry":
            add("headset", "Headset on USB", quest["usb"] == "connected", quest["usb_message"],
                "The arm program wakes the headset's tracking over USB")
            return out
        ips = [state["left_ip"], state["right_ip"]] if dual else [state["single_ip"]]
        if kind == "guarded" and side:
            ips = [state["left_ip"], state["right_ip"]]  # the read-only arm is connected too
        answered = bench.ping_many(ips)
        missing = [ip for ip, ok in answered.items() if not ok]
        add("ping", "Arms answer on the network", not missing,
            "all answer" if not missing else f"no answer from {', '.join(missing)}",
            "Check the arm power and the network cable; Bench step 1 checks the PC side")
        others = driver_processes()
        add("exclusive", "No other program connected to the arms", not others,
            ", ".join(f"{p['script']} (pid {p['pid']})" for p in others), "Stop it in its terminal (Ctrl+C)")
        if kind == "contact":
            return out
        pc = self.workstation()
        add("workstation", "PC ready (mains power, performance mode, direct route to the arms)", pc.get("ok"),
            pc.get("detail", ""), "Bench step 1 has the fix buttons")
        add("contact", "Read-only arm check passed (last 8 h)", self.contact_ok(),
            "" if self.contact_ok() else "not run, older than 8 h, or the bench changed since",
            "Run the read-only check (button below)")
        if kind == "live":
            add("validated", "First-run check on this bench", bool(state.get("validated")),
                f"done {state['validated']['at']} by {state['validated']['operator'] or 'someone'}" if state.get("validated")
                else "not done: one arm at a time at 0.40 (Bench step 7)", "Bench step 7", level="warn")
        add("headset", "Headset on USB", quest["usb"] == "connected", quest["usb_message"],
            "The arm program wakes the headset's tracking over USB")
        cam = self.services.states["camera"].healthy_since is not None
        add("camera", "Camera recorder running", cam, "" if cam else "episodes can't be recorded until it runs",
            "System tab", level="warn")
        return out

    def blocking(self, kind: str, side: str | None = None) -> list[dict]:
        return [c for c in self.checks(kind, side) if not c["ok"] and c["level"] == "block"]

    def summary(self) -> dict:
        state = bench.load_state()
        return {
            "state": state, "complete": bench.complete(state) and bench.PROFILE.exists(), "ready_to_save": bench.complete(state),
            "saved": bench.STATE_FILE.exists(), "profile": str(bench.PROFILE) if bench.PROFILE.exists() else None,
            "contact": self.contact if self.contact_ok() else None,
            "responses": {k: v[2] for k, v in bench.RESPONSES.items()}
            | {"custom": f"Custom: {state['custom_move']:.2f} move / {state['custom_turn']:.2f} turn"},
            "tracking_stops": bench.TRACKING_STOPS,
            "caps_s": bench.CAPS_S, "clearances_mm": bench.CLEARANCES_MM, "floor_offsets_mm": bench.FLOOR_OFFSETS_MM,
            "ranges": {k: v[:2] for k, v in bench.RANGES.items()},
            "source_ip": bench.SOURCE_IP, "subnet": bench.SUBNET,
        }


def create_app(services: ServiceManager, quest: quest_mod.QuestManager, benchy: Bench, *, quit_callback=None):
    app = FastAPI(title="Collection app")
    client = httpx.AsyncClient()
    runner = benchy.runner
    started = time.time()

    async def blocking(fn, *args, **kwargs):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -- page
    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/console/{name}")
    async def static(name: str) -> FileResponse:
        if not STATIC.match(name) or not (WEB / name).is_file():
            raise HTTPException(status_code=404)
        return FileResponse(WEB / name, media_type=MEDIA[name.rsplit(".", 1)[1]], headers={"Cache-Control": "no-store"})

    # -- status
    @app.get("/api/app/ping")
    async def ping() -> dict:
        return {"app": "bimanual_collect", "pid": os.getpid(), "started": started}

    async def camera_status() -> tuple[dict | None, str | None]:
        try:
            response = await client.get(f"{CAMERA}/api/status", timeout=1.5)
            response.raise_for_status()
            return response.json(), None
        except (httpx.HTTPError, ValueError) as exc:
            return None, str(exc) or exc.__class__.__name__

    @app.get("/api/app/status")
    async def status() -> dict:
        camera, camera_error = await camera_status()
        quest_status, service_status, arms = await asyncio.gather(
            asyncio.to_thread(quest.status), asyncio.to_thread(services.status), asyncio.to_thread(runner.status))
        return {"time": time.time(), "services": service_status, "camera": camera, "camera_error": camera_error,
                "quest": quest_status, "arms": arms, "brief": runner.brief(), "bench": await asyncio.to_thread(benchy.summary)}

    # -- arms
    @app.get("/api/arms/brief")
    async def arms_brief() -> dict:
        return runner.brief()

    @app.get("/api/arms/checks")
    async def arms_checks(kind: str = "live", side: str | None = None) -> list[dict]:
        return await blocking(benchy.checks, kind, side)

    @app.post("/api/arms/start")
    async def arms_start(payload: dict) -> dict:
        kind = str(payload.get("kind", ""))
        side = payload.get("side")
        if kind not in ("live", "guarded", "contact", "dry", "sim"):
            raise HTTPException(status_code=400, detail="unknown run kind")
        expires = payload.get("expires_at")
        if expires is not None and (type(expires) not in (int, float) or not 0 < expires < float("inf")):
            raise HTTPException(status_code=400, detail="invalid start deadline")
        if expires is not None and time.time() >= expires:
            raise HTTPException(status_code=408, detail="Start expired in transit. Check the arm status and try again.")
        state = bench.load_state()
        spec = await blocking(bench.run_spec, state, kind, side)
        if kind in HARDWARE_KINDS:
            confirm = payload.get("confirm") or {}
            if not (confirm.get("resting") is True and confirm.get("switch") is True):
                raise HTTPException(status_code=400, detail="confirm both: arms at rest with the workspace clear, "
                                                            "and someone at the power switch")
            if spec["token"] and payload.get("token") != spec["token"]:
                raise HTTPException(status_code=400, detail="the arms shown in the dialog are out of date: reopen it")
        failing = await blocking(benchy.blocking, kind, side)
        if failing:
            raise HTTPException(status_code=400, detail="not ready: " + "; ".join(
                f"{c['label']} ({c['detail']})" if c["detail"] else c["label"] for c in failing))
        if expires is not None and time.time() >= expires:
            raise HTTPException(status_code=408, detail="Start expired during checks. Check the arm status and try again.")
        result = await blocking(runner.start, kind=kind, title=spec["title"], command=spec["command"], cwd=Path(spec["cwd"]),
                                label=spec["label"], profile=spec["profile"], arms=spec["arms"], cap_s=spec["cap_s"],
                                error_limit_rad=spec["error_limit_rad"], headset_mode=spec["headset_mode"])
        return result

    @app.get("/api/arms/plan")
    async def arms_plan(kind: str = "live", side: str | None = None) -> dict:
        """What a start would run and confirm: title, arms and sides, live token, run length."""
        spec = await blocking(bench.run_spec, bench.load_state(), kind, side)
        return {k: spec[k] for k in ("kind", "title", "arms", "token", "cap_s", "error_limit_rad", "headset_mode", "profile")}

    @app.post("/api/arms/stop")
    async def arms_stop(payload: dict | None = None) -> dict:
        return await blocking(runner.stop, (payload or {}).get("run_id"))

    @app.post("/api/arms/force-quit")
    async def arms_force_quit() -> dict:
        return await blocking(runner.force_quit)

    @app.get("/api/arms/log")
    async def arms_log() -> PlainTextResponse:
        run = runner.run
        if run is None:
            raise HTTPException(status_code=404, detail="no run yet")
        return PlainTextResponse("\n".join(tail(Path(run.log), 2000, 2_000_000)))

    # -- bench wizard
    @app.get("/api/bench")
    async def bench_get() -> dict:
        return await asyncio.to_thread(benchy.summary)

    @app.post("/api/bench")
    async def bench_save(payload: dict) -> dict:
        state = await blocking(bench.save_state, payload)
        written = {}
        if bench.complete(state):
            written = await blocking(bench.write_profiles, state)
        benchy.cache.drop()
        return {"state": state, "written": written, "complete": bench.complete(state)}

    @app.post("/api/bench/validated")
    async def bench_validated(payload: dict) -> dict:
        record = {"at": time.strftime("%Y-%m-%d %H:%M"), "operator": str(payload.get("operator", ""))[:60],
                  "checklist": {k: bool(v) for k, v in (payload.get("checklist") or {}).items()}}
        if not record["checklist"] or not all(record["checklist"].values()):
            raise HTTPException(status_code=400, detail="tick every motion that was checked")
        state = await blocking(bench.save_state, {"validated": record})
        return {"state": state}

    @app.get("/api/bench/network")
    async def bench_network() -> dict:
        return await blocking(bench.network_status)

    @app.post("/api/bench/network/fix")
    async def bench_network_fix(payload: dict) -> dict:
        result = await blocking(bench.fix_network, str(payload.get("ifname", "")))
        benchy.cache.drop("workstation")
        return result

    @app.post("/api/bench/power")
    async def bench_power() -> dict:
        result = await blocking(bench.set_performance)
        benchy.cache.drop("workstation")
        return result

    @app.post("/api/bench/scan")
    async def bench_scan() -> list[dict]:
        return await blocking(bench.scan)

    @app.post("/api/bench/ping")
    async def bench_ping(payload: dict) -> dict:
        ips = [str(ip) for ip in (payload.get("ips") or [])][:8]
        return await blocking(bench.ping_many, ips)

    @app.get("/api/bench/check")
    async def bench_check(fresh: bool = False) -> dict:
        if fresh:
            benchy.cache.drop("profile", "workstation")
        profile, pc = await asyncio.gather(asyncio.to_thread(benchy.profile_check), asyncio.to_thread(benchy.workstation))
        return {"profile": profile, "workstation": pc}

    @app.post("/api/bench/picture")
    async def bench_picture() -> dict:
        return await blocking(bench.sim_picture)

    @app.get("/api/bench/picture.png")
    async def bench_picture_file() -> FileResponse:
        path = rig.load().state / "bench_scene.png"
        if not path.exists():
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    # -- headset
    @app.get("/api/quest")
    async def quest_status() -> dict:
        return await asyncio.to_thread(quest.status)

    @app.post("/api/quest/open")
    async def quest_open(payload: dict | None = None) -> dict:
        mode = str((payload or {}).get("mode") or "bimanual")
        if mode not in ("bimanual", "left", "right"):
            raise HTTPException(status_code=400, detail="mode must be bimanual, left or right")
        return await blocking(quest.open_page, mode)

    @app.post("/api/quest/prepare")
    async def quest_prepare() -> dict:
        device = quest.device
        if quest.state != "connected" or device is None:
            raise HTTPException(status_code=400, detail=quest_mod.MESSAGES.get(quest.state, "no headset"))
        await blocking(quest.prepare, device["serial"])
        return await asyncio.to_thread(quest.status)

    # -- services and system
    @app.get("/api/services")
    async def services_status() -> dict:
        return await asyncio.to_thread(services.status)

    @app.post("/api/services/{name}/restart")
    async def services_restart(name: str) -> dict:
        if name not in services.states:
            raise HTTPException(status_code=404)
        run = runner.run
        if name == "quest_page" and run is not None and run.active:
            raise HTTPException(status_code=400, detail="an arm run is using the headset page: stop it first")
        await asyncio.to_thread(services.restart, name)
        return await asyncio.to_thread(services.status)

    @app.get("/api/services/{name}/log")
    async def services_log(name: str) -> PlainTextResponse:
        if name not in services.states:
            raise HTTPException(status_code=404)
        from .services import log_path

        return PlainTextResponse("\n".join(tail(log_path(services.states[name].service), 400, 400_000)))

    @app.get("/api/doctor")
    async def doctor_checks() -> list[dict]:
        from .doctor import run_checks

        return [check.__dict__ for check in await asyncio.to_thread(run_checks)]

    @app.get("/api/system")
    async def system_info() -> dict:
        from . import shortcut, system_setup

        where = shortcut.paths()
        return {"needs": await asyncio.to_thread(system_setup.needs), "can_ask_password": system_setup.can_ask_password(),
                "shortcut": bool(where["menu"].exists()), "command": " ".join(["sudo", *system_setup.command()[1:]])}

    @app.post("/api/system/setup")
    async def system_run(payload: dict | None = None) -> dict:
        from . import system_setup

        network_if = (payload or {}).get("network_if") or None
        if network_if and not any(n["name"] == network_if for n in bench.interfaces()):
            raise HTTPException(status_code=400, detail=f"{network_if} is not a wired port on this PC")
        result = await asyncio.to_thread(system_setup.run, network_if, bool((payload or {}).get("dry_run")))
        benchy.cache.drop()
        return result

    @app.post("/api/system/calibration")
    async def system_calibration() -> dict:
        from .calibration import fetch_calibration
        from .zed_v4l2 import video_devices

        def fetch() -> dict:
            got = {}
            for device in video_devices():
                if device.is_zed and device.serial and not rig.load().calibration(device.serial).exists():
                    got[device.serial] = str(fetch_calibration(device.serial).relative_to(rig.ROOT))
            return got

        fetched = await blocking(fetch)
        return {"fetched": fetched, "detail": f"downloaded {len(fetched)} file(s)" if fetched else "nothing missing"}

    @app.post("/api/system/shortcut")
    async def system_shortcut() -> dict:
        from . import shortcut

        return await blocking(shortcut.install)

    @app.post("/api/system/installer")
    async def system_installer() -> dict:
        script = rig.ROOT / "scripts" / "install_gui.sh"
        log = rig.load().state / "logs" / "install.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as handle:
            subprocess.Popen(["bash", str(script)], cwd=rig.ROOT, stdin=subprocess.DEVNULL, stdout=handle,
                             stderr=subprocess.STDOUT, start_new_session=True)
        return {"started": True, "detail": "The installer window opens on this PC's screen."}

    @app.post("/api/app/quit")
    async def quit_app() -> dict:
        run = runner.run
        if run is not None and run.active:
            raise HTTPException(status_code=400, detail="stop the arm run first")
        if quit_callback is None:
            raise HTTPException(status_code=400, detail="this app can't quit itself")
        threading.Thread(target=quit_callback, daemon=True).start()
        return {"quitting": True}

    # -- everything else under these prefixes is the camera service's (episodes, jobs, recording, views)
    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request) -> Response:
        if not path.startswith(PROXY_PREFIXES):
            raise HTTPException(status_code=404)
        url = f"{CAMERA}/{path}" + (f"?{request.url.query}" if request.url.query else "")
        streaming = path.endswith(".mjpg") or path.startswith("api/file")
        headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "accept", "range")}
        timeout = httpx.Timeout(connect=2.0, read=None if streaming else 120.0, write=30.0, pool=5.0)
        try:
            upstream = await client.send(client.build_request(request.method, url, content=await request.body(),
                                                              headers=headers, timeout=timeout), stream=True)
        except httpx.HTTPError:
            return JSONResponse({"detail": "The camera service is not running (System tab)."}, status_code=503)
        keep = ("content-type", "content-length", "content-range", "accept-ranges", "cache-control", "last-modified", "etag")
        return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code,
                                 headers={k: v for k, v in upstream.headers.items() if k.lower() in keep},
                                 background=BackgroundTask(upstream.aclose))

    app.router.on_shutdown.append(client.aclose)
    return app


# ----------------------------------------------------------------------------- start-up
def app_running(timeout: float = 1.0) -> bool:
    try:
        return http_json(f"http://127.0.0.1:{PORT}/api/app/ping", timeout=timeout).get("app") == "bimanual_collect"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def notify(message: str) -> None:
    """Tell the user something when there is no terminal (desktop icon)."""
    print(message, flush=True)
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", "Collection app", message], check=False)


def open_browser() -> None:
    if shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", URL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    else:
        webbrowser.open(URL)


def launch_detached(extra: list[str]) -> None:
    log = rig.load().state / "logs" / "app.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    with open(log, "ab") as handle:
        handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} app start\n".encode())
        subprocess.Popen([sys.executable, "-m", "bimanual_collect", "app", *extra], cwd=rig.ROOT, env=env,
                         stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--open", action="store_true", help="start in the background if needed, then open the browser")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser (terminal use)")
    parser.add_argument("--mode", help="camera mode for this session (default: rig config)")
    parser.add_argument("--fps", type=int, help="camera frame rate for this session")
    args = parser.parse_args(argv)
    camera_args = [*(["--mode", args.mode] if args.mode else []), *(["--fps", str(args.fps)] if args.fps else [])]

    if args.open:
        if not app_running():
            launch_detached(camera_args)
            deadline = time.time() + 30
            while time.time() < deadline and not app_running():
                time.sleep(0.3)
            if not app_running():
                notify(f"The collection app did not start; see {rig.load().state / 'logs' / 'app.log'}")
                return 1
        open_browser()
        return 0

    if app_running():
        print(f"The collection app is already running: {URL}")
        if not args.no_browser:
            open_browser()
        return 0
    if port_open(PORT):
        print(f"Port {PORT} is used by another program; stop it first.", file=sys.stderr)
        return 1

    services = ServiceManager(camera_args)
    quest = quest_mod.QuestManager()
    benchy = Bench(services, quest)
    import uvicorn

    server: uvicorn.Server | None = None

    def quit_all() -> None:
        time.sleep(0.3)  # let the HTTP answer go out
        services.close()
        services.stop_all()
        if server is not None:
            server.should_exit = True

    app = create_app(services, quest, benchy, quit_callback=quit_all)
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    print(f"Collection app: {URL}", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, open_browser).start()
    try:
        server.run()
    finally:
        services.close()
        quest.close()
        run = benchy.runner.run
        if run is not None and run.active:
            print("An arm run is still active: it keeps running, and so do the camera and headset services. "
                  "Open the app again to see or stop it.", flush=True)
        elif any(state.wanted for state in services.states.values()):
            services.stop_all()
        benchy.runner.close()
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.exit(main())
