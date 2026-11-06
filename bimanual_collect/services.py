"""The two background services the app keeps running, as detached processes:

  camera     bimanual_collect.camera_server  127.0.0.1:8444  cameras, recorder, episode/job API (nice 10)
  quest_page bimanual_collect.quest_page     127.0.0.1:8443  the headset page + pose relay

Each runs in its own session with its output in <state>/logs/<name>.log, so restarting or closing the app never
interrupts a recording or the relay an arm run depends on. A new app adopts services that are already running.
Only these two modules are ever matched or stopped: the pattern is `<python> -m bimanual_collect.<module>`, never a
mere mention of the name.
"""
from __future__ import annotations

import grp
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import rig


@dataclass(frozen=True)
class Service:
    name: str
    module: str
    port: int
    health_path: str
    nice: int
    title: str


SERVICES = {
    "camera": Service("camera", "bimanual_collect.camera_server", 8444, "/health", 10, "Cameras and recorder"),
    "quest_page": Service("quest_page", "bimanual_collect.quest_page", 8443, "/quest/status", 0, "Headset page and relay"),
}
BACKOFF_S = (3, 5, 10, 15, 30)


def python_executable() -> str:
    return os.environ.get("BIMANUAL_PYTHON") or sys.executable


def clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONUNBUFFERED"] = "1"
    return env


def service_pids(module: str) -> list[int]:
    """Processes running `<python> -m <module>` (anchored: the program and its first two arguments)."""
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if len(argv) >= 3 and Path(argv[0].decode(errors="ignore")).name.startswith("python") \
                and argv[1] == b"-m" and argv[2] == module.encode():
            found.append(int(proc.name))
    return sorted(found)


def port_open(port: int, timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def http_json(url: str, method: str = "GET", body: dict | None = None, timeout: float = 2.0):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def healthy(service: Service, timeout: float = 1.5) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{service.port}{service.health_path}", timeout=timeout).close()
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def log_path(service: Service) -> Path:
    return rig.load().state / "logs" / f"{service.name}.log"


def tail(path: Path, lines: int = 40, max_bytes: int = 64_000) -> list[str]:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            text = handle.read().decode(errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


def video_signature() -> tuple[str, ...]:
    """Which video devices exist (a camera being plugged in changes it)."""
    try:
        return tuple(sorted(p.name for p in Path("/sys/class/video4linux").iterdir()))
    except OSError:
        return ()


@dataclass
class ServiceState:
    service: Service
    wanted: bool = True
    pid: int | None = None
    started_at: float | None = None
    healthy_since: float | None = None
    failures: int = 0
    restarts: int = 0
    next_try: float = 0.0
    last_exit: str | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)

    def status(self) -> dict:
        pids = service_pids(self.service.module)
        alive = bool(pids)
        ok = self.healthy_since is not None
        if ok:
            state = "running"
        elif alive and self.started_at and time.time() - self.started_at < 60:
            state = "starting"
        elif alive:
            state = "not responding"
        elif not self.wanted:
            state = "stopped"
        else:
            state = "restarting" if self.failures else "starting"
        return {
            "name": self.service.name, "title": self.service.title, "port": self.service.port, "state": state,
            "pids": pids, "restarts": self.restarts, "failures": self.failures,
            "retry_in_s": None if ok or not self.wanted else max(0, round(self.next_try - time.time())),
            "last_exit": self.last_exit, "log": str(log_path(self.service)),
            "log_tail": [] if ok else tail(log_path(self.service), 12),
        }


class ServiceManager:
    """Keeps the services running: adopts running copies, starts missing ones, restarts crashed ones with
    backoff (immediately when a camera is plugged in), and stops them on request."""

    def __init__(self, camera_args: list[str] | None = None, start: bool = True) -> None:
        self.camera_args = list(camera_args or [])
        self.states = {name: ServiceState(service) for name, service in SERVICES.items()}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._video = video_signature()
        self._thread: threading.Thread | None = None
        if start:
            self.start()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="services", daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._stop.set()

    # -- control
    def _launch(self, state: ServiceState) -> None:
        service = state.service
        log = log_path(service)
        log.parent.mkdir(parents=True, exist_ok=True)
        args = self.camera_args if service.name == "camera" else []
        command = [python_executable(), "-m", service.module, *args]
        if service.name == "camera" and rig.load().uses_zed_sdk:
            try:
                zed_group = grp.getgrnam("zed").gr_gid
            except KeyError:
                pass
            else:
                if zed_group not in (*os.getgroups(), os.getegid()):
                    command = ["sg", "zed", "-c", "exec " + shlex.join(command)]
        with open(log, "ab") as handle:
            handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {' '.join(command)}\n".encode())
            handle.flush()
            nice = service.nice
            state.process = subprocess.Popen(
                command, cwd=rig.ROOT, env=clean_env(), stdin=subprocess.DEVNULL, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True,
                preexec_fn=(lambda: os.nice(nice)) if nice else None)
        state.pid, state.started_at, state.healthy_since = state.process.pid, time.time(), None

    def stop_service(self, name: str, timeout_s: float = 15.0) -> None:
        state = self.states[name]
        with self._lock:
            state.wanted = False
        if name == "camera" and healthy(state.service):
            try:  # close an open episode cleanly first (finishes the video, writes episode.json)
                http_json(f"http://127.0.0.1:{state.service.port}/record/stop", "POST", {}, timeout=30)
            except (urllib.error.URLError, OSError, ValueError):
                pass
        pids = service_pids(state.service.module)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + timeout_s
        while time.time() < deadline and service_pids(state.service.module):
            time.sleep(0.2)
        for pid in service_pids(state.service.module):  # these are not arm processes: a hard stop is safe
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if state.process is not None:
            try:
                state.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        state.pid, state.process, state.healthy_since = None, None, None

    def restart(self, name: str) -> None:
        self.stop_service(name)
        state = self.states[name]
        with self._lock:
            state.wanted, state.failures, state.next_try = True, 0, 0.0
        self._tick()

    def want(self, name: str, wanted: bool = True) -> None:
        with self._lock:
            self.states[name].wanted = wanted
            if wanted:
                self.states[name].next_try = 0.0

    def stop_all(self) -> None:
        for name in self.states:
            self.stop_service(name)

    # -- supervision
    def _tick(self) -> None:
        video = video_signature()
        plugged = video != self._video
        self._video = video
        for state in self.states.values():
            service = state.service
            if not state.wanted:
                continue
            if state.process is not None and state.process.poll() is not None:
                state.last_exit = f"exit code {state.process.returncode}"
                state.process = None
            pids = service_pids(service.module)
            if healthy(service):
                if state.healthy_since is None:
                    state.healthy_since = time.time()
                state.failures, state.pid = 0, pids[0] if pids else state.pid
                continue
            state.healthy_since = None
            if pids:
                continue  # starting up, or hung: the UI shows it and offers Restart
            if state.started_at is not None and state.failures == 0 and state.last_exit is None:
                state.last_exit = "stopped"
            now = time.time()
            if service.name == "camera" and plugged:
                state.next_try = 0.0
            if now < state.next_try:
                continue
            if port_open(service.port):
                state.last_exit = f"port {service.port} is used by another program"
                state.next_try = now + BACKOFF_S[-1]
                continue
            if state.started_at is not None:
                state.failures += 1
                state.restarts += 1
            state.next_try = now + BACKOFF_S[min(state.failures, len(BACKOFF_S) - 1)]
            self._launch(state)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                with self._lock:
                    self._tick()
            except Exception as exc:  # noqa: BLE001 - the supervisor must keep going
                print(f"services: {exc}", flush=True)
            self._stop.wait(2.0)

    def status(self) -> dict:
        return {name: state.status() for name, state in self.states.items()}
