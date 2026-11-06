"""Starts and stops the teleop repo's arm launchers for the app, one at a time, and reports what they are doing.

The launchers (teleop/scripts/run_dual_hardware.py, run_hardware.py, ...) have no control socket. The only safe
stop is the one Ctrl+C gives them: a single SIGINT, which makes them return the arms to rest before closing the
drivers. A second SIGINT during that shutdown, or SIGTERM / SIGKILL, would skip the return to rest (teleop
AGENTS.md rule 10). So this module:
  - starts the launcher in its own session with SIGINT at its default action and unbuffered output to a log file
    (not a pipe: closing or restarting the app can't break the launcher's output);
  - sends exactly one SIGINT on Stop and never anything else once the arms may have moved;
  - follows the launcher's printed lines to know its phase, and its telemetry.csv for live numbers;
  - remembers the run in <state>/arm_run.json so a restarted app picks it up again.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import rig
from .activity import launcher_of

# Processes that open a driver session to an arm. Only one may run at a time (the controller takes one client).
DRIVER_SCRIPTS = (b"run_dual_hardware.py", b"run_hardware.py", b"run_cad_hardware.py", b"preflight_hardware.py",
                  b"preflight_dual_hardware.py", b"park_arms.py", b"commission_one_arm.py", b"run_six_axis_diagnostic.py",
                  b"leader_follower.py")
SIM_SCRIPTS = (b"run_dual_sim.py", b"run_live_sim.py")

# kind -> (touches the arms, needs the headset, produces telemetry)
KINDS = {
    "live": (True, True, True),        # the arms move
    "guarded": (True, True, True),     # one arm moves, the other is connected read-only and screens every command
    "contact": (True, False, False),   # read-only: connect, read joints and versions, disconnect
    "dry": (False, True, True),        # the hardware launcher with simulated arms (no --live): no arm is contacted
    "sim": (False, True, True),        # MuJoCo simulation of the bench
}

# (pattern, phase). Checked in order on every line; the last match wins for that line.
PHASES = [
    (r"offline dual-arm checks PASSED|^combined scene: home pose clear|guarded .*paths clear", "checked"),
    (r"^preflight: workstation", "pc_checked"),
    (r"waiting for fresh (Quest )?tracking", "waiting_controllers"),
    (r"^preflight: (left |right )?(arm @|driver=)|^preflight: stationary|read-only controller state", "connected"),
    (r"^startup: (measured pose|ramping|opening)", "moving_home"),
    (r"^\s*(left|right): held q", "moving_home"),
    (r"^home reached", "home"),
    (r"^(LIVE|DRY RUN)", "live"),
    (r"^operator stop|^stopped; ", "stopping"),
    (r"^telemetry: ", "stopping"),
    (r"^shutdown: returning", "returning"),
    (r"^rest reached: (left|right) arm$", "returning"),
    (r"^rest reached: driver cleanup|^shutdown summary", "at_rest"),
    (r"NO-MOTION (DUAL |STATE )?PREFLIGHT PASSED|DISCOVERY-ONLY PREFLIGHT PASSED", "contact_ok"),
]
MOTION_PHASES = {"moving_home", "home", "live", "stopping", "returning", "at_rest"}
ERRORS = [
    (r"SAFETY STOP: (.*)", "Safety stop: {0}"),
    (r"SAFETY STOP: adb get-state failed(.*)", "The headset is not on USB: connect it (the program wakes its tracking over USB)"),
    (r"PREFLIGHT FAILED: (.*)", "Check failed: {0}"),
    (r"(?:live )?workstation preflight failed: (.*)", "PC check failed: {0}"),
    (r"WORKSTATION PREFLIGHT FAILED: (.*)", "PC check failed: {0}"),
    (r"(dual )?Quest preflight failed: (.*)", "Headset not ready: {1}"),
    (r"live (dual-arm )?output remains disabled[:;]? ?(.*)", "Live output refused: {1}"),
    (r"invalid (?:dual-arm |hardware safety )?configuration: (.*)", "Bench settings rejected: {0}"),
]
EMERGENCY = re.compile(r"EMERGENCY SHUTDOWN WARNING: (.*)")
LABELS = {
    "waiting_headset": "Waiting for the headset: enter VR, both controllers tracked, grips released",
    "starting": "Starting",
    "checked": "Checks passed",
    "pc_checked": "PC checks passed",
    "waiting_controllers": "Waiting for both controllers (grips released)",
    "connected": "Connected to the arms",
    "moving_home": "Moving to home: keep the grips released",
    "home": "At home: hold a grip to drive that arm",
    "live": "Live",
    "stopping": "Stopping: returning to rest",
    "returning": "Returning to rest",
    "at_rest": "At rest",
    "contact_ok": "Read-only check passed",
    "done": "Finished",
    "failed": "Stopped with an error",
    "cancelled": "Cancelled",
}
STOP_GRACE_S = 20.0  # after one SIGINT, a launcher that is still alive this long needs the power switch
HEADSET_WAIT_S = 120.0


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")[:60] or "run"


def proc_start_ticks(pid: int) -> int | None:
    """The process's start time in clock ticks (identifies it across pid reuse); None if gone or a zombie."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = stat.rsplit(")", 1)[1].split()
    return None if fields[0] == "Z" else int(fields[19])


def driver_processes(exclude: set[int] = frozenset()) -> list[dict]:
    """Running processes that can hold an arm's driver session (started by anyone, e.g. in a terminal)."""
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in exclude:
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        script = launcher_of(argv, DRIVER_SCRIPTS)
        if script:
            found.append({"pid": int(proc.name), "script": script})
    return found


@dataclass
class TelemetryTail:
    """Reads a growing telemetry.csv incrementally and keeps the numbers the UI shows."""

    path: Path
    offset: int = 0
    header: list[str] | None = None
    partial: str = ""
    rows: deque = field(default_factory=lambda: deque(maxlen=900))  # ~10 s at 90 Hz
    first_ns: int | None = None
    ik_failures: dict = field(default_factory=dict)
    held_ticks: int = 0
    max_error: dict = field(default_factory=dict)
    total_rows: int = 0

    def poll(self) -> None:
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self.offset)
                chunk = handle.read(4_000_000)
        except OSError:
            return
        if not chunk:
            return
        self.offset += len(chunk)
        text = self.partial + chunk.decode(errors="replace")
        lines = text.split("\n")
        self.partial = lines.pop()
        if self.header is None and lines:
            self.header = next(csv.reader([lines.pop(0)]))
        for values in csv.reader(io.StringIO("\n".join(lines))):
            if self.header is None or len(values) != len(self.header):
                continue
            self._add(dict(zip(self.header, values)))

    def sides(self) -> list[str]:
        header = self.header or []
        sides = [s for s in ("left", "right") if f"{s}_q_feedback" in header]
        return sides or [""]

    def _add(self, row: dict) -> None:
        try:
            t = int(row["pc_monotonic_ns"])
        except (KeyError, ValueError):
            return
        self.total_rows += 1
        if self.first_ns is None:
            self.first_ns = t
        if row.get("control_state", "normal") not in ("", "normal", "returning_home", "at_home", "home_hold"):
            self.held_ticks += 1
        for side in self.sides():
            p = f"{side}_" if side else ""
            status = row.get(f"{p}ik_status", "")
            if status not in ("", "ok", "orientation_parked", "None"):
                self.ik_failures[side] = self.ik_failures.get(side, 0) + 1
            if row.get(f"{p}feedback_sample_fresh") == "True":
                try:
                    err = max(abs(float(v)) for v in json.loads(row.get(f"{p}q_feedback_error") or "[]"))
                except (ValueError, TypeError):
                    err = None
                if err is not None:
                    self.max_error[side] = max(self.max_error.get(side, 0.0), err)
                    row["_err_" + side] = err
        self.rows.append((t, row))

    def summary(self) -> dict:
        if not self.rows:
            return {"rows": 0}
        t_last, last = self.rows[-1]
        window = [t for t, _ in self.rows if t_last - t <= 2e9]
        loop_hz = (len(window) - 1) / ((window[-1] - window[0]) / 1e9) if len(window) > 2 and window[-1] > window[0] else None
        arms = {}
        for side in self.sides():
            p = f"{side}_" if side else ""
            recent = [row.get("_err_" + side) for t, row in self.rows if t_last - t <= 5e9]
            recent = [e for e in recent if e is not None]
            grip = last.get(f"{p}grip") or last.get(f"{p}quest_grip") or "0"
            arms[side or "arm"] = {
                "tracked": last.get(f"{p}tracked", "True") == "True",
                "grip": round(float(grip or 0), 2),
                "clutch": last.get(f"{p}clutch_engaged") == "True",
                "error_rad": round(max(recent), 4) if recent else None,
                "max_error_rad": round(self.max_error.get(side, 0.0), 4),
                "ik_failures": self.ik_failures.get(side, 0),
                "held": last.get(f"{p}held") == "True",
            }
        return {
            "rows": self.total_rows, "loop_hz": None if loop_hz is None else round(loop_hz, 1),
            "elapsed_s": round((t_last - self.first_ns) / 1e9, 1), "control_state": last.get("control_state", ""),
            "held_ticks": self.held_ticks, "arms": arms, "age_s": round(time.time() - self.path.stat().st_mtime, 1)
            if self.path.exists() else None,
        }


@dataclass
class Run:
    id: str
    kind: str
    title: str
    command: list[str]
    cwd: str
    log: str
    label: str
    cap_s: float | None
    error_limit_rad: float | None
    profile: str
    arms: dict
    headset_mode: str = "bimanual"  # what the headset must stream before launch: bimanual | left | right
    created: float = field(default_factory=time.time)
    pid: int | None = None
    pid_ticks: int | None = None
    started: float | None = None
    phase: str = "starting"
    phase_at: float = field(default_factory=time.time)
    live_at: float | None = None
    motion_started: bool = False
    sigint_at: float | None = None
    finished: float | None = None
    returncode: int | None = None
    outcome: str | None = None  # done | failed | cancelled | stopped
    error: str | None = None
    emergency: str | None = None
    message: str = ""
    lines: deque = field(default_factory=lambda: deque(maxlen=400))
    contact: dict = field(default_factory=dict)  # read-only check: per arm versions and joints
    adopted: bool = False
    telemetry: str | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)
    log_offset: int = field(default=0, repr=False)
    tail: TelemetryTail | None = field(default=None, repr=False)

    @property
    def touches_arms(self) -> bool:
        return KINDS[self.kind][0]

    @property
    def active(self) -> bool:
        return self.finished is None

    def record(self) -> dict:
        keys = ("id", "kind", "title", "command", "cwd", "log", "label", "cap_s", "error_limit_rad", "profile", "arms",
                "headset_mode", "created", "pid", "pid_ticks", "started", "sigint_at", "motion_started", "live_at")
        return {key: getattr(self, key) for key in keys}

    def status(self) -> dict:
        now = time.time()
        stop_overdue = bool(self.active and self.sigint_at and now - self.sigint_at > STOP_GRACE_S)
        time_left = None
        if self.live_at and self.cap_s and self.active:
            time_left = max(0.0, self.cap_s - (now - self.live_at))
        return {
            "id": self.id, "kind": self.kind, "title": self.title, "label": self.label, "profile": self.profile,
            "touches_arms": self.touches_arms, "headset_mode": self.headset_mode,
            "arms": self.arms, "active": self.active, "phase": self.phase, "phase_label": LABELS.get(self.phase, self.phase),
            "phase_s": round(now - self.phase_at, 1), "motion_started": self.motion_started,
            "stop_sent": self.sigint_at is not None, "stop_overdue": stop_overdue,
            "can_force_quit": self.can_force_quit(), "started": self.started, "finished": self.finished,
            "elapsed_s": None if not self.started else round((self.finished or now) - self.started, 1),
            "time_left_s": None if time_left is None else round(time_left), "cap_s": self.cap_s,
            "error_limit_rad": self.error_limit_rad, "outcome": self.outcome, "returncode": self.returncode,
            "error": self.error, "emergency": self.emergency, "message": self.message, "adopted": self.adopted,
            "contact": {k: v for k, v in self.contact.items() if not k.startswith("_")}, "telemetry_dir": self.telemetry,
            "live": self.tail.summary() if self.tail is not None else None,
            "log": list(self.lines)[-60:], "log_file": self.log,
        }

    def can_force_quit(self) -> bool:
        if not self.active or self.pid is None or self.sigint_at is None or time.time() - self.sigint_at < 5:
            return False  # only for a program that ignored Stop for a few seconds
        if not self.touches_arms:
            return True
        # Before any motion (waiting for controllers, connecting) a hung launcher may be stopped (AGENTS rule 8).
        # After motion started, only the arm's power switch may end it without the return to rest.
        return not self.motion_started


class ArmRunner:
    """One arm process at a time. `presence` says whether the headset is ready (ok, message); it is polled while
    a run that needs the headset waits to launch."""

    def __init__(self, state_dir: Path | None = None, runs_roots=None, presence: Callable[[Run], tuple[bool, str]] | None = None,
                 poll_s: float = 0.25) -> None:
        settings = rig.load()
        self.state_dir = Path(state_dir or settings.state)
        self.runs_roots = [Path(p) for p in (runs_roots or settings.teleop_runs)]
        self.presence = presence
        self.poll_s = poll_s
        self.run: Run | None = None
        self.history: deque[dict] = deque(maxlen=20)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._adopt()
        self._thread = threading.Thread(target=self._loop, name="arm-runner", daemon=True)
        self._thread.start()

    # -- persistence
    @property
    def record_path(self) -> Path:
        return self.state_dir / "arm_run.json"

    def _save(self) -> None:
        run = self.run
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        if run is None or not run.active or run.pid is None:
            self.record_path.unlink(missing_ok=True)
            return
        tmp = self.record_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(run.record(), indent=1))
        tmp.replace(self.record_path)

    def _adopt(self) -> None:
        try:
            data = json.loads(self.record_path.read_text())
        except (OSError, ValueError):
            return
        pid, ticks = data.get("pid"), data.get("pid_ticks")
        if not pid or proc_start_ticks(pid) != ticks:
            self.record_path.unlink(missing_ok=True)
            return
        run = Run(**{k: data[k] for k in ("id", "kind", "title", "command", "cwd", "log", "label", "cap_s",
                                          "error_limit_rad", "profile", "arms")})
        for key in ("headset_mode", "created", "pid", "pid_ticks", "started", "sigint_at", "motion_started", "live_at"):
            setattr(run, key, data.get(key))
        run.adopted = True
        self.run = run
        self._read_log(run)

    # -- starting
    def busy_reason(self) -> str | None:
        with self._lock:
            if self.run is not None and self.run.active:
                return f"{self.run.title} is still running"
        others = driver_processes()
        if others:
            names = ", ".join(f"{p['script']} (pid {p['pid']})" for p in others)
            return f"another program is connected to the arms: {names}. Stop it first (in its own terminal: Ctrl+C)."
        return None

    def start(self, *, kind: str, title: str, command: list[str], cwd: Path, label: str, profile: str, arms: dict,
              cap_s: float | None = None, error_limit_rad: float | None = None, headset_mode: str = "bimanual") -> dict:
        if kind not in KINDS:
            raise ValueError(f"unknown run kind {kind!r}")
        label = safe_label(label)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        logs = self.state_dir / "logs" / "arms"
        logs.mkdir(parents=True, exist_ok=True)
        run = Run(id=f"{stamp}_{label}", kind=kind, title=title, command=[str(c) for c in command], cwd=str(cwd),
                  log=str(logs / f"{stamp}_{label}.log"), label=label, cap_s=cap_s, error_limit_rad=error_limit_rad,
                  profile=profile, arms=arms, headset_mode=headset_mode)
        with self._lock:
            reason = self.busy_reason()
            if reason:
                raise RuntimeError(reason)
            self.run = run
            if KINDS[kind][1] and self.presence is not None:
                self._set_phase(run, "waiting_headset")
            else:
                self._launch(run)
        self._wake.set()
        return run.status()

    def _launch(self, run: Run) -> None:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONUNBUFFERED"] = "1"
        with open(run.log, "ab") as handle:
            handle.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {run.title}\n=== {' '.join(run.command)}\n".encode())
            handle.flush()

            def child_setup() -> None:
                # Ctrl+C semantics even if the app itself was started with SIGINT ignored (e.g. from a launcher).
                signal.signal(signal.SIGINT, signal.SIG_DFL)

            run.process = subprocess.Popen(run.command, cwd=run.cwd, env=env, stdin=subprocess.DEVNULL, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=child_setup)
        run.pid, run.started = run.process.pid, time.time()
        run.pid_ticks = proc_start_ticks(run.pid)
        self._set_phase(run, "starting")
        self._save()

    # -- stopping
    def stop(self, run_id: str | None = None) -> dict:
        """Ask the run to end: cancels a run still waiting for the headset; otherwise sends ONE SIGINT."""
        with self._lock:
            run = self.run
            if run is None or not run.active:
                raise RuntimeError("nothing is running")
            if run_id is not None and run.id != run_id:
                raise RuntimeError("the arm run changed; check the current status")
            if run.pid is None:
                self._finish(run, "cancelled", None)
                return run.status()
            self._read_log(run)  # the time cap or a safety stop may already have started shutdown
            if run.sigint_at is not None or run.phase in ("stopping", "returning", "at_rest"):
                return run.status()
            try:
                os.kill(run.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            run.sigint_at = time.time()
            self._set_phase(run, "stopping")
            self._save()
            return run.status()

    def force_quit(self) -> dict:
        with self._lock:
            run = self.run
            if run is None or not run.active or run.pid is None:
                raise RuntimeError("nothing is running")
            if not run.can_force_quit():
                if run.touches_arms and run.motion_started:
                    raise RuntimeError("the arms may have moved: a forced quit would skip the return to rest. "
                                       "If the launcher does not stop, cut the arm power at the switch.")
                raise RuntimeError("press Stop first and give it a few seconds")
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.kill(run.pid, sig)
                except ProcessLookupError:
                    break
                for _ in range(30):
                    if not self._alive(run):
                        break
                    time.sleep(0.1)
            run.message = "force-quit before any arm motion"
            return run.status()

    # -- monitoring
    def _alive(self, run: Run) -> bool:
        if run.process is not None:
            return run.process.poll() is None
        return run.pid is not None and proc_start_ticks(run.pid) == run.pid_ticks

    def _set_phase(self, run: Run, phase: str) -> None:
        if run.phase != phase:
            run.phase, run.phase_at = phase, time.time()
        if phase in MOTION_PHASES and KINDS[run.kind][0]:
            run.motion_started = True
        if phase == "live" and run.live_at is None:
            run.live_at = time.time()

    def _parse(self, run: Run, line: str) -> None:
        run.lines.append(line)
        text = line.strip()
        for pattern, phase in PHASES:
            if re.search(pattern, text):
                if not (phase == "stopping" and run.phase in ("returning", "at_rest")):
                    self._set_phase(run, phase)
        for pattern, template in ERRORS:
            match = re.search(pattern, text)
            if match:
                run.error = template.format(*[g or "" for g in match.groups()]).strip()
        if run.motion_started and text.startswith("SAFETY STOP:") and run.phase not in ("returning", "at_rest"):
            self._set_phase(run, "stopping")
        match = EMERGENCY.search(text)
        if match:
            run.emergency = match.group(1).strip()
        contact = re.match(r"^(left|right) arm @ (\S+)", text)
        if contact:
            run.contact.setdefault(contact.group(1), {})["ip"] = contact.group(2)
            run.contact["_current"] = contact.group(1)
        side = run.contact.get("_current", "arm")
        for key in ("driver", "firmware"):
            found = re.search(rf"\b{key}(?:_version)?[=:]\s*'?v?([0-9][\w.]*)", text)
            if found:
                run.contact.setdefault(side, {})[key] = found.group(1)
        joints = re.search(r"\b(?:q_rad|arm_q_rad)[=:]\s*\[([^\]]*)\]", text)
        if joints:
            run.contact.setdefault(side, {})["q_rad"] = [round(float(v), 3) for v in joints.group(1).split(",") if v.strip()]
        gripper = re.search(r"\bgripper_m[=:]\s*(-?[0-9.]+)", text)
        if gripper:
            run.contact.setdefault(side, {})["gripper_m"] = round(float(gripper.group(1)), 4)
        telemetry = re.match(r"^telemetry: (\S+)", text)
        if telemetry:
            run.telemetry = telemetry.group(1)

    def _read_log(self, run: Run) -> None:
        try:
            with open(run.log, "rb") as handle:
                handle.seek(run.log_offset)
                chunk = handle.read()
        except OSError:
            return
        if not chunk:
            return
        complete = chunk.rfind(b"\n") + 1
        if complete == 0:
            return
        run.log_offset += complete
        for raw in chunk[:complete].decode(errors="replace").splitlines():
            if raw.startswith("=== "):
                continue
            self._parse(run, raw.rstrip())

    def _find_telemetry(self, run: Run) -> None:
        if run.tail is not None or not KINDS[run.kind][2] or run.started is None:
            return
        newest = None
        for root in self.runs_roots:
            for day in sorted((p for p in root.glob("*") if p.is_dir()), reverse=True)[:2]:
                for folder in day.glob(f"*_{run.label}"):
                    csv_path = folder / "telemetry.csv"
                    if csv_path.exists() and folder.stat().st_mtime >= run.started - 5:
                        if newest is None or folder.stat().st_mtime > newest.parent.stat().st_mtime:
                            newest = csv_path
        if newest is not None:
            run.tail = TelemetryTail(newest)
            run.telemetry = str(newest.parent)

    def _finish(self, run: Run, outcome: str, returncode: int | None) -> None:
        run.finished, run.returncode = time.time(), returncode
        if run.emergency:
            outcome = "failed"
        elif run.error and outcome == "done":
            outcome = "failed"
        elif outcome == "done" and returncode not in (0, None):
            outcome = "failed"
            last = next((line.strip() for line in reversed(run.lines) if line.strip()), "")
            run.error = run.error or (last[:200] if last else f"the launcher exited with code {returncode}")
        run.outcome = outcome
        self._set_phase(run, {"done": "done", "failed": "failed", "cancelled": "cancelled"}.get(outcome, "done"))
        if run.process is not None and run.process.stdout:
            run.process.stdout.close()
        self.history.appendleft(run.status() | {"log": list(run.lines)[-20:]})
        self._save()

    def _tick(self) -> None:
        with self._lock:
            run = self.run
            if run is None or not run.active:
                return
            if run.pid is None:  # waiting for the headset
                ok, message = self.presence(run) if self.presence else (True, "")
                run.message = message
                if ok:
                    self._launch(run)
                elif time.time() - run.created > HEADSET_WAIT_S:
                    run.error = f"the headset was not ready within {int(HEADSET_WAIT_S)} s: {message}"
                    self._finish(run, "cancelled", None)
                return
            self._read_log(run)
            self._find_telemetry(run)
            if run.tail is not None:
                run.tail.poll()
                # The simulators print no "LIVE" line: their control loop has started once telemetry rows appear.
                if run.kind == "sim" and run.tail.total_rows and run.phase in ("starting", "checked"):
                    self._set_phase(run, "live")
            if not self._alive(run):
                self._read_log(run)
                if run.tail is not None:
                    run.tail.poll()
                code = run.process.returncode if run.process is not None else None
                self._finish(run, "done", code)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001 - monitoring must keep going while an arm runs
                print(f"arm runner: {exc}", flush=True)
            self._wake.wait(self.poll_s)
            self._wake.clear()

    def close(self) -> None:
        """Stop monitoring (the launcher, if any, keeps running and is picked up by the next app)."""
        self._stop.set()
        self._wake.set()

    # -- reporting
    def status(self) -> dict:
        with self._lock:
            current = None if self.run is None else self.run.status()
        return {"run": current, "history": list(self.history)[:5]}

    def brief(self) -> dict:
        """One line for the headset header."""
        run = self.run
        if run is None:
            return {"line": "", "level": "none"}
        if run.emergency:
            return {"line": "CUT ARM POWER NOW", "level": "bad"}
        if not run.active:
            if run.finished and time.time() - run.finished < 15:
                if run.error:
                    return {"line": f"ARMS: {run.error}"[:90], "level": "bad"}
                return {"line": f"ARMS {LABELS.get(run.phase, run.phase).lower()}", "level": "warn"}
            return {"line": "", "level": "none"}
        if run.error:
            return {"line": f"ARMS: {run.error}"[:90], "level": "bad"}
        sim = "" if run.touches_arms else " (sim)"
        if run.phase == "live":
            status = run.status()
            live = status["live"] or {}
            home_state = live.get("control_state") if (live.get("age_s") or 0.0) < 2.0 else None
            if home_state == "returning_home":
                return {"line": f"ARMS{sim} returning home · keep holding X", "level": "warn", "homing": True}
            if home_state == "at_home":
                return {"line": f"ARMS{sim} at home · release X", "level": "ok", "homing": True}
            if home_state in ("home_hold", "home_blocked"):
                return {"line": f"ARMS{sim} return paused · release X to retry", "level": "warn", "homing": True}
            left = status["time_left_s"]
            tail = f" {int(left) // 60}:{int(left) % 60:02d} left" if left is not None else ""
            return {"line": f"ARMS LIVE{sim}{tail}", "level": "ok"}
        if run.phase in ("stopping", "returning"):
            return {"line": f"ARMS{sim} returning to rest", "level": "warn"}
        if run.phase == "moving_home":
            return {"line": f"ARMS{sim} moving to home: grips released", "level": "warn"}
        if run.phase == "home":
            return {"line": f"ARMS{sim} at home: hold a grip to drive", "level": "ok"}
        return {"line": f"ARMS{sim}: {LABELS.get(run.phase, run.phase).lower()}", "level": "warn"}
