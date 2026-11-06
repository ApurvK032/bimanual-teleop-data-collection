"""Background jobs for the console: process a session, export accepted episodes, compute depth. One at a time.

Each job is a `python -m bimanual_collect ...` subprocess at nice 10, in the Python environment it needs (export
uses the LeRobot env, RAFT depth the GPU env). The tools themselves pause while a recording or an arm run is
active, so a job started mid-session waits instead of costing camera frames.
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from . import rig


@dataclass
class Job:
    id: int
    kind: str
    command: list[str]
    label: str
    status: str = "queued"  # queued | running | done | failed | cancelled
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    returncode: int | None = None
    log: deque = field(default_factory=lambda: deque(maxlen=400))
    process: subprocess.Popen | None = None

    def summary(self, log_lines: int = 40) -> dict:
        lines = list(self.log)
        return {
            "id": self.id, "kind": self.kind, "label": self.label, "status": self.status,
            "created": self.created, "started": self.started, "finished": self.finished,
            "returncode": self.returncode, "paused": bool(lines and lines[-1].startswith("paused:")),
            "log": lines[-log_lines:],
        }


def job_command(kind: str, args: dict) -> tuple[list[str], str]:
    """The command line for a job kind, validated (paths must lie inside the raw data folder)."""
    settings = rig.load()
    root = settings.data_raw.resolve()

    def inside(path: str) -> str:
        resolved = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if root not in (resolved, *resolved.parents):
            raise ValueError(f"{path} is not inside the data folder")
        if not resolved.exists():
            raise ValueError(f"{path} does not exist")
        return str(resolved)

    paths = [inside(p) for p in args.get("paths", [])]
    if not paths:
        raise ValueError("no sessions or episodes given")
    if kind == "process":
        command = [sys.executable, "-m", "bimanual_collect", "process", *paths]
        if args.get("force"):
            command.append("--force")
        depth = args.get("depth", "none")
        if depth not in ("none", "sgbm", "raft"):
            raise ValueError("depth must be none, sgbm or raft")
        if depth != "none":
            command += ["--depth", depth]
        label = f"process {len(paths)} item(s)" + (f" + {depth} depth" if depth != "none" else "")
    elif kind == "export":
        repo_id = str(args.get("repo_id", "")).strip()
        if not repo_id or "/" in repo_id or repo_id.startswith("."):
            raise ValueError("give the dataset a simple name (letters, digits, - or _)")
        python = str(settings.lerobot_python)
        if not Path(python).exists():
            raise ValueError(f"LeRobot environment not found: {python} (paths.lerobot_python in the rig config)")
        command = [python, "-m", "bimanual_collect", "export", *paths, "--repo-id", repo_id]
        for key, flag in (("image_size", "--image-size"), ("eyes", "--eyes"), ("crf", "--crf"), ("task", "--task")):
            if args.get(key) not in (None, ""):
                command += [flag, str(args[key])]
        if args.get("overwrite"):
            command.append("--overwrite")
        label = f"export {repo_id}"
    elif kind == "depth":
        method = args.get("method", "sgbm")
        if method not in ("sgbm", "raft"):
            raise ValueError("method must be sgbm or raft")
        python = sys.executable if method == "sgbm" else str(settings.depth_python)
        command = [python, "-m", "bimanual_collect", "depth", paths[0], "--method", method]
        label = f"{method} depth"
    else:
        raise ValueError(f"unknown job kind {kind!r}")
    return command, label


class JobRunner:
    def __init__(self) -> None:
        self.jobs: dict[int, Job] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="jobs", daemon=True)
        self._thread.start()

    def submit(self, kind: str, args: dict) -> dict:
        command, label = job_command(kind, args)
        with self._lock:
            job = Job(next(self._ids), kind, command, label)
            self.jobs[job.id] = job
        self._wake.set()
        return job.summary()

    def cancel(self, job_id: int) -> dict:
        with self._lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise ValueError(f"no job {job_id}")
        if job.status == "queued":
            job.status, job.finished = "cancelled", time.time()
        elif job.status == "running" and job.process is not None:
            job.process.terminate()
            job.status = "cancelled"
        return job.summary()

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j.id, reverse=True)
        return [job.summary(log_lines=12) for job in jobs[:30]]

    def get(self, job_id: int) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise ValueError(f"no job {job_id}")
        return job.summary(log_lines=400)

    def _next(self) -> Job | None:
        with self._lock:
            queued = [job for job in self.jobs.values() if job.status == "queued"]
        return min(queued, key=lambda j: j.id) if queued else None

    def _loop(self) -> None:
        while True:
            job = self._next()
            if job is None:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        job.status, job.started = "running", time.time()
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONUNBUFFERED"] = "1"
        try:
            job.process = subprocess.Popen(job.command, cwd=rig.ROOT, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1,
                                           preexec_fn=lambda: os.nice(10))
            for line in job.process.stdout:
                line = line.rstrip()
                if line and not line.startswith(("Svt[", "SVT [")):  # the AV1 encoder is chatty
                    job.log.append(line)
            job.returncode = job.process.wait()
        except OSError as exc:
            job.log.append(f"could not start: {exc}")
            job.returncode = -1
        job.finished = time.time()
        if job.status != "cancelled":
            job.status = "done" if job.returncode == 0 else "failed"
