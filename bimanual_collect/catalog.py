"""Episode bookkeeping: quality checks (QC) per episode, human review, and the dataset index.

Files per episode (next to episode.json):
  episode_qc.json   written by `process`: measured values, each check against its limit, verdict + reasons
  review.json       written by `review`: accepted / rejected, task label, note
And for the whole data folder: <data_raw>/index.csv, one row per recorded episode (rebuilt on every change).

QC limits come from the rig config's `qc:` section (defaults below). An episode that fails any check is
`needs_review`; nothing is exported unless it was accepted in review (or `export --all` is used).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from . import rig

DEFAULT_LIMITS = {
    "max_camera_drop_pct": 2.0,  # frames the camera error-flagged (the previous image is reused in the dataset)
    "max_torn_pct": 1.0,  # sideways-wrapped frames (frame_check)
    "max_sync_abs_ms": 15.0,  # camera-vs-arm offset from sync_check
    "min_sync_r": 0.6,  # correlation of image motion with arm motion at the best offset
    "min_duration_s": 5.0,
    "require_task": True,
}
INDEX_COLUMNS = ["episode", "task", "outcome", "operator", "status", "duration_s", "mode", "fps", "frames",
                 "camera_drop_pct", "torn_frames", "sync_ms", "sync_r", "steps", "qc", "review", "reasons"]


def limits() -> dict:
    return {**DEFAULT_LIMITS, **(rig.load().qc or {})}


def relative(path: Path) -> str:
    """Episode path relative to the raw data folder (portable across machines)."""
    try:
        return str(Path(path).resolve().relative_to(rig.load().data_raw.resolve()))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def review(episode: Path) -> dict | None:
    return read_json(Path(episode) / "review.json")


def task_of(episode: Path, meta: dict | None = None) -> str:
    """The task label: review > recording-time label > empty."""
    return str(label_of(episode, "task", meta) or "").strip()


def label_of(episode: Path, key: str, meta: dict | None = None):
    """A label (task / outcome / operator / note): review.json wins over the labels set while recording."""
    reviewed = review(episode) or {}
    if reviewed.get(key):
        return reviewed[key]
    meta = meta if meta is not None else read_json(Path(episode) / "episode.json") or {}
    return (meta.get("labels") or {}).get(key) or (meta.get(key) if key == "task" else None)


def camera_drop_pct(meta: dict) -> float:
    kept, missing = int(meta.get("frames_recorded") or 0), int(meta.get("camera_frames_missing") or 0)
    return round(100.0 * missing / max(1, kept + missing), 2)


def evaluate(episode: Path, meta: dict, torn: list[int], alignment: dict, sync: dict) -> dict:
    """QC record for one episode. alignment / sync: {"ok": bool, "error": str|None, ...measured values}."""
    lim = limits()
    frames = int(meta.get("frames_recorded", 0))
    task = task_of(episode, meta)
    status = str(meta.get("status", "complete"))
    reason = meta.get("stop_reason")
    checks = [
        ("recording", status if status == "complete" else f"{status}: {reason}", "complete",
         status == "complete"),
        ("stopped by", reason or "operator", "operator", reason in (None, "operator")),
        ("camera drops", f"{camera_drop_pct(meta)}%", f"<= {lim['max_camera_drop_pct']}%",
         camera_drop_pct(meta) <= lim["max_camera_drop_pct"]),
        *[(f"camera {cam['name']} drops", f"{camera_drop_pct(cam)}%", f"<= {lim['max_camera_drop_pct']}%",
           camera_drop_pct(cam) <= lim["max_camera_drop_pct"])
          for cam in meta.get("cameras", []) if not cam.get("primary")],
        ("torn frames", f"{round(100.0 * len(torn) / max(1, frames), 2)}%", f"<= {lim['max_torn_pct']}%",
         100.0 * len(torn) / max(1, frames) <= lim["max_torn_pct"]),
        ("recorder drops", int(meta.get("recorder_queue_drops") or 0), "0", int(meta.get("recorder_queue_drops") or 0) == 0),
        ("duration", f"{float(meta.get('duration_s', 0.0)):.1f} s", f">= {lim['min_duration_s']} s",
         float(meta.get("duration_s", 0.0)) >= lim["min_duration_s"]),
        ("alignment", alignment.get("error") or f"{alignment.get('steps')} steps", "the arm log covers it",
         bool(alignment.get("ok"))),
    ]
    if sync.get("ok"):
        shift, r = sync["shift_ms"], sync["r"]
        checks.append(("sync", f"{shift:+.1f} ms, r {r:.2f}",
                       f"|shift| <= {lim['max_sync_abs_ms']} ms, r >= {lim['min_sync_r']}",
                       abs(shift) <= lim["max_sync_abs_ms"] and r >= lim["min_sync_r"]))
    else:
        checks.append(("sync", sync.get("error") or "not run", "measured", False))
    if lim["require_task"]:
        checks.append(("task label", task or "(empty)", "not empty", bool(task)))
    reasons = [f"{name}: {value} (limit {limit})" for name, value, limit, ok in checks if not ok]
    return {
        "episode": relative(episode),
        "processed": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task": task,
        "outcome": label_of(episode, "outcome", meta),
        "status": status,
        "duration_s": meta.get("duration_s"),
        "mode": (meta.get("camera") or {}).get("mode"),
        "fps": (meta.get("video") or {}).get("nominal_fps"),
        "frames": frames,
        "camera_drop_pct": camera_drop_pct(meta),
        "torn_frames": torn,
        "alignment": alignment,
        "sync": sync,
        "checks": [{"name": n, "value": v, "limit": lim_, "ok": bool(ok)} for n, v, lim_, ok in checks],
        "verdict": "pass" if not reasons else "needs_review",
        "reasons": reasons,
    }


def all_episodes(root: Path | None = None) -> list[Path]:
    root = root or rig.load().data_raw
    return sorted(p.parent for p in root.glob("*/session_*/episode_*/episode.json"))


def index_row(episode: Path) -> dict:
    meta = read_json(episode / "episode.json") or {}
    qc = read_json(episode / "episode_qc.json") or {}
    reviewed = review(episode) or {}
    sync = qc.get("sync") or {}
    return {
        "episode": relative(episode),
        "task": task_of(episode, meta),
        "outcome": label_of(episode, "outcome", meta) or "",
        "operator": label_of(episode, "operator", meta) or "",
        "status": meta.get("status", "complete") if meta else "",
        "duration_s": meta.get("duration_s", ""),
        "mode": (meta.get("camera") or {}).get("mode", ""),
        "fps": (meta.get("video") or {}).get("nominal_fps", ""),
        "frames": meta.get("frames_recorded", ""),
        "camera_drop_pct": camera_drop_pct(meta) if meta else "",
        "torn_frames": len(qc["torn_frames"]) if "torn_frames" in qc else "",
        "sync_ms": sync.get("shift_ms", "") if sync.get("ok") else "",
        "sync_r": sync.get("r", "") if sync.get("ok") else "",
        "steps": (qc.get("alignment") or {}).get("steps", ""),
        "qc": qc.get("verdict", "not processed"),
        "review": reviewed.get("status", "unreviewed"),
        "reasons": "; ".join(qc.get("reasons", [])),
    }


def write_index(root: Path | None = None) -> Path:
    root = root or rig.load().data_raw
    root.mkdir(parents=True, exist_ok=True)  # a fresh checkout has no data folder yet
    path = root / "index.csv"
    rows = [index_row(episode) for episode in all_episodes(root)]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def is_accepted(episode: Path) -> bool:
    return (review(episode) or {}).get("status") == "accepted"
