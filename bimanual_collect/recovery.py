"""Repair episodes left "recording" by a crash (power loss, killed process): remux each camera's video, trim
frames.csv to the frames that decode, fill in the stop time, and mark the episode "recovered".

The recorder runs this on start-up; `process` runs it for any such episode it meets (never on a live recording).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path


def write_json(path: Path, data: dict) -> None:
    """Atomic: a crash never leaves a half-written episode.json."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _count_packets(video: Path) -> int | None:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                             "stream=nb_read_packets", "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def recover_episode(directory: Path) -> dict | None:
    """Repair an episode left "recording" by a crash: remux each video, trim frames.csv to what decodes."""
    directory = Path(directory)
    path = directory / "episode.json"
    meta = json.loads(path.read_text())
    if meta.get("status") != "recording":
        return None
    last_rows = []
    for cam in meta.get("cameras", []):
        if cam.get("svo2"):
            cam["svo2"]["status"] = "unverified"
        video, frames_csv = directory / cam["video"], directory / cam["frames"]
        fixed = video.with_name("video.recovered.mkv")
        if video.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-c", "copy", str(fixed)],
                           capture_output=True)
            if fixed.exists() and fixed.stat().st_size > 0:
                os.replace(fixed, video)
        packets = _count_packets(video) if video.exists() else 0
        rows = []
        if frames_csv.exists():
            with open(frames_csv, newline="") as handle:
                reader = csv.DictReader(handle)
                rows = [row for row in reader if row.get("arrival_epoch_ns")]
        rows = rows[: packets or 0]
        with open(frames_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["video_frame", "camera_sequence", "exposure_monotonic_ns",
                                                        "arrival_monotonic_ns", "arrival_epoch_ns"])
            writer.writeheader()
            writer.writerows(rows)
        sequences = [int(row["camera_sequence"]) for row in rows]
        expected = sequences[-1] - sequences[0] + 1 if sequences else 0
        cam.update({"frames_recorded": len(rows), "camera_frames_missing": max(0, expected - len(rows)),
                    "camera_frames_error_dropped": None, "recorder_queue_drops": None, "encoder_exit_code": None})
        if cam.get("primary") and rows:
            last_rows = rows
    if last_rows:
        meta["stopped_monotonic_ns"] = int(last_rows[-1]["arrival_monotonic_ns"])
        meta["stopped_epoch_ns"] = int(last_rows[-1]["arrival_epoch_ns"])
        meta["duration_s"] = round((meta["stopped_monotonic_ns"] - meta["started_monotonic_ns"]) / 1e9, 3)
    primary = next((cam for cam in meta.get("cameras", []) if cam.get("primary")), None)
    if primary is not None:
        for key in ("frames_recorded", "camera_frames_missing", "camera_frames_error_dropped", "recorder_queue_drops",
                    "encoder_exit_code"):
            meta[key] = primary.get(key)
    meta["status"] = "recovered"
    meta["stop_reason"] = "the recorder stopped unexpectedly; repaired on the next start"
    write_json(path, meta)
    return {"directory": str(directory), "frames": meta.get("frames_recorded", 0)}


def recover_unfinished(data_root: Path) -> list[dict]:
    recovered = []
    for path in sorted(Path(data_root).glob("*/session_*/episode_*/episode.json")):
        try:
            if json.loads(path.read_text()).get("status") == "recording":
                result = recover_episode(path.parent)
                if result:
                    recovered.append(result)
        except (OSError, ValueError, KeyError) as exc:
            print(f"could not recover {path.parent}: {exc}", flush=True)
    return recovered
