"""Read side of a recorded episode: which cameras it holds, and where each camera's files are.

Two layouts exist:
  format 2 (current):        episode.json lists "cameras"; files under cameras/<name>/{video.mkv, frames.csv}
  format 1 (before):         one ZED; zed_sbs.mkv + zed_frames.csv in the episode folder
Every tool goes through this module, so both layouts keep working.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CameraRecord:
    name: str
    video: Path
    frames_csv: Path
    width: int  # full frame (both eyes for a stereo camera)
    height: int
    fps: float  # nominal
    stereo: bool
    primary: bool
    pixel_format: str = "yuyv"
    info: dict = field(default_factory=dict)  # model, serial, mode, device, calibration, counters

    @property
    def eye_width(self) -> int:
        return self.width // 2 if self.stereo else self.width

    def rows(self) -> list[dict]:
        with open(self.frames_csv, newline="") as handle:
            return list(csv.DictReader(handle))


def read_meta(episode: Path) -> dict:
    return json.loads((Path(episode) / "episode.json").read_text())


def cameras(episode: Path, meta: dict | None = None) -> list[CameraRecord]:
    episode = Path(episode)
    meta = meta if meta is not None else read_meta(episode)
    if meta.get("cameras"):
        records = []
        for cam in meta["cameras"]:
            width, height = cam["size"]
            records.append(CameraRecord(
                name=cam["name"], video=episode / cam["video"], frames_csv=episode / cam["frames"],
                width=int(width), height=int(height), fps=float(cam["nominal_fps"]), stereo=bool(cam["stereo"]),
                primary=bool(cam.get("primary")), pixel_format=cam.get("pixel_format", "yuyv"), info=cam,
            ))
        if records and not any(r.primary for r in records):
            first = records[0]
            records[0] = CameraRecord(**{**first.__dict__, "primary": True})
        return records
    video = meta["video"]
    width, height = video["size"]
    info = {**(meta.get("camera") or {}), "frames_recorded": meta.get("frames_recorded"),
            "camera_frames_missing": meta.get("camera_frames_missing"),
            "recorder_queue_drops": meta.get("recorder_queue_drops")}
    return [CameraRecord(name="zed", video=episode / video.get("file", "zed_sbs.mkv"),
                         frames_csv=episode / "zed_frames.csv", width=int(width), height=int(height),
                         fps=float(video["nominal_fps"]), stereo=True, primary=True, info=info)]


def primary(episode: Path, meta: dict | None = None) -> CameraRecord:
    return next(record for record in cameras(episode, meta) if record.primary)


def camera(episode: Path, name: str, meta: dict | None = None) -> CameraRecord:
    for record in cameras(episode, meta):
        if record.name == name:
            return record
    raise KeyError(f"{episode}: no camera named {name!r}")


def status(meta: dict) -> str:
    """complete | aborted | recovered | recording (format 1 episodes were always written at a clean stop)."""
    return str(meta.get("status", "complete"))
