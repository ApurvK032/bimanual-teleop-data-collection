"""Rig configuration: where data lives and how this bench is set up.

Loaded from the first file that exists: $BIMANUAL_RIG, configs/rig.yaml (your bench, git-ignored),
configs/rig.example.yaml (committed defaults). Relative paths resolve against the repository root.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (ROOT / "configs" / "rig.yaml", ROOT / "configs" / "rig.example.yaml")

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Rig:
    source: Path
    camera_serial: str | None
    camera_mode: str
    camera_fps: int
    camera_backend: str
    calibration_pattern: str
    data_raw: Path
    data_lerobot: Path
    state: Path
    teleop: Path
    teleop_runs: tuple[Path, ...]
    arm_profile: Path
    boxes_2k: dict[str, Box]
    depth_near_mm: float
    depth_far_mm: float
    export: dict
    qc: dict
    recording: dict
    tasks: tuple[str, ...]
    operator: str
    arms: int | None
    cameras: tuple[dict, ...]
    lerobot_python: Path
    depth_python: Path

    @property
    def uses_zed_sdk(self) -> bool:
        if not self.cameras:
            return self.camera_backend == "zed_sdk"
        return any(c.get("type", "zed") == "zed" and c.get("backend", self.camera_backend) == "zed_sdk"
                   for c in self.cameras)

    def calibration(self, serial: str | None = None) -> Path:
        """The camera's factory calibration file (python -m bimanual_collect calibration downloads it)."""
        if "{serial}" in self.calibration_pattern:
            serial = serial or self.camera_serial
            if serial is None:
                from .zed_v4l2 import zed_serial

                serial = zed_serial()
            return resolve(self.calibration_pattern.format(serial=serial))
        return resolve(self.calibration_pattern)

    def box_2k(self, name: str) -> Box:
        return self.boxes_2k[name]


def resolve(path: str | Path) -> Path:
    path = Path(os.path.expanduser(str(path)))
    return path if path.is_absolute() else ROOT / path


def config_path() -> Path:
    override = os.environ.get("BIMANUAL_RIG")
    if override:
        return resolve(override)
    for candidate in CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no rig config: create {CANDIDATES[0]} from {CANDIDATES[1].name}")


@cache
def load(path: str | None = None) -> Rig:
    source = resolve(path) if path else config_path()
    data = yaml.safe_load(source.read_text()) or {}
    camera, paths = data.get("camera", {}), data.get("paths", {})
    depth = data.get("depth", {})
    serial = camera.get("serial")
    return Rig(
        source=source,
        camera_serial=None if serial in (None, "") else str(serial),
        camera_mode=str(camera.get("mode", "1080")),
        camera_fps=int(camera.get("fps", 30)),
        camera_backend=str(camera.get("backend", "v4l2")),
        calibration_pattern=str(camera.get("calibration", "calibration/SN{serial}.conf")),
        data_raw=resolve(paths.get("data_raw", "data/raw")),
        data_lerobot=resolve(paths.get("data_lerobot", "data/lerobot")),
        state=resolve(paths.get("state", "state")),
        teleop=resolve(paths.get("teleop", "teleop")),
        teleop_runs=tuple(resolve(p) for p in paths.get("teleop_runs", ["teleop/runs"])),
        arm_profile=resolve(paths.get("arm_profile", "teleop/configs/dual_widowxai.yaml")),
        boxes_2k={name: tuple(int(v) for v in box) for name, box in (data.get("boxes_2k") or {}).items()},
        depth_near_mm=float(depth.get("near_mm", 900)),
        depth_far_mm=float(depth.get("far_mm", 1600)),
        export=dict(data.get("export") or {}),
        qc=dict(data.get("qc") or {}),
        recording=dict(data.get("recording") or {}),
        tasks=tuple(str(t) for t in (data.get("tasks") or [])),
        operator=str(data.get("operator") or ""),
        arms=None if data.get("arms") in (None, "", "auto") else int(data["arms"]),
        cameras=tuple(dict(c) for c in (data.get("cameras") or [])),
        lerobot_python=resolve(paths.get("lerobot_python", ".venv-lerobot/bin/python")),
        depth_python=resolve(paths.get("depth_python", ".venv-stereo/bin/python")),
    )
