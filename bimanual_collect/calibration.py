"""ZED factory calibration (Stereolabs .conf): per-mode intrinsics, and pixel boxes carried between modes.

Boxes in the rig config are written once in 2K left-eye pixels. HD1080 is a tighter native-scale crop of the same
sensor and HD720/VGA are binned wider views, so a box maps to another mode through the pinhole model:
u' = (u - cx_2k) * fx' / fx_2k + cx' (same for v), using the factory intrinsics of each mode.
"""
from __future__ import annotations

import configparser
from functools import cache
from pathlib import Path

from . import rig

SECTION = {2208: "2K", 1920: "FHD", 1280: "HD", 672: "VGA"}  # per-eye width -> calibration section
EYE_HEIGHT = {2208: 1242, 1920: 1080, 1280: 720, 672: 376}


@cache
def _read(calib: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    if not config.read(calib):
        raise FileNotFoundError(f"calibration file not found: {calib} (python -m bimanual_collect calibration downloads it)")
    return config


def intrinsics(calib: Path, eye_width: int, eye: str = "LEFT") -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) of one eye in the mode with this per-eye width."""
    section = _read(str(calib))[f"{eye}_CAM_{SECTION[eye_width]}"]
    return float(section["fx"]), float(section["fy"]), float(section["cx"]), float(section["cy"])


def map_box_from_2k(box: rig.Box, calib: Path, eye_width: int) -> rig.Box:
    """A left-eye box in 2K pixels -> the same scene area in the mode with this per-eye width (clipped)."""
    if eye_width == 2208:
        return tuple(int(v) for v in box)
    fx0, fy0, cx0, cy0 = intrinsics(calib, 2208)
    fx, fy, cx, cy = intrinsics(calib, eye_width)
    x0, y0, x1, y1 = box
    width, height = eye_width, EYE_HEIGHT[eye_width]

    def u(x: float) -> int:
        return int(round(min(max((x - cx0) * fx / fx0 + cx, 0), width)))

    def v(y: float) -> int:
        return int(round(min(max((y - cy0) * fy / fy0 + cy, 0), height)))

    return u(x0), v(y0), u(x1), v(y1)


def episode_calibration(meta: dict) -> Path:
    """The factory calibration recorded with an episode, else the rig's file for that camera serial."""
    camera = meta.get("camera") or {}
    recorded = camera.get("factory_calibration")
    if recorded:
        for candidate in (rig.resolve(recorded), rig.ROOT / "calibration" / Path(recorded).name):
            if candidate.exists():
                return candidate
    return rig.load().calibration(camera.get("serial"))


def episode_box(name: str, meta: dict) -> rig.Box:
    """A rig box (2K pixels) mapped to an episode's camera mode."""
    eye_width = int(meta["video"]["size"][0]) // 2
    return map_box_from_2k(rig.load().box_2k(name), episode_calibration(meta), eye_width)


def stereo_rectification(calib: Path, eye_width: int, height: int):
    """Rectification of the ZED pair for one mode: (kl, dl, kr, dr, r1, r2, p1, p2, baseline_mm).

    Convention (checked by row alignment, 0 px median error): R = Rodrigues([RX, CV, RZ]), T = (-Baseline, TY, TZ).
    2K and HD1080 use OpenCV's alpha=0 (valid pixels only, f ~1051 px). HD720 and VGA are binned views wider than
    the field the factory lens model was fitted on; there OpenCV's valid-pixel zoom extrapolates the distortion
    polynomial past its range and zooms in ~1.6x (wrong depth), so those keep the left camera's own focal length
    and principal point instead.
    """
    import cv2
    import numpy as np

    config = _read(str(calib))
    section = SECTION[eye_width]

    def camera(name: str):
        c = config[name]
        k = np.array([[float(c["fx"]), 0, float(c["cx"])], [0, float(c["fy"]), float(c["cy"])], [0, 0, 1]])
        # Column vectors: OpenCV 5 rejects flat (5,) / (3,) arrays in stereoRectify; 4.x accepted them.
        return k, np.array([float(c[key]) for key in ("k1", "k2", "p1", "p2", "k3")]).reshape(5, 1)

    kl, dl = camera(f"LEFT_CAM_{section}")
    kr, dr = camera(f"RIGHT_CAM_{section}")
    stereo = config["STEREO"]
    baseline = float(stereo["Baseline"])
    rotation = cv2.Rodrigues(np.array([float(stereo[f"RX_{section}"]), float(stereo[f"CV_{section}"]),
                                       float(stereo[f"RZ_{section}"])]))[0]
    translation = np.array([-baseline, float(stereo["TY"]), float(stereo["TZ"])]).reshape(3, 1)
    r1, r2, p1, p2, *_ = cv2.stereoRectify(kl, dl, kr, dr, (eye_width, height), rotation, translation,
                                           flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    if eye_width in (1280, 672):
        f, cx, cy = kl[0, 0], kl[0, 2], kl[1, 2]
        p1 = np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]], dtype=np.float64)
        p2 = p1.copy()
        p2[0, 3] = -f * baseline
    return kl, dl, kr, dr, r1, r2, p1, p2, baseline


def map_box_to_2k(box: rig.Box, calib: Path, eye_width: int) -> rig.Box:
    """The inverse of map_box_from_2k: a left-eye box in this mode's pixels -> 2K pixels (for the rig config)."""
    if eye_width == 2208:
        return tuple(int(v) for v in box)
    fx0, fy0, cx0, cy0 = intrinsics(calib, 2208)
    fx, fy, cx, cy = intrinsics(calib, eye_width)
    x0, y0, x1, y1 = box
    return (int(round((x0 - cx) * fx0 / fx + cx0)), int(round((y0 - cy) * fy0 / fy + cy0)),
            int(round((x1 - cx) * fx0 / fx + cx0)), int(round((y1 - cy) * fy0 / fy + cy0)))


CALIBRATION_URL = "https://calib.stereolabs.com/?SN={serial}"


def fetch_calibration(serial: str, out_dir: Path | None = None) -> Path:
    """Download a ZED's factory calibration (public, by serial) to calibration/SN<serial>.conf."""
    import urllib.request

    if not str(serial).isdigit():
        raise ValueError(f"not a ZED serial number: {serial!r}")
    with urllib.request.urlopen(CALIBRATION_URL.format(serial=serial), timeout=30) as response:
        text = response.read().decode()
    config = configparser.ConfigParser()
    config.read_string(text)
    if "STEREO" not in config or "LEFT_CAM_2K" not in config:
        raise ValueError("the download does not look like a ZED calibration file")
    out_dir = out_dir or rig.ROOT / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"SN{serial}.conf"
    out.write_text(text)
    return out


def main() -> None:
    """python -m bimanual_collect calibration [--serial N]: download each connected ZED's factory calibration."""
    import argparse

    from .zed_v4l2 import video_devices

    parser = argparse.ArgumentParser(description="Download ZED factory calibration files from Stereolabs, by serial")
    parser.add_argument("--serial", help="camera serial (default: every connected ZED)")
    args = parser.parse_args()
    serials = [args.serial] if args.serial else [d.serial for d in video_devices() if d.is_zed and d.serial]
    if not serials:
        raise SystemExit("no ZED found; connect the camera or pass --serial")
    for serial in serials:
        print(f"wrote {fetch_calibration(serial).relative_to(rig.ROOT)}")
