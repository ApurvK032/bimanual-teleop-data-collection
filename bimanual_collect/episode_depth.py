"""Per-frame depth for a recorded ZED episode, from its lossless stereo pair (no ZED SDK).

    python -m bimanual_collect depth data/raw/<date>/<session>/episode_0000
    ... --every 15            every 15th frame only (quick look)
    ... --frames 0,100,200    just these video frames
    ... --threads 16          after teleop is over (default 8, and the process runs at nice 10)
    .venv-stereo/bin/python -m bimanual_collect depth <episode> --method raft   # GPU (torch), denser

Both methods rectify both eyes with the factory calibration (convention verified by row alignment, 0 px row
error) and work one frame at a time (nothing averaged: the cloth moves).
  sgbm  OpenCV SGBM 3-way, block 5, on the luma. Only what both eyes clearly match: a few cm of noise, holes on dark,
        plain or one-eye-only areas. -> <episode>/rgbd/
  raft  torchvision's RAFT (optical-flow network, official C_T_SKHT_V2 weights) on the rectified colour pair at half
        size; disparity = horizontal flow. Dense: it also fills plain, dark and one-eye-only areas from context, so
        a left-right check marks which pixels both directions agree on (confident/). -> <episode>/rgbd_raft/
Writes:
  depth/<video_frame>.png      uint16 depth in mm, aligned to the RECTIFIED left eye; 0 = no depth
  rgb/<video_frame>.jpg        rectified left eye, natural colour (q95), pixel-aligned with depth
  confident/<video_frame>.png  raft only: 255 = left->right and right->left agree within 1.5 px, 0 = filled/uncertain
  frames.csv                   video frame, camera sequence, exposure (CLOCK_MONOTONIC ns), depth fraction, median mm
  intrinsics.json              rectified pinhole (fx fy cx cy), baseline, size, units, method
  preview.mp4                  half-size rgb | colour depth, for viewing only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from . import episode_io, rig
from .calibration import episode_calibration, stereo_rectification
from .frame_check import ensure as ensure_frame_check
from .frame_check import wait_while_busy
from .zed_format import EyeDecoder, iter_frames

# Disparity search window per eye width: 2K covers ~0.66-3.9 m (table ~1.47 m = 86 px, arms ~100 px).
DISPARITY = {2208: (32, 160), 1920: (32, 160), 1280: (16, 96), 672: (8, 48)}
BLOCK = 5


class StereoDepth:
    """Rectification + SGBM for one eye size, with the convention verified by row alignment (0 px row error)."""

    def __init__(self, eye_width: int, height: int, calib: Path, serial: str = "") -> None:
        self.calib, self.serial = Path(calib), serial
        kl, dl, kr, dr, r1, r2, p1, p2, self.baseline_mm = stereo_rectification(self.calib, eye_width, height)
        size = (eye_width, height)
        self.left_map = cv2.initUndistortRectifyMap(kl, dl, r1, p1, size, cv2.CV_16SC2)
        self.right_map = cv2.initUndistortRectifyMap(kr, dr, r2, p2, size, cv2.CV_16SC2)
        self.f, self.cx, self.cy = float(p1[0, 0]), float(p1[0, 2]), float(p1[1, 2])
        self.min_disp, self.num_disp = DISPARITY[eye_width]
        self.size = size
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disp, numDisparities=self.num_disp, blockSize=BLOCK, P1=8 * BLOCK * BLOCK,
            P2=32 * BLOCK * BLOCK, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=200, speckleRange=2,
            preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def rectify_left(self, image: np.ndarray) -> np.ndarray:
        return cv2.remap(image, *self.left_map, cv2.INTER_LINEAR)

    def rectify_right(self, image: np.ndarray) -> np.ndarray:
        return cv2.remap(image, *self.right_map, cv2.INTER_LINEAR)

    def depth_mm(self, left_luma: np.ndarray, right_luma: np.ndarray) -> np.ndarray:
        """SGBM. uint16 mm on the rectified left grid; 0 where nothing was measured."""
        rl = cv2.remap(left_luma, *self.left_map, cv2.INTER_LINEAR)
        rr = cv2.remap(right_luma, *self.right_map, cv2.INTER_LINEAR)
        disparity = self.matcher.compute(rl, rr).astype(np.float32) / 16.0
        valid = disparity >= self.min_disp
        valid[:, : self.min_disp + self.num_disp] = False  # no partner in the right image
        depth = np.where(valid, self.f * self.baseline_mm / np.maximum(disparity, 1e-3), 0)
        return np.clip(np.round(depth), 0, 65535).astype(np.uint16)

    def depth_from_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """Dense disparity (px, rectified left grid) -> uint16 mm; 0 outside the search range or the right image."""
        x = np.arange(disparity.shape[1], dtype=np.float32)[None, :]
        valid = ((disparity >= self.min_disp) & (disparity <= self.min_disp + self.num_disp)
                 & (x - disparity >= 0))
        depth = np.where(valid, self.f * self.baseline_mm / np.maximum(disparity, 1e-3), 0)
        return np.clip(np.round(depth), 0, 65535).astype(np.uint16)

    def intrinsics(self, method: str) -> dict:
        return {
            "frame": f"rectified left eye of the ZED 2i {self.serial}".strip() + " (depth/ and rgb/ share this pixel grid)",
            "width": self.size[0], "height": self.size[1], "fx": self.f, "fy": self.f, "cx": self.cx, "cy": self.cy,
            "baseline_mm": self.baseline_mm, "depth_units": "mm, uint16; 0 = no depth",
            "method": method, "calibration": self.calib.name,
        }


class RaftMatcher:
    """Dense disparity from torchvision's RAFT (optical flow; official weights) on the rectified colour pair."""

    def __init__(self, scale: float = 0.5, iters: int = 20) -> None:
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        self.torch = torch
        self.weights = Raft_Large_Weights.C_T_SKHT_V2
        self.model = raft_large(weights=self.weights, progress=False).eval().to("cuda")
        self.scale, self.iters = scale, iters

    def describe(self) -> str:
        return (f"torchvision RAFT-large {self.weights.name} on the rectified colour pair at scale {self.scale}, "
                f"{self.iters} updates; disparity = -horizontal flow; confident = left->right and right->left agree "
                "within 1.5 px and vertical flow <= 1.5 px; per frame, no temporal averaging")

    def disparity(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(disparity px on the full-size rectified left grid, confident mask)."""
        torch = self.torch
        h, w = left_rgb.shape[:2]
        sw, sh = int(round(w * self.scale)), int(round(h * self.scale))
        pad_w, pad_h = (8 - sw % 8) % 8, (8 - sh % 8) % 8  # RAFT needs sizes divisible by 8

        def prep(image):
            small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
            small = cv2.copyMakeBorder(small, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            return torch.from_numpy(np.ascontiguousarray(small)).permute(2, 0, 1)

        a, b = prep(left_rgb), prep(right_rgb)
        first, second = self.weights.transforms()(torch.stack([a, b]), torch.stack([b, a]))  # both directions
        with torch.inference_mode():
            flow = self.model(first.cuda(), second.cuda(), num_flow_updates=self.iters)[-1]
        flow = flow[:, :, :sh, :sw].float().cpu().numpy()

        def full(field):
            return cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR) / self.scale

        d_left, vertical, d_right = -full(flow[0, 0]), full(flow[0, 1]), full(flow[1, 0])
        xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        ys = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
        d_back = cv2.remap(d_right, xs - d_left, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=-1e4)
        confident = (np.abs(d_left - d_back) <= 1.5) & (np.abs(vertical) <= 1.5)
        return d_left.astype(np.float32), confident


def colour_depth(depth: np.ndarray, near_mm: float, far_mm: float) -> np.ndarray:
    scaled = np.clip((far_mm - depth.astype(np.float32)) / (far_mm - near_mm), 0, 1)  # near = warm
    image = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[depth == 0] = 0
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episode", type=Path)
    parser.add_argument("--method", choices=["sgbm", "raft"], default="sgbm")
    parser.add_argument("--every", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--frames", help="comma-separated video frames instead of --every")
    parser.add_argument("--threads", type=int, default=8, help="OpenCV threads (keep low while teleoperating)")
    parser.add_argument("--raft-scale", type=float, default=0.5, help="raft: image scale fed to the network")
    parser.add_argument("--raft-iters", type=int, default=20, help="raft: refinement updates")
    parser.add_argument("--near-mm", type=float, default=rig.load().depth_near_mm, help="preview colour range")
    parser.add_argument("--far-mm", type=float, default=rig.load().depth_far_mm)
    parser.add_argument("--no-rgb", action="store_true", help="skip rgb/ (depth only)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    os.nice(10)
    cv2.setNumThreads(args.threads)
    meta = json.loads((args.episode / "episode.json").read_text())
    cam = episode_io.primary(args.episode, meta)
    if not cam.stereo:
        raise SystemExit(f"{args.episode}: the primary camera {cam.name!r} is not a stereo camera")
    width, height = cam.width, cam.height
    eye = cam.eye_width
    rows = cam.rows()
    wanted = ({int(v) for v in args.frames.split(",")} if args.frames else set(range(0, len(rows), args.every)))
    torn = set(ensure_frame_check(args.episode))  # sideways-wrapped frames get no depth (frame_check.py)
    wanted -= torn
    out = args.out or args.episode / ("rgbd" if args.method == "sgbm" else "rgbd_raft")
    (out / "depth").mkdir(parents=True, exist_ok=True)
    if not args.no_rgb:
        (out / "rgb").mkdir(exist_ok=True)
    stereo = StereoDepth(eye, height, episode_calibration(meta), str(meta.get("camera", {}).get("serial", "")))
    raft = None
    if args.method == "raft":
        raft = RaftMatcher(args.raft_scale, args.raft_iters)
        (out / "confident").mkdir(exist_ok=True)
        method = raft.describe()
    else:
        method = (f"OpenCV SGBM 3-way, block {BLOCK}, disparities {stereo.min_disp}-{stereo.min_disp + stereo.num_disp}"
                  " px, per frame (no temporal averaging)")
    (out / "intrinsics.json").write_text(json.dumps(stereo.intrinsics(method), indent=2) + "\n")
    decoder = EyeDecoder(colour_fix=True)

    view_w, view_h = eye // 2 // 2 * 2, height // 2 // 2 * 2  # H.264 needs even sizes (2K: 1104x620)
    preview_fps = max(1, round(float(meta["video"]["nominal_fps"]) / max(1, args.every)))
    preview = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2 * view_w}x{view_h}",
         "-r", str(preview_fps), "-i", "pipe:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "preview.mp4")],
        stdin=subprocess.PIPE)
    started, done = time.perf_counter(), 0
    band = stereo.min_disp + stereo.num_disp
    with open(out / "frames.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["video_frame", "camera_sequence", "exposure_monotonic_ns", "measured_fraction", "median_mm"]
        writer.writerow(header + (["confident_fraction"] if raft else []))
        for index, yuyv in enumerate(iter_frames(cam.video, width, height)):
            if index >= len(rows) or index > max(wanted):
                break
            if index not in wanted:
                continue
            wait_while_busy()
            rgb = stereo.rectify_left(decoder.bgr(yuyv, 0, eye))
            extra = []
            if raft is None:
                luma = yuyv[:, :, 0]
                depth = stereo.depth_mm(np.ascontiguousarray(luma[:, :eye]), np.ascontiguousarray(luma[:, eye:]))
            else:
                right = stereo.rectify_right(decoder.bgr(yuyv, eye, eye))
                disparity, confident = raft.disparity(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                                                      cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
                depth = stereo.depth_from_disparity(disparity)
                confident &= depth > 0
                cv2.imwrite(str(out / "confident" / f"{index:05d}.png"), confident.astype(np.uint8) * 255)
                extra = [round(float(confident[:, band:].mean()), 4)]
            cv2.imwrite(str(out / "depth" / f"{index:05d}.png"), depth)
            if not args.no_rgb:
                cv2.imwrite(str(out / "rgb" / f"{index:05d}.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            measured = depth[:, band:] > 0  # same region for both methods (SGBM has no partner left of `band`)
            writer.writerow([index, rows[index]["camera_sequence"], rows[index]["exposure_monotonic_ns"],
                             round(float(measured.mean()), 4),
                             int(np.median(depth[depth > 0])) if measured.any() else 0] + extra)
            pair = np.hstack([cv2.resize(rgb, (view_w, view_h), interpolation=cv2.INTER_AREA),
                              cv2.resize(colour_depth(depth, args.near_mm, args.far_mm), (view_w, view_h),
                                         interpolation=cv2.INTER_NEAREST)])
            cv2.putText(pair, f"frame {index}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                        cv2.LINE_AA)
            preview.stdin.write(pair.tobytes())
            done += 1
    preview.stdin.close()
    preview.wait()
    seconds = time.perf_counter() - started
    print(f"wrote {done} {args.method} depth frames to {out} in {seconds:.1f} s "
          f"({seconds / max(done, 1):.2f} s/frame incl. decoding); torn frames skipped: {sorted(torn)}")


if __name__ == "__main__":
    main()
