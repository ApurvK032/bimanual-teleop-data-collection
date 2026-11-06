"""Find torn (sideways-wrapped) frames in a recorded ZED episode, and keep heavy processing off the live session.

    python -m bimanual_collect check-frames data/raw/<date>/<session>/episode_NNNN

Torn frames: the camera can deliver a frame assembled from shifted data. It has the full size and no error flag,
but the picture is wrapped sideways (seen while a CPU-heavy job ran on the PC during recording). The camera is
fixed and most of the view is static, so phase correlation of each frame's left-eye luma (quarter size) against the
last good frame gives ~0 px shift (p95 0.04 px); a torn frame gives tens to hundreds of px. A shift that persists
for several frames is taken as the view really moving (reported, not dropped).
Writes <episode>/frame_check.json. lerobot_align treats the listed frames as camera-dropped (previous image repeated),
sync_check skips them, episode_depth skips them, export_lerobot runs this check first.

Guard: wait_while_busy() pauses the calling tool while a recording or an arm launcher is active, because a
CPU-heavy job during recording raised torn and error-flagged frames from ~0.3% to ~10% in testing.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from . import episode_io
from .activity import HARDWARE_LAUNCHERS, running_launchers
from .zed_format import iter_frames  # noqa: E402

RESULT = "frame_check.json"
SCALE = 4
THRESHOLD_PX = 16.0  # full-resolution px
PERSIST_FRAMES = 4


def busy() -> str | None:
    """What heavy processing should wait for right now, or None."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8444/record/status", timeout=0.5) as response:
            if json.load(response).get("recording"):
                return "a ZED recording"
    except (OSError, ValueError):
        pass
    if running_launchers(HARDWARE_LAUNCHERS):
        return "an arm run"
    return None


def wait_while_busy(poll_s: float = 2.0) -> None:
    reason = busy()
    if reason is None:
        return
    print(f"paused: {reason} is active (heavy processing can cost camera frames); resuming when it ends", flush=True)
    while busy() is not None:
        time.sleep(poll_s)
    print("resumed", flush=True)


def _shift_px(a: np.ndarray, b: np.ndarray, window: np.ndarray) -> float:
    (dx, dy), _ = cv2.phaseCorrelate(a, b, window)
    return float(np.hypot(dx, dy)) * SCALE


def detect(episode: Path) -> dict:
    cam = episode_io.primary(episode)
    width, height = cam.width, cam.height
    size = (cam.eye_width // SCALE, height // SCALE)
    window = cv2.createHanningWindow(size, cv2.CV_32F)
    ref, ref_index, pending, torn, moved, shifts, frames = None, -1, [], [], [], [], 0
    for index, yuyv in enumerate(iter_frames(cam.video, width, height)):
        if index % 30 == 0:
            wait_while_busy()
        frames += 1
        small = cv2.resize(np.ascontiguousarray(yuyv[:, : width // 2, 0]), size,
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        if ref is None:
            ref, ref_index = small, index
            continue
        shift = _shift_px(ref, small, window)
        if shift <= THRESHOLD_PX:
            torn.extend(i for i, _, _ in pending)
            pending = []
            shifts.append(shift)
            ref, ref_index = small, index
            continue
        pending.append((index, small, shift))
        if len(pending) >= PERSIST_FRAMES and all(
                _shift_px(pending[0][1], s, window) <= THRESHOLD_PX for _, s, _ in pending[1:]):
            if ref_index == 0 and pending[0][0] == 1:
                torn.append(0)  # the first frame was the odd one out
            else:
                moved.append({"video_frame": pending[0][0], "shift_px": round(pending[0][2], 1)})
            ref, ref_index, pending = pending[-1][1], pending[-1][0], []
    torn.extend(i for i, _, _ in pending)
    torn = sorted(set(torn))
    return {
        "frames": frames,
        "torn_video_frames": torn,
        "view_moved_at": moved,
        "good_frame_shift_px_p95": round(float(np.percentile(shifts, 95)), 3) if shifts else None,
        "method": f"phase correlation of left-eye luma at 1/{SCALE} size vs the last good frame; torn if the shift "
                  f"exceeds {THRESHOLD_PX:g} px and does not persist {PERSIST_FRAMES}+ frames",
    }


def ensure(episode: Path) -> list[int]:
    """Torn video frames of an episode, computed once and cached in <episode>/frame_check.json."""
    path = Path(episode) / RESULT
    if not path.exists():
        path.write_text(json.dumps(detect(Path(episode)), indent=2) + "\n")
    return list(json.loads(path.read_text())["torn_video_frames"])


def torn_frames(episode: Path) -> list[int] | None:
    """Cached result only (no decoding); None if the check has not run."""
    path = Path(episode) / RESULT
    return list(json.loads(path.read_text())["torn_video_frames"]) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--redo", action="store_true", help="recompute even if frame_check.json exists")
    args = parser.parse_args()
    for episode in args.episodes:
        if args.redo and (episode / RESULT).exists():
            (episode / RESULT).unlink()
        ensure(episode)
        result = json.loads((episode / RESULT).read_text())
        print(f"{episode}: {len(result['torn_video_frames'])} torn of {result['frames']} "
              f"{result['torn_video_frames'][:20]}; view moved at {result['view_moved_at']}; "
              f"good-frame shift p95 {result['good_frame_shift_px_p95']} px")


if __name__ == "__main__":
    main()
