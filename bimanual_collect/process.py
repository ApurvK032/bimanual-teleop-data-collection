"""After a session: check every episode, give it a QC verdict, and update the index and the session report.

    python -m bimanual_collect process data/raw/<date>/<session>            # every episode of the session
    python -m bimanual_collect process data/raw/<date>/<session> --depth sgbm
    python -m bimanual_collect process data/raw/2026-*/session_*  --force   # redo episodes already processed

Per episode: torn-frame check -> alignment with the arm log (no write) -> camera-vs-arm sync check -> optional
per-frame depth -> QC verdict (catalog.py). Writes <episode>/episode_qc.json, <episode>/thumbs.jpg,
<session>/report.md and <data_raw>/index.csv. Heavy work pauses while a recording or an arm run is active.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from . import catalog, episode_io, rig
from .frame_check import ensure as ensure_frame_check
from .frame_check import wait_while_busy
from .lerobot_align import AlignmentError, align_episode, episode_dirs
from .recovery import recover_episode
from .sync_check import SyncError
from .sync_check import check as sync_check
from .zed_format import EyeDecoder, decode_command


def frame_at(video: Path, index: int, fps: float, width: int, height: int) -> np.ndarray | None:
    """One decoded YUYV frame by index (FFV1 is intra-only, so seeking is cheap)."""
    command = decode_command(video)
    command[3:3] = ["-ss", f"{index / fps:.4f}"]  # before -i: fast seek
    command[-1:-1] = ["-frames:v", "1"]
    data = subprocess.run(command, capture_output=True).stdout
    size = width * height * 2
    return np.frombuffer(data[:size], np.uint8).reshape(height, width, 2) if len(data) >= size else None


def thumbnails(episode: Path, meta: dict, out: Path, count: int = 3, width: int = 480) -> Path | None:
    """Left-eye natural-colour stills at 10% / 50% / 90% of the episode, side by side."""
    cam = episode_io.primary(episode, meta)
    w, h, fps = cam.width, cam.height, cam.fps
    frames = int(meta.get("frames_recorded") or len(cam.rows()))
    decoder, tiles = EyeDecoder(colour_fix=True), []
    for fraction in np.linspace(0.1, 0.9, count):
        yuyv = frame_at(cam.video, int(fraction * max(frames - 1, 0)), fps, w, h)
        if yuyv is None:
            continue
        left = decoder.bgr(yuyv, 0, cam.eye_width)
        tiles.append(cv2.resize(left, (width, round(width * h / cam.eye_width)), interpolation=cv2.INTER_AREA))
    if not tiles:
        return None
    cv2.imwrite(str(out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def preview(episode: Path, meta: dict, out: Path, width: int = 640) -> Path | None:
    """A small H.264 clip of the primary camera (left eye) for the console; viewing only, camera colour."""
    cam = episode_io.primary(episode, meta)
    crop = "crop=iw/2:ih:0:0," if cam.stereo else ""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(cam.video), "-vf",
         f"scale=in_range=pc:out_range=pc,{crop}scale={width}:-2,format=yuv420p", "-c:v", "libx264", "-preset",
         "veryfast", "-crf", "28", "-movflags", "+faststart", "-an", str(out)], capture_output=True)
    return out if result.returncode == 0 and out.exists() else None


def run_depth(episode: Path, method: str) -> dict:
    python = sys.executable if method == "sgbm" else str(rig.load().depth_python)
    result = subprocess.run([python, "-m", "bimanual_collect.episode_depth", str(episode), "--method", method],
                            cwd=rig.ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        return {"method": method, "ok": False, "error": (result.stderr or result.stdout).strip().splitlines()[-1:]}
    return {"method": method, "ok": True, "folder": "rgbd" if method == "sgbm" else "rgbd_raft"}


def process_episode(episode: Path, depth: str = "none", sync_video: bool = False) -> dict:
    wait_while_busy()
    meta = json.loads((episode / "episode.json").read_text())
    if meta.get("status") == "recording":  # left open by a crash (nothing is recording now: we waited above)
        recover_episode(episode)
        meta = json.loads((episode / "episode.json").read_text())
    torn = ensure_frame_check(episode)
    try:
        steps = align_episode(episode)
        report = steps.report
        alignment = {"ok": True, "error": None, "steps": report["steps"], "fps": report["fps"],
                     "telemetry": str(Path(report["telemetry"]).parent.name),
                     "max_state_sample_gap_ms": report["max_state_sample_gap_ms"],
                     "max_action_age_ms": report["max_action_age_ms"]}
    except (AlignmentError, ValueError, KeyError, OSError) as exc:
        alignment = {"ok": False, "error": str(exc).replace(str(episode), "episode").replace(f"{rig.ROOT}/", "")}
    if alignment["ok"]:
        try:
            result = sync_check(episode, video=sync_video)
            sync = {"ok": True, "error": None, "shift_ms": result["best_shift_ms"], "r": result["correlation"],
                    "block_spread_ms": result["block_shift_spread_ms"], "verdict": result["verdict"]}
        except (SyncError, ValueError, OSError) as exc:
            sync = {"ok": False, "error": str(exc)}
    else:
        sync = {"ok": False, "error": "skipped (no arm log aligned with this episode)"}
    record = catalog.evaluate(episode, meta, torn, alignment, sync)
    if depth != "none":
        wait_while_busy()
        record["depth"] = run_depth(episode, depth)
    thumbs = thumbnails(episode, meta, episode / "thumbs.jpg")
    record["thumbnails"] = thumbs.name if thumbs else None
    clip = preview(episode, meta, episode / "preview.mp4")
    record["preview"] = clip.name if clip else None
    (episode / "episode_qc.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def session_report(session: Path) -> Path:
    """Markdown summary of a session's episodes (QC, sync, review), with thumbnails and sync plots."""
    episodes = sorted(p for p in session.glob("episode_*") if (p / "episode.json").exists())
    lines = [f"# {session.parent.name} / {session.name}", "",
             "| Episode | Task | Length | Mode | Camera drops | Sync | QC | Review |",
             "|---|---|---|---|---|---|---|---|"]
    details = []
    for episode in episodes:
        row = catalog.index_row(episode)
        sync = f"{row['sync_ms']:+.1f} ms (r {row['sync_r']:.2f})" if row["sync_ms"] != "" else "-"
        lines.append(f"| {episode.name} | {row['task'] or '-'} | {row['duration_s']} s | {row['mode']} @ {row['fps']} "
                     f"| {row['camera_drop_pct']}% | {sync} | {row['qc']} | {row['review']} |")
        details += ["", f"## {episode.name}", ""]
        if row["reasons"]:
            details += [f"- {reason}" for reason in row["reasons"].split("; ")]
            details.append("")
        if (episode / "thumbs.jpg").exists():
            details.append(f"![{episode.name}]({episode.name}/thumbs.jpg)")
        if (episode / "sync" / "sync_plot.png").exists():
            details.append(f"![sync]({episode.name}/sync/sync_plot.png)")
    path = session / "report.md"
    path.write_text("\n".join(lines + details) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="session or episode folders")
    parser.add_argument("--depth", choices=["none", "sgbm", "raft"], default="none",
                        help="also compute per-frame depth (raft needs .venv-stereo and a GPU)")
    parser.add_argument("--sync-video", action="store_true", help="also write the sync preview video")
    parser.add_argument("--force", action="store_true", help="redo episodes that already have episode_qc.json")
    args = parser.parse_args()

    episodes = episode_dirs(args.paths)
    for episode in episodes:
        if (episode / "episode_qc.json").exists() and not args.force:
            print(f"{catalog.relative(episode)}: already processed (--force to redo)")
            continue
        record = process_episode(episode, args.depth, args.sync_video)
        print(f"{record['episode']}: {record['verdict']}" + "".join(f"\n    {r}" for r in record["reasons"]),
              flush=True)
    for session in sorted({episode.parent for episode in episodes}):
        print(f"report -> {session_report(session)}")
    print(f"index  -> {catalog.write_index()}")


if __name__ == "__main__":
    main()
