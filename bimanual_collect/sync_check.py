"""Check that a recorded ZED episode and the teleop arm log line up in time (one-arm or two-arm runs).

    python -m bimanual_collect sync data/raw/<date>/<session>/episode_0000
    ... --telemetry <run>/telemetry.csv   default: the teleop run that overlaps the episode (as lerobot_align)
    ... --roi x0,y0,x1,y1                 left-eye pixels around the arm(s); default: the rig's sync_roi box
    ... --video                           also a preview MP4: ZED left eye + arm traces with a moving cursor

Image motion = mean absolute brightness change between consecutive left-eye frames inside the ROI (quarter
scale). Arm motion = summed |change| of the measured joints between the same two exposure times (fresh encoder
reads only, linearly interpolated). The time shift that best correlates the two is the camera-vs-arm offset:
about 0 ms means each frame's exposure timestamp and the arm log agree. It is also measured on 4 time blocks to
show the spread. Keep the operator out of the ROI: their motion leads the arm's and would bias the result.
Writes <episode>/sync/: sync_report.json, sync_plot.png (+ sync_preview.mp4). Reads the episode and telemetry only.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import episode_io
from .calibration import episode_box
from .frame_check import ensure as ensure_frame_check
from .frame_check import wait_while_busy
from .lerobot_align import find_telemetry
from .zed_format import EyeDecoder, iter_frames

csv.field_size_limit(10_000_000)
SCALE = 4  # image motion is measured at quarter scale (averages sensor noise, fast)
MAX_SHIFT_MS = 400
MARGIN_NS = MAX_SHIFT_MS * 1_000_000
JOINT_COLOURS = [(60, 60, 220), (40, 160, 40), (200, 120, 30), (30, 170, 220), (180, 60, 180), (120, 120, 120)]


def _int(text) -> int:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return 0


def load_arms(csv_path: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """Fresh encoder reads per arm, {side: (t_ns (M,), values (M, 7) = 6 joints + gripper)}, and run stats.

    One-arm telemetry has columns q_feedback, ...; two-arm telemetry has left_q_feedback, right_q_feedback, ...
    On control ticks without a new encoder read the launchers repeat the last values with a read time of 0 and
    feedback_sample_fresh = False. Those rows are skipped, so each sample keeps its true read time.
    """
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "q_feedback" in fields:
            prefixes = {"arm": ""}
        else:
            prefixes = {side: f"{side}_" for side in ("left", "right") if f"{side}_q_feedback" in fields}
        if not prefixes:
            raise SystemExit(f"{csv_path}: no q_feedback columns, not teleop telemetry")
        samples: dict[str, dict[int, list[float]]] = {side: {} for side in prefixes}
        worst_error: dict[str, list[float]] = {side: [] for side in prefixes}
        ticks, quest_sequence, ik_failures = [], [], 0
        for row in reader:
            now = _int(row.get("pc_monotonic_ns"))
            if now <= 0:
                continue
            ticks.append(now)
            if row.get("quest_sequence"):
                quest_sequence.append(_int(row["quest_sequence"]))
            for side, prefix in prefixes.items():
                if row.get(f"{prefix}ik_status", "") not in ("", "ok"):
                    ik_failures += 1
                if (row.get(f"{prefix}feedback_sample_fresh") or "").strip().lower() not in ("", "true", "1"):
                    continue
                try:
                    q = json.loads(row[f"{prefix}q_feedback"])
                    grip = float(row[f"{prefix}gripper_feedback_m"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not isinstance(q, list) or len(q) != 6:
                    continue
                samples[side][_int(row.get(f"{prefix}feedback_read_monotonic_ns")) or now] = [*q, grip]
                try:
                    worst_error[side].append(max(abs(v) for v in json.loads(row[f"{prefix}q_feedback_error"])))
                except (KeyError, TypeError, ValueError):
                    pass
    if not ticks:
        raise SystemExit(f"{csv_path}: no telemetry rows")
    arms, per_arm = {}, {}
    for side, by_time in samples.items():
        if len(by_time) < 2:
            raise SystemExit(f"{csv_path}: fewer than 2 fresh encoder reads for {side}")
        t = np.array(sorted(by_time), dtype=np.int64)
        arms[side] = (t, np.array([by_time[k] for k in t]))
        errors = np.array(worst_error[side]) if worst_error[side] else np.array([np.nan])
        per_arm[side] = {
            "fresh_encoder_reads": int(t.size),
            "encoder_read_rate_hz": round((t.size - 1) / ((t[-1] - t[0]) / 1e9), 2),
            # None when the run logged no tracking errors (e.g. simulation)
            "worst_joint_tracking_error_rad_p95": round(float(np.nanpercentile(errors, 95)), 4)
            if np.isfinite(errors).any() else None,
            "worst_joint_tracking_error_rad_max": round(float(np.nanmax(errors)), 4) if np.isfinite(errors).any() else None,
            "gripper_opening_mm_min_max": [round(float(arms[side][1][:, 6].min() * 1e3), 2),
                                           round(float(arms[side][1][:, 6].max() * 1e3), 2)],
        }
    tick = np.array(ticks, dtype=np.int64)
    period_ms = np.diff(tick) / 1e6
    steps = np.diff(np.array(quest_sequence)) if len(quest_sequence) > 1 else np.array([1])
    stats = {
        "rows": int(tick.size),
        "duration_s": round((tick[-1] - tick[0]) / 1e9, 3),
        "loop_rate_hz": round((tick.size - 1) / ((tick[-1] - tick[0]) / 1e9), 3),
        "loop_period_ms_p50_p99_max": [round(float(np.percentile(period_ms, p)), 2) for p in (50, 99)]
        + [round(float(period_ms.max()), 2)],
        "ik_failures": ik_failures,
        "quest_sequence_skipped": int(np.clip(steps - 1, 0, None).sum()),
        "quest_sequence_repeated": int((steps == 0).sum()),
        "arms": per_arm,
        "first_monotonic_ns": int(tick[0]),
        "last_monotonic_ns": int(tick[-1]),
    }
    return arms, stats


def arm_motion(arms: dict, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Summed |joint change| (rad, 6 joints per arm) between times t0 and t1 (encoder reads interpolated)."""
    total = np.zeros(t0.shape, dtype=np.float64)
    for t, values in arms.values():
        for joint in range(6):
            total += np.abs(np.interp(t1, t, values[:, joint]) - np.interp(t0, t, values[:, joint]))
    return total


def correlation_curve(motion, arms, exposure, pairs, shifts_ns) -> np.ndarray:
    t0, t1 = exposure[pairs], exposure[pairs + 1]
    image = motion[pairs]
    curve = np.full(len(shifts_ns), np.nan)
    if image.std() == 0:
        return curve
    for i, shift in enumerate(shifts_ns):
        arm = arm_motion(arms, t0 + shift, t1 + shift)
        if arm.std() > 0:
            curve[i] = np.corrcoef(image, arm)[0, 1]
    return curve


def peak(shifts_ms: np.ndarray, curve: np.ndarray) -> tuple[float, float]:
    """Best shift (ms, parabola-refined) and its correlation."""
    if np.all(np.isnan(curve)):
        return float("nan"), float("nan")
    k = int(np.nanargmax(curve))
    best = float(shifts_ms[k])
    if 0 < k < len(curve) - 1 and not np.isnan(curve[k - 1 : k + 2]).any():
        a, b, c = curve[k - 1 : k + 2]
        denominator = a - 2 * b + c
        if denominator < 0:
            best += 0.5 * (a - c) / denominator * float(shifts_ms[1] - shifts_ms[0])
    return round(best, 1), round(float(curve[k]), 3)


def draw_trace(canvas, xs, ys, colour, thickness=2) -> None:
    keep = np.isfinite(xs) & np.isfinite(ys)
    if keep.sum() >= 2:
        points = np.stack([xs[keep], ys[keep]], 1).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], False, colour, thickness, cv2.LINE_AA)


def text(canvas, message, origin, scale=0.55, colour=(40, 40, 40), thickness=1) -> None:
    cv2.putText(canvas, message, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def unit(series: np.ndarray) -> np.ndarray:
    top = np.nanpercentile(series, 99) if np.isfinite(series).any() else 0
    return series / top if top > 0 else series * 0


def sync_plot(path, shifts_ms, curve, best, r_best, mid_s, image_n, arm_n, motion_map, roi_small, verdict) -> None:
    """One PNG: correlation vs shift, the ROI on the motion map, and both motion traces over time."""
    canvas = np.full((800, 1280, 3), 255, np.uint8)
    # correlation vs shift
    x0, y0, w, h = 70, 60, 640, 300
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (200, 200, 200), 1)
    lo = min(0.0, float(np.nanmin(curve))) if np.isfinite(curve).any() else 0.0
    sx = lambda ms: x0 + (ms + MAX_SHIFT_MS) / (2 * MAX_SHIFT_MS) * w  # noqa: E731
    sy = lambda r: y0 + h - (r - lo) / (1.0 - lo) * h  # noqa: E731
    cv2.line(canvas, (int(sx(0)), y0), (int(sx(0)), y0 + h), (200, 200, 200), 1)
    for ms in range(-MAX_SHIFT_MS, MAX_SHIFT_MS + 1, 100):
        text(canvas, f"{ms:+d}" if ms else "0", (int(sx(ms)) - 16, y0 + h + 20), 0.45)
    draw_trace(canvas, sx(shifts_ms), sy(curve), (200, 110, 20), 2)
    if np.isfinite(best):
        cv2.circle(canvas, (int(sx(best)), int(sy(r_best))), 6, (30, 30, 200), -1, cv2.LINE_AA)
    text(canvas, "Match between image motion and arm motion vs time shift", (x0, 30), 0.6, thickness=2)
    text(canvas, "shift (ms): arm log time minus camera exposure time", (x0 + 150, y0 + h + 42), 0.45)
    text(canvas, f"best shift {best:+.1f} ms, correlation {r_best:.2f}", (x0 + 12, y0 + 26), 0.6, (30, 30, 200), 2)
    # motion map with the ROI
    mm = cv2.applyColorMap(np.clip(motion_map * 25, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    a, b, c, d = roi_small
    cv2.rectangle(mm, (a, b), (c, d), (80, 220, 80), 2)
    mm = cv2.resize(mm, (500, int(500 * mm.shape[0] / mm.shape[1])))
    canvas[y0 : y0 + mm.shape[0], 750 : 750 + mm.shape[1]] = mm
    text(canvas, "Where the image changed (green box = arm area used)", (750, 30), 0.55, thickness=2)
    # traces over time
    x0, y0, w, h = 70, 440, 1160, 300
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (200, 200, 200), 1)
    span = max(float(mid_s[-1]), 1e-6)
    tx = x0 + mid_s / span * w
    draw_trace(canvas, tx, y0 + h - np.clip(image_n, 0, 1.2) / 1.2 * h, (200, 110, 20), 2)
    draw_trace(canvas, tx, y0 + h - np.clip(arm_n, 0, 1.2) / 1.2 * h, (30, 140, 240), 2)
    for s in range(0, int(span) + 1, 5):
        text(canvas, f"{s}s", (int(x0 + s / span * w) - 8, y0 + h + 20), 0.45)
    text(canvas, "Image change in the arm area (blue)  vs  arm joint motion from the encoders (orange), "
         "at the best shift", (x0, y0 - 14), 0.55, thickness=2)
    text(canvas, verdict, (x0, 790), 0.6, (30, 30, 200) if "NOT" in verdict else (30, 130, 30), 2)
    cv2.imwrite(str(path), canvas)


def preview_video(path, episode, size, rows, exposure, arms, motion, roi, telemetry_span, torn=frozenset(),
                  fps: float = 15) -> None:
    """ZED left eye (half size, natural colour, viewing only) above the arm traces, cursor at each exposure."""
    width, height = size
    eye_w, view_w, view_h, strip_h, px_per_s = width // 2, width // 4, height // 2, 299, 160
    t_start = exposure[0]
    pad = view_w // 2
    span_px = int((exposure[-1] - t_start) / 1e9 * px_per_s) + 2 * pad + 2
    strip = np.full((strip_h, span_px, 3), 255, np.uint8)
    xs = lambda t: pad + (np.asarray(t, np.float64) - t_start) / 1e9 * px_per_s  # noqa: E731
    bands = {"joints": (8, 150), "gripper": (160, 210), "motion": (220, 290)}
    for top, bottom in bands.values():
        cv2.rectangle(strip, (0, top), (span_px - 1, bottom), (225, 225, 225), 1)
    for t, values in arms.values():
        centred = values[:, :6] - np.median(values[:, :6], axis=0)
        reach = max(float(np.percentile(np.abs(centred), 99.5)), 1e-3)
        top, bottom = bands["joints"]
        for joint in range(6):
            ys = (top + bottom) / 2 - np.clip(centred[:, joint] / reach, -1, 1) * (bottom - top) / 2
            draw_trace(strip, xs(t), ys, JOINT_COLOURS[joint], 2)
        top, bottom = bands["gripper"]
        draw_trace(strip, xs(t), bottom - np.clip(values[:, 6] / 0.04, 0, 1) * (bottom - top), (40, 40, 40), 2)
    mid = (exposure[:-1] + exposure[1:]) / 2
    top, bottom = bands["motion"]
    draw_trace(strip, xs(mid), bottom - np.clip(unit(motion), 0, 1.2) / 1.2 * (bottom - top), (200, 110, 20), 2)
    arm = arm_motion(arms, exposure[:-1], exposure[1:])
    arm[(exposure[:-1] < telemetry_span[0]) | (exposure[1:] > telemetry_span[1])] = np.nan
    draw_trace(strip, xs(mid), bottom - np.clip(unit(arm), 0, 1.2) / 1.2 * (bottom - top), (30, 140, 240), 2)
    labels = [("arm joints (encoders, each centred)", 24), ("gripper opening 0-40 mm", 176),
              ("image change in arm area (blue) / arm motion (orange)", 236)]
    out_h = view_h + strip_h
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{view_w}x{out_h}",
               "-r", f"{fps:g}", "-i", "pipe:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    decoder = EyeDecoder(colour_fix=True)
    x0, y0, x1, y1 = (v // 2 for v in roi)
    for index, yuyv in enumerate(iter_frames(episode_io.primary(episode).video, width, height)):
        if index >= len(rows):
            break
        view = cv2.resize(decoder.bgr(yuyv, 0, eye_w), (view_w, view_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(view, (x0, y0), (x1, y1), (80, 220, 80), 2)
        covered = telemetry_span[0] <= exposure[index] <= telemetry_span[1]
        caption = (f"t {(exposure[index] - t_start) / 1e9:6.2f} s   frame {index}   camera seq "
                   f"{rows[index]['camera_sequence']}" + ("" if covered else "   (no arm log here)")
                   + ("   TORN FRAME (skipped)" if index in torn else ""))
        cv2.rectangle(view, (0, 0), (view_w, 34), (0, 0, 0), -1)
        text(view, caption, (10, 24), 0.65, (255, 255, 255), 1)
        centre = int(round(xs(exposure[index])))
        window = strip[:, centre - pad : centre - pad + view_w].copy()
        cv2.line(window, (pad, 0), (pad, strip_h), (0, 0, 220), 2)
        for label, y in labels:
            text(window, label, (8, y), 0.5, (90, 90, 90))
        encoder.stdin.write(np.vstack([view, window]).tobytes())
    encoder.stdin.close()
    encoder.wait()


class SyncError(ValueError):
    """The episode cannot be checked (no overlapping arm log, too little motion data)."""


def check(episode: Path, telemetry: Path | None = None, roi: tuple[int, int, int, int] | None = None,
          video: bool = False) -> dict:
    """Run the timing check on one episode; writes <episode>/sync/ and returns the report."""
    episode = Path(episode)
    meta = json.loads((episode / "episode.json").read_text())
    cam = episode_io.primary(episode, meta)
    size = [cam.width, cam.height]
    telemetry_path = telemetry or find_telemetry(meta)
    arms, stats = load_arms(telemetry_path)
    rows = cam.rows()
    exposure = np.array([int(r["exposure_monotonic_ns"]) for r in rows], dtype=np.int64)
    sequence = np.array([int(r["camera_sequence"]) for r in rows], dtype=np.int64)
    roi = tuple(roi) if roi else episode_box("sync_roi", meta)
    a, b, c, d = (v // SCALE for v in roi)
    torn = set(ensure_frame_check(episode))  # wrapped frames: skipped, like frames the camera dropped

    # image motion inside the ROI, plus a whole-frame motion map to show where the ROI sits
    small_size = (cam.eye_width // SCALE, size[1] // SCALE)
    previous, motion, motion_sum = None, [], np.zeros(small_size[::-1], np.float64)
    for index, yuyv in enumerate(iter_frames(cam.video, *size)):
        if index >= len(rows):
            break
        if index % 30 == 0:
            wait_while_busy()
        luma = cv2.resize(np.ascontiguousarray(yuyv[:, : size[0] // 2, 0]), small_size, interpolation=cv2.INTER_AREA)
        luma = luma.astype(np.int16)
        if previous is not None:
            if index in torn or index - 1 in torn:
                motion.append(float("nan"))
            else:
                change = np.abs(luma - previous)
                motion.append(float(change[b:d, a:c].mean()))
                motion_sum += change
        previous = luma
    motion = np.array(motion)
    frames = len(motion) + 1
    exposure, sequence, rows = exposure[:frames], sequence[:frames], rows[:frames]

    span = (max(t[0] for t, _ in arms.values()), min(t[-1] for t, _ in arms.values()))
    pairs = np.flatnonzero(
        (np.diff(sequence) == 1) & (exposure[:-1] - MARGIN_NS >= span[0]) & (exposure[1:] + MARGIN_NS <= span[1])
        & np.isfinite(motion)
    )
    if pairs.size < 30:
        raise SyncError(f"only {pairs.size} frame pairs overlap the arm log; nothing to compare")
    shifts_ms = np.arange(-MAX_SHIFT_MS, MAX_SHIFT_MS + 1, 1.0)
    shifts_ns = (shifts_ms * 1e6).astype(np.int64)
    curve = correlation_curve(motion, arms, exposure, pairs, shifts_ns)
    best, r_best = peak(shifts_ms, curve)
    blocks = []
    for block in np.array_split(pairs, 4):
        shift, r = peak(shifts_ms, correlation_curve(motion, arms, exposure, block, shifts_ns))
        blocks.append({"from_s": round((exposure[block[0]] - exposure[0]) / 1e9, 1),
                       "to_s": round((exposure[block[-1]] - exposure[0]) / 1e9, 1), "shift_ms": shift, "r": r})
    good_blocks = [blk["shift_ms"] for blk in blocks if np.isfinite(blk["r"]) and blk["r"] >= 0.5]
    spread = round(max(good_blocks) - min(good_blocks), 1) if len(good_blocks) >= 2 else None
    period_ms = float(np.median(np.diff(exposure)) / 1e6)
    in_sync = bool(np.isfinite(best) and abs(best) <= period_ms / 3 and r_best >= 0.5
                   and (spread is None or spread <= period_ms / 2))
    verdict = (f"IN SYNC: camera and arm log agree to {best:+.1f} ms (one camera frame = {period_ms:.0f} ms)"
               if in_sync else f"NOT CONFIRMED IN SYNC: best shift {best:+.1f} ms, correlation {r_best:.2f}")

    arrival = np.array([int(r["arrival_monotonic_ns"]) for r in rows], dtype=np.int64)
    report = {
        "episode": str(episode),
        "telemetry": str(telemetry_path),
        "verdict": verdict,
        "in_sync": in_sync,
        "best_shift_ms": best,
        "best_shift_meaning": "arm log time minus camera exposure-start time at the best match "
                              "(about +half the exposure time is expected)",
        "correlation": r_best,
        "correlation_at_0_ms": round(float(curve[MAX_SHIFT_MS]), 3),
        "blocks": blocks,
        "block_shift_spread_ms": spread,
        "frame_pairs_compared": int(pairs.size),
        "roi_left_eye_px": list(roi),
        "torn_frames_skipped": sorted(torn),
        "camera": {
            "frames": frames,
            "camera_sequence_gaps": int((np.diff(sequence) - 1).sum()),
            "fps": round((frames - 1) / ((exposure[-1] - exposure[0]) / 1e9), 3),
            "exposure_to_arrival_ms_p50": round(float(np.median(arrival - exposure) / 1e6), 1),
            "recorder_queue_drops": meta.get("recorder_queue_drops"),
        },
        "arm_log_within_episode_s": [round((stats["first_monotonic_ns"] - exposure[0]) / 1e9, 2),
                                     round((stats["last_monotonic_ns"] - exposure[0]) / 1e9, 2)],
        "teleop": {k: v for k, v in stats.items() if not k.endswith("monotonic_ns")},
    }
    out = episode / "sync"
    out.mkdir(exist_ok=True)
    (out / "sync_report.json").write_text(json.dumps(report, indent=2) + "\n")
    mid_s = ((exposure[pairs] + exposure[pairs + 1]) / 2 - exposure[0]) / 1e9
    arm_best = arm_motion(arms, exposure[pairs] + int(best * 1e6), exposure[pairs + 1] + int(best * 1e6))
    sync_plot(out / "sync_plot.png", shifts_ms, curve, best, r_best, mid_s, unit(motion[pairs]), unit(arm_best),
              motion_sum / max(frames - 1, 1), (a, b, c, d), verdict)
    if video:
        preview_video(out / "sync_preview.mp4", episode, size, rows, exposure, arms, motion, roi, span, torn,
                      float(meta["video"].get("nominal_fps", 15)))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episode", type=Path)
    parser.add_argument("--telemetry", type=Path, help="teleop telemetry.csv (default: found by time)")
    parser.add_argument("--roi", help="x0,y0,x1,y1 in left-eye pixels (default: the rig's sync_roi box, mapped to the mode)")
    parser.add_argument("--video", action="store_true", help="also write sync_preview.mp4")
    args = parser.parse_args()
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    try:
        report = check(args.episode, args.telemetry, roi, args.video)
    except SyncError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps({k: report[k] for k in ("verdict", "best_shift_ms", "correlation", "block_shift_spread_ms",
                                             "camera", "teleop")}, indent=2))
    print(f"wrote {args.episode / 'sync'}")


if __name__ == "__main__":
    main()
