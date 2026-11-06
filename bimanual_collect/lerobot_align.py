"""Line up one recorded episode with the teleop telemetry (one or two arms), one dataset step per camera frame.

numpy only (no LeRobot needed), so it runs in the main venv and can be inspected before exporting:

    python -m bimanual_collect align data/raw/<date>/<session>/episode_0000

Conventions (all times CLOCK_MONOTONIC: V4L2 exposure timestamps and the teleop's time.perf_counter_ns()):
  - step k <-> camera sequence first+k at the camera's nominal rate (episode.json). A frame the camera dropped
    (error-flagged buffer) keeps the previous image and is marked `image_repeated`.
  - observation.state = per arm 6 measured joints [rad] + measured gripper [m], linearly interpolated to the
    frame's start-of-exposure time.
  - action = per arm the most recent commanded 6 joints [rad] + commanded gripper [m] at that time (the target
    the teleop had just sent the arm).
  - order: left_joint_0..5, left_gripper, right_joint_0..5, right_gripper (one arm: joint_0..5, gripper).
    Left/right are the operator's sides
    (controller hands); which arm that is depends on the bench, see the run's config_snapshot.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import episode_io, rig

DEFAULT_RUNS_ROOTS = rig.load().teleop_runs
SIDES = ("left", "right")
JOINTS = [*(f"joint_{i}" for i in range(6)), "gripper"]


def arm_layout(fieldnames) -> list[tuple[str, str]]:
    """[(side, column prefix)] from a telemetry header: two arms (left_/right_ columns) or one (no prefix)."""
    names = set(fieldnames or [])
    if {"left_q_feedback", "right_q_feedback"} <= names:
        return [("left", "left_"), ("right", "right_")]
    if "q_feedback" in names:
        return [("arm", "")]
    return []


def state_names(sides) -> list[str]:
    """left_joint_0..5, left_gripper, right_... for two arms; joint_0..5, gripper for one."""
    sides = list(sides)
    return list(JOINTS) if sides == ["arm"] else [f"{side}_{name}" for side in sides for name in JOINTS]


STATE_NAMES = state_names(SIDES)  # two arms
MAX_SAMPLE_GAP_NS = 100_000_000  # a frame further than this from arm data is not covered
EDGE_NS = 50_000_000  # frames before the first / after the last arm sample by more than this are trimmed
csv.field_size_limit(10_000_000)


class AlignmentError(RuntimeError):
    pass


@dataclass
class ArmSeries:
    feedback_t: np.ndarray  # (M,) ns
    feedback: np.ndarray  # (M, 7) 6 joints + gripper, measured
    command_t: np.ndarray  # (K,) ns
    command: np.ndarray  # (K, 7) 6 joints + gripper, commanded


@dataclass
class Telemetry:
    path: Path
    rows: int
    first_monotonic_ns: int
    last_monotonic_ns: int
    epoch_minus_monotonic_ns: int
    arms: dict[str, ArmSeries]  # in layout order: {"left", "right"} or {"arm"}

    @property
    def sides(self) -> list[str]:
        return list(self.arms)


def _int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _vector(value: str, length: int) -> list[float] | None:
    if not value:
        return None
    try:
        values = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(values, list) or len(values) != length:
        return None
    return [float(item) for item in values]


def _float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def load_telemetry(csv_path: Path) -> Telemetry:
    """Read a teleop telemetry.csv, one or two arms (vectors are JSON lists in cells)."""
    rows = 0
    first = last = offset = None
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        layout = arm_layout(reader.fieldnames)
        if not layout:
            raise AlignmentError(f"{csv_path} has no arm joint columns (q_feedback / left_q_feedback)")
        feedback = {side: {} for side, _ in layout}  # time -> 7 values (dedupes repeated reads)
        commands = {side: {} for side, _ in layout}
        for row in reader:
            now = _int(row.get("pc_monotonic_ns", ""))
            if now <= 0:
                continue
            rows += 1
            first = now if first is None else first
            last = now
            if offset is None and _int(row.get("pc_epoch_ns", "")) > 0:
                offset = _int(row["pc_epoch_ns"]) - now
            for side, prefix in layout:
                q = _vector(row.get(f"{prefix}q_feedback", ""), 6)
                grip = _float(row.get(f"{prefix}gripper_feedback_m", ""))
                # Ticks without a new encoder read repeat the last values with read time 0 and fresh = False
                # (run_dual_hardware.py / run_hardware.py). Keep only real reads, each at its own read time.
                fresh = (row.get(f"{prefix}feedback_sample_fresh") or "").strip().lower()
                if q is not None and grip is not None and fresh in ("", "true", "1"):
                    t = _int(row.get(f"{prefix}feedback_read_monotonic_ns", "")) or now
                    feedback[side][t] = [*q, grip]
                q = _vector(row.get(f"{prefix}q_cmd", ""), 6)
                grip = _float(row.get(f"{prefix}gripper_cmd_m", ""))
                if q is not None and grip is not None:
                    t = _int(row.get(f"{prefix}command_send_monotonic_ns", "")) or now
                    commands[side][t] = [*q, grip]
    if not rows:
        raise AlignmentError(f"{csv_path} has no telemetry rows")
    arms = {}
    for side, _ in layout:
        if not feedback[side] or not commands[side]:
            raise AlignmentError(f"{csv_path}: no {side}-arm joint data")
        ft = np.array(sorted(feedback[side]), dtype=np.int64)
        ct = np.array(sorted(commands[side]), dtype=np.int64)
        arms[side] = ArmSeries(
            ft, np.array([feedback[side][t] for t in ft]), ct, np.array([commands[side][t] for t in ct])
        )
    return Telemetry(Path(csv_path), rows, first, last, offset or 0, arms)


def _edge_rows(csv_path: Path) -> tuple[dict, dict] | None:
    """First and last data rows of a telemetry.csv without reading the whole file."""
    with open(csv_path, newline="", encoding="utf-8") as handle:
        header = handle.readline()
        first_line = handle.readline()
        if not first_line:
            return None
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 1_000_000))
        tail = handle.read().splitlines()
    names = next(csv.reader([header]))
    first = dict(zip(names, next(csv.reader([first_line]))))
    for line in reversed(tail):
        try:
            values = next(csv.reader([line]))
        except StopIteration:
            continue
        if len(values) == len(names):
            return first, dict(zip(names, values))
    return None


def find_telemetry(episode: dict, roots=DEFAULT_RUNS_ROOTS, arms: int | None = None) -> Path:
    """The teleop run whose telemetry covers this episode best (same boot, largest time overlap).

    Only runs with arm joint data count (CAD-leader logs don't); with `arms`, only runs with that many arms.
    """
    start, stop = episode["started_monotonic_ns"], episode["stopped_monotonic_ns"]
    episode_offset = episode["started_epoch_ns"] - start
    best, best_overlap = None, 0
    for root in roots:
        for path in Path(root).rglob("telemetry.csv"):
            edges = _edge_rows(path)
            if edges is None:
                continue
            layout = arm_layout(edges[0].keys())
            if not layout or (arms is not None and len(layout) != arms):
                continue
            first, last = (_int(edge.get("pc_monotonic_ns", "")) for edge in edges)
            offset = _int(edges[0].get("pc_epoch_ns", "")) - first
            if not first or not last or abs(offset - episode_offset) > 1_000_000_000:
                continue  # different boot (CLOCK_MONOTONIC restarts) or no timestamps
            overlap = min(stop, last) - max(start, first)
            if overlap > best_overlap:
                best, best_overlap = path, overlap
    if best is None:
        raise AlignmentError(
            f"no teleop telemetry overlaps episode {episode.get('episode_index')} "
            f"(searched {', '.join(str(r) for r in roots)}); pass --telemetry"
        )
    return best


@dataclass
class CameraSteps:
    """Another camera's frames matched to the primary camera's steps."""

    name: str
    video_frame: np.ndarray  # (S,) decoded frame shown at each step
    repeated: np.ndarray  # (S,) bool: no frame of this camera close to the step, the previous one is shown
    exposure_monotonic_ns: np.ndarray  # (S,)
    offset_ms: np.ndarray  # (S,) frame exposure minus step time


@dataclass
class Steps:
    episode_dir: Path
    telemetry_path: Path
    fps: int
    video_frame: np.ndarray  # (S,) which decoded video frame each step shows
    image_repeated: np.ndarray  # (S,) bool: camera dropped this frame, previous image reused
    camera_sequence: np.ndarray  # (S,)
    exposure_monotonic_ns: np.ndarray  # (S,) actual, or interpolated for repeated steps
    state: np.ndarray  # (S, 7 x arms) float32
    action: np.ndarray  # (S, 7 x arms) float32
    report: dict = field(default_factory=dict)
    names: list[str] = field(default_factory=lambda: list(STATE_NAMES))
    primary: str = "zed"
    cameras: dict[str, CameraSteps] = field(default_factory=dict)  # the other cameras


def _interpolate(times: np.ndarray, values: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t, times, values[:, column]) for column in range(values.shape[1])], axis=1)


def _hold(times: np.ndarray, values: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = np.searchsorted(times, t, side="right") - 1
    valid = index >= 0
    index = np.clip(index, 0, len(times) - 1)
    return values[index], np.where(valid, t - times[index], np.iinfo(np.int64).max)


def _nearest_gap(times: np.ndarray, t: np.ndarray) -> np.ndarray:
    index = np.clip(np.searchsorted(times, t), 1, len(times) - 1)
    return np.minimum(np.abs(t - times[index - 1]), np.abs(times[index] - t))


def match_camera(name: str, rows: list[dict], fps: float, step_times: np.ndarray) -> CameraSteps:
    """For each step, the frame whose exposure is nearest; if none lies within half a frame period (camera dropped
    it, or a slower camera), the latest earlier frame is shown and the step is flagged as repeated."""
    exposure = np.array([int(r["exposure_monotonic_ns"]) for r in rows], dtype=np.int64)
    video = np.array([int(r["video_frame"]) for r in rows], dtype=np.int64)
    order = np.argsort(exposure)
    exposure, video = exposure[order], video[order]
    nearest = np.clip(np.searchsorted(exposure, step_times), 1, len(exposure) - 1) if len(exposure) > 1 \
        else np.zeros(len(step_times), dtype=np.int64)
    if len(exposure) > 1:
        left = nearest - 1
        nearest = np.where(np.abs(exposure[left] - step_times) <= np.abs(exposure[nearest] - step_times), left, nearest)
    half_period = 0.5e9 / max(fps, 1e-6)
    close = np.abs(exposure[nearest] - step_times) <= half_period
    previous = np.clip(np.searchsorted(exposure, step_times, side="right") - 1, 0, len(exposure) - 1)
    chosen = np.where(close, nearest, previous)
    return CameraSteps(name, video[chosen], ~close, exposure[chosen], (exposure[chosen] - step_times) / 1e6)


def _torn_frames(episode_dir: Path) -> set[int]:
    path = Path(episode_dir) / "frame_check.json"
    return set(json.loads(path.read_text())["torn_video_frames"]) if path.exists() else set()


def align_episode(episode_dir: Path, telemetry_path: Path | None = None, roots=DEFAULT_RUNS_ROOTS,
                  arms: int | None = None) -> Steps:
    """One step per primary-camera frame; `arms` (1 or 2) limits which teleop runs may match (default: rig)."""
    episode_dir = Path(episode_dir)
    meta = json.loads((episode_dir / "episode.json").read_text())
    camera = episode_io.primary(episode_dir, meta)
    rows = camera.rows()
    # Torn frames (frame_check.py: full size, not flagged by the driver, picture wrapped sideways) count as dropped
    # by the camera, so their step repeats the previous good image.
    torn = _torn_frames(episode_dir)
    rows = [row for row in rows if int(row["video_frame"]) not in torn]
    if not rows:
        raise AlignmentError(f"{episode_dir} has no frames")
    fps = int(round(camera.fps))
    arms = arms if arms is not None else rig.load().arms
    telemetry = load_telemetry(telemetry_path or find_telemetry(meta, roots, arms))

    sequence = np.array([int(row["camera_sequence"]) for row in rows], dtype=np.int64)
    exposure = np.array([int(row["exposure_monotonic_ns"]) for row in rows], dtype=np.int64)
    video_index = np.array([int(row["video_frame"]) for row in rows], dtype=np.int64)
    grid = np.arange(sequence[0], sequence[-1] + 1)
    present = np.isin(grid, sequence)
    step_exposure = np.round(np.interp(grid, sequence, exposure)).astype(np.int64)
    step_video = video_index[np.searchsorted(sequence, grid, side="right") - 1]

    covered = np.ones(len(grid), dtype=bool)
    state_parts, action_parts, max_state_gap, max_action_age = [], [], 0, 0
    for side in telemetry.sides:
        arm = telemetry.arms[side]
        gap = _nearest_gap(arm.feedback_t, step_exposure) if len(arm.feedback_t) > 1 else np.zeros(len(grid))
        inside = (step_exposure >= arm.feedback_t[0] - EDGE_NS) & (step_exposure <= arm.feedback_t[-1] + EDGE_NS)
        covered &= inside & (gap <= MAX_SAMPLE_GAP_NS)
        state_parts.append(_interpolate(arm.feedback_t, arm.feedback, step_exposure))
        held, age = _hold(arm.command_t, arm.command, step_exposure)
        covered &= age <= MAX_SAMPLE_GAP_NS
        action_parts.append(held)
        max_state_gap = max(max_state_gap, int(gap[covered].max()) if covered.any() else 0)
        max_action_age = max(max_action_age, int(age[covered].max()) if covered.any() else 0)
    if not covered.any():
        raise AlignmentError(f"{episode_dir}: no frame lies within the arm telemetry of {telemetry.path}")
    first, last = int(np.argmax(covered)), len(covered) - 1 - int(np.argmax(covered[::-1]))
    if not covered[first : last + 1].all():
        holes = int((~covered[first : last + 1]).sum())
        raise AlignmentError(f"{episode_dir}: arm telemetry has a gap over {holes} frames mid-episode")
    keep = slice(first, last + 1)
    state = np.concatenate(state_parts, axis=1)[keep].astype(np.float32)
    action = np.concatenate(action_parts, axis=1)[keep].astype(np.float32)
    report = {
        "episode": str(episode_dir),
        "telemetry": str(telemetry.path),
        "arms": len(telemetry.sides),
        "fps": fps,
        "steps": int(last - first + 1),
        "image_repeated_steps": int((~present[keep]).sum()),
        "torn_frames_replaced": sorted(torn),
        "torn_frame_check": "done" if (episode_dir / "frame_check.json").exists() else "NOT RUN (bimanual_collect check-frames)",
        "trimmed_start_steps_without_arm_data": first,
        "trimmed_end_steps_without_arm_data": len(grid) - 1 - last,
        "max_state_sample_gap_ms": round(max_state_gap / 1e6, 2),
        "max_action_age_ms": round(max_action_age / 1e6, 2),
        "duration_s": round((last - first + 1) / fps, 3),
    }
    others = {}
    for other in episode_io.cameras(episode_dir, meta):
        if other.primary:
            continue
        other_rows = other.rows()
        if not other_rows:
            raise AlignmentError(f"{episode_dir}: camera {other.name} recorded no frames")
        matched = match_camera(other.name, other_rows, other.fps, step_exposure[keep])
        others[other.name] = matched
        report.setdefault("cameras", {})[other.name] = {
            "repeated_steps": int(matched.repeated.sum()),
            "max_offset_ms": round(float(np.abs(matched.offset_ms[~matched.repeated]).max()), 2)
            if (~matched.repeated).any() else None,
        }
    return Steps(episode_dir, telemetry.path, fps, step_video[keep], ~present[keep], grid[keep],
                 step_exposure[keep], state, action, report, state_names(telemetry.sides), camera.name, others)


def episode_dirs(paths: list[Path]) -> list[Path]:
    """Accept episode folders or session folders (all their episodes, in order)."""
    found = []
    for path in map(Path, paths):
        if (path / "episode.json").exists():
            found.append(path)
        else:
            found.extend(sorted(p for p in path.glob("episode_*") if (p / "episode.json").exists()))
    if not found:
        raise AlignmentError(f"no recorded episodes under {', '.join(map(str, paths))}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Check how ZED episodes line up with the dual-arm telemetry")
    parser.add_argument("episodes", nargs="+", type=Path, help="episode or session folders")
    parser.add_argument("--telemetry", type=Path, help="telemetry.csv to use (default: find by time)")
    parser.add_argument("--runs-root", type=Path, action="append", help="where teleop runs are (default: rig config)")
    args = parser.parse_args()
    roots = tuple(args.runs_root) if args.runs_root else DEFAULT_RUNS_ROOTS
    ok = True
    for episode in episode_dirs(args.episodes):
        try:
            print(json.dumps(align_episode(episode, args.telemetry, roots).report, indent=2))
        except AlignmentError as exc:
            ok = False
            print(f"NOT EXPORTABLE: {exc}", file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
