"""Export recorded episodes to a LeRobot v3.0 dataset: one or two arms + the recorded cameras.

    .venv-lerobot/bin/python -m bimanual_collect export \
        data/raw/<date>/<session> [more sessions or episodes] --repo-id cloth_bimanual
    # -> data/lerobot/cloth_bimanual/ (a LeRobot dataset) + data/lerobot/cloth_bimanual.export.json

Uses LeRobot's own LeRobotDataset writer (create / add_frame / save_episode / finalize), so LeRobot training
code reads the result directly. Per step (see lerobot_align.py for the exact timing rules):
  observation.state  float32[14]  measured   left_joint_0..5, left_gripper, right_joint_0..5, right_gripper
                     (float32[7], joint_0..5, gripper, for a one-arm run)
  action             float32[14]  commanded  same order
  observation.images.zed_left / zed_right   per eye, natural colour (the live view's colour fix;
                                             --camera-colour keeps the camera's own); capture size, or
                                             --image-size WxH (area-averaged) for policy training
  observation.frame_repeated                 1 when the camera dropped that frame and the previous image is reused
  observation.zed_exposure_monotonic_ns      start of exposure of the image shown
The lossless recording in data/raw stays the master; LeRobot stores lossy video (AV1 here, CRF --crf).
"""
from __future__ import annotations

import argparse
import functools
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import catalog, episode_io, rig
from .frame_check import ensure as ensure_frame_check
from .frame_check import wait_while_busy
from .lerobot_align import DEFAULT_RUNS_ROOTS, AlignmentError, align_episode, episode_dirs
from .zed_format import EyeDecoder, iter_frames

EXPORT = rig.load().export
ROBOT_TYPES = {1: "widowxai_quest", 2: "widowxai_bimanual_quest"}  # unless the rig config names one
DEFAULT_ROOT = rig.load().data_lerobot


@dataclass
class CameraOut:
    """How one recorded camera becomes dataset features."""

    name: str
    primary: bool
    stereo: bool
    width: int  # capture frame (both eyes for stereo)
    height: int
    out_size: tuple[int, int]  # per image
    views: tuple[tuple[str, int], ...]  # (feature key, x0 of the eye in the frame)
    colour_fix: bool

    @property
    def repeated_key(self) -> str:
        return "observation.frame_repeated" if self.primary else f"observation.{self.name}_frame_repeated"

    @property
    def exposure_key(self) -> str:
        return f"observation.{self.name}_exposure_monotonic_ns"


def fit_size(width: int, height: int, size: tuple[int, int] | None) -> tuple[int, int]:
    """The requested size, or (for a different aspect ratio) the requested height at the camera's own aspect."""
    if size is None:
        return width, height
    out_w, out_h = size
    if abs(width / height - out_w / out_h) > 0.01:
        out_w = max(2, round(out_h * width / height / 2) * 2)
    return out_w, out_h


def camera_outputs(cameras, eyes: tuple[str, ...], size: tuple[int, int] | None, camera_colour: bool) -> list[CameraOut]:
    outputs = []
    for cam in cameras:
        views = tuple((f"observation.images.{cam.name}_{eye}", 0 if eye == "left" else cam.eye_width)
                      for eye in eyes) if cam.stereo else ((f"observation.images.{cam.name}", 0),)
        fix = not camera_colour and str(cam.info.get("model", "ZED" if cam.stereo else "")).startswith("ZED")
        outputs.append(CameraOut(cam.name, cam.primary, cam.stereo, cam.width, cam.height,
                                 fit_size(cam.eye_width, cam.height, size), views, fix))
    return outputs


def dataset_features(outputs: list[CameraOut], names: list[str]) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": list(names)},
        "action": {"dtype": "float32", "shape": (len(names),), "names": list(names)},
    }
    for out in outputs:
        # Not "*image*_repeated": LeRobot treats any feature whose name contains "image" as a picture when it
        # merges statistics across episodes, and stops on the shape.
        features[out.repeated_key] = {"dtype": "int64", "shape": (1,), "names": None}
        features[out.exposure_key] = {"dtype": "int64", "shape": (1,), "names": None}
        for key, _ in out.views:
            features[key] = {"dtype": "video", "shape": (out.out_size[1], out.out_size[0], 3),
                             "names": ["height", "width", "channels"]}
    return features


class FrameCursor:
    """Walks one camera's video forward, decoding each frame once."""

    def __init__(self, video: Path, width: int, height: int) -> None:
        self.video = video
        self._frames = iter_frames(video, width, height)
        self.index, self.yuyv = -1, None

    def get(self, index: int) -> np.ndarray:
        if index < self.index:
            raise SystemExit(f"EXPORT STOPPED: {self.video}: frames requested out of order ({index} < {self.index})")
        while self.index < index:
            self.yuyv = next(self._frames, None)
            if self.yuyv is None:
                raise SystemExit(f"EXPORT STOPPED: {self.video} ends before video frame {index}")
            self.index += 1
        return self.yuyv


def view_images(decoder: EyeDecoder, yuyv: np.ndarray, out: CameraOut) -> dict:
    """RGB per view (eye), resized (area average) to the output size."""
    images = {}
    eye_width = out.width // 2 if out.stereo else out.width
    for key, x0 in out.views:
        rgb = cv2.cvtColor(decoder.bgr(yuyv, x0, eye_width), cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != out.out_size:
            rgb = cv2.resize(rgb, out.out_size, interpolation=cv2.INTER_AREA)
        images[key] = rgb
    return images


def parse_size(text: str | None) -> tuple[int, int] | None:
    """'640x360' -> (640, 360); None/'native' -> None (keep the capture size)."""
    if text in (None, "", "native"):
        return None
    width, height = (int(v) for v in str(text).lower().split("x"))
    if width <= 0 or height <= 0:
        raise ValueError(f"bad image size {text!r}")
    return width, height


def arm_ips(telemetry_path: Path) -> dict:
    """Which arm was on which side in this episode's teleop run (from the run's config snapshot)."""
    try:
        snapshot = json.loads((Path(telemetry_path).parent / "config_snapshot.json").read_text())
    except (OSError, ValueError):
        return {}
    arms = snapshot.get("arms")
    if isinstance(arms, dict) and "left" in arms:
        return {side: str((arms.get(side) or {}).get("robot_ip", "unknown")) for side in ("left", "right")}
    robot_ip = (snapshot.get("hardware") or {}).get("robot_ip")
    return {"arm": str(robot_ip)} if robot_ip else {}


def set_video_quality(crf: int) -> None:
    """LeRobot's writer always encodes at its default CRF 30. Set ours in this process only (no file edits)."""
    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    lerobot_dataset.encode_video_frames = functools.partial(lerobot_dataset.encode_video_frames, crf=crf)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episodes", nargs="+", type=Path, help="session or episode folders under data/raw")
    parser.add_argument("--repo-id", required=True, help="dataset name, e.g. cloth_bimanual")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="parent folder for the dataset")
    parser.add_argument("--eyes", choices=["both", "left"], default=str(EXPORT.get("eyes", "both")))
    parser.add_argument("--image-size", default=EXPORT.get("image_size"),
                        help="per-eye image size WxH, e.g. 640x360 (default: rig config, else the capture size)")
    parser.add_argument("--crf", type=int, default=int(EXPORT.get("crf", 18)),
                        help="AV1 quality, lower = better (LeRobot default 30)")
    parser.add_argument("--vcodec", default="libsvtav1", choices=["libsvtav1", "h264", "hevc"])
    parser.add_argument("--task", help="task text for every episode (default: the one typed when recording)")
    parser.add_argument("--camera-colour", action="store_true", help="skip the natural-colour fix")
    parser.add_argument("--telemetry", type=Path, help="one telemetry.csv for all episodes (default: by time)")
    parser.add_argument("--runs-root", type=Path, action="append", help="teleop runs folder (default: rig config)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset of this name")
    parser.add_argument("--all", action="store_true", help="include episodes not accepted in review")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    roots = tuple(args.runs_root) if args.runs_root else DEFAULT_RUNS_ROOTS
    episodes = episode_dirs(args.episodes)
    if not args.all:
        skipped = [catalog.relative(path) for path in episodes if not catalog.is_accepted(path)]
        episodes = [path for path in episodes if catalog.is_accepted(path)]
        if skipped:
            print(f"skipping {len(skipped)} episode(s) not accepted in review (review --accept, or --all): "
                  + ", ".join(skipped), flush=True)
        if not episodes:
            raise SystemExit("EXPORT STOPPED: no accepted episodes (python -m bimanual_collect review)")
    for path in episodes:  # torn (sideways-wrapped) frames become camera-dropped steps in the alignment
        torn = ensure_frame_check(path)
        if torn:
            print(f"{path}: {len(torn)} torn frame(s) {torn[:10]} replaced by the previous good image", flush=True)
    try:  # align everything first, so a bad episode stops the export before anything is written
        aligned = [align_episode(path, args.telemetry, roots) for path in episodes]
    except AlignmentError as exc:
        raise SystemExit(f"EXPORT STOPPED: {exc}") from None
    metas = [json.loads((path / "episode.json").read_text()) for path in episodes]
    layouts = {tuple(steps.names) for steps in aligned}
    if len(layouts) != 1:
        raise SystemExit("EXPORT STOPPED: episodes mix one-arm and two-arm recordings; export them separately")
    names = list(layouts.pop())
    robot_type = str(EXPORT.get("robot_type") or ROBOT_TYPES[len(names) // 7])
    camera_sets = [episode_io.cameras(path, meta) for path, meta in zip(episodes, metas)]
    signatures = {tuple((c.name, c.width, c.height, c.stereo, c.primary) for c in cams) for cams in camera_sets}
    rates = {steps.fps for steps in aligned}
    if len(signatures) != 1 or len(rates) != 1:
        raise SystemExit(f"EXPORT STOPPED: episodes differ in cameras/modes {sorted(signatures)} or rates {rates}; "
                         "export them separately")
    eyes = ("left", "right") if args.eyes == "both" else ("left",)
    out_size = parse_size(args.image_size)
    outputs = camera_outputs(camera_sets[0], eyes, out_size, args.camera_colour)

    output = args.root / args.repo_id
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"EXPORT STOPPED: {output} exists (use --overwrite to replace it)")
        if not (output / "meta" / "info.json").exists():
            raise SystemExit(f"EXPORT STOPPED: {output} exists and is not a LeRobot dataset; not deleting it")
        shutil.rmtree(output)
    args.root.mkdir(parents=True, exist_ok=True)

    set_video_quality(args.crf)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=rates.pop(),
        features=dataset_features(outputs, names),
        root=output,
        robot_type=robot_type,
        use_videos=True,
        image_writer_threads=8,
        vcodec=args.vcodec,
    )
    decoders = {True: EyeDecoder(colour_fix=True), False: EyeDecoder(colour_fix=False)}
    started = time.time()
    reports = []
    for path, meta, steps, cams in zip(episodes, metas, aligned, camera_sets):
        task = args.task or catalog.task_of(path, meta) or "cloth manipulation"
        records = {cam.name: cam for cam in cams}
        cursors = {out.name: FrameCursor(records[out.name].video, out.width, out.height) for out in outputs}
        cache: dict[str, tuple[int, dict]] = {}
        for k in range(len(steps.state)):
            if k % 30 == 0:
                wait_while_busy()
            frame = {"observation.state": steps.state[k], "action": steps.action[k], "task": task}
            for out in outputs:
                if out.primary:
                    index, repeated, exposure = steps.video_frame[k], steps.image_repeated[k], steps.exposure_monotonic_ns[k]
                else:
                    matched = steps.cameras[out.name]
                    index, repeated, exposure = matched.video_frame[k], matched.repeated[k], matched.exposure_monotonic_ns[k]
                if out.name not in cache or cache[out.name][0] != index:
                    cache[out.name] = (index, view_images(decoders[out.colour_fix], cursors[out.name].get(int(index)), out))
                frame.update(cache[out.name][1])
                frame[out.repeated_key] = np.array([int(repeated)], dtype=np.int64)
                frame[out.exposure_key] = np.array([int(exposure)], dtype=np.int64)
            dataset.add_frame(frame)
        dataset.save_episode()
        reports.append({**steps.report, "task": task, "arms": arm_ips(steps.telemetry_path)})
        print(f"episode {len(reports) - 1}: {steps.report['steps']} steps from {path}", flush=True)
    dataset.finalize()

    manifest = {
        "dataset": str(output),
        "lerobot_codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "eyes": list(eyes),
        "cameras": [{"name": out.name, "primary": out.primary, "stereo": out.stereo,
                     "capture_size": [out.width // 2 if out.stereo else out.width, out.height],
                     "image_size": list(out.out_size), "features": [key for key, _ in out.views],
                     "colour": "natural-colour fix" if out.colour_fix else "camera colour"} for out in outputs],
        "image_size_per_eye": list(outputs[0].out_size),
        "capture_size_per_eye": [outputs[0].width // 2 if outputs[0].stereo else outputs[0].width, outputs[0].height],
        "resize": "none" if out_size is None else "OpenCV INTER_AREA from the capture size",
        "video": {"vcodec": args.vcodec, "crf": args.crf},
        "state_action_names": names,
        "arms": "per episode (episodes[].arms, from each teleop run's config snapshot)",
        "episodes": reports,
        "export_seconds": round(time.time() - started, 1),
    }
    manifest_path = args.root / f"{args.repo_id}.export.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"dataset -> {output}\nmanifest -> {manifest_path}")


if __name__ == "__main__":
    main()
