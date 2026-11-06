"""Export frames of a recorded episode as PNG images (full size, both eyes).

    python -m bimanual_collect frames data/raw/<date>/<session>/episode_0000 --every 15
    python -m bimanual_collect frames <episode> --frames 0,100,200 --raw

Writes <episode>/png/<video_frame>_seq<camera_sequence>_{left,right}.png in natural colour (the same fix as
the live view), and with --raw also *_raw.png in camera colour (use those for stereo depth).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from . import episode_io
from .zed_format import EyeDecoder, iter_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episode", type=Path)
    parser.add_argument("--every", type=int, help="keep every Nth frame (default: about 1 per second)")
    parser.add_argument("--frames", help="comma-separated video frame numbers instead of --every")
    parser.add_argument("--raw", action="store_true", help="also write camera-colour images")
    parser.add_argument("--camera", help="camera name (default: the primary camera)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    cam = episode_io.camera(args.episode, args.camera) if args.camera else episode_io.primary(args.episode)
    width, height = cam.width, cam.height
    every = args.every or max(1, round(cam.fps))
    rows = cam.rows()
    wanted = {int(item) for item in args.frames.split(",")} if args.frames else set(range(0, len(rows), every))
    out = args.out or args.episode / "png"
    out.mkdir(parents=True, exist_ok=True)
    natural, camera = EyeDecoder(colour_fix=True), EyeDecoder(colour_fix=False)
    written = 0
    for index, yuyv in enumerate(iter_frames(cam.video, width, height)):
        if index >= len(rows):
            break
        if index not in wanted:
            continue
        stem = f"{index:05d}_seq{rows[index]['camera_sequence']}"
        half = cam.eye_width
        for eye, x0 in ((("left", 0), ("right", half)) if cam.stereo else (("image", 0),)):
            cv2.imwrite(str(out / f"{stem}_{eye}.png"), natural.bgr(yuyv, x0, half))
            if args.raw:
                cv2.imwrite(str(out / f"{stem}_{eye}_raw.png"), camera.bgr(yuyv, x0, half))
        written += 1
    print(f"wrote {written} frame pairs to {out}")


if __name__ == "__main__":
    main()
