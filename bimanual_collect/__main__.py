"""Command line: python -m bimanual_collect <command> [args]   (run from the repository root)."""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "app": ("app", "the collection app: one browser UI for cameras, arms, headset, episodes (http://localhost:8440/)"),
    "session": ("app", "the app in this terminal, without opening the browser (Ctrl+C stops)"),
    "install-shortcut": ("shortcut", "add the collection app to the desktop and the app menu"),
    "frames": ("episode_frames", "save chosen frames of an episode as natural-colour PNGs"),
    "check-frames": ("frame_check", "find torn frames in an episode"),
    "sync": ("sync_check", "camera-vs-arm timing check for an episode"),
    "align": ("lerobot_align", "align camera frames with arm telemetry (no write)"),
    "depth": ("episode_depth", "per-frame depth from the stereo pair (SGBM or RAFT)"),
    "doctor": ("doctor", "check this PC is ready (packages, cameras, FFmpeg, envs) and how to fix what isn't"),
    "calibration": ("calibration", "download each connected ZED's factory calibration from Stereolabs"),
    "setup": ("setup_wizard", "write configs/rig.yaml for this bench (detects cameras, fetches calibrations)"),
    "process": ("process", "after a session: check every episode, QC verdict, index, session report"),
    "review": ("review", "list episodes with QC; accept / reject; set task labels"),
    "export": ("export_lerobot", "write accepted episodes as a LeRobot v3.0 dataset"),
    "sim-scene": ("sim_scene", "two-arm + table MuJoCo scene check and picture"),
    "camera-server": ("camera_server", "camera + recorder service on :8444 (started by the app)"),
    "quest-page": ("quest_page", "headset page + pose relay on :8443 (started by the app)"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = [__doc__.strip(), "", "commands:"]
    lines += [f"  {name:<{width}}  {help_text}" for name, (_, help_text) in COMMANDS.items()]
    lines += ["", "python -m bimanual_collect <command> --help for its options"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int | None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"unknown command {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module_name = COMMANDS[name][0]
    if name == "session":
        rest = ["--no-browser", *rest]
    module = importlib.import_module(f"bimanual_collect.{module_name}")
    sys.argv = [f"bimanual_collect {name}", *rest]
    return module.main()


if __name__ == "__main__":
    sys.exit(main())
