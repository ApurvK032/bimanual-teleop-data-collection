"""Write configs/rig.yaml for this bench: detect the cameras, fetch ZED calibrations, ask a few questions.

    python -m bimanual_collect setup            # asks, with sensible defaults
    python -m bimanual_collect setup --yes      # takes the defaults (first ZED as the primary, 720p30)

The file is made from configs/rig.example.yaml (comments kept); an existing configs/rig.yaml is backed up first.
Pixel boxes are drawn afterwards in the console (Setup tab), on a live picture.
"""
from __future__ import annotations

import argparse
import re
import shutil
import time

import yaml

from . import rig
from .calibration import fetch_calibration
from .zed_v4l2 import RATES, SIZES, video_devices


def ask(prompt: str, default: str, assume_yes: bool) -> str:
    if assume_yes:
        print(f"{prompt} [{default}]: {default}")
        return default
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def build_config(example: str, *, serial: str | None, mode: str, fps: int, arms: str, operator: str,
                 tasks: list[str], extra_cameras: list[dict], arm_profile: str) -> str:
    """Edit the example's values in place, keeping every comment."""
    text = example

    def set_line(pattern: str, value: str) -> None:
        nonlocal text
        new, count = re.subn(pattern, lambda m: m.group(1) + value + (m.group(2) or ""), text, count=1, flags=re.M)
        if count != 1:
            raise ValueError(f"rig.example.yaml changed: no line matching {pattern!r}")
        text = new

    set_line(r"^(  serial: )\S+(\s+#.*)?$", f'"{serial}"' if serial else "null")
    set_line(r'^(  mode: )"?\w+"?(\s+#.*)?$', f'"{mode}"')
    set_line(r"^(  fps: )\d+()$", str(fps))
    set_line(r"^(arms: )\S+(\s+#.*)?$", arms)
    set_line(r'^(operator: )"[^"]*"(\s+#.*)?$', yaml.safe_dump(operator, default_style='"').strip())
    set_line(r"^(  arm_profile: )\S+(\s+#.*)?$", arm_profile)
    task_lines = "".join(f"  - {yaml.safe_dump(t).strip().removesuffix('...').strip()}\n" for t in tasks)
    text, count = re.subn(r"^(tasks:[^\n]*\n)(?:  - [^\n]*\n)*", lambda m: m.group(1) + task_lines, text,
                          count=1, flags=re.M)
    if count != 1:
        raise ValueError("rig.example.yaml changed: no tasks list")
    text = re.sub(r"^cameras:\n(?:[ -][^\n]*\n)*\n?", "", text, flags=re.M)  # an earlier run's camera list
    if extra_cameras:
        primary = {"name": "zed", "type": "zed", "serial": serial, "mode": mode, "fps": fps, "primary": True}
        block = yaml.safe_dump({"cameras": [primary, *extra_cameras]}, sort_keys=False)
        text = text.replace("\npaths:\n", f"\n{block}\npaths:\n", 1)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true", help="take every default without asking")
    parser.add_argument("--no-download", action="store_true", help="don't fetch missing ZED calibrations")
    args = parser.parse_args()
    yes = args.yes

    devices = video_devices()
    zeds = [d for d in devices if d.is_zed]
    others = [d for d in devices if not d.is_zed]
    print("Cameras found:")
    for d in devices:
        print(f"  {d.node}: {d.model} serial {d.serial or '?'} USB {d.usb_mbps} Mbps")
    if not devices:
        print("  none (plug the cameras in, then run this again; writing a config without cameras anyway)")

    serial = zeds[0].serial if zeds else None
    if len(zeds) > 1:
        serial = ask("Several ZEDs: serial of the overhead (primary) one", zeds[0].serial, yes)
    mode = ask(f"ZED mode ({' / '.join(SIZES)})", "720", yes)
    if mode not in SIZES:
        raise SystemExit(f"unknown mode {mode!r}")
    fps = int(ask(f"Frame rate ({' / '.join(map(str, RATES[mode]))})", str(30 if 30 in RATES[mode] else RATES[mode][0]), yes))
    if fps not in RATES[mode]:
        raise SystemExit(f"mode {mode} runs at {RATES[mode]} fps")

    extra = []
    for d in others:
        if ask(f"Record {d.name} ({d.node}) too? (y/n)", "n", yes).lower().startswith("y"):
            name = ask("  name for it (e.g. wrist_right)", f"cam{len(extra) + 1}", yes)
            entry = {"name": name, "type": "uvc", "width": 640, "height": 480, "fps": 30}
            entry.update({"serial": d.serial} if d.serial else {"match": d.name})
            extra.append(entry)

    current = rig.load()
    arms = ask("Arms (1 / 2 / null = whatever the teleop run logged)", str(current.arms or "null"), yes)
    operator = ask("Default operator name", current.operator, yes)
    tasks_text = ask("Task presets, separated by ';'", "; ".join(current.tasks) or "fold the towel", yes)
    tasks = [t.strip() for t in tasks_text.split(";") if t.strip()]
    profile = ask("Two-arm teleop profile (for the sim scene check)",
                  str(current.arm_profile.relative_to(rig.ROOT)) if current.arm_profile.is_relative_to(rig.ROOT)
                  else str(current.arm_profile), yes)

    # Start from this bench's own config when there is one (keeps its boxes, QC limits, paths), else the example.
    values = dict(serial=serial, mode=mode, fps=fps, arms=arms, operator=operator, tasks=tasks, extra_cameras=extra,
                  arm_profile=profile)
    base = rig.CANDIDATES[0] if rig.CANDIDATES[0].exists() else rig.CANDIDATES[1]
    try:
        config = build_config(base.read_text(), **values)
    except ValueError:
        print(f"{base.relative_to(rig.ROOT)} is hand-edited beyond recognition; starting from the example")
        config = build_config(rig.CANDIDATES[1].read_text(), **values)
    target = rig.CANDIDATES[0]
    if target.exists():
        backup = target.with_name(f"rig.yaml.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(target, backup)
        print(f"backed up the old config to {backup.relative_to(rig.ROOT)}")
    target.write_text(config)
    rig.load.cache_clear()
    rig.load(str(target))  # parses
    print(f"wrote {target.relative_to(rig.ROOT)}")

    for d in zeds:
        calibration = rig.load().calibration(d.serial or None)
        if calibration.exists() or not d.serial:
            continue
        if args.no_download:
            print(f"missing calibration for {d.serial}: python -m bimanual_collect calibration --serial {d.serial}")
            continue
        try:
            print(f"downloaded {fetch_calibration(d.serial).relative_to(rig.ROOT)}")
        except Exception as exc:  # noqa: BLE001 - network
            print(f"could not download the calibration for {d.serial} ({exc}); try python -m bimanual_collect calibration later")
    print("Next: python -m bimanual_collect doctor, then start a session and draw the boxes in the console's Setup tab.")


if __name__ == "__main__":
    main()
