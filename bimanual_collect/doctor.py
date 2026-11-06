"""Check that this PC is ready to collect data, and say how to fix what isn't.

    python -m bimanual_collect doctor

Checks: Python packages, the teleop submodule and its page, FFmpeg encoders, v4l2-ctl, camera permissions, every
configured camera (found, USB 3, calibration file), the data folder, the Quest (adb), the arm driver environment,
the PC's power mode and route to the arms, the LeRobot and GPU depth environments. Exit code 1 if anything required
fails. The app's System tab shows the same list, with a button for each fix it can make itself (`action`).
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import rig


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str
    fix: str = ""
    action: str = ""  # a fix the app can run: system_setup | power | calibration | installer | shortcut | bench


def _run(command: list[str], timeout: float = 20) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def check_python() -> list[Check]:
    checks = [Check("python", "ok" if sys.version_info >= (3, 10) else "fail",
                    f"{sys.version.split()[0]} ({sys.executable})", "use Python 3.10 or newer")]
    for module, extra in (("numpy", ""), ("cv2", "opencv-python-headless"), ("yaml", "pyyaml"),
                          ("fastapi", "the [session] extra"), ("uvicorn", "the [session] extra")):
        try:
            imported = importlib.import_module(module)
            checks.append(Check(f"package {module}", "ok", getattr(imported, "__version__", "installed")))
        except ImportError:
            checks.append(Check(f"package {module}", "fail", "missing", f"pip install -e '.[session]' ({extra or module})"))
    return checks


def check_teleop() -> list[Check]:
    settings = rig.load()
    checks = []
    try:
        import widowxai_quest_teleop  # noqa: F401
        checks.append(Check("teleop package", "ok", "importable"))
    except ImportError:
        return [Check("teleop package", "fail", "widowxai_quest_teleop not importable",
                      "git submodule update --init teleop && pip install -e ./teleop")]
    model = settings.teleop / "third_party" / "trossen_arm_mujoco" / "trossen_arm_mujoco" / "assets" / "wxai"
    model = model / "wxai_follower.xml"
    checks.append(Check("teleop arm model", "ok" if model.exists() else "fail",
                        "present" if model.exists() else "missing (the arm programs and the simulator load it)",
                        "run the installer again (or: git submodule update --init --recursive teleop)", "installer"))
    head = _run(["git", "-C", str(settings.teleop), "rev-parse", "--short", "HEAD"])
    checks.append(Check("teleop submodule", "ok" if head and head.returncode == 0 else "warn",
                        head.stdout.strip() if head and head.returncode == 0 else f"{settings.teleop} is not a git checkout"))
    try:
        from . import quest_page

        with tempfile.TemporaryDirectory() as tmp:
            quest_page.build_web(Path(tmp))
        checks.append(Check("Quest page patches", "ok", "all apply to the teleop page"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("Quest page patches", "fail", str(exc)[:200],
                            "the teleop page changed: check out the pinned submodule commit (git submodule update)"))
    runs = settings.teleop_runs
    checks.append(Check("teleop runs folder", "ok" if any(Path(r).exists() for r in runs) else "warn",
                        ", ".join(str(r) for r in runs), "created by the first teleop run; nothing to do"))
    return checks


def check_tools() -> list[Check]:
    checks = []
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return [Check("ffmpeg", "fail", "not found", "System setup (or: sudo apt install ffmpeg)", "system_setup")]
    version = _run(["ffmpeg", "-version"])
    checks.append(Check("ffmpeg", "ok", version.stdout.splitlines()[0][:60] if version else ffmpeg))
    encoders = _run(["ffmpeg", "-hide_banner", "-encoders"])
    listing = encoders.stdout if encoders else ""
    for name, needed_for, required in (("ffv1", "lossless recording", True), ("libx264", "previews", False),
                                       ("libsvtav1", "LeRobot export (AV1)", False)):
        present = f" {name} " in listing
        checks.append(Check(f"encoder {name}", "ok" if present else ("fail" if required else "warn"),
                            f"{'present' if present else 'missing'} ({needed_for})",
                            "install an FFmpeg build with it (Ubuntu's ffmpeg package has all three)"))
    checks.append(Check("v4l2-ctl", "ok" if shutil.which("v4l2-ctl") else "fail",
                        "present" if shutil.which("v4l2-ctl") else "not found",
                        "System setup (or: sudo apt install v4l-utils)", "system_setup"))
    return checks


def check_cameras() -> list[Check]:
    from .zed_v4l2 import video_devices

    settings = rig.load()
    checks = []
    devices = video_devices()
    entries = list(settings.cameras) or [{"name": "zed", "type": "zed", "serial": settings.camera_serial}]
    for entry in entries:
        name, kind = entry.get("name", "?"), entry.get("type", "zed")
        if kind == "zed":
            zeds = [d for d in devices if d.is_zed and (not entry.get("serial") or d.serial == str(entry["serial"]))]
            if not zeds:
                rules = list(Path("/etc/udev/rules.d").glob("98-zed-video.rules"))
                checks.append(Check(f"camera {name}", "fail", "no ZED found",
                                    "plug it into a USB 3 port" + ("" if rules else "; then System setup (camera access)"),
                                    "" if rules else "system_setup"))
                continue
            zed = zeds[0]
            ok_usb = (zed.usb_mbps or 0) >= 5000
            checks.append(Check(f"camera {name}", "ok" if ok_usb else "fail",
                                f"{zed.model} {zed.serial} on {zed.node}, USB {zed.usb_mbps} Mbps",
                                "" if ok_usb else "it needs USB 3: reseat or flip the USB-C plug at the camera"))
            calibration = settings.calibration(zed.serial or None)
            checks.append(Check(f"calibration {name}", "ok" if calibration.exists() else "fail",
                                str(calibration.relative_to(rig.ROOT)) if calibration.exists() else "missing",
                                "download it from Stereolabs (button, or: python -m bimanual_collect calibration)",
                                "calibration"))
            readable = Path(zed.node).exists() and _run(["v4l2-ctl", "-d", zed.node, "--info"])
            if readable is not None and readable.returncode != 0:
                checks.append(Check(f"permissions {name}", "fail", f"{zed.node} not readable",
                                    "System setup (camera access), then replug the camera", "system_setup"))
        else:
            matches = [d for d in devices if not d.is_zed and (not entry.get("serial") or d.serial == str(entry["serial"]))
                       and (not entry.get("match") or entry["match"].lower() in d.name.lower())]
            checks.append(Check(f"camera {name}", "ok" if len(matches) == 1 else "fail",
                                f"{matches[0].name} on {matches[0].node}" if len(matches) == 1 else
                                f"{len(matches)} cameras match serial={entry.get('serial')} match={entry.get('match')}",
                                "" if len(matches) == 1 else "check the camera's serial/match in configs/rig.yaml"))
    return checks


def check_storage() -> list[Check]:
    settings = rig.load()
    data = settings.data_raw
    try:
        data.mkdir(parents=True, exist_ok=True)
        probe = data / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        writable = True
    except OSError:
        writable = False
    free_gb = shutil.disk_usage(data if data.exists() else rig.ROOT).free / 1e9
    stop_gb = float(settings.recording.get("min_free_to_start_gb", 20))
    return [
        Check("data folder", "ok" if writable else "fail", str(data), "make it writable or change paths.data_raw"),
        Check("free disk", "ok" if free_gb > 3 * stop_gb else ("warn" if free_gb > stop_gb else "fail"),
              f"{free_gb:.0f} GB free (about {free_gb / 0.9:.0f} min at 720p30)", "free some space"),
        Check("rig config", "ok" if settings.source.name == "rig.yaml" else "warn",
              str(settings.source.relative_to(rig.ROOT)), "python -m bimanual_collect setup (writes configs/rig.yaml)"),
    ]


def check_quest() -> list[Check]:
    if not shutil.which("adb"):
        return [Check("adb", "warn", "not found (needed for the Quest over USB)",
                      "System setup (or: sudo apt install adb)", "system_setup")]
    state = _run(["adb", "get-state"], timeout=5)
    connected = bool(state and state.stdout.strip() == "device")
    return [Check("Quest", "ok" if connected else "warn", "connected" if connected else "not connected",
                  "connect it by USB, enable developer mode and accept USB debugging")]


def check_environments() -> list[Check]:
    settings = rig.load()
    checks = []
    lerobot = settings.lerobot_python
    if lerobot.exists():
        result = _run([str(lerobot), "-c", "import lerobot, importlib.metadata as m; print(m.version('lerobot'))"], 60)
        ok = bool(result and result.returncode == 0)
        checks.append(Check("LeRobot env", "ok" if ok else "warn",
                            f"lerobot {result.stdout.strip()}" if ok else f"{lerobot}: lerobot not importable",
                            "run the installer again (creates .venv-lerobot)", "installer"))
    else:
        checks.append(Check("LeRobot env", "warn", f"{lerobot} missing (needed for export)",
                            "run the installer again", "installer"))
    depth = settings.depth_python
    if depth.exists():
        result = _run([str(depth), "-c", "import torch, torchvision; print(torch.cuda.is_available())"], 60)
        cuda = bool(result and result.returncode == 0 and result.stdout.strip() == "True")
        checks.append(Check("GPU depth env", "ok" if cuda else "warn",
                            "torch with CUDA" if cuda else "torch without CUDA (RAFT depth will be slow)",
                            "scripts/install.sh --gpu-depth"))
    else:
        checks.append(Check("GPU depth env", "warn", f"{depth} missing (optional: RAFT depth)",
                            "scripts/install.sh --gpu-depth"))
    return checks


def check_arms() -> list[Check]:
    """The arm driver environment, the PC's power mode and its route to the arms (what the launchers check)."""
    import yaml

    from . import bench

    settings = rig.load()
    checks = []
    python = settings.teleop / ".venv-arm19" / "bin" / "python"
    try:
        wanted = str(yaml.safe_load(settings.arm_profile.read_text())["hardware"]["driver_version_tested"])
    except (OSError, KeyError, TypeError, ValueError):
        wanted = None
    if python.exists():
        result = _run([str(python), "-c", "import importlib.metadata as m; print(m.version('trossen-arm'))"], 60)
        version = result.stdout.strip() if result and result.returncode == 0 else None
        ok = version is not None and (wanted is None or version == wanted)
        checks.append(Check("arm driver env", "ok" if ok else "fail",
                            f"trossen-arm {version}" + (f" (profile wants {wanted})" if wanted and not ok else "")
                            if version else "trossen-arm not installed",
                            "run the installer again (builds teleop/.venv-arm19)", "installer"))
    else:
        checks.append(Check("arm driver env", "fail", "teleop/.venv-arm19 missing (needed to drive the arms)",
                            "run the installer again (builds teleop/.venv-arm19)", "installer"))
    power = bench.power_profile()
    checks.append(Check("power mode", "ok" if power["ok"] else "warn", power["detail"] or "unknown",
                        "set performance mode (the arm programs require it)", "power"))
    state = bench.load_state()
    ips = [state["left_ip"], state["right_ip"]] if state["arms"] == 2 else [state["single_ip"]]
    for ip in filter(None, ips):
        route = bench.route_to(ip)
        checks.append(Check(f"route to arm {ip}", "ok" if route["ok"] else "warn", route["detail"],
                            f"give a wired port {bench.SOURCE_IP} (Bench tab, step 1)", "bench"))
    return checks


def run_checks(cameras: bool = True) -> list[Check]:
    checks = check_python() + check_teleop() + check_tools()
    if cameras:
        checks += check_cameras()
    return checks + check_storage() + check_quest() + check_arms() + check_environments()


def main() -> int:
    marks = {"ok": "✓", "warn": "!", "fail": "✗"}
    checks = run_checks()
    width = max(len(c.name) for c in checks)
    for check in checks:
        line = f" {marks[check.status]}  {check.name:<{width}}  {check.detail}"
        print(line)
        if check.status != "ok" and check.fix:
            print(f"    {'':<{width}}  -> {check.fix}")
    failed = sum(c.status == "fail" for c in checks)
    warned = sum(c.status == "warn" for c in checks)
    print(f"\n{len(checks) - failed - warned} ok, {warned} warning(s), {failed} problem(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
