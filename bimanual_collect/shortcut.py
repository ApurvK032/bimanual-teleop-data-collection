"""Add the collection app to the app menu and the desktop (no terminal needed afterwards).

    python -m bimanual_collect install-shortcut           menu entry + desktop icon
    python -m bimanual_collect install-shortcut --remove  take them away again

Clicking the icon runs `python -m bimanual_collect app --open`: it starts the app in the background if it isn't
running, then opens http://localhost:8440/ in the browser.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import rig

NAME = "bimanual-collect"
ICON = Path(__file__).resolve().parent / "web" / "console" / "icon.svg"


def desktop_entry(python: str) -> str:
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        "Version=1.0",
        "Name=Robot data collection",
        "GenericName=Teleoperation and recording",
        "Comment=Set up the bench, drive the arms with the Quest, record episodes",
        f"Exec=env -u PYTHONPATH {python} -m bimanual_collect app --open",
        f"Path={rig.ROOT}",
        f"Icon={NAME}",
        "Terminal=false",
        "Categories=Science;Development;",
        "StartupNotify=false",
        "",
    ])


def paths() -> dict[str, Path]:
    override = os.environ.get("BIMANUAL_SHORTCUT_ROOT")  # tests: keep the icons out of the real desktop
    if override:
        root = Path(override)
        return {"menu": root / "applications" / f"{NAME}.desktop", "icon": root / "icons" / f"{NAME}.svg",
                "desktop": root / "Desktop" / f"{NAME}.desktop"}
    home = Path.home()
    data = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
    desktop = None
    try:
        found = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, timeout=5).stdout.strip()
        desktop = Path(found) if found and Path(found) != home else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    desktop = desktop or (home / "Desktop")
    return {"menu": data / "applications" / f"{NAME}.desktop",
            "icon": data / "icons" / "hicolor" / "scalable" / "apps" / f"{NAME}.svg",
            "desktop": desktop / f"{NAME}.desktop" if desktop.is_dir() else None}


def install(python: str | None = None) -> dict[str, str]:
    python = python or str(rig.ROOT / ".venv" / "bin" / "python")
    if not Path(python).exists():
        python = sys.executable
    where = paths()
    where["icon"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ICON, where["icon"])
    entry = desktop_entry(python)
    done = {}
    for key in ("menu", "desktop"):
        target = where[key]
        if target is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(entry)
        target.chmod(0o755)
        if key == "desktop" and shutil.which("gio"):
            # GNOME asks "Allow Launching?" for desktop files unless they are marked trusted.
            subprocess.run(["gio", "set", str(target), "metadata::trusted", "true"], capture_output=True, check=False)
        done[key] = str(target)
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(where["menu"].parent)], capture_output=True, check=False)
    return done


def remove() -> list[str]:
    removed = []
    for target in paths().values():
        if target is not None and target.exists():
            target.unlink()
            removed.append(str(target))
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    if args.remove:
        for path in remove():
            print(f"removed {path}")
        return
    for key, path in install().items():
        print(f"{key}: {path}")
    print("Click 'Robot data collection' in the app menu or on the desktop to start.")


if __name__ == "__main__":
    main()
