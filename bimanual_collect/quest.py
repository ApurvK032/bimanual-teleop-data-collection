"""The Quest headset, seen from the PC: USB (adb) state, port forwarding, and what the headset page streams.

adb side (every 2 s): when a headset shows up authorised, forward the page and camera ports to it
(`adb reverse` 8443/8444) and keep it awake with the proximity sensor overridden (the same broadcasts as the
teleop repo's restart_localhost.sh), so WebXR keeps running while the headset is off the face. Re-checked every
20 s, because the forwards vanish when the headset reconnects.

Page side: the Quest page service's /quest/status (is the page open, which mode, controllers tracked, grips).
`ready_for()` turns both into "can this arm run start" plus a plain instruction when it can't.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
import urllib.error

from .services import http_json

PORTS = (8443, 8444)
PAGE_URL = "http://localhost:8443/"
BROWSER = "com.oculus.browser"
GRIP_RELEASED = 0.2


def adb(*args: str, serial: str | None = None, timeout: float = 5.0) -> subprocess.CompletedProcess:
    command = ["adb", *(["-s", serial] if serial else []), *args]
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def parse_devices(text: str) -> list[dict]:
    """`adb devices -l` -> [{serial, state, model}]"""
    devices = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        info = dict(part.split(":", 1) for part in parts[2:] if ":" in part)
        state = "no_permission" if "no permissions" in line else parts[1]
        devices.append({"serial": parts[0], "state": state, "model": info.get("model", ""),
                        "product": info.get("product", "")})
    return devices


MESSAGES = {
    "no_adb": "adb is not installed: run System setup (System tab).",
    "none": "No headset on USB. Connect the Quest with a USB-C cable (developer mode on).",
    "unauthorized": "Put the headset on and accept 'Allow USB debugging' (tick 'Always allow from this computer').",
    "no_permission": "This PC has no permission for the headset's USB port: run System setup (System tab).",
    "offline": "The headset is not answering over USB: unplug and replug the cable.",
    "connected": "Headset connected.",
}


def usb_state(device: dict | None) -> str:
    """adb's device state -> the app's: none | connected | unauthorized | no_permission | offline."""
    if device is None:
        return "none"
    state = {"device": "connected"}.get(device["state"], device["state"])
    return state if state in MESSAGES else "offline"


class QuestManager:
    def __init__(self, start: bool = True, page_status_url: str = "http://127.0.0.1:8443/quest/status") -> None:
        self.page_status_url = page_status_url
        self.adb_path = shutil.which("adb")
        self.device: dict | None = None
        self.state = "no_adb" if self.adb_path is None else "none"
        self.prepared_serial: str | None = None
        self.prepared_at = 0.0
        self.last_error: str | None = None
        self.opened_page_at: float | None = None
        self._page: dict | None = None
        self._page_at = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if start:
            threading.Thread(target=self._loop, name="quest", daemon=True).start()

    # -- adb
    def _poll_adb(self) -> None:
        if self.adb_path is None:
            self.adb_path = shutil.which("adb")
            if self.adb_path is None:
                return
        try:
            devices = parse_devices(adb("devices", "-l").stdout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.last_error = str(exc)
            return
        quests = [d for d in devices if "quest" in (d["model"] + d["product"]).lower() or d["product"] == "eureka"]
        device = (quests or devices or [None])[0]
        with self._lock:
            self.device = device
            self.state = usb_state(device)
        if self.state != "connected":
            self.prepared_serial = None
            return
        stale = time.time() - self.prepared_at > 20
        if self.prepared_serial != device["serial"] or stale:
            self.prepare(device["serial"], quick=self.prepared_serial == device["serial"])

    def prepare(self, serial: str, quick: bool = False) -> None:
        """Forward the ports; on first sight also keep the headset awake with tracking running."""
        try:
            if quick:
                listed = adb("reverse", "--list", serial=serial).stdout
                if all(f"tcp:{port}" in listed for port in PORTS):
                    self.prepared_at = time.time()
                    return
            for port in PORTS:
                adb("reverse", f"tcp:{port}", f"tcp:{port}", serial=serial)
            if not quick:
                for action in ("automation_disable", "prox_close"):
                    adb("shell", "am", "broadcast", "-a", f"com.oculus.vrpowermanager.{action}", serial=serial)
                adb("shell", "svc", "power", "stayon", "true", serial=serial)
                adb("shell", "input", "keyevent", "KEYCODE_WAKEUP", serial=serial)
            self.prepared_serial, self.prepared_at, self.last_error = serial, time.time(), None
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.last_error = f"could not set up the headset: {exc}"

    def open_page(self, mode: str = "bimanual") -> dict:
        """Open the headset page in the Quest browser (mode: bimanual | left | right)."""
        if self.state != "connected" or self.device is None:
            raise RuntimeError(MESSAGES.get(self.state, "no headset"))
        query = "?input=bimanual" if mode == "bimanual" else f"?input=single&hand={mode}&mode=real"
        result = adb("shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", PAGE_URL + query, BROWSER,
                     serial=self.device["serial"], timeout=10)
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0 or "Error" in output:
            raise RuntimeError(f"the headset did not open the page ({output or result.returncode}). "
                               f"Open the Browser in the headset and go to {PAGE_URL}")
        self.opened_page_at = time.time()
        return {"opened": PAGE_URL + query}

    # -- page
    def page(self) -> dict | None:
        if time.time() - self._page_at < 0.4:
            return self._page
        try:
            self._page = http_json(self.page_status_url, timeout=0.8)
        except (urllib.error.URLError, OSError, ValueError):
            self._page = None
        self._page_at = time.time()
        return self._page

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_adb()
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
            self._stop.wait(2.0)

    def close(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        page = self.page()
        with self._lock:
            device = dict(self.device) if self.device else None
            state = self.state
        return {
            "adb": self.adb_path is not None, "usb": state, "usb_message": MESSAGES.get(state, state),
            "device": device, "prepared": self.prepared_serial is not None and state == "connected",
            "error": self.last_error, "page": page, "page_url": PAGE_URL,
        }


def ready_for(page: dict | None, mode: str, mapping: str = "real") -> tuple[bool, str]:
    """Is the headset streaming what an arm run of this mode needs? mode: bimanual | left | right."""
    if not page:
        return False, "The headset page service is not running (System tab)."
    if not page.get("pages") and not page.get("streaming"):
        return False, f"Open {PAGE_URL} in the headset (Quest card: 'Open page in headset')."
    if not page.get("streaming"):
        return False, "Press 'Enter Passthrough/VR' on the headset page and hold the controllers."
    want = "bimanual" if mode == "bimanual" else f"single {mode}"
    if page.get("mode") != want:
        human = "Bimanual" if mode == "bimanual" else f"Single, {mode} hand"
        return False, f"The headset page streams '{page.get('mode')}'; set input to {human}, Apply, re-enter VR."
    hands = page.get("hands") or {}
    sides = ("left", "right") if mode == "bimanual" else (mode,)
    for side in sides:
        hand = hands.get(side)
        if not hand or not hand.get("tracked"):
            return False, f"The {side} controller is not tracked: hold it in view of the headset."
        if hand.get("mapping") and hand["mapping"] != mapping:
            return False, f"The {side} hand is set to '{hand['mapping']}'; set it to Behind / Parallel (real)."
        if float(hand.get("grip") or 0) > GRIP_RELEASED:
            return False, f"Release the {side} grip."
    return True, "Headset ready."
