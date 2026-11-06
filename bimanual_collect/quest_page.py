"""Quest page + pose relay (port 8443) for data collection, built on the teleop repo's relay.

At start-up this copies the teleop web page into <state>/web, applies the patches below, and serves
it on localhost (the Quest reaches it through `adb reverse tcp:8443 tcp:8443`). The relay forwards
pose packets to the teleop launcher, which handles arm motion.

Patches (each must match the repo's text exactly once, or start-up stops and says which one failed):
  1. camera panels keep the camera's native pixels (the repo draws every feed into 640x480);
  2. the panel header/placeholder text is scaled to match, plus a red REC badge while recording;
  3. right-controller A starts/stops an episode (only with both grips released); after a stop, a 6 s prompt
    takes A = success / B = failure (B's camera cycling waits until the prompt ends);
  4. bimanual pose packets carry the left-controller home button;
  5. index.html loads web/collect_hd.js (recording controls on the page and in the headset).

Also added here (the relay itself is unchanged; its hub methods are wrapped to observe traffic):
  GET /quest/status   is the headset page open, which mode it streams, is each controller tracked, grips, age
  GET /collect/arms   the arm-run line for the headset header, fetched from the collection app (:8440)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

from fastapi.responses import FileResponse
from widowxai_quest_teleop import relay

from . import rig

HERE = Path(__file__).resolve().parent
REPO_WEB = relay.WEB_ROOT
LOCAL_WEB = rig.load().state / "web"

PATCHES = {
    "client.js": [
        (
            "home button in bimanual poses",
            '    type: "bimanual_pose",\n',
            '    type: "bimanual_pose",\n'
            "    home_button: window.collectHomeButton?.(inputSources),\n",
        ),
    ],
    "camera_view.js": [
        (
            "native-resolution canvas",
            "    drawFrame(bitmap) {\n      const width = this.canvas.width;\n",
            "    drawFrame(bitmap) {\n"
            "      // collect: keep the camera's native pixels (largest 4:3 region, no downscaling).\n"
            "      const nativeWidth = Math.min(bitmap.width, Math.round(bitmap.height * 4 / 3));\n"
            "      const nativeHeight = Math.round(nativeWidth * 3 / 4);\n"
            "      if (this.canvas.width !== nativeWidth || this.canvas.height !== nativeHeight) {\n"
            "        this.canvas.width = nativeWidth;\n"
            "        this.canvas.height = nativeHeight;\n"
            "      }\n"
            "      const width = this.canvas.width;\n",
        ),
        (
            "header scale start",
            "    drawHeader() {\n      const context = this.context;\n",
            "    drawHeader() {\n      const context = this.context;\n"
            "      // collect: lay the header out on a 640-wide grid, scaled to the native canvas.\n"
            "      const uiScale = this.canvas.width / 640;\n"
            "      const uiWidth = 640;\n"
            "      context.save();\n"
            "      context.scale(uiScale, uiScale);\n",
        ),
        ("header bar width", "context.fillRect(0, 0, this.canvas.width, 76);", "context.fillRect(0, 0, uiWidth, 76);"),
        ("header dot", "context.arc(this.canvas.width - 124, 29, 7, 0, Math.PI * 2);",
         "context.arc(uiWidth - 124, 29, 7, 0, Math.PI * 2);"),
        (
            "header end + REC badge",
            '      context.fillText(stale ? "STALE" : "LIVE", this.canvas.width - 108, 36);\n    }\n',
            '      context.fillText(stale ? "STALE" : "LIVE", uiWidth - 108, 36);\n'
            "      if (window.collectDrawRecordBadge) window.collectDrawRecordBadge(context, uiWidth);\n"
            "      context.restore();\n    }\n",
        ),
        (
            "placeholder scale",
            '      const context = this.context;\n      context.fillStyle = "#0b1721";\n'
            "      context.fillRect(0, 0, this.canvas.width, this.canvas.height);\n"
            '      context.strokeStyle = "#38536a";\n      context.lineWidth = 4;\n'
            "      roundedRect(context, 15, 15, this.canvas.width - 30, this.canvas.height - 30, 22);\n",
            "      const context = this.context;\n"
            "      // collect: 640-wide layout grid scaled to the native canvas.\n"
            "      const uiScale = this.canvas.width / 640;\n"
            "      const uiWidth = 640;\n"
            "      const uiHeight = this.canvas.height / uiScale;\n"
            "      context.save();\n"
            "      context.scale(uiScale, uiScale);\n"
            '      context.fillStyle = "#0b1721";\n'
            "      context.fillRect(0, 0, uiWidth, uiHeight);\n"
            '      context.strokeStyle = "#38536a";\n      context.lineWidth = 4;\n'
            "      roundedRect(context, 15, 15, uiWidth - 30, uiHeight - 30, 22);\n",
        ),
        ("placeholder title", "this.canvas.width / 2, 215);", "uiWidth / 2, 215);"),
        ("placeholder message", "this.canvas.width / 2, 268);", "uiWidth / 2, 268);"),
        (
            "placeholder end",
            '      context.textAlign = "start";\n      this.commitFrame(notify);\n',
            '      context.textAlign = "start";\n      context.restore();\n      this.commitFrame(notify);\n',
        ),
        (
            "A button records (and takes A/B during the success prompt)",
            "    handleControllerButtons(inputSources) {\n",
            "    handleControllerButtons(inputSources) {\n"
            "      if (window.collectHandleRecordButton && window.collectHandleRecordButton(inputSources)) {\n"
            "        // collect: the post-episode prompt is using B; don't also cycle the camera view.\n"
            "        this.previousBPressed = buttonValue(inputSources.right?.gamepad, QUEST_B_BUTTON) >= 0.7;\n"
            "        return;\n"
            "      }\n",
        ),
    ],
    "index.html": [
        (
            "load collect_hd.js",
            '  <script src="/camera_view.js"></script>\n',
            '  <script src="/camera_view.js"></script>\n  <script src="/collect_hd.js"></script>\n',
        ),
    ],
}


class PatchError(RuntimeError):
    pass


def patch_text(name: str, text: str) -> str:
    for label, old, new in PATCHES.get(name, []):
        count = text.count(old)
        if count != 1:
            raise PatchError(
                f"{name}: patch '{label}' matched {count} times (expected 1). The teleop repo's page changed; "
                "update PATCHES in bimanual_collect/quest_page.py."
            )
        text = text.replace(old, new)
    return text


def build_web(target: Path = LOCAL_WEB) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for source in REPO_WEB.iterdir():
        if source.is_file():
            (target / source.name).write_text(patch_text(source.name, source.read_text(encoding="utf-8")),
                                              encoding="utf-8")
    shutil.copy2(HERE / "web" / "collect_hd.js", target / "collect_hd.js")
    return target


class HeadsetWatch:
    """Observes the relay's clients and pose packets (read-only; forwarding is untouched)."""

    def __init__(self) -> None:
        self.clients: dict[object, dict] = {}
        self.last: dict | None = None
        self.arrivals: deque[float] = deque(maxlen=400)

    def install(self, hub) -> None:
        register, unregister, publish_pose = hub.register, hub.unregister, hub.publish_pose

        async def watched_register(websocket):
            client = await register(websocket)
            agent = websocket.headers.get("user-agent", "")
            self.clients[client] = {"agent": agent, "since": time.time(),
                                    "page": not agent.lower().startswith("python")}
            return client

        def watched_unregister(client):
            self.clients.pop(client, None)
            return unregister(client)

        def watched_publish(message, source, *, message_type):
            try:
                self.observe(message, source)
            except (ValueError, TypeError, KeyError, AttributeError):
                pass
            return publish_pose(message, source, message_type=message_type)

        hub.register, hub.unregister, hub.publish_pose = watched_register, watched_unregister, watched_publish

    def observe(self, message: str, source) -> None:
        packet = json.loads(message)
        now = time.monotonic()
        self.arrivals.append(now)

        def hand(data) -> dict | None:
            if not isinstance(data, dict):
                return None
            return {"tracked": bool(data.get("tracked", True)), "grip": round(float(data.get("grip") or 0.0), 2),
                    "mapping": data.get("mapping_mode")}

        if packet.get("type") == "bimanual_pose":
            mode, hands = "bimanual", {side: hand(packet.get(side)) for side in ("left", "right")}
        else:
            selected = packet.get("selected_hand")
            mode = f"single {selected}"
            hands = {side: (hand(packet.get(side)) if side == selected else None) for side in ("left", "right")}
            if hands.get(selected):
                hands[selected]["mapping"] = packet.get("mapping_mode")
        self.last = {"at": now, "mode": mode, "sequence": packet.get("sequence"), "hands": hands,
                     "agent": self.clients.get(source, {}).get("agent", "")}

    def status(self) -> dict:
        now = time.monotonic()
        pages = [{"agent": c["agent"], "headset": any(k in c["agent"] for k in ("Quest", "OculusBrowser")),
                  "connected_s": round(time.time() - c["since"])} for c in self.clients.values() if c["page"]]
        last = self.last
        age = None if last is None else now - last["at"]
        streaming = age is not None and age < 0.5
        rate = sum(1 for t in self.arrivals if now - t < 1.0)
        return {
            "pages": pages, "listeners": sum(1 for c in self.clients.values() if not c["page"]),
            "streaming": streaming, "age_s": None if age is None else round(age, 3), "rate_hz": rate,
            "mode": None if last is None else last["mode"], "sequence": None if last is None else last["sequence"],
            "hands": None if last is None else last["hands"],
            "from_headset": bool(last and any(k in last["agent"] for k in ("Quest", "OculusBrowser"))),
        }


APP_URL = "http://127.0.0.1:8440"


def arms_line() -> dict:
    try:
        with urllib.request.urlopen(f"{APP_URL}/api/arms/brief", timeout=0.5) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError):
        return {"line": "", "level": "none"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8443)
    args = parser.parse_args()
    web = build_web()
    relay.WEB_ROOT = web  # the relay's page routes read this module global per request

    watch = HeadsetWatch()
    watch.install(relay.hub)

    @relay.app.get("/quest/status")
    async def quest_status() -> dict:
        return watch.status()

    @relay.app.get("/collect/arms")
    async def collect_arms() -> dict:
        return await asyncio.to_thread(arms_line)

    @relay.app.get("/collect_hd.js")
    async def collect_script() -> FileResponse:
        return FileResponse(web / "collect_hd.js", media_type="text/javascript", headers={"Cache-Control": "no-store"})

    import uvicorn

    print(f"Quest page (repo page + collect patches from {web}) on http://localhost:{args.port}", flush=True)
    uvicorn.run(relay.app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
