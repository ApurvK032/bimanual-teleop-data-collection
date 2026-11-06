from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response


WEB_ROOT = Path(__file__).with_name("web")
LATEST_STATE_MESSAGE_TYPES = frozenset(("pose", "bimanual_pose"))


@dataclass(eq=False)
class Client:
    websocket: WebSocket
    generation: int
    pose_queue: asyncio.Queue[str] = field(default_factory=lambda: asyncio.Queue(maxsize=1))
    event_queue: asyncio.Queue[str] = field(default_factory=lambda: asyncio.Queue(maxsize=32))


class LatestStateHub:
    def __init__(self) -> None:
        self.clients: set[Client] = set()
        self.connection_generation = 0
        self.pose_messages = 0
        self.pose_messages_by_type = {message_type: 0 for message_type in LATEST_STATE_MESSAGE_TYPES}
        self.overwritten_pose_messages = 0

    async def register(self, websocket: WebSocket) -> Client:
        await websocket.accept()
        self.connection_generation += 1
        client = Client(websocket=websocket, generation=self.connection_generation)
        self.clients.add(client)
        return client

    def unregister(self, client: Client) -> None:
        self.clients.discard(client)

    def publish_pose(self, message: str, source: Client, *, message_type: str) -> None:
        if message_type not in LATEST_STATE_MESSAGE_TYPES:
            raise ValueError(f"unsupported latest-state message type: {message_type!r}")
        self.pose_messages += 1
        self.pose_messages_by_type[message_type] += 1
        for client in tuple(self.clients):
            if client is source:
                continue
            if client.pose_queue.full():
                try:
                    client.pose_queue.get_nowait()
                    self.overwritten_pose_messages += 1
                except asyncio.QueueEmpty:
                    pass
            client.pose_queue.put_nowait(message)

    def publish_event(self, message: str, source: Client | None = None) -> None:
        for client in tuple(self.clients):
            if client is source:
                continue
            if client.event_queue.full():
                try:
                    client.event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            client.event_queue.put_nowait(message)


hub = LatestStateHub()
app = FastAPI(title="WidowXAI Quest latest-state relay")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/client.js")
async def client_script() -> FileResponse:
    return FileResponse(WEB_ROOT / "client.js", media_type="text/javascript", headers={"Cache-Control": "no-store"})


@app.get("/camera_view.js")
async def camera_view_script() -> FileResponse:
    return FileResponse(
        WEB_ROOT / "camera_view.js",
        media_type="text/javascript",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/style.css")
async def stylesheet() -> FileResponse:
    return FileResponse(WEB_ROOT / "style.css", media_type="text/css", headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "clients": len(hub.clients),
            "pose_messages": hub.pose_messages,
            "pose_messages_by_type": dict(hub.pose_messages_by_type),
            "overwritten_pose_messages": hub.overwritten_pose_messages,
        }
    )


async def _sender(client: Client) -> None:
    while True:
        try:
            message = client.event_queue.get_nowait()
        except asyncio.QueueEmpty:
            pose_task = asyncio.create_task(client.pose_queue.get())
            event_task = asyncio.create_task(client.event_queue.get())
            done, pending = await asyncio.wait(
                {pose_task, event_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            message = next(iter(done)).result()
        await client.websocket.send_text(message)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    client = await hub.register(websocket)
    sender = asyncio.create_task(_sender(client))
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            message_type = payload.get("type")
            if message_type in LATEST_STATE_MESSAGE_TYPES:
                payload["relay_arrival_monotonic_ns"] = time.perf_counter_ns()
                payload["relay_arrival_epoch_ns"] = time.time_ns()
                payload["reconnect_generation"] = client.generation
                hub.publish_pose(
                    json.dumps(payload, separators=(",", ":")),
                    client,
                    message_type=message_type,
                )
            elif message_type == "clock_ping":
                response = {
                    "type": "clock_pong",
                    "sequence": payload.get("sequence"),
                    "client_send_monotonic_ms": payload.get("client_send_monotonic_ms"),
                    "client_send_epoch_ms": payload.get("client_send_epoch_ms"),
                    "pc_receive_monotonic_ns": time.perf_counter_ns(),
                    "pc_receive_epoch_ns": time.time_ns(),
                    "pc_send_monotonic_ns": time.perf_counter_ns(),
                    "pc_send_epoch_ns": time.time_ns(),
                }
                client.event_queue.put_nowait(json.dumps(response, separators=(",", ":")))
            else:
                hub.publish_event(raw, source=client)
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        hub.unregister(client)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Quest WebXR page and latest-state WebSocket relay")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8443)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
