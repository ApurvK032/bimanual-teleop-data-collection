from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request

from websockets.sync.client import connect

from widowxai_quest_teleop.transport import BimanualQuestReceiver


def _available_loopback_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def run_smoke_test(port: int | None = None) -> dict[str, object]:
    port = _available_loopback_port() if port is None else int(port)
    http_url = f"http://127.0.0.1:{port}"
    websocket_url = f"ws://127.0.0.1:{port}/ws"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    relay = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "widowxai_quest_teleop.relay",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    receiver = None
    try:
        health = None
        for _ in range(30):
            try:
                with urllib.request.urlopen(f"{http_url}/health", timeout=0.5) as response:
                    health = json.load(response)
                break
            except OSError:
                time.sleep(0.1)
        assert health is not None and health["status"] == "ok"
        with connect(websocket_url) as subscriber:
            with connect(websocket_url) as source:
                source.send(
                    json.dumps(
                        {
                            "type": "pose",
                            "schema_version": 1,
                            "sequence": 42,
                            "capture_monotonic_ms": 100.0,
                            "capture_epoch_ms": 200.0,
                            "send_monotonic_ms": 101.0,
                            "left": {
                                "position": [0.1, 0.2, 0.3],
                                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                                "grip": 1.0,
                                "trigger": 0.0,
                            },
                        }
                    )
                )
                single = json.loads(subscriber.recv(timeout=2.0))
                source_generation = single["reconnect_generation"]

                # Exercise the actual PC receiver and its capacity-one mailbox,
                # not just a raw WebSocket subscriber. This is the integration
                # boundary the bimanual relay regression previously skipped.
                receiver = BimanualQuestReceiver(
                    websocket_url, mapping_modes={"left": "real", "right": "real"}
                )
                receiver.start()
                for _ in range(100):
                    if receiver.reconnects > 0:
                        break
                    time.sleep(0.01)
                assert receiver.reconnects > 0
                source.send(
                    json.dumps(
                        {
                            "type": "bimanual_pose",
                            "schema_version": 2,
                            "sequence": 43,
                            "capture_monotonic_ms": 110.0,
                            "capture_epoch_ms": 210.0,
                            "send_monotonic_ms": 111.0,
                            "left": {"tracked": False, "mapping_mode": "real"},
                            "right": {"tracked": False, "mapping_mode": "real"},
                        }
                    )
                )
                bimanual = json.loads(subscriber.recv(timeout=2.0))
                first_sample, mailbox_generation = receiver.mailbox.wait_take_latest(0, 2.0)
                assert first_sample is not None
                assert first_sample.sequence == 43
                assert first_sample.reconnect_generation == source_generation

            # Replacing the browser/source connection must produce a new
            # generation so both clutch anchors are invalidated together.
            with connect(websocket_url) as reconnected_source:
                reconnected_source.send(
                    json.dumps(
                        {
                            "type": "bimanual_pose",
                            "schema_version": 2,
                            "sequence": 44,
                            "capture_monotonic_ms": 120.0,
                            "capture_epoch_ms": 220.0,
                            "send_monotonic_ms": 121.0,
                            "left": {"tracked": False, "mapping_mode": "real"},
                            "right": {"tracked": False, "mapping_mode": "real"},
                        }
                    )
                )
                reconnected = json.loads(subscriber.recv(timeout=2.0))
                reconnected_sample, _ = receiver.mailbox.wait_take_latest(
                    mailbox_generation, 2.0
                )
                assert reconnected_sample is not None
                assert reconnected_sample.sequence == 44
                assert reconnected_sample.reconnect_generation > source_generation

        for received in (single, bimanual, reconnected):
            assert "relay_arrival_monotonic_ns" in received
            assert "relay_arrival_epoch_ns" in received
            assert received["reconnect_generation"] > 0
        assert single["sequence"] == 42
        assert bimanual["sequence"] == 43
        assert bimanual["reconnect_generation"] == source_generation
        assert reconnected["sequence"] == 44
        assert reconnected["reconnect_generation"] > source_generation

        with urllib.request.urlopen(f"{http_url}/health", timeout=2.0) as response:
            final_health = json.load(response)
        assert final_health["pose_messages"] == 3
        assert final_health["pose_messages_by_type"] == {"pose": 1, "bimanual_pose": 2}
        with urllib.request.urlopen(f"{http_url}/", timeout=2.0) as response:
            assert response.status == 200
        return final_health
    finally:
        if receiver is not None:
            receiver.stop()
        relay.terminate()
        relay.wait(timeout=3.0)


def main() -> None:
    health = run_smoke_test()
    print(
        "relay smoke test passed "
        f"({health['pose_messages_by_type']}, reconnect generation advanced)"
    )


if __name__ == "__main__":
    main()
