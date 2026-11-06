"""Stand-in for the Quest: stream two scripted controllers to the teleop relay (no headset needed).

    .venv/bin/python tests/scripted_controllers.py --seconds 20

Sends bimanual_pose packets (schema 2, the Quest page's format) to ws://127.0.0.1:8443/ws at 72 Hz: both grips
held (deadman engaged after a short release), hands tracing slow circles a few centimetres across, triggers opening
and closing the grippers. Use it with teleop/scripts/run_dual_sim.py to exercise the whole pipeline (recording,
sync check, alignment, export) without a headset or arms. Never point a live hardware launcher at it.
"""
from __future__ import annotations

import argparse
import json
import math
import time

from websockets.sync.client import connect


def controller(t: float, side: int, grip: float, radius_m: float) -> dict:
    phase = 2 * math.pi * 0.25 * t + (0 if side < 0 else math.pi / 2)
    return {
        "tracked": True,
        "mapping_mode": "real",
        "position": [0.25 * side + radius_m * math.cos(phase), 1.05 + 0.5 * radius_m * math.sin(2 * phase),
                     -0.35 + radius_m * math.sin(phase)],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "grip": grip,
        "trigger": 0.5 + 0.5 * math.sin(2 * math.pi * 0.2 * t + side),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="ws://127.0.0.1:8443/ws")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--rate-hz", type=float, default=72.0)
    parser.add_argument("--radius-m", type=float, default=0.04)
    parser.add_argument("--release-s", type=float, default=1.0,
                        help="grips released for this long first (launchers wait for released grips before starting)")
    args = parser.parse_args()

    period = 1.0 / args.rate_hz
    with connect(args.url) as socket:
        start = time.monotonic()
        sequence = 0
        while (now := time.monotonic()) - start < args.seconds:
            t = now - start
            grip = 0.0 if t < args.release_s else 1.0  # released first, then held: a clean clutch engage
            now_ms = time.monotonic() * 1000
            packet = {
                "type": "bimanual_pose", "schema_version": 2, "input_mode": "bimanual", "sequence": sequence,
                "capture_monotonic_ms": now_ms, "capture_epoch_ms": time.time() * 1000,
                "enqueue_monotonic_ms": now_ms, "send_monotonic_ms": now_ms,
                "left": controller(t, -1, grip, args.radius_m), "right": controller(t, 1, grip, args.radius_m),
                "head": None, "operator_head": None,
            }
            socket.send(json.dumps(packet))
            sequence += 1
            time.sleep(max(0.0, start + (sequence * period) - time.monotonic()))
    print(f"sent {sequence} packets over {args.seconds:.0f} s")


if __name__ == "__main__":
    main()
