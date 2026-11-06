"""Stand-in for the teleop arm launchers in tests: prints their phase lines, writes a telemetry.csv, and handles
Ctrl+C (SIGINT) the way they do: return to rest, then exit 0. Never touches hardware.

    python fake_launcher.py <runs_root> <label> <mode> [live_seconds]

modes: normal | safety (safety stop while live, exit 1) | emergency (rest fails on stop) | hang (ignores the stop)
       | contact (read-only preflight output) | crash (traceback before connecting)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def say(text: str) -> None:
    print(text, flush=True)


def telemetry(runs_root: Path, label: str) -> Path:
    folder = runs_root / time.strftime("%Y-%m-%d") / f"{time.strftime('%Y%m%d-%H%M%S')}_{label}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def main() -> int:
    runs_root, label, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    live_s = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    if mode == "contact":
        say("=== offline dual-arm checks ===")
        say("offline dual-arm checks PASSED")
        say("live dual-arm gates: SATISFIED")
        say("=== read-only controller state (no position mode, no commands) ===")
        for side, ip in (("left", "192.0.2.3"), ("right", "192.0.2.2")):
            say(f"{side:>5} arm @ {ip}")
            say("        driver=1.9.3 firmware=1.9.4")
            say("        q_rad=[-0.066, -0.0, 0.003, -0.001, 0.02, 0.012]")
            say("        gripper_m=0.002400")
        say("NO-MOTION DUAL PREFLIGHT PASSED for both controllers")
        return 0
    say("=== offline dual-arm checks ===")
    say("offline dual-arm checks PASSED")
    say("preflight: workstation AC=mains-no-battery profile=performance robot_source=192.0.2.10 devices=['eth1']")
    say("preflight: waiting for fresh tracking on BOTH controllers with grips released")
    time.sleep(0.2)
    if mode == "crash":
        raise RuntimeError("simulated crash before connecting")
    say(" left arm @ 192.0.2.3 driver=1.9.3 firmware=1.9.4 q=[0, 0, 0, 0, 0, 0]")
    say("preflight: right arm @ 192.0.2.2 driver=1.9.3 firmware=1.9.4 q=[0, 0, 0, 0, 0, 0]")
    say("startup: ramping BOTH arms from measured pose to home in lockstep")
    time.sleep(0.2)
    say("home reached on both arms: keep BOTH grips released")
    folder = telemetry(runs_root, label)
    header = ["pc_epoch_ns", "pc_monotonic_ns", "control_state"]
    for side in ("left", "right"):
        header += [f"{side}_tracked", f"{side}_grip", f"{side}_clutch_engaged", f"{side}_q_feedback",
                   f"{side}_feedback_sample_fresh", f"{side}_q_feedback_error", f"{side}_ik_status", f"{side}_held"]
    csv_file = open(folder / "telemetry.csv", "w", buffering=1)
    csv_file.write(",".join(header) + "\n")
    say(f"LIVE BIMANUAL: 90 Hz, 25 ms driver horizon, left<-left controller, right<-right controller, "
        f"each grip is that arm's deadman, {live_s:.0f} s maximum")
    stop_reason = "time"
    try:
        start = time.monotonic()
        tick = 0
        while time.monotonic() - start < live_s:
            row = [str(time.time_ns()), str(time.perf_counter_ns()), "normal"]
            for _side in ("left", "right"):
                error = json.dumps([0.001 * (tick % 7), 0.002, 0.0, 0.0, 0.0, 0.0])
                row += ["True", "1.0", "True", "\"[0,0,0,0,0,0]\"", "True" if tick % 2 else "False",
                        '"' + error.replace('"', '""') + '"', "ok", "False"]
            csv_file.write(",".join(row) + "\n")
            tick += 1
            if mode == "safety" and tick == 40:
                print("SAFETY STOP: left joint 3 tracking error 0.130 rad exceeds 0.120 rad", file=sys.stderr, flush=True)
                stop_reason = "safety"
                break
            time.sleep(1 / 90)
        if stop_reason == "time":
            say(f"telemetry: {folder}")
    except KeyboardInterrupt:
        stop_reason = "operator"
        say("operator stop: returning both arms to rest before driver cleanup")
        if mode == "hang":
            while True:  # a launcher stuck in its shutdown (only the power switch helps)
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    pass
    finally:
        csv_file.close()
    for side in ("left", "right"):
        say(f"shutdown: returning {side} arm to rest with a 2 s blocking move")
        time.sleep(0.1)
        if mode == "emergency" and stop_reason == "operator":
            print(f"EMERGENCY SHUTDOWN WARNING: {side} arm did not reach rest; cut controller power now",
                  file=sys.stderr, flush=True)
        else:
            say(f"rest reached: {side} arm")
    say("shutdown summary: left[connected=True moved=True rest=True], right[connected=True moved=True rest=True]")
    return 1 if stop_reason == "safety" else 0


if __name__ == "__main__":
    sys.exit(main())
