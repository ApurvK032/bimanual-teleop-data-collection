from __future__ import annotations

import argparse
import time
from pathlib import Path

from widowxai_quest_teleop.hold_to_run import (
    HoldToRunError,
    LinuxEvdevHoldToRun,
)


def candidate_devices() -> list[str]:
    stable = sorted(str(path) for path in Path("/dev/input/by-id").glob("*-event-*"))
    events = sorted(str(path) for path in Path("/dev/input").glob("event*"))
    return stable or events


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only release/press/release qualification for the CAD physical "
            "hold-to-run input; this never imports or opens the robot driver"
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--device", help="Linux evdev event device path")
    selection.add_argument(
        "--list",
        action="store_true",
        help="list candidate stable event-device paths and exit",
    )
    parser.add_argument(
        "--key-code",
        type=int,
        default=57,
        help="Linux EV_KEY code to test (default: 57, KEY_SPACE)",
    )
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()

    if args.list:
        devices = candidate_devices()
        if not devices:
            raise SystemExit("no Linux evdev devices were found")
        print("Candidate hold-to-run event devices:")
        for device in devices:
            print(f"  {device}")
        return

    if not 0.0 < args.duration <= 60.0:
        raise SystemExit("--duration must be within (0, 60] seconds")

    control = LinuxEvdevHoldToRun(args.device, args.key_code)
    try:
        initial = control.open(require_released=True)
        print("PHYSICAL OUTPUT DISABLED: no robot driver is imported or opened")
        print(
            f"hold-to-run ready device={control.device} key_code={control.key_code}; "
            "press and hold once, then release"
        )
        previous = initial.pressed
        deadline = time.perf_counter() + args.duration
        while time.perf_counter() < deadline:
            sample = control.sample()
            if sample.pressed != previous:
                print("PRESSED" if sample.pressed else "RELEASED")
                previous = sample.pressed
            if sample.press_generation >= 1 and sample.release_generation >= 1:
                print("RESULT PASS: observed a fresh physical press and release")
                return
            time.sleep(0.01)
    except KeyboardInterrupt:
        raise SystemExit("RESULT STOPPED: qualification interrupted") from None
    except HoldToRunError as exc:
        raise SystemExit(f"RESULT FAIL: {exc}") from None
    finally:
        control.close()
    raise SystemExit(
        "RESULT FAIL: no complete release/press/release cycle was observed before timeout"
    )


if __name__ == "__main__":
    main()
