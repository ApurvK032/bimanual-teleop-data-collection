from __future__ import annotations

import argparse
import statistics
import time

from widowxai_quest_teleop.transport import QuestReceiver


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure unique Quest sample arrival timing; no robot output")
    parser.add_argument("--url", default="ws://127.0.0.1:8443/ws")
    parser.add_argument("--hand", choices=("auto", "left", "right"), default="auto")
    parser.add_argument(
        "--mapping-mode",
        choices=("auto", "real", "mirror"),
        default="auto",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    receiver = QuestReceiver(
        args.url,
        hand=None if args.hand == "auto" else args.hand,
        mapping_mode=None if args.mapping_mode == "auto" else args.mapping_mode,
    )
    receiver.start()
    arrivals: list[int] = []
    sequences: list[int] = []
    selections: set[tuple[str, str]] = set()
    generation = 0
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < args.duration:
            sample, generation = receiver.mailbox.wait_for_newer(generation, 0.5)
            if sample is None:
                continue
            arrivals.append(sample.pc_arrival_monotonic_ns)
            sequences.append(sample.sequence)
            selections.add((sample.hand, sample.mapping_mode))
    finally:
        receiver.stop()
    if len(arrivals) < 2:
        raise SystemExit("fewer than two Quest samples received")
    periods_ms = [(b - a) / 1e6 for a, b in zip(arrivals, arrivals[1:])]
    drops = sum(max(0, b - a - 1) for a, b in zip(sequences, sequences[1:]))
    print(f"unique samples: {len(arrivals)} ({len(arrivals) / args.duration:.1f} Hz)")
    print(f"inter-arrival ms median/p95/max: {statistics.median(periods_ms):.3f} / {percentile(periods_ms, 0.95):.3f} / {max(periods_ms):.3f}")
    print(f"sequence gaps: {drops}")
    print(
        "input selections: "
        + ", ".join(f"{hand}/{mode}" for hand, mode in sorted(selections))
    )
    print(f"mailbox overwrites: {receiver.mailbox.overwrite_count}")
    print(f"reconnects: {receiver.reconnects}; malformed messages: {receiver.bad_messages}")


if __name__ == "__main__":
    main()
