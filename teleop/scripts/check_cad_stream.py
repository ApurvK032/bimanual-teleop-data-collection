from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from widowxai_quest_teleop.cad_input import (
    CadFreshnessWatchdog,
    CadJointSample,
    CadUdpReceiver,
    wrapped_angle_delta,
)
from widowxai_quest_teleop.config import load_config


def summarize_cad_samples(
    samples: list[CadJointSample],
    *,
    duration_s: float,
    source_deadband_rad: np.ndarray,
    discontinuities: int,
    invalid_packets: int,
    final_fresh: bool,
) -> dict:
    if not samples:
        return {
            "passed": False,
            "failures": ["no valid locked-root samples"],
            "samples": 0,
        }
    q = np.asarray([sample.q for sample in samples], dtype=float)
    sequences = np.asarray([sample.sequence for sample in samples], dtype=np.int64)
    source_age_ms = np.asarray(
        [sample.arrival_epoch_ns - sample.source_time_ns for sample in samples],
        dtype=float,
    ) / 1e6
    median_q = np.median(q, axis=0)
    deviation = np.abs(
        np.asarray([wrapped_angle_delta(row, median_q) for row in q], dtype=float)
    )
    deviation_p95 = np.percentile(deviation, 95.0, axis=0)
    deviation_max = np.max(deviation, axis=0)
    if q.shape[0] > 1:
        steps = np.abs(
            np.asarray(
                [wrapped_angle_delta(q[index], q[index - 1]) for index in range(1, q.shape[0])]
            )
        )
        step_p95 = np.percentile(steps, 95.0, axis=0)
        step_max = np.max(steps, axis=0)
        sequence_gaps = int(np.sum(np.maximum(0, np.diff(sequences) - 1)))
    else:
        step_p95 = np.zeros(5)
        step_max = np.zeros(5)
        sequence_gaps = 0

    effective_rate_hz = len(samples) / max(float(duration_s), 1e-9)
    deadband = np.asarray(source_deadband_rad, dtype=float).reshape(5)
    failures: list[str] = []
    if effective_rate_hz < 20.0:
        failures.append(f"valid source rate {effective_rate_hz:.1f} Hz is below 20 Hz")
    if invalid_packets:
        failures.append(f"receiver rejected {invalid_packets} packet(s)")
    if discontinuities:
        failures.append(f"watchdog observed {discontinuities} discontinuity event(s)")
    if not final_fresh:
        failures.append("stream was not fresh at the end of the check")
    if sequence_gaps:
        failures.append(f"local UDP stream skipped {sequence_gaps} sequence value(s)")
    noisy = np.flatnonzero(deviation_p95 > deadband + 1e-12)
    if noisy.size:
        failures.append(
            "stationary p95 deviation exceeds the configured deadband for joint(s) "
            + ",".join(str(int(index)) for index in noisy)
        )

    return {
        "passed": not failures,
        "failures": failures,
        "samples": len(samples),
        "effective_rate_hz": effective_rate_hz,
        "first_sequence": int(sequences[0]),
        "last_sequence": int(sequences[-1]),
        "sequence_gaps": sequence_gaps,
        "invalid_packets": int(invalid_packets),
        "discontinuities": int(discontinuities),
        "final_fresh": bool(final_fresh),
        "source_age_ms_p50": float(np.percentile(source_age_ms, 50.0)),
        "source_age_ms_p95": float(np.percentile(source_age_ms, 95.0)),
        "source_age_ms_max": float(np.max(source_age_ms)),
        "median_q_rad": median_q.tolist(),
        "stationary_deviation_p95_rad": deviation_p95.tolist(),
        "stationary_deviation_max_rad": deviation_max.tolist(),
        "step_p95_rad": step_p95.tolist(),
        "step_max_rad": step_max.tolist(),
        "configured_deadband_rad": deadband.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a stationary locked-root M3T stream without contacting an arm"
    )
    parser.add_argument("--config", default="configs/cad_hardware_commissioning.yaml")
    parser.add_argument("--udp-host")
    parser.add_argument("--udp-port", type=int)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 1.0:
        raise SystemExit("--duration must be finite and at least one second")

    config = load_config(args.config)
    cad = config["cad"]
    receiver = CadUdpReceiver(
        args.udp_host or cad["udp_host"],
        cad["udp_port"] if args.udp_port is None else args.udp_port,
        expected_names=cad["expected_names"],
        max_packet_age_s=cad["max_packet_age_s"],
        max_future_skew_s=cad["max_future_skew_s"],
        require_root_locked=True,
    )
    watchdog = CadFreshnessWatchdog(
        cad["stale_timeout_s"],
        cad["fresh_samples_to_recover"],
        np.asarray(cad["max_source_step_rad"], dtype=float),
    )
    samples: list[CadJointSample] = []
    started = time.perf_counter()
    final_status = watchdog.poll()
    try:
        receiver.start()
        print(
            f"Keep the physical leader stationary for {args.duration:g} s; "
            f"checking udp://{receiver.host}:{receiver.port}"
        )
        while time.perf_counter() - started < args.duration:
            sample, _generation = receiver.mailbox.take_latest()
            now_ns = time.perf_counter_ns()
            if sample is None:
                final_status = watchdog.poll(now_ns)
            else:
                samples.append(sample)
                final_status = watchdog.observe(sample, now_ns)
            time.sleep(0.002)
    finally:
        receiver.stop()

    report = summarize_cad_samples(
        samples,
        duration_s=args.duration,
        source_deadband_rad=np.asarray(cad["source_deadband_rad"], dtype=float),
        discontinuities=watchdog.discontinuities,
        invalid_packets=receiver.invalid_packets,
        final_fresh=final_status.fresh,
    )
    report["receiver_packets"] = receiver.received_packets
    report["receiver_valid_packets"] = receiver.valid_packets
    report["receiver_last_error"] = receiver.last_error
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"report: {args.output}")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
