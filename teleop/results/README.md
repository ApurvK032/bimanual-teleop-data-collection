# Major run results

This directory contains the compact, publishable record of the major physical
experiments. Raw telemetry remains local under ignored `runs/` because the
complete archive is roughly a gigabyte.

[`major_runs.csv`](major_runs.csv) contains only summary statistics calculated
from the original telemetry. Run IDs match the local raw directories so an
experiment can be recovered when the archive is available.

## Latest accepted run

`20260723-171126_four-mode-50pct-continuous`:

- Right controller, Mirrored mapping, 50% task scale, full gripper;
- 4,627 rows over 51.398 seconds;
- 90.003 Hz actual command rate;
- zero IK failures;
- 0.324 ms browser capture-to-send median;
- 1.667 ms PC arrival-to-command median;
- 11.097 ms median command interval;
- 0.003 rad median maximum joint tracking error;
- approximately 27.25 ms command-to-encoder trajectory lag;
- approximately 30.0 ms estimated captured-pose-to-encoder response.

## What is measured

The following values are direct statistics from timestamps or values recorded
in one clock domain:

- browser capture-to-send;
- PC socket-arrival-to-driver-command;
- command interval/rate;
- IK failures;
- command and feedback joint values;
- feedback tracking error.

## What is estimated

Command-to-encoder lag is obtained by aligning the recorded command trajectory
with 50 Hz encoder feedback and selecting the delay with the lowest normalized
joint error. It is based on real command/encoder data but is not an event-level
latency measurement; practical uncertainty is approximately ±3–5 ms.

Captured-pose-to-encoder response is:

```text
browser capture-to-send
  + assumed 0.8 ms USB one-way transport
  + PC arrival-to-command
  + trajectory-aligned command-to-encoder lag
```

The 0.8 ms transport term is half of the reference article's 1.6 ms median USB
round trip, not a direct measurement of this setup. Quest and PC clocks are not
yet synchronized well enough to subtract their absolute timestamps directly.

True hand-motion-to-arm-motion latency additionally includes Quest
tracking/exposure and up to one 90 Hz WebXR frame phase (0–11.1 ms). Therefore
the approximately 30 ms value must be described as an estimated
controller-capture-to-encoder response, not a guaranteed physical end-to-end
maximum.
