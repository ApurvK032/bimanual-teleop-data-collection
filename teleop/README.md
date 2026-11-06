# WidowXAI + Meta Quest 3 Teleoperation

> **In this repository** the teleop code lives in this `teleop/` folder and is installed with the
> repository's `Install.sh` (see the [top-level README](../README.md)). The standalone install steps
> below are for the teleop project on its own. Added in this copy: in two-arm runs, holding **X** on
> the left controller (both grips released) brings both arms back to home.

Low-latency, clutch-relative teleoperation of a Trossen WidowXAI arm from a
Meta Quest 3 Touch Plus controller.

This project adapts Aurel Arnold's
[VR Teleoperation Stack](https://aurelarnold.xyz/blog/vr-teleoperation-stack/)
and the
[Dream Machines reference kit](https://github.com/Dream-Machines-Robotics/vr-teleop-kit)
to the WidowXAI. It adds the official robot model, Trossen driver integration,
rest/home lifecycle, task-frame calibration, physical safety checks, and
latency telemetry.

> Experimental research software: validate changes in MuJoCo first, firmly
> mount the arm, clear its workspace, and remain ready to cut controller power.

## Status

| Capability | Status |
|---|---|
| Quest 3 WebXR input over USB | Working |
| Left/right controller selection | Working |
| Right controller + Mirrored mapping | Physically accepted at 50% task scale |
| Full-stroke trigger gripper | Working |
| MuJoCo on Windows and Ubuntu | Working |
| One physical WidowXAI on Ubuntu | Working |
| Automatic rest → home → rest | Working |
| Bimanual WebXR capture, dual-arm MuJoCo, coordinated runtime | Working |
| Two-arm physical operation | Behind accepted at 40%; staged 60% response is not accepted |
| Configurable scene/left-wrist/right-wrist camera views | Operator-accepted with D455 + two D405s |
| Five-joint camera/CAD leader to one WidowXAI | Guarded software path complete; physical sign commissioning pending |

The current single-arm profile is
[`configs/quest_50pct_hardware.yaml`](configs/quest_50pct_hardware.yaml).
The validated left-controller Step-3 profile remains available as a rollback
baseline. See [`configs/README.md`](configs/README.md) before changing profiles.

The latest accepted run sustained 90.0 Hz for 51.4 seconds with zero IK
failures. Software timestamps and encoder-trajectory alignment estimate about
30 ms from captured controller pose to arm encoder response. This is not a
camera-verified hand-to-arm measurement; exact components and caveats are in
[`results/`](results/).

## How it works

```text
Quest controller
  → WebXR frame callback
  → WebSocket over USB ADB reverse
  → latest-state relay
  → clutch-relative calibrated mapping
  → filtered decoupled WidowXAI IK
  → safety checks
  → MuJoCo or Trossen driver
  → WidowXAI
```

Camera video uses a deliberately separate path: each selected V4L2 color camera
is captured and JPEG encoded by its own FFmpeg process, then exposed through a
latest-frame WebSocket service on port 8444. Camera traffic never enters the
90 Hz pose WebSocket on port 8443. The browser assigns stable camera serials to
Scene, Left wrist, and Right wrist without a code change or service restart.

Grip acts as a deadman and clutch. Pressing it anchors the current controller
and robot poses. Releasing it holds the last safe command and lets the operator
reposition their hand. Re-gripping creates a new zero-delta anchor without a
jump.

## Quick start: MuJoCo

Clone with the pinned Trossen model submodules:

```bash
git clone --recurse-submodules \
  <teleop repository>
cd vr-telop-widowxai

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

Connect the Quest by USB, accept USB debugging, and start the local relay:

```bash
bash scripts/restart_localhost.sh
```

To additionally start the configurable camera service and expose both ports
through USB ADB reverse:

```bash
bash scripts/restart_localhost.sh --with-cameras
```

The Camera Setup section lists compatible cameras attached to the PC. Assign
each device once to Scene, Left wrist, or Right wrist, enable the views wanted
in passthrough, and press **Apply camera setup**. Assignments persist by serial;
an unselected camera produces no passthrough panel. The adaptive overview shows
one, two, or three world-anchored panels high above the work area so they do not
obstruct the direct passthrough task view. With both
grips released, right-controller **B** smoothly cycles the enabled focused
views and returns to overview.

Open `http://localhost:8443/` in the Quest Browser. For the accepted profile,
select **Right + Mirrored**, press **Apply input**, and enter passthrough.

Run MuJoCo:

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_live_sim.py \
  --config configs/quest_50pct_hardware.yaml \
  --label quest-50pct-mujoco
```

Windows setup, Quest details, physical-arm prerequisites, calibration, and
troubleshooting are in the
**[setup and operation guide](docs/SETUP_AND_OPERATION.md)**.
Current milestones, measured performance, and open blockers are tracked in
**[project progress](docs/PROJECT_PROGRESS.md)**.

## Physical arm

Physical output requires Ubuntu, CPython 3.10, Trossen driver 1.9.3 (matching the
lab controllers' firmware 1.9.4 since 2026-10-20; build it with
`bash scripts/setup_legacy_hardware_env.sh`), and wired Ethernet to the arm.
Every physical acceptance so far was recorded on firmware 1.8.3 / driver 1.8.6
and must be repeated on 1.9.4 before it is relied on. Native Windows is supported only for MuJoCo.

Before enabling output:

1. verify all six translations/rotations and the gripper in MuJoCo;
2. firmly mount the arm and clear its workspace;
3. run the no-motion preflight and dry run;
4. keep grip released during the automatic rest-to-home move;
5. remain ready to cut controller power.

The full, copyable commands and lifecycle behavior are documented in
[`docs/SETUP_AND_OPERATION.md`](docs/SETUP_AND_OPERATION.md#run-one-physical-widowxai).

The independent M3T/CAD leader path now reuses the same proven Trossen backend
and rest/home/rest safety lifecycle. Its first physical profile is limited to
10% scale, ±2° about home, 15 seconds, five tracked joints, and no gripper. See
the [CAD leader commissioning guide](docs/CAD_LEADER_TELEOP.md); the direction
mapping remains pending until the next supervised lab session. A time-based
speed-adaptive joint filter smooths visual jitter after the raw packet
watchdog; it cannot conceal a stale stream, jump, or tracker restart. Pending
live mapping is single-axis-only: `--commission-joint` is mandatory and all
unselected follower joints remain at home. Live CAD motion also requires an
explicit Linux evdev hold-to-run input; the terminal `e/r` latch is available
only in MuJoCo and hardware dry runs.

The MuJoCo-only CAD profile starts and returns to all-zero joint rest. Its
viewer directly accepts `E` to engage, `R` to release/re-anchor, and `Q` to
quit; these keys no longer have to be entered in the terminal.

## Results

Large raw traces remain local under the ignored `runs/` directory. The
repository contains only compact publishable data:

- [`results/major_runs.csv`](results/major_runs.csv) — major physical runs;
- [`results/README.md`](results/README.md) — measured versus estimated latency;
- [`scripts/organize_runs.py`](scripts/organize_runs.py) — local run organizer.

## Two arms (bimanual)

Both Quest controllers are captured in one timestamped WebXR frame, each drives
its own arm through its own calibration, and both arms are solved and screened
for cross-arm collision in **one** MuJoCo scene before either command is sent.
The single-arm page, profile, and tests are unchanged and remain the rollback
path.

Select **Bimanual** on the Quest page, then run the simulation:

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_dual_sim.py \
  --config configs/dual_widowxai.yaml
```

For the measured side-by-side base layout, add `--tabletop --inline-viewer`.
It places the parallel, equal-height bases exactly 500 mm apart and adds a
collidable 1000 × 700 × 40 mm tabletop whose rear edge is 50.8 mm (2 in)
behind the base centerline:

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_dual_sim.py \
  --config configs/dual_widowxai.yaml \
  --tabletop --inline-viewer \
  --label dual-500mm-tabletop-mujoco
```

The 500 mm base separation, aligned orientation, and 50.8 mm rear inset were
measured on 2026-08-12. The table width, depth, and thickness remain simulation
assumptions; all values are printed at launch and saved in the telemetry
configuration snapshot. Override them with `--table-width-m`,
`--table-depth-m`, `--table-thickness-m`, and `--base-rear-inset-m` when the
bench changes. An arm/table contact is a coordinated collision rejection, not
just a visual overlap.

To reproduce the previously rejected load-yield oscillation without exposing
the physical arms, run the same scene with the explicit simulation-only stress
flag:

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_dual_sim.py \
  --config configs/dual_widowxai.yaml \
  --tabletop --inline-viewer --jerk-stress \
  --label dual-jerk-stress-mujoco
```

This adds a 35 ms command delay, a 30 ms first-order response, 50 Hz encoder
samples, and the historical three-sample 0.045 rad dynamic measured-pose
yield. Hold both grips and make a fast but controlled controller reversal to
stress it. If the guard activates, release both grips and re-grip to re-arm.
The mode is intentionally expected to oscillate; it is a failure-reproduction
tool, not a proposed fix. The flag is accepted only by the MuJoCo launcher and
is never enabled by the normal simulation or either hardware launcher.

No-motion preflight, including simultaneous and one-arm-moving path screening
in the combined scene:

```bash
env -u PYTHONPATH .venv/bin/python scripts/preflight_dual_hardware.py \
  --config configs/dual_widowxai.yaml
```

`scripts/run_dual_hardware.py` implements the full physical lifecycle
(lockstep rest→home ramp, per-arm grippers, coordinated fault hold, sequential
return to rest). Both Behind calibrations are physically accepted at 40%, so
`configs/dual_widowxai.yaml` satisfies the stored calibration gate without the
temporary override. Its response is currently staged at 60% translation and
60% rotation for validation. Collision recovery is capped at 0.010 rad per
joint per tick. The experimental measured-feedback load yield was removed
after one physical activation caused 1.36 s of command-direction oscillation;
it must not be restored without a delay-stable design and simulation that
models encoder lag. The 0.08 rad hard feedback stop, 30 mm clearance, accepted
reach envelope, and 10 ms command-skew gate remain unchanged. Live mode still
requires explicit current authorization, every independent safety gate, and
its own
`LIVE-WIDOWXAI-DUAL-<left-ip>-<right-ip>` token; the single-arm
`LIVE-WIDOWXAI-<ip>` token can never enable two arms. Before a hardware backend
is constructed, every live launcher now also verifies AC power, the
`performance` power profile, and a direct robot-Ethernet route sourced from
`192.168.1.10`.

Base transforms in `configs/dual_widowxai.yaml` record the measured 2026-08-12
bench geometry: 500 mm separation, no forward/vertical offset, equal height,
parallel bases, no relative yaw, and a 50.8 mm rear-edge inset. Both explicit
`base_transform.measurement_status` values are `measured`. The calibration
override remains available only for deliberate experimental profiles; it does
not convert candidate evidence into acceptance.

Architecture, staged validation, and the definition of done are in
[`docs/DUAL_ARM_EXTENSION.md`](docs/DUAL_ARM_EXTENSION.md).

## Arm-to-arm leader/follower

`scripts/leader_follower.py` is the small Trossen-driver path for using the
physical left arm (`192.168.1.2`) as the leader and the physical right arm
(`192.168.1.3`) as the follower:

```bash
env -u PYTHONPATH .venv-arm19/bin/python scripts/leader_follower.py \
  --duration 3600
```

This is Trossen's official seven-axis position/velocity-copy and 0.1-gain force
feedback pattern with the bench IPs and a one-hour default duration. It returns
both arms to home and rest on exit and requires the operator to support the
left leader when external-effort mode begins. This minimal path does not use
Quest input, the project's combined collision checker, or its detailed tracking
telemetry; use it only with separated workspaces and an operator ready to stop
the run.

## Repository map

```text
configs/                    accepted, rollback, and experimental profiles
docs/                       operating and extension guides
results/                    compact publishable milestone metrics
scripts/                    setup, simulation, calibration, and hardware tools
src/widowxai_quest_teleop/  transport, mapping, IK, safety, and telemetry
tests/                      offline unit and integration tests
third_party/                pinned official Trossen model submodules
runs/                       ignored local raw telemetry
```

`AGENTS.md` holds the arm-safety rules and known failure signatures,
known failure signatures, and recovery points for future development sessions.

## Documentation

- [Setup and operation](docs/SETUP_AND_OPERATION.md)
- [Configuration catalog](configs/README.md)
- [Latency and results](results/README.md)
- [Dual-arm extension](docs/DUAL_ARM_EXTENSION.md)
- [Camera/CAD leader commissioning](docs/CAD_LEADER_TELEOP.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Acknowledgements

- Aurel Arnold and
  [Dream Machines Robotics](https://github.com/Dream-Machines-Robotics/vr-teleop-kit)
- [Trossen Robotics](https://www.trossenrobotics.com/) for the WidowXAI model,
  driver, and documentation

Project-specific code is available under the [MIT License](LICENSE). Adapted
components and pinned submodules retain their original licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
