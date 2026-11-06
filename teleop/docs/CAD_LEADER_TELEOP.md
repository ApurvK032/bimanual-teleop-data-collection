# Camera-tracked leader to physical WidowXAI

This path reuses the proven WidowXAI driver, firmware checks, rest/home
lifecycle, command shaping, feedback supervision, and return-to-rest behavior
from the Quest project. Its input is the five-joint UDP stream from the M3T CAD
tracker instead of a Quest controller pose.

It is currently a **guarded physical commissioning path**, not an accepted
full-workspace teleoperator. The software side is complete and tested offline;
the five signs still require isolated operator validation at the bench. While
the mapping is pending, live output refuses to start unless exactly one
follower joint is selected with `--commission-joint 0` through `4`.

## What is controlled

| M3T source | WidowXAI follower | First-run behavior |
|---|---|---|
| `link1_link` | `joint_0` | candidate sign `+`, 10% scale |
| `link2_link` | `joint_1` | candidate sign `+`, 10% scale |
| `link3_link` | `joint_2` | candidate sign `-`, 10% scale |
| `link4_link` | `joint_3` | candidate sign `-`, 10% scale |
| `link5_link` | `joint_4` | candidate sign `-`, 10% scale |
| not tracked | `joint_5` | fixed at home |
| not tracked | gripper | mode and command disabled |

The follower starts at all-zero rest, ramps through the proven two-second move
to `[0, 60, 75, -60, 0, 0]` degrees, and anchors the current visual leader pose
to that home pose when the hold-to-run control is pressed. Therefore a new camera
calibration or a different visual zero does not create an absolute robot jump.

The commissioning envelope is only ±2° around home for joints 0–4. The run is
capped at 15 seconds. Expanding either limit requires a separate accepted
profile after physical sign and smoothness validation; do not weaken this file
in place.

## Safety behavior

Physical output requires all of the following:

- `root_locked=true` in every accepted M3T packet;
- exact five-joint names, finite values, recent timestamps, and monotonic
  sequences;
- three continuous fresh packets before recovery;
- no source change above 0.12 rad per frame;
- raw-packet jump/freshness checks followed by a time-based, speed-adaptive
  circular joint filter that updates once per unique M3T packet;
- a Linux evdev key, foot pedal, or gamepad button whose physical state is
  queried on every control iteration;
- an explicit input-device path on every live command, with the input released
  both before preflight and after the automatic home ramp;
- a CAD-specific live token that cannot enable the Quest launcher;
- an extra first-run override while the sign mapping is marked pending;
- a mandatory isolated follower joint selection while the mapping is pending;
- the arm measured within 0.08 rad of all-zero rest before position mode;
- official model/controller limits, self-collision checks, velocity,
  acceleration, and per-tick gates;
- 50 Hz time-aligned encoder checks with the unchanged 0.08 rad hard stop;
- return to all-zero rest on normal exit, Ctrl+C, or a safety exception.
- AC power, the `performance` power profile, and a direct route to the robot
  sourced from the dedicated `192.168.1.10` Ethernet address.

Releasing the physical control immediately holds the last safe command. Loss
of camera packets, an unlocked root, a tracker restart, or a source jump also
holds and requires a physical release followed by a fresh press to create a
new zero-delta anchor. Failure or disconnection of the evdev device records a
deadman fault, sends one held command, stops the process, and returns to rest.
The terminal `e`/`r` latch is disabled in live mode; it remains only in MuJoCo
and dry runs. The code never chases delayed encoder feedback when re-anchoring.

## Before the first session

The hardware environment is CPython 3.10 with `trossen-arm==1.9.3` in
`.venv-arm19`; both controllers run firmware 1.9.4 since 2026-10-20. The CAD
path was developed on firmware 1.8.3 and has not been re-run on 1.9.4. The
follower is normally `192.168.1.2` and the PC
robot-LAN address is `192.168.1.10/24`, with no gateway. Never probe controller
port 50001 using `nc`, telnet, or a raw socket.

Run the offline suite:

```bash
cd "<teleop folder>"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv-arm19/bin/python -m pytest -q -p no:cacheprovider
```

## Lab commissioning sequence

Use AC power and the performance power profile. Keep arm Ethernet separate from
the camera hub if possible; the shared dock previously reset Ethernet and
camera devices together under load. The live launcher now checks these
workstation conditions and the direct robot route before constructing the
Trossen backend. It will intentionally reject the current Wi-Fi route until the
dedicated robot Ethernet interface is restored.

Select a stable `/dev/input/by-id/...-event-*` path for a USB foot pedal,
gamepad button, or keyboard key. A foot pedal or a second operator is preferred
so the person moving the printed leader never has to let go of it. The default
Linux key code is 57 (`KEY_SPACE`). List candidates and qualify a complete
release/press/release cycle without importing the robot driver:

```bash
env -u PYTHONPATH .venv/bin/python scripts/check_hold_to_run.py --list
export CAD_HOLD_DEVICE=/dev/input/by-id/<your-device>-event-kbd
env -u PYTHONPATH .venv/bin/python scripts/check_hold_to_run.py \
  --device "$CAD_HOLD_DEVICE" \
  --key-code 57
```

Do not continue unless the checker prints `RESULT PASS`. Do not run the robot
launcher with `sudo`; correct the evdev device permissions instead. The live
launcher rejects a missing, unsupported, inaccessible, or already-held input
before workstation checks or robot construction.

Check the workstation without importing the arm driver or contacting a robot:

```bash
env -u PYTHONPATH .venv/bin/python scripts/check_workstation.py \
  --config configs/cad_hardware_commissioning.yaml
```

1. Firmly mount the follower, clear both arm workspaces, keep the controller
   power cutoff in reach, and put the follower at all-zero rest.
2. Recalibrate the camera session and launch the M3T overlay from the CAD
   repository. For the present ZED/D455 placement:

   ```bash
   cd <CAD-pose_detection folder>
   ./experiments/m3t_leader_arm_assembly/run_joints_d455_mat.sh
   ```

   Apply the detector with `d`, start tracking with `t`, wait for a correct
   overlay, then press `l`. The ZED currently validates the mat seed; live M3T
   evidence still comes from D455 RGB-D.
3. With the printed leader completely stationary, validate five seconds of the
   exact stream. This command never imports or contacts the robot driver:

   ```bash
   cd "<teleop folder>"
   env -u PYTHONPATH .venv/bin/python scripts/check_cad_stream.py \
     --config configs/cad_hardware_commissioning.yaml \
     --duration 5
   ```

   Do not continue unless it prints `"passed": true`.
4. Stop the stream checker. Only one process can bind UDP port 5055. Preview the
   mapping in MuJoCo from the all-zero joint pose and validate each source joint
   direction separately. The legacy config filename is retained for command
   compatibility, but its simulation rest pose is `[0,0,0,0,0,0]`. Start with
   follower joint 0:

   ```bash
   env -u PYTHONPATH .venv/bin/python scripts/run_cad_sim.py \
     --config configs/cad_home_commissioning_mujoco.yaml \
     --commission-joint 0 \
     --label cad-j0-mujoco
   ```

   Click the MuJoCo window and press `E` to engage, `R` to hold/re-anchor, and
   `Q` to stop. The terminal alternatives are `e`, `r`, or `q` followed by
   Enter. Repeat separate MuJoCo runs with `--commission-joint 1`, `2`, `3`,
   and `4`. Even if every source estimate moves, only the selected follower
   joint may leave zero rest. Joint 0 has a deliberate 0.15 rad source
   deadband, so rotate the printed leader joint more than about 8.6° before
   expecting simulated motion.
5. Perform the official-driver read-only check. It configures a driver session
   but does not enable position mode or send a command:

   ```bash
   env -u PYTHONPATH .venv-arm19/bin/python scripts/preflight_hardware.py \
     --config configs/cad_hardware_commissioning.yaml
   ```
6. Exercise the complete hardware lifecycle against the in-memory backend:

   ```bash
   env -u PYTHONPATH .venv-arm19/bin/python scripts/run_cad_hardware.py \
     --config configs/cad_hardware_commissioning.yaml \
     --commission-joint 0 \
     --duration 10 \
     --label cad-j0-hardware-dry-run
   ```
7. Only after explicit same-session authorization, run the physical sign
   commissioning:

   ```bash
   env -u PYTHONPATH .venv-arm19/bin/python scripts/run_cad_hardware.py \
     --config configs/cad_hardware_commissioning.yaml \
     --live \
     --confirm-live LIVE-WIDOWXAI-CAD-192.168.1.2 \
     --accept-unvalidated-mapping \
     --commission-joint 0 \
     --deadman-device "$CAD_HOLD_DEVICE" \
     --deadman-key-code 57 \
     --duration 15 \
     --label cad-j0-sign-physical-01
   ```

   Keep the physical control released during the automatic rest-to-home ramp.
   After `home reached`, press and continuously hold it while moving only the
   printed-leader joint mapped to the selected follower joint, slowly in both
   directions. Release the control to hold; then type `q` to return the arm to
   all-zero rest. Start a new complete process with joint 1 and a matching
   label, then repeat for joints 2, 3, and 4. The pending profile rejects
   `--live` if `--commission-joint` is omitted, even when the mapping override
   and live token are present. Release the physical control immediately if any
   direction, sound, smoothness, or overlay behavior is wrong.

## What remains before full teleoperation

The bridge is fail-closed, but these evidence gaps prevent calling the complete
system “perfect” today:

1. The five candidate direction signs need operator acceptance from five
   separate single-joint hardware runs.
2. M3T supplies no per-joint confidence score. A smooth visual drift can pass a
   freshness/step watchdog, so full-range or unattended output is not allowed.
3. The current live tracker uses D455 RGB-D after a ZED/D455 mat check; ZED is
   not yet a second live M3T modality.
4. Wrist joint 5 and the gripper are not observed. They must stay fixed until
   the leader model/stream is extended.
5. Stationary noise, latency, occlusion recovery, and deliberate-motion traces
   must be measured at the final camera distance before increasing the ±2°
   envelope or 10% scale.

After a clean first physical run, inspect the exact telemetry under
`runs/YYYY-MM-DD/`. Promote the mapping status only after the operator confirms
all five signs and the trace shows no stale episode, discontinuity, limiter
fault, deadman fault, feedback fault, or abnormal send time. The trace records
the deadman source, device, key code, state-query age, and press/release
generations.
