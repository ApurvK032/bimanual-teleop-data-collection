# Setup and operation

This guide contains the detailed setup, Quest, MuJoCo, and single-arm operating
instructions. Start with the repository [README](../README.md) for the project
overview and current validated profile.

> This is experimental research software, not a safety-certified robot
> controller. Validate every mapping or control change in MuJoCo before using
> hardware. Firmly mount the arm, clear its workspace, and remain ready to cut
> controller power.

## Requirements

### Simulation

- Windows 10/11 or Ubuntu
- Python 3.10 or newer
- Meta Quest 3 with Developer Mode enabled
- Android Platform Tools (`adb`)
- a data-capable USB cable

### Physical WidowXAI

- Ubuntu 20.04, 22.04, or 24.04
- CPython 3.10
- `trossen-arm==1.9.3`
- WidowXAI controller firmware `1.9.x` (both lab arms run `1.9.4` since
  2026-10-20; all earlier physical evidence was recorded on `1.8.3` with
  driver `1.8.6` and has not yet been repeated on `1.9.4`)
- wired Ethernet access to the arm controller

The physical Trossen driver is not supported by this project on native Windows.
Use Windows for MuJoCo development and Ubuntu for physical output.

## Clone and install

Clone the repository and its pinned Trossen model submodules:

```bash
git clone --recurse-submodules \
  <teleop repository>
cd vr-telop-widowxai
git submodule update --init --recursive
```

### Ubuntu simulation environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

### Windows simulation environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

Run a Quest-free MuJoCo smoke test:

```bash
python scripts/run_sim.py --duration 5 --realtime --viewer
```

## Connect the Quest

Enable Quest Developer Mode, connect USB, wear the headset once, and accept the
USB-debugging prompt.

On Ubuntu, this command restores ADB forwarding, applies the development
mounted-state override, and starts the relay:

```bash
bash scripts/restart_localhost.sh
```

On Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_quest_usb.ps1
widowxai-quest-relay
```

The equivalent manual setup is:

```bash
adb devices -l
adb reverse tcp:8443 tcp:8443
widowxai-quest-relay
```

Open `http://localhost:8443/` in the Quest Browser, choose the controller and
mapping, press **Apply input**, and enter passthrough.

- **Behind / Parallel** follows the corresponding native task motion.
- **Mirrored** flips left/right, forward/back, screw, and nod-no while retaining
  up/down and nod-yes.

The accepted profile uses **Right + Mirrored**. The choice is stored in the
browser and locked when a physical run begins. Changing it during a run causes
a safety stop.

The mounted-state override keeps WebXR awake but cannot replace optical
tracking. If the headset is off your face, keep it upright with its cameras
facing the controller workspace.

### Configure scene and wrist operator views

Connect the cameras and use the camera-enabled restart command:

```bash
bash scripts/restart_localhost.sh --with-cameras
```

This creates two independent ADB reverse paths:

- `tcp:8443` serves the operator page and the capacity-one 90 Hz pose relay;
- `tcp:8444` serves separately encoded, latest-frame camera WebSockets.

The camera service discovers physical V4L2 cameras with a 640 × 480, 30 FPS
YUYV color node, including RealSense D405 and D455 cameras. On the first run,
stable serial order preserves the old two-camera behavior by assigning the
first camera to Scene and the second to Right wrist. After that, open the
Camera Setup section in the browser and configure:

- Scene;
- Left wrist;
- Right wrist.

Each row has a physical-camera dropdown and a **Show in passthrough** checkbox.
A device can be assigned only once. Selecting **No camera** disables that view;
unchecking a selected view hides/stops it without forgetting its assignment.
Press **Apply camera setup** before entering passthrough. The configuration is
saved by stable serial on the PC, not by changing `/dev/video*` numbers.

Serials can still be supplied during first-run or recovery startup:

```bash
bash scripts/restart_localhost.sh --with-cameras \
  --scene-serial <V4L2-ID_SERIAL_SHORT> \
  --left-wrist-serial <V4L2-ID_SERIAL_SHORT> \
  --right-wrist-serial <V4L2-ID_SERIAL_SHORT>
```

The serial used here is the V4L2/ASIC `ID_SERIAL_SHORT` printed by the camera
service, which may differ from the RealSense marketing serial printed by
`rs-enumerate-devices -s`.

The initial passthrough layout adapts to one, two, or three enabled views and
keeps them 0.18 m above the earlier layout to preserve the direct task view.
Their position and orientation are anchored
to the Quest `local-floor` world frame when passthrough starts, so head movement
does not drag the panels along with it. With multiple views enabled and both
grips released, right-controller **B** smoothly cycles overview, then each
enabled camera in Scene → Left wrist → Right wrist order. Focus mode retains
every other enabled camera as a small preview.

The ordinary page includes the same three-state cycle button for desktop
verification. A missing or stale stream shows `VIDEO LOST`; camera traffic
never shares `/ws` with controller poses. This first milestone provides RGB
only and does not yet make video freshness a physical-motion safety gate.

## Run in MuJoCo

Keep the relay running and start:

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_live_sim.py \
  --config configs/quest_50pct_hardware.yaml \
  --label quest-50pct-mujoco
```

Before hardware, verify:

1. left/right;
2. up/down;
3. forward/back;
4. screw/twist;
5. nod yes;
6. nod no;
7. clutch release and re-engagement;
8. trigger-controlled gripper behavior.

Do not proceed if any sign, axis, re-anchor, or gripper behavior is wrong.

## Run one physical WidowXAI

### Build the hardware environment

the arms (firmware 1.9.4) require Python 3.10 and driver 1.9.3. The script
builds `.venv-arm19`; the older `.venv-arm18` (driver 1.8.6) can no longer
connect, and its version gate fails before any arm contact:

```bash
bash scripts/setup_legacy_hardware_env.sh

env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv-arm19/bin/python -m pytest -q -p no:cacheprovider
```

### Configure the arm network

The tested controller is `192.168.1.2`. Give the Ubuntu Ethernet interface the
dedicated address `192.168.1.10/24`, with no gateway on that link.
If the controller uses another address, copy the configuration, update
`hardware.robot_ip`, and use the same address in the live confirmation token.

Never probe driver TCP port `50001` with `nc`, telnet, or a raw socket. This
controller can be wedged by a second client. Use `ping` only for basic network
visibility.

### Perform no-motion checks

With the arm firmly mounted and supported:

```bash
env -u PYTHONPATH .venv/bin/python scripts/check_workstation.py \
  --config configs/quest_50pct_hardware.yaml

env -u PYTHONPATH .venv-arm19/bin/python scripts/preflight_hardware.py \
  --config configs/quest_50pct_hardware.yaml
```

Test the Quest/control path without physical output:

```bash
env -u PYTHONPATH .venv-arm19/bin/python scripts/run_hardware.py \
  --config configs/quest_50pct_hardware.yaml \
  --duration 15 \
  --label quest-50pct-dry-run
```

### Run hardware

Clear the workspace, keep grip released during startup, and remain ready to cut
power:

```bash
env -u PYTHONPATH .venv-arm19/bin/python scripts/run_hardware.py \
  --config configs/quest_50pct_hardware.yaml \
  --live \
  --confirm-live LIVE-WIDOWXAI-192.168.1.2 \
  --duration 30 \
  --label right-mirror-50pct
```

The launcher:

1. verifies AC power, `performance` mode, and a direct robot route from
   `192.168.1.10` before constructing the hardware backend;
2. requires fresh Quest tracking with grip released;
3. locks the selected hand and mapping before connecting;
4. checks driver, firmware, feedback, joint limits, and the startup path;
5. moves from all-zero rest to `[0, 60, 75, -60, 0, 0]` degrees;
6. opens the gripper in a separate blocking move;
7. enables grip-controlled relative teleoperation;
8. returns the arm to all-zero rest after normal, Ctrl+C, and safety-stop exits.

The accepted profile can run without an internal deadline, but explicit short
`--duration` values are recommended during development.

## Two-arm development

The commands above operate one arm from one selected controller. Do not repeat
them in two terminals to control two arms.

The coordinated dual-arm runtime is implemented in
`scripts/run_dual_hardware.py`: one bimanual Quest packet, independent
calibration/IK/driver state, a combined collision model, coordinated hold, and
rest/home/rest lifecycle. Both Behind mappings are accepted at 40%. The active
60% response is a staged benchmark that failed tracking/skew tests and is not a
generally accepted operating scope; dual grippers and longer reliability also
remain pending.

Run the read-only workstation and combined-model checks before a dry run:

```bash
env -u PYTHONPATH .venv/bin/python scripts/check_workstation.py \
  --config configs/dual_widowxai.yaml
env -u PYTHONPATH .venv/bin/python scripts/preflight_dual_hardware.py \
  --config configs/dual_widowxai.yaml
env -u PYTHONPATH .venv-arm19/bin/python scripts/run_dual_hardware.py \
  --config configs/dual_widowxai.yaml \
  --duration 15
```

The first command is expected to fail while either arm route uses Wi-Fi. Follow
[`DUAL_ARM_EXTENSION.md`](DUAL_ARM_EXTENSION.md) for the live token, calibration
and measured-base gates, staged validation, and exact physical checklist.

## Camera/CAD leader input

The five-joint M3T leader uses the same official Trossen backend and physical
rest/home/rest lifecycle through a separate, bounded launcher. It does not use
Quest IK and it cannot control unobserved WidowX joint 5 or the gripper. Follow
the complete [CAD leader commissioning guide](CAD_LEADER_TELEOP.md) for the
stationary stream gate, exact MuJoCo profile, dry run, CAD-specific live token,
and five separate supervised ±2° physical sign tests. While signs remain
pending, every live run requires one `--commission-joint 0…4` selection and
holds all unselected follower joints at home. It also requires an explicit
Linux evdev hold-to-run device; terminal `e/r` is disabled for live CAD motion.

## Calibration

Calibration is Quest-only and never imports or contacts the robot driver:

```bash
env -u PYTHONPATH .venv/bin/python scripts/calibrate_task_frame.py \
  --config configs/quest_50pct_hardware.yaml \
  --output configs/calibrations/my_guided_6dof.json \
  --repeats 2 \
  --overwrite
```

Wear the headset, face robot-forward, and follow the six guided gestures.
Inspect the reported axis errors, add an intentionally reviewed profile to
`configs/calibrations/task_profiles.json`, and validate it in MuJoCo.

## Results and local telemetry

Raw telemetry is written under `runs/YYYY-MM-DD/` and intentionally ignored by
Git. Each run records its configuration snapshot, timestamped CSV telemetry,
and a small JSON summary.

Use:

```bash
env -u PYTHONPATH .venv/bin/python scripts/organize_runs.py
```

to organize local traces and refresh the local manifest. Publishable milestone
metrics live in [`results/major_runs.csv`](../results/major_runs.csv); latency
definitions and caveats are in [`results/README.md`](../results/README.md).

## Troubleshooting

### Waiting for fresh Quest tracking

Confirm that the relay is running, `adb reverse --list` contains port 8443,
WebXR/passthrough is open, the selected controller is visible to the headset,
and grip is released.

### Quest localhost stopped

```bash
bash scripts/restart_localhost.sh
```

### Driver hangs after TCP/UDP connection messages

Stop the launcher. Do not open another socket or repeatedly reconnect. If the
official driver handshake still hangs after the local process exits, support
the arm and power-cycle the controller.

### Native Windows hardware

Use Windows only for simulation. Run the physical driver on a supported Ubuntu
host.
