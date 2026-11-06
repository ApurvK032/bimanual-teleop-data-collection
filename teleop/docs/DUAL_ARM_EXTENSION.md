# Dual-arm extension

This document defines a safe path to two arms without duplicating two
single-arm processes and hoping they remain synchronized.

> **Current status: implemented; Behind mapping accepted at 40% with response
> staged at 60% translation / 60% rotation and bounded collision recovery for
> validation. Front-facing Mirrored tracks cleanly at 45%, but its newly
> re-checked front/back direction is rejected pending a one-axis sign correction.
> The unstable dynamic measured-feedback load yield is removed.**
>
> The bimanual transport, dual-arm configuration schema, combined MuJoCo scene,
> per-arm runtime, coordinated controller, dual telemetry, and all three
> launchers exist and are exercised by the offline suite. The measured bench
> has completed a simultaneous physical Behind run with correct response in
> every direction. Remaining work is validation depth, not architecture:
>
> 1. both controllers passed the separately authorized read-only preflight;
> 2. Left/Behind and Right/Behind are physically accepted at 40%, so
>    `require_live_dual_arm_config` accepts the normal profile without the
>    unvalidated-calibration override;
> 3. correct only the Mirrored front/back translation sign and pass a
>    Quest-driven MuJoCo check before another physical Mirrored run;
> 4. dual grippers and longer-duration reliability remain to be validated.
>
> Base geometry is no longer a blocker. On 2026-08-12 the owner measured the
> bases exactly 500 mm apart, aligned in X and Z, at equal height, parallel with
> no relative yaw, and both 50.8 mm (2 in) forward of the table's rear edge.
>
> Live output additionally requires the dual token
> `LIVE-WIDOWXAI-DUAL-<left-ip>-<right-ip>`.

## Implemented components

| Concern | Where |
|---|---|
| Bimanual WebXR capture (schema v2) | `src/widowxai_quest_teleop/web/client.js`, `index.html` |
| Bimanual sample types | `types.py` — `ControllerSample`, `BimanualQuestSample` |
| Packet parsing and one-mailbox receiver | `transport.py` — `parse_bimanual_pose_message`, `BimanualQuestReceiver` |
| Per-arm config with fail-closed rules | `config.py` — `parse_dual_arm_config`, `require_live_dual_arm_config` |
| Combined two-arm scene and collision | `dual_arm_model.py` — `DualArmCollisionModel` |
| Reusable per-arm pipeline | `arm_runtime.py` — `ArmRuntime` |
| Consume-once / solve-both / screen / send | `dual_arm_coordinator.py` — `DualArmCoordinator` |
| Dual telemetry with skew and fault state | `telemetry.py` — `DUAL_ARM_TELEMETRY_COLUMNS` |
| Launchers | `scripts/run_dual_sim.py`, `preflight_dual_hardware.py`, `run_dual_hardware.py` |
| Profile | `configs/dual_widowxai.yaml` |
| Tests | `tests/test_dual_arm_transport.py`, `tests/test_dual_arm_safety.py` |

### Notes recorded during implementation

The two arms each own a `WidowXAIModel`. `DecoupledIK` uses the model's
`MjData` as scratch between `fk` and `jacobian`, so two solvers sharing one
model would silently corrupt each other's Jacobians.

The combined scene is built with `MjSpec.attach(child, prefix=..., frame=...)`
from two copies of `wxai_follower.xml`. The gripper-carriage self-collision
whitelist had to become prefix-aware; the unprefixed literal would stop
matching and report every closed gripper as a self-collision.

`mj_geomDistance` in mujoco 3.8.1 returns exactly `0.0` for some box-box pairs
once `distmax` exceeds the true separation — reproduced on this model with two
link boxes 0.31 m apart. The clearance check keeps its search cutoff just above
the configured margin and raises every result to a bounding-sphere lower bound,
which is provable and can only under-report clearance. Contact detection via
`mj_forward` is unaffected and remains the primary gate.

## What exists today

The validated runtime controls one arm from one selected Quest controller. The
Quest page discovers both controllers, but captures and transmits only the hand
chosen in the **Hand** menu. The hardware process then locks that hand and its
Behind/Mirrored mapping for the entire run.

The general accepted physical baseline is **Right + Mirrored** at 50% task
scale. Right + Behind is additionally accepted on physical arm `.3` at 40%
with full gripper control. Neither result automatically accepts the left-hand
calibration or simultaneous dual-arm motion.

Do not remove the current hand/mapping controls from the validated single-arm
mode. A dual-arm mode must instead:

- obtain left and right controller poses from the same WebXR frame;
- transmit both poses and both button states in one packet;
- map the left controller to the configured left arm and the right controller
  to the configured right arm;
- use a separately measured task-frame calibration for each arm; and
- retain the existing single-arm page and profile as a rollback path.

Behind/Mirrored is a single-arm operator-view convention, not a substitute for
calibrating two arm bases. Do not apply one global mirror switch to both arms.

## Target architecture

```text
Quest left + right controllers
  -> one timestamped bimanual WebXR packet
  -> latest-state relay
  -> shared bimanual safety coordinator
       -> left calibration/filter/clutch/IK -> left arm driver
       -> right calibration/filter/clutch/IK -> right arm driver
  -> dual-arm MuJoCo collision model and telemetry
```

This follows the reference kit's bimanual principle—both controllers produce
one coordinated action—while retaining the WidowXAI-specific lifecycle and
safety gates developed in this repository.

Each arm needs independent state:

- controller hand, grip, trigger, and wrist-pivot calibration;
- Quest-to-arm-base task transform;
- clutch anchor and pose filter;
- WidowXAI model instance and IK state;
- arm IP, driver connection, limits, feedback history, and gripper state;
- rest/home poses and lifecycle completion.

The two arms share:

- one WebXR frame timestamp and reconnect generation;
- a single latest-state mailbox;
- cross-arm collision checking;
- startup/shutdown coordination;
- a fault policy that can hold or return both arms safely;
- one telemetry record per control tick.

## Developer starting point

1. Clone the repository with submodules and build the normal simulation
   environment using [`SETUP_AND_OPERATION.md`](SETUP_AND_OPERATION.md).
2. Run the complete offline test suite.
3. Run the accepted single-arm profile in MuJoCo and save its result as a
   regression baseline.
4. Create a dedicated development branch for the dual-arm work.
5. Keep every physical backend disabled until the dual-arm MuJoCo and fault
   tests below pass.

The single-arm implementation is the reference behavior. Refactor reusable
pieces out of it; do not replace it with an untested two-arm-only launcher.

## Hardware and network prerequisites

Before physical development, obtain:

- two firmly mounted WidowXAI follower arms;
- two controller IP addresses that are unique on the same dedicated LAN;
- an Ubuntu workstation that can reach both arms over wired Ethernet;
- confirmed compatible driver/firmware pairs for both controllers;
- measured transforms from a shared world frame to each arm base;
- a dual-arm MuJoCo scene matching those base transforms;
- independent power isolation or an immediately reachable shared power cutoff;
- a Quest 3 and both tracked Touch Plus controllers.

Do not copy the IP addresses below into hardware. Discover and document the
actual lab configuration without raw-probing TCP port `50001`.

## Required changes

### 1. Bimanual transport

Change the WebXR page from a selected-hand pose to one packet containing both
controller poses and button states. Preserve sequence numbers, capture/send
timestamps, reconnect handling, and capacity-one/latest-state behavior.

Do not merge two independently timed WebSocket streams at the driver layer.
One frame-level packet makes controller alignment and stale detection explicit.

Recommended packet shape:

```json
{
  "type": "bimanual_pose",
  "schema_version": 2,
  "sequence": 123,
  "capture_monotonic_ms": 456.7,
  "left": {
    "position": [0, 0, 0],
    "orientation_xyzw": [0, 0, 0, 1],
    "grip": 0,
    "trigger": 0
  },
  "right": {
    "position": [0, 0, 0],
    "orientation_xyzw": [0, 0, 0, 1],
    "grip": 0,
    "trigger": 0
  }
}
```

Reject the entire packet if its shared timestamps or sequence are invalid.
Represent per-controller optical loss explicitly instead of silently reusing an
old pose.

### 2. Configuration schema

Replace single `quest` and `hardware` blocks with explicit arms:

```yaml
project:
  mode: dual_widowxai

quest:
  mode: bimanual
  websocket_url: ws://127.0.0.1:8443/ws

arms:
  left:
    controller_hand: left
    robot_ip: LEFT_ARM_IP
    calibration: configs/calibrations/dual_left_accepted.json
    base_transform:
      measurement_status: measured
      position_m: [MEASURED_X, MEASURED_Y, MEASURED_Z]
      quaternion_wxyz: [MEASURED_W, MEASURED_X, MEASURED_Y, MEASURED_Z]
  right:
    controller_hand: right
    robot_ip: RIGHT_ARM_IP
    calibration: configs/calibrations/dual_right_accepted.json
    base_transform:
      measurement_status: measured
      position_m: [MEASURED_X, MEASURED_Y, MEASURED_Z]
      quaternion_wxyz: [MEASURED_W, MEASURED_X, MEASURED_Y, MEASURED_Z]

safety:
  cross_arm_collision: true
  coordinated_fault_hold: true
```

Keep global transport, cadence, telemetry, and coordinated safety policy
separate from per-arm settings. Do not copy the example IPs without checking
the actual controllers.

Fail configuration loading when:

- both arms use the same IP;
- both arms reference the same logical controller hand;
- either calibration is missing or not explicitly accepted;
- base transforms are missing or not explicitly recorded as measured;
- cross-arm collision checking is disabled for live output.

### 3. Dual-arm MuJoCo model

Create one scene containing both arms at their measured base transforms.
Validate:

- each arm's self-collision;
- left-arm versus right-arm collision;
- tool/gripper clearance;
- rest-to-home and home-to-rest paths for both arms;
- one-arm motion while the other arm holds.

Two independent single-arm MuJoCo instances cannot detect cross-arm collisions.

### 4. Coordinated control loop

Consume each Quest sequence once, solve both arms, collision-check the combined
state, and only then send commands. A practical first policy is:

- independent grips engage each arm;
- loss of one controller holds only that arm;
- loss of the shared WebXR/relay connection holds both;
- driver or collision failure prevents new motion on both;
- shutdown returns arms sequentially if simultaneous paths are not proven safe.

The correct send ordering and acceptable skew should be measured. Do not claim
synchronization merely because two driver calls occur in the same Python loop.

### 5. Hardware abstraction

Refactor the current launcher into:

- a reusable per-arm runtime object;
- a bimanual coordinator owning two driver backends;
- one explicit live confirmation covering both controller IPs;
- cleanup that records which arms connected, entered position mode, moved,
  held, and reached rest.

Never raw-probe either controller's TCP port. Connect each official driver only
after both Quest profiles and both complete startup paths pass offline checks.

## Code map

| Area | Current file | Dual-arm work |
|---|---|---|
| WebXR capture | `src/widowxai_quest_teleop/web/client.js` | Capture both `gripSpace` poses in one frame and emit schema v2 |
| Sample types | `src/widowxai_quest_teleop/types.py` | Add a bimanual sample with optional validity per hand |
| Parsing/mailbox | `src/widowxai_quest_teleop/transport.py` | Parse both hands into one capacity-one sample |
| Configuration | `src/widowxai_quest_teleop/config.py` | Validate global plus per-arm blocks |
| Mapping/IK | `mapping.py`, `ik.py`, `model.py` | Instantiate isolated state per arm |
| Hardware lifecycle | `scripts/run_hardware.py`, `hardware.py` | Extract reusable per-arm runtime and add a coordinator |
| Simulation | `scripts/run_live_sim.py` | Add a scene containing both arms and cross-collision checks |
| Telemetry | `telemetry.py` | Record both targets, commands, feedback, skew, and fault state per tick |
| Acceptance | `tests/` | Add bimanual transport, clutch, collision, lifecycle, and failure tests |

Do not weaken the current single-arm tests while extracting reusable classes.
Add dual-arm tests alongside them.

## Entry points

Separate entry points; the single-arm commands are unchanged.

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_dual_sim.py \
  --config configs/dual_widowxai.yaml
```

Add `--tabletop --inline-viewer` for the measured 500 mm side-by-side base
layout. The default rear inset is the measured 50.8 mm (2 in); the default
1000 × 700 × 40 mm table dimensions remain simulation assumptions. All four
values have command-line overrides and are recorded in the run snapshot. Table
contacts participate in the combined collision verdict.

No-motion offline checks. Add `--contact-arms` to additionally open a
read-only driver session to each controller without enabling position mode:

```bash
env -u PYTHONPATH .venv-arm19/bin/python scripts/preflight_dual_hardware.py \
  --config configs/dual_widowxai.yaml
```

The live launcher requires a confirmation token containing both verified arm
IPs. The single-arm `LIVE-WIDOWXAI-<IP>` token never enables two arms:

```bash
# Only with explicit operator authorization, both calibrations accepted,
# and measured base transforms in the profile.
env -u PYTHONPATH .venv-arm19/bin/python scripts/run_dual_hardware.py \
  --config configs/dual_widowxai.yaml --live \
  --confirm-live LIVE-WIDOWXAI-DUAL-<left-ip>-<right-ip>
```

Omitting `--live` runs the complete lifecycle against two dry-run backends.

## Validation sequence

1. Unit-test bimanual packet parsing, stale handling, and per-arm clutching.
2. Replay synthetic left-only, right-only, simultaneous, and reconnect cases.
3. Validate all twelve signed controller motions in the dual-arm MuJoCo scene.
4. Validate cross-arm collision rejection and coordinated shutdown in MuJoCo.
5. Run one physical arm through the new coordinator while the second backend is
   dry-run.
6. Swap sides and repeat.
7. Connect both arms with grips released and perform read-only state checks.
8. Run short, low-scale, one-arm-at-a-time motion.
9. Only then test slow simultaneous motion.

Record command skew, per-arm encoder response, IK failures, collision margins,
stale/reanchor events, and shutdown completion in every stage.

For the twelve motion checks, validate these independently for each controller:

1. left/right;
2. up/down;
3. forward/back;
4. screw/twist;
5. nod yes;
6. nod no.

Then validate independent grip engagement, simultaneous engagement, each
gripper trigger, one-controller tracking loss, shared relay loss, one-driver
failure, and Ctrl+C during motion.

## Definition of done

Dual-arm support is ready for routine laboratory testing only when:

- both controller poses originate from one timestamped WebXR frame;
- each controller drives only its assigned arm through an accepted calibration;
- all twelve signed motions pass in the combined MuJoCo scene;
- self-collision and cross-arm collision paths fail closed;
- one-arm and shared tracking failures produce the documented holds;
- partial driver connection and mid-run driver loss cannot leave either arm in
  uncontrolled motion;
- startup and shutdown complete safely from every tested intermediate state;
- command skew and latency are recorded rather than assumed;
- the existing accepted one-arm profile still passes its full regression suite;
- the README is updated from “planned” only after physical acceptance evidence
  is recorded.

## Reuse versus replacement

Reusable without major redesign:

- relative pose mapping;
- per-arm filters and decoupled IK;
- task-frame calibration format;
- Trossen backend and version checks;
- telemetry primitives;
- per-arm rest/home lifecycle.

Must be redesigned:

- the selected-hand WebXR payload;
- the single-arm configuration schema;
- the single-arm launcher and safety state machine;
- MuJoCo scene/model ownership;
- collision screening and coordinated shutdown.

## Prohibited shortcuts

- Do not launch two copies of `scripts/run_hardware.py`.
- Do not give two processes independent views of the same Quest stream.
- Do not use two separate MuJoCo models for physical collision approval.
- Do not reuse the Right + Mirrored calibration for the second arm.
- Do not treat the current Hand/Mirrored UI as a bimanual controller selector.
- Do not connect either physical backend until both profiles and both startup
  paths pass the coordinated offline preflight.
