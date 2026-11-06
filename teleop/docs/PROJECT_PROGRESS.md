# Project progress and measured performance

Last updated: 2026-10-23

This is the short operational ledger. `AGENTS.md` holds the arm-safety rules,
safety rules, failure analysis, and exact milestone chronology. Raw run evidence
stays under `runs/`; compact publishable metrics stay under `results/`.

## Current system state

| Area | Status | Best evidence / performance | What it does not prove |
|---|---|---|---|
| Arm firmware / driver | Upgraded; right-arm solo re-checked | Both controllers flashed 1.8.3 → 1.9.4 on 2026-10-20; stored settings unchanged except a new zero `position_offset` per joint; project moved to driver 1.9.3 in `.venv-arm19`. Right arm `.3` solo at 40% Behind with full gripper ran the full 45 s (4,050 rows at 90.00 Hz, zero IK failures, no input pauses) on 2026-10-23, and the owner reports it works | Right-hand wrist calibration and roll/pitch mapping are not yet tuned; every other physical result below was measured on firmware 1.8.3 / driver 1.8.6 and not yet repeated |
| Single-arm Quest teleoperation | Accepted baseline | Right controller + Mirrored at 50% task scale; latest accepted run 4,627 rows over 51.398 s at 90.003 Hz with zero IK failures | The other three hand/mapping combinations are not accepted at this scale |
| Single-arm response | Accepted as smooth and responsive | Approximately 30 ms estimated controller-capture-to-encoder response from software/driver telemetry | Not a synchronized high-speed-camera measurement |
| Bimanual Quest transport | Physically exercised and direction-accepted | Schema-v2 packets drove both arms together for 3,786 rows at 89.874 Hz overall; the operator confirmed correct response in every direction | Transport evidence does not validate Mirrored/front-facing mode or dual grippers |
| Dual feedback telemetry | Physically exercised | Truthful per-arm feedback showed left/right joint-2 maxima of 0.007212/0.004884 rad with no fault in a 44.996 s simultaneous run | One clean run is not a long-duration reliability study |
| Dual MuJoCo tabletop | Operator-accepted | 10,211 rows over 114.878 s at 88.877 Hz; zero IK failures; 218 deliberate cross-arm clearance rejections held both arms exactly | Does not validate drivers, encoders, measurement tolerances, or physical collision distance |
| Dual-arm bench geometry | Measured | Bases exactly 500 mm apart, no X/Z offset, equal height, parallel with no yaw difference; both centers 50.8 mm (2 in) from the rear edge | Overall table width/depth/thickness and physical collision clearance remain unmeasured |
| Dual-arm controllers | Coordinated physical pass; Behind directions accepted | Both use driver 1.8.6/firmware 1.8.3; they completed a simultaneous 44.996 s run with 4.336 ms maximum send skew and correct operator-observed response in every direction | Dual grippers, post-run rest confirmation, and longer reliability remain pending |
| Camera/CAD leader to one WidowXAI | Guarded software integration complete | MuJoCo starts at all-zero rest with direct viewer E/R/Q control; five named M3T joints pass through raw locked-root/timestamp/jump watchdogs, per-packet adaptive circular smoothing, relative anchoring, mandatory pending-mapping single-joint isolation, kernel-polled physical hold-to-run, 10% scaling, ±2° physical home envelope, shaping, 50 Hz time-aligned feedback, telemetry, workstation/network preflight, and rest/home/rest hardware lifecycle | Five separate physical sign runs, physical deadman qualification, real visual drift/occlusion behavior, live ZED fusion, joint 5, and gripper are pending lab evidence |
| Left-arm home cycle | Physical pass | `.2` alone completed rest → home → rest; maximum home error 0.009119 rad, final rest error 0.008202 rad, modeled separation 60 mm against 30 mm margin | Does not validate signed teleoperation, gripper motion, or coordinated dual motion |
| Right-arm home cycle | Physical pass | `.3` alone completed rest → home → rest; 0.359 mm resting model artifact cleared by 0.2% without deepening; maximum home error 0.008738 rad, final rest error 0.012016 rad | Does not validate signed teleoperation, gripper motion, or coordinated dual motion |
| Right/Behind mapping on arm `.3` | Operator-accepted at 40% | 4,043 rows/44.987 s at 89.849 Hz; 3,705 active rows, full 0.442–40.036 mm physical gripper range, zero IK failures/arm limiter flags; max joint tracking error 0.010144 rad; operator confirmed correct directions and intended behavior | Does not accept 45%/50% gains, the left-hand mapping, or simultaneous dual motion |
| First simultaneous physical dual run | Technical and signed-direction pass | 4,045 rows/44.996 s at 89.874 Hz; both arms active together for 3,786 rows; zero IK/limiter/fault/collision-rejection rows; max tracking error 0.010517 rad; send skew 0.754 ms p95 / 4.336 ms max; operator confirmed both arms worked perfectly in every direction | Triggers were intentionally unused; shutdown was not separately confirmed |
| Behind 60% response | Physical tracking failure; not accepted | Requested 120 s was capped to 60 s, but the run stopped at 20.681 s when left wrist joint 4 reached 0.081099 rad tracking error against the unchanged 0.08 stop; right simultaneously reached 0.075754 rad; both arms returned to rest | Do not raise the threshold or duration cap; isolate rotation gain from translation gain before retrying |
| Behind 60%/60% follow-up | Command-skew stop; not accepted | 1,905 rows/21.617 s; terminal measured 10.432 ms sequential send skew against the 10 ms gate; no collision or feedback fault in saved rows; both arms returned to rest | Faulting skew is not saved because the exception precedes telemetry logging; instrument per-arm driver-send duration before tuning |
| Behind 60% translation / 45% rotation marker-cap attempt | External-load release tracking stop; not accepted | 3,577 rows/40.008 s at 89.381 Hz; 1,215 both-engaged rows; zero IK failures; cap release produced encoder jumps with smooth outgoing commands, then left joint 4 reached 0.082283 rad error; collision hold had recovered earlier | Keep the 0.08 hard stop. Any softer response must avoid continuously chasing delayed measured positions; right gripper rest was not confirmed on shutdown |
| 60%/60% measured-feedback load-yield experiment | Physically rejected and removed | The 5,325-row run triggered once at 0.052184 rad, then spent 125 rows/1.360 s chasing delayed encoder positions. Every joint reversed command direction 16–32 times; peak error reached 0.072888 rad left and 0.057501 rad right. The operator reported both arms shaking strongly before settling | Dynamic measured-pose yield is removed from config and live control. Bounded collision recovery and per-driver send timing remain; hard 0.08 rad stop, 30 mm clearance, and 10 ms skew gate are unchanged |
| Front-facing Mirrored dual run | Technically clean at 45%; front/back direction rejected | Latest full run: 4,049 rows/44.991 s at 89.996 Hz, 2,766 both-engaged rows, zero IK/fault/hold/collision rows, 0.007872/0.013752 rad left/right maximum error, and 1.121 ms maximum send skew. Assignment, lateral/vertical motion, and rotations felt correct | Both calibrations use position signs `[-1,-1,+1]`; telemetry confirms hand-forward drives robot `−X` with −0.93/−0.95 correlation. Test candidate `[+1,-1,+1]` in MuJoCo before physical reuse |
| Three-camera Quest operator view | Operator-accepted | One D455 plus two D405 feeds at 640×480×30 FPS; measured rates 29.97, 30.02, and 30.00 FPS | Glass-to-glass latency and three-stream impact on active WebXR pose rate remain unmeasured |
| Camera UI | Operator-accepted | Persistent serial-based Scene/Left wrist/Right wrist assignment; adaptive 1/2/3-panel world anchor; B-button focus cycle; 380 ms transitions; common +0.18 m vertical offset | Camera loss currently warns but does not force robot hold |
| Offline regression suite | Passing after the driver 1.9.3 migration | 293 passed + 1 expected skip in `.venv`, 294 passed in `.venv-arm19` (including an offline check of the real 1.9.3 driver API); CAD filtering, raw-watchdog separation, wrap/reset behavior, live workstation route checks, pending-mapping single-joint isolation, kernel-polled hold-to-run faults, zero-rest simulation, viewer keys, and existing safety paths are covered | Offline tests do not accept CAD directions, physically qualify the selected deadman device, repair USB topology, or make another tight-object release safe |

## Camera milestone details

The accepted workstation exposes one RealSense D455 and two D405s. All three
negotiate USB SuperSpeed (5 Gb/s). At the final acceptance snapshot:

| Role | V4L2 serial | Measured FPS | Latest JPEG size |
|---|---|---:|---:|
| Scene | `208223060917` | 29.970 | 20,147 bytes |
| Left wrist | `323743070189` | 30.019 | 28,926 bytes |
| Right wrist | `323743071937` | 30.004 | 28,253 bytes |

The instantaneous encoded payload represented by those frame sizes is roughly
2.3 MB/s total before WebSocket overhead. Earlier Quest ADB testing sustained
approximately 47 MB/s, so raw USB forwarding capacity is not the current
bottleneck. This comparison is not a video-latency measurement.

Accepted behavior:

- each logical role selects any compatible free camera from the browser;
- one physical serial cannot be assigned to multiple roles;
- assignments persist on the PC and can change without editing code;
- an unselected or unchecked role does not appear in passthrough;
- overview adapts to one, two, or three enabled panels;
- B cycles overview and each enabled focus view while grips are released;
- non-primary feeds remain visible as previews;
- panels remain fixed in the Quest `local-floor` world frame;
- every layout uses a 380 ms eased transition and a common +0.18 m height
  adjustment, accepted as no longer obstructing the direct work area.

Known camera limitations:

- RGB freshness is display-only and is not yet a physical-motion safety input;
- glass-to-glass latency has not been measured;
- active WebXR pose rate with all three streams has not been recorded in a
  controlled before/after trace;
- the shared USB-C dock has reset once, simultaneously removing Ethernet and
  the hub-connected D405s. It recovered, but dock power/cabling should be
  stabilized before physical dual-arm operation.

## Dual-arm work completed

1. Bimanual WebXR packets and the latest-state relay were corrected and tested.
2. Dual per-arm runtime, synchronized command proposal, combined collision
   rejection, and dual telemetry were implemented.
3. Feedback timestamps/reference states were made truthful so a future joint-2
   fault remains diagnosable.
4. Placeholder base geometry was made a fail-closed live-output blocker, then
   replaced with the measured 2026-08-12 bench transforms.
5. A provisional 500 mm tabletop MuJoCo scene was created with table collision.
6. A simulation-only bimanual calibration was captured for the left controller
   and explicitly derived for the right controller.
7. The operator accepted the dual MuJoCo mapping and controllability.
8. Deliberate simulated arm approaches confirmed coordinated collision hold.
9. The PC/arm address conflict was removed and both controllers are now visible
   as distinct Ethernet devices.
10. The exact 500 mm aligned base layout and 50.8 mm rear inset were recorded
    in all dual profiles; the old 100 mm simulated rear-inset assumption was
    removed while overall table dimensions remain explicitly provisional.
11. No-contact preflight against that measured geometry passed simultaneous
    rest/home, gripper endpoint, and both one-arm-moving path screens. Modeled
    home/home clearance is 60 mm against the configured 30 mm margin.
12. The separately authorized official-driver read-only preflight passed for
    both controllers. Position mode remained disabled, no command was sent,
    and both sessions closed normally.
13. With explicit motion authorization and the physical checklist confirmed,
    the left arm alone completed rest → home → rest. The right controller
    remained read-only and neither gripper was placed in position mode.
14. With separate authorization, the right arm completed the same isolated
    cycle. Its shallow near-zero model artifact was bounded to 2 mm, measured
    at 0.359 mm, proven non-deepening, and cleared by 0.2% of the outbound path.
15. The dual profile's exact Right/Behind candidate was exercised on arm `.3`
    at 0.20 translation/rotation scale with arm `.2` read-only. The run was
    technically clean; operator confirmation of the six signed directions is
    still required before changing calibration acceptance evidence.
16. The separate 0.40 translation/rotation profile passed a guarded 44.987 s
    physical run on arm `.3` with full trigger-controlled gripper motion. It had
    zero IK failures, no arm limiter flags, maximum joint tracking error
    0.010144 rad, and physical gripper travel from 0.442 to 40.036 mm. The live
    stationary-arm guard accepted every command. The operator subsequently
    accepted all directions, orientation response, gripper feel, and shutdown.
17. That verdict is preserved in a separate accepted calibration document. The
    dual profile's right side now uses the exact accepted 0.40 / 0.070 m /
    0.16 rad scope; the untested provisional 0.45 settings were not promoted.
18. The first dual-hardware command exposed a direct-script import-path bug and
    stopped before connecting to either arm. The launcher now supports the
    documented clean-`PYTHONPATH` command, with a subprocess regression test;
    no physical motion occurred during the failed invocation.
19. With an explicit left-calibration override, both arms then completed the
    first simultaneous coordinated run at 40%. Joint-2 tracking remained
    healthy on both sides, send skew stayed below its gate, and no collision or
    control fault occurred. The owner clarified the 45 s duration was
    intentional and that triggers were deliberately unused.
20. The operator subsequently confirmed that both arms worked perfectly in
    every direction. Left/Behind is therefore accepted at the exact 40% scope
    used by the run, alongside the already accepted Right/Behind mapping. The
    accepted file preserves the tested matrix without changing any limit. Dual
    grippers and Mirrored/front-facing mode remain separate validation items.
21. The front-facing Mirrored profile then completed a 44.990 s physical pass
    at 45%. The right controller correctly drove `.2`, the left controller
    correctly drove `.3`, and the operator reported perfect behavior. The run
    had zero faults, holds, IK failures, collision rows, or rejected proposals.
    Left/Mirror is now accepted at that exact scope; both triggers remained
    unused, so dual grippers remain separate.
22. A Behind marker-task run at 60% exercised both triggers and reached the
    collision margin 75 times; the coordinated hold recovered normally. At
    20.681 s, both wrist joint-4 errors rose and the left crossed the unchanged
    0.08 rad feedback stop. The reference was fresh/interpolated, so this is a
    real 60%-scope tracking problem rather than the historical stale-reference
    hypothesis. Both arms returned to rest.
23. The 60% translation / 45% rotation marker-cap run proved that reducing
    rotation gain alone does not address a tight-object release. The video and
    telemetry align: commands remained smooth while both measured arms recoiled,
    after which left joint 4 crossed the hard tracking gate.
24. The first dynamic load-yield response was physically rejected. It followed
    each new delayed encoder sample at up to 0.006 rad/tick, causing repeated
    target reversals for 1.36 s. That path is removed. A future solution must
    use a latched target or a fail-closed soft stop rather than a live
    command-to-measurement chasing loop.
25. The rejected controller is now available only through
    `run_dual_sim.py --jerk-stress`, together with a 35 ms command-delay / 30 ms
    first-order encoder model and 50 Hz feedback. Focused tests reproduce its
    moving-target direction reversal and prove the normal builder leaves it
    disabled. This is a failure-reproduction harness, not a hardware fix, and
    a Quest-driven manual stress run is still pending.

## Remaining validation work

1. Confirm normal return-to-rest after the next run; the completed run's
   shutdown was not separately reported.
2. Exercise both triggers in a short separated-workspace run if simultaneous
   dual gripper behavior is required; this run left both grippers open.
3. The historical joint-2 fault did not recur in this 45 s run. Repeat under a
   longer representative task before calling it closed; keep the 0.08 rad stop.
4. Repeat dual motion with both grippers in separated work volumes. Do not
   deliberately collide the physical arms.
5. Add a camera-loss policy before unattended remote teleoperation. The safest
   default is to force a clutch release/hold when every operator-critical view
   is stale.
6. Measure three-camera glass-to-glass latency and Quest pose-rate impact during
   an active bimanual WebXR session.
7. Create a Mirrored calibration candidate that changes only the front/back
   translation sign from `-1` to `+1`, then verify all axes in Quest-driven
   MuJoCo before another physical Mirrored run.
8. Fix the bimanual wrist-calibration arming UX and add in-VR calibration,
   clutch, and reach-limit status.
9. For the camera/CAD leader, run the stationary stream qualifier and exact
   home-relative MuJoCo profile, then perform the 15-second isolated five-axis
   physical sign commissioning. Keep joint 5 and gripper disabled and do not
   expand the 10%/±2° profile until that evidence is accepted.

## GitHub issue audit

All four issues in `ApurvK032/aurel-vr-teleop-widowxai` are still open upstream. Local
implementation and evidence are ahead of their tracker state:

- **#1 joint-2 tracking fault:** not reproduced in either 45 s dual run; the
  unchanged 0.08 rad stop remained active and the worst observed maximum was
  0.016873 rad. Treat as provisionally healthy, not closed, until a longer
  representative run passes.
- **#2 Mirrored/front-facing mode:** partially resolved at 45%. The swapped
  hand assignment, lateral/vertical translation, rotations, and tracking are
  clean, but the 2026-08-14 re-test rejected front/back direction. The current
  first translation sign is `-1`; a `+1` candidate needs MuJoCo validation.
- **#3 camera operator view:** core feature implemented and operator-accepted
  with one D455 and two D405 roles, independent camera transport, dropdowns,
  toggles, overview/focus cycling, and world-anchored transitions. Remaining
  closure work is glass-to-glass latency, active pose-rate impact, and a
  camera-loss hold policy.
- **#4 calibration/reach UX:** still outstanding. Wrist calibration now arms
  only from the 2D Calibrate wrist button in every mode (2026-10-23: the
  right-hand page had auto-armed it, so closing the gripper while clutched
  paused streaming for 5 s). The headset still does not persistently show
  per-hand calibration, clutch, or near-limit state.

## Recommended next milestone

Behind at 40% remains the accepted dual scope. Do not repeat the physical
Mirrored test until its front/back candidate passes MuJoCo, and do not repeat
the tight marker-cap benchmark yet. The unstable soft load-yield remains
removed from physical control; bounded collision recovery and per-arm
send-duration telemetry remain with every hard gate unchanged. Use its isolated
`--jerk-stress` reproduction in MuJoCo to capture an oscillation trace, then
design and compare a latched soft stop against exactly the same delayed-feedback
case before considering another loaded-release test.

## Change-control state

The current work is on local branch `bimanual-teleop-updated`. Camera, bimanual
stabilization, documentation, and test changes are intentionally uncommitted
and unpushed until the owner asks for publication. No physical robot command was
sent during the camera and LAN milestones.
