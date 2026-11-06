# Teleop: arm-safety rules

The teleop stack (Quest 3 -> IK -> combined two-arm collision check -> Trossen drivers) is used by the data
collection app in this repository. These rules apply to anyone, person or tool, who drives the arms with it. The
app keeps them: it asks for the confirmations in rule 2 before every live run, stops runs with a single interrupt
(rule 10), and never clears controller errors or probes the control port (rules 6 and 7).

## Non-negotiable physical-arm rules

1. Never move the physical arm without an explicit command from the user in the
   current conversation. Inspection, documentation, simulation, and offline
   telemetry analysis do not authorize motion.
2. Before any physical run, require a clear workspace, a firmly mounted arm,
   and a person ready to cut controller power.
3. Keep the grip released during preflight and the automatic rest-to-home move.
   Grip is the live deadman only after the launcher prints `home reached`.
4. Use MuJoCo first after calibration, mapping, IK, timing, or limit changes.
5. Never bypass the `--live` and exact
   `--confirm-live LIVE-WIDOWXAI-192.168.1.2` gates.
6. Never automatically clear controller errors, flash firmware, update firmware,
   or change parameters stored on the arm.
7. Never probe TCP port `50001` with `nc`, telnet, a raw socket, or a generic
   readiness check. The controller appears to support one client and a raw probe
   wedged its server. Inspect local sockets with `ss` and use ICMP `ping` only.
8. If the driver hangs after printing the TCP and UDP connection lines but
   before driver/firmware versions, stop the process. If a stale local session
   clears but the next driver handshake still hangs, have the user power-cycle
   the controller with the arm supported. Do not repeatedly reconnect.
9. Do not assume a safety stop is permission to weaken all limits. Diagnose the
   exact check and retain hard joint, collision, stale-stream, driver-version,
   feedback, and shutdown gates.
10. On any normal or safety-stop exit after motion starts, the launcher must
    return the arm to all-zero rest. If the return fails, hold measured position
    if possible and instruct the user to cut controller power.


## Known failure signatures and interpretation

| Message/symptom | Meaning and response |
|---|---|
| `waiting for fresh Quest tracking with grip released` | WebXR session/controller pose is not fresh, headset is asleep, wrong hand page is open, or grip is held. Do not bypass. |
| Hangs after TCP/UDP connection lines | Controller handshake/server problem. Stop; inspect local sockets without connecting; power-cycle if repeated. |
| Tiny negative joint command at zero boundary | Numerical/encoder noise. Clamp only within a small tolerance; retain physical limit. |
| `gripper command -0.000000 outside` | Floating-point boundary issue; clamp to configured physical range. |
| `gripper command exceeded per-tick cap` | Trigger discontinuity; rate-limit smoothly without reducing full stroke. |
| Startup gripper tracking error | Do not supervise arm and gripper as one simultaneous ramp on this legacy setup; use separate blocking gripper move. |
| Joint tracking error during live run | Could be reference timing rather than a stall. Compare to delayed/interpolated command, but never remove the hard stall threshold blindly. |
| Table vibration/audible wrist pulses | Usually short-horizon replanning, high-rate reversals, catch-up bursts, or noisy orientation commands. MuJoCo cannot validate this physical effect. |
| Random direction after headset movement | Coordinate-frame/calibration or optical tracking discontinuity, not necessarily IK failure. Release grip and recalibrate/validate. |
| `TCP connection closed unexpectedly` on shutdown | Controller/driver connection loss. Attempt measured hold only if possible; tell user to cut power. |
| Arm freezes about 5 s when the trigger is pressed while gripping, then re-anchors | Quest page started a wrist-calibration capture on the grip + trigger chord. Fixed 2026-10-23: calibration arms only from the Calibrate wrist button. Reload the Quest page to pick up the fix. |
| `wrong Trossen driver for this arm profile: installed=1.8.6, required=1.9.3` | Launched from the retired `.venv-arm18`. Use `.venv-arm19`. No arm connection was attempted. |
| `Trossen driver/firmware mismatch` | Driver and controller firmware major.minor differ. Install the matching driver; never downgrade firmware or loosen the gate. |

