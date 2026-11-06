# Configuration guide

Configuration files stay at stable paths because telemetry snapshots and
rollback tags refer to them. Use the profiles below instead of choosing a file
only from its name.

## Recommended

### `quest_50pct_hardware.yaml`

Current single-arm profile:

- selects Left/Right and Behind/Mirrored from the Quest page before connecting
  to the arm;
- locks that choice for the complete run;
- runs at the Quest cadence (nominally 90 Hz);
- uses a 25 ms Trossen interpolation horizon and conservative velocity
  feedforward;
- fails closed unless AC power, the `performance` profile, and the dedicated
  direct robot-Ethernet route are present;
- maps controller motion at 50% of the reference article scale;
- enables the full 0–40 mm gripper stroke;
- returns the arm to all-zero rest on exit.

Right + Mirrored is operator-accepted at this 50% scale. Right + Behind and
Left + Behind are accepted separately at 40% and must not be inferred as
accepted at 50%; Left + Mirrored is accepted separately at 45% and must not be
inferred as accepted at 50%.

### `step3_velocity_feedforward_25ms_hardware.yaml`

Tagged left-controller rollback baseline (`milestone-3-step-3-best`):

- 30% task scale;
- full gripper;
- nominal 90 Hz Quest-synchronized commands;
- 25 ms driver horizon;
- conservative velocity feedforward;
- maximum 60-second live duration.

## Task profiles

`calibrations/task_profiles.json` maps the four page selections to fixed
calibration documents:

| Quest page | Profile | Status at 50% |
|---|---|---|
| Left + Behind/Parallel | `left_behind` | Accepted at 40%; 50% pending |
| Left + Mirrored | `left_mirror` | Accepted at 45%; 50% pending |
| Right + Behind/Parallel | `right_behind` | Accepted at 40%; 50% pending |
| Right + Mirrored | `right_mirror` | Accepted |

Mirrored semantics flip left/right and forward/back translation, screw, and
nod-no. Up/down and nod-yes remain unchanged.

## Rollback ladder

These files preserve the measured tuning sequence:

1. `safe_demo_30pct.yaml`
2. `calibrated_adaptive_30pct_hardware.yaml`
3. `no_catchup_30pct_hardware.yaml`
4. `send_barrier_30pct_hardware.yaml`
5. `step2_25ms_30pct_hardware.yaml`
6. `step3_velocity_feedforward_25ms_hardware.yaml`

Matching MuJoCo/replay profiles and the article-rate experiments are retained
for controlled comparisons. Do not use an experimental profile on hardware
without first testing it in MuJoCo.

## Specialized profiles

- `six_axis_arm_diagnostic.yaml` drives a deterministic Quest-free diagnostic.
- `cad_rest_commissioning.yaml` is the independent MuJoCo-only CAD/leader-arm
  experiment.
- `cad_home_commissioning_mujoco.yaml` retains its legacy filename but now
  previews the guarded 10% mapping and ±2° simulation phase from all-zero
  rest, matching the printed leader's requested initial visualization. The
  physical profile keeps its separate rest-to-home lifecycle.
- `cad_hardware_commissioning.yaml` is the guarded first physical CAD profile:
  five tracked joints at 10% scale, ±2° about home, 15 seconds maximum, fixed
  joint 5, disabled gripper, raw-stream watchdogs, adaptive source smoothing,
  workstation/network preflight, and candidate signs that require the explicit
  first-run acceptance flag. Pending live mapping additionally requires
  `--commission-joint 0` through `4`; every unselected follower joint is held
  exactly at home. Live output also requires an explicitly selected Linux
  evdev hold-to-run device that is polled every control iteration; a missing,
  held, unsupported, or disconnected device fails closed. Follow
  `docs/CAD_LEADER_TELEOP.md`; do not expand this evidence profile in place.
- `right_mirror_axis_validation_*`, `right_real_axis_validation_hardware.yaml`,
  and `temporary_*` preserve calibration experiments and are not normal run
  profiles.
- `right_dual_bench_axis_validation_hardware.yaml` applies the exact right-hand
  Behind candidate from the dual profile to physical arm `.3` at the earlier
  reduced 0.20 scale. Run it only with the stationary-arm guard flags on
  `run_hardware.py`: `--dual-guard-config configs/dual_widowxai.yaml
  --moving-side right`. Arm `.2` remains read-only and its fresh measured
  geometry screens every command and shutdown path.
- `right_dual_bench_40pct_full_gripper_hardware.yaml` is the requested second
  stage: 0.40 translation/rotation scale, 0.070 m / 0.16 rad reach, full
  trigger-controlled 0–0.040 m gripper stroke, and a 45 s cap. While arm `.2`
  is mounted beside it, it requires the same `.2` read-only stationary-arm
  guard; with `.2` removed from the bench it may run solo without the guard
  flags (owner decision, 2026-10-23). The operator accepted its directions,
  orientation response, gripper, and shutdown on 2026-08-12; it does not replace
  the clean 0.20 evidence profile or validate higher gains.

Change one control variable per experiment. Never weaken physical joint,
collision, stale-stream, feedback, driver-version, or shutdown gates merely to
make a run launch.

## Dual-arm profile

`dual_widowxai.yaml` is the bimanual profile: both Quest controllers, two arms,
one coordinated runtime. It replaces the single `quest` and `hardware.robot_ip`
blocks with per-arm `arms.left` / `arms.right` entries plus a `safety` block,
and it is validated by `config.parse_dual_arm_config` rather than by the
single-arm loader.

Its base transforms record the bench geometry measured on 2026-08-12: the arm
bases are exactly 500 mm apart with no forward or vertical offset, equal height,
parallel axes, no relative yaw, and both base centers 50.8 mm (2 in) forward of
the table's rear edge. Their explicit `measurement_status` is `measured`. Live
mode still rejects any other status, including when the calibration-only
override is supplied.

Both Behind mappings are accepted at 40%. The active profile now stages 60%
translation and 60% rotation response for the owner-requested marker task;
that gain is not yet accepted for general use. Post-collision recovery is
limited to 0.010 rad per joint per tick. The measured-feedback load-yield
experiment is deliberately absent after its physical activation oscillated
both arms. Position/rotation reach remains 0.070 m / 0.16 rad per clutch; the
hard 0.08 rad feedback stop and 0.030 m collision margin are unchanged.
Mirrored remains at its separately accepted 45% scope.

Live two-arm output additionally requires an explicitly accepted calibration
for each arm and the dual `LIVE-WIDOWXAI-DUAL-<left-ip>-<right-ip>` token. Do
not use the single-arm profiles or token for two arms.

For a bimanual MuJoCo-only task-frame capture, calibrate each controller
separately from the same schema-v2 stream. For example, the left hand uses:

```bash
env -u PYTHONPATH .venv/bin/python scripts/calibrate_task_frame.py \
  --config configs/dual_widowxai.yaml \
  --output configs/calibrations/left_behind_mujoco_20260811.json \
  --repeats 2 \
  --overwrite \
  --bimanual-hand left \
  --mapping-mode real \
  --simulation-only
```

The current MuJoCo profile is `dual_widowxai_mujoco_calibrated.yaml`. It uses the
captured left file above and, at the owner's request, an explicitly documented
identity transfer of that task frame for the right controller. The derived
right file does not claim right-hand capture evidence. Both files are stamped
`simulation_only`, and the profile sets both hardware gates false. Never replace
the calibrations in `dual_widowxai.yaml` with them.

Run the calibrated 500 mm tabletop scene with:

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_dual_sim.py \
  --config configs/dual_widowxai_mujoco_calibrated.yaml \
  --tabletop \
  --inline-viewer \
  --label dual-500mm-mujoco-calibrated
```
