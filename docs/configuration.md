# Configuration

## The bench (arms): set in the app

The app's **Bench** tab saves the arm setup in `configs/local/bench.json` (git-ignored) and writes the teleop
profiles from it: `configs/local/bench.yaml` (what every run uses) and, for two arms, `bench_left.yaml` /
`bench_right.yaml` (the first-run check). These files are regenerated on every save: change them in the app, not by
hand. Settings: one or two arms, their addresses and sides, the base separation, the response (0.40 / 0.60 /
0.70-0.60), the joint tracking stop (0.08 or 0.12 rad), the run time limit, grippers on/off, the arm-to-arm collision distance and an optional table floor (table size, base distance from the operator-side edge, floor below the surface). Each drop-down also takes a custom value: response 0.20-1.00, tracking stop 0.05-0.15 rad, run time 0.5-30 min, arm distance 1-5 cm, floor 0-10 mm. Saving also
sets `arms` and `paths.arm_profile` in `configs/rig.yaml`. A hand-made profile already pointed to by
`paths.arm_profile` in `configs/local/` is read in as the starting point.

## The rig (`configs/rig.yaml`)

One file per bench, git-ignored. `python -m bimanual_collect setup` writes it; without it,
`configs/rig.example.yaml` is used. `$BIMANUAL_RIG` can point at another file. Relative paths are relative to the
repository root. The app's System tab shows the values in use.

| Key | Default | Meaning |
|---|---|---|
| `arms` | `null` | `1`, `2`, or `null` = whatever the teleop run logged. Limits which runs can match an episode. |
| `camera.serial` | `null` | the ZED's serial; `null` = the only ZED connected |
| `camera.mode` / `camera.fps` | `"1080"` / `15` | `2k` (15 fps), `1080` (30/15), `720` (60/30/15), `vga` (100/60/30/15) |
| `camera.backend` | `v4l2` | Set to `zed_sdk` to record SVO2 alongside MKV. A ZED in `cameras` can override `backend`. |
| `camera.calibration` | `calibration/SN{serial}.conf` | factory calibration file (`python -m bimanual_collect calibration` downloads it) |
| `cameras` | (absent) | several cameras: a list replacing `camera:` (below) |
| `paths.data_raw` | `data/raw` | recorded episodes |
| `paths.data_lerobot` | `data/lerobot` | exported datasets |
| `paths.state` | `state` | generated files: logs, patched Quest page, camera roles |
| `paths.teleop` / `paths.teleop_runs` | `teleop` / `[teleop/runs]` | the teleop repo and where its telemetry goes |
| `paths.lerobot_python` | `.venv-lerobot/bin/python` | Python with `lerobot` (export jobs) |
| `paths.depth_python` | `.venv-stereo/bin/python` | Python with `torch` + `torchvision` (RAFT depth jobs) |
| `paths.arm_profile` | example profile | the teleop profile (`configs/local/bench.yaml` once the bench is saved) |
| `boxes_2k.sync_roi` | | box around the arms (operator outside) for the timing check, in 2K left-eye pixels |
| `boxes_2k.video_crop` | | crop for review videos |
| `boxes_2k.table` | | the table top |
| `depth.near_mm` / `far_mm` | 900 / 1600 | colour range of depth previews |
| `export.robot_type` | `null` | `null` = `widowxai_bimanual_quest` (2 arms) / `widowxai_quest` (1 arm) |
| `export.crf` | 18 | AV1 quality (lower = better; LeRobot's default is 30) |
| `export.eyes` | `both` | stereo cameras: `both` or `left` |
| `export.image_size` | `null` | per-image `WxH`, e.g. `640x360`; `null` = capture size |
| `qc.max_camera_drop_pct` | 2.0 | camera error-flagged frames, per camera |
| `qc.max_torn_pct` | 1.0 | sideways-wrapped frames |
| `qc.max_sync_abs_ms` / `qc.min_sync_r` | 15.0 / 0.6 | camera-vs-arm offset and how clearly it was measured |
| `qc.min_duration_s` | 5.0 | shortest acceptable episode |
| `qc.require_task` | `true` | an episode needs a task label to pass |
| `recording.min_free_to_start_gb` | 20 | refuse to start an episode below this |
| `recording.warn_free_gb` / `stop_free_gb` | 50 / 10 | warn / stop a running episode |
| `operator` | `""` | default operator name |
| `tasks` | list | task presets offered in the app |

Boxes are easiest to set in the app (Bench tab → Camera → draw → Save boxes): they are stored in 2K pixels and
mapped to whatever mode the camera runs in. Redraw them after moving the camera.

## Camera notes

For SVO2, install ZED SDK 5.5 or later with its Python bindings in the recorder environment (`.venv` by default).
Set `camera.backend: zed_sdk` in `configs/rig.yaml` and restart the collection app. The recorder uses the GPU's
hardware encoder and runs with depth disabled. For depth after capture, open the SVO2 with `NEURAL`, `NEURAL_LIGHT`
or `NEURAL_PLUS` in the ZED SDK. [ZED SDK recording and playback](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/recording)

Why 1080 at 15 fps: on the same scene, depth from 1080 or 2K resolves thin parts (gripper fingers, a cable loop,
cloth folds) that 720 smears, and the ZED SDK's NEURAL_PLUS depth is about twice as steady; 15 fps keeps the USB
link and disk load low (about 2.6 GB per minute with the SVO2). Use 720 at 30 fps when frame rate matters more than
detail.

| Mode | Per eye | fps | View | Camera drops while recording (measured) |
|---|---|---|---|---|
| `2k` | 2208×1242 | 15 | native-scale crop of the sensor | 0–0.4% |
| **`1080` (default, 15 fps)** | 1920×1080 | 30, 15 | tighter native-scale crop, same detail as 2K | 0% at 15 fps; 0.6–3.9% at 30 fps |
| `720` | 1280×720 | 60, 30, 15 | binned, wider view, half the detail | 0% at 30 fps |
| `vga` | 672×376 | 100, 60, 30, 15 | binned, widest view | not measured |

- **USB 3 is required**: on USB 2 a ZED only offers 1344×376. If the System tab reports USB 2, reseat or flip the
  USB-C plug at the camera.
- **Mains flicker**: V4L2 capture sets the camera's anti-flicker to 60 Hz (North America). For 50 Hz mains, change
  `power_line_frequency` in `bimanual_collect/zed_v4l2.py`.
- **Colour**: recordings keep the camera's colour. The live views and the export apply a fixed two-point colour
  correction measured under one room's lighting (`bimanual_collect/zed_format.py`); to redo it for other lighting,
  measure the mean colour of known-black and known-white surfaces and solve per channel for gain and offset.

## Several cameras

```yaml
cameras:
  - name: zed            # becomes observation.images.zed_left / zed_right
    type: zed
    serial: "12345678"
    mode: "720"
    fps: 30
    primary: true        # sets the dataset's step clock
  - name: wrist_right    # becomes observation.images.wrist_right
    type: uvc
    match: "RealSense"   # part of the camera's name, or serial: "..."
    width: 640
    height: 480
    fps: 30
```

Every camera records its own lossless video and frame log. In the dataset, each step shows each camera's frame
nearest in exposure time to the primary camera's frame; when a camera has no frame within half its frame period, the
previous one is shown and `observation.<name>_frame_repeated` is 1. The natural-colour fix is applied only to ZED
cameras unless `colour_fix: true` is set for another camera.
