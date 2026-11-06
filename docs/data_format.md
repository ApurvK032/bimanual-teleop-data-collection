# Data format

## Raw episodes: `data/raw/<date>/session_HHMMSS/episode_NNNN/`

Written while recording (format 2):

| File | Contents |
|---|---|
| `episode.json` | Written at start (`status: recording`) and at the end. `status`: `complete`, `aborted` (camera lost, encoder died; `stop_reason` says which) or `recovered` (repaired after a crash). Labels `{task, operator, outcome, note}` (outcome: success / failure / unsure). Start/stop times (monotonic and epoch). `cameras`: one entry per camera (name, primary, stereo, size, nominal fps, model, serial, mode, calibration, counters: frames recorded, missing, error-dropped, recorder drops, encoder exit code). The primary camera's fields are repeated at the top level (`video`, `camera`, `frames_recorded`...). |
| `cameras/<name>/video.mkv` | Lossless FFV1, YUYV (BT.601, **full range**, camera colour). V4L2 supplies YUYV directly and the SDK backend converts its BGR output to YUYV. Stereo cameras store both eyes side by side (left = columns `[0, W/2)`). Decode with `ffmpeg -i video.mkv -vf scale=in_range=pc:out_range=pc,format=yuyv422 -color_range pc -f rawvideo out.yuyv`. |
| `cameras/<name>/video.svo2` | With `camera.backend: zed_sdk`, a native SVO2 from the same grabs using the SDK's hardware `H265_LOSSLESS` mode. The camera's `svo2` metadata gives the path, finalization status and frame count. |
| `cameras/<name>/SN<serial>.conf` | Calibration for the SVO2. Keep it with the recording and use its folder as `InitParameters.optional_settings_path` when replaying. |
| `cameras/<name>/frames.csv` | One row per stored frame: `video_frame, camera_sequence, exposure_monotonic_ns, arrival_monotonic_ns, arrival_epoch_ns`. A sequence gap marks dropped frames. The SDK backend stores the SDK `IMAGE` timestamp in `exposure_monotonic_ns` after conversion to `CLOCK_MONOTONIC`. The original SDK epoch timestamp is `exposure_monotonic_ns + arrival_epoch_ns - arrival_monotonic_ns`. |
| `cameras/<name>/encoder.log` | FFmpeg's output (empty when all went well). |

Older episodes (format 1) have one ZED and `zed_sbs.mkv` + `zed_frames.csv` in the episode folder;
every tool reads both layouts (`episode_io.py`).

Written afterwards:

| File | By | Contents |
|---|---|---|
| `frame_check.json` | `process` / `check-frames` | torn frames of the primary camera, method |
| `sync/` | `process` / `sync` | `sync_report.json` (offset, r, blocks), `sync_plot.png`, optionally `sync_preview.mp4` |
| `episode_qc.json` | `process` | every check with value, limit and result; verdict `pass` / `needs_review`; reasons |
| `thumbs.jpg`, `preview.mp4` | `process` | primary camera stills (10/50/90%) and a small H.264 clip, for the console |
| `review.json` | `review` / console | `accepted` / `rejected`, task, outcome, note (override the recording-time labels) |
| `rgbd/`, `rgbd_raft/` | `depth` | `depth/<frame>.png` (uint16 mm, 0 = none), `rgb/<frame>.jpg` (rectified left, aligned), `confident/` (RAFT), `frames.csv`, `intrinsics.json`, `preview.mp4` |

Per session: `report.md`. For the whole data folder: `data/raw/index.csv`, one row per episode (task, outcome,
operator, status, length, mode, drops, torn frames, sync offset, steps, QC, review, reasons).

## LeRobot dataset (`export`)

LeRobot dataset format v3.0, written with LeRobot's own `LeRobotDataset` writer, one step per primary-camera frame.

| Feature | Type | Contents |
|---|---|---|
| `observation.state` | float32[14] or [7] | measured `left_joint_0..5, left_gripper, right_joint_0..5, right_gripper` (one arm: `joint_0..5, gripper`), rad and m |
| `action` | same | commanded, same order |
| `observation.images.<camera>_left` / `_right` | video | stereo camera, per eye (`--eyes left` keeps only the left) |
| `observation.images.<camera>` | video | mono camera |
| `observation.frame_repeated` | int64[1] | 1 when the primary camera dropped the frame (or it was torn) and the previous image is shown |
| `observation.<camera>_frame_repeated` | int64[1] | the same for each other camera |
| `observation.<camera>_exposure_monotonic_ns` | int64[1] | capture timestamp of the image shown, per camera, using the source described above |

With the default single ZED named `zed`: `observation.images.zed_left`, `zed_right`, `observation.frame_repeated`,
`observation.zed_exposure_monotonic_ns`. Video: AV1 (SVT-AV1) at CRF 18 by default (LeRobot's default is 30); images
at the capture size or `--image-size` (other aspect ratios keep their own aspect at that height). Robot type:
`widowxai_bimanual_quest` or `widowxai_quest`. Next to the dataset, `<repo-id>.export.json` records the source
episodes, alignment reports, cameras and settings.

## Arm telemetry columns used

From the teleop repo's `telemetry.csv`: `pc_monotonic_ns`, `pc_epoch_ns`, and per arm (prefix `left_` / `right_`, or
none for one arm): `q_feedback` (6 joints, JSON list), `gripper_feedback_m`, `feedback_sample_fresh`,
`feedback_read_monotonic_ns`, `q_cmd`, `gripper_cmd_m`, `command_send_monotonic_ns`.
