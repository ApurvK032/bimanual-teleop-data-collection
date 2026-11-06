# Third-party components and credits

The license in `LICENSE` covers only the original code of this repository (the `bimanual_collect/` package, the
web app, scripts, tests and docs). The following keep their own licenses:

| Component | Where | License / terms |
|---|---|---|
| Teleoperation stack for the WidowX AI (adapted from Aurel Arnold's *VR Teleoperation Stack* and Dream Machines Robotics' `vr-teleop-kit`; adds the robot model, Trossen driver integration, task-frame calibration, safety checks, latency telemetry) | `teleop/` | MIT (`teleop/LICENSE`); the parts derived from `vr-teleop-kit` stay under Apache-2.0: notice in `teleop/THIRD_PARTY_NOTICES.md`, full text in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) |
| Trossen Robotics arm model and description | `teleop/third_party/` (submodules) | their own repositories' licenses |
| `trossen-arm` driver | installed from PyPI | Trossen Robotics' terms |
| LeRobot (Hugging Face) | installed from PyPI; dataset format v3.0 | Apache-2.0 ([text](LICENSES/Apache-2.0.txt)) |
| ZED SDK / `pyzed` | installed by the user | Stereolabs' terms; not redistributed here |
| FFmpeg, OpenCV, NumPy, MuJoCo, PyTorch | installed from their packages | their own licenses |

Credit and thanks to Aurel Arnold, Dream Machines Robotics, Trossen Robotics, Hugging Face (LeRobot) and Stereolabs.
