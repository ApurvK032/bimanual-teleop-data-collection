from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from widowxai_quest_teleop.camera_service import (
    ROLE_NAMES,
    CameraDevice,
    CameraRuntime,
    FfmpegCamera,
    LatestJpegFrame,
    assign_camera_roles,
    create_app,
    empty_configuration,
    normalize_configuration,
    split_jpeg_stream,
)


JPEG_A = b"\xff\xd8first\xff\xd9"
JPEG_B = b"\xff\xd8second\xff\xd9"


def camera(serial: str, video_index: int) -> CameraDevice:
    return CameraDevice("", Path(f"/dev/video{video_index}"), serial, "D405")


class FakeCamera:
    def __init__(self, device: CameraDevice, **options: object) -> None:
        self.device = device
        self.width = int(options.get("width", 640))
        self.height = int(options.get("height", 480))
        self.fps = int(options.get("fps", 30))
        self.frames = LatestJpegFrame()
        self.started = False

    def start(self) -> None:
        self.started = True
        if self.frames.snapshot() is None:
            self.frames.publish(JPEG_A)

    def stop(self) -> None:
        self.started = False

    def health(self) -> dict[str, object]:
        return {
            "role": self.device.role,
            "serial": self.device.serial,
            "healthy": self.started,
            "process_alive": self.started,
        }


def test_jpeg_stream_parser_keeps_partial_tail_and_returns_complete_frames() -> None:
    buffer = bytearray(b"garbage" + JPEG_A + JPEG_B[:-2])

    assert split_jpeg_stream(buffer) == [JPEG_A]
    assert buffer == bytearray(JPEG_B[:-2])

    buffer.extend(JPEG_B[-2:])
    assert split_jpeg_stream(buffer) == [JPEG_B]
    assert buffer == bytearray()


def test_latest_frame_mailbox_skips_obsolete_frames() -> None:
    mailbox = LatestJpegFrame()
    first = mailbox.publish(JPEG_A, ready_monotonic_ns=1_000_000_000)
    second = mailbox.publish(JPEG_B, ready_monotonic_ns=1_100_000_000)

    assert first.sequence == 1
    assert second.sequence == 2
    assert mailbox.snapshot() == second
    assert mailbox.wait_after(first.sequence, 0.01) == second
    assert mailbox.wait_after(second.sequence, 0.01) is None
    assert mailbox.measured_fps() == pytest.approx(10.0)


def test_camera_roles_are_stable_and_can_be_selected_by_serial() -> None:
    devices = [camera("235", 12), camera("115", 6)]

    automatic = assign_camera_roles(devices)
    assert automatic["scene"].serial == "115"
    assert automatic["wrist"].serial == "235"
    assert automatic["scene"].role == "scene"
    assert automatic["wrist"].role == "wrist"

    selected = assign_camera_roles(devices, scene_serial="235", wrist_serial="115")
    assert selected["scene"].path == Path("/dev/video12")
    assert selected["wrist"].path == Path("/dev/video6")


def test_camera_roles_require_two_distinct_devices() -> None:
    with pytest.raises(RuntimeError, match="two D405"):
        assign_camera_roles([camera("115", 6)])
    with pytest.raises(RuntimeError, match="different cameras"):
        assign_camera_roles(
            [camera("115", 6), camera("235", 12)],
            scene_serial="115",
            wrist_serial="115",
        )


def test_three_role_configuration_disables_empty_views_and_rejects_duplicates() -> None:
    normalized = normalize_configuration(
        {
            "roles": {
                "scene": {"serial": "115", "enabled": True},
                "left_wrist": {"serial": None, "enabled": True},
                "right_wrist": {"serial": "235", "enabled": False},
            }
        }
    )

    assert tuple(normalized) == ROLE_NAMES
    assert normalized["scene"] == {"serial": "115", "enabled": True}
    assert normalized["left_wrist"] == {"serial": None, "enabled": False}
    assert normalized["right_wrist"] == {"serial": "235", "enabled": False}
    with pytest.raises(ValueError, match="assigned to both"):
        normalize_configuration(
            {
                "scene": {"serial": "115", "enabled": True},
                "left_wrist": {"serial": "115", "enabled": False},
            }
        )


def test_runtime_persists_dynamic_assignments_and_only_starts_enabled_roles(tmp_path: Path) -> None:
    devices = [camera("115", 6), camera("235", 12), camera("315", 18)]
    config_path = tmp_path / "cameras.json"
    runtime = CameraRuntime(
        devices,
        discoverer=lambda: devices,
        camera_factory=FakeCamera,
        config_path=config_path,
    )
    state = runtime.configure(
        {
            "scene": {"serial": "115", "enabled": True},
            "left_wrist": {"serial": "235", "enabled": False},
            "right_wrist": {"serial": "315", "enabled": True},
        }
    )

    assert set(runtime.cameras) == {"scene", "right_wrist"}
    assert state["roles"]["left_wrist"] == {"serial": "235", "enabled": False}
    assert config_path.exists()
    assigned = {item["serial"]: item["assigned_role"] for item in state["devices"]}
    assert assigned == {"115": "scene", "235": "left_wrist", "315": "right_wrist"}
    assert runtime.health()["status"] == "ok"

    restored = CameraRuntime(
        devices,
        discoverer=lambda: devices,
        camera_factory=FakeCamera,
        config_path=config_path,
    )
    restored.start(fallback_configuration=empty_configuration())
    assert set(restored.cameras) == {"scene", "right_wrist"}
    assert restored.configuration == runtime.configuration
    runtime.stop()
    restored.stop()


def test_first_run_preserves_old_two_camera_scene_and_wrist_behavior() -> None:
    runtime = CameraRuntime(
        [camera("235", 12), camera("115", 6)],
        discoverer=lambda: [],
        camera_factory=FakeCamera,
    )

    configuration = runtime.first_run_configuration()

    assert configuration["scene"] == {"serial": "115", "enabled": True}
    assert configuration["left_wrist"] == {"serial": None, "enabled": False}
    assert configuration["right_wrist"] == {"serial": "235", "enabled": True}


def test_ffmpeg_command_uses_low_latency_yuyv_capture() -> None:
    stream = FfmpegCamera(CameraDevice("scene", Path("/dev/video6"), "115", "D405"))
    command = stream.command()

    assert command[0] == "ffmpeg"
    assert "nobuffer" in command
    assert "low_delay" in command
    assert "yuyv422" in command
    assert "640x480" in command
    assert "image2pipe" in command


def test_camera_app_exposes_health_snapshot_and_websocket_routes() -> None:
    scene = FfmpegCamera(CameraDevice("scene", Path("/dev/video6"), "115", "D405"))
    scene.frames.publish(JPEG_A, ready_monotonic_ns=time.monotonic_ns())
    runtime = CameraRuntime([camera("115", 6)], discoverer=lambda: [camera("115", 6)])
    runtime.configuration = normalize_configuration(
        {"scene": {"serial": "115", "enabled": True}}
    )
    runtime.cameras = {"scene": scene}
    app = create_app(runtime)
    routes = {route.path: route for route in app.routes}

    assert "/health" in routes
    assert "/configuration" in routes
    assert "/camera/{role}.jpg" in routes
    assert "/ws/{role}" in routes
    response = asyncio.run(routes["/camera/{role}.jpg"].endpoint(role="scene"))
    assert response.status_code == 200
    assert response.body == JPEG_A
    assert response.headers["x-camera-sequence"] == "1"


def test_quest_page_includes_camera_cycle_and_separate_service() -> None:
    web = Path(__file__).resolve().parents[1] / "src" / "widowxai_quest_teleop" / "web"
    html = (web / "index.html").read_text(encoding="utf-8")
    client = (web / "client.js").read_text(encoding="utf-8")
    camera_javascript = (web / "camera_view.js").read_text(encoding="utf-8")

    assert 'id="scene-preview"' in html
    assert 'id="left-wrist-preview"' in html
    assert 'id="right-wrist-preview"' in html
    assert 'id="scene-camera-select"' in html
    assert 'id="left-wrist-camera-select"' in html
    assert 'id="right-wrist-camera-select"' in html
    assert 'id="apply-camera-setup"' in html
    assert 'id="cycle-camera-view"' in html
    assert 'src="/camera_view.js"' in html
    assert "cameraView.handleControllerButtons" in client
    assert "cameraView.render(session, viewerPose)" in client
    assert "applyCameraConfiguration" in client
    assert 'fetch(`${CAMERA_SERVICE_URL}/configuration`' in client
    assert "const QUEST_B_BUTTON = 5" in camera_javascript
    assert "Release both grips to change camera view" in camera_javascript
    assert "CAMERA_PORT = 8444" in camera_javascript
    assert 'CAMERA_ROLES = ["scene", "left_wrist", "right_wrist"]' in camera_javascript
    assert "TRANSITION_DURATION_MS = 380" in camera_javascript
    assert "PANEL_VERTICAL_OFFSET_M = 0.18" in camera_javascript
    assert "bottom + PANEL_VERTICAL_OFFSET_M" in camera_javascript
    assert "this.worldAnchorMatrix = new Float32Array(viewerPose.transform.matrix)" in camera_javascript
    assert "view.transform.inverse.matrix,\n          this.worldAnchorMatrix" in camera_javascript
    assert "panelLayout(this.enabledRoles(), this.mode())" in camera_javascript
