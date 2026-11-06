from __future__ import annotations

from widowxai_quest_teleop.viewer import close_passive_viewer


class FakePassiveViewer:
    def __init__(self) -> None:
        self.simulate = object()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.simulate = None

    def _sim(self):
        return self.simulate


def test_passive_viewer_wait_handles_completed_native_shutdown() -> None:
    viewer = FakePassiveViewer()
    close_passive_viewer(viewer, timeout_s=0.1)
    assert viewer.close_calls == 1
