from __future__ import annotations

import time
from typing import Any


def close_passive_viewer(viewer: Any, timeout_s: float = 2.0) -> None:
    """Close MuJoCo's daemon viewer thread before Python tears down GLFW."""

    viewer.close()
    # MuJoCo 3.8.1's Linux Handle.close() requests exit but doesn't join the
    # daemon render thread. Its weak reference clears after native destruction.
    simulate_ref = getattr(viewer, "_sim", None)
    if not callable(simulate_ref):
        return
    deadline = time.perf_counter() + float(timeout_s)
    while simulate_ref() is not None and time.perf_counter() < deadline:
        time.sleep(0.01)
    if simulate_ref() is not None:
        raise RuntimeError("MuJoCo passive viewer did not finish shutting down")
