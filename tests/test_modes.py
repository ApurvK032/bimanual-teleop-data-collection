"""Camera modes and pixel boxes carried between modes (no camera needed)."""
from __future__ import annotations

import pytest

from bimanual_collect.calibration import map_box_from_2k
from bimanual_collect.zed_v4l2 import check_mode


def test_modes_and_rates() -> None:
    assert check_mode("1080") == (3840, 1080, 30)
    assert check_mode("1080", 15) == (3840, 1080, 15)
    assert check_mode("2k") == (4416, 1242, 15)
    assert check_mode("720") == (2560, 720, 30)  # default is the fastest rate up to 30
    assert check_mode("720", 60) == (2560, 720, 60)
    with pytest.raises(ValueError):
        check_mode("2k", 30)  # 2K only runs at 15 fps
    with pytest.raises(ValueError):
        check_mode("4k")


def test_box_maps_through_intrinsics(tmp_path) -> None:
    # Intrinsics shaped like a ZED 2i: FHD is a centred native-scale crop of 2K, HD is binned (half focal).
    calib = tmp_path / "SN1.conf"
    calib.write_text(
        "[LEFT_CAM_2K]\nfx=1000\nfy=1000\ncx=1100\ncy=600\n"
        "[LEFT_CAM_FHD]\nfx=1000\nfy=1000\ncx=956\ncy=519\n"
        "[LEFT_CAM_HD]\nfx=500\nfy=500\ncx=640\ncy=360\n"
    )
    box = (800, 10, 1300, 1000)
    assert map_box_from_2k(box, calib, 2208) == box
    assert map_box_from_2k(box, calib, 1920) == (656, 0, 1156, 919)  # shifted by the crop; top clipped at 0
    assert map_box_from_2k(box, calib, 1280) == (490, 65, 740, 560)  # half scale about the centre
