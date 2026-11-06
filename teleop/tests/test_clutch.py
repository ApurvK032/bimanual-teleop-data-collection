import numpy as np

from widowxai_quest_teleop.clutch import ClutchController
from widowxai_quest_teleop.mapping import ClutchPoseMapper
from widowxai_quest_teleop.types import Pose


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def make_clutch() -> ClutchController:
    mapper = ClutchPoseMapper(
        "configs/calibrations/left_behind_full_pose_good_20260602.json",
        position_reach_limit_m=None,
        rotation_reach_limit_rad=None,
    )
    return ClutchController(mapper)


def update(clutch: ClutchController, grip: float, fresh: bool, x: float = 0.0) -> Pose | None:
    return clutch.update(
        grip=grip,
        stream_fresh=fresh,
        controller_pose=Pose(np.array([x, 0.0, 0.0]), IDENTITY),
        robot_pose=Pose(np.array([0.3, 0.0, 0.2]), IDENTITY),
        wrist_pivot=np.array([0.2, 0.0, 0.2]),
    )


def test_release_holds_until_a_new_grip_edge() -> None:
    clutch = make_clutch()
    assert update(clutch, 1.0, True) is not None
    assert update(clutch, 1.0, True, 0.05) is not None
    assert update(clutch, 0.0, True, 0.05) is None
    assert not clutch.mapper.engaged
    reanchored = update(clutch, 1.0, True, 1.0)
    assert reanchored is not None
    np.testing.assert_allclose(reanchored.position, [0.3, 0.0, 0.2])


def test_stale_stream_reanchors_at_current_pose_without_catch_up() -> None:
    clutch = make_clutch()
    assert update(clutch, 1.0, True) is not None
    assert update(clutch, 1.0, True, 0.05) is not None
    assert update(clutch, 1.0, False) is None
    assert not clutch.mapper.engaged
    reanchored = update(clutch, 1.0, True, 1.0)
    assert reanchored is not None
    np.testing.assert_allclose(reanchored.position, [0.3, 0.0, 0.2])
