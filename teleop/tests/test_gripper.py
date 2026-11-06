import pytest

from widowxai_quest_teleop.gripper import trigger_to_gripper_position


def test_trigger_maps_the_full_gripper_stroke_without_scaling() -> None:
    assert trigger_to_gripper_position(0.0, 0.040, 0.0) == pytest.approx(0.040)
    assert trigger_to_gripper_position(0.5, 0.040, 0.0) == pytest.approx(0.020)
    assert trigger_to_gripper_position(1.0, 0.040, 0.0) == pytest.approx(0.0)
    assert trigger_to_gripper_position(-1.0, 0.040, 0.0) == pytest.approx(0.040)
    assert trigger_to_gripper_position(2.0, 0.040, 0.0) == pytest.approx(0.0)


def test_gripper_mapping_rejects_invalid_endpoints() -> None:
    with pytest.raises(ValueError, match="below"):
        trigger_to_gripper_position(0.5, 0.0, 0.040)
