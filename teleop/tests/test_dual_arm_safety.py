from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from widowxai_quest_teleop import safety
from widowxai_quest_teleop.config import (
    ArmPlacement,
    DUAL_ARM_SIDES,
    DualArmConfigError,
    calibration_acceptance_status,
    dual_live_confirmation_token,
    load_config,
    parse_dual_arm_config,
    require_live_dual_arm_config,
)
from widowxai_quest_teleop.dual_arm_coordinator import build_dual_arm_system
from widowxai_quest_teleop.dual_arm_model import (
    CollisionReport,
    DualArmCollisionModel,
    TabletopGeometry,
)
from widowxai_quest_teleop.telemetry import (
    CAD_TELEMETRY_COLUMNS,
    DUAL_ARM_TELEMETRY_COLUMNS,
    TELEMETRY_COLUMNS,
    TelemetryLogger,
)
from widowxai_quest_teleop.mapping import ClutchPoseMapper
from widowxai_quest_teleop.hardware import HardwareSafetyError
from widowxai_quest_teleop.model import WidowXAIModel
from widowxai_quest_teleop.transport import parse_bimanual_pose_message
from widowxai_quest_teleop.types import Pose
from widowxai_quest_teleop.one_arm_safety import screen_gripper_path, screen_selected_path

DUAL_CONFIG = "configs/dual_widowxai.yaml"
MIRRORED_DUAL_CONFIG = "configs/dual_widowxai_mirrored.yaml"
MUJOCO_CALIBRATED_DUAL_CONFIG = "configs/dual_widowxai_mujoco_calibrated.yaml"
HOME_Q = np.array([0.0, 1.0471975512, 1.3089969390, -1.0471975512, 0.0, 0.0])
REST_Q = np.zeros(6)


def mark_base_transforms_measured(config: dict) -> None:
    for side in ("left", "right"):
        config["arms"][side]["base_transform"]["measurement_status"] = "measured"


def yawed(base_yaw_rad: float) -> np.ndarray:
    """A raised pose yawed about the base only.

    Negating the whole joint vector would fold the arm into its own base and
    produce a self-collision instead of the cross-arm case under test.
    """

    return np.array([base_yaw_rad, 1.0, 1.2, -1.0, 0.0, 0.0])


@pytest.fixture
def dual_config() -> dict:
    return load_config(DUAL_CONFIG)


@pytest.fixture
def mirrored_dual_config() -> dict:
    return load_config(MIRRORED_DUAL_CONFIG)


@pytest.fixture(scope="module")
def collision_model() -> DualArmCollisionModel:
    return DualArmCollisionModel(
        {
            "left": ArmPlacement([0.0, 0.15, 0.0], [1.0, 0.0, 0.0, 0.0]),
            "right": ArmPlacement([0.0, -0.15, 0.0], [1.0, 0.0, 0.0, 0.0]),
        },
        clearance_m=0.03,
    )


# -- configuration fails closed -------------------------------------------


def test_shipped_dual_profile_loads_and_records_both_arms(dual_config) -> None:
    arms = parse_dual_arm_config(dual_config)

    assert arms["left"].controller_hand == "left"
    assert arms["right"].controller_hand == "right"
    # Verified against the physical bench: .2 sits on the operator's left when
    # standing behind the arms, .3 on the right.
    assert arms["left"].robot_ip == "192.168.1.2"
    assert arms["right"].robot_ip == "192.168.1.3"
    assert dual_config["_dual_arm"]["base_separation_m"] == pytest.approx(0.50)
    assert arms["left"].placement.measurement_status == "measured"
    assert arms["right"].placement.measurement_status == "measured"
    assert arms["left"].placement.position_m.tolist() == pytest.approx([0.0, 0.25, 0.0])
    assert arms["right"].placement.position_m.tolist() == pytest.approx([0.0, -0.25, 0.0])
    assert arms["left"].placement.quaternion_wxyz.tolist() == pytest.approx(
        [1.0, 0.0, 0.0, 0.0]
    )
    assert arms["right"].placement.quaternion_wxyz.tolist() == pytest.approx(
        [1.0, 0.0, 0.0, 0.0]
    )
    # The owner requested 60% translation and rotation response. The accepted
    # 70 mm / 0.16 rad per-clutch reach envelope and hard safety gates remain.
    for side in ("left", "right"):
        assert arms[side].settings["translation_scale"] == pytest.approx(0.60)
        assert arms[side].settings["rotation_scale"] == pytest.approx(0.60)
        assert arms[side].settings["position_reach_limit_m"] == pytest.approx(0.070)
        assert arms[side].settings["rotation_reach_limit_rad"] == pytest.approx(0.16)
    assert dual_config["safety"]["cross_arm_clearance_m"] == pytest.approx(0.030)
    assert dual_config["safety"]["collision_recovery_max_joint_delta_rad"] == pytest.approx(
        [0.010] * 6
    )
    assert dual_config["hardware"]["max_feedback_error_rad"] == pytest.approx(0.08)
    # Each arm uses a calibration measured for its own controller hand.
    assert "left" in arms["left"].calibration
    assert "right" in arms["right"].calibration


def test_mujoco_calibrated_dual_profile_is_hardware_disabled() -> None:
    config = load_config(MUJOCO_CALIBRATED_DUAL_CONFIG)
    arms = parse_dual_arm_config(config)

    assert not config["hardware"]["enabled"]
    assert not config["hardware"]["require_explicit_enable"]
    assert arms["left"].calibration_status == "simulation_only"
    assert arms["right"].calibration_status == "simulation_only"
    assert not arms["left"].calibration_accepted
    assert not arms["right"].calibration_accepted
    assert arms["left"].controller_hand == "left"
    assert arms["right"].controller_hand == "right"
    assert config["_dual_arm"]["base_separation_m"] == pytest.approx(0.50)
    assert arms["left"].placement.measurement_status == "measured"
    assert arms["right"].placement.measurement_status == "measured"


def test_same_ip_on_both_arms_is_rejected(dual_config) -> None:
    dual_config["arms"]["left"]["robot_ip"] = dual_config["arms"]["right"]["robot_ip"]
    with pytest.raises(DualArmConfigError, match="same controller IP"):
        parse_dual_arm_config(dual_config)


def test_same_controller_hand_on_both_arms_is_rejected(dual_config) -> None:
    dual_config["arms"]["left"]["controller_hand"] = "right"
    with pytest.raises(DualArmConfigError, match="same logical controller hand"):
        parse_dual_arm_config(dual_config)


def test_missing_base_transform_is_rejected(dual_config) -> None:
    del dual_config["arms"]["right"]["base_transform"]
    with pytest.raises(DualArmConfigError, match="base_transform is required"):
        parse_dual_arm_config(dual_config)


def test_base_transform_measurement_status_must_be_explicit(dual_config) -> None:
    del dual_config["arms"]["right"]["base_transform"]["measurement_status"]
    with pytest.raises(DualArmConfigError, match="measurement_status is required"):
        parse_dual_arm_config(dual_config)


def test_missing_calibration_file_is_rejected(dual_config) -> None:
    dual_config["arms"]["left"]["calibration"] = "configs/calibrations/does_not_exist.json"
    with pytest.raises(DualArmConfigError, match="calibration file is missing"):
        parse_dual_arm_config(dual_config)


def test_coincident_arm_bases_are_rejected(dual_config) -> None:
    # Must stay equal to the shipped right-arm position, or this stops testing
    # coincidence and silently passes for the wrong reason.
    dual_config["arms"]["left"]["base_transform"]["position_m"] = [0.0, -0.25, 0.0]
    with pytest.raises(DualArmConfigError, match="same measured position"):
        parse_dual_arm_config(dual_config)


def test_non_bimanual_quest_mode_is_rejected(dual_config) -> None:
    dual_config["quest"]["mode"] = "single"
    with pytest.raises(DualArmConfigError, match="quest.mode: bimanual"):
        parse_dual_arm_config(dual_config)


def test_safety_block_must_be_explicit(dual_config) -> None:
    dual_config["safety"]["cross_arm_collision"] = "yes"
    with pytest.raises(DualArmConfigError, match="explicit boolean"):
        parse_dual_arm_config(dual_config)


def test_live_output_requires_cross_arm_collision_checking(dual_config) -> None:
    arms = parse_dual_arm_config(dual_config)
    dual_config["safety"]["cross_arm_collision"] = False
    with pytest.raises(DualArmConfigError, match="cross_arm_collision must be enabled"):
        require_live_dual_arm_config(dual_config, arms)


def test_shipped_dual_profile_has_both_calibrations_accepted(dual_config) -> None:
    arms = parse_dual_arm_config(dual_config)
    assert arms["left"].calibration_accepted
    assert arms["right"].calibration_accepted
    assert require_live_dual_arm_config(dual_config, arms) == {}


def test_operator_override_unblocks_only_the_calibration_gate(dual_config) -> None:
    """The override must not become a general safety bypass."""

    mark_base_transforms_measured(dual_config)
    dual_config["arms"]["left"]["calibration"] = (
        "configs/calibrations/left_behind_all_motions_20260723_candidate.json"
    )
    arms = parse_dual_arm_config(dual_config)
    pending = require_live_dual_arm_config(
        dual_config, arms, allow_unvalidated_calibrations=True
    )
    assert set(pending) == {"left"}

    # Recorded in the config, so the run's config snapshot carries it.
    override = dual_config["_dual_arm"]["calibration_override"]
    assert override["authorized_by"] == "operator"
    assert override["scope"] == "calibration_acceptance_only"
    assert set(override["pending"]) == {"left"}

    # Collision and coordinated-hold gates still refuse, override or not.
    dual_config["safety"]["cross_arm_collision"] = False
    with pytest.raises(DualArmConfigError, match="cross_arm_collision must be enabled"):
        require_live_dual_arm_config(
            dual_config, arms, allow_unvalidated_calibrations=True
        )
    dual_config["safety"]["cross_arm_collision"] = True
    dual_config["safety"]["coordinated_fault_hold"] = False
    with pytest.raises(DualArmConfigError, match="coordinated_fault_hold must be enabled"):
        require_live_dual_arm_config(
            dual_config, arms, allow_unvalidated_calibrations=True
        )


def test_calibration_override_does_not_bypass_placeholder_bases(dual_config) -> None:
    for side in ("left", "right"):
        dual_config["arms"][side]["base_transform"]["measurement_status"] = "placeholder"
    arms = parse_dual_arm_config(dual_config)

    with pytest.raises(DualArmConfigError, match="measured base transform"):
        require_live_dual_arm_config(
            dual_config, arms, allow_unvalidated_calibrations=True
        )

    assert "calibration_override" not in dual_config["_dual_arm"]


def test_override_does_not_modify_the_calibration_files(dual_config) -> None:
    """Acceptance evidence in the repo must never be rewritten by a run."""

    from pathlib import Path

    mark_base_transforms_measured(dual_config)
    dual_config["arms"]["left"]["calibration"] = (
        "configs/calibrations/left_behind_all_motions_20260723_candidate.json"
    )
    arms = parse_dual_arm_config(dual_config)
    paths = [Path(arms[side].calibration) for side in ("left", "right")]
    before = [path.read_bytes() for path in paths]

    require_live_dual_arm_config(dual_config, arms, allow_unvalidated_calibrations=True)

    assert [path.read_bytes() for path in paths] == before
    assert calibration_acceptance_status(arms["left"].calibration).startswith(
        "candidate"
    )
    assert calibration_acceptance_status(arms["right"].calibration) == (
        "accepted_by_operator_at_40pct"
    )


def test_calibration_acceptance_is_read_not_inferred() -> None:
    accepted = calibration_acceptance_status(
        "configs/calibrations/right_mirror_20260723_accepted.json"
    )
    candidate = calibration_acceptance_status(
        "configs/calibrations/left_behind_all_motions_20260723_candidate.json"
    )
    assert accepted.startswith("accepted")
    assert not candidate.startswith("accepted")


def test_dual_live_token_can_never_be_a_single_arm_token(dual_config) -> None:
    arms = parse_dual_arm_config(dual_config)
    token = dual_live_confirmation_token(arms)

    # Order is left-then-right, so swapping which physical arm is which side
    # changes the token. That is deliberate: a token minted for one side
    # assignment must not authorise the opposite one.
    assert token == "LIVE-WIDOWXAI-DUAL-192.168.1.2-192.168.1.3"
    assert token != "LIVE-WIDOWXAI-DUAL-192.168.1.3-192.168.1.2"
    assert token != "LIVE-WIDOWXAI-192.168.1.2"
    assert token != "LIVE-WIDOWXAI-192.168.1.3"
    assert not token.startswith("LIVE-WIDOWXAI-192.")


# -- mirrored operator-view profile ----------------------------------------


def mirror_pair_sample(sequence: int, position, mapping_mode: str):
    """One bimanual frame with both hands at the same raw WebXR pose."""

    block = {
        "tracked": True,
        "mapping_mode": mapping_mode,
        "position": list(position),
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "grip": 0.9,
        "trigger": 0.0,
    }
    return parse_bimanual_pose_message(
        {
            "type": "bimanual_pose",
            "schema_version": 2,
            "sequence": sequence,
            "capture_monotonic_ms": float(sequence) * 11.0,
            "capture_epoch_ms": 1.7e12 + sequence,
            "send_monotonic_ms": float(sequence) * 11.0,
            "left": dict(block),
            "right": dict(block),
        },
        mapping_modes={"left": mapping_mode, "right": mapping_mode},
    )


def transported_translation(hand: str, mapping_mode: str, calibration: str):
    """Arm-frame translation for one raw hand step, through the real packet path.

    The sagittal reflection lives in the transport, not the mapper, so a test
    that drives ``ClutchPoseMapper`` directly would never exercise mirror mode.
    """

    start = np.array([0.30, 0.20, 0.25])
    step = np.array([0.05, 0.03, 0.02])
    first = mirror_pair_sample(1, start, mapping_mode).for_hand(hand)
    second = mirror_pair_sample(2, start + step, mapping_mode).for_hand(hand)
    mapper = ClutchPoseMapper(
        calibration,
        position_reach_limit_m=None,
        rotation_reach_limit_rad=None,
    )
    anchor = Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mapper.engage(first.controller_pose, anchor, np.zeros(3))
    return mapper.update(second.controller_pose, anchor).position


def test_mirrored_profile_assigns_the_near_hand_to_each_arm(mirrored_dual_config) -> None:
    arms = parse_dual_arm_config(mirrored_dual_config)

    # Facing the arms, +y (192.168.1.2) is on the operator's right, so the
    # hand-to-arm assignment is swapped relative to the Behind profile.
    assert arms["left"].robot_ip == "192.168.1.2"
    assert arms["left"].controller_hand == "right"
    assert arms["right"].robot_ip == "192.168.1.3"
    assert arms["right"].controller_hand == "left"
    assert arms["left"].mapping_mode == "mirror"
    assert arms["right"].mapping_mode == "mirror"
    assert mirrored_dual_config["_dual_arm"]["base_separation_m"] == pytest.approx(0.50)


def test_mirrored_profile_keeps_every_dual_arm_safety_gate(mirrored_dual_config) -> None:
    safety = mirrored_dual_config["safety"]
    assert safety["cross_arm_collision"] is True
    assert safety["coordinated_fault_hold"] is True
    assert safety["cross_arm_clearance_m"] == pytest.approx(0.030)
    assert mirrored_dual_config["hardware"]["max_demo_duration_s"] == pytest.approx(60.0)


def test_mirrored_profile_has_both_calibrations_accepted(
    mirrored_dual_config,
) -> None:
    arms = parse_dual_arm_config(mirrored_dual_config)
    assert arms["left"].calibration_accepted
    assert arms["right"].calibration_accepted
    assert require_live_dual_arm_config(mirrored_dual_config, arms) == {}


def test_calibration_captured_for_the_other_hand_is_rejected(mirrored_dual_config) -> None:
    mirrored_dual_config["arms"]["left"]["calibration"] = mirrored_dual_config["arms"][
        "right"
    ]["calibration"]
    with pytest.raises(DualArmConfigError, match="captured for the left hand"):
        parse_dual_arm_config(mirrored_dual_config)


def test_calibration_captured_under_the_other_mapping_mode_is_rejected(
    mirrored_dual_config,
) -> None:
    mirrored_dual_config["arms"]["left"]["mapping_mode"] = "real"
    with pytest.raises(DualArmConfigError, match="captured under mirror"):
        parse_dual_arm_config(mirrored_dual_config)


def test_mirror_mode_is_a_proper_rotation_of_the_behind_mapping() -> None:
    """Mirrored is a 180 degree yaw, not a reflection, once the page reflection
    and the calibration's cancelling task frame are both applied."""

    yaw_180 = np.array([-1.0, -1.0, 1.0])
    for hand, behind, mirror in (
        (
            "left",
            "configs/calibrations/left_behind_all_motions_20260812_accepted.json",
            "configs/calibrations/left_mirror_all_motions_20260812_accepted.json",
        ),
        (
            "right",
            "configs/calibrations/right_behind_all_motions_20260812_accepted.json",
            "configs/calibrations/right_mirror_20260723_accepted.json",
        ),
    ):
        behind_translation = transported_translation(hand, "real", behind)
        mirror_translation = transported_translation(hand, "mirror", mirror)
        assert np.allclose(mirror_translation, yaw_180 * behind_translation)
        # Left/right and forward/back reverse, up/down does not.
        assert not np.allclose(mirror_translation, behind_translation)


def test_mirroring_both_hands_leaves_them_independent() -> None:
    """Mirror is a per-hand input convention, so it cannot cross the arms."""

    sample = mirror_pair_sample(1, np.array([0.30, 0.20, 0.25]), "mirror")
    left = sample.for_hand("left")
    right = sample.for_hand("right")
    # Same raw pose in, same mirrored pose out: neither hand's reflection is
    # taken about the other hand or about the shared world frame.
    assert np.allclose(left.controller_pose.position, right.controller_pose.position)
    assert np.allclose(left.controller_pose.position, [-0.30, 0.20, 0.25])


def test_one_hand_left_on_the_wrong_page_mode_discards_the_whole_frame() -> None:
    """Both dropdowns must read Mirrored; a half-switched page is not partial."""

    with pytest.raises(ValueError, match="left mapping mode real, expected mirror"):
        parse_bimanual_pose_message(
            {
                "type": "bimanual_pose",
                "schema_version": 2,
                "sequence": 1,
                "capture_monotonic_ms": 11.0,
                "capture_epoch_ms": 1.7e12,
                "send_monotonic_ms": 11.0,
                "left": {"tracked": True, "mapping_mode": "real",
                         "position": [0.3, 0.2, 0.25],
                         "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
                "right": {"tracked": True, "mapping_mode": "mirror",
                          "position": [0.3, -0.2, 0.25],
                          "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            },
            mapping_modes={"left": "mirror", "right": "mirror"},
        )


def test_expected_mapping_modes_are_keyed_by_controller_hand(mirrored_dual_config) -> None:
    """The launchers key the transport by hand; keying by arm side would send
    the swapped profile's expectations to the wrong controller."""

    arms = parse_dual_arm_config(mirrored_dual_config)
    by_hand = {
        arms[side].controller_hand: arms[side].mapping_mode for side in DUAL_ARM_SIDES
    }
    assert set(by_hand) == {"left", "right"}
    assert by_hand == {"left": "mirror", "right": "mirror"}


# -- combined scene --------------------------------------------------------


def test_combined_scene_contains_both_prefixed_arms(collision_model) -> None:
    model = collision_model
    assert model.model.nq == 16
    assert model.model.nu == 14
    assert not set(model.arm_qpos_indices["left"]) & set(model.arm_qpos_indices["right"])
    assert model.ee_site_ids["left"] != model.ee_site_ids["right"]
    assert model.wrist_site_ids["left"] != model.wrist_site_ids["right"]


def test_forward_places_both_arms_and_never_zeroes_the_other(collision_model) -> None:
    """Zeroing the shared qpos while setting one arm would silently evaluate
    the other arm at its zero pose, making cross-arm screening meaningless."""

    collision_model.forward({"left": HOME_Q, "right": REST_Q}, {"left": 0.04, "right": 0.0})
    assert np.allclose(collision_model.data.qpos[collision_model.arm_qpos_indices["left"]], HOME_Q)
    assert np.allclose(collision_model.data.qpos[collision_model.arm_qpos_indices["right"]], REST_Q)

    collision_model.forward({"left": REST_Q, "right": HOME_Q}, {"left": 0.0, "right": 0.04})
    assert np.allclose(collision_model.data.qpos[collision_model.arm_qpos_indices["right"]], HOME_Q)


def test_closed_gripper_is_not_reported_as_a_self_collision(collision_model) -> None:
    """The carriage whitelist must carry the attach prefix, or every closed
    gripper reads as a self-collision."""

    report = collision_model.check(
        {"left": REST_Q, "right": REST_Q}, {"left": 0.0, "right": 0.0}
    )
    assert report.colliding is False


def test_arms_reaching_toward_each_other_are_rejected(collision_model) -> None:
    clear = collision_model.check(
        {"left": HOME_Q, "right": HOME_Q}, {"left": 0.04, "right": 0.04}
    )
    assert clear.colliding is False

    # Both arms yaw inward across the 300 mm gap.
    report = collision_model.check(
        {"left": yawed(-0.5), "right": yawed(0.5)}, {"left": 0.04, "right": 0.04}
    )
    assert report.colliding is True
    assert report.kind.startswith("cross-arm")
    assert report.side is None, "a cross-arm hazard is not attributable to one arm"


def test_clearance_margin_rejects_before_contact() -> None:
    placements = {
        "left": ArmPlacement([0.0, 0.15, 0.0], [1.0, 0.0, 0.0, 0.0]),
        "right": ArmPlacement([0.0, -0.15, 0.0], [1.0, 0.0, 0.0, 0.0]),
    }
    strict = DualArmCollisionModel(placements, clearance_m=0.03)
    contact_only = DualArmCollisionModel(placements, clearance_m=0.0)

    q = {"left": yawed(-0.20), "right": yawed(0.20)}
    grippers = {"left": 0.04, "right": 0.04}

    assert contact_only.check(q, grippers).colliding is False
    strict_report = strict.check(q, grippers)
    assert strict_report.colliding is True
    assert strict_report.kind == "cross-arm-clearance"


def test_cross_arm_distance_never_over_reports_clearance(collision_model) -> None:
    """mj_geomDistance returns a spurious 0.0 for some box pairs in mujoco
    3.8.1. The bounding-sphere guard must keep the reported separation a valid
    lower bound, and must not introduce discontinuities along a smooth path."""

    separations = []
    for alpha in np.linspace(0.0, 1.0, 41):
        collision_model.forward(
            {"left": REST_Q + alpha * (HOME_Q - REST_Q), "right": HOME_Q},
            {"left": 0.04, "right": 0.04},
        )
        separations.append(collision_model.minimum_cross_arm_distance())

    separations = np.array(separations)
    assert np.all(separations > 0.0), "spurious zero separation on a clear path"
    # No single step may collapse the reported clearance by more than the
    # cutoff itself; a spurious zero shows up as exactly that discontinuity.
    assert np.max(np.abs(np.diff(separations))) < 0.06


def test_simultaneous_path_screening_covers_both_arms(collision_model) -> None:
    clear = collision_model.first_collision_on_path(
        {"left": REST_Q, "right": REST_Q},
        {"left": HOME_Q, "right": HOME_Q},
        start_gripper={"left": 0.0, "right": 0.0},
        end_gripper={"left": 0.04, "right": 0.04},
        samples=51,
    )
    assert clear is None

    found = collision_model.first_collision_on_path(
        {"left": REST_Q, "right": REST_Q},
        {"left": yawed(-0.6), "right": yawed(0.6)},
        samples=51,
    )
    assert found is not None
    alpha, report = found
    assert 0.0 < alpha <= 1.0
    assert report.kind.startswith("cross-arm")


def test_measured_500mm_layout_keeps_rest_and_home_clear(dual_config) -> None:
    dual_config["simulation_environment"] = {
        "status": "provisional",
        "tabletop": {
            "width_m": 1.0,
            "depth_m": 0.7,
            "thickness_m": 0.04,
            "rear_edge_x_m": -0.0508,
            "top_z_m": 0.0,
        },
    }
    arms, model, _ = build_system(dual_config)

    assert model.tabletop is not None
    assert model.tabletop.width_m == pytest.approx(1.0)
    assert model.tabletop.depth_m == pytest.approx(0.7)
    assert model.check(
        {"left": REST_Q, "right": REST_Q}, {"left": 0.0, "right": 0.0}
    ).colliding is False
    assert model.check(
        {"left": HOME_Q, "right": HOME_Q}, {"left": 0.04, "right": 0.04}
    ).colliding is False
    assert arms["left"].arm_config.placement.position_m[1] == pytest.approx(0.25)
    assert arms["right"].arm_config.placement.position_m[1] == pytest.approx(-0.25)


def test_one_arm_commissioning_paths_clear_with_other_at_measured_rest(dual_config) -> None:
    arms = parse_dual_arm_config(dual_config)
    single = WidowXAIModel(dual_config["model"]["xml_path"])
    dual = DualArmCollisionModel(
        {side: arms[side].placement for side in ("left", "right")},
        xml_path=dual_config["model"]["xml_path"],
        clearance_m=dual_config["safety"]["cross_arm_clearance_m"],
    )

    minimum = screen_selected_path(
        single,
        dual,
        moving_side="left",
        moving_start=REST_Q,
        moving_end=HOME_Q,
        holding_q=REST_Q,
        moving_gripper_m=0.0,
        holding_gripper_m=0.0,
        samples=501,
        clearance_m=0.03,
    )

    assert minimum >= 0.03


def test_one_arm_commissioning_allows_only_a_clearing_marginal_start(dual_config) -> None:
    arms = parse_dual_arm_config(dual_config)
    single = WidowXAIModel(dual_config["model"]["xml_path"])
    dual = DualArmCollisionModel(
        {side: arms[side].placement for side in ("left", "right")},
        xml_path=dual_config["model"]["xml_path"],
        clearance_m=0.03,
    )
    drooped_right = np.array(
        [-0.001335, -0.004005, -0.005150, -0.001717, 0.000572, -0.000191]
    )

    minimum = screen_selected_path(
        single,
        dual,
        moving_side="right",
        moving_start=drooped_right,
        moving_end=HOME_Q,
        holding_q=REST_Q,
        moving_gripper_m=0.0,
        holding_gripper_m=0.0,
        samples=501,
        clearance_m=0.03,
        allow_marginal_start_m=0.002,
    )
    assert minimum >= 0.03

    with pytest.raises(HardwareSafetyError, match="never clears"):
        screen_selected_path(
            single,
            dual,
            moving_side="right",
            moving_start=drooped_right,
            moving_end=drooped_right,
            holding_q=REST_Q,
            moving_gripper_m=0.0,
            holding_gripper_m=0.0,
            samples=11,
            clearance_m=0.03,
            allow_marginal_start_m=0.002,
        )


def test_stationary_arm_guard_screens_full_gripper_stroke(dual_config) -> None:
    arms = parse_dual_arm_config(dual_config)
    dual = DualArmCollisionModel(
        {side: arms[side].placement for side in ("left", "right")},
        xml_path=dual_config["model"]["xml_path"],
        clearance_m=0.03,
    )

    minimum = screen_gripper_path(
        dual,
        moving_side="right",
        moving_q=HOME_Q,
        holding_q=REST_Q,
        moving_start_gripper_m=0.0,
        moving_end_gripper_m=0.04,
        holding_gripper_m=0.0,
        clearance_m=0.03,
        samples=501,
    )

    assert minimum >= 0.03


def test_tabletop_contact_is_an_environment_collision(dual_config) -> None:
    dual_config["simulation_environment"] = {
        "tabletop": {
            "width_m": 1.0,
            "depth_m": 0.7,
            "thickness_m": 0.04,
            "rear_edge_x_m": -0.0508,
            "top_z_m": 0.0,
        }
    }
    _, model, _ = build_system(dual_config)
    below_table_q = np.array(
        [0.721916, 3.009115, 0.564246, -0.999520, 0.331046, -2.774283]
    )

    report = model.check(
        {"left": below_table_q, "right": HOME_Q},
        {"left": 0.04, "right": 0.04},
    )

    assert report.colliding is True
    assert report.kind == "environment"
    assert report.side == "left"
    assert report.bodies is not None
    assert report.bodies[1] == "tabletop"


def test_tabletop_dimensions_fail_closed() -> None:
    with pytest.raises(ValueError, match="dimensions must be finite and positive"):
        TabletopGeometry(1.0, 0.0, 0.04, -0.10)


# -- coordinated control ---------------------------------------------------


def bimanual_sample(
    sequence: int,
    *,
    grip: float = 0.9,
    left_tracked: bool = True,
    x_offset: float = 0.0,
    home_button: bool | None = None,
):
    def hand(sign: float, tracked: bool) -> dict:
        block = {"tracked": tracked, "mapping_mode": "real"}
        if tracked:
            block.update(
                {
                    "position": [0.30 + x_offset, sign * 0.20, 0.25],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "grip": grip,
                    "trigger": 0.0,
                }
            )
        return block

    return parse_bimanual_pose_message(
        {
            "type": "bimanual_pose",
            "schema_version": 2,
            "sequence": sequence,
            "capture_monotonic_ms": float(sequence) * 11.0,
            "capture_epoch_ms": 1.7e12 + sequence,
            "send_monotonic_ms": float(sequence) * 11.0,
            "home_button": home_button,
            "left": hand(1.0, left_tracked),
            "right": hand(-1.0, True),
        }
    )


def build_system(dual_config):
    arms_config = parse_dual_arm_config(dual_config)
    return build_dual_arm_system(
        dual_config,
        arms_config,
        initial_q=HOME_Q,
        initial_gripper_m=0.04,
        control_gripper=True,
    )


@pytest.fixture
def home_run(dual_config, monkeypatch):
    arms, collision, coordinator = build_system(dual_config)
    for side, arm in arms.items():
        arm.q_command[0] += 0.10 if side == "left" else -0.10
        arm.q_feedback = arm.q_command.copy()
        arm.gripper_command_m = 0.025
        arm.hold()
    sequence = 0
    now_ns = 1_000_000_000
    monkeypatch.setattr(safety, "perf_counter_ns", lambda: now_ns)

    def step(pressed, *, grip=0.0, tracked=True, generation=0, delay_ms=11):
        nonlocal sequence, now_ns
        sequence += 1
        now_ns += delay_ms * 1_000_000
        sample = None if pressed is None else replace(
            bimanual_sample(sequence, grip=grip, left_tracked=tracked, home_button=pressed),
            pc_arrival_monotonic_ns=now_ns,
            reconnect_generation=generation,
        )
        tick = coordinator.step(sample, limiter_dt=0.011)
        for arm in arms.values():
            arm.q_feedback = arm.q_command.copy()
        return tick

    return arms, collision, step


def test_home_needs_release_then_hold_and_preserves_grippers(home_run):
    arms, _, step = home_run
    for _ in range(40):
        assert step(True).control_state == "home_hold"
    assert abs(arms["left"].q_command[0] - HOME_Q[0]) == pytest.approx(0.10)
    step(False)
    for _ in range(20):
        assert step(True).control_state == "normal"
    for _ in range(300):
        tick = step(True)
        if tick.control_state == "at_home":
            break
    assert tick.control_state == "at_home"
    for arm in arms.values():
        np.testing.assert_allclose(arm.q_command, HOME_Q, atol=1e-4)
        assert arm.gripper_command_m == pytest.approx(0.025)
        assert not arm.mapper.engaged


@pytest.mark.parametrize("interruption", ["release", "grip", "tracking", "stale", "reconnect", "collision"])
def test_home_interruption_holds_both_arms(home_run, monkeypatch, interruption):
    arms, collision, step = home_run
    for _ in range(8):
        step(False)
    for _ in range(42):
        tick = step(True)
    assert tick.control_state == "returning_home"
    previous = {side: arm.q_command.copy() for side, arm in arms.items()}
    if interruption == "release":
        step(False)
    elif interruption == "grip":
        step(True, grip=0.9)
    elif interruption == "tracking":
        step(True, tracked=False)
    elif interruption == "stale":
        step(None, delay_ms=1000)
    elif interruption == "reconnect":
        step(True, generation=1)
    else:
        with monkeypatch.context() as patch:
            patch.setattr(collision, "check", lambda *args: CollisionReport(colliding=True))
            assert step(True).control_state == "home_blocked"
    # The press cannot resume after a fault. A release needs a new dwell.
    for _ in range(10 if interruption == "release" else 50):
        step(True, generation=1 if interruption == "reconnect" else 0)
    for side, arm in arms.items():
        np.testing.assert_array_equal(arm.q_command, previous[side])


def test_each_controller_drives_only_its_assigned_arm(dual_config) -> None:
    arms, _, coordinator = build_system(dual_config)

    assert arms["left"].controller_hand == "left"
    assert arms["right"].controller_hand == "right"
    for sequence in range(8):
        coordinator.step(bimanual_sample(sequence), limiter_dt=0.011)

    # Both arms clutched from their own controller.
    assert arms["left"].mapper.engaged
    assert arms["right"].mapper.engaged
    assert arms["left"].mapper.reanchor_generation >= 1
    assert arms["right"].mapper.reanchor_generation >= 1


def test_losing_one_controller_holds_only_that_arm(dual_config) -> None:
    arms, _, coordinator = build_system(dual_config)
    for sequence in range(8):
        coordinator.step(bimanual_sample(sequence), limiter_dt=0.011)

    left_committed = arms["left"].q_command.copy()
    tick = coordinator.step(bimanual_sample(8, left_tracked=False), limiter_dt=0.011)

    assert tick.proposals["left"].tracked is False
    assert tick.proposals["right"].tracked is True
    assert tick.proposals["left"].active is False
    assert "left" in tick.held_sides
    assert "right" not in tick.held_sides
    assert tick.stream_lost is False
    np.testing.assert_allclose(tick.proposals["left"].q_command, left_committed)
    np.testing.assert_allclose(arms["left"].q_command, left_committed)


def test_losing_the_shared_stream_holds_both_arms(dual_config) -> None:
    arms, _, coordinator = build_system(dual_config)
    for sequence in range(8):
        coordinator.step(bimanual_sample(sequence), limiter_dt=0.011)

    # No packet at all: the relay or the WebXR session is gone. Freshness is
    # judged against the wall clock, so the stale timeout must actually pass.
    time.sleep(float(dual_config["quest"]["stale_timeout_s"]) * 1.5)
    tick = coordinator.step(None, limiter_dt=0.011)

    assert tick.stream_lost is True
    assert set(tick.held_sides) == {"left", "right"}
    assert tick.proposals["left"].active is False
    assert tick.proposals["right"].active is False


def test_a_cross_arm_collision_rejects_both_arms_commands(dual_config) -> None:
    """Neither arm may proceed from a combined state that was never sent."""

    arms, collision_model, coordinator = build_system(dual_config)
    for sequence in range(8):
        coordinator.step(bimanual_sample(sequence), limiter_dt=0.011)

    committed = {side: arms[side].q_command.copy() for side in ("left", "right")}

    class AlwaysColliding:
        def check(self, *_args, **_kwargs):
            from widowxai_quest_teleop.dual_arm_model import CollisionReport

            return CollisionReport(
                colliding=True, kind="cross-arm", bodies=("left_link_6", "right_link_6")
            )

    coordinator.collision_model = AlwaysColliding()
    tick = coordinator.step(bimanual_sample(9), limiter_dt=0.011)

    assert tick.accepted is False
    assert "cross-arm" in tick.fault_reason
    assert set(tick.held_sides) == {"left", "right"}
    for side in ("left", "right"):
        assert np.allclose(arms[side].q_command, committed[side]), (
            f"{side} arm advanced past a rejected combined state"
        )
        assert arms[side].rejected_commands == 1


def test_a_clear_tick_resumes_through_a_bounded_collision_ramp(dual_config) -> None:
    """Recovery must not accept a far-ahead clear target in one tick."""

    arms, _, coordinator = build_system(dual_config)
    for sequence in range(8):
        coordinator.step(bimanual_sample(sequence), limiter_dt=0.011)

    class ToggleCollision:
        colliding = True

        def check(self, *_args, **_kwargs):
            if self.colliding:
                return CollisionReport(
                    colliding=True,
                    kind="cross-arm-clearance",
                    separation_m=0.029,
                )
            return CollisionReport(colliding=False, separation_m=0.031)

    model = ToggleCollision()
    coordinator.collision_model = model
    blocked = coordinator.step(bimanual_sample(8, x_offset=0.10), limiter_dt=0.011)
    assert blocked.accepted is False
    assert set(blocked.held_sides) == {"left", "right"}
    committed = {side: arms[side].q_command.copy() for side in ("left", "right")}

    model.colliding = False
    resumed = coordinator.step(bimanual_sample(9, x_offset=0.10), limiter_dt=0.011)
    assert resumed.accepted is True
    assert resumed.collision.colliding is False
    assert resumed.control_state == "collision_recovery"
    assert any(
        "collision_recovery" in resumed.proposals[side].limiter_flags
        for side in ("left", "right")
    )
    for side in ("left", "right"):
        assert arms[side].rejected_commands == 1
        assert np.max(np.abs(arms[side].q_command - committed[side])) <= 0.010 + 1e-12
        np.testing.assert_allclose(arms[side].feedforward_velocity, np.zeros(6))


def test_external_arm_fault_rejects_and_holds_both_commands(dual_config) -> None:
    arms, _, coordinator = build_system(dual_config)
    for sequence in range(8):
        coordinator.step(bimanual_sample(sequence), limiter_dt=0.011)
    committed = {side: arms[side].q_command.copy() for side in ("left", "right")}

    tick = coordinator.step(
        bimanual_sample(8),
        limiter_dt=0.011,
        external_fault="left driver feedback unavailable",
    )

    assert tick.accepted is False
    assert tick.fault_reason == "left driver feedback unavailable"
    assert set(tick.held_sides) == {"left", "right"}
    for side in ("left", "right"):
        np.testing.assert_allclose(arms[side].q_command, committed[side])
        assert arms[side].rejected_commands == 1


def test_reconnect_releases_both_arms_then_reanchors_together(dual_config) -> None:
    """A reconnect means the controllers and arms may no longer agree on where
    'here' is. Both arms must drop their clutch anchor, and neither may
    re-engage until the stream proves fresh again."""

    arms, _, coordinator = build_system(dual_config)
    for sequence in range(8):
        coordinator.step(bimanual_sample(sequence), limiter_dt=0.011)
    assert arms["left"].mapper.engaged and arms["right"].mapper.engaged
    generations = {
        side: arms[side].mapper.reanchor_generation for side in ("left", "right")
    }

    reconnected = bimanual_sample(9)
    reconnected = type(reconnected)(**{**reconnected.__dict__, "reconnect_generation": 5})
    tick = coordinator.step(reconnected, limiter_dt=0.011)

    # Immediately after the reconnect neither arm may drive.
    for side in ("left", "right"):
        assert tick.proposals[side].active is False
        assert arms[side].mapper.engaged is False

    # Once the stream proves fresh again, both re-anchor on a new generation.
    for sequence in range(10, 16):
        sample = bimanual_sample(sequence)
        sample = type(sample)(**{**sample.__dict__, "reconnect_generation": 5})
        coordinator.step(sample, limiter_dt=0.011)

    for side in ("left", "right"):
        assert arms[side].mapper.engaged is True
        assert arms[side].mapper.reanchor_generation > generations[side]


def test_coordinator_rejects_two_arms_claiming_one_controller(dual_config) -> None:
    arms, collision_model, _ = build_system(dual_config)
    arms["left"].controller_hand = "right"
    from widowxai_quest_teleop.dual_arm_coordinator import DualArmCoordinator

    with pytest.raises(ValueError, match="same controller hand"):
        DualArmCoordinator(arms, collision_model)


def test_each_arm_owns_an_independent_model_and_solver(dual_config) -> None:
    """Two DecoupledIK instances sharing one WidowXAIModel would corrupt each
    other's Jacobians through the shared MjData scratch."""

    arms, _, _ = build_system(dual_config)

    assert arms["left"].model is not arms["right"].model
    assert arms["left"].model.data is not arms["right"].model.data
    assert arms["left"].solver is not arms["right"].solver
    assert arms["left"].mapper is not arms["right"].mapper
    assert arms["left"].watchdog is not arms["right"].watchdog
    assert arms["left"].controller_filter is not arms["right"].controller_filter


# -- telemetry -------------------------------------------------------------


def test_dual_telemetry_declares_both_arms_and_skew() -> None:
    for side in ("left", "right"):
        assert f"{side}_q_cmd" in DUAL_ARM_TELEMETRY_COLUMNS
        assert f"{side}_ik_status" in DUAL_ARM_TELEMETRY_COLUMNS
        assert f"{side}_tracked" in DUAL_ARM_TELEMETRY_COLUMNS
        assert f"{side}_stream_fresh" in DUAL_ARM_TELEMETRY_COLUMNS
        assert f"{side}_feedback_reference_state" in DUAL_ARM_TELEMETRY_COLUMNS
        assert f"{side}_feedback_newest_command_age_ms" in DUAL_ARM_TELEMETRY_COLUMNS
        assert f"{side}_feedback_history_span_ms" in DUAL_ARM_TELEMETRY_COLUMNS
        assert f"{side}_command_send_duration_ms" in DUAL_ARM_TELEMETRY_COLUMNS
    assert "command_skew_ms" in DUAL_ARM_TELEMETRY_COLUMNS
    assert "control_state" in DUAL_ARM_TELEMETRY_COLUMNS
    assert "cross_arm_collision" in DUAL_ARM_TELEMETRY_COLUMNS
    assert "coordinated_hold" in DUAL_ARM_TELEMETRY_COLUMNS
    # No unprefixed single-arm joint column may leak in and hide an arm.
    assert "q_cmd" not in DUAL_ARM_TELEMETRY_COLUMNS


def test_undeclared_telemetry_column_raises_instead_of_being_dropped(tmp_path) -> None:
    """DictWriter silently discards unknown keys; a lost arm column would look
    like a clean run."""

    with TelemetryLogger(
        "t", {}, tmp_path, columns=list(TELEMETRY_COLUMNS), strict_columns=True
    ) as logger:
        with pytest.raises(ValueError, match="undeclared columns"):
            logger.log(right_q_cmd=[0.0] * 6)


def test_every_launcher_logs_only_columns_its_schema_declares() -> None:
    """Guards against an arm's data being silently dropped by DictWriter.

    CAD, single-arm, and dual-arm launchers each have an explicit schema. A new
    launcher must be assigned here rather than silently dropping source or
    feedback evidence into the wrong schema.
    """

    import ast
    from pathlib import Path

    single = set(TELEMETRY_COLUMNS)
    cad = set(CAD_TELEMETRY_COLUMNS)
    dual = set(DUAL_ARM_TELEMETRY_COLUMNS)
    dual_scripts = {"run_dual_sim.py", "run_dual_hardware.py"}
    cad_scripts = {"run_cad_sim.py", "run_cad_hardware.py"}

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    undeclared: dict[str, list[str]] = {}
    for path in sorted(scripts_dir.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "log"
            ):
                continue
            keys = {keyword.arg for keyword in node.keywords if keyword.arg}
            if not keys:
                continue
            declared = (
                dual
                if path.name in dual_scripts
                else cad
                if path.name in cad_scripts
                else single
            )
            extra = sorted(keys - declared)
            if extra:
                undeclared.setdefault(path.name, []).extend(extra)

    assert not undeclared, f"launcher logs columns no schema declares: {undeclared}"


def test_dual_telemetry_counts_ik_failures_from_both_arms(tmp_path) -> None:
    with TelemetryLogger(
        "t",
        {},
        tmp_path,
        columns=DUAL_ARM_TELEMETRY_COLUMNS,
        ik_status_columns=("left_ik_status", "right_ik_status"),
        strict_columns=True,
    ) as logger:
        logger.log(left_ik_status="ok", right_ik_status="ok")
        logger.log(left_ik_status="numerical_failure", right_ik_status="ok")
        logger.log(left_ik_status="ok", right_ik_status="nonfinite_rejected")
        logger.log(left_ik_status="orientation_parked", right_ik_status="ok")
    assert logger.rows == 4
    assert logger.ik_failures == 2
