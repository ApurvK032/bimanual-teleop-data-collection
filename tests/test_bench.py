"""Bench wizard: profile generation from the examples, validation, launcher commands (no arm, no network)."""
from __future__ import annotations

import os

import pytest
import yaml

from bimanual_collect import bench, rig

STATE = bench.DEFAULTS | {"arms": 2, "left_ip": "192.0.2.3", "right_ip": "192.0.2.2", "identified": True,
                          "spacing_mm": 590, "spacing_measured": True, "response": "fast", "tracking_stop_rad": 0.12,
                          "cap_s": 120, "gripper": True}


@pytest.fixture(autouse=True)
def documentation_subnet(monkeypatch):
    """Tests use the documentation range 192.0.2.0/24 for the arm network (no real lab addresses in the repo)."""
    monkeypatch.setattr(bench, "SOURCE_IP", "192.0.2.10")
    monkeypatch.setattr(bench, "SUBNET", "192.0.2.0/24")


@pytest.fixture
def local(tmp_path, monkeypatch):
    """Point the bench files and the rig config at a temp folder."""
    monkeypatch.setattr(bench, "LOCAL", tmp_path)
    monkeypatch.setattr(bench, "STATE_FILE", tmp_path / "bench.json")
    monkeypatch.setattr(bench, "PROFILE", tmp_path / "bench.yaml")
    monkeypatch.setattr(bench, "SIDE_PROFILES", {s: tmp_path / f"bench_{s}.yaml" for s in ("left", "right")})
    rig_file = tmp_path / "rig.yaml"
    rig_file.write_text(rig.CANDIDATES[1].read_text())
    monkeypatch.setattr(rig, "CANDIDATES", (rig_file, rig.CANDIDATES[1]))
    monkeypatch.setenv("BIMANUAL_RIG", str(rig_file))
    rig.load.cache_clear()
    yield tmp_path
    rig.load.cache_clear()


def test_set_yaml_values_keeps_layout_and_comments():
    text = "a:\n  b: 1  # keep me\n  c:\n    d: 2  # example value\nb: 3\n"
    out = bench.set_yaml_values(text, {("a", "b"): "5", ("a", "c", "d"): "[1, 2]", ("b",): "x"})
    assert out == "a:\n  b: 5  # keep me\n  c:\n    d: [1, 2]\nb: x\n"
    with pytest.raises(ValueError, match="a.zz"):
        bench.set_yaml_values(text, {("a", "zz"): "1"})


def test_dual_profile_values():
    profile = yaml.safe_load(bench.dual_profile(bench.validate(STATE)))
    left, right = profile["arms"]["left"], profile["arms"]["right"]
    assert (left["robot_ip"], right["robot_ip"]) == ("192.0.2.3", "192.0.2.2")
    assert left["base_transform"]["position_m"] == [0.0, 0.295, 0.0]
    assert right["base_transform"]["position_m"] == [0.0, -0.295, 0.0]
    assert left["base_transform"]["measurement_status"] == "measured"
    assert (left["translation_scale"], left["rotation_scale"]) == (0.70, 0.60)
    hardware = profile["hardware"]
    assert hardware["max_demo_duration_s"] == 120.0 and hardware["max_feedback_error_rad"] == 0.12
    assert hardware["workstation_preflight"]["expected_robot_source_ip"] == bench.SOURCE_IP
    # everything else is the example's, untouched
    example = yaml.safe_load(bench.DUAL_EXAMPLE.read_text())
    assert profile["safety"] == example["safety"] and profile["control"] == example["control"]


def test_single_profiles_follow_the_hand():
    right = yaml.safe_load(bench.single_profile("right", "192.0.2.2", bench.validate(STATE), response="validated"))
    assert right["quest"]["hand"] == "right" and "right_behind" in right["quest"]["calibration"]
    assert right["quest"]["translation_scale"] == 0.40 and right["hardware"]["robot_ip"] == "192.0.2.2"


def test_validation_refuses_bad_benches():
    with pytest.raises(ValueError, match="different addresses"):
        bench.validate(STATE | {"right_ip": "192.0.2.3"})
    with pytest.raises(ValueError, match="must be on"):
        bench.validate(STATE | {"left_ip": "198.51.100.3"})
    with pytest.raises(ValueError, match="PC's own"):
        bench.validate(STATE | {"left_ip": "192.0.2.10"})
    with pytest.raises(ValueError, match="separation"):
        bench.validate(STATE | {"spacing_mm": 100})
    with pytest.raises(ValueError, match="tracking stop"):
        bench.validate(STATE | {"tracking_stop_rad": 0.3})


def test_save_writes_profiles_and_points_the_rig(local):
    state = bench.save_state(STATE)
    written = bench.write_profiles(state)
    assert set(written) == {"left", "right", "profile"}
    assert yaml.safe_load(bench.PROFILE.read_text())["arms"]["left"]["robot_ip"] == "192.0.2.3"
    settings = rig.load()
    assert settings.arms == 2 and settings.arm_profile.name == "bench.yaml"
    # a changed address invalidates the side check and the first-run check
    state = bench.save_state({"validated": {"at": "now", "operator": "x", "checklist": {"a": True}}})
    assert state["validated"]
    state = bench.save_state({"right_ip": "192.0.2.4"})
    assert state["validated"] is None and state["identified"] is False


def test_import_of_a_hand_made_profile(local):
    assert bench.import_profile(bench.DUAL_EXAMPLE) == {}  # the example is not a bench
    text = bench.DUAL_EXAMPLE.read_text()
    (local / "p.yaml").write_text(text)
    found = bench.import_profile(local / "p.yaml")
    assert found["arms"] == 2 and found["left_ip"] == "192.0.2.3" and found["spacing_mm"] == 590
    assert found["response"] == "fast" and found["tracking_stop_rad"] == 0.12 and found["cap_s"] == 120


def test_launcher_commands():
    state = bench.validate(STATE)
    live = bench.run_spec(state, "live")
    assert "--live" in live["command"] and live["token"] == "LIVE-WIDOWXAI-DUAL-192.0.2.3-192.0.2.2"
    assert live["command"][live["command"].index("--confirm-live") + 1] == live["token"]
    assert live["command"][1] == "-u" and live["headset_mode"] == "bimanual"
    dry = bench.run_spec(state, "dry")
    assert "--live" not in dry["command"] and "--confirm-live" not in dry["command"] and dry["token"] is None
    sim = bench.run_spec(state, "sim")
    assert "scripts/run_dual_sim.py" in sim["command"] and "--live" not in sim["command"]
    contact = bench.run_spec(state, "contact")
    assert "--contact-arms" in contact["command"] and "--live" not in contact["command"]
    guarded = bench.run_spec(state, "guarded", "left")
    assert guarded["command"][guarded["command"].index("--moving-side") + 1] == "left"
    assert guarded["token"] == "LIVE-WIDOWXAI-192.0.2.3" and guarded["headset_mode"] == "left"
    assert str(bench.SIDE_PROFILES["left"]) in guarded["command"]
    single = bench.run_spec(bench.validate(STATE | {"arms": 1, "single_ip": "192.0.2.2", "single_hand": "right"}), "live")
    assert "scripts/run_hardware.py" in single["command"] and single["token"] == "LIVE-WIDOWXAI-192.0.2.2"


MODEL = rig.ROOT / "teleop" / "third_party" / "trossen_arm_mujoco" / "trossen_arm_mujoco" / "assets" / "wxai"


@pytest.mark.skipif(not (MODEL / "wxai_follower.xml").exists(), reason="needs the teleop submodule and its arm model")
def test_generated_profile_passes_the_teleop_checks(tmp_path):
    from bimanual_collect import profile_check

    path = tmp_path / "bench.yaml"
    path.write_text(bench.dual_profile(bench.validate(STATE)))
    cwd = os.getcwd()
    try:
        result = profile_check.check(path)
    finally:
        os.chdir(cwd)
    assert result["ok"] and result["gates"] == "satisfied", result
    assert result["token"] == "LIVE-WIDOWXAI-DUAL-192.0.2.3-192.0.2.2"


def test_clearance_and_table_floor_in_the_profile(local):
    state = bench.validate(STATE | {"clearance_mm": 20, "table_floor": True, "table_width_mm": 1450,
                                    "table_depth_mm": 720, "table_rear_inset_mm": 60, "table_floor_offset_mm": 2})
    profile = yaml.safe_load(bench.dual_profile(state))
    assert profile["safety"]["cross_arm_clearance_m"] == 0.020
    assert profile["simulation_environment"]["tabletop"] == {"width_m": 1.45, "depth_m": 0.72, "thickness_m": 0.04,
                                                             "rear_edge_x_m": -0.06, "top_z_m": -0.002}
    # off by default: the example's safety block untouched, no table
    plain = yaml.safe_load(bench.dual_profile(bench.validate(STATE)))
    assert plain["safety"]["cross_arm_clearance_m"] == 0.030 and "simulation_environment" not in plain
    # a saved profile reads back into the same settings
    path = local / "bench.yaml"
    path.write_text(bench.dual_profile(state))
    found = bench.import_profile(path)
    assert {k: found[k] for k in ("clearance_mm", "table_floor", "table_width_mm", "table_depth_mm",
                                  "table_rear_inset_mm", "table_floor_offset_mm")} == \
        {"clearance_mm": 20, "table_floor": True, "table_width_mm": 1450, "table_depth_mm": 720,
         "table_rear_inset_mm": 60, "table_floor_offset_mm": 2}
    for bad in ({"clearance_mm": 5}, {"table_floor_offset_mm": 11}, {"table_width_mm": 100}, {"cap_s": 10},
                {"tracking_stop_rad": 0.2}, {"response": "custom", "custom_move": 1.5}, {"response": "wild"}):
        with pytest.raises(ValueError):
            bench.validate(STATE | bad)


@pytest.mark.skipif(not (MODEL / "wxai_follower.xml").exists(), reason="needs the teleop submodule and its arm model")
def test_profile_with_table_floor_passes_the_teleop_checks(tmp_path):
    from bimanual_collect import profile_check

    path = tmp_path / "bench.yaml"
    path.write_text(bench.dual_profile(bench.validate(STATE | {"clearance_mm": 20, "table_floor": True,
                                                                "table_floor_offset_mm": 2})))
    cwd = os.getcwd()
    try:
        result = profile_check.check(path)
    finally:
        os.chdir(cwd)
    assert result["ok"] and result["gates"] == "satisfied", result
    assert any("configured margin 0.020" in line for line in result["checks"]), result["checks"]


def test_custom_values(local):
    state = bench.validate(STATE | {"response": "custom", "custom_move": 0.55, "custom_turn": 0.45, "cap_s": 450,
                                    "tracking_stop_rad": 0.1, "clearance_mm": 18, "table_floor": True,
                                    "table_floor_offset_mm": 7})
    profile = yaml.safe_load(bench.dual_profile(state))
    left = profile["arms"]["left"]
    assert (left["translation_scale"], left["rotation_scale"]) == (0.55, 0.45)
    assert profile["hardware"]["max_demo_duration_s"] == 450.0 and profile["hardware"]["max_feedback_error_rad"] == 0.10
    assert profile["safety"]["cross_arm_clearance_m"] == 0.018
    assert profile["simulation_environment"]["tabletop"]["top_z_m"] == -0.007
    # the first-run one-arm profiles stay at the validated 0.40
    single = yaml.safe_load(bench.single_profile("left", "192.0.2.3", state, response="validated"))
    assert single["quest"]["translation_scale"] == 0.40
    path = local / "bench.yaml"
    path.write_text(bench.dual_profile(state))
    found = bench.import_profile(path)
    assert (found["response"], found["custom_move"], found["custom_turn"], found["cap_s"]) == ("custom", 0.55, 0.45, 450)
