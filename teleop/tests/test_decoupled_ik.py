import numpy as np

from widowxai_quest_teleop.config import load_config
from widowxai_quest_teleop.decoupled_ik import DecoupledIK
from widowxai_quest_teleop.model import WidowXAIModel


def test_reachable_pose_converges_without_limit_violation(model: WidowXAIModel) -> None:
    config = load_config()
    solver = DecoupledIK.from_config(model, config)
    q_start = np.array([0.0, 1.0, 0.55, 0.2, -0.15, 0.1])
    q_goal = np.array([0.12, 1.12, 0.67, 0.35, -0.28, 0.28])
    target, _ = model.fk(q_goal)
    q = q_start.copy()
    for _ in range(160):
        q, diagnostics = solver.solve(target, q)
    assert diagnostics.status == "ok"
    assert diagnostics.position_residual_m < 0.008
    assert diagnostics.orientation_residual_rad < 0.04
    assert np.all(q >= model.joint_limits[:, 0])
    assert np.all(q <= model.joint_limits[:, 1])
    assert np.all(np.isfinite(q))


def test_wrist_only_target_does_not_move_arm_group(model: WidowXAIModel) -> None:
    config = load_config()
    solver = DecoupledIK.from_config(model, config)
    q_start = np.array([0.1, 1.05, 0.62, 0.0, 0.0, 0.0])
    q_goal = q_start.copy()
    q_goal[3:] = [0.25, -0.30, 0.45]
    target, _ = model.fk(q_goal)
    q = q_start.copy()
    for _ in range(120):
        q, diagnostics = solver.solve(target, q)
    assert diagnostics.orientation_residual_rad < 0.04
    assert np.linalg.norm(q[:3] - q_start[:3]) < 0.02


def test_unreachable_target_stays_finite_and_bounded(model: WidowXAIModel) -> None:
    config = load_config()
    solver = DecoupledIK.from_config(model, config)
    q = np.array(config["model"]["simulation_start_q_rad"], dtype=float)
    pose, _ = model.fk(q)
    unreachable = type(pose)(pose.position + np.array([2.0, -2.0, 2.0]), pose.quaternion_wxyz)
    for _ in range(50):
        q, diagnostics = solver.solve(unreachable, q)
    assert np.all(np.isfinite(q))
    assert np.all(q >= model.joint_limits[:, 0])
    assert np.all(q <= model.joint_limits[:, 1])
    assert diagnostics.minimum_joint_limit_margin_rad >= -1e-12
    assert diagnostics.position_residual_m > 0.1


def test_reference_per_tick_joint_delta_caps_are_applied_once(model: WidowXAIModel) -> None:
    config = load_config()
    solver = DecoupledIK.from_config(model, config)
    q = np.array(config["model"]["simulation_start_q_rad"], dtype=float)
    pose, _ = model.fk(q)
    target = type(pose)(pose.position + np.array([1.0, 1.0, 1.0]), pose.quaternion_wxyz)
    result, diagnostics = solver.solve(target, q)
    caps = np.asarray(config["ik"]["max_dq_per_joint_rad"])
    assert np.all(np.abs(result - q) <= caps + 1e-12)
    assert diagnostics.iterations == 1


def test_deployment_joint_margin_is_absorbing_not_a_gate_failure(model: WidowXAIModel) -> None:
    config = load_config()
    solver = DecoupledIK.from_config(model, config)
    deployment_limits = model.joint_limits.copy()
    deployment_limits[:, 0] += 0.05
    deployment_limits[:, 1] -= 0.05
    solver.set_joint_limits(deployment_limits)

    q = np.array(config["model"]["simulation_start_q_rad"], dtype=float)
    pose, _ = model.fk(q)
    target = type(pose)(pose.position + np.array([3.0, 3.0, 3.0]), pose.quaternion_wxyz)
    for _ in range(500):
        q, diagnostics = solver.solve(target, q)
    assert np.all(q >= deployment_limits[:, 0] - 1e-12)
    assert np.all(q <= deployment_limits[:, 1] + 1e-12)
    assert diagnostics.minimum_joint_limit_margin_rad >= -1e-12
