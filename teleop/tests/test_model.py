import numpy as np
import mujoco

from widowxai_quest_teleop.model import WidowXAIModel


def test_wrist_anchor_is_invariant_to_wrist_joints(model: WidowXAIModel) -> None:
    q = np.array([0.2, 1.0, 0.7, 0.0, 0.0, 0.0])
    _, anchor_reference = model.fk(q)
    for wrist_q in (
        (0.5, 0.4, -1.0),
        (-0.8, -0.7, 2.0),
        (1.2, 1.0, -2.4),
    ):
        candidate = q.copy()
        candidate[3:] = wrist_q
        _, anchor = model.fk(candidate)
        np.testing.assert_allclose(anchor.position, anchor_reference.position, atol=1e-11)


def test_site_position_jacobians_match_finite_difference(model: WidowXAIModel) -> None:
    q = np.array([0.25, 1.1, 0.65, 0.3, -0.25, 0.4])
    epsilon = 1e-6
    for site in ("ee", "wrist"):
        model.forward(q)
        analytic, _ = model.jacobian(site)
        numeric = np.zeros((3, 6))
        for joint in range(6):
            plus = q.copy()
            minus = q.copy()
            plus[joint] += epsilon
            minus[joint] -= epsilon
            model.forward(plus)
            p_plus = model.pose(site).position
            model.forward(minus)
            p_minus = model.pose(site).position
            numeric[:, joint] = (p_plus - p_minus) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, numeric, atol=2e-6, rtol=2e-5)


def test_joint_limits_are_from_official_model(model: WidowXAIModel) -> None:
    expected = np.array(
        [
            [-3.05433, 3.05433],
            [0.0, 3.14159],
            [0.0, 2.35619],
            [-1.5708, 1.5708],
            [-1.5708, 1.5708],
            [-3.14159, 3.14159],
        ]
    )
    np.testing.assert_allclose(model.joint_limits, expected, atol=1e-8)


def test_reference_style_viewer_has_lighting_ground_and_no_target_object() -> None:
    scene = WidowXAIModel()
    assert scene.model.nlight > 0
    assert scene.model.geom("floor").id >= 0
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "target_cube") == -1


def test_viewer_qpos_is_applied_directly(model: WidowXAIModel) -> None:
    data = mujoco.MjData(model.model)
    q = np.array([0.2, 1.0, 0.7, 0.3, -0.2, 0.4])
    model.set_viewer_qpos(data, q, 0.031)
    np.testing.assert_array_equal(data.qpos[model.qpos_indices], q)
    np.testing.assert_array_equal(data.qpos[model.gripper_qpos_indices], [0.031, 0.031])


def test_startup_path_collision_check_uses_official_geometry(model: WidowXAIModel) -> None:
    staged = np.array([0.0, np.pi / 3, np.pi / 6, np.pi / 5, 0.0, 0.0])
    assert model.first_self_collision_on_path(np.zeros(6), staged) is None
    assert not model.in_self_collision(staged, gripper_q=0.0)

    known_colliding = np.array(
        [0.74635137, 3.10690731, 0.50730820, -1.06747787, 0.35355442, -2.86549445]
    )
    assert model.in_self_collision(known_colliding)
    assert model.first_self_collision_on_path(known_colliding, staged) == 0.0
