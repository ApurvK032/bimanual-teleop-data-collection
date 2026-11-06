from __future__ import annotations

"""Collision screening for one moving arm beside one stationary arm."""

import mujoco
import numpy as np

from .config import DUAL_ARM_SIDES
from .dual_arm_model import DualArmCollisionModel
from .hardware import HardwareSafetyError
from .model import WidowXAIModel

MARGINAL_START_CONTACT_M = 0.002


def _self_contact(
    single_model: WidowXAIModel,
    q: np.ndarray,
    gripper_m: float,
) -> tuple[bool, float]:
    """Return non-carriage contact presence and deepest penetration."""

    single_model.set_viewer_qpos(single_model.data, q, gripper_m)
    deepest = 0.0
    found = False
    for index in range(single_model.data.ncon):
        contact = single_model.data.contact[index]
        names = {
            mujoco.mj_id2name(
                single_model.model,
                mujoco.mjtObj.mjOBJ_BODY,
                single_model.model.geom_bodyid[contact.geom1],
            ),
            mujoco.mj_id2name(
                single_model.model,
                mujoco.mjtObj.mjOBJ_BODY,
                single_model.model.geom_bodyid[contact.geom2],
            ),
        }
        if names == {"carriage_right", "carriage_left"}:
            continue
        found = True
        deepest = max(deepest, max(0.0, -float(contact.dist)))
    return found, deepest


def screen_self_collision_path(
    single_model: WidowXAIModel,
    *,
    start_q: np.ndarray,
    end_q: np.ndarray,
    start_gripper_m: float,
    end_gripper_m: float | None = None,
    samples: int,
    label: str,
    allow_marginal_start_m: float = 0.0,
) -> tuple[float, float | None]:
    """Screen one arm's own collision geometry along a straight joint path.

    Any contact fails closed, except that a measured start already in model
    contact is accepted when ``allow_marginal_start_m`` is positive, the start
    penetration is within it, the path never deepens it, clears it, and never
    re-enters contact. This is the 2026-08-12 owner-accepted rule for the right
    arm's known near-rest ``link_2``/``link_4`` model overlap.

    Returns the start penetration depth and the path fraction where a marginal
    start cleared; a clear start returns ``(0.0, None)``.
    """

    if samples < 2:
        raise ValueError("path screen requires at least two samples")
    start = np.asarray(start_q, dtype=float).reshape(6)
    end = np.asarray(end_q, dtype=float).reshape(6)
    start_gripper = max(0.0, float(start_gripper_m))
    end_gripper = start_gripper if end_gripper_m is None else max(0.0, float(end_gripper_m))

    start_contact, start_depth = _self_contact(single_model, start, start_gripper)
    marginal = bool(start_contact and allow_marginal_start_m > 0.0)
    if start_contact and not marginal:
        raise HardwareSafetyError(f"{label} self-collision model rejects the path at 0.0%")
    if marginal and start_depth > float(allow_marginal_start_m):
        raise HardwareSafetyError(
            f"{label} measured start contact is {start_depth * 1000:.3f} mm, "
            f"above the {float(allow_marginal_start_m) * 1000:.3f} mm marginal limit"
        )

    cleared_at: float | None = None
    penetration_tolerance_m = 1e-6
    for alpha in np.linspace(0.0, 1.0, samples):
        q = start + alpha * (end - start)
        gripper = start_gripper + alpha * (end_gripper - start_gripper)
        contact, depth = _self_contact(single_model, q, gripper)
        if not marginal:
            if contact:
                raise HardwareSafetyError(
                    f"{label} self-collision model rejects the path near {alpha * 100:.1f}%"
                )
            continue
        if contact and depth > start_depth + penetration_tolerance_m:
            raise HardwareSafetyError(
                f"{label} path deepens the measured start contact near "
                f"{alpha * 100:.1f}% ({depth * 1000:.3f} mm vs "
                f"{start_depth * 1000:.3f} mm at rest)"
            )
        if not contact and cleared_at is None:
            cleared_at = float(alpha)
        elif contact and cleared_at is not None:
            raise HardwareSafetyError(
                f"{label} path re-enters self-collision near {alpha * 100:.1f}%"
            )

    if marginal and cleared_at is None:
        raise HardwareSafetyError(f"{label} measured start contact never clears along the path")
    return (float(start_depth) if marginal else 0.0), cleared_at


def cross_arm_separation(
    dual_model: DualArmCollisionModel,
    *,
    moving_side: str,
    moving_q: np.ndarray,
    holding_q: np.ndarray,
    moving_gripper_m: float,
    holding_gripper_m: float,
    clearance_m: float,
) -> float:
    """Validate one combined state without rejecting stationary-arm droop."""

    if moving_side not in DUAL_ARM_SIDES:
        raise ValueError(f"unknown moving side: {moving_side}")
    other = next(side for side in DUAL_ARM_SIDES if side != moving_side)
    dual_model.forward(
        {
            moving_side: np.asarray(moving_q, dtype=float).reshape(6),
            other: np.asarray(holding_q, dtype=float).reshape(6),
        },
        {
            moving_side: max(0.0, float(moving_gripper_m)),
            other: max(0.0, float(holding_gripper_m)),
        },
    )
    separation = float(dual_model.minimum_cross_arm_distance())
    if separation < float(clearance_m):
        raise HardwareSafetyError(
            f"{moving_side} command reaches {separation:.6f} m cross-arm separation, "
            f"below the {float(clearance_m):.6f} m margin"
        )
    return separation


def screen_gripper_path(
    dual_model: DualArmCollisionModel,
    *,
    moving_side: str,
    moving_q: np.ndarray,
    holding_q: np.ndarray,
    moving_start_gripper_m: float,
    moving_end_gripper_m: float,
    holding_gripper_m: float,
    clearance_m: float,
    samples: int,
) -> float:
    """Screen a stationary arm pose across a moving gripper stroke."""

    if samples < 2:
        raise ValueError("gripper path screen requires at least two samples")
    minimum = np.inf
    for alpha in np.linspace(0.0, 1.0, samples):
        moving_gripper = float(moving_start_gripper_m) + alpha * (
            float(moving_end_gripper_m) - float(moving_start_gripper_m)
        )
        minimum = min(
            minimum,
            cross_arm_separation(
                dual_model,
                moving_side=moving_side,
                moving_q=moving_q,
                holding_q=holding_q,
                moving_gripper_m=moving_gripper,
                holding_gripper_m=holding_gripper_m,
                clearance_m=clearance_m,
            ),
        )
    return float(minimum)


def screen_selected_path(
    single_model: WidowXAIModel,
    dual_model: DualArmCollisionModel,
    *,
    moving_side: str,
    moving_start: np.ndarray,
    moving_end: np.ndarray,
    holding_q: np.ndarray,
    moving_gripper_m: float,
    holding_gripper_m: float,
    samples: int,
    clearance_m: float,
    moving_end_gripper_m: float | None = None,
    allow_marginal_start_m: float = 0.0,
) -> float:
    """Screen moving-arm self collision and distance to a stationary arm.

    The holding arm's own near-zero model contact is not treated as a moving
    path fault: it is already physically resting there and receives no command.
    Its complete geometry still participates in cross-arm distance checking.
    """

    if moving_side not in DUAL_ARM_SIDES:
        raise ValueError(f"unknown moving side: {moving_side}")
    if samples < 2:
        raise ValueError("path screen requires at least two samples")
    start = np.asarray(moving_start, dtype=float).reshape(6)
    end = np.asarray(moving_end, dtype=float).reshape(6)
    holding = np.asarray(holding_q, dtype=float).reshape(6)
    start_gripper = max(0.0, float(moving_gripper_m))
    end_gripper = (
        start_gripper
        if moving_end_gripper_m is None
        else max(0.0, float(moving_end_gripper_m))
    )

    start_depth, cleared_at = screen_self_collision_path(
        single_model,
        start_q=start,
        end_q=end,
        start_gripper_m=start_gripper,
        end_gripper_m=end_gripper,
        samples=samples,
        label=moving_side,
        allow_marginal_start_m=allow_marginal_start_m,
    )

    minimum = np.inf
    minimum_alpha = 0.0
    for alpha in np.linspace(0.0, 1.0, samples):
        q = start + alpha * (end - start)
        gripper = start_gripper + alpha * (end_gripper - start_gripper)
        separation = cross_arm_separation(
            dual_model,
            moving_side=moving_side,
            moving_q=q,
            holding_q=holding,
            moving_gripper_m=gripper,
            holding_gripper_m=holding_gripper_m,
            clearance_m=clearance_m,
        )
        if separation < minimum:
            minimum = separation
            minimum_alpha = float(alpha)

    if cleared_at is not None:
        print(
            f"  {moving_side}: measured {start_depth * 1000:.3f} mm start contact "
            f"never deepens and clears by {cleared_at * 100:.1f}%"
        )
    if minimum < float(clearance_m):
        # cross_arm_separation normally raises first; retain this path-level
        # message if that implementation changes.
        raise HardwareSafetyError(
            f"{moving_side} path reaches {minimum:.6f} m cross-arm separation near "
            f"{minimum_alpha * 100:.1f}%, below the {float(clearance_m):.6f} m margin"
        )
    return float(minimum)
