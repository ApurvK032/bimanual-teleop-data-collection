from __future__ import annotations

"""Maintenance utility: park one or both WidowXAI arms at a measured pose.

This exists because a clean shutdown parks at exactly zero, and joint 2
settling a few tenths of a degree negative is enough for the pinned MuJoCo
model to reject the NEXT startup ramp (it trips at -0.0040 rad). Parking with
a deliberate positive bias takes that coin flip out of the loop.

It is not a teleoperation path. It uses the project's vetted backend
primitives -- never raw driver calls -- and keeps the joint-limit, collision,
and settling checks. The one thing it can relax, and only behind an explicit
flag, is a marginal contact at the MEASURED starting pose: the arm is already
physically sitting there, so the model does not get to veto where it is.
"""

import argparse
import sys

import numpy as np

from widowxai_quest_teleop.config import (
    DUAL_ARM_SIDES,
    DualArmConfigError,
    load_config,
    parse_dual_arm_config,
)
from widowxai_quest_teleop.hardware import (
    END_EFFECTOR_PROFILE_TO_VARIANT,
    HardwareSafetyError,
    HardwareUnavailableError,
    TrossenArmBackend,
)
from widowxai_quest_teleop.model import WidowXAIModel

# A contact this shallow at a pose the arm is physically resting in is model
# conservatism, not a fold. For reference the closed gripper's own carriage
# pair reports ~0.18 mm and is already whitelisted in model.py.
MARGINAL_START_PENETRATION_M = 0.002


def describe_contacts(model: WidowXAIModel, q_arm, gripper_m: float):
    """Return (deepest_penetration_m, [(body1, body2, depth_m), ...])."""

    import mujoco

    model.data.qpos[:] = 0.0
    model.data.qpos[model.qpos_indices] = np.asarray(q_arm, dtype=float).reshape(6)
    model.data.qpos[model.gripper_qpos_indices] = float(gripper_m)
    mujoco.mj_forward(model.model, model.data)

    found = []
    deepest = 0.0
    for index in range(model.data.ncon):
        contact = model.data.contact[index]
        names = {
            mujoco.mj_id2name(
                model.model, mujoco.mjtObj.mjOBJ_BODY, model.model.geom_bodyid[contact.geom1]
            ),
            mujoco.mj_id2name(
                model.model, mujoco.mjtObj.mjOBJ_BODY, model.model.geom_bodyid[contact.geom2]
            ),
        }
        if names == {"carriage_right", "carriage_left"}:
            continue
        depth = float(-contact.dist)
        found.append((*sorted(names), depth))
        deepest = max(deepest, depth)
    return deepest, found


def report_arm(side: str, ip: str, state, model: WidowXAIModel) -> float:
    tolerances = np.asarray(state.position_tolerances, dtype=float).reshape(7)
    deepest, contacts = describe_contacts(model, state.q_arm, state.gripper_position_m)
    print(f"  {side:>5} arm @ {ip}")
    print(f"        q_rad              = {np.round(state.q_arm, 6).tolist()}")
    print(f"        q_deg              = {np.round(np.degrees(state.q_arm), 3).tolist()}")
    print(f"        gripper_m          = {state.gripper_position_m:.6f}")
    print(f"        position_tolerances= {np.round(tolerances, 6).tolist()}")
    print(f"        joint2             = {state.q_arm[2]:+.6f} rad "
          f"({np.degrees(state.q_arm[2]):+.3f} deg), model trips below -0.0040")
    if contacts:
        for body1, body2, depth in contacts:
            print(f"        MODEL CONTACT      : {body1} vs {body2}  {depth * 1000:.3f} mm")
        print(f"        deepest            = {deepest * 1000:.3f} mm")
    else:
        print("        model contacts     : none")
    return deepest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Park WidowXAI arms at a measured pose and report joint values"
    )
    parser.add_argument("--config", default="configs/dual_widowxai.yaml")
    parser.add_argument(
        "--sides",
        default="left,right",
        help="comma-separated subset of left,right (default both)",
    )
    parser.add_argument(
        "--joint2-bias-rad",
        type=float,
        default=0.05,
        help=(
            "positive joint-2 offset for the park pose (default 0.05 rad = 2.9 deg, "
            "about 12x the -0.0040 trip threshold); pass 0 to park at exact zero"
        ),
    )
    parser.add_argument("--duration", type=float, default=3.0, help="blocking move seconds")
    parser.add_argument(
        "--end-effector-profile",
        choices=tuple(END_EFFECTOR_PROFILE_TO_VARIANT),
        default="legacy_1_8",
    )
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help="read-only: connect, report, and exit without enabling position mode",
    )
    parser.add_argument(
        "--allow-marginal-start",
        action="store_true",
        help=(
            "permit a start-pose model contact shallower than "
            f"{MARGINAL_START_PENETRATION_M * 1000:.0f} mm; the arm is physically "
            "resting there, so the model does not get to veto its own start"
        ),
    )
    parser.add_argument("--confirm", default="", help="must equal PARK-WIDOWXAI-<ip>")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        arms_config = parse_dual_arm_config(config)
    except DualArmConfigError as exc:
        raise SystemExit(f"invalid dual-arm configuration: {exc}") from None

    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    unknown = [s for s in sides if s not in DUAL_ARM_SIDES]
    if unknown:
        raise SystemExit(f"unknown sides: {unknown}")

    rest_q = np.asarray(config["hardware"]["rest_q_rad"], dtype=float).reshape(6)
    park_q = rest_q.copy()
    park_q[2] += float(args.joint2_bias_rad)
    model = WidowXAIModel(config["model"]["xml_path"])
    park_q = model.clamp_joints(park_q)

    print("=== park target ===")
    print(f"  q_rad = {np.round(park_q, 6).tolist()}")
    print(f"  q_deg = {np.round(np.degrees(park_q), 3).tolist()}")
    deepest, contacts = describe_contacts(model, park_q, 0.0)
    if contacts:
        raise SystemExit(
            f"REFUSED: the park target itself is in model collision ({deepest * 1000:.3f} mm); "
            "choose a different --joint2-bias-rad"
        )
    print("  park target is model-clear")
    print()

    variant = END_EFFECTOR_PROFILE_TO_VARIANT[str(args.end_effector_profile)]
    backends: dict[str, TrossenArmBackend] = {}
    try:
        print("=== measured state (read-only, no position mode) ===")
        states = {}
        starts = {}
        for side in sides:
            ip = arms_config[side].robot_ip
            backend = TrossenArmBackend(
                ip,
                end_effector_variant=variant,
                required_driver_version=str(config["hardware"]["driver_version_tested"]),
            )
            backends[side] = backend
            states[side] = backend.connect()
            starts[side] = report_arm(side, ip, states[side], model)

        if args.measure_only:
            print()
            print("measure-only: position mode was not enabled; no command was sent")
            return

        for side in sides:
            ip = arms_config[side].robot_ip
            expected = f"PARK-WIDOWXAI-{ip}"
            if args.confirm != expected:
                raise SystemExit(
                    f"motion disabled; to park the {side} arm pass --confirm {expected} "
                    "(park one arm at a time)"
                )

        print()
        print("=== screening measured -> park ===")
        for side in sides:
            state = states[side]
            start_depth = starts[side]
            if start_depth > 0.0:
                if not args.allow_marginal_start:
                    raise HardwareSafetyError(
                        f"{side} arm start pose reports a {start_depth * 1000:.3f} mm model "
                        "contact; it is physically resting there, so re-run with "
                        "--allow-marginal-start to proceed"
                    )
                if start_depth > MARGINAL_START_PENETRATION_M:
                    raise HardwareSafetyError(
                        f"{side} arm start pose contact is {start_depth * 1000:.3f} mm, deeper "
                        f"than the {MARGINAL_START_PENETRATION_M * 1000:.0f} mm marginal limit; "
                        "this may be a real fold -- inspect the arm before moving it"
                    )
                print(
                    f"  {side}: start contact {start_depth * 1000:.3f} mm accepted as marginal"
                )
            # A marginal start contact does not clear on the very next sample --
            # it persists until the joint climbs back past the trip threshold.
            # So the invariant is not "no contact", it is "the move never makes
            # the contact deeper, and it does clear". That is checkable and it
            # proves the arm is travelling OUT of the marginal region.
            worst = 0.0
            worst_alpha = 0.0
            cleared_at = None
            tolerance_m = 1e-6
            for alpha in np.linspace(0.0, 1.0, 201):
                q = state.q_arm + alpha * (park_q - state.q_arm)
                depth, _ = describe_contacts(model, q, state.gripper_position_m)
                if depth > worst:
                    worst, worst_alpha = depth, alpha
                if depth > start_depth + tolerance_m:
                    raise HardwareSafetyError(
                        f"{side} arm path DEEPENS the model contact at {alpha * 100:.1f}% "
                        f"({depth * 1000:.3f} mm vs {start_depth * 1000:.3f} mm at the start); "
                        "this move drives further in, not out -- refusing"
                    )
                if depth == 0.0 and cleared_at is None:
                    cleared_at = alpha
            if cleared_at is None:
                raise HardwareSafetyError(
                    f"{side} arm path never leaves model contact (worst "
                    f"{worst * 1000:.3f} mm at {worst_alpha * 100:.1f}%); refusing"
                )
            if start_depth > 0.0:
                print(
                    f"  {side}: contact never deepens (peak {worst * 1000:.3f} mm <= start "
                    f"{start_depth * 1000:.3f} mm) and clears at {cleared_at * 100:.1f}%"
                )
            else:
                print(f"  {side}: measured -> park path is clear")

        # Cross-arm screening. park_arms originally used only the single-arm
        # model, which cannot see the other arm at all -- it would happily
        # drive one arm through the other. On a two-arm bench every move must
        # be checked against where the OTHER arm is actually standing.
        if len(sides) < len(DUAL_ARM_SIDES):
            others = [s for s in DUAL_ARM_SIDES if s not in sides]
            print()
            print("=== cross-arm screening ===")
            other_states = {}
            for other in others:
                ip = arms_config[other].robot_ip
                backend = TrossenArmBackend(
                    ip,
                    end_effector_variant=variant,
                    required_driver_version=str(config["hardware"]["driver_version_tested"]),
                )
                backends[other] = backend
                other_states[other] = backend.connect()
                print(f"  {other} arm holds at {np.round(other_states[other].q_arm, 4).tolist()}")

            from widowxai_quest_teleop.dual_arm_model import DualArmCollisionModel

            dual = DualArmCollisionModel(
                {s: arms_config[s].placement for s in DUAL_ARM_SIDES},
                xml_path=config["model"]["xml_path"],
                clearance_m=float(config["safety"]["cross_arm_clearance_m"]),
            )
            for side in sides:
                other = others[0]
                grippers = {
                    side: max(0.0, states[side].gripper_position_m),
                    other: max(0.0, other_states[other].gripper_position_m),
                }
                separations = []
                for alpha in np.linspace(0.0, 1.0, 201):
                    q = {
                        side: states[side].q_arm + alpha * (park_q - states[side].q_arm),
                        other: other_states[other].q_arm,
                    }
                    dual.forward(q, grippers)
                    separations.append(dual.minimum_cross_arm_distance())
                separations = np.array(separations)
                start_sep = float(separations[0])
                worst = float(separations.min())
                print(
                    f"  {side}: cross-arm separation {start_sep * 1000:.1f} mm -> "
                    f"{separations[-1] * 1000:.1f} mm, minimum {worst * 1000:.1f} mm"
                )
                if worst < start_sep - 1e-9:
                    raise HardwareSafetyError(
                        f"moving the {side} arm brings it CLOSER to the {other} arm "
                        f"({worst * 1000:.1f} mm vs {start_sep * 1000:.1f} mm at the start). "
                        "Separate the arms by hand with the controllers powered down "
                        "instead of commanding this move."
                    )
                if worst <= 0.0:
                    raise HardwareSafetyError(
                        f"the {side} arm path reaches {worst * 1000:.1f} mm of the {other} "
                        "arm; refusing to command it"
                    )
                print(f"  {side}: never approaches the {other} arm")

        print()
        print("=== parking ===")
        for side in sides:
            backend = backends[side]
            backend.enable_position_control(include_gripper=False)
            print(f"  {side}: {args.duration:g} s blocking move to park")
            backend.move_to_rest(
                park_q,
                states[side].gripper_position_m,
                duration_s=float(args.duration),
                include_gripper=False,
            )

        print()
        print("=== settled state ===")
        for side in sides:
            settled = backends[side].read_state()
            report_arm(side, arms_config[side].robot_ip, settled, model)
            error = float(np.max(np.abs(settled.q_arm - park_q)))
            limit = float(config["hardware"]["max_feedback_error_rad"])
            status = "OK" if error <= limit else "EXCEEDS LIMIT"
            print(f"        settle error       = {error:.6f} rad ({status}, limit {limit})")
    except (HardwareSafetyError, HardwareUnavailableError) as exc:
        print(f"SAFETY STOP: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
    finally:
        for side, backend in backends.items():
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 - report, never mask cleanup
                print(f"WARNING: {side} driver cleanup failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
