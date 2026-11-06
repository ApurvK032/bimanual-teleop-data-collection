from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .arm_runtime import ArmProposal, ArmRuntime
from .config import ArmConfig
from .dual_arm_model import CollisionReport, DualArmCollisionModel
from .types import BimanualQuestSample

DUAL_ARM_SIDES = ("left", "right")
HOME_HOLD_NS = 350_000_000
HOME_GRIP_RELEASED = 0.2
HOME_FEEDBACK_TOLERANCE_RAD = 0.025
HOME_MAX_SPEED_RAD_S = 0.6
HOME_MAX_ACCELERATION_RAD_S2 = 1.2
HOME_MAX_JERK_RAD_S3 = 8.0


@dataclass
class HomeReturn:
    start_q: dict[str, np.ndarray]
    duration_s: float
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class CoordinatedTick:
    """The outcome of one coordinated control tick for both arms."""

    proposals: dict[str, ArmProposal]
    collision: CollisionReport
    accepted: bool
    stream_lost: bool
    fault_reason: str = ""
    held_sides: tuple[str, ...] = field(default_factory=tuple)
    control_state: str = "normal"

    @property
    def moving_sides(self) -> tuple[str, ...]:
        if not self.accepted:
            return ()
        return tuple(
            side for side, proposal in sorted(self.proposals.items()) if proposal.active
        )


class DualArmCoordinator:
    """Owns the one place where two arms become one coordinated action.

    The ordering is deliberate and is the whole reason a dual-arm runtime
    cannot be two single-arm processes:

    1. consume one Quest sequence exactly once;
    2. judge each controller's freshness independently;
    3. solve both arms against that one packet;
    4. screen the *combined* commanded state for cross-arm collision;
    5. only then commit and send.

    A combined state that fails step 4 is rejected for both arms, even when
    only one arm was moving. Neither arm may proceed from a state that was
    never sent.
    """

    def __init__(
        self,
        arms: dict[str, ArmRuntime],
        collision_model: DualArmCollisionModel,
        *,
        coordinated_fault_hold: bool = True,
        cross_arm_collision: bool = True,
        collision_recovery_max_joint_delta_rad: np.ndarray | None = None,
        collision_recovery_max_gripper_delta_m: float = 0.001,
        home_q: np.ndarray | None = None,
    ) -> None:
        missing = [side for side in DUAL_ARM_SIDES if side not in arms]
        if missing:
            raise ValueError(f"dual-arm coordinator is missing arms: {missing}")
        hands = [arms[side].controller_hand for side in DUAL_ARM_SIDES]
        if len(set(hands)) != len(hands):
            raise ValueError("both arm runtimes claim the same controller hand")
        self.arms = dict(arms)
        self.collision_model = collision_model
        self.coordinated_fault_hold = bool(coordinated_fault_hold)
        self.cross_arm_collision = bool(cross_arm_collision)
        recovery_delta = (
            np.full(6, 0.010)
            if collision_recovery_max_joint_delta_rad is None
            else np.asarray(collision_recovery_max_joint_delta_rad, dtype=float).reshape(6)
        )
        if not np.all(np.isfinite(recovery_delta)) or np.any(recovery_delta <= 0.0):
            raise ValueError("collision recovery joint deltas must be finite and positive")
        recovery_gripper_delta = float(collision_recovery_max_gripper_delta_m)
        if not np.isfinite(recovery_gripper_delta) or recovery_gripper_delta <= 0.0:
            raise ValueError("collision recovery gripper delta must be finite and positive")
        self.collision_recovery_max_joint_delta_rad = recovery_delta.copy()
        self.collision_recovery_max_gripper_delta_m = recovery_gripper_delta
        self.collision_recovery_active = False
        # Deliberately absent unless run_dual_sim.py enables the rejected
        # measured-feedback controller through --jerk-stress. Hardware and
        # normal MuJoCo never construct this object from configuration.
        self._experimental_load_guard: object | None = None
        self.experimental_load_decision: object | None = None
        self.rejected_ticks = 0
        self.stream_loss_ticks = 0
        self.last_reconnect_generation: int | None = None
        self.home_q = None
        if home_q is not None and self.cross_arm_collision and all(
            arm.joint_limiter is not None for arm in self.arms.values()
        ):
            self.home_q = np.asarray(home_q, dtype=float).reshape(6).copy()
        self._home_armed = False
        self._home_pressed_ns: int | None = None
        self._home_sequence = -1
        self._home_return: HomeReturn | None = None

    def observe_reconnect(self, sample: BimanualQuestSample | None) -> bool:
        """Force both arms to re-anchor across a relay reconnect.

        A reconnect means the operator's controllers and the arms may no longer
        agree on where "here" is. Re-anchoring both together avoids one arm
        keeping a stale clutch anchor while the other takes a new one.
        """

        if sample is None:
            return False
        changed = (
            self.last_reconnect_generation is not None
            and sample.reconnect_generation != self.last_reconnect_generation
        )
        self.last_reconnect_generation = sample.reconnect_generation
        if changed:
            self._disarm_home()
            self._home_sequence = -1
            for arm in self.arms.values():
                arm.force_reanchor()
        return changed

    def _disarm_home(self) -> None:
        self._home_armed = False
        self._home_pressed_ns = None
        self._home_return = None

    def _home_state(self, sample: BimanualQuestSample | None) -> str:
        """Require a fresh release and deliberate hold in the current connection."""
        fresh = all(arm.last_freshness.fresh for arm in self.arms.values())
        if not fresh:
            self._disarm_home()
        if sample is None:
            return "normal"
        if sample.sequence <= self._home_sequence:
            self._disarm_home()
            return "home_hold" if sample.home_button is True else "normal"
        self._home_sequence = sample.sequence
        controllers = (sample.for_hand(side) for side in DUAL_ARM_SIDES)
        if not fresh or sample.home_button is None or not all(
            controller is not None and 0 <= controller.grip <= HOME_GRIP_RELEASED
            for controller in controllers
        ):
            self._disarm_home()
        elif not sample.home_button:
            self._home_armed = True
            self._home_pressed_ns = None
            self._home_return = None
        elif self._home_armed:
            if self._home_pressed_ns is None:
                self._home_pressed_ns = sample.pc_arrival_monotonic_ns
            if sample.pc_arrival_monotonic_ns - self._home_pressed_ns >= HOME_HOLD_NS:
                return "returning_home"
            return "normal"
        return "home_hold" if sample.home_button is True else "normal"

    def _home_commands(self, dt: float) -> dict[str, np.ndarray]:
        if self._home_return is None:
            start_q = {side: arm.q_command.copy() for side, arm in self.arms.items()}
            duration_s = 1.0
            # Bound the derivatives of a quintic with zero endpoint speed and acceleration.
            for side, arm in self.arms.items():
                distance = np.abs(self.home_q - start_q[side])
                limiter = arm.joint_limiter
                speed = np.minimum(limiter.max_velocity, HOME_MAX_SPEED_RAD_S)
                acceleration = np.minimum(limiter.max_acceleration, HOME_MAX_ACCELERATION_RAD_S2)
                jerk = HOME_MAX_JERK_RAD_S3
                if limiter.max_jerk is not None:
                    jerk = np.minimum(limiter.max_jerk, jerk)
                duration_s = max(
                    duration_s,
                    float(np.max(1.875 * distance / speed)),
                    float(np.max(np.sqrt(6.0 * distance / acceleration))),
                    float(np.max(np.cbrt(60.0 * distance / jerk))),
                )
            self._home_return = HomeReturn(start_q, duration_s)
        motion = self._home_return
        motion.elapsed_s = min(motion.duration_s, motion.elapsed_s + dt)
        progress = motion.elapsed_s / motion.duration_s
        if progress == 1.0:
            return dict.fromkeys(DUAL_ARM_SIDES, self.home_q)
        blend = progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)
        return {side: start + blend * (self.home_q - start) for side, start in motion.start_q.items()}

    def enable_simulation_jerk_stress(self, guard: object) -> None:
        """Attach the historically rejected load-yield controller for MuJoCo.

        This is an explicit runtime call rather than a config-driven builder
        option so no hardware launcher can accidentally resurrect it.
        """

        required = ("observe", "max_yield_delta_rad")
        if any(not hasattr(guard, name) for name in required):
            raise TypeError("simulation jerk-stress guard has the wrong interface")
        self._experimental_load_guard = guard

    def step(
        self,
        sample: BimanualQuestSample | None,
        *,
        limiter_dt: float,
        robot_q_source: dict[str, object] | None = None,
        feedback_error_source: dict[str, object] | None = None,
        feedback_sample_fresh: bool = False,
        external_fault: str = "",
    ) -> CoordinatedTick:
        self.observe_reconnect(sample)
        sources = robot_q_source or {}

        for arm in self.arms.values():
            arm.observe(sample)

        home_enabled = self.home_q is not None and self._experimental_load_guard is None
        control_state = self._home_state(sample) if home_enabled else "normal"
        homing = control_state == "returning_home"
        suppress_clutch = home_enabled and sample is not None and sample.home_button is True
        home_commands = self._home_commands(limiter_dt) if homing else {}

        proposals: dict[str, ArmProposal] = {}
        for side in DUAL_ARM_SIDES:
            arm = self.arms[side]
            proposals[side] = arm.solve(
                sample,
                limiter_dt=limiter_dt,
                robot_q_source=sources.get(side),
                home_command_q=home_commands.get(side),
                suppress_clutch=suppress_clutch,
            )

        # Loss of the shared WebXR/relay connection is distinguishable from one
        # controller leaving the volume: neither hand is fresh at the same time.
        stream_lost = not any(proposal.freshness.fresh for proposal in proposals.values())
        if stream_lost:
            self.stream_loss_ticks += 1

        held_sides = tuple(
            side
            for side in DUAL_ARM_SIDES
            if not proposals[side].tracked or not proposals[side].freshness.fresh
        )

        if external_fault:
            self._disarm_home()
            return self._reject(
                proposals,
                CollisionReport(colliding=False),
                stream_lost,
                external_fault,
                held_sides,
            )

        if self._experimental_load_guard is not None:
            if feedback_error_source is None:
                raise ValueError(
                    "simulation jerk stress requires delayed feedback errors"
                )
            grip_by_side = {
                side: (
                    0.0
                    if proposals[side].sample is None
                    else float(proposals[side].sample.grip)
                )
                for side in DUAL_ARM_SIDES
            }
            decision = self._experimental_load_guard.observe(
                {
                    side: np.asarray(feedback_error_source[side], dtype=float)
                    for side in DUAL_ARM_SIDES
                },
                feedback_sample_fresh=bool(feedback_sample_fresh),
                grip_by_side=grip_by_side,
            )
            self.experimental_load_decision = decision
            if decision.triggered:
                for arm in self.arms.values():
                    arm.force_reanchor()
            if decision.active:
                return self._commit_experimental_load_yield(
                    proposals,
                    sources,
                    stream_lost=stream_lost,
                    phase=decision.phase,
                )

        report = CollisionReport(colliding=False)
        if self.cross_arm_collision:
            report = self.collision_model.check(
                {side: proposals[side].q_command for side in DUAL_ARM_SIDES},
                {side: proposals[side].gripper_command_m for side in DUAL_ARM_SIDES},
            )
        if report.colliding:
            self.collision_recovery_active = True
            self._disarm_home()
            return self._reject(
                proposals,
                report,
                stream_lost,
                f"combined scene rejects the commanded state: {report.describe()}",
                held_sides,
                control_state="home_blocked" if homing else "collision_hold",
            )

        bounded_override = False
        recovery_complete = True
        if self.collision_recovery_active:
            if not homing:
                control_state = "collision_recovery"
            bounded: dict[str, ArmProposal] = {}
            for side in DUAL_ARM_SIDES:
                arm = self.arms[side]
                proposal = proposals[side]
                q_delta = np.clip(
                    proposal.q_command - arm.q_command,
                    -self.collision_recovery_max_joint_delta_rad,
                    self.collision_recovery_max_joint_delta_rad,
                )
                gripper_delta = float(
                    np.clip(
                        proposal.gripper_command_m - arm.gripper_command_m,
                        -self.collision_recovery_max_gripper_delta_m,
                        self.collision_recovery_max_gripper_delta_m,
                    )
                )
                q_command = arm.q_command + q_delta
                gripper_command = arm.gripper_command_m + gripper_delta
                side_complete = np.allclose(
                    q_command, proposal.q_command, rtol=0.0, atol=1e-12
                ) and np.isclose(
                    gripper_command,
                    proposal.gripper_command_m,
                    rtol=0.0,
                    atol=1e-12,
                )
                recovery_complete = recovery_complete and bool(side_complete)
                flags = list(proposal.limiter_flags)
                if not side_complete:
                    flags.append("collision_recovery")
                bounded[side] = replace(
                    proposal,
                    q_command=q_command,
                    gripper_command_m=gripper_command,
                    feedforward_velocity=np.zeros(6),
                    limiter_flags=flags,
                )
            proposals = bounded
            bounded_override = True
            report = self.collision_model.check(
                {side: proposals[side].q_command for side in DUAL_ARM_SIDES},
                {side: proposals[side].gripper_command_m for side in DUAL_ARM_SIDES},
            )
            if report.colliding:
                self._disarm_home()
                return self._reject(
                    proposals,
                    report,
                    stream_lost,
                    f"combined scene rejects collision recovery: {report.describe()}",
                    held_sides,
                    control_state="home_blocked" if homing else "collision_hold",
                )

        for side in DUAL_ARM_SIDES:
            if bounded_override or homing:
                self.arms[side].commit_bounded_override(proposals[side])
            else:
                self.arms[side].commit(proposals[side])
        if bounded_override and recovery_complete:
            self.collision_recovery_active = False
        if homing and bounded_override and not recovery_complete:
            self._disarm_home()
            control_state = "home_hold"
        elif homing and self._home_return.elapsed_s == self._home_return.duration_s and all(
            np.max(np.abs(arm.q_feedback - self.home_q)) <= HOME_FEEDBACK_TOLERANCE_RAD
            for arm in self.arms.values()
        ):
            control_state = "at_home"
        return CoordinatedTick(
            proposals=proposals,
            collision=report,
            accepted=True,
            stream_lost=stream_lost,
            held_sides=held_sides,
            control_state=control_state,
        )

    def _commit_experimental_load_yield(
        self,
        proposals: dict[str, ArmProposal],
        feedback_by_side: dict[str, object],
        *,
        stream_lost: bool,
        phase: str,
    ) -> CoordinatedTick:
        """Continuously chase the newest delayed measurement, on purpose.

        This reproduces the 2026-08-14 physical failure. Do not reuse this
        method as a recovery design: its moving destination is precisely what
        created the repeated command-direction reversals.
        """

        bounded: dict[str, ArmProposal] = {}
        max_delta = float(self._experimental_load_guard.max_yield_delta_rad)
        for side in DUAL_ARM_SIDES:
            arm = self.arms[side]
            measured = np.asarray(
                feedback_by_side.get(side, arm.q_feedback), dtype=float
            ).reshape(6)
            q_command = arm.q_command + np.clip(
                measured - arm.q_command,
                -max_delta,
                max_delta,
            )
            flags = list(proposals[side].limiter_flags)
            flags.append("experimental_dynamic_load_yield")
            bounded[side] = replace(
                proposals[side],
                q_des=measured.copy(),
                q_command=q_command,
                gripper_des_m=arm.gripper_command_m,
                gripper_command_m=arm.gripper_command_m,
                feedforward_velocity=np.zeros(6),
                active=False,
                limiter_flags=flags,
            )

        report = self.collision_model.check(
            {side: bounded[side].q_command for side in DUAL_ARM_SIDES},
            {side: bounded[side].gripper_command_m for side in DUAL_ARM_SIDES},
        )
        if report.colliding:
            return self._reject(
                bounded,
                report,
                stream_lost,
                f"combined scene rejects experimental load yield: {report.describe()}",
                tuple(DUAL_ARM_SIDES),
                control_state="load_yield_collision_hold",
            )

        for side in DUAL_ARM_SIDES:
            self.arms[side].commit_bounded_override(bounded[side])
        return CoordinatedTick(
            proposals=bounded,
            collision=report,
            accepted=True,
            stream_lost=stream_lost,
            held_sides=tuple(DUAL_ARM_SIDES),
            control_state="load_yield",
            fault_reason=f"simulation-only rejected controller: {phase}",
        )

    def _reject(
        self,
        proposals: dict[str, ArmProposal],
        report: CollisionReport,
        stream_lost: bool,
        reason: str,
        held_sides: tuple[str, ...],
        *,
        control_state: str = "fault_hold",
    ) -> CoordinatedTick:
        self.rejected_ticks += 1
        for arm in self.arms.values():
            arm.reject()
        return CoordinatedTick(
            proposals=proposals,
            collision=report,
            accepted=False,
            stream_lost=stream_lost,
            fault_reason=reason,
            held_sides=tuple(DUAL_ARM_SIDES) if self.coordinated_fault_hold else held_sides,
            control_state=control_state,
        )

    def hold_all(self) -> None:
        for arm in self.arms.values():
            arm.hold()


def build_dual_arm_system(
    config: dict[str, Any],
    arms_config: dict[str, ArmConfig],
    *,
    initial_q: np.ndarray,
    initial_gripper_m: float,
    control_gripper: bool = True,
) -> tuple[dict[str, ArmRuntime], DualArmCollisionModel, DualArmCoordinator]:
    """Build both arm runtimes, the combined scene, and the coordinator.

    Shared by the simulation, preflight, and hardware launchers so all three
    exercise the same construction path rather than three near-copies.
    """

    safety = config.get("safety", {})
    simulation_environment = config.get("simulation_environment", {}) or {}
    if not isinstance(simulation_environment, dict):
        raise ValueError("simulation_environment must be a mapping")
    collision_model = DualArmCollisionModel(
        {side: arms_config[side].placement for side in DUAL_ARM_SIDES},
        xml_path=config["model"]["xml_path"],
        clearance_m=float(safety.get("cross_arm_clearance_m", 0.0)),
        tabletop=simulation_environment.get("tabletop"),
    )
    arms = {
        side: ArmRuntime(
            arms_config[side],
            config,
            initial_q=initial_q,
            initial_gripper_m=initial_gripper_m,
            control_gripper=control_gripper,
        )
        for side in DUAL_ARM_SIDES
    }
    coordinator = DualArmCoordinator(
        arms,
        collision_model,
        coordinated_fault_hold=bool(safety.get("coordinated_fault_hold", True)),
        cross_arm_collision=bool(safety.get("cross_arm_collision", True)),
        collision_recovery_max_joint_delta_rad=np.asarray(
            safety.get("collision_recovery_max_joint_delta_rad", [0.010] * 6),
            dtype=float,
        ),
        collision_recovery_max_gripper_delta_m=float(
            safety.get("collision_recovery_max_gripper_delta_m", 0.001)
        ),
        home_q=arms["left"].model.clamp_joints(np.asarray(config["model"]["simulation_start_q_rad"])),
    )
    return arms, collision_model, coordinator
