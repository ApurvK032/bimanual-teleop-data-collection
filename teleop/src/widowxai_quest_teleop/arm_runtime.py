from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .clutch import ClutchController
from .config import ArmConfig
from .decoupled_ik import DecoupledIK, IKDiagnostics
from .gripper import trigger_to_gripper_position
from .mapping import ClutchPoseMapper
from .model import WidowXAIModel
from .motion_limiter import VelocityFeedforwardFilter, limiter_from_config
from .pose_filter import ControllerPoseFilter
from .safety import FreshnessStatus, FreshSequenceWatchdog
from .types import BimanualQuestSample, Pose, QuestSample


@dataclass
class ArmProposal:
    """One arm's candidate command for a single control tick.

    A proposal is not authoritative. The coordinator screens both arms'
    proposals against the combined scene first, and only a committed proposal
    becomes the arm's new command.
    """

    side: str
    q_command: np.ndarray
    gripper_command_m: float
    q_des: np.ndarray
    gripper_des_m: float
    feedforward_velocity: np.ndarray
    target_pose: Pose | None
    diagnostics: IKDiagnostics | None
    freshness: FreshnessStatus
    engaged: bool
    active: bool
    tracked: bool
    sample: QuestSample | None
    ik_start_monotonic_ns: int = 0
    ik_end_monotonic_ns: int = 0
    limiter_flags: list[str] = field(default_factory=list)


class ArmRuntime:
    """One arm's complete controller-to-joint pipeline with no shared state.

    Every object held here is per-instance by construction: mapper, clutch,
    pose filter, IK solver, limiters, feedforward, and freshness watchdog. The
    arm also owns its own ``WidowXAIModel`` because ``DecoupledIK`` uses the
    model's ``MjData`` as scratch between ``fk`` and ``jacobian``; two solvers
    sharing one model would silently corrupt each other's Jacobians.
    """

    def __init__(
        self,
        arm_config: ArmConfig,
        config: dict[str, Any],
        *,
        initial_q: np.ndarray,
        initial_gripper_m: float,
        control_gripper: bool = True,
    ) -> None:
        self.side = arm_config.side
        self.arm_config = arm_config
        self.controller_hand = arm_config.controller_hand
        self.mapping_mode = arm_config.mapping_mode
        self.control_gripper = bool(control_gripper)

        control = config["control"]
        quest = config["quest"]
        hardware = config.get("hardware", {}) or {}

        self.model = WidowXAIModel(config["model"]["xml_path"])
        self.solver = DecoupledIK.from_config(self.model, config)
        self.mapper = ClutchPoseMapper(
            arm_config.calibration,
            translation_scale=arm_config.settings["translation_scale"],
            rotation_scale=arm_config.settings["rotation_scale"],
            position_reach_limit_m=arm_config.settings["position_reach_limit_m"],
            rotation_reach_limit_rad=arm_config.settings["rotation_reach_limit_rad"],
        )
        self.clutch = ClutchController(self.mapper)
        self.controller_filter = ControllerPoseFilter(control)
        self.watchdog = FreshSequenceWatchdog(
            quest["stale_timeout_s"],
            quest["fresh_samples_to_recover"],
        )

        self.q_command = np.asarray(initial_q, dtype=float).reshape(6).copy()
        self.q_des = self.q_command.copy()
        self.gripper_command_m = float(initial_gripper_m)
        self.gripper_des_m = self.gripper_command_m
        self.gripper_open_m = float(hardware.get("gripper_open_m", 0.044))
        self.gripper_min_m = float(hardware.get("gripper_min_demo_m", 0.0))

        self.joint_limiter = limiter_from_config(
            control.get("joint_command_limits"),
            self.q_command,
        )
        self.gripper_limiter = (
            limiter_from_config(
                control.get("gripper_command_limits"),
                np.array([self.gripper_command_m]),
            )
            if self.control_gripper
            else None
        )
        feedforward_config = hardware.get("arm_velocity_feedforward", {}) or {}
        self.feedforward_filter = (
            VelocityFeedforwardFilter(
                self.q_command,
                filter_alpha=feedforward_config["filter_alpha"],
                gain=feedforward_config["gain"],
                max_velocity=np.asarray(
                    feedforward_config["max_velocity_rad_s"], dtype=float
                ),
            )
            if feedforward_config.get("enabled", False)
            else None
        )
        self.feedforward_velocity = np.zeros(6)

        self.q_feedback = self.q_command.copy()
        self.gripper_feedback_m = self.gripper_command_m
        self._filtered_controller: Pose | None = None
        self._last_sample: QuestSample | None = None
        self._target_pose: Pose | None = None
        self.last_freshness = self.watchdog.poll()
        self.holds = 0
        self.rejected_commands = 0

    # -- input ------------------------------------------------------------

    def observe(
        self,
        sample: BimanualQuestSample | None,
        *,
        now_ns: int | None = None,
    ) -> FreshnessStatus:
        """Judge this arm's controller freshness for one packet.

        An untracked hand is polled rather than observed. The packet's shared
        sequence keeps advancing while a controller is out of view, so counting
        it as a unique sample would report a stale controller as fresh and let
        one arm run on a frozen pose.
        """

        projected = None if sample is None else sample.for_hand(self.controller_hand)
        if projected is None:
            self.last_freshness = self.watchdog.poll(now_ns)
        else:
            self._last_sample = projected
            self.last_freshness = self.watchdog.observe(projected, now_ns)
        return self.last_freshness

    # -- control ----------------------------------------------------------

    def solve(
        self,
        sample: BimanualQuestSample | None,
        *,
        limiter_dt: float,
        robot_q_source: np.ndarray | None = None,
        home_command_q: np.ndarray | None = None,
        suppress_clutch: bool = False,
    ) -> ArmProposal:
        """Produce this arm's candidate command without committing it."""

        projected = None if sample is None else sample.for_hand(self.controller_hand)
        tracked = projected is not None
        freshness = self.last_freshness
        source_q = self.q_feedback if robot_q_source is None else robot_q_source
        robot_pose, wrist_pose = self.model.fk(source_q)

        pose_sample = projected or self._last_sample
        if pose_sample is not None:
            if tracked:
                self._filtered_controller = self.controller_filter.update(
                    pose_sample.controller_pose,
                    pose_sample.capture_monotonic_ms / 1000.0,
                )
                controller_pose = self._filtered_controller
            else:
                controller_pose = self._filtered_controller or pose_sample.controller_pose
            self._target_pose = self.clutch.update(
                grip=pose_sample.grip,
                stream_fresh=freshness.fresh and tracked and not suppress_clutch,
                controller_pose=controller_pose,
                robot_pose=robot_pose,
                wrist_pivot=wrist_pose.position,
                head_quaternion_wxyz=pose_sample.head_quaternion_wxyz,
            )
        else:
            self._target_pose = None

        active = (
            self._target_pose is not None
            and freshness.fresh
            and tracked
            and self.mapper.engaged
        )
        limiter_flags: list[str] = []
        ik_start = ik_end = 0
        diagnostics = None
        q_des = self.q_command.copy()
        gripper_des = self.gripper_command_m
        q_command = self.q_command.copy()
        gripper_command = self.gripper_command_m
        feedforward_velocity = np.zeros(6)

        if home_command_q is not None:
            q_des = home_command_q
            q_command = home_command_q
        elif active:
            ik_start = time.perf_counter_ns()
            q_des, diagnostics = self.solver.solve(self._target_pose, self.q_command)
            ik_end = time.perf_counter_ns()
            if self.control_gripper and pose_sample is not None:
                gripper_des = trigger_to_gripper_position(
                    pose_sample.trigger,
                    self.gripper_open_m,
                    self.gripper_min_m,
                )
            if self.joint_limiter is None:
                q_command = q_des.copy()
            else:
                joint_result = self.joint_limiter.step(q_des, limiter_dt)
                q_command = joint_result.command
                limiter_flags.extend(joint_result.flags("joint"))
            if self.gripper_limiter is None:
                gripper_command = gripper_des
            else:
                gripper_result = self.gripper_limiter.step(
                    np.array([gripper_des]), limiter_dt
                )
                gripper_command = float(gripper_result.command[0])
                limiter_flags.extend(gripper_result.flags("gripper"))
            if self.feedforward_filter is not None:
                feedforward_velocity = self.feedforward_filter.update(
                    q_command, limiter_dt
                )
        else:
            self._reset_shaping()
        if not self.mapper.engaged:
            self._filtered_controller = None
            self.controller_filter.reset()

        return ArmProposal(
            side=self.side,
            q_command=q_command,
            gripper_command_m=gripper_command,
            q_des=q_des,
            gripper_des_m=gripper_des,
            feedforward_velocity=feedforward_velocity,
            target_pose=self._target_pose,
            diagnostics=diagnostics,
            freshness=freshness,
            engaged=self.mapper.engaged,
            active=active or home_command_q is not None,
            tracked=tracked,
            sample=pose_sample,
            ik_start_monotonic_ns=ik_start,
            ik_end_monotonic_ns=ik_end,
            limiter_flags=limiter_flags,
        )

    def commit(self, proposal: ArmProposal) -> None:
        self.q_command = np.asarray(proposal.q_command, dtype=float).reshape(6).copy()
        self.q_des = np.asarray(proposal.q_des, dtype=float).reshape(6).copy()
        self.gripper_command_m = float(proposal.gripper_command_m)
        self.gripper_des_m = float(proposal.gripper_des_m)
        self.feedforward_velocity = np.asarray(
            proposal.feedforward_velocity, dtype=float
        ).reshape(6)

    def commit_bounded_override(self, proposal: ArmProposal) -> None:
        """Commit an externally bounded safe state and reset all shaping there."""

        self.commit(proposal)
        self._reset_shaping()

    def reject(self) -> None:
        """Discard a proposal and restore shaping to the last accepted command.

        Called when the *other* arm, or the combined scene, made this tick
        unsafe. The arm holds its last accepted command rather than continuing
        from a state that was never sent.
        """

        self.rejected_commands += 1
        self._reset_shaping()

    def hold(self) -> None:
        """Hold this arm's last accepted command and re-anchor on recovery."""

        self.holds += 1
        self._reset_shaping()
        if not self.mapper.engaged:
            self._filtered_controller = None
            self.controller_filter.reset()

    def _reset_shaping(self) -> None:
        if self.joint_limiter is not None:
            self.joint_limiter.reset(self.q_command)
        if self.gripper_limiter is not None:
            self.gripper_limiter.reset(np.array([self.gripper_command_m]))
        if self.feedforward_filter is not None:
            self.feedforward_filter.reset(self.q_command)
        self.feedforward_velocity = np.zeros(6)

    def force_reanchor(self) -> None:
        self.clutch.force_reanchor()
        self.controller_filter.reset()
        self._filtered_controller = None
