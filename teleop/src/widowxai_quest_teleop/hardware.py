from __future__ import annotations

import importlib
import importlib.metadata
import platform
import re
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol

import numpy as np


class HardwareUnavailableError(RuntimeError):
    pass


class HardwareSafetyError(RuntimeError):
    pass


_VERSION_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)(?:\.\d+)?")
END_EFFECTOR_PROFILE_TO_VARIANT = {
    # The unversioned follower profile proven with trossen-arm 1.8.6. Driver
    # 1.9.3 (for the arms' firmware 1.9.4) provides the same name with
    # identical parameters; neither driver has the dated follower names.
    "legacy_1_8": "wxai_v0_follower",
    "20250509": "wxai_v0_follower_20250509",
    "20260626": "wxai_v0_follower_20260626",
}
SUPPORTED_FOLLOWER_END_EFFECTORS = tuple(END_EFFECTOR_PROFILE_TO_VARIANT.values())


def version_major_minor(version: str) -> tuple[int, int]:
    """Return a Trossen version's compatibility pair or fail closed."""

    match = _VERSION_RE.search(str(version))
    if match is None:
        raise HardwareSafetyError(f"could not parse Trossen version {version!r}")
    return int(match.group(1)), int(match.group(2))


def require_compatible_versions(driver_version: str, firmware_version: str) -> None:
    """Trossen requires the driver and firmware major/minor to match."""

    driver_pair = version_major_minor(driver_version)
    firmware_pair = version_major_minor(firmware_version)
    if driver_pair != firmware_pair:
        raise HardwareSafetyError(
            "Trossen driver/firmware mismatch: "
            f"driver={driver_version}, firmware={firmware_version}; major.minor must match"
        )


@dataclass(frozen=True)
class HardwareDiscovery:
    robot_ip: str
    firmware_version: str
    driver_version: str


@dataclass(frozen=True)
class HardwareState:
    q_arm: np.ndarray
    gripper_position_m: float
    joint_limits: np.ndarray
    firmware_version: str
    driver_version: str = "unknown"
    position_tolerances: np.ndarray | None = None

    def __post_init__(self) -> None:
        q_arm = np.asarray(self.q_arm, dtype=float).reshape(6).copy()
        gripper = float(self.gripper_position_m)
        limits = np.asarray(self.joint_limits, dtype=float).reshape(7, 2).copy()
        if self.position_tolerances is None:
            tolerances = np.full(7, 1e-6)
        else:
            tolerances = np.asarray(self.position_tolerances, dtype=float).reshape(7).copy()
        if not np.all(np.isfinite(q_arm)) or not np.isfinite(gripper):
            raise HardwareSafetyError("hardware feedback contains non-finite positions")
        if not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
            raise HardwareSafetyError("hardware reported invalid joint limits")
        if not np.all(np.isfinite(tolerances)) or np.any(tolerances < 0.0):
            raise HardwareSafetyError("hardware reported invalid position tolerances")
        positions = np.concatenate([q_arm, [gripper]])
        if np.any(positions < limits[:, 0] - tolerances) or np.any(
            positions > limits[:, 1] + tolerances
        ):
            raise HardwareSafetyError("hardware feedback is outside the reported joint limits")
        object.__setattr__(self, "q_arm", q_arm)
        object.__setattr__(self, "gripper_position_m", gripper)
        object.__setattr__(self, "joint_limits", limits)
        object.__setattr__(self, "position_tolerances", tolerances)


class HardwareBackend(Protocol):
    def connect(self) -> HardwareState: ...
    def enable_position_control(self, *, include_gripper: bool = True) -> None: ...
    def read_state(self) -> HardwareState: ...
    def send_positions(
        self,
        q_command: np.ndarray,
        gripper_position_m: float,
        *,
        include_gripper: bool = True,
        arm_feedforward_velocity: np.ndarray | None = None,
    ) -> None: ...
    def move_to_rest(
        self,
        q_rest: np.ndarray,
        gripper_position_m: float,
        *,
        duration_s: float,
        include_gripper: bool = True,
    ) -> None: ...
    def move_gripper_blocking(self, gripper_position_m: float, *, duration_s: float) -> None: ...
    def safe_hold(self) -> None: ...
    def close(self) -> None: ...


class CommandGate:
    """Reject unsafe commands without filtering, buffering, or modifying them."""

    def __init__(
        self,
        initial_q: np.ndarray,
        initial_gripper_m: float,
        joint_limits: np.ndarray,
        max_joint_delta_rad: np.ndarray,
        *,
        joint_limit_margin_rad: float,
        gripper_limits_m: tuple[float, float],
        max_gripper_delta_m: float,
    ) -> None:
        self.previous_q = np.asarray(initial_q, dtype=float).reshape(6).copy()
        self.previous_gripper_m = float(initial_gripper_m)
        limits = np.asarray(joint_limits, dtype=float).reshape(7, 2)
        margin = float(joint_limit_margin_rad)
        if not np.isfinite(margin) or margin < 0.0:
            raise HardwareSafetyError("joint-limit margin must be finite and nonnegative")
        if not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
            raise HardwareSafetyError("command gate received invalid joint limits")
        self.lower = limits[:6, 0] + margin
        self.upper = limits[:6, 1] - margin
        if np.any(self.lower >= self.upper):
            raise HardwareSafetyError("joint-limit margin leaves no valid command range")
        self.max_joint_delta = np.asarray(max_joint_delta_rad, dtype=float).reshape(6).copy()
        if not np.all(np.isfinite(self.max_joint_delta)) or np.any(self.max_joint_delta <= 0.0):
            raise HardwareSafetyError("joint-step caps must be finite and positive")
        self.gripper_lower_m, self.gripper_upper_m = map(float, gripper_limits_m)
        self.max_gripper_delta_m = float(max_gripper_delta_m)
        if (
            not np.isfinite(self.gripper_lower_m)
            or not np.isfinite(self.gripper_upper_m)
            or self.gripper_lower_m >= self.gripper_upper_m
        ):
            raise HardwareSafetyError("gripper limits must be finite and ordered")
        if not np.isfinite(self.max_gripper_delta_m) or self.max_gripper_delta_m <= 0.0:
            raise HardwareSafetyError("gripper step cap must be finite and positive")
        self.validate(self.previous_q, self.previous_gripper_m, update=False)

    def validate(self, q_command: np.ndarray, gripper_position_m: float, *, update: bool = True) -> None:
        q = np.asarray(q_command, dtype=float).reshape(6)
        gripper = float(gripper_position_m)
        if not np.all(np.isfinite(q)) or not np.isfinite(gripper):
            raise HardwareSafetyError("non-finite hardware command rejected")
        outside = np.flatnonzero((q < self.lower) | (q > self.upper))
        if outside.size:
            index = int(outside[0])
            raise HardwareSafetyError(
                f"joint {index} command {q[index]:.6f} outside the demo limit margin "
                f"[{self.lower[index]:.6f}, {self.upper[index]:.6f}]"
            )
        if np.any(np.abs(q - self.previous_q) > self.max_joint_delta + 1e-12):
            raise HardwareSafetyError("joint command exceeded the configured per-tick cap")
        if not self.gripper_lower_m <= gripper <= self.gripper_upper_m:
            raise HardwareSafetyError(
                f"gripper command {gripper:.6f} outside the demo range "
                f"[{self.gripper_lower_m:.6f}, {self.gripper_upper_m:.6f}]"
            )
        if abs(gripper - self.previous_gripper_m) > self.max_gripper_delta_m + 1e-12:
            raise HardwareSafetyError("gripper command exceeded the configured per-tick cap")
        if update:
            self.previous_q = q.copy()
            self.previous_gripper_m = gripper


class DryRunBackend:
    """In-memory backend for exercising the complete demo without arm I/O."""

    def __init__(self, initial_q: np.ndarray, joint_limits: np.ndarray, gripper_position_m: float = 0.044) -> None:
        self.q_arm = np.asarray(initial_q, dtype=float).reshape(6).copy()
        arm_limits = np.asarray(joint_limits, dtype=float).reshape(6, 2)
        self.limits = np.vstack([arm_limits, np.array([0.0, 0.044])])
        self.gripper = float(gripper_position_m)
        self.enabled = False
        self.commands = 0

    def connect(self) -> HardwareState:
        return self.read_state()

    def enable_position_control(self, *, include_gripper: bool = True) -> None:
        self.enabled = True

    def read_state(self) -> HardwareState:
        return HardwareState(self.q_arm, self.gripper, self.limits, "dry-run")

    def send_positions(
        self,
        q_command: np.ndarray,
        gripper_position_m: float,
        *,
        include_gripper: bool = True,
        arm_feedforward_velocity: np.ndarray | None = None,
    ) -> None:
        if not self.enabled:
            raise HardwareSafetyError("dry-run position control is not enabled")
        self.q_arm = np.asarray(q_command, dtype=float).reshape(6).copy()
        if arm_feedforward_velocity is not None:
            velocity = np.asarray(arm_feedforward_velocity, dtype=float).reshape(6)
            if not np.all(np.isfinite(velocity)):
                raise HardwareSafetyError("dry-run feedforward velocity is non-finite")
        if include_gripper:
            self.gripper = float(gripper_position_m)
        self.commands += 1

    def move_to_rest(
        self,
        q_rest: np.ndarray,
        gripper_position_m: float,
        *,
        duration_s: float,
        include_gripper: bool = True,
    ) -> None:
        if not self.enabled:
            raise HardwareSafetyError("dry-run position control is not enabled")
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise HardwareSafetyError("rest move duration must be finite and positive")
        self.q_arm = np.asarray(q_rest, dtype=float).reshape(6).copy()
        if include_gripper:
            self.gripper = float(gripper_position_m)
        self.commands += 1

    def move_gripper_blocking(self, gripper_position_m: float, *, duration_s: float) -> None:
        if not self.enabled:
            raise HardwareSafetyError("dry-run position control is not enabled")
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise HardwareSafetyError("gripper move duration must be finite and positive")
        if not np.isfinite(gripper_position_m):
            raise HardwareSafetyError("non-finite gripper command rejected")
        self.gripper = float(gripper_position_m)
        self.commands += 1

    def safe_hold(self) -> None:
        return

    def close(self) -> None:
        self.enabled = False


class TrossenArmBackend:
    """Thin adapter over Trossen's official ``trossen_arm`` Python API."""

    def __init__(
        self,
        robot_ip: str,
        *,
        driver_module: ModuleType | object | None = None,
        platform_name: str | None = None,
        driver_version: str | None = None,
        required_driver_version: str | None = None,
        command_goal_time_s: float = 0.0,
        end_effector_variant: str | None = None,
    ) -> None:
        self.robot_ip = str(robot_ip)
        self._module = driver_module
        self._platform_name = platform.system() if platform_name is None else platform_name
        self._driver_version = driver_version
        self._required_driver_version = required_driver_version
        self._command_goal_time_s = float(command_goal_time_s)
        self._end_effector_variant = end_effector_variant
        if not np.isfinite(self._command_goal_time_s) or self._command_goal_time_s < 0.0:
            raise ValueError("command_goal_time_s must be finite and nonnegative")
        self._driver = None
        self._enabled = False
        self._gripper_enabled = False
        self._last_positions: np.ndarray | None = None
        self._joint_limits: np.ndarray | None = None
        self._position_tolerances: np.ndarray | None = None
        self._firmware_version = "unknown"

    def _load_module(self):
        if self._platform_name not in ("Linux", "Darwin"):
            raise HardwareUnavailableError(
                "Trossen's official trossen_arm driver supports Ubuntu and macOS, not Windows. "
                "No physical command was attempted."
            )
        if self._module is None:
            try:
                self._module = importlib.import_module("trossen_arm")
            except ImportError as exc:
                raise HardwareUnavailableError(
                    "Official trossen_arm Python driver is not installed. Install the version "
                    "whose major.minor matches the arm firmware."
                ) from exc
        return self._module

    def _installed_driver_version(self) -> str:
        if self._driver_version is None:
            try:
                self._driver_version = importlib.metadata.version("trossen-arm")
            except importlib.metadata.PackageNotFoundError as exc:
                raise HardwareUnavailableError(
                    "Could not determine the installed trossen-arm version. Reinstall the "
                    "firmware-matched official driver in this virtual environment."
                ) from exc
        if (
            self._required_driver_version is not None
            and self._driver_version != self._required_driver_version
        ):
            raise HardwareUnavailableError(
                "wrong Trossen driver for this arm profile: "
                f"installed={self._driver_version}, required={self._required_driver_version}; "
                "no arm connection was attempted"
            )
        return self._driver_version

    @staticmethod
    def _enum_value(value: object) -> int:
        return int(getattr(value, "value", value))

    def _discover_result(self, module):
        if not hasattr(module.TrossenArmDriver, "discover"):
            raise HardwareUnavailableError(
                "trossen-arm 1.8.x has no discovery API; use the legacy no-motion state "
                "preflight with the known controller IP and legacy_1_8 follower profile"
            )
        subnet = self.robot_ip.rsplit(".", 1)[0]
        host = int(self.robot_ip.rsplit(".", 1)[1])
        results = module.TrossenArmDriver.discover(
            subnet=subnet,
            ip_start=host,
            ip_end=host,
            timeout=0.05,
        )
        if len(results) != 1 or str(results[0].ip) != self.robot_ip:
            raise HardwareUnavailableError(f"official discovery did not find exactly one arm at {self.robot_ip}")
        result = results[0]
        if self._enum_value(result.error_state) != 0:
            raise HardwareSafetyError(f"arm discovery reported error state {result.error_state}")
        if self._enum_value(result.model) != self._enum_value(module.Model.wxai_v0):
            raise HardwareSafetyError("discovered controller is not a WidowXAI")
        self._firmware_version = str(result.firmware_version)
        require_compatible_versions(self._installed_driver_version(), self._firmware_version)
        return result

    def discover(self) -> HardwareDiscovery:
        """Validate one controller using discovery only; do not configure a driver."""

        module = self._load_module()
        self._discover_result(module)
        return HardwareDiscovery(
            self.robot_ip,
            self._firmware_version,
            self._installed_driver_version(),
        )

    def connect(self) -> HardwareState:
        module = self._load_module()
        driver_version = self._installed_driver_version()
        has_discovery = hasattr(module.TrossenArmDriver, "discover")
        if has_discovery:
            self._discover_result(module)

        if self._end_effector_variant not in SUPPORTED_FOLLOWER_END_EFFECTORS:
            choices = ", ".join(SUPPORTED_FOLLOWER_END_EFFECTORS)
            raise HardwareSafetyError(
                "physical end-effector profile was not selected explicitly; "
                f"choose one of: {choices}"
            )
        try:
            end_effector = getattr(module.StandardEndEffector, self._end_effector_variant)
        except AttributeError as exc:
            raise HardwareUnavailableError(
                f"installed driver does not provide {self._end_effector_variant}"
            ) from exc

        driver = module.TrossenArmDriver()
        self._driver = driver
        try:
            driver.configure(
                module.Model.wxai_v0,
                end_effector,
                self.robot_ip,
                False,
            )
            if not has_discovery:
                if not hasattr(driver, "get_controller_version"):
                    raise HardwareUnavailableError(
                        "legacy driver cannot report the controller firmware version"
                    )
                self._firmware_version = str(driver.get_controller_version())
                require_compatible_versions(driver_version, self._firmware_version)
            return self.read_state()
        except Exception:
            try:
                driver.cleanup()
            finally:
                self._driver = None
            raise

    def enable_position_control(self, *, include_gripper: bool = True) -> None:
        if self._driver is None:
            raise HardwareSafetyError("driver is not connected")
        self._driver.set_arm_modes(self._module.Mode.position)
        if include_gripper:
            self._driver.set_gripper_mode(self._module.Mode.position)
        self._enabled = True
        self._gripper_enabled = bool(include_gripper)

    def read_state(self) -> HardwareState:
        if self._driver is None:
            raise HardwareSafetyError("driver is not connected")
        positions = np.asarray(self._driver.get_all_positions(), dtype=float).reshape(7)
        if self._joint_limits is None:
            limits_raw = self._driver.get_joint_limits()
            self._joint_limits = np.array(
                [[limit.position_min, limit.position_max] for limit in limits_raw],
                dtype=float,
            )
            self._position_tolerances = np.array(
                [getattr(limit, "position_tolerance", 1e-4) for limit in limits_raw],
                dtype=float,
            )
        limits = self._joint_limits
        self._last_positions = positions.copy()
        self._joint_limits = limits.copy()
        return HardwareState(
            positions[:6],
            positions[6],
            limits,
            self._firmware_version,
            self._installed_driver_version(),
            self._position_tolerances,
        )

    def send_positions(
        self,
        q_command: np.ndarray,
        gripper_position_m: float,
        *,
        include_gripper: bool = True,
        arm_feedforward_velocity: np.ndarray | None = None,
    ) -> None:
        if self._driver is None or not self._enabled:
            raise HardwareSafetyError("position control is not enabled")
        q = np.asarray(q_command, dtype=float).reshape(6)
        if arm_feedforward_velocity is None:
            self._driver.set_arm_positions(q, self._command_goal_time_s, False)
        else:
            feedforward = np.asarray(arm_feedforward_velocity, dtype=float).reshape(6)
            if not np.all(np.isfinite(feedforward)):
                raise HardwareSafetyError("arm feedforward velocity is non-finite")
            self._driver.set_arm_positions(
                q,
                self._command_goal_time_s,
                False,
                feedforward,
            )
        if include_gripper:
            if not self._gripper_enabled:
                raise HardwareSafetyError("gripper position control is not enabled")
            self._driver.set_gripper_position(
                float(gripper_position_m),
                self._command_goal_time_s,
                False,
            )
        self._last_positions = np.concatenate([q, [float(gripper_position_m)]])

    def move_to_rest(
        self,
        q_rest: np.ndarray,
        gripper_position_m: float,
        *,
        duration_s: float,
        include_gripper: bool = True,
    ) -> None:
        """Reproduce the proven XRoboToolkit blocking sleep transition."""

        if self._driver is None or not self._enabled:
            raise HardwareSafetyError("position control is not enabled")
        duration = float(duration_s)
        if not np.isfinite(duration) or duration <= 0.0:
            raise HardwareSafetyError("rest move duration must be finite and positive")
        q = np.asarray(q_rest, dtype=float).reshape(6)
        if not np.all(np.isfinite(q)) or not np.isfinite(gripper_position_m):
            raise HardwareSafetyError("non-finite rest command rejected")

        self._driver.set_arm_modes(self._module.Mode.position)
        self._driver.set_arm_positions(q, duration, True)
        if include_gripper:
            self._driver.set_gripper_mode(self._module.Mode.position)
            self._driver.set_gripper_position(float(gripper_position_m), duration, True)
            self._gripper_enabled = True
        self._last_positions = np.concatenate([q, [float(gripper_position_m)]])

    def move_gripper_blocking(self, gripper_position_m: float, *, duration_s: float) -> None:
        """Use the previous XRoboToolkit's one-shot blocking gripper move."""

        if self._driver is None or not self._enabled:
            raise HardwareSafetyError("position control is not enabled")
        duration = float(duration_s)
        gripper = float(gripper_position_m)
        if not np.isfinite(duration) or duration <= 0.0:
            raise HardwareSafetyError("gripper move duration must be finite and positive")
        if not np.isfinite(gripper):
            raise HardwareSafetyError("non-finite gripper command rejected")
        self._driver.set_gripper_mode(self._module.Mode.position)
        self._driver.set_gripper_position(gripper, duration, True)
        self._gripper_enabled = True
        if self._last_positions is not None:
            self._last_positions[6] = gripper

    def safe_hold(self) -> None:
        if self._driver is None or not self._enabled:
            return
        try:
            state = self.read_state()
            if self._gripper_enabled:
                self._driver.set_arm_modes(self._module.Mode.position)
                self._driver.set_gripper_mode(self._module.Mode.position)
                self._driver.set_arm_positions(state.q_arm, 0.0, False)
                self._driver.set_gripper_position(
                    state.gripper_position_m,
                    0.0,
                    False,
                )
                self._last_positions = np.concatenate(
                    [state.q_arm, [state.gripper_position_m]]
                )
            else:
                self._driver.set_arm_modes(self._module.Mode.position)
                self._driver.set_arm_positions(state.q_arm, 0.0, False)
                self._last_positions = np.concatenate(
                    [state.q_arm, [state.gripper_position_m]]
                )
        except Exception as exc:
            raise HardwareSafetyError(
                "could not confirm a measured-position shutdown hold; "
                "driver cleanup will continue and controller power should be cut"
            ) from exc

    def close(self) -> None:
        if self._driver is None:
            return
        try:
            self._driver.cleanup()
        finally:
            self._driver = None
            self._enabled = False
            self._gripper_enabled = False
