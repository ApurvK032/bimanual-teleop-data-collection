from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
TASK_PROFILE_CATALOG = (
    PROJECT_ROOT
    / "configs"
    / "calibrations"
    / "task_profiles.json"
)
TASK_PROFILE_NAMES = (
    "left_behind",
    "left_mirror",
    "right_behind",
    "right_mirror",
)


def task_profile_for_quest_selection(hand: str, mapping_mode: str) -> str:
    selected_hand = str(hand).lower()
    selected_mode = str(mapping_mode).lower()
    if selected_hand not in ("left", "right"):
        raise ValueError(f"unsupported Quest hand: {hand}")
    semantics_by_mode = {"real": "behind", "mirror": "mirror"}
    if selected_mode not in semantics_by_mode:
        raise ValueError(f"unsupported Quest mapping mode: {mapping_mode}")
    return f"{selected_hand}_{semantics_by_mode[selected_mode]}"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {resolved}")
    config["_config_path"] = str(resolved)
    return config


def apply_task_profile(
    config: dict[str, Any],
    profile_name: str,
    catalog_path: str | Path = TASK_PROFILE_CATALOG,
) -> dict[str, Any]:
    """Apply one fixed hand/semantic mapping to an already loaded config."""

    if profile_name not in TASK_PROFILE_NAMES:
        raise ValueError(
            f"unknown task profile {profile_name!r}; choose one of {TASK_PROFILE_NAMES}"
        )
    resolved_catalog = resolve_project_path(catalog_path)
    catalog = json.loads(resolved_catalog.read_text(encoding="utf-8"))
    profile_path = catalog.get("profiles", {}).get(profile_name)
    if not isinstance(profile_path, str):
        raise ValueError(f"task profile catalog has no path for {profile_name}")
    resolved_profile = resolve_project_path(profile_path)
    profile = json.loads(resolved_profile.read_text(encoding="utf-8"))
    quest_input = profile.get("quest_input")
    if not isinstance(quest_input, dict):
        raise ValueError(f"task profile {profile_name} has no quest_input")
    expected_hand, expected_semantics = profile_name.split("_", maxsplit=1)
    hand = quest_input.get("hand")
    mapping_mode = quest_input.get("mapping_mode")
    semantics = profile.get("task_mapping_semantics", {}).get("name")
    if hand != expected_hand:
        raise ValueError(
            f"task profile {profile_name} reports hand {hand!r}, expected {expected_hand!r}"
        )
    expected_mode = "real" if expected_semantics == "behind" else "mirror"
    if mapping_mode != expected_mode:
        raise ValueError(
            f"task profile {profile_name} reports mode {mapping_mode!r}, "
            f"expected {expected_mode!r}"
        )
    if semantics != expected_semantics:
        raise ValueError(
            f"task profile {profile_name} reports semantics {semantics!r}, "
            f"expected {expected_semantics!r}"
        )

    config["quest"]["hand"] = hand
    config["quest"]["mapping_mode"] = mapping_mode
    config["quest"]["calibration"] = profile_path
    config["project"]["mode"] = (
        f"{config['project']['mode']}__task_profile_{profile_name}"
    )
    config["_task_profile"] = {
        "name": profile_name,
        "catalog": str(resolved_catalog),
        "calibration": str(resolved_profile),
        "hand": hand,
        "mapping_mode": mapping_mode,
        "semantics": semantics,
    }
    return config


# --------------------------------------------------------------------------
# Dual-arm configuration
# --------------------------------------------------------------------------

DUAL_ARM_SIDES = ("left", "right")


class DualArmConfigError(ValueError):
    """Raised when a dual-arm configuration is unsafe or incomplete."""


@dataclass(frozen=True)
class ArmPlacement:
    """One arm base expressed in the shared dual-arm world frame."""

    position_m: np.ndarray
    quaternion_wxyz: np.ndarray
    measurement_status: str = "synthetic"

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=float).reshape(3).copy()
        quaternion = np.asarray(self.quaternion_wxyz, dtype=float).reshape(4).copy()
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            raise DualArmConfigError("arm base transform must contain finite values")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-9:
            raise DualArmConfigError("arm base quaternion must be non-degenerate")
        status = str(self.measurement_status).strip().lower()
        if status not in ("placeholder", "measured", "synthetic"):
            raise DualArmConfigError(
                "arm base measurement_status must be placeholder, measured, or synthetic"
            )
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "quaternion_wxyz", quaternion / norm)
        object.__setattr__(self, "measurement_status", status)

    @property
    def measured(self) -> bool:
        return self.measurement_status == "measured"


@dataclass(frozen=True)
class ArmConfig:
    """Everything one arm needs that the other arm must not share."""

    side: str
    controller_hand: str
    mapping_mode: str
    robot_ip: str
    calibration: str
    calibration_status: str
    placement: ArmPlacement
    settings: dict[str, Any]

    @property
    def calibration_accepted(self) -> bool:
        return self.calibration_status.startswith("accepted")


def _load_calibration(calibration_path: str | Path) -> dict[str, Any]:
    resolved = resolve_project_path(calibration_path)
    if not resolved.exists():
        raise DualArmConfigError(f"calibration file is missing: {resolved}")
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DualArmConfigError(f"calibration file is not valid JSON: {resolved}") from exc
    if not isinstance(document, dict):
        raise DualArmConfigError(f"calibration file must be a mapping: {resolved}")
    return document


def calibration_quest_selection(
    calibration_path: str | Path,
) -> tuple[str | None, str | None]:
    """Report the Quest hand and mapping mode a calibration was captured under.

    Anything the file does not record comes back as ``None``, so an older
    calibration reads as unknown rather than as agreement.
    """

    quest_input = _load_calibration(calibration_path).get("quest_input")
    if not isinstance(quest_input, dict):
        return (None, None)

    def recorded(key: str) -> str | None:
        value = quest_input.get(key)
        return str(value).lower() if isinstance(value, str) and value.strip() else None

    return (recorded("hand"), recorded("mapping_mode"))


def calibration_acceptance_status(calibration_path: str | Path) -> str:
    """Report a calibration's recorded operator acceptance.

    Acceptance is currently written in three unnormalised places. This reads
    all of them and fails closed to ``"unrecorded"`` rather than guessing, so a
    dual-arm live gate can never infer acceptance that nobody recorded.
    """

    document = _load_calibration(calibration_path)
    for section, key in (
        ("physical_validation", "status"),
        ("physical_axis_acceptance", "status"),
        ("physical_axis_validation", "status"),
        ("task_mapping_semantics", "status"),
    ):
        block = document.get(section)
        if isinstance(block, dict):
            status = block.get(key)
            if isinstance(status, str) and status.strip():
                return status.strip()
    return "unrecorded"


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    block = config.get(key)
    if not isinstance(block, dict):
        raise DualArmConfigError(f"dual-arm configuration requires a {key!r} mapping")
    return block


def _parse_placement(side: str, raw: Any) -> ArmPlacement:
    if not isinstance(raw, dict):
        raise DualArmConfigError(
            f"arms.{side}.base_transform is required; measure it against the shared "
            "world frame instead of copying an example"
        )
    for key in ("position_m", "quaternion_wxyz", "measurement_status"):
        if key not in raw:
            raise DualArmConfigError(f"arms.{side}.base_transform.{key} is required")
    status = str(raw["measurement_status"]).strip().lower()
    if status not in ("placeholder", "measured"):
        raise DualArmConfigError(
            f"arms.{side}.base_transform.measurement_status must be "
            "'placeholder' or 'measured'"
        )
    try:
        return ArmPlacement(raw["position_m"], raw["quaternion_wxyz"], status)
    except (TypeError, ValueError) as exc:
        raise DualArmConfigError(f"arms.{side}.base_transform is invalid: {exc}") from None


def _parse_arm(side: str, raw: Any) -> ArmConfig:
    if not isinstance(raw, dict):
        raise DualArmConfigError(f"arms.{side} must be a mapping")

    controller_hand = str(raw.get("controller_hand", "")).lower()
    if controller_hand not in DUAL_ARM_SIDES:
        raise DualArmConfigError(
            f"arms.{side}.controller_hand must be 'left' or 'right'"
        )
    mapping_mode = str(raw.get("mapping_mode", "real")).lower()
    if mapping_mode not in ("real", "mirror"):
        raise DualArmConfigError(f"arms.{side}.mapping_mode must be 'real' or 'mirror'")

    robot_ip = str(raw.get("robot_ip", "")).strip()
    if not robot_ip:
        raise DualArmConfigError(f"arms.{side}.robot_ip is required")
    try:
        ipaddress.ip_address(robot_ip)
    except ValueError:
        raise DualArmConfigError(
            f"arms.{side}.robot_ip is not a valid IP address: {robot_ip!r}"
        ) from None

    calibration = raw.get("calibration")
    if not isinstance(calibration, str) or not calibration.strip():
        raise DualArmConfigError(f"arms.{side}.calibration is required")
    status = calibration_acceptance_status(calibration)

    scales = {
        f"arms.{side}.translation_scale": raw.get("translation_scale"),
        f"arms.{side}.rotation_scale": raw.get("rotation_scale"),
        f"arms.{side}.position_reach_limit_m": raw.get("position_reach_limit_m"),
        f"arms.{side}.rotation_reach_limit_rad": raw.get("rotation_reach_limit_rad"),
    }
    for name, value in scales.items():
        if value is None:
            raise DualArmConfigError(f"{name} is required")
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise DualArmConfigError(f"{name} must be finite and positive")

    return ArmConfig(
        side=side,
        controller_hand=controller_hand,
        mapping_mode=mapping_mode,
        robot_ip=robot_ip,
        calibration=calibration,
        calibration_status=status,
        placement=_parse_placement(side, raw.get("base_transform")),
        settings=dict(raw),
    )


def parse_dual_arm_config(config: dict[str, Any]) -> dict[str, ArmConfig]:
    """Validate the per-arm blocks and fail closed on any ambiguity.

    Cross-arm collision checking and calibration acceptance are additionally
    gated at live-output time by :func:`require_live_dual_arm_config`. This
    function stays usable for MuJoCo work, which the project requires before
    any physical dual-arm attempt.
    """

    quest = _require_mapping(config, "quest")
    if str(quest.get("mode", "")).lower() != "bimanual":
        raise DualArmConfigError("dual-arm configuration requires quest.mode: bimanual")

    arms_block = _require_mapping(config, "arms")
    missing = [side for side in DUAL_ARM_SIDES if side not in arms_block]
    if missing:
        raise DualArmConfigError(f"dual-arm configuration is missing arms: {missing}")
    unknown = sorted(set(arms_block) - set(DUAL_ARM_SIDES))
    if unknown:
        raise DualArmConfigError(f"unknown dual-arm arm entries: {unknown}")

    arms = {side: _parse_arm(side, arms_block[side]) for side in DUAL_ARM_SIDES}

    hands = [arm.controller_hand for arm in arms.values()]
    if len(set(hands)) != len(hands):
        raise DualArmConfigError(
            "both arms reference the same logical controller hand; one controller "
            "cannot drive two arms"
        )
    ips = [arm.robot_ip for arm in arms.values()]
    if len(set(ips)) != len(ips):
        raise DualArmConfigError(f"both arms use the same controller IP: {ips[0]}")

    # A task frame is only valid for the hand and page mode it was captured
    # under. A profile that swaps the hand-to-arm assignment or switches to
    # Mirrored makes this easy to get wrong, and the mismatch is otherwise
    # invisible until an arm moves the wrong way.
    for side in DUAL_ARM_SIDES:
        arm = arms[side]
        hand, mapping_mode = calibration_quest_selection(arm.calibration)
        if hand is not None and hand != arm.controller_hand:
            raise DualArmConfigError(
                f"arms.{side} drives the {arm.controller_hand} controller but its "
                f"calibration was captured for the {hand} hand: {arm.calibration}"
            )
        if mapping_mode is not None and mapping_mode != arm.mapping_mode:
            raise DualArmConfigError(
                f"arms.{side} expects {arm.mapping_mode} mapping but its calibration "
                f"was captured under {mapping_mode}: {arm.calibration}"
            )

    placements = [arms[side].placement.position_m for side in DUAL_ARM_SIDES]
    separation_m = float(np.linalg.norm(placements[0] - placements[1]))
    if separation_m <= 0.0:
        raise DualArmConfigError("both arm bases are at the same measured position")

    safety = config.get("safety")
    if not isinstance(safety, dict):
        raise DualArmConfigError("dual-arm configuration requires a 'safety' mapping")
    for key in ("cross_arm_collision", "coordinated_fault_hold"):
        if not isinstance(safety.get(key), bool):
            raise DualArmConfigError(f"safety.{key} must be an explicit boolean")
    clearance = float(safety.get("cross_arm_clearance_m", 0.0))
    if not np.isfinite(clearance) or clearance < 0.0:
        raise DualArmConfigError("safety.cross_arm_clearance_m must be finite and nonnegative")
    max_skew_s = float(safety.get("max_command_skew_s", 0.0))
    if not np.isfinite(max_skew_s) or max_skew_s <= 0.0:
        raise DualArmConfigError("safety.max_command_skew_s must be finite and positive")

    config["_dual_arm"] = {
        "sides": list(DUAL_ARM_SIDES),
        "base_separation_m": separation_m,
        "base_transform_status": {
            side: arms[side].placement.measurement_status for side in DUAL_ARM_SIDES
        },
        "controller_hands": {side: arms[side].controller_hand for side in DUAL_ARM_SIDES},
        "robot_ips": {side: arms[side].robot_ip for side in DUAL_ARM_SIDES},
        "calibration_status": {
            side: arms[side].calibration_status for side in DUAL_ARM_SIDES
        },
    }
    return arms


def require_live_dual_arm_config(
    config: dict[str, Any],
    arms: dict[str, ArmConfig],
    *,
    allow_unvalidated_calibrations: bool = False,
) -> dict[str, str]:
    """Additional gates that only a physical dual-arm run must satisfy.

    ``allow_unvalidated_calibrations`` is an explicit operator override for the
    calibration-acceptance gate only. It never relaxes measured base geometry,
    cross-arm collision, coordinated fault hold, or any
    joint/stale/feedback/driver/shutdown gate, and it does not modify the
    calibration files: the pending statuses are returned so the caller can
    record the override in the run's evidence.
    """

    safety = config.get("safety", {})
    if not safety.get("cross_arm_collision", False):
        raise DualArmConfigError(
            "safety.cross_arm_collision must be enabled for live dual-arm output"
        )
    if not safety.get("coordinated_fault_hold", False):
        raise DualArmConfigError(
            "safety.coordinated_fault_hold must be enabled for live dual-arm output"
        )
    pending = {
        side: arm.calibration_status
        for side, arm in sorted(arms.items())
        if not arm.calibration_accepted
    }
    unmeasured = {
        side: arm.placement.measurement_status
        for side, arm in sorted(arms.items())
        if not arm.placement.measured
    }
    if pending and not allow_unvalidated_calibrations:
        geometry_note = (
            ""
            if not unmeasured
            else f"; live also requires measured base transforms: {unmeasured}"
        )
        raise DualArmConfigError(
            "live dual-arm output requires an explicitly accepted calibration for "
            f"each arm; pending: {pending}{geometry_note}"
        )
    if unmeasured:
        raise DualArmConfigError(
            "live dual-arm output requires a measured base transform for each arm; "
            f"unmeasured: {unmeasured}. The calibration override does not bypass "
            "bench geometry."
        )
    if pending:
        # Record the override in the config so TelemetryLogger's config
        # snapshot carries it. The calibration files stay untouched, so the
        # repository's acceptance evidence is never falsified by a run.
        config.setdefault("_dual_arm", {})["calibration_override"] = {
            "authorized_by": "operator",
            "scope": "calibration_acceptance_only",
            "pending": dict(pending),
        }
    return pending


def dual_live_confirmation_token(arms: dict[str, ArmConfig]) -> str:
    """Build the dual-arm live token.

    Deliberately distinct from the single-arm ``LIVE-WIDOWXAI-<ip>`` token so a
    single-arm confirmation can never enable two arms.
    """

    ips = "-".join(arms[side].robot_ip for side in DUAL_ARM_SIDES)
    return f"LIVE-WIDOWXAI-DUAL-{ips}"
