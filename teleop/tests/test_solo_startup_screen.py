"""Solo (no stationary-arm guard) startup screen for the right arm's rest overlap.

The owner accepted a 2 mm marginal-start rule on 2026-08-12 for the known
near-rest link_2/link_4 model overlap and, on 2026-10-23, approved sharing it
with the solo startup check. Every other collision check stays zero-tolerance.
"""

import numpy as np
import pytest

from scripts.run_hardware import return_to_rest
from widowxai_quest_teleop.hardware import HardwareSafetyError, HardwareState
from widowxai_quest_teleop.model import WidowXAIModel
from widowxai_quest_teleop.one_arm_safety import (
    MARGINAL_START_CONTACT_M,
    screen_self_collision_path,
)

HOME_Q = np.array([0.0, np.pi / 3.0, 5.0 * np.pi / 12.0, -np.pi / 3.0, 0.0, 0.0])
REST_Q = np.zeros(6)
# Right arm .3 at rest on firmware 1.9.4, as printed by the 2026-10-23 launcher.
MEASURED_RIGHT_REST_Q = np.array([-0.001, -0.004, -0.007, -0.012, 0.002, -0.0])


@pytest.fixture(scope="module")
def single_model() -> WidowXAIModel:
    return WidowXAIModel()


def screen(model, start, end, *, allow=MARGINAL_START_CONTACT_M, samples=501):
    return screen_self_collision_path(
        model,
        start_q=start,
        end_q=end,
        start_gripper_m=0.0,
        samples=samples,
        label="arm 192.168.1.3",
        allow_marginal_start_m=allow,
    )


def test_marginal_start_limit_is_the_accepted_2mm() -> None:
    assert MARGINAL_START_CONTACT_M == pytest.approx(0.002)


def test_measured_right_rest_contact_is_accepted_only_because_it_clears(single_model) -> None:
    depth, cleared_at = screen(single_model, MEASURED_RIGHT_REST_Q, HOME_Q)

    assert 0.0 < depth <= MARGINAL_START_CONTACT_M
    assert cleared_at is not None and cleared_at < 0.01


def test_without_the_allowance_the_same_start_still_fails_closed(single_model) -> None:
    with pytest.raises(HardwareSafetyError, match=r"rejects the path at 0\.0%"):
        screen(single_model, MEASURED_RIGHT_REST_Q, HOME_Q, allow=0.0)


def test_clear_start_gets_no_allowance_for_later_contact(single_model) -> None:
    with pytest.raises(HardwareSafetyError, match="rejects the path near"):
        screen(single_model, REST_Q, MEASURED_RIGHT_REST_Q)


def test_start_contact_above_2mm_is_rejected(single_model) -> None:
    deep = MEASURED_RIGHT_REST_Q.copy()
    deep[2] = -0.015  # about 2.8 mm of model overlap

    with pytest.raises(HardwareSafetyError, match=r"above the 2\.000 mm marginal limit"):
        screen(single_model, deep, HOME_Q)


def test_path_that_deepens_the_start_contact_is_rejected(single_model) -> None:
    deeper = MEASURED_RIGHT_REST_Q.copy()
    deeper[2] = -0.010  # about 1.6 mm, deeper than the measured start

    with pytest.raises(HardwareSafetyError, match="deepens the measured start contact"):
        screen(single_model, MEASURED_RIGHT_REST_Q, deeper)


def test_start_contact_that_never_clears_is_rejected(single_model) -> None:
    with pytest.raises(HardwareSafetyError, match="never clears"):
        screen(single_model, MEASURED_RIGHT_REST_Q, MEASURED_RIGHT_REST_Q, samples=11)


def test_exact_rest_to_home_is_clear_without_any_allowance(single_model) -> None:
    assert screen(single_model, REST_Q, HOME_Q, allow=0.0) == (0.0, None)


def test_return_to_rest_stays_zero_tolerance_from_a_contact_start(single_model) -> None:
    limits = np.vstack([np.tile([-3.2, 3.2], (6, 1)), [0.0, 0.044]])

    class Backend:
        def read_state(self):
            return HardwareState(
                MEASURED_RIGHT_REST_Q, 0.0, limits, "1.9.4", "1.9.3", np.full(7, 1e-4)
            )

        def move_to_rest(self, *_args, **_kwargs):
            raise AssertionError("a rejected return path must not be commanded")

    config = {
        "hardware": {
            "rest_q_rad": [0.0] * 6,
            "rest_gripper_m": 0.0,
            "shutdown_move_duration_s": 2.0,
            "startup_collision_samples": 501,
        }
    }
    with pytest.raises(HardwareSafetyError, match=r"rejects the return-to-rest path near 0\.0%"):
        return_to_rest(Backend(), single_model, config, control_gripper=True)
