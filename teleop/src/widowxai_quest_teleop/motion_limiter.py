from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def bounded_command_period(elapsed_s: float, loop_rate_hz: float) -> float:
    """Use actual timing for early frames but never integrate beyond one nominal tick."""
    elapsed = float(elapsed_s)
    rate = float(loop_rate_hz)
    if not np.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("elapsed command time must be finite and positive")
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("loop rate must be finite and positive")
    return max(1e-6, min(elapsed, 1.0 / rate))


def configured_minimum_command_interval(control: dict) -> float:
    """Return an optional latest-state command spacing guard."""

    interval = float(control.get("minimum_command_interval_s", 0.0))
    if not np.isfinite(interval) or interval < 0.0:
        raise ValueError("minimum command interval must be finite and nonnegative")
    return interval


def configured_command_spacing_stage(control: dict) -> str:
    """Return where a configured command interval is enforced."""

    stage = str(control.get("command_spacing_stage", "before_consume"))
    if stage not in ("before_consume", "before_send"):
        raise ValueError(
            "command spacing stage must be 'before_consume' or 'before_send'"
        )
    return stage


def minimum_command_spacing_wait(
    last_send_s: float,
    minimum_interval_s: float,
    now_s: float,
) -> float:
    """Delay until one spacing interval has elapsed; never repay missed time."""

    last_send = float(last_send_s)
    interval = float(minimum_interval_s)
    now = float(now_s)
    if not np.isfinite(last_send) or not np.isfinite(now):
        raise ValueError("command timestamps must be finite")
    if not np.isfinite(interval) or interval < 0.0:
        raise ValueError("minimum command interval must be finite and nonnegative")
    return max(0.0, last_send + interval - now)


def _jerk_safe_acceleration(
    candidate: float,
    velocity: float,
    max_velocity: float,
    max_jerk: float,
    period: float,
) -> float:
    """Bound acceleration early enough to respect velocity without a jerk impulse."""

    if candidate == 0.0:
        return 0.0
    direction = 1.0 if candidate > 0.0 else -1.0
    remaining_velocity = max(0.0, max_velocity - direction * velocity)
    # After this interval, the remaining acceleration still needs
    # ``a**2 / (2*j)`` velocity headroom to ramp continuously to zero. Solving
    # a*dt + a**2/(2*j) <= remaining_velocity gives the positive root below.
    jdt = max_jerk * period
    safe_magnitude = max(
        0.0,
        -jdt + np.sqrt(jdt * jdt + 2.0 * max_jerk * remaining_velocity),
    )
    return direction * min(abs(candidate), safe_magnitude)


@dataclass(frozen=True)
class MotionLimitResult:
    command: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    velocity_limited: np.ndarray
    acceleration_limited: np.ndarray
    jerk_limited: np.ndarray

    def flags(self, prefix: str) -> list[str]:
        result: list[str] = []
        if np.any(self.velocity_limited):
            result.append(f"{prefix}_velocity")
        if np.any(self.acceleration_limited):
            result.append(f"{prefix}_acceleration")
        if np.any(self.jerk_limited):
            result.append(f"{prefix}_jerk")
        return result


class AccelerationLimitedCommand:
    """Track position targets with explicit velocity and acceleration bounds.

    This is a hardware-facing safety adapter. It is intentionally separate from
    the article-faithful IK step cap so simulation can preview the exact command
    stream that will later be offered to the physical driver.
    """

    def __init__(
        self,
        initial_position: np.ndarray,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        max_jerk: np.ndarray | None = None,
    ) -> None:
        initial = np.asarray(initial_position, dtype=float).reshape(-1)
        velocity = np.asarray(max_velocity, dtype=float).reshape(initial.shape)
        acceleration = np.asarray(max_acceleration, dtype=float).reshape(initial.shape)
        if not np.all(np.isfinite(initial)):
            raise ValueError("initial command must be finite")
        if not np.all(np.isfinite(velocity)) or np.any(velocity <= 0.0):
            raise ValueError("maximum velocities must be finite and positive")
        if not np.all(np.isfinite(acceleration)) or np.any(acceleration <= 0.0):
            raise ValueError("maximum accelerations must be finite and positive")
        jerk = None if max_jerk is None else np.asarray(max_jerk, dtype=float).reshape(initial.shape)
        if jerk is not None and (not np.all(np.isfinite(jerk)) or np.any(jerk <= 0.0)):
            raise ValueError("maximum jerks must be finite and positive")
        self.max_velocity = velocity.copy()
        self.max_acceleration = acceleration.copy()
        self.max_jerk = None if jerk is None else jerk.copy()
        self.position = initial.copy()
        self.velocity = np.zeros_like(initial)
        self.acceleration = np.zeros_like(initial)

    def reset(self, position: np.ndarray) -> None:
        value = np.asarray(position, dtype=float).reshape(self.position.shape)
        if not np.all(np.isfinite(value)):
            raise ValueError("reset command must be finite")
        self.position = value.copy()
        self.velocity.fill(0.0)
        self.acceleration.fill(0.0)

    def step(self, target: np.ndarray, dt: float) -> MotionLimitResult:
        goal = np.asarray(target, dtype=float).reshape(self.position.shape)
        period = float(dt)
        if not np.all(np.isfinite(goal)):
            raise ValueError("target command must be finite")
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("command period must be finite and positive")

        error = goal - self.position
        # Leave enough distance to brake after this tick. This discrete form is
        # less prone to one-tick overshoot than sqrt(2*a*distance) alone.
        acceleration_step = self.max_acceleration * period
        braking_speed = np.maximum(
            0.0,
            np.sqrt(acceleration_step * acceleration_step + 2.0 * self.max_acceleration * np.abs(error))
            - acceleration_step,
        )
        requested_velocity = np.sign(error) * np.minimum(self.max_velocity, braking_speed)
        unconstrained_velocity = np.divide(error, period)
        velocity_limited = np.abs(unconstrained_velocity) > self.max_velocity + 1e-12

        previous_velocity = self.velocity.copy()
        previous_acceleration = self.acceleration.copy()
        velocity_delta = requested_velocity - previous_velocity
        acceleration_limited = np.abs(velocity_delta) > acceleration_step + 1e-12
        requested_acceleration = np.clip(
            np.divide(velocity_delta, period),
            -self.max_acceleration,
            self.max_acceleration,
        )
        if self.max_jerk is None:
            next_acceleration = requested_acceleration
            jerk_limited = np.zeros_like(next_acceleration, dtype=bool)
        else:
            acceleration_delta = requested_acceleration - previous_acceleration
            jerk_step = self.max_jerk * period
            jerk_limited = np.abs(acceleration_delta) > jerk_step + 1e-12
            next_acceleration = previous_acceleration + np.clip(
                acceleration_delta,
                -jerk_step,
                jerk_step,
            )
            next_acceleration = np.clip(
                next_acceleration,
                -self.max_acceleration,
                self.max_acceleration,
            )
            for index in range(next_acceleration.size):
                next_acceleration[index] = _jerk_safe_acceleration(
                    next_acceleration[index],
                    previous_velocity[index],
                    self.max_velocity[index],
                    self.max_jerk[index],
                    period,
                )
        next_velocity = previous_velocity + next_acceleration * period
        next_velocity = np.clip(next_velocity, -self.max_velocity, self.max_velocity)
        # Record the acceleration actually represented by the bounded velocity.
        next_acceleration = (next_velocity - previous_velocity) / period
        next_position = self.position + next_velocity * period

        # The braking equation converges to the target through very small
        # floating-point residues. Canonicalize only machine-precision values
        # so a physical boundary target such as gripper 0.0 cannot become
        # -0.00000000000000000001 on the following tick. This is not a range
        # clamp: meaningful overshoot remains visible to the command gate.
        scale = np.maximum.reduce(
            [
                np.ones_like(goal),
                np.abs(goal),
                np.abs(self.position),
                np.abs(next_position),
            ]
        )
        at_target = np.abs(next_position - goal) <= 64.0 * np.finfo(float).eps * scale
        next_position = np.where(at_target, goal, next_position)
        next_velocity = np.where(at_target, 0.0, next_velocity)
        next_acceleration = np.where(at_target, 0.0, next_acceleration)

        self.position = next_position
        self.velocity = next_velocity
        self.acceleration = next_acceleration
        return MotionLimitResult(
            command=next_position.copy(),
            velocity=next_velocity.copy(),
            acceleration=next_acceleration.copy(),
            velocity_limited=velocity_limited,
            acceleration_limited=acceleration_limited,
            jerk_limited=jerk_limited,
        )


class VelocityFeedforwardFilter:
    """Estimate a bounded, smoothed endpoint velocity from position commands."""

    def __init__(
        self,
        initial_position: np.ndarray,
        *,
        filter_alpha: float,
        gain: float,
        max_velocity: np.ndarray,
    ) -> None:
        initial = np.asarray(initial_position, dtype=float).reshape(-1)
        velocity = np.asarray(max_velocity, dtype=float).reshape(initial.shape)
        alpha = float(filter_alpha)
        feedforward_gain = float(gain)
        if not np.all(np.isfinite(initial)):
            raise ValueError("feedforward initial position must be finite")
        if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("feedforward filter alpha must be within (0, 1]")
        if not np.isfinite(feedforward_gain) or not 0.0 <= feedforward_gain <= 1.0:
            raise ValueError("feedforward gain must be within [0, 1]")
        if not np.all(np.isfinite(velocity)) or np.any(velocity <= 0.0):
            raise ValueError("feedforward velocity caps must be finite and positive")
        self.filter_alpha = alpha
        self.gain = feedforward_gain
        self.max_velocity = velocity.copy()
        self.previous_position = initial.copy()
        self.filtered_velocity = np.zeros_like(initial)

    def reset(self, position: np.ndarray) -> None:
        value = np.asarray(position, dtype=float).reshape(self.previous_position.shape)
        if not np.all(np.isfinite(value)):
            raise ValueError("feedforward reset position must be finite")
        self.previous_position = value.copy()
        self.filtered_velocity.fill(0.0)

    def update(self, position: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(position, dtype=float).reshape(self.previous_position.shape)
        period = float(dt)
        if not np.all(np.isfinite(value)):
            raise ValueError("feedforward position must be finite")
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("feedforward period must be finite and positive")
        raw_velocity = (value - self.previous_position) / period
        # Do not carry an old velocity through a direction reversal. A stale
        # opposite-sign endpoint velocity would be worse than no feedforward
        # while the driver replans the new segment.
        reversing = raw_velocity * self.filtered_velocity < 0.0
        self.filtered_velocity[reversing] = 0.0
        self.filtered_velocity = (
            (1.0 - self.filter_alpha) * self.filtered_velocity
            + self.filter_alpha * raw_velocity
        )
        self.previous_position = value.copy()
        return np.clip(
            self.gain * self.filtered_velocity,
            -self.max_velocity,
            self.max_velocity,
        )


def limiter_from_config(
    section: dict | None,
    initial_position: np.ndarray,
) -> AccelerationLimitedCommand | None:
    if not section or not section.get("enabled", False):
        return None
    return AccelerationLimitedCommand(
        initial_position,
        np.asarray(section["max_velocity"], dtype=float),
        np.asarray(section["max_acceleration"], dtype=float),
        None
        if section.get("max_jerk") is None
        else np.asarray(section["max_jerk"], dtype=float),
    )
