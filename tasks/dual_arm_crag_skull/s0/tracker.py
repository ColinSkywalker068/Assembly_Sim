"""Velocity-level dual-arm tracker: the README's 12-axis joint QP, without distance constraints.

    min_qdot ||W^(1/2) (J qdot - v_des)||^2 + lambda ||qdot||^2

subject to per-joint box bounds that fold together the velocity limits, the acceleration limits
relative to the previous command, and the braking distance to each joint limit: a joint may only
move toward a limit as fast as it can still stop before it at the maximum deceleration,
|qdot| <= -a dt/2 + sqrt((a dt/2)^2 + 2 a d) with d the remaining distance. J is block-diagonal
over the two arms. Arm-arm and environment distance constraints are NOT included: they need a
general QP with linearized distance rows. A desired twist the bounds cannot deliver is reported as
a failure with no command (qdot None), never silently slowed down.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import lsq_linear

from ..contracts.actions import ARMS


@dataclass(frozen=True)
class TrackerResult:
    ok: bool
    qdot: Mapping[str, np.ndarray] | None
    reason: str | None
    twist_residual: Mapping[str, np.ndarray]


def braking_velocity(distance, max_acceleration, dt):
    """Largest speed toward a limit `distance` away that still stops in time (discrete, one step of lag)."""
    distance = np.maximum(np.asarray(distance, dtype=float), 0.0)
    half = max_acceleration * dt / 2
    return -half + np.sqrt(half * half + 2 * max_acceleration * distance)


def joint_velocity_bounds(q, qdot_previous, lower, upper, max_velocity, max_acceleration, dt):
    # Within ~1 mm of a limit the braking bound alone exceeds d/dt, so keep the one-step bound too.
    toward_upper = np.minimum(braking_velocity(upper - q, max_acceleration, dt), (upper - q) / dt)
    toward_lower = np.minimum(braking_velocity(q - lower, max_acceleration, dt), (q - lower) / dt)
    low = np.maximum.reduce([-max_velocity, -toward_lower, qdot_previous - max_acceleration * dt])
    high = np.minimum.reduce([max_velocity, toward_upper, qdot_previous + max_acceleration * dt])
    return low, high


def track(model, q, qdot_previous, desired_twist, *, dt, max_acceleration, weights=np.ones(6), damping=1e-3,
          linear_tolerance_m_s=1e-3, angular_tolerance_rad_s=1e-2):
    """One servo step for both arms. `desired_twist[arm]` is (v, omega) of the TCP in the world frame."""
    blocks, targets, lows, highs = [], [], [], []
    for arm in ARMS:
        arm_model = model.arms[arm]
        low, high = joint_velocity_bounds(np.asarray(q[arm], float), np.asarray(qdot_previous[arm], float),
                                          arm_model.lower, arm_model.upper, arm_model.max_velocity,
                                          np.asarray(max_acceleration, float), dt)
        if (low > high).any():
            joint = int(np.argmax(low - high))
            empty = {a: np.zeros(6) for a in ARMS}
            return TrackerResult(False, None, f'infeasible_bounds: {arm} joint_{joint + 1} cannot satisfy position, '
                                              f'velocity and acceleration limits together', empty)
        blocks.append(arm_model.jacobian(q[arm]))
        targets.append(np.asarray(desired_twist[arm], dtype=float))
        lows.append(low)
        highs.append(high)
    jacobian, target = block_diag(*blocks), np.concatenate(targets)
    root_weights = np.sqrt(np.tile(np.asarray(weights, dtype=float), len(ARMS)))
    matrix = np.vstack([root_weights[:, None] * jacobian, np.sqrt(damping) * np.eye(jacobian.shape[1])])
    vector = np.concatenate([root_weights * target, np.zeros(jacobian.shape[1])])
    solution = lsq_linear(matrix, vector, bounds=(np.concatenate(lows), np.concatenate(highs)), method='bvls').x
    residual = jacobian @ solution - target
    qdot = {arm: solution[6 * i:6 * (i + 1)] for i, arm in enumerate(ARMS)}
    residuals = {arm: residual[6 * i:6 * (i + 1)] for i, arm in enumerate(ARMS)}
    for arm, value in residuals.items():
        if np.linalg.norm(value[:3]) > linear_tolerance_m_s or np.linalg.norm(value[3:]) > angular_tolerance_rad_s:
            return TrackerResult(False, None, f'tracking_infeasible: {arm} twist residual '
                                              f'{np.linalg.norm(value[:3]):.4f} m/s, {np.linalg.norm(value[3:]):.4f} rad/s',
                                 residuals)
    return TrackerResult(True, qdot, None, residuals)
