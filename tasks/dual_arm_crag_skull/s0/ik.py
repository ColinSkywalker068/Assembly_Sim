"""Damped-least-squares IK that fails loudly instead of returning a clipped or distant answer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..perception import rigid_transform


@dataclass(frozen=True)
class IKOptions:
    position_tolerance_m: float = 1e-4
    rotation_tolerance_rad: float = 1e-3
    max_iterations: int = 100
    damping: float = 0.05
    max_step_rad: float = 0.2
    limit_margin_rad: float = 1e-3
    max_seed_jump_rad: float = 0.5


@dataclass(frozen=True)
class IKResult:
    """`q` is set only on success; failures carry a reason from IK_FAILURES and the final errors."""

    ok: bool
    q: np.ndarray | None
    reason: str | None
    position_error_m: float
    rotation_error_rad: float
    iterations: int


IK_FAILURES = ('seed_outside_limits', 'unconverged', 'joint_limit', 'seed_jump')


def pose_error(target, current):
    """World-frame (dp, rotvec) taking `current` to `target`."""
    rotation = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
    return np.concatenate([target[:3, 3] - current[:3, 3], rotation])


def solve_ik(arm, target, seed, options=IKOptions()):
    """Solve arm.tcp(q) == target from `seed`.

    Iterates DLS steps projected into [lower + margin, upper - margin]; a `q` is returned only if
    it meets both tolerances inside that box. Otherwise the result is a failure (`joint_limit` when
    the last step was pushing a joint past the box), as it is when the converged solution is
    further than `max_seed_jump_rad` from the seed on any joint (a branch flip, not a small move).
    """
    target = rigid_transform(target)
    seed = np.asarray(seed, dtype=float)
    low, high = arm.lower + options.limit_margin_rad, arm.upper - options.limit_margin_rad
    if seed.shape != (arm.dof,) or not np.isfinite(seed).all() or (seed < low).any() or (seed > high).any():
        return IKResult(False, None, 'seed_outside_limits', np.inf, np.inf, 0)
    q = seed.copy()
    pinned = np.zeros(arm.dof, dtype=bool)
    weights = np.ones(6)
    for iteration in range(options.max_iterations + 1):
        error = pose_error(target, arm.tcp(q))
        position, rotation = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
        if position <= options.position_tolerance_m and rotation <= options.rotation_tolerance_rad:
            if np.abs(q - seed).max() > options.max_seed_jump_rad:
                return IKResult(False, None, 'seed_jump', position, rotation, iteration)
            return IKResult(True, q, None, position, rotation, iteration)
        if iteration == options.max_iterations:
            break
        jacobian = arm.jacobian(q) * weights[:, None]
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + options.damping ** 2 * np.eye(6), error * weights)
        norm = np.abs(step).max()
        if norm > options.max_step_rad:
            step *= options.max_step_rad / norm
        proposed = q + step
        pinned = (proposed < low) | (proposed > high)
        q = np.clip(proposed, low, high)
    reason = 'joint_limit' if pinned.any() else 'unconverged'
    return IKResult(False, None, reason, position, rotation, options.max_iterations)
