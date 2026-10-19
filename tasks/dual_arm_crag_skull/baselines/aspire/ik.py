"""ASPIRE IKProvider over S0's damped-least-squares IK for the CRX arms, registered as the "crx_dls" IK solver.

The seed is the current joint vector of the requested arm in the RobotState ASPIRE passes (CapApi substitutes the
caller's seed_joints there). A solution is returned only if s0.ik.solve_ik meets its tolerances inside the joint limits;
every other outcome is a typed failure, never a substituted or clipped pose. The joint path to a solution is checked
separately, sample by sample, by the adapter's S0 checks.
"""
from __future__ import annotations

import math

import numpy as np
from cap_harness.contracts import IKResult, Pose
from cap_harness.errors import ApiError, ErrorCode

from ...s0.ik import IKOptions, solve_ik
from .adapter import BASE_FRAME, EMBODIMENT

IK_SOLVER = 'crx_dls'
# A pose goal may be far from the current configuration; S0's default 0.5 rad seed jump is meant for chunk waypoints.
DEFAULT_OPTIONS = IKOptions(max_seed_jump_rad=math.pi)


def _finite_or_none(value):
    return float(value) if np.isfinite(value) else None


class CrxIKProvider:
    def __init__(self, model, options=DEFAULT_OPTIONS):
        self.model, self.options = model, options

    def _failure(self, arm, code, message, **details):
        return IKResult(ok=False, arm=arm, embodiment=EMBODIMENT, error=ApiError(code=code, message=message,
                                                                                 details=details))

    def solve_ik(self, target_pose, robot_state, *, arm='primary'):
        if arm not in self.model.arms:
            return self._failure(arm, ErrorCode.NOT_FOUND, f'unknown arm {arm!r}; arms are {tuple(self.model.arms)}')
        if not isinstance(target_pose, Pose) or target_pose.frame != BASE_FRAME:
            return self._failure(arm, ErrorCode.INVALID_REQUEST, f'target_pose must be a Pose in {BASE_FRAME!r}')
        if robot_state.embodiment != EMBODIMENT or arm not in robot_state.joint_positions:
            return self._failure(arm, ErrorCode.INVALID_REQUEST,
                                 f'robot_state must be a {EMBODIMENT} state with {arm!r}')
        result = solve_ik(self.model.arms[arm], target_pose.as_matrix(), robot_state.joint_positions[arm], self.options)
        diagnostics = {'provider': IK_SOLVER, 'iterations': result.iterations,
                       'position_error_m': _finite_or_none(result.position_error_m),
                       'rotation_error_rad': _finite_or_none(result.rotation_error_rad)}
        if result.ok:
            return IKResult(ok=True, joint_positions=result.q, arm=arm, diagnostics=diagnostics, embodiment=EMBODIMENT)
        code = ErrorCode.INVALID_REQUEST if result.reason == 'seed_outside_limits' else ErrorCode.IK_FAILED
        return IKResult(ok=False, arm=arm, embodiment=EMBODIMENT, diagnostics=diagnostics,
                        error=ApiError(code=code, message=f'CRX IK failed for {arm}: {result.reason}',
                                       details={'reason': result.reason, **diagnostics}))
