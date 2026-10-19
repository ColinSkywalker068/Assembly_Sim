import numpy as np
import pytest

from tasks.dual_arm_crag_skull.contracts.actions import ARMS
from tasks.dual_arm_crag_skull.s0.ik import pose_error
from tasks.dual_arm_crag_skull.s0.kinematics import robotiq_proxy_model
from tasks.dual_arm_crag_skull.s0.tracker import braking_velocity, joint_velocity_bounds, track

MODEL = robotiq_proxy_model()
HOME = np.array([0.42959, -0.22024, -0.68657, -0.00021, -1.10447, -0.42928])
ZERO = {arm: np.zeros(6) for arm in ARMS}
WIDE = np.full(6, 1e6)  # test-only acceleration bound that never binds
DT = 1 / 60


def state(left=HOME, right=HOME):
    return {'left': np.array(left), 'right': np.array(right)}


def test_unbounded_solution_is_damped_least_squares():
    twist = {'left': np.array([0, 0, -.05, 0, 0, 0]), 'right': np.array([.02, 0, 0, 0, 0, .1])}
    result = track(MODEL, state(), ZERO, twist, dt=DT, max_acceleration=WIDE, damping=1e-4)
    assert result.ok, result.reason
    for arm in ARMS:
        jacobian = MODEL.arms[arm].jacobian(HOME)
        expected = np.linalg.solve(jacobian.T @ jacobian + 1e-4 * np.eye(6), jacobian.T @ twist[arm])
        np.testing.assert_allclose(result.qdot[arm], expected, atol=1e-8)


def test_resting_arm_stays_at_rest():
    result = track(MODEL, state(), ZERO, {'left': np.array([0, 0, -.05, 0, 0, 0]), 'right': np.zeros(6)},
                   dt=DT, max_acceleration=WIDE)
    np.testing.assert_allclose(result.qdot['right'], 0, atol=1e-12)


def test_twist_beyond_the_velocity_box_is_reported_without_a_command():
    result = track(MODEL, state(), ZERO, {'left': np.array([0, 0, -3.0, 0, 0, 0]), 'right': np.zeros(6)},
                   dt=DT, max_acceleration=WIDE)
    assert not result.ok and result.reason.startswith('tracking_infeasible: left')
    assert result.qdot is None  # a slowed-down command is never handed out
    assert np.linalg.norm(result.twist_residual['left'][:3]) > 1.0


def test_acceleration_bound_limits_the_change_per_step():
    arm = MODEL.arms['left']
    low, high = joint_velocity_bounds(HOME, np.zeros(6), arm.lower, arm.upper, arm.max_velocity, np.full(6, 1.0), DT)
    np.testing.assert_allclose(low, -1.0 * DT)
    np.testing.assert_allclose(high, 1.0 * DT)
    twist = {'left': np.array([0, 0, -.2, 0, 0, 0]), 'right': np.zeros(6)}
    result = track(MODEL, state(), ZERO, twist, dt=DT, max_acceleration=np.full(6, 1.0))
    assert not result.ok and 'tracking_infeasible' in result.reason and result.qdot is None


def test_box_that_cannot_stop_before_the_limit_is_infeasible():
    arm = MODEL.arms['left']
    q = HOME.copy()
    q[0] = arm.upper[0]
    previous = {'left': np.array([1.0, 0, 0, 0, 0, 0]), 'right': np.zeros(6)}
    low, high = joint_velocity_bounds(q, previous['left'], arm.lower, arm.upper, arm.max_velocity, np.full(6, 6.0), DT)
    assert low[0] > high[0]
    result = track(MODEL, state(left=q), previous, ZERO, dt=DT, max_acceleration=np.full(6, 6.0))
    assert not result.ok and result.reason.startswith('infeasible_bounds: left joint_1')


def test_joint_at_its_lower_limit_cannot_be_driven_further_down():
    arm = MODEL.arms['left']
    q = HOME.copy()
    q[0] = arm.lower[0]
    low, _ = joint_velocity_bounds(q, np.zeros(6), arm.lower, arm.upper, arm.max_velocity, WIDE, DT)
    assert low[0] == 0.0
    twist = {'left': arm.jacobian(q)[:, 0] * -.5, 'right': np.zeros(6)}  # exactly q1dot = -0.5 rad/s
    result = track(MODEL, state(left=q), ZERO, twist, dt=DT, max_acceleration=WIDE)
    assert not result.ok and 'tracking_infeasible' in result.reason and result.qdot is None


def test_braking_velocity_formula():
    assert braking_velocity(0.0, 6.0, DT) == 0.0
    v = braking_velocity(.083, 6.0, DT)
    # Held velocity commands that drop by a*dt per step travel v^2/(2a) + v*dt/2 before stopping.
    assert v * v / (2 * 6.0) + v * DT / 2 == pytest.approx(.083)
    assert braking_velocity(-.01, 6.0, DT) == 0.0  # already past the limit: no motion toward it


def test_every_accepted_command_can_still_stop_before_the_limit():
    # joint_1 is driven toward its upper limit at 1 rad/s with 6 rad/s^2 available: the tracker must give up
    # (no command) while a stop is still possible, not keep going until the bounds become empty.
    arm = MODEL.arms['left']
    acceleration = np.full(6, 6.0)
    q = HOME.copy()
    q[0] = arm.upper[0] - .5
    qdot = {'left': np.array([1.0, 0, 0, 0, 0, 0]), 'right': np.zeros(6)}
    for step in range(200):
        twist = {'left': arm.jacobian(q) @ np.array([1.0, 0, 0, 0, 0, 0]), 'right': np.zeros(6)}
        result = track(MODEL, state(left=q), qdot, twist, dt=DT, max_acceleration=acceleration, damping=1e-8)
        if not result.ok:
            assert result.reason.startswith('tracking_infeasible') and result.qdot is None, result.reason
            break
        qdot = result.qdot
        q = q + qdot['left'] * DT
        remaining, v = arm.upper[0] - q[0], qdot['left'][0]
        # Discrete stoppability: the next commands v - a dt, v - 2 a dt, ... still fit before the limit.
        assert remaining >= v * v / (2 * acceleration[0]) - v * DT / 2 - 1e-9, (step, remaining, v)
    else:
        raise AssertionError('tracker never refused the twist')
    assert step > 10  # it did follow the twist while that was safe


def test_closed_loop_follows_a_straight_line():
    q = state()
    qdot = dict(ZERO)
    start = MODEL.arms['left'].tcp(HOME)
    for step in range(1, 61):
        target = start.copy()
        target[2, 3] -= .05 * step * DT  # 5 cm/s down for one second
        current = MODEL.arms['left'].tcp(q['left'])
        twist = {'left': np.array([0, 0, -.05, 0, 0, 0]) + pose_error(target, current) / DT * .5, 'right': np.zeros(6)}
        result = track(MODEL, q, qdot, twist, dt=DT, max_acceleration=np.full(6, 20.0), linear_tolerance_m_s=5e-3)
        assert result.ok, (step, result.reason)
        qdot = result.qdot
        q = {arm: q[arm] + qdot[arm] * DT for arm in ARMS}
    final = MODEL.arms['left'].tcp(q['left'])
    np.testing.assert_allclose(final[:3, 3], start[:3, 3] + [0, 0, -.05], atol=1e-3)
    np.testing.assert_allclose(q['right'], HOME)
