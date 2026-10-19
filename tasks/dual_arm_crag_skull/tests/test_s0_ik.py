import numpy as np
import pytest

from core.robots.kinematics import T
from tasks.dual_arm_crag_skull.s0.ik import IKOptions, solve_ik
from tasks.dual_arm_crag_skull.s0.kinematics import robotiq_proxy_model

MODEL = robotiq_proxy_model()
LEFT = MODEL.arms['left']
HOME = np.array([0.42959, -0.22024, -0.68657, -0.00021, -1.10447, -0.42928])


@pytest.mark.parametrize('seed', range(8))
def test_round_trip_from_a_nearby_seed(seed):
    rng = np.random.default_rng(seed)
    q_true = np.clip(HOME + rng.uniform(-.6, .6, 6), LEFT.lower + .1, LEFT.upper - .1)
    start = q_true + rng.uniform(-.1, .1, 6)
    result = solve_ik(LEFT, LEFT.tcp(q_true), start)
    assert result.ok, result
    assert result.position_error_m <= 1e-4 and result.rotation_error_rad <= 1e-3
    np.testing.assert_allclose(LEFT.tcp(result.q), LEFT.tcp(q_true), atol=2e-4)


def test_unreachable_target_is_unconverged_not_clipped():
    target = T([-.78 + 3.0, 0, .9], [1, 0, 0, 0])
    result = solve_ik(LEFT, target, HOME)
    assert not result.ok and result.q is None
    assert result.reason in ('unconverged', 'joint_limit') and result.position_error_m > 1.0


def test_joint_limit_failure_reports_the_limit():
    q_true = HOME.copy()
    q_true[4] = LEFT.upper[4] + .2  # pose only reachable with joint 5 beyond its limit near this seed
    seed = HOME.copy()
    seed[4] = LEFT.upper[4] - .05
    result = solve_ik(LEFT, LEFT.tcp(q_true), seed, IKOptions(max_seed_jump_rad=.6, max_iterations=60))
    assert not result.ok and result.q is None
    assert result.reason == 'joint_limit', result


def test_large_branch_change_is_rejected_as_seed_jump():
    q_target = HOME + np.array([.0, .0, .0, .0, .0, .35])
    result = solve_ik(LEFT, LEFT.tcp(q_target), HOME, IKOptions(max_seed_jump_rad=.1))
    assert not result.ok and result.reason == 'seed_jump'
    assert result.position_error_m <= 1e-4  # it did converge; the jump is what is refused


def test_seed_outside_limits_is_refused():
    seed = HOME.copy()
    seed[0] = LEFT.upper[0] + .01
    assert solve_ik(LEFT, LEFT.tcp(HOME), seed).reason == 'seed_outside_limits'
