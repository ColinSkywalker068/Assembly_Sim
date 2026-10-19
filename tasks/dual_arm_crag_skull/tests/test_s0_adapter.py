import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tasks.dual_arm_crag_skull.contracts.actions import ARMS, decode_chunk, encode_chunk
from tasks.dual_arm_crag_skull.s0 import adapter as adapter_module
from tasks.dual_arm_crag_skull.s0.adapter import S0Adapter, S0Limits, braking_profile
from tasks.dual_arm_crag_skull.s0.ik import IKOptions, solve_ik
from tasks.dual_arm_crag_skull.s0.kinematics import ROBOTIQ_PROXY_MAX_WIDTH_M, load_kinematics, robotiq_proxy_model

MODEL = robotiq_proxy_model()
HOME = np.array([0.42959, -0.22024, -0.68657, -0.00021, -1.10447, -0.42928])
TEST_ACCELERATION = np.full(6, 20.0)  # test-only value: the real CRX limit is unknown (see kinematics JSON)
STEP_DOWN = np.array([[0, 0, -.01], [0, 0, -.02], [0, 0, -.03], [0, 0, -.04]])  # 0.1 m/s at 10 Hz


def limits(**overrides):
    args = dict(max_joint_velocity_rad_s=load_kinematics()['max_joint_velocity_rad_s'],
                max_joint_acceleration_rad_s2=TEST_ACCELERATION,
                gripper_width_range_m={arm: (0.0, ROBOTIQ_PROXY_MAX_WIDTH_M) for arm in ARMS},
                arm_arm_clearance_m=0.02, table_top_z_m=0.75, table_clearance_m=0.01, tcp_min_height_m=0.02,
                servo_hz=60, max_observation_age_s=0.5, anchor_position_tolerance_m=0.002,
                anchor_rotation_tolerance_rad=0.005)
    args.update(overrides)
    return S0Limits(**args)


def joints_at(arm, position, seed=HOME):
    target = MODEL.arms[arm].tcp(HOME)
    target[:3, 3] = position
    result = solve_ik(MODEL.arms[arm], target, seed, IKOptions(max_seed_jump_rad=3.0))
    assert result.ok, result
    return result.q


def chunk_from(anchors, targets, widths, *, active=('left',), times=None, valid_until=1.6):
    times = 1.1 + .1 * np.arange(len(targets['left'])) if times is None else np.asarray(times)
    return encode_chunk(request_id='req', chunk_id=3, observation_time=1.0, valid_until=valid_until, anchors=anchors,
                        waypoint_times=times, targets=targets, widths=widths, active_arms=active)


def make_chunk(q_anchor, widths_now, *, moves=STEP_DOWN, widths=None, active=('left',), **kwargs):
    """Active arms translate by `moves` (N, 3) from their anchors; inactive arms hold."""
    moves = np.asarray(moves, dtype=float)
    anchors = {arm: MODEL.arms[arm].tcp(q_anchor[arm]) for arm in ARMS}
    targets = {arm: np.repeat(anchors[arm][None], len(moves), axis=0) for arm in ARMS}
    for arm in active:
        targets[arm][:, :3, 3] += moves
    chunk_widths = {arm: np.full(len(moves), widths_now[arm]) for arm in ARMS}
    chunk_widths.update(widths or {})
    return chunk_from(anchors, targets, chunk_widths, active=active, **kwargs)


def state(left=HOME, right=HOME):
    return {'left': np.array(left), 'right': np.array(right)}, {'left': .05, 'right': .03}


def plan(chunk, q_now, widths_now, now=1.0, velocities=None, ik_options=IKOptions(), tcp_at_observation=None,
         adapter=None, commanded=None, **limit_overrides):
    adapter = adapter or S0Adapter(MODEL, limits(**limit_overrides), ik_options)
    velocities = velocities or {arm: np.zeros(6) for arm in ARMS}
    # By default S1 anchored the chunk on exactly the TCP the robot had at observation time, and the
    # grippers sit where they were last commanded (measured == commanded).
    tcp_at_observation = tcp_at_observation or {arm: np.array(chunk.anchors[arm]) for arm in ARMS}
    commanded = widths_now if commanded is None else commanded
    return adapter.plan(chunk, now=now, joints=q_now, joint_velocities=velocities, widths=widths_now,
                        commanded_widths=commanded, tcp_at_observation=tcp_at_observation)


def test_feasible_chunk_becomes_servo_rate_trajectory():
    q_now, widths_now = state()
    chunk = make_chunk(q_now, widths_now, widths={'left': np.array([.05, .045, .04, .035])})
    status, trajectory = plan(chunk, q_now, widths_now)
    assert status.status == 'accepted', status.reason
    assert not trajectory.hold
    steps = np.diff(trajectory.times)
    np.testing.assert_allclose(steps, 1 / 60, atol=1e-9)
    assert trajectory.times[0] == pytest.approx(1 + 1 / 60) and trajectory.times[-1] == pytest.approx(1.4)
    for k, time in enumerate(chunk.waypoint_times):
        index = int(np.argmin(np.abs(trajectory.times - time)))
        reached = MODEL.arms['left'].tcp(trajectory.joints['left'][index])[:3, 3]
        np.testing.assert_allclose(reached, MODEL.arms['left'].tcp(HOME)[:3, 3] + STEP_DOWN[k], atol=2e-4)
    assert trajectory.widths['left'][-1] == pytest.approx(.035)
    np.testing.assert_allclose(trajectory.joints['right'], np.repeat(HOME[None], len(trajectory.times), axis=0))
    np.testing.assert_allclose(trajectory.widths['right'], .03)


def test_waypoints_already_in_the_past_are_dropped():
    # At now=1.25 the robot has executed half the chunk: 2.5 cm down, moving at 0.1 m/s.
    q_anchor, widths_now = state()
    chunk = make_chunk(q_anchor, widths_now)
    left = MODEL.arms['left']
    q_mid = joints_at('left', left.tcp(HOME)[:3, 3] + [0, 0, -.025], seed=HOME)
    velocity = np.linalg.solve(left.jacobian(q_mid), np.array([0, 0, -.1, 0, 0, 0]))
    q_now = {'left': q_mid, 'right': HOME}
    status, trajectory = plan(chunk, q_now, widths_now, now=1.25,
                              velocities={'left': velocity, 'right': np.zeros(6)})
    assert status.status == 'accepted', status.reason
    assert trajectory.times[0] > 1.25 and trajectory.times[-1] == pytest.approx(1.4)
    assert len(trajectory.times) == 9


def test_late_start_without_catching_up_is_rejected_not_retimed():
    # Same chunk, but the robot never moved: the remaining waypoints need a catch-up spike.
    q_now, widths_now = state()
    expect_rejection(make_chunk(q_now, widths_now), q_now, widths_now, 'acceleration exceeded', now=1.25)


def expect_rejection(chunk, q_now, widths_now, fragment, now=1.0, ik_options=IKOptions(), commanded=None,
                     **limit_overrides):
    status, hold = plan(chunk, q_now, widths_now, now=now, ik_options=ik_options, commanded=commanded,
                        **limit_overrides)
    commanded = widths_now if commanded is None else commanded
    assert status.status == 'rejected' and fragment in status.reason, status.reason
    assert hold.hold and hold.times.tolist() == [now]
    for arm in ARMS:
        np.testing.assert_array_equal(hold.joints[arm][0], q_now[arm])
        assert hold.widths[arm][0] == commanded[arm]  # never releases a held fragment
    assert dict(status.measured_width) == widths_now  # measured openings are reported, not commanded
    return status


def test_stale_chunk_is_rejected():
    q_now, widths_now = state()
    expect_rejection(make_chunk(q_now, widths_now, valid_until=1.2), q_now, widths_now, 'stale', now=1.21)


def test_inactive_arm_motion_is_rejected():
    q_now, widths_now = state()
    both = make_chunk(q_now, widths_now, active=('left', 'right'))
    decoded = decode_chunk(both)
    moving = chunk_from(both.anchors, {arm: decoded[arm][0] for arm in ARMS},
                        {arm: decoded[arm][1] for arm in ARMS}, active=('left',))
    expect_rejection(moving, q_now, widths_now, 'inactive arm moves: right')


def test_inactive_arm_width_change_is_rejected():
    q_now, widths_now = state()
    chunk = make_chunk(q_now, widths_now, widths={'right': np.full(4, .02)})
    expect_rejection(chunk, q_now, widths_now, 'inactive arm right')


def test_unreachable_target_is_rejected():
    q_now, widths_now = state()
    chunk = make_chunk(q_now, widths_now, moves=[[2.5, 0, 0]] * 4)
    expect_rejection(chunk, q_now, widths_now, 'left IK')


def test_joint_limit_is_rejected():
    left = HOME.copy()
    left[5] = MODEL.arms['left'].upper[5] - .003
    q_now, widths_now = state(left=left)
    anchors = {arm: MODEL.arms[arm].tcp(q_now[arm]) for arm in ARMS}
    targets = {arm: np.repeat(anchors[arm][None], 4, axis=0) for arm in ARMS}
    for k in range(4):
        beyond = left.copy()
        beyond[5] += .02 * (k + 1)  # only reachable by turning joint 6 past its limit
        targets['left'][k] = MODEL.arms['left'].tcp(beyond)
    chunk = chunk_from(anchors, targets, {'left': np.full(4, .05), 'right': np.full(4, .03)})
    expect_rejection(chunk, q_now, widths_now, 'IK joint_limit')


def test_velocity_limit_is_rejected():
    # 1.2 m/s TCP; the per-waypoint IK jump bound is relaxed so the velocity check is what decides.
    q_now, widths_now = state()
    expect_rejection(make_chunk(q_now, widths_now, moves=STEP_DOWN * 12), q_now, widths_now, 'velocity exceeded',
                     ik_options=IKOptions(max_seed_jump_rad=1.5))


def test_branch_flip_between_waypoints_is_rejected_by_ik():
    q_now, widths_now = state()
    expect_rejection(make_chunk(q_now, widths_now, moves=STEP_DOWN * 15), q_now, widths_now, 'IK seed_jump')


def test_acceleration_limit_is_rejected():
    q_now, widths_now = state()
    expect_rejection(make_chunk(q_now, widths_now), q_now, widths_now, 'acceleration exceeded',
                     max_joint_acceleration_rad_s2=np.full(6, .2))


def test_width_over_the_gripper_range_is_rejected():
    q_now, widths_now = state()
    chunk = make_chunk(q_now, widths_now, widths={'left': np.array([.05, .07, .09, .09])})
    expect_rejection(chunk, q_now, widths_now, 'width 0.0900 m outside')


def test_arm_arm_collision_is_rejected():
    q_now, widths_now = state(left=joints_at('left', [-.12, 0, 1.0]), right=joints_at('right', [.12, 0, 1.0]))
    status, _ = plan(make_chunk(q_now, widths_now, moves=np.zeros((4, 3))), q_now, widths_now,
                     arm_arm_clearance_m=.05)
    assert status.status == 'accepted', status.reason  # holding still is clear
    toward = [[.01, 0, 0], [.02, 0, 0], [.03, 0, 0], [.04, 0, 0]]
    status = expect_rejection(make_chunk(q_now, widths_now, moves=toward), q_now, widths_now, 'arm-arm clearance',
                              arm_arm_clearance_m=.05)
    assert 'left:tool' in status.reason and 'right:tool' in status.reason


def test_tcp_below_the_table_limit_is_rejected():
    # Pointing straight down 5 cm above the table: links and tool box are clear, only the TCP rule fires.
    q_now, widths_now = state(left=joints_at('left', [-.42, 0, .80]))
    assert MODEL.arms['left'].tool_box_corners(q_now['left'])[:, 2].min() > .75
    expect_rejection(make_chunk(q_now, widths_now, moves=np.zeros((4, 3))), q_now, widths_now, 'left TCP',
                     tcp_min_height_m=.10, table_clearance_m=0.0)


def test_arm_link_into_the_table_is_rejected():
    q_now, widths_now = state(left=joints_at('left', [-.42, 0, .80]))
    expect_rejection(make_chunk(q_now, widths_now), q_now, widths_now, 'from the table', table_clearance_m=.25)


def test_invalid_measured_state_fails_without_a_trajectory():
    q_now, widths_now = state()
    chunk = make_chunk(q_now, widths_now)
    status, trajectory = plan(chunk, {'left': np.full(6, np.nan), 'right': HOME}, widths_now)
    assert status.status == 'failed' and trajectory is None


def test_limits_require_explicit_values():
    with pytest.raises(ValueError):
        limits(max_joint_acceleration_rad_s2=np.zeros(6))
    with pytest.raises(ValueError):
        limits(gripper_width_range_m={'left': (0, .085)})
    with pytest.raises(ValueError):
        limits(servo_hz=60.0)


def tilted_low_pose():
    """TCP 3 cm above the table, approach tilted 20 deg and rolled 45 deg: a pad corner is in the table."""
    arm = MODEL.arms['left']
    rotation = (Rotation.from_euler('y', 20, degrees=True).as_matrix() @ arm.tcp(HOME)[:3, :3]
                @ Rotation.from_euler('x', 45, degrees=True).as_matrix())
    target = np.eye(4)
    target[:3, :3], target[:3, 3] = rotation, [-.42, 0, .78]
    result = solve_ik(arm, target, HOME, IKOptions(max_seed_jump_rad=3.0))
    assert result.ok, result
    return result.q


def test_tool_box_rejects_a_low_tilted_pose_the_old_rules_accepted():
    q_low = tilted_low_pose()
    arm = MODEL.arms['left']
    assert arm.tool_box_corners(q_low)[:, 2].min() < .75  # really below the table top
    assert arm.tcp(q_low)[2, 3] - .75 >= .02  # the TCP-height rule alone is satisfied
    q_now, widths_now = state(left=q_low)
    expect_rejection(make_chunk(q_now, widths_now, moves=np.zeros((4, 3))), q_now, widths_now, 'left:tool box')


def test_rejection_while_moving_brakes_within_the_acceleration_limit():
    q_now, widths_now = state()
    velocities = {'left': np.array([1.5, 0, 0, 0, 0, 0]), 'right': np.array([0, -.4, 0, 0, 0, 0])}
    chunk = make_chunk(q_now, widths_now, valid_until=1.2)
    status, brake = plan(chunk, q_now, widths_now, now=1.21, velocities=velocities)
    assert status.status == 'rejected' and 'stale' in status.reason and brake.hold
    # Each arm's fastest joint sets that arm's duration and all its joints stop then (arms are separate
    # controllers); the command lasts until the slower arm has stopped.
    assert len(brake.times) > 1 and brake.times[-1] == pytest.approx(1.21 + 1.5 / 20.0)
    for arm in ARMS:
        stop_time = np.abs(velocities[arm]).max() / 20.0
        t = np.clip(brake.times - 1.21, 0, stop_time)[:, None]
        deceleration = velocities[arm] / stop_time
        assert (np.abs(deceleration) <= TEST_ACCELERATION + 1e-9).all()
        expected = q_now[arm] + velocities[arm] * t - .5 * deceleration * t * t  # exact constant deceleration
        np.testing.assert_allclose(brake.joints[arm], expected, atol=1e-12)
        np.testing.assert_array_equal(brake.widths[arm], widths_now[arm])  # never releases
    np.testing.assert_allclose(brake.joints['left'][-1][0], HOME[0] + .5 * 1.5 * 1.5 / 20.0)
    assert brake.note is None


def test_braking_profile_stops_all_joints_together():
    q0, qd0, limit = np.zeros(3), np.array([1.0, -2.0, .5]), np.array([4.0, 4.0, 4.0])
    positions, duration, stop = braking_profile(q0, qd0, limit, np.linspace(0, 1, 101))
    assert duration == pytest.approx(.5)
    np.testing.assert_allclose(stop, .5 * qd0 * .5)
    np.testing.assert_allclose(positions[-1], stop)
    speed = np.gradient(positions, .01, axis=0)
    assert np.abs(speed[60:]).max() < 1e-9  # stopped after 0.5 s, every joint at once


def test_moving_inactive_arm_brakes_instead_of_jumping_to_a_hold():
    q_now, widths_now = state()
    velocities = {'left': np.zeros(6), 'right': np.array([1.0, 0, 0, 0, 0, 0])}
    status, trajectory = plan(make_chunk(q_now, widths_now), q_now, widths_now, velocities=velocities)
    assert status.status == 'accepted', status.reason
    right = trajectory.joints['right'][:, 0]
    first_step = right[0] - HOME[0]
    assert first_step == pytest.approx(1.0 / 60 - .5 * 20.0 * (1 / 60) ** 2 / 1.0, rel=1e-6)
    assert np.all(np.diff(right) >= -1e-12) and right[-1] == pytest.approx(HOME[0] + .5 * 1.0 / 20.0)


def test_chunk_that_cannot_brake_inside_the_joint_limits_is_rejected():
    # Joint 6 turns toward its upper limit at 0.5 rad/s and ends 0.04 rad short: with 2 rad/s^2 the stop
    # needs 0.0625 rad, so the chunk is refused even though every waypoint is inside the limits.
    arm = MODEL.arms['left']
    q0 = HOME.copy()
    q0[5] = arm.upper[5] - .04 - .2
    q_now, widths_now = state(left=q0)
    anchors = {name: MODEL.arms[name].tcp(q_now[name]) for name in ARMS}
    targets = {name: np.repeat(anchors[name][None], 4, axis=0) for name in ARMS}
    for k in range(4):
        q = q0.copy()
        q[5] += .05 * (k + 1)
        targets['left'][k] = arm.tcp(q)
    chunk = chunk_from(anchors, targets, {'left': np.full(4, .05), 'right': np.full(4, .03)})
    velocities = {'left': np.array([0, 0, 0, 0, 0, .5]), 'right': np.zeros(6)}
    status, brake = plan(chunk, q_now, widths_now, velocities=velocities, max_joint_acceleration_rad_s2=np.full(6, 2.0))
    assert status.status == 'rejected' and 'cannot brake to a stop inside the joint_6 limit' in status.reason
    assert brake.hold and 'braking left would pass the joint_6 limit' not in status.reason
    status, _ = plan(chunk, q_now, widths_now, velocities=velocities, max_joint_acceleration_rad_s2=np.full(6, 8.0))
    assert status.status == 'accepted', status.reason


def test_anchor_must_match_the_tcp_measured_at_observation():
    q_now, widths_now = state()
    chunk = make_chunk(q_now, widths_now, moves=np.zeros((4, 3)))
    shifted = {arm: np.array(chunk.anchors[arm]) for arm in ARMS}
    shifted['left'][0, 3] += .01
    status, brake = plan(chunk, q_now, widths_now, tcp_at_observation=shifted)
    assert status.status == 'rejected' and 'left anchor is 10.0 mm' in status.reason and brake.hold
    broken = {**shifted, 'left': np.diag([1., 1., 1., 2.])}
    status, _ = plan(chunk, q_now, widths_now, tcp_at_observation=broken)
    assert status.status == 'rejected' and 'tcp_at_observation is not a valid pose' in status.reason


def test_replayed_and_out_of_order_chunks_are_rejected():
    q_now, widths_now = state()
    adapter = S0Adapter(MODEL, limits())
    first = make_chunk(q_now, widths_now, moves=np.zeros((4, 3)))
    assert plan(first, q_now, widths_now, adapter=adapter)[0].status == 'accepted'
    status, _ = plan(first, q_now, widths_now, adapter=adapter)
    assert status.status == 'rejected' and 'is not after the last accepted chunk 3' in status.reason
    older = encode_chunk(request_id='req', chunk_id=2, observation_time=1.0, valid_until=1.6, anchors=first.anchors,
                         waypoint_times=first.waypoint_times, targets={a: decode_chunk(first)[a][0] for a in ARMS},
                         widths={a: decode_chunk(first)[a][1] for a in ARMS}, active_arms=('left',))
    assert plan(older, q_now, widths_now, adapter=adapter)[0].status == 'rejected'
    newer = encode_chunk(request_id='req', chunk_id=4, observation_time=1.0, valid_until=1.6, anchors=first.anchors,
                         waypoint_times=first.waypoint_times, targets={a: decode_chunk(first)[a][0] for a in ARMS},
                         widths={a: decode_chunk(first)[a][1] for a in ARMS}, active_arms=('left',))
    assert plan(newer, q_now, widths_now, adapter=adapter)[0].status == 'accepted'


def test_s0_limits_observation_age_even_when_s1_claims_a_long_validity():
    q_now, widths_now = state()
    chunk = make_chunk(q_now, widths_now, moves=np.zeros((4, 3)), times=1.6 + .1 * np.arange(4), valid_until=1e9)
    status, _ = plan(chunk, q_now, widths_now, now=1.55)
    assert status.status == 'rejected' and 'observation is 0.550 s old' in status.reason


def test_contract_errors_while_planning_become_rejections(monkeypatch):
    q_now, widths_now = state()
    chunk = make_chunk(q_now, widths_now)

    def broken_decode(_):
        raise ValueError('decode failed for a test')
    monkeypatch.setattr(adapter_module, 'decode_chunk', broken_decode)
    status, brake = plan(chunk, q_now, widths_now)
    assert status.status == 'rejected' and 'decode failed for a test' in status.reason and brake.hold


def test_nonfinite_clock_fails_without_a_trajectory():
    q_now, widths_now = state()
    status, trajectory = plan(make_chunk(q_now, widths_now), q_now, widths_now, now=float('nan'))
    assert status.status == 'failed' and trajectory is None


def test_adapter_refuses_a_model_without_tool_geometry():
    data = load_kinematics()
    from tasks.dual_arm_crag_skull.s0.kinematics import DualArmModel, ROBOTIQ_PROXY_FLANGE_T_TCP
    bare = DualArmModel({arm: ROBOTIQ_PROXY_FLANGE_T_TCP for arm in ARMS}, {arm: None for arm in ARMS}, data=data,
                        tool_box={'left': data['robotiq_proxy_tool_box'], 'right': None})
    with pytest.raises(ValueError, match='missing'):
        S0Adapter(bare, limits())


# Support arm (right) was commanded to 0.020 m and is blocked by a fragment at 0.030 m. Holding the
# measured 0.030 m would remove the squeeze, so every hold must keep the commanded 0.020 m.
BLOCKED_MEASURED = {'left': .05, 'right': .030}
BLOCKED_COMMANDED = {'left': .05, 'right': .020}


def test_rejection_holds_the_commanded_width_of_a_blocked_gripper():
    q_now, _ = state()
    chunk = make_chunk(q_now, BLOCKED_COMMANDED, valid_until=1.2)
    status = expect_rejection(chunk, q_now, BLOCKED_MEASURED, 'stale', now=1.21, commanded=BLOCKED_COMMANDED)
    assert status.measured_width['right'] == .030


def test_braking_while_moving_keeps_the_commanded_width():
    q_now, _ = state()
    velocities = {'left': np.zeros(6), 'right': np.array([0, -.4, 0, 0, 0, 0])}
    chunk = make_chunk(q_now, BLOCKED_COMMANDED, valid_until=1.2)
    status, brake = plan(chunk, q_now, BLOCKED_MEASURED, now=1.21, velocities=velocities, commanded=BLOCKED_COMMANDED)
    assert status.status == 'rejected' and brake.hold and len(brake.times) > 1
    np.testing.assert_array_equal(brake.widths['right'], .020)
    np.testing.assert_array_equal(brake.widths['left'], .05)


def test_inactive_arm_is_judged_against_the_commanded_width():
    q_now, _ = state()
    holds_command = make_chunk(q_now, BLOCKED_COMMANDED)
    status, trajectory = plan(holds_command, q_now, BLOCKED_MEASURED, commanded=BLOCKED_COMMANDED)
    assert status.status == 'accepted', status.reason
    np.testing.assert_array_equal(trajectory.widths['right'], .020)
    holds_measured = make_chunk(q_now, BLOCKED_MEASURED)  # S1 copied the measured opening for the support arm
    expect_rejection(holds_measured, q_now, BLOCKED_MEASURED, 'inactive arm right: chunk width differs from the last '
                     'commanded width', commanded=BLOCKED_COMMANDED)


def test_active_width_motion_starts_from_the_commanded_width():
    q_now, _ = state()
    measured, commanded = {'left': .045, 'right': .03}, {'left': .05, 'right': .03}  # left closing onto a fragment
    chunk = make_chunk(q_now, commanded, widths={'left': np.array([.04, .035, .03, .025])})
    status, trajectory = plan(chunk, q_now, measured, commanded=commanded)
    assert status.status == 'accepted', status.reason
    expected = np.interp(trajectory.times, [1.0, *chunk.waypoint_times], [.05, .04, .035, .03, .025])
    np.testing.assert_allclose(trajectory.widths['left'], expected, atol=1e-15)


@pytest.mark.parametrize('commanded, fragment', [
    ({'left': .05, 'right': float('nan')}, 'right: commanded width nan'),
    ({'left': .05, 'right': ROBOTIQ_PROXY_MAX_WIDTH_M + .001}, 'not inside the gripper range'),
    ({'left': -.001, 'right': .03}, 'left: commanded width -0.001'),
    ({'left': .05}, 'right'),
])
def test_invalid_commanded_widths_fail_without_a_trajectory(commanded, fragment):
    q_now, widths_now = state()
    status, trajectory = plan(make_chunk(q_now, widths_now), q_now, widths_now, commanded=commanded)
    assert status.status == 'failed' and trajectory is None and fragment in status.reason, status.reason


def joint_path(steps=10, hz=60.0, step=np.array([0.0, 0.002, -0.002, 0.0, 0.002, 0.003])):
    times = 1.0 + np.arange(1, steps + 1) / hz
    path = HOME + step * np.arange(1, steps + 1)[:, None]
    return times, {'left': path, 'right': np.repeat(HOME[None], steps, axis=0)}


PERIOD = 1 / 60  # the backends report the velocity of the period just commanded (RobotSnapshot.velocity_window_s)


def check_path(times, joints, *, qd_now=None, qd_window_s=PERIOD, **limit_overrides):
    q_now = {arm: HOME.copy() for arm in ARMS}
    qd_now = qd_now or {arm: np.zeros(6) for arm in ARMS}
    adapter_module.check_joint_path(MODEL, limits(**limit_overrides), now=1.0, q_now=q_now, qd_now=qd_now,
                                    qd_window_s=qd_window_s, times=times, joints=joints)


def test_direct_joint_path_within_the_limits_is_accepted():
    times, joints = joint_path()
    check_path(times, joints, max_joint_acceleration_rad_s2=np.full(6, 200.0))


@pytest.mark.parametrize('mutate, overrides, fragment', [
    (lambda t, j: (t[::-1], j), {}, 'strictly increasing'),
    (lambda t, j: (t - 1.0, j), {}, 'after now'),
    (lambda t, j: (t, {**j, 'left': j['left'][:, :5]}), {}, 'finite joint targets'),
    (lambda t, j: (t, {**j, 'left': j['left'] + np.array([0, 0, 0, 0, 0, 7.0])}), {}, 'left joint limit'),
    (lambda t, j: (t, j), {'max_joint_velocity_rad_s': np.full(6, 0.1)}, 'left velocity exceeded on joint_6'),
    (lambda t, j: (t, j), {'max_joint_acceleration_rad_s2': np.full(6, 5.0)}, 'left acceleration exceeded'),
    (lambda t, j: (t, j), {'tcp_min_height_m': 0.5}, 'TCP'),
])
def test_direct_joint_path_violations_are_rejected(mutate, overrides, fragment):
    times, joints = mutate(*joint_path())
    overrides = {'max_joint_acceleration_rad_s2': np.full(6, 200.0), **overrides}
    with pytest.raises(adapter_module.Rejected, match=fragment):
        check_path(times, joints, **overrides)


def test_direct_joint_path_counts_acceleration_from_the_current_velocity():
    times, joints = joint_path()
    moving = {'left': np.array([0.0, 0.12, -0.12, 0.0, 0.12, 0.18]), 'right': np.zeros(6)}  # already at path speed
    check_path(times, joints, qd_now=moving, max_joint_acceleration_rad_s2=np.full(6, 5.0))


@pytest.mark.parametrize('now, q_left, qd_left, fragment', [
    (1.0, np.full(6, np.nan), np.zeros(6), 'measured joint state'),
    (1.0, HOME, np.full(6, np.nan), 'measured joint state'),
    (1.0, HOME[:5], np.zeros(6), 'measured joint state'),
    (float('nan'), HOME, np.zeros(6), 'clock'),
    (float('-inf'), HOME, np.zeros(6), 'clock'),
    (True, HOME, np.zeros(6), 'clock'),
])
def test_direct_joint_path_rejects_a_measured_state_or_clock_it_cannot_trust(now, q_left, qd_left, fragment):
    times, joints = joint_path()
    with pytest.raises(adapter_module.Rejected, match=fragment):
        adapter_module.check_joint_path(MODEL, limits(max_joint_acceleration_rad_s2=np.full(6, 200.0)), now=now,
                                        q_now={'left': q_left, 'right': HOME}, qd_now={'left': qd_left,
                                                                                       'right': np.zeros(6)},
                                        qd_window_s=PERIOD, times=times, joints=joints)
    with pytest.raises(adapter_module.Rejected, match='both arms'):
        adapter_module.check_joint_path(MODEL, limits(max_joint_acceleration_rad_s2=np.full(6, 200.0)), now=1.0,
                                        q_now={'left': HOME}, qd_now={'left': np.zeros(6)}, qd_window_s=PERIOD,
                                        times=times, joints=joints)


def test_direct_joint_path_measures_the_first_step_from_the_current_joints():
    times, joints = joint_path()
    jump = joints['left'] + np.array([0.5, 0, 0, 0, 0, 0])  # every step small, but the first one starts 0.5 rad away
    with pytest.raises(adapter_module.Rejected, match='left velocity exceeded on joint_1 at t=1.017'):
        check_path(times, {**joints, 'left': jump}, max_joint_acceleration_rad_s2=np.full(6, 1e6))


def test_direct_joint_path_checks_the_right_arm_too():
    times, joints = joint_path()
    with pytest.raises(adapter_module.Rejected, match='right velocity exceeded'):
        check_path(times, {'left': np.repeat(HOME[None], len(times), axis=0), 'right': joints['left']},
                   max_joint_velocity_rad_s=np.full(6, 0.1), max_joint_acceleration_rad_s2=np.full(6, 200.0))


@pytest.mark.parametrize('times', [np.array([1.1, 1.1, 1.2]), np.array([1.0, 1.1, 1.2])])
def test_direct_joint_path_needs_times_strictly_after_now_and_increasing(times):
    joints = {arm: np.repeat(HOME[None], 3, axis=0) for arm in ARMS}
    with pytest.raises(adapter_module.Rejected, match='strictly increasing and after now'):
        check_path(times, joints)


def single_joint_path(times, positions, *, q_now, qd_now):
    joints = {'left': np.repeat(HOME[None], len(times), axis=0), 'right': np.repeat(HOME[None], len(times), axis=0)}
    joints['left'][:, 0] = positions
    start = HOME.copy()
    start[0] = q_now
    return joints, {'left': start, 'right': HOME.copy()}, {'left': np.array([qd_now, 0, 0, 0, 0, 0]),
                                                          'right': np.zeros(6)}


@pytest.mark.parametrize('window, speeds', [(PERIOD, ((0.30, True), (0.36, False))),  # a period's average: dv/dt
                                             (0.0, ((0.15, True), (0.25, False)))])  # instantaneous: 2 dv/dt
def test_the_first_step_reads_the_measured_velocity_as_the_backend_declares_it(window, speeds):
    """From rest to `speed` in one 1/60 s step under 20 rad/s^2, then held: as the velocity of the period before
    (what every simulated backend reports) the step is speed / dt (0.30 -> 18, 0.36 -> 21.6); as an instantaneous
    velocity it needs constant acceleration over the step, 2 speed / dt (0.15 -> 18, 0.25 -> 30). The hold reads
    speed / dt either way (0.36 -> 21.6 rejects there too, which is why 0.30 is the one the window decides)."""
    dt = 1 / 60
    for speed, accepted in speeds:
        times = np.array([1.0 + dt, 1.0 + 2 * dt])
        joints, q_now, qd_now = single_joint_path(times, [HOME[0] + speed * dt, HOME[0] + speed * dt], q_now=HOME[0],
                                                  qd_now=0.0)
        call = lambda: adapter_module.check_joint_path(MODEL, limits(), now=1.0, q_now=q_now, qd_now=qd_now,
                                                       qd_window_s=window, times=times, joints=joints)
        if accepted:
            call()
        else:
            with pytest.raises(adapter_module.Rejected, match='left acceleration exceeded on joint_1'):
                call()


@pytest.mark.parametrize('window', [None, -PERIOD, float('nan'), float('inf'), True, '0.0167'])
def test_a_velocity_whose_meaning_is_not_declared_is_rejected(window):
    times, joints = joint_path()
    with pytest.raises(adapter_module.Rejected, match='velocity'):
        check_path(times, joints, qd_window_s=window, max_joint_acceleration_rad_s2=np.full(6, 200.0))


def test_a_motion_split_into_two_checks_reads_as_it_does_in_one():
    """A constant-acceleration ramp sampled at the servo rate, checked whole or as two paths (the second from the
    state the first ends in, its velocity the last period's): the same accelerations, none doubled at the split."""
    dt, accel, n = 1 / 60, 18.0, 6  # up to 1.65 rad/s, under joint 1's 2.09
    velocity = accel * (np.arange(n) + 0.5) * dt  # each period's velocity, as a sampled ramp from rest has it
    positions = HOME[0] + np.cumsum(velocity) * dt
    times = 1.0 + np.arange(1, n + 1) * dt
    for limit, accepted in ((18.0 + 1e-6, True), (17.9, False)):
        whole = lambda: adapter_module.check_joint_path(
            MODEL, limits(max_joint_acceleration_rad_s2=np.full(6, limit)), now=1.0,
            **_single_state(HOME[0], 0.0), qd_window_s=dt, times=times, joints=_single_joints(positions))
        split = 3
        second = lambda: adapter_module.check_joint_path(
            MODEL, limits(max_joint_acceleration_rad_s2=np.full(6, limit)), now=times[split - 1],
            **_single_state(positions[split - 1], velocity[split - 1]), qd_window_s=dt, times=times[split:],
            joints=_single_joints(positions[split:]))
        for call in (whole, second):
            if accepted:
                call()
            else:
                with pytest.raises(adapter_module.Rejected, match='left acceleration exceeded on joint_1'):
                    call()


def _single_state(q, qd):
    start = HOME.copy()
    start[0] = q
    return {'q_now': {'left': start, 'right': HOME.copy()},
            'qd_now': {'left': np.array([qd, 0, 0, 0, 0, 0]), 'right': np.zeros(6)}}


def _single_joints(positions):
    joints = {'left': np.repeat(HOME[None], len(positions), axis=0), 'right': np.repeat(HOME[None], len(positions), axis=0)}
    joints['left'][:, 0] = positions
    return joints


def test_uneven_steps_judge_acceleration_over_the_mean_spacing():
    """1 rad/s for 10 ms, then at rest for 100 ms: -1 / 0.055 = -18.2 rad/s^2, not -1 / 0.1 = -10."""
    times = np.array([1.01, 1.11])
    joints, q_now, qd_now = single_joint_path(times, [HOME[0] + 0.01, HOME[0] + 0.01], q_now=HOME[0], qd_now=1.0)
    adapter_module.check_joint_path(MODEL, limits(max_joint_acceleration_rad_s2=np.full(6, 19.0)), now=1.0,
                                    q_now=q_now, qd_now=qd_now, qd_window_s=PERIOD, times=times, joints=joints)
    with pytest.raises(adapter_module.Rejected, match='left acceleration exceeded on joint_1'):
        adapter_module.check_joint_path(MODEL, limits(max_joint_acceleration_rad_s2=np.full(6, 15.0)), now=1.0,
                                        q_now=q_now, qd_now=qd_now, qd_window_s=PERIOD, times=times, joints=joints)


def test_direct_joint_path_must_be_able_to_brake_inside_the_joint_limits():
    upper, dt, speed = MODEL.arms['left'].upper[0], 1 / 60, 0.9
    for margin, accepted in ((0.03, True), (0.01, False)):  # braking at 20 rad/s^2 from 0.9 rad/s takes 0.02 rad
        end = upper - margin
        joints, q_now, qd_now = single_joint_path(np.array([1.0 + dt]), [end], q_now=end - speed * dt, qd_now=speed)
        call = lambda: adapter_module.check_joint_path(MODEL, limits(), now=1.0, q_now=q_now, qd_now=qd_now,
                                                       qd_window_s=PERIOD, times=np.array([1.0 + dt]), joints=joints)
        if accepted:
            call()
        else:
            with pytest.raises(adapter_module.Rejected, match='cannot brake to a stop inside the joint_1 limit'):
                call()
