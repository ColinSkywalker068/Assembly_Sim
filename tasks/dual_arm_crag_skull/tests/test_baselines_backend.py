from dataclasses import replace

import numpy as np
import pytest

from tasks.dual_arm_crag_skull.baselines.backend import (AssemblyBackend, CameraFrame, KinematicBackend,
                                                         ObservedPose, RobotSnapshot)
from tasks.dual_arm_crag_skull.baselines.task import task_from_layout
from tasks.dual_arm_crag_skull.contracts.actions import ARMS
from tasks.dual_arm_crag_skull.s0.kinematics import ROBOTIQ_PROXY_MAX_WIDTH_M, robotiq_proxy_model
from tasks.dual_arm_crag_skull.tests.baselines_fixtures import HALF, write_layout

MODEL = robotiq_proxy_model()
HOME = np.array([0.42959, -0.22024, -0.68657, -0.00021, -1.10447, -0.42928])
NUDGE = np.array([0.0, 0.05, -0.05, 0.0, 0.05, 0.1])


def backend():
    return KinematicBackend(MODEL, home_joints={arm: HOME for arm in ARMS},
                            width_range_m={arm: (0.0, ROBOTIQ_PROXY_MAX_WIDTH_M) for arm in ARMS}, control_hz=20)


@pytest.fixture
def task(tmp_path):
    """frag_b's box centre sits exactly at the left TCP at home, so closing there grasps it."""
    base = task_from_layout(write_layout(tmp_path / 'layout'), anchor_id='frag_a')
    start = np.eye(4)
    start[:3, 3] = MODEL.arms['left'].tcp(HOME)[:3, 3] - [0, 0, HALF]
    fragments = tuple(replace(f, initial_world_T_fragment=start) if f.fragment_id == 'frag_b' else f
                      for f in base.fragments)
    return replace(base, fragments=fragments)


def targets(left=HOME, right=HOME):
    return {'left': np.array(left), 'right': np.array(right)}


def test_kinematic_backend_satisfies_the_protocol_and_needs_a_reset(task):
    sim = backend()
    assert isinstance(sim, AssemblyBackend)
    with pytest.raises(RuntimeError):
        sim.robot_state()
    state = sim.reset(task)
    assert state.time_s == 0.0 and state.widths == {arm: ROBOTIQ_PROXY_MAX_WIDTH_M for arm in ARMS}
    np.testing.assert_array_equal(state.joints['left'], HOME)
    poses = sim.privileged_fragment_poses()
    np.testing.assert_array_equal(poses['frag_b'], task.fragment('frag_b').initial_world_T_fragment)
    poses['frag_b'][0, 3] = 5.0
    assert sim.privileged_fragment_poses()['frag_b'][0, 3] != 5.0
    observed = sim.observed_fragment_poses()['frag_c']
    assert observed.source == 'oracle_sim' and not observed.world_T_fragment.flags.writeable
    assert sim.camera_frames()['agent'].synthetic


def test_step_tracks_targets_and_reports_velocity(task):
    sim = backend()
    sim.reset(task)
    state = sim.step(targets(HOME + NUDGE * 0.1), {arm: 0.05 for arm in ARMS})
    assert state.time_s == pytest.approx(0.05)
    np.testing.assert_allclose(state.joints['left'], HOME + NUDGE * 0.1)
    np.testing.assert_allclose(state.joint_velocities['left'], NUDGE * 0.1 * 20)
    np.testing.assert_allclose(state.tcp['left'], MODEL.arms['left'].tcp(HOME + NUDGE * 0.1))
    with pytest.raises(ValueError):
        state.joints['left'][0] = 1.0


def test_closing_at_a_fragment_carries_it_and_opening_releases_it(task):
    sim = backend()
    sim.reset(task)
    grasp_tcp = MODEL.arms['left'].tcp(HOME)
    sim.step(targets(), {'left': 0.03, 'right': ROBOTIQ_PROXY_MAX_WIDTH_M})
    assert sim.holding() == {'left': 'frag_b', 'right': None}
    start = task.fragment('frag_b').initial_world_T_fragment
    state = sim.step(targets(HOME + NUDGE), {'left': 0.03, 'right': ROBOTIQ_PROXY_MAX_WIDTH_M})
    expected = state.tcp['left'] @ np.linalg.inv(grasp_tcp) @ start
    np.testing.assert_allclose(sim.privileged_fragment_poses()['frag_b'], expected, atol=1e-12)
    sim.step(targets(HOME + NUDGE), {'left': 0.06, 'right': ROBOTIQ_PROXY_MAX_WIDTH_M})
    assert sim.holding()['left'] is None
    sim.step(targets(HOME), {'left': 0.06, 'right': ROBOTIQ_PROXY_MAX_WIDTH_M})
    np.testing.assert_allclose(sim.privileged_fragment_poses()['frag_b'], expected, atol=1e-12)


def test_the_anchor_is_held_by_a_fixture_and_never_grasped(task):
    """The anchor's box centre sits at the left TCP, every other fragment at its staging pose, out of reach: closing
    there takes nothing."""
    start = np.eye(4)
    start[:3, 3] = MODEL.arms['left'].tcp(HOME)[:3, 3] - [0, 0, HALF]
    staged = task_from_layout(task.layout, anchor_id='frag_a').fragments
    fixed = replace(task, fragments=tuple(replace(f, initial_world_T_fragment=start) if f.fragment_id == 'frag_a'
                                          else f for f in staged))
    sim = backend()
    sim.reset(fixed)
    sim.step(targets(), {'left': 0.0, 'right': ROBOTIQ_PROXY_MAX_WIDTH_M})
    sim.step(targets(HOME + NUDGE), {'left': 0.0, 'right': ROBOTIQ_PROXY_MAX_WIDTH_M})
    assert sim.holding() == {'left': None, 'right': None}
    np.testing.assert_array_equal(sim.privileged_fragment_poses()['frag_a'], start)


def test_closing_away_from_every_fragment_grasps_nothing(task):
    sim = backend()
    sim.reset(task)
    sim.step(targets(HOME + NUDGE), {arm: ROBOTIQ_PROXY_MAX_WIDTH_M for arm in ARMS})
    sim.step(targets(HOME + NUDGE), {'left': 0.0, 'right': 0.0})
    assert sim.holding() == {'left': None, 'right': None}


@pytest.mark.parametrize('joints, widths', [
    ({'left': HOME}, {arm: 0.05 for arm in ARMS}),
    (targets(HOME[:5]), {arm: 0.05 for arm in ARMS}),
    (targets(HOME + np.array([10.0, 0, 0, 0, 0, 0])), {arm: 0.05 for arm in ARMS}),
    (targets(), {'left': 0.2, 'right': 0.05}),
])
def test_invalid_targets_are_refused_without_moving(task, joints, widths):
    sim = backend()
    sim.reset(task)
    with pytest.raises(ValueError):
        sim.step(joints, widths)
    state = sim.robot_state()
    assert state.time_s == 0.0 and state.widths['left'] == ROBOTIQ_PROXY_MAX_WIDTH_M
    np.testing.assert_array_equal(state.joints['left'], HOME)


def test_camera_frames_carry_their_sample_time(task):
    sim = backend()
    sim.reset(task)
    assert sim.camera_frames()['agent'].time_s == 0.0
    state = sim.step(targets(HOME + NUDGE * 0.1), {arm: 0.05 for arm in ARMS})
    frame = sim.camera_frames()['agent']
    assert frame.time_s == state.time_s
    for bad in (float('nan'), float('inf'), True):
        with pytest.raises(ValueError, match='sample time'):
            replace(frame, time_s=bad)


def snapshot(**changes):
    args = dict(time_s=0.5, joints={arm: HOME for arm in ARMS}, joint_velocities={arm: np.zeros(6) for arm in ARMS},
                widths={arm: np.float32(0.05) for arm in ARMS}, tcp={arm: MODEL.arms[arm].tcp(HOME) for arm in ARMS},
                velocity_window_s=0.05)
    args.update(changes)
    return RobotSnapshot(**args)


def test_robot_snapshots_are_validated_and_normalized():
    state = snapshot()
    assert type(state.widths['left']) is float and state.time_s == 0.5
    bad_tcp = MODEL.arms['left'].tcp(HOME).copy()
    bad_tcp[:3, :3] *= 2.0
    for changes in ({'time_s': float('nan')}, {'time_s': True},
                    {'joints': {'left': np.full(6, np.nan), 'right': HOME}},
                    {'joints': {'left': np.eye(2), 'right': HOME}},
                    {'joint_velocities': {'left': np.zeros(5), 'right': np.zeros(6)}},
                    {'widths': {'left': float('nan'), 'right': 0.05}}, {'widths': {'left': -0.01, 'right': 0.05}},
                    {'tcp': {'left': bad_tcp, 'right': bad_tcp}}, {'joints': {'left': HOME}},
                    {'velocity_window_s': -0.01}, {'velocity_window_s': float('nan')}, {'velocity_window_s': True},
                    {'velocity_window_s': None}):
        with pytest.raises(ValueError):
            snapshot(**changes)


def test_the_kinematic_double_reports_each_period_s_velocity_and_says_so(task):
    """Its velocity is the last control period's (target - previous) * control_hz: a period's average, declared as
    such so S0 reads a path's first step as it reads every later one."""
    sim = backend()
    state = sim.reset(task)
    assert state.velocity_window_s == pytest.approx(1 / sim.control_hz) == sim.velocity_window_s
    state = sim.step(targets(HOME + NUDGE * 0.1), {arm: 0.05 for arm in ARMS})
    np.testing.assert_allclose(state.joint_velocities['left'], NUDGE * 0.1 * sim.control_hz)
    assert state.velocity_window_s == pytest.approx(1 / sim.control_hz)


def test_observed_poses_carry_their_time_and_source(task):
    sim = backend()
    sim.reset(task)
    sim.step(targets(HOME + NUDGE * 0.1), {arm: 0.05 for arm in ARMS})
    observed = sim.observed_fragment_poses()['frag_c']
    assert (observed.source, observed.time_s) == ('oracle_sim', pytest.approx(0.05))
    with pytest.raises(ValueError, match='time'):
        ObservedPose(np.eye(4), 'oracle_sim', float('nan'))
    with pytest.raises(ValueError, match='source'):
        ObservedPose(np.eye(4), 'guess', 0.0)


def test_camera_frames_need_rectified_intrinsics_and_a_rigid_pose(task):
    sim = backend()
    sim.reset(task)
    frame = sim.camera_frames()['agent']
    skewed = frame.intrinsics.copy()
    skewed[0, 1] = 0.5
    negative = frame.intrinsics.copy()
    negative[0, 0] = -8.0
    for intrinsics in (np.zeros((4, 4)), [[np.nan]], skewed, negative):
        with pytest.raises(ValueError, match='intrinsics'):
            replace(frame, intrinsics=intrinsics)
    scaled = frame.world_T_camera.copy()
    scaled[:3, :3] *= 1.5
    with pytest.raises(ValueError, match='rigid'):
        replace(frame, world_T_camera=scaled)


@pytest.mark.parametrize('changes', [{'control_hz': 0.0}, {'control_hz': float('nan')}, {'capture_radius_m': -1.0},
                                     {'width_range_m': {arm: (0.1, 0.05) for arm in ARMS}},
                                     {'home_joints': {arm: np.full(6, np.nan) for arm in ARMS}}])
def test_kinematic_backend_parameters_are_checked(changes):
    args = dict(home_joints={arm: HOME for arm in ARMS},
                width_range_m={arm: (0.0, ROBOTIQ_PROXY_MAX_WIDTH_M) for arm in ARMS}, control_hz=20)
    args.update(changes)
    with pytest.raises(ValueError):
        KinematicBackend(MODEL, **args)


def test_a_gripper_never_takes_the_fragment_the_other_one_holds(task):
    """frag_b sits midway between the TCPs, the nearest fragment to both; with a 10 m capture radius both grippers
    would take it, but only the first to close does."""
    left, right = (MODEL.arms[arm].tcp(HOME)[:3, 3] for arm in ARMS)
    start = np.eye(4)
    start[:3, 3] = (left + right) / 2 - [0, 0, HALF]
    shared = replace(task, fragments=tuple(replace(f, initial_world_T_fragment=start) if f.fragment_id == 'frag_b'
                                           else f for f in task.fragments))
    sim = KinematicBackend(MODEL, home_joints={arm: HOME for arm in ARMS},
                           width_range_m={arm: (0.0, ROBOTIQ_PROXY_MAX_WIDTH_M) for arm in ARMS}, control_hz=20,
                           capture_radius_m=10.0)
    sim.reset(shared)
    centres = {key: sim._box_centre(key) for key in shared.fragment_ids}
    assert min(centres, key=lambda key: np.linalg.norm(centres[key] - right)) == 'frag_b'
    sim.step(targets(), {'left': 0.0, 'right': ROBOTIQ_PROXY_MAX_WIDTH_M})
    sim.step(targets(), {'left': 0.0, 'right': 0.0})
    held = sim.holding()
    assert held['left'] == 'frag_b' and held['right'] not in (None, 'frag_b')


def test_a_reused_command_buffer_does_not_move_the_robot_afterwards(task):
    sim = backend()
    sim.reset(task)
    command = targets(HOME + NUDGE)
    sim.step(command, {arm: ROBOTIQ_PROXY_MAX_WIDTH_M for arm in ARMS})
    command['left'][0] += 0.5
    np.testing.assert_allclose(sim.robot_state().joints['left'], HOME + NUDGE)
