"""The scripted pick and place (sim/pick.py) on CPU: grasp choice, motion timing, S0 checks, and a whole run on
KinematicBackend with the YAM model."""
import math
from dataclasses import replace

import numpy as np
import pytest

from core.workcell.config import load_workcell_preset
from tasks.dual_arm_crag_skull.baselines.backend import KinematicBackend
from tasks.dual_arm_crag_skull.baselines.task import task_from_layout
from tasks.dual_arm_crag_skull.contracts.actions import ARMS
from tasks.dual_arm_crag_skull.s0.adapter import Rejected, S0Limits, check_joint_path
from tasks.dual_arm_crag_skull.s0.kinematics import load_kinematics, yam_model, yam_tool_geometry, yam_width_range_m
from tasks.dual_arm_crag_skull.sim import pick
from tasks.dual_arm_crag_skull.tests.baselines_fixtures import BOX_F, BOX_V, write_layout

MODEL = yam_model()
WORKCELL = load_workcell_preset('dual_arm')
HOME = {arm: np.array(WORKCELL.robot(arm).home_joint_positions, dtype=float) for arm in ARMS}
WIDTHS = {arm: yam_width_range_m() for arm in ARMS}
TABLE = 0.75
LIMITS = S0Limits(max_joint_velocity_rad_s=np.asarray(load_kinematics()['max_joint_velocity_rad_s']),
                  max_joint_acceleration_rad_s2=np.full(6, 20.0), gripper_width_range_m=WIDTHS,
                  arm_arm_clearance_m=0.02, table_top_z_m=TABLE, table_clearance_m=0.01, tcp_min_height_m=0.02,
                  servo_hz=60, max_observation_age_s=0.5, anchor_position_tolerance_m=0.002,
                  anchor_rotation_tolerance_rad=0.005, contact_table_clearance_m=0.002, contact_region_radius_m=0.12,
                  contact_max_speed_m_s=0.05, finger_opening_margin_m=0.004)  # the TEST values every method runs under
REACH = yam_tool_geometry()[2]['parts']['fingers']['max'][0] - MODEL.arms['left'].flange_T_tcp[0, 3]


def box_points(size, centre, yaw_deg):
    """The surface of a box standing on the table, turned by yaw, sampled as the grasp planner needs it."""
    corners = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (0.0, 1.0)]) * size
    yaw = math.radians(yaw_deg)
    turn = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return pick.surface_points(corners @ turn.T + [centre[0], centre[1], TABLE], BOX_F)


def test_grasp_closes_across_the_narrow_side_through_the_middle():
    """The pads meet a box's faces at its width (to the reach grid's resolution) with the TCP on its middle."""
    grasp = pick.choose_grasp('f', box_points((0.06, 0.03, 0.04), (0.3, -0.1), 30), TABLE, 0.094)
    assert math.degrees(grasp.yaw_rad) % 180 == pytest.approx(120)
    assert grasp.chord_m == pytest.approx(0.03, abs=1e-4)
    np.testing.assert_allclose(grasp.center_xy, (0.3, -0.1), atol=1e-4)
    assert grasp.open_width_m == pytest.approx(0.03 + 2 * pick.OPEN_MARGIN_M, abs=2e-4)
    assert grasp.clear_width_m == pytest.approx(0.03, abs=2e-4)
    assert grasp.close_width_m == pytest.approx(grasp.chord_m - pick.SQUEEZE_M)


def test_grasp_lines_can_be_offset_across_the_closing_axis():
    """Closing across a 60 x 30 mm box's narrow side, lines parallel to the one through the middle meet its faces at
    the same chord, their centres moved along its length by the offset; a line well past its end meets nothing."""
    points = box_points((0.06, 0.03, 0.04), (0.3, -0.1), 0)
    found = pick.grasp_candidates('f', points, TABLE, 0.094, yaws_deg=(90,), offsets_m=(0.0, 0.01, -0.02, 0.2))
    assert len(found) == 3
    across = np.array([math.sin(math.pi / 2), -math.cos(math.pi / 2)])
    for grasp, offset in zip(found, (0.0, 0.01, -0.02)):
        assert grasp.chord_m == pytest.approx(found[0].chord_m, abs=1e-4)
        assert (np.asarray(grasp.center_xy) - found[0].center_xy) @ across == pytest.approx(offset, abs=1e-6)


def test_grasp_lines_can_be_laid_through_a_given_point():
    """Through a point 10 mm along the 60 mm box from its middle (its centre of mass, say), the first line closes
    across the box there; the offsets are taken from that line."""
    points = box_points((0.06, 0.03, 0.04), (0.3, -0.1), 0)
    found = pick.grasp_candidates('f', points, TABLE, 0.094, yaws_deg=(90,), offsets_m=(0.0, 0.004),
                                  through_xy=(0.31, -0.1))
    assert found[0].center_xy[0] == pytest.approx(0.31, abs=1e-6)
    assert found[1].center_xy[0] == pytest.approx(0.314, abs=1e-6)
    assert found[0].chord_m == pytest.approx(0.03, abs=1e-4)


def test_a_wide_part_the_fingers_would_land_on_blocks_a_top_down_grasp():
    """A 120 mm roof 50 mm up over a 30 mm foot: the rails beside the window come down on it at any opening."""
    foot = box_points((0.03, 0.03, 0.03), (0.0, 0.0), 0)
    roof = box_points((0.12, 0.12, 0.01), (0.0, 0.0), 0) + [0, 0, 0.05]
    with pytest.raises(ValueError, match='no top-down grasp'):
        pick.choose_grasp('f', np.vstack([foot, roof]), TABLE, 0.094)


def test_a_bar_in_the_window_does_not_stop_the_pads_but_must_be_cleared_on_the_way_down():
    """A 3 mm bar along the closing axis, 33 mm up on the middle line, over a 30 mm foot: the closing pads pass it
    through the window and meet the foot, but the solid fingertips below the window would come down on it, so the
    open fingers clear its 50 mm length."""
    foot = box_points((0.03, 0.03, 0.03), (0.0, 0.0), 0)
    bar = box_points((0.05, 0.003, 0.003), (0.0, 0.0), 0) + [0, 0, 0.0345]
    grasp = pick.choose_grasp('f', np.vstack([foot, bar]), TABLE, 0.094, yaws_deg=(0,))
    assert grasp.chord_m == pytest.approx(0.03, abs=1e-4)
    assert grasp.open_width_m == pytest.approx(0.05 + 2 * pick.OPEN_MARGIN_M, abs=2e-4)
    assert grasp.clear_width_m == pytest.approx(0.05, abs=2e-4)  # what the fingers must open to to rise past it


def test_a_fragment_wider_than_the_opening_everywhere_is_refused():
    with pytest.raises(ValueError, match='no top-down grasp'):
        pick.choose_grasp('f', box_points((0.1, 0.1, 0.04), (0.0, 0.0), 0), TABLE, 0.094)


def test_surface_points_cover_every_triangle_densely():
    """Mesh vertices alone miss a box's long faces wherever no vertex lies (a voxel mesh has one every pitch)."""
    vertices = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.04, 0]], dtype=float)
    points = pick.surface_points(vertices, np.array([[0, 1, 2]]))
    assert len(points) >= 50 * 20 / 2
    inside = (points >= -1e-12).all(axis=1) & (points[:, 0] / 0.1 + points[:, 1] / 0.04 <= 1 + 1e-9)
    assert inside.all() and np.isclose(points[:, 2], 0).all()
    gaps = [np.min(np.linalg.norm(points[:, :2] - probe, axis=1)) for probe in ((0.05, 0.0), (0.03, 0.02), (0.0, 0.03))]
    assert max(gaps) <= pick.SURFACE_SPACING_M


def test_down_pose_points_the_approach_axis_at_the_table():
    pose = pick.down_pose((0.1, 0.2), 0.8, math.radians(30))
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0)
    np.testing.assert_allclose(pose[:3, 0], [0, 0, -1])
    np.testing.assert_allclose(pose[:3, 1], [math.cos(math.radians(30)), math.sin(math.radians(30)), 0], atol=1e-12)


def test_raised_cosine_starts_slow_and_ends_exactly():
    fractions = pick.raised_cosine(30)
    assert fractions[-1] == 1.0 and (np.diff(fractions) > 0).all() and fractions[0] < 0.01


def test_joint_moves_are_timed_inside_s0s_limits():
    q1 = HOME['left'] + np.array([0.6, -0.3, 0.4, 0.9, -0.5, 1.2])
    segment = pick.joint_segment('move', LIMITS, HOME['left'], q1, 0.05, 60)
    np.testing.assert_allclose(segment.joints[-1], q1)
    speed, accel = pick._joint_rates(HOME['left'], segment.joints, 60)
    assert (speed <= pick.SPEED_FRACTION * LIMITS.max_joint_velocity_rad_s + 1e-12).all()
    assert (accel <= pick.SPEED_FRACTION * LIMITS.max_joint_acceleration_rad_s2 + 1e-12).all()
    times = 1.0 + np.arange(1, len(segment.joints) + 1) / 60
    for window in (0.0, 1 / 60):  # timed for the stricter reading (an instantaneous velocity), so it passes both
        check_joint_path(MODEL, LIMITS, now=1.0, q_now=HOME, qd_now={arm: np.zeros(6) for arm in ARMS},
                         qd_window_s=window, times=times,
                         joints={'left': segment.joints, 'right': np.repeat(HOME['right'][None], len(times), axis=0)},
                         openings={'left': 0.05, 'right': WIDTHS['right'][1]})


def test_cartesian_lines_stay_straight_and_slow():
    start = MODEL.arms['left'].tcp(HOME['left'])
    target = start.copy()
    target[2, 3] -= 0.12
    segment = pick.cartesian_segment('down', MODEL.arms['left'], LIMITS, HOME['left'], target, 0.05, 60,
                                     max_speed_m_s=0.03)
    positions = np.array([MODEL.arms['left'].tcp(q)[:3, 3] for q in segment.joints])
    np.testing.assert_allclose(positions[:, :2], np.repeat(start[None, :2, 3], len(positions), axis=0), atol=2e-4)
    np.testing.assert_allclose(positions[-1], target[:3, 3], atol=2e-4)
    speeds = np.linalg.norm(np.diff(np.vstack([start[:3, 3], positions]), axis=0), axis=1) * 60
    assert speeds.max() <= 0.03 * 1.01


def kinematic_scene(tmp_path):
    task = task_from_layout(write_layout(tmp_path / 'layout'), anchor_id='frag_a')
    backend = KinematicBackend(MODEL, home_joints=HOME, width_range_m=WIDTHS, control_hz=60)
    backend.reset(task)
    return task, backend


def cube_points(task, key, pose):
    return pick.surface_points(BOX_V.astype(float) @ pose[:3, :3].T + pose[:3, 3], BOX_F)


def test_a_whole_pick_and_place_on_the_kinematic_backend(tmp_path):
    task, backend = kinematic_scene(tmp_path)
    start = backend.privileged_fragment_poses()
    observed = backend.observed_fragment_poses()['frag_b'].world_T_fragment[:2, 3]
    seen, bounds, real = [], [], backend.bound_reach
    backend.bound_reach = lambda value: bounds.append(value) or real(value)  # each segment bounds the fingers' reach
    record = pick.run_pick_and_place(backend, MODEL, LIMITS, arm='left', fragment_id='frag_b',
                                     points=cube_points(task, 'frag_b', start['frag_b']), carry_xy=(0.0, -0.05),
                                     tip_reach_m=REACH, on_step=lambda name, state: seen.append(name))
    assert len(bounds) == len(record.segments) and all(bound is not None for bound in bounds)
    end = backend.privileged_fragment_poses()
    moved = end['frag_b'][:3, 3] - start['frag_b'][:3, 3]
    np.testing.assert_allclose(moved, [0.0, -0.05, 0.0], atol=1e-3)
    for key in ('frag_a', 'frag_c'):
        np.testing.assert_array_equal(end[key], start[key])
    assert backend.holding() == {'left': None, 'right': None}
    names = [segment.name for segment in record.segments]
    assert names == ['approach', 'descend', 'close', 'squeeze', 'lift', 'hold', 'carry', 'lower', 'release',
                     'retreat', 'settle']
    windows = {segment.name: segment.window for segment in record.segments}
    assert {name for name, window in windows.items() if window and window.kind == 'grasp'} == {
        'descend', 'close', 'squeeze', 'lift'}
    assert {name for name, window in windows.items() if window and window.kind == 'place'} == {
        'lower', 'release', 'retreat'}
    # the S1 contract: the pick window on the fragment's observed pose (not the grasp TCP), the place window on its
    # target, the observed pose moved by the carry
    for name in ('descend', 'close', 'squeeze', 'lift'):
        np.testing.assert_allclose(windows[name].center_xy, observed, atol=1e-12)
    for name in ('lower', 'release', 'retreat'):
        np.testing.assert_allclose(windows[name].center_xy, observed + [0.0, -0.05], atol=1e-12)
    grasp, widths = record.grasp, {segment.name: (segment.first_width_m, segment.last_width_m)
                                   for segment in record.segments}
    assert widths['approach'] == widths['descend'] == (grasp.open_width_m, grasp.open_width_m)
    assert widths['close'][1] == widths['squeeze'][0] == widths['lift'][0] == grasp.close_width_m
    assert widths['carry'] == widths['lower'] == (grasp.close_width_m, grasp.close_width_m)
    assert widths['release'][1] == widths['retreat'][0] == grasp.open_width_m > grasp.close_width_m
    assert len(seen) == sum(segment.count for segment in record.segments)


def test_s0_refuses_a_descent_too_fast_for_the_contact_band_before_anything_moves(tmp_path, monkeypatch):
    task, backend = kinematic_scene(tmp_path)
    monkeypatch.setattr(pick, 'CONTACT_SPEED_FRACTION', 3.0)
    reached = []
    with pytest.raises(Rejected, match='move at'):
        pick.run_pick_and_place(backend, MODEL, LIMITS, arm='left', fragment_id='frag_b',
                                points=cube_points(task, 'frag_b', backend.privileged_fragment_poses()['frag_b']),
                                carry_xy=(0.0, -0.05), tip_reach_m=REACH,
                                on_step=lambda name, state: reached.append(name))
    assert set(reached) == {'approach'}


def test_without_the_contact_phase_s0_refuses_the_descent(tmp_path):
    task, backend = kinematic_scene(tmp_path)
    limits = replace(LIMITS, contact_table_clearance_m=None, contact_region_radius_m=None, contact_max_speed_m_s=None)
    reached = []
    with pytest.raises(Rejected, match='contact phase is not configured'):
        pick.run_pick_and_place(backend, MODEL, limits, arm='left', fragment_id='frag_b',
                                points=cube_points(task, 'frag_b', backend.privileged_fragment_poses()['frag_b']),
                                carry_xy=(0.0, -0.05), tip_reach_m=REACH,
                                on_step=lambda name, state: reached.append(name))
    assert set(reached) == {'approach'}


def test_the_open_pads_need_their_margin_inside_the_opening():
    """An 80 mm chord fits a 94 mm opening, but not with 10 mm of clearance on each side."""
    with pytest.raises(ValueError, match='no top-down grasp'):
        pick.choose_grasp('f', box_points((0.08, 0.08, 0.04), (0.0, 0.0), 0), TABLE, 0.094)


def test_the_tcp_sits_where_both_pads_meet_the_fragment_at_once_not_on_its_centre():
    """An L of two boxes: along the closing axis the pads meet the narrow arm, whose middle is off the L's centre;
    from the chosen TCP both tips stop at the same joint, so neither pushes the fragment before the other touches."""
    from tasks.dual_arm_crag_skull.sim.fingers import yam_finger_reach
    long_arm = box_points((0.10, 0.02, 0.03), (0.0, 0.0), 0)
    short_arm = box_points((0.02, 0.06, 0.03), (0.04, 0.04), 0)
    points = np.vstack([long_arm, short_arm])
    grasp = pick.choose_grasp('f', points, TABLE, 0.094)
    closing = np.array([math.cos(grasp.yaw_rad), math.sin(grasp.yaw_rad)])
    across = np.array([closing[1], -closing[0]])
    offsets = points[:, :2] - np.asarray(grasp.center_xy)
    assert abs(np.asarray(grasp.center_xy) - (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2).max() > 0.005
    stops = yam_finger_reach().stop_joints(points[:, 2] - TABLE - pick.FINGERTIP_HEIGHT_M, offsets @ across,
                                           offsets @ closing)
    assert stops['tip_left'] == pytest.approx(stops['tip_right'], abs=1e-12)


def test_joint_moves_are_retimed_for_acceleration_as_well_as_speed():
    """With speed to spare and little acceleration, the move must still be slow enough to start and stop."""
    gentle = replace(LIMITS, max_joint_velocity_rad_s=np.full(6, 100.0), max_joint_acceleration_rad_s2=np.full(6, 2.0))
    segment = pick.joint_segment('move', gentle, HOME['left'], HOME['left'] + 0.4, 0.05, 60)
    speed, accel = pick._joint_rates(HOME['left'], segment.joints, 60)
    assert (accel <= pick.SPEED_FRACTION * 2.0 + 1e-9).all() and (speed < 0.05 * 100.0).all()


def test_the_turned_pose_keeps_pointing_down():
    """When IK needs the 180-degree turn about the approach axis, the TCP still points at the table."""
    target = pick.down_pose((-0.4, 0.1), 0.86, 0.3)
    tried, original = [], pick.solve_ik

    def solve(arm, pose, seed, options):  # the first candidate fails, as if out of reach
        tried.append(pose)
        solution = original(arm, pose, seed, options)
        return solution if len(tried) == 2 else replace(solution, ok=False, q=None, reason='joint_limit')

    try:
        pick.solve_ik = solve
        _, used = pick.solve_reachable(MODEL.arms['left'], target, HOME['left'])
    finally:
        pick.solve_ik = original
    np.testing.assert_allclose(used[:3, 0], [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(used[:3, 1], -target[:3, 1], atol=1e-12)
    np.testing.assert_allclose(used[:3, 3], target[:3, 3])
    assert np.linalg.det(used[:3, :3]) == pytest.approx(1.0)


class Lagging:
    """A backend whose measured arm trails its targets: S0 must check every segment from what it measures."""

    def __init__(self, inner, arm, offset, velocity):
        self._inner, self._arm, self._offset, self._velocity = inner, arm, offset, velocity

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _measured(self, state):
        joints = {**state.joints, self._arm: np.asarray(state.joints[self._arm]) + self._offset}
        velocities = {**state.joint_velocities, self._arm: np.asarray(state.joint_velocities[self._arm]) + self._velocity}
        return replace(state, joints=joints, joint_velocities=velocities)

    def robot_state(self):
        return self._measured(self._inner.robot_state())

    def step(self, joints, widths):
        return self._measured(self._inner.step(joints, widths))


def test_s0_checks_each_segment_from_the_measured_velocity(tmp_path):
    """An arm measured turning at 2.5 rad/s cannot follow a path planned from rest within S0's acceleration limit:
    the segment is refused, which only a check from the measured state can do (segments are planned from the
    measured joints, so those always agree with the path's start)."""
    task, backend = kinematic_scene(tmp_path)
    lagging = Lagging(backend, 'left', np.zeros(6), np.array([0, 0, 0, 0, 0, 2.5]))
    with pytest.raises(Rejected):
        pick.run_pick_and_place(lagging, MODEL, LIMITS, arm='left', fragment_id='frag_b',
                                points=cube_points(task, 'frag_b', backend.privileged_fragment_poses()['frag_b']),
                                carry_xy=(0.0, -0.05), tip_reach_m=REACH)


def test_joint_rates_measure_the_first_step_as_strictly_as_s0_can():
    """From rest the first segment's acceleration over half a period, 2 v1 / dt: check_joint_path's reading for an
    instantaneous measured velocity, twice its reading for a period's average (what the backends report)."""
    speed, accel = pick._joint_rates(np.zeros(6), np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]), 60)
    assert speed[0] == pytest.approx(6.0) and accel[0] == pytest.approx(720.0)


def test_grasp_candidates_list_every_yaw_that_fits_and_the_narrowest_is_chosen():
    points = box_points((0.06, 0.03, 0.04), (0.3, -0.1), 30)
    candidates = list(pick.grasp_candidates('f', points, TABLE, 0.094))
    assert len({round(math.degrees(c.yaw_rad)) for c in candidates}) == len(candidates) > 10
    assert all(c.open_width_m <= 0.094 for c in candidates)
    narrowest = min(candidates, key=lambda c: c.chord_m)
    assert pick.choose_grasp('f', points, TABLE, 0.094) == narrowest
    assert list(pick.grasp_candidates('f', points, TABLE, 0.02)) == []  # nothing fits a 20 mm opening
