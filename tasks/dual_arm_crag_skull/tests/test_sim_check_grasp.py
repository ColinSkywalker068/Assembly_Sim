"""The CPU half of sim/check_grasp.py: its criteria, the carry direction and the candidate screening."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tasks.dual_arm_crag_skull.baselines.task import task_from_layout
from tasks.dual_arm_crag_skull.sim.backend import RestTolerance
from tasks.dual_arm_crag_skull.sim.check_grasp import (CRITERIA, attachments, dynamic_body, free_carry, judge,
                                                     measure, screen)
from tasks.dual_arm_crag_skull.sim.pick import APPROACH_HEIGHT_M, LIFT_M
from tasks.dual_arm_crag_skull.tests.baselines_fixtures import write_layout
from tasks.dual_arm_crag_skull.tests.test_sim_pick import HOME, LIMITS, MODEL, REACH


def passing():
    return {'completed': True, 'refusal': None,
            'grasp': {'chord_m': 0.04, 'close_width_m': 0.034, 'squeezed_width_m': 0.038},
            'lift_target_m': 0.1, 'lift_m': 0.099, 'hold_slip_m': 1e-4, 'hold_slip_deg': 0.1,
            'transport_slip_m': 2e-4, 'transport_slip_deg': 0.2, 'attachments': [], 'carried_dynamic': True,
            'carry_error_m': 0.002, 'place': {'height_error_m': 0.001, 'xy_error_m': 0.004, 'tilt_deg': 1.0,
                                              'at_rest': True},
            'max_step_m': 0.003, 'others_moved_m': 0.0, 'others_turned_deg': 0.0,
            'cameras': {'valid': True, 'time_matches': True}}


def test_a_report_within_every_criterion_passes():
    assert judge(passing()) == []


@pytest.mark.parametrize('change, message', [
    (lambda r: r.update(completed=False, refusal='S0: descend rejected'), 'did not complete'),
    (lambda r: r['grasp'].update(squeezed_width_m=0.005), 'closed on nothing'),
    (lambda r: r['grasp'].update(squeezed_width_m=0.034), 'closed on nothing'),  # reached the command: on air
    (lambda r: r['grasp'].update(squeezed_width_m=0.0369), 'closed on nothing'),
    (lambda r: r['grasp'].update(squeezed_width_m=0.05), 'wider than the chord'),
    (lambda r: r.update(lift_m=0.05), 'lifted'),
    (lambda r: r.update(lift_m=float('nan')), 'lifted'),
    (lambda r: r.update(hold_slip_m=0.003), 'slipped'),
    (lambda r: r.update(hold_slip_deg=3.0), 'slipped'),
    (lambda r: r.update(transport_slip_m=0.003), 'slipped'),
    (lambda r: r.update(transport_slip_deg=3.0), 'slipped'),
    (lambda r: r.update(attachments=['/World/Grippers/Left/Weld']), 'attached'),
    (lambda r: r.update(carried_dynamic=False), 'attached'),
    (lambda r: r.update(carry_error_m=0.02), 'carried'),
    (lambda r: r['place'].update(height_error_m=0.01), 'not back on the table'),
    (lambda r: r['place'].update(xy_error_m=0.02), 'from its place target'),
    (lambda r: r['place'].update(tilt_deg=15.0), 'tilted'),
    (lambda r: r['place'].update(at_rest=False), 'not at rest'),
    (lambda r: r.update(max_step_m=0.02), 'jumped'),
    (lambda r: r.update(others_moved_m=0.003), 'other fragments'),
    (lambda r: r.update(others_turned_deg=3.0), 'other fragments'),
    (lambda r: r['cameras'].update(valid=False), 'cameras'),
    (lambda r: r['cameras'].update(time_matches=False), 'cameras'),
])
def test_any_criterion_missed_fails(change, message):
    report = copy.deepcopy(passing())
    change(report)
    failures = judge(report)
    assert failures and any(message in failure for failure in failures), failures


def test_carry_goes_where_nothing_else_is(tmp_path):
    task = task_from_layout(write_layout(tmp_path / 'layout'), anchor_id='frag_a')
    poses = task.initial_poses()
    carry = free_carry(task, 'frag_b', poses, distance_m=0.05)
    assert np.linalg.norm(carry) == pytest.approx(0.05)
    moved = poses['frag_b'][:3, 3] + [*carry, 0]
    for key in ('frag_a', 'frag_c'):
        assert np.linalg.norm(moved[:2] - poses[key][:2, 3]) > 0.06


def test_screening_keeps_the_candidates_s0_accepts_end_to_end(tmp_path):
    task = task_from_layout(write_layout(tmp_path / 'layout'), anchor_id='frag_a')
    results = screen(task, MODEL, LIMITS, HOME, tip_reach_m=REACH, fragment_ids=('frag_b',))
    assert [(r['fragment_id'], r['arm']) for r in results] == [('frag_b', 'left'), ('frag_b', 'right')]
    left = next(r for r in results if r['arm'] == 'left')
    assert left['accepted'] and left['refusal'] is None and len(left['carry_xy']) == 2


def test_carry_turns_away_from_a_neighbour_in_the_first_direction(tmp_path):
    from dataclasses import replace
    task = task_from_layout(write_layout(tmp_path / 'layout'), anchor_id='frag_a')
    poses = task.initial_poses()
    blocked = poses['frag_c'].copy()
    blocked[:2, 3] = poses['frag_b'][:2, 3] + [0.0, -0.07]
    crowded = replace(task, fragments=tuple(replace(f, initial_world_T_fragment=blocked) if f.fragment_id == 'frag_c'
                                            else f for f in task.fragments))
    assert free_carry(task, 'frag_b', poses) == (0.0, -0.05)
    carry = free_carry(crowded, 'frag_b', crowded.initial_poses())
    assert carry != (0.0, -0.05) and np.linalg.norm(carry) == pytest.approx(0.05)


def test_carry_stays_out_of_the_robot_bases(tmp_path):
    from dataclasses import replace
    task = task_from_layout(write_layout(tmp_path / 'layout'), anchor_id='frag_a')
    near_base = task.initial_poses()['frag_b'].copy()
    near_base[:2, 3] = [-0.60, 0.0]  # beside the base zone: only moving away from it, along +x, clears it
    moved = replace(task, fragments=tuple(replace(f, initial_world_T_fragment=near_base) if f.fragment_id == 'frag_b'
                                          else f for f in task.fragments),
                    exclusions_m=((-0.95, -0.61, -0.17, 0.17),))
    assert free_carry(moved, 'frag_b', moved.initial_poses()) == (0.05, 0.0)


def test_the_lift_returns_to_the_approach_height_the_criterion_expects():
    assert LIFT_M == APPROACH_HEIGHT_M


def _pose(xyz, *, yaw_deg=0.0, tilt_deg=0.0):
    """world_T_body: a turn about world z by yaw_deg after one about world x by tilt_deg, at xyz."""
    yaw, tilt = math.radians(yaw_deg), math.radians(tilt_deg)
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(tilt), -math.sin(tilt)], [0.0, math.sin(tilt), math.cos(tilt)]])
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rz @ rx, xyz
    return pose


START = np.array([0.20, 0.10, 0.80])
CARRY = (-0.05, 0.0)
GRASP = SimpleNamespace(chord_m=0.056, yaw_rad=0.0, open_width_m=0.076, close_width_m=0.050)
SQUEEZED = 0.0561
STEPS = (('approach', 10), ('descend', 12), ('close', 6), ('squeeze', 6), ('lift', 12), ('hold', 30), ('carry', 12),
         ('lower', 12), ('release', 6), ('retreat', 12), ('settle', 60))


def synthetic_log(*, slip=None, end_offset=(0.0, 0.0, 0.0), end_yaw_deg=0.0, end_tilt_deg=0.0, settle_bump_m=0.0,
                  settle_steps=None):
    """A 60 Hz pick and place: the fragment rides rigidly 20 mm below the TCP from the squeeze to the release, except
    for `slip` (segment, metres down the fingers over that segment). Returns (log, start, end)."""
    dt, time_s, log = 1 / 60, 0.0, []
    grasp_tcp = START + [0.0, 0.0, 0.02]
    pre, carried = grasp_tcp + [0.0, 0.0, LIFT_M], grasp_tcp + [*CARRY, LIFT_M]
    placed = START + [*CARRY, 0.0]
    final = _pose(placed + end_offset, yaw_deg=end_yaw_deg, tilt_deg=end_tilt_deg)
    tcp_path = {'approach': (pre, pre), 'descend': (pre, grasp_tcp), 'close': (grasp_tcp, grasp_tcp),
                'squeeze': (grasp_tcp, grasp_tcp), 'lift': (grasp_tcp, pre), 'hold': (pre, pre),
                'carry': (pre, carried), 'lower': (carried, placed + [0.0, 0.0, 0.02]),
                'release': (placed + [0.0, 0.0, 0.02],) * 2, 'retreat': (placed + [0.0, 0.0, 0.02], carried),
                'settle': (carried, carried)}
    dropped = 0.0
    for name, count in STEPS:
        count = settle_steps if name == 'settle' and settle_steps is not None else count
        first, last = (np.asarray(p, dtype=float) for p in tcp_path[name])
        for k in range(1, count + 1):
            time_s += dt
            tcp = _pose(first + (last - first) * k / count)
            if slip is not None and slip[0] == name:
                dropped = slip[1] * k / count
            if name in ('approach', 'descend', 'close'):
                fragment = _pose(START)
            elif name in ('squeeze', 'lift', 'hold', 'carry', 'lower'):
                fragment = _pose(tcp[:3, 3] - [0.0, 0.0, 0.02 + dropped])
            elif name == 'settle':
                bump = settle_bump_m if k == count // 2 else 0.0
                fragment = final.copy()
                fragment[0, 3] += bump
            else:
                fragment = final
            width = (GRASP.open_width_m if name in ('approach', 'descend', 'release', 'retreat', 'settle') else
                     max(SQUEEZED, GRASP.open_width_m - (GRASP.open_width_m - GRASP.close_width_m) * k / count)
                     if name == 'close' else SQUEEZED)
            log.append((name, time_s, fragment, tcp, width))
    other = _pose([0.4, -0.1, 0.8])
    return log, {'frag': _pose(START), 'other': other}, {'frag': final, 'other': other}


def measured(**kwargs):
    log, start, end = synthetic_log(**kwargs)
    return measure(log, SimpleNamespace(grasp=GRASP), start, end, 'frag', CARRY, RestTolerance())


def test_measure_reads_a_clean_pick_and_place():
    report = measured()
    assert report['grasp']['squeezed_width_m'] == SQUEEZED and report['grasp']['close_width_m'] == GRASP.close_width_m
    assert report['lift_m'] == pytest.approx(LIFT_M) and report['lift_target_m'] == LIFT_M
    for key in ('hold_slip_m', 'hold_slip_deg', 'transport_slip_m', 'transport_slip_deg', 'carry_error_m'):
        assert report[key] == pytest.approx(0.0, abs=1e-12), key
    assert report['place'] == {'height_error_m': pytest.approx(0.0, abs=1e-12),
                               'xy_error_m': pytest.approx(0.0, abs=1e-12), 'tilt_deg': pytest.approx(0.0, abs=1e-9),
                               'at_rest': True}
    assert report['max_step_m'] == pytest.approx(LIFT_M / 12)
    assert report['others_moved_m'] == 0.0 and report['others_turned_deg'] == 0.0
    passing_parts = {**passing(), **report, 'cameras': {'valid': True, 'time_matches': True}, 'attachments': [],
                     'carried_dynamic': True}
    assert judge(passing_parts) == []


@pytest.mark.parametrize('segment', ['lift', 'carry', 'lower'])
def test_slip_while_moving_is_measured_relative_to_the_tcp(segment):
    report = measured(slip=(segment, 0.015))
    assert report['transport_slip_m'] == pytest.approx(0.015 if segment != 'lower' else 0.0, abs=1e-12)
    assert report['hold_slip_m'] == pytest.approx(0.0, abs=1e-12)


def test_slip_during_the_hold_is_measured_over_the_hold():
    report = measured(slip=('hold', 0.003))
    holds = dict(STEPS)['hold']  # the slip has begun by the first hold entry: one step of it is before the window
    assert report['hold_slip_m'] == pytest.approx(0.003 * (holds - 1) / holds)
    assert report['transport_slip_m'] == pytest.approx(0.003)


def test_the_place_target_is_the_start_moved_by_the_carry():
    assert measured()['place']['xy_error_m'] == pytest.approx(0.0, abs=1e-12)
    report = measured(end_offset=(0.003, -0.004, 0.002))
    assert report['place']['xy_error_m'] == pytest.approx(0.005)
    assert report['place']['height_error_m'] == pytest.approx(0.002)


def test_tilt_is_the_turn_of_the_vertical_axis_not_of_a_yaw():
    assert measured(end_yaw_deg=30.0)['place']['tilt_deg'] == pytest.approx(0.0, abs=1e-9)
    assert measured(end_tilt_deg=5.0)['place']['tilt_deg'] == pytest.approx(5.0)


def test_rest_needs_the_whole_window_still_and_long_enough():
    assert measured()['place']['at_rest']
    assert not measured(settle_bump_m=0.001)['place']['at_rest']  # moved and came back inside the window
    assert not measured(settle_steps=20)['place']['at_rest']  # a settle shorter than the rest window


def test_attachments_lists_every_joint_on_the_fragment():
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    for path in ('/World/Fragments/frag_b', '/World/Fragments/frag_b/body', '/World/Fragments/frag_c',
                 '/World/Grippers/Left/tip_left'):
        UsdGeom.Xform.Define(stage, path)
    assert attachments(stage, '/World/Fragments/frag_b') == []
    joint = UsdPhysics.FixedJoint.Define(stage, '/World/Grippers/Left/Weld')
    joint.CreateBody0Rel().SetTargets(['/World/Grippers/Left/tip_left'])
    joint.CreateBody1Rel().SetTargets(['/World/Fragments/frag_b/body'])
    other = UsdPhysics.FixedJoint.Define(stage, '/World/Grippers/Left/Other')
    other.CreateBody1Rel().SetTargets(['/World/Fragments/frag_c'])
    assert attachments(stage, '/World/Fragments/frag_b') == ['/World/Grippers/Left/Weld']


def test_only_an_enabled_non_kinematic_rigid_body_is_dynamic():
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    paths = {name: f'/World/Fragments/{name}' for name in ('plain', 'dynamic', 'kinematic', 'disabled')}
    for path in paths.values():
        UsdGeom.Xform.Define(stage, path)
    for name in ('dynamic', 'kinematic', 'disabled'):
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(paths[name]))
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(paths['kinematic'])).CreateKinematicEnabledAttr(True)
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(paths['disabled'])).CreateRigidBodyEnabledAttr(False)
    assert {name: dynamic_body(stage, path) for name, path in paths.items()} == {
        'plain': False, 'dynamic': True, 'kinematic': False, 'disabled': False}


def test_s0_limits_need_a_whole_servo_rate():
    from tasks.dual_arm_crag_skull.sim.check_grasp import _whole_hz
    assert _whole_hz(60.0) == 60
    with pytest.raises(ValueError, match='whole servo rate'):
        _whole_hz(59.5)


def test_the_scripted_reference_runs_under_the_baselines_s0_test_limits():
    """One S0 for every method: the scripted runs' TEST limits are the baselines' at the same servo rate and widths."""
    from dataclasses import fields
    from types import SimpleNamespace
    from tasks.dual_arm_crag_skull.baselines.aspire.__main__ import test_limits
    from tasks.dual_arm_crag_skull.contracts.actions import ARMS
    from tasks.dual_arm_crag_skull.s0.kinematics import yam_width_range_m
    from tasks.dual_arm_crag_skull.sim.check_grasp import s0_test_limits
    widths = {arm: yam_width_range_m() for arm in ARMS}
    task = SimpleNamespace(table_top_z_m=0.75)
    ours, theirs = s0_test_limits(task, SimpleNamespace(width_range=widths, control_hz=60.0)), test_limits(task, widths)
    for field in fields(ours):
        assert np.array_equal(np.asarray(getattr(ours, field.name), dtype=object),
                              np.asarray(getattr(theirs, field.name), dtype=object)), field.name
    assert ours.finger_opening_margin_m is not None
