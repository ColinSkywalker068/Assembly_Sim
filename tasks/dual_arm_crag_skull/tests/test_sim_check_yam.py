"""The judging half of sim/check_yam.py (the Isaac half runs on a GPU)."""
import copy

import numpy as np
import pytest

from core.robots.kinematics import T
from tasks.dual_arm_crag_skull.sim.check_yam import (BOX_PROBES, COOKED_PROTRUSION_M, COOKED_SHRINK_M,
                                                      POSITION_TOLERANCE_M, ROTATION_TOLERANCE_DEG,
                                                      TCP_FROM_PADS_TOLERANCE_M, WIDTH_TOLERANCE_M, judge, pad_midpoint,
                                                      pose_error)


def test_pose_error_in_millimetres_and_degrees():
    assert pose_error(np.eye(4), np.eye(4)) == (0.0, 0.0)
    assert pose_error(np.eye(4), T([0.0, 0.001, 0.0], [1, 0, 0, 0])) == pytest.approx((0.001, 0.0))
    half = np.radians(0.2) / 2
    assert pose_error(T([1, 2, 3], [1, 0, 0, 0]), T([1, 2, 3], [np.cos(half), 0, 0, np.sin(half)])) == \
        pytest.approx((0.0, 0.2))


def test_pose_error_resolves_small_angles_of_float32_rotations():
    """PhysX poses are float32; an arccos of the trace cannot resolve angles below about 0.03 deg there."""
    half = np.radians(0.001) / 2
    measured = T([0.0, 0.0, 0.0], [np.cos(half), np.sin(half), 0.0, 0.0]).astype(np.float32)
    assert pose_error(np.eye(4), measured)[1] == pytest.approx(0.001, rel=0.05)


def passing():
    pose = {'errors': {part: (POSITION_TOLERANCE_M / 2, ROTATION_TOLERANCE_DEG / 2)
                       for part in ('flange', 'gripper_body', 'tip_left', 'tip_right')},
            'tcp_from_pads_m': TCP_FROM_PADS_TOLERANCE_M / 2}
    widths = [{'target_m': 0.03, 'measured_m': 0.03 + WIDTH_TOLERANCE_M / 2}]
    boxes = [{'depth_m': depth, 'width_m': width, 'side_m': side, 'shift_m': shift, 'tolerance_m': tolerance,
              'excess_m': tolerance / 2} for depth, width, tolerance, side, shift in BOX_PROBES]
    cooked = {'colliders': 600, 'valid': 600, 'max_vertices': 64, 'protrusion_m': COOKED_PROTRUSION_M / 2,
              'shrink_m': COOKED_SHRINK_M / 2, 'polygon_limit_warnings': 0, 'gpu_incompatible': 0}
    return {'arms': {arm: {'dofs_found': True, 'widths': copy.deepcopy(widths), 'poses': [copy.deepcopy(pose)],
                           'boxes': copy.deepcopy(boxes)} for arm in ('left', 'right')},
            'cooked': cooked}


def test_a_report_within_every_criterion_passes():
    assert judge(passing()) == []


@pytest.mark.parametrize('change, message', [
    (lambda r: r['arms']['left'].update(dofs_found=False), 'finger DOFs missing'),
    (lambda r: r['arms']['right']['widths'][0].update(measured_m=0.03 + 1.5 * WIDTH_TOLERANCE_M), 'width'),
    (lambda r: r['arms']['left']['widths'][0].update(measured_m=float('nan')), 'width'),
    (lambda r: r['arms']['left']['poses'][0]['errors'].update(gripper_body=(2 * POSITION_TOLERANCE_M, 0.0)),
     'gripper_body off'),
    (lambda r: r['arms']['left']['poses'][0].update(tcp_from_pads_m=2 * TCP_FROM_PADS_TOLERANCE_M), 'TCP is'),
    (lambda r: r['arms']['right']['poses'][0].update(tcp_from_pads_m=float('nan')), 'TCP is'),
    (lambda r: r['arms']['right']['poses'][0]['errors'].update(tip_left=(0.0, 2 * ROTATION_TOLERANCE_DEG)),
     'tip_left off'),
    (lambda r: r['arms']['right']['poses'][0]['errors'].update(flange=(float('nan'), 0.0)), 'flange off'),
    (lambda r: r.update(arms={}), 'no arm'),
    (lambda r: r['arms']['left']['boxes'][0].update(excess_m=0.0018), 'box'),
    (lambda r: r['arms']['right']['boxes'][1].update(excess_m=-0.0015), 'box'),
    (lambda r: r['arms']['right']['boxes'][2].update(excess_m=float('nan')), 'box'),
    (lambda r: r['arms']['left'].update(boxes=[]), 'box probes'),
    (lambda r: r['cooked'].update(valid=599), 'cooked'),
    (lambda r: r['cooked'].update(colliders=0, valid=0), 'cooked'),
    (lambda r: r['cooked'].update(max_vertices=65), 'cooked'),
    (lambda r: r['cooked'].update(protrusion_m=2 * COOKED_PROTRUSION_M), 'cooked'),
    (lambda r: r['cooked'].update(shrink_m=1.5 * COOKED_SHRINK_M), 'cooked'),
    (lambda r: r['cooked'].update(shrink_m=float('nan')), 'cooked'),
    (lambda r: r['cooked'].update(polygon_limit_warnings=1), 'cooked'),
    (lambda r: r['cooked'].update(gpu_incompatible=2), 'cooked'),
])
def test_any_criterion_missed_fails(change, message):
    report = passing()
    change(report)
    failures = judge(report)
    assert failures and any(message in failure for failure in failures)


def test_pad_midpoint_finds_the_grasp_site_from_the_tip_meshes():
    """I2RT defines the grasp site from the tip meshes; this estimate lands within 0.15 mm of it at every opening."""
    from core.robots.linear_4310 import load_linear_4310
    gripper = load_linear_4310()
    approach, opening = gripper.mount_T_tcp[:3, 2], gripper.slide_axes['tip_right']
    for q in (0.0, 0.02, gripper.travel_m):
        estimate = pad_midpoint(gripper.tips_at(q), approach, opening)
        assert np.linalg.norm(estimate - gripper.mount_T_tcp[:3, 3]) < 1.5e-4


def test_pad_midpoint_does_not_jump_under_a_tiny_rotation():
    """The fingertip ends are curved (vertex depths 0, 1, 5, 11 um ...), so an estimate built on a thin end slice
    jumped by 0.16 mm under float32 PhysX poses; extremes over the whole pad must not."""
    from core.robots.linear_4310 import load_linear_4310
    gripper = load_linear_4310()
    approach, opening = gripper.mount_T_tcp[:3, 2], gripper.slide_axes['tip_right']
    tips = gripper.tips_at(0.0)
    base = pad_midpoint(tips, approach, opening)
    for angle in (1e-6, -1e-6, 3e-6):
        turn = T([0.0, 0.0, 0.0], [np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])[:3, :3]
        turned = {name: points @ turn.T for name, points in tips.items()}
        assert np.linalg.norm(pad_midpoint(turned, approach, opening) - base) < 2e-6


def test_box_probe_stops_are_predicted_from_the_tip_meshes():
    """On the pad a box stops the fingers at its own width; on the slope above it a little narrower, as the meshes
    say (Isaac must then match the prediction, not the box)."""
    from core.robots.linear_4310 import load_linear_4310
    from tasks.dual_arm_crag_skull.sim.check_yam import predicted_stop_width
    gripper = load_linear_4310()
    assert predicted_stop_width(gripper, 0.008, 0.040) == pytest.approx(0.040, abs=2e-5)
    assert predicted_stop_width(gripper, 0.008, 0.020) == pytest.approx(0.020, abs=2e-5)
    assert predicted_stop_width(gripper, 0.025, 0.040) == pytest.approx(0.03968, abs=5e-5)


def test_a_rod_in_the_window_lets_the_fingers_close_past_it():
    """A 3 mm rod 35 mm up from the fingertips, on the middle line, meets no material on either tip: the fingers close
    fully. A collider that bridged the window would stop them at its length."""
    from core.robots.linear_4310 import load_linear_4310
    from tasks.dual_arm_crag_skull.sim.check_yam import predicted_stop_width
    gripper = load_linear_4310()
    rods = [probe for probe in BOX_PROBES if probe[3] <= 0.004]
    assert rods
    for depth, width, _, side, _ in rods:
        assert predicted_stop_width(gripper, depth, width, side / 2) == gripper.width(gripper.contact_joint_m) == 0.0
    assert predicted_stop_width(gripper, 0.035, 0.020, 0.006) < 0.020  # a 12 mm box there meets the rails


def test_cooked_hulls_may_fall_short_more_than_they_may_stick_out():
    """PhysX drops near-coplanar points from every hull (measured up to 0.083 mm inward, never outward)."""
    assert COOKED_PROTRUSION_M < COOKED_SHRINK_M <= 1e-4


def test_a_pair_of_one_pad_probes_gives_the_pads_standoff_and_their_middle():
    """Measured on Isaac with the coupled fingers at 64 position iterations (2026-10-21): 1 mm towards the driven tip
    -0.12 mm, away +0.17 mm: the two pads stand 0.025 mm off the faces together (what both pads on a box in the middle
    gave before the fingers were coupled; at 32 iterations, +0.26 / +0.75, 0.5 mm together) and their middle lies
    0.07 mm off the TCP towards the driven tip."""
    from tasks.dual_arm_crag_skull.sim.check_yam import box_excesses
    probe = lambda shift, excess: {'depth_m': 0.005, 'width_m': 0.04, 'side_m': 0.012, 'shift_m': shift,
                                   'tolerance_m': 5e-4, 'excess_m': excess}
    (_, standoff, tolerance), (_, middle, middle_tolerance) = box_excesses([probe(0.001, -0.00012),
                                                                            probe(-0.001, 0.00017)])
    assert standoff == pytest.approx(0.000025) and abs(standoff) <= tolerance
    assert middle == pytest.approx(0.0000725) and middle_tolerance == TCP_FROM_PADS_TOLERANCE_M
    (_, slow, _), _ = box_excesses([probe(0.001, 0.00026), probe(-0.001, 0.00075)])  # 32 iterations
    assert slow > tolerance
    assert box_excesses([probe(0.001, 0.0)])[0][1] == float('inf')  # a pair with one probe missing fails
    report = passing()
    report['arms']['left']['boxes'][0]['excess_m'] = 0.0012  # one probe of a pair 1.2 mm off: the mean is over 0.5 mm
    assert any('stood off' in failure for failure in judge(report))


def test_the_coupling_probe_is_judged_against_the_1_mm_probe_the_same_way():
    """The box 5 mm off the middle and the one 1 mm off it the same way stop on the same pad, 2 (dD - c) wider than
    predicted each: judged by their difference, a TCP 0.3 mm off the pads' middle passes (it would fail alone at the
    0.5 mm tolerance), and uncoupled fingers, closing on both faces 10 mm short, fail."""
    from tasks.dual_arm_crag_skull.sim.check_yam import box_excesses
    probe = lambda shift, excess: {'depth_m': 0.005, 'width_m': 0.04, 'side_m': 0.012, 'shift_m': shift,
                                   'tolerance_m': 5e-4, 'excess_m': excess}

    def coupling(far, near):
        judged = [entry for entry in box_excesses([probe(0.001, near), probe(-0.001, 0.0), probe(0.005, far)])
                  if '5 mm off' in entry[0]]
        assert len(judged) == 1 and 'beyond the box 1 mm off it' in judged[0][0]
        return judged[0][1]
    assert coupling(-0.00005, -0.00012) == pytest.approx(0.00007)  # measured on Isaac, 2026-10-21
    assert abs(coupling(-0.0006, -0.0006)) <= 5e-4  # c = 0.3 mm
    assert abs(coupling(-0.0101, -0.0001)) > 5e-4  # uncoupled
    alone = box_excesses([probe(0.005, -0.0006)])  # no 1 mm probe that way: judged on its own
    assert len(alone) == 1 and alone[0][1] == -0.0006


def test_the_boxes_are_measured_one_pad_at_a_time_and_one_tells_coupled_fingers_from_uncoupled():
    """Each box sits 1 mm off the middle one way and then the other, so one pad touches it; the coupled fingers stop on
    a box 5 mm off the middle 10 mm wider than on one in the middle, where uncoupled ones would close on both faces."""
    from core.robots.linear_4310 import load_linear_4310
    from tasks.dual_arm_crag_skull.sim.check_yam import predicted_stop_width
    gripper = load_linear_4310()
    boxes = [probe for probe in BOX_PROBES if probe[3] > 0.004 and abs(probe[4]) < 0.004]
    assert sorted(probe[4] for probe in boxes) == [-0.001] * 4 + [0.001] * 4
    far = [probe for probe in BOX_PROBES if abs(probe[4]) >= 0.004]
    assert len(far) == 1 and far[0][4] >= 10 * far[0][2]  # the 10 mm difference is far over the tolerance
    depth, width, _, side, shift = far[0]
    middle = predicted_stop_width(gripper, depth, width, side / 2)
    assert predicted_stop_width(gripper, depth, width, side / 2, shift) == pytest.approx(middle + 2 * shift, abs=2e-4)
    assert predicted_stop_width(gripper, depth, width, side / 2, -shift) == pytest.approx(middle + 2 * shift, abs=2e-4)
