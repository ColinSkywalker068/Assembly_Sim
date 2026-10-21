"""S0 takes the fingers where they can be: with `finger_opening_margin_m` set, its finger and tool boxes are the YAM
tips' slabs at every opening up to the widest the gripper can have in the checked motion (the measured opening, the
last command and every new one) plus the margin, instead of one box over the whole stroke.

The test pose: the YAM pointing down, turned 30 deg about its own z, so the closing axis dips 30 deg; the TCP 25 mm
over the table. One box over the whole stroke reaches 70.7 mm along the closing axis, 35 mm under the TCP: 10 mm into
the table. Open to 32 mm (30 + the 2 mm margin) the last 5 mm of the tips reach 20.4 mm out, 10.2 mm under the TCP:
14.7 mm over the table, above the normal 10 mm tool clearance.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tasks.dual_arm_crag_skull.contracts.actions import ARMS, encode_chunk
from tasks.dual_arm_crag_skull.s0 import adapter as adapter_module
from tasks.dual_arm_crag_skull.s0.adapter import Rejected, S0Adapter, widest_opening
from tasks.dual_arm_crag_skull.s0.ik import IKOptions, solve_ik
from tasks.dual_arm_crag_skull.s0.kinematics import yam_model, yam_width_range_m
from tasks.dual_arm_crag_skull.tests.test_s0_adapter import HOME, limits as base_limits

YAM = yam_model()
TABLE = 0.75
OPEN = 0.03
MARGIN = 0.002
CONTACT = dict(contact_table_clearance_m=0.002, contact_region_radius_m=0.15, contact_max_speed_m_s=0.05)
TILTED = (Rotation.from_euler('y', 90, degrees=True) * Rotation.from_euler('z', 30, degrees=True)).as_matrix()


def limits(**overrides):
    return base_limits(**{'gripper_width_range_m': {arm: yam_width_range_m() for arm in ARMS}, **CONTACT,
                          'finger_opening_margin_m': MARGIN, 'max_joint_acceleration_rad_s2': np.full(6, 200.0),
                          **overrides})


def tcp_joints(position, rotation, seed=HOME):
    target = np.eye(4)
    target[:3, :3], target[:3, 3] = rotation, position
    result = solve_ik(YAM.arms['left'], target, seed, IKOptions(max_seed_jump_rad=3.0))
    assert result.ok, result
    return result.q


Q_TILT = tcp_joints([-0.42, 0.0, TABLE + 0.025], TILTED)
Q_HOME = {arm: HOME.copy() for arm in ARMS}


def hold(q=Q_TILT, count=6, **kwargs):
    """check_joint_path of the left arm holding q (and the right arm home) for `count` servo periods."""
    defaults = dict(openings={'left': OPEN, 'right': 0.0})
    defaults.update(kwargs)
    limit = defaults.pop('limits', None) or limits()
    adapter_module.check_joint_path(YAM, limit, now=1.0, q_now={'left': q, 'right': HOME},
                                    qd_now={arm: np.zeros(6) for arm in ARMS}, qd_window_s=1 / 60,
                                    times=1.0 + np.arange(1, count + 1) / 60,
                                    joints={'left': np.repeat(q[None], count, axis=0),
                                            'right': np.repeat(HOME[None], count, axis=0)}, **defaults)


def test_the_test_pose_is_what_the_docstring_says():
    arm = YAM.arms['left']
    assert arm.tcp(Q_TILT)[2, 3] - TABLE == pytest.approx(0.025, abs=1e-4)  # S0 IK is good to 0.1 mm
    assert arm.finger_box_corners(Q_TILT)[:, 2].min() - TABLE == pytest.approx(-0.0104, abs=5e-4)
    assert arm.finger_box_corners(Q_TILT, OPEN + MARGIN)[:, 2].min() - TABLE == pytest.approx(0.0147, abs=5e-4)
    assert arm.tool_box_corners(Q_TILT, OPEN + MARGIN)[:, 2].min() - TABLE > 0.010


def test_the_fingers_at_their_opening_are_off_the_table_where_the_whole_stroke_box_is_in_it():
    hold()  # passes S0's normal rules: no contact needed
    with pytest.raises(Rejected, match='tool box'):
        hold(limits=limits(finger_opening_margin_m=None))  # S0 as before: the whole stroke
    with pytest.raises(Rejected, match='tool box'):
        hold(openings={'left': yam_width_range_m()[1], 'right': 0.0})  # fully open is the whole stroke again


def test_the_margin_is_added_to_the_opening():
    arm = YAM.arms['left']
    # the opening at which the boxes' lowest corner reaches the 10 mm floor, less the margin, still passes; more fails
    widths = np.linspace(0.0, 0.09, 901)
    low = np.array([arm.tool_box_corners(Q_TILT, w)[:, 2].min() - TABLE for w in widths])
    edge = widths[np.flatnonzero(low < 0.010)[0]]
    hold(openings={'left': edge - MARGIN - 2e-4, 'right': 0.0})
    with pytest.raises(Rejected, match='tool box'):
        hold(openings={'left': edge - MARGIN + 2e-4, 'right': 0.0})


@pytest.mark.parametrize('openings', [None, {'left': OPEN}, {'left': np.nan, 'right': 0.0},
                                      {'left': -0.001, 'right': 0.0}, {'left': 'wide', 'right': 0.0}])
def test_with_the_margin_set_every_check_needs_both_grippers_widest_openings(openings):
    with pytest.raises(Rejected, match='opening'):
        hold(openings=openings)


def test_without_the_margin_openings_are_not_needed():
    hold(q=HOME, limits=limits(finger_opening_margin_m=None), openings=None)


@pytest.mark.parametrize('value', [-0.001, np.nan, np.inf, True])
def test_the_margin_must_be_a_finite_non_negative_length(value):
    with pytest.raises(ValueError, match='finger_opening_margin_m'):
        limits(finger_opening_margin_m=value)


def test_the_widest_opening_is_where_the_gripper_is_was_sent_or_is_sent():
    assert widest_opening(0.03, 0.02, [0.01, 0.025]) == 0.03  # held wider than commanded by what it grips
    assert widest_opening(0.01, 0.05, [0.02]) == 0.05  # may still be on its way to the last command
    assert widest_opening(0.01, 0.02, [0.02, 0.06, 0.03]) == 0.06
    assert widest_opening(0.01, 0.02) == 0.02
    with pytest.raises(Rejected, match='width'):
        widest_opening(np.nan, 0.02, [0.01])


def test_below_holds_each_box_part_under_the_floor():
    low = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.5, 1.5)])  # crosses z = 1
    high = low + [0.0, 0.0, 2.0]  # all above
    part = adapter_module._below(np.vstack([high, low]), 1.0)
    assert (part[:, 2] <= 1.0 + 1e-12).all() and len(part) == 8  # 4 corners below, 4 edge crossings
    np.testing.assert_allclose(np.sort(part[:, 2]), [0.5] * 4 + [1.0] * 4)
    assert len(adapter_module._below(high, 1.0)) == 0


# ---------------------------------------------------------------- the S1 path: S0Adapter

def chunk_holding(q_left, width):
    """The left arm holds q_left and its gripper goes to `width`; the right arm holds home at OPEN."""
    anchors = {'left': YAM.arms['left'].tcp(q_left), 'right': YAM.arms['right'].tcp(HOME)}
    targets = {arm: np.repeat(anchors[arm][None], 3, axis=0) for arm in ARMS}
    return encode_chunk(request_id='req', chunk_id=1, observation_time=1.0, valid_until=1.6, anchors=anchors,
                        waypoint_times=1.1 + .1 * np.arange(3), targets=targets,
                        widths={'left': np.full(3, width), 'right': np.full(3, OPEN)}, active_arms=('left',))


@pytest.mark.parametrize('measured, commanded, chunk_width, accepted', [
    (OPEN, OPEN, OPEN, True),
    (0.09, OPEN, OPEN, False),  # fingers held wide open (by something between them): the box is where they are
    (OPEN, 0.09, OPEN, False),  # the last command was wide open: they may still be on their way there
    (OPEN, OPEN, 0.09, False),  # the chunk opens them: the box is as wide as it sends them
])
def test_s0_adapter_takes_the_fingers_where_they_are_were_sent_and_are_sent(measured, commanded, chunk_width,
                                                                              accepted):
    adapter = S0Adapter(YAM, limits())
    chunk = chunk_holding(Q_TILT, chunk_width)
    status, _ = adapter.plan(chunk, now=1.0, joints={'left': Q_TILT, 'right': HOME},
                             joint_velocities={arm: np.zeros(6) for arm in ARMS},
                             widths={'left': measured, 'right': OPEN},
                             commanded_widths={'left': commanded, 'right': OPEN},
                             tcp_at_observation={arm: np.array(chunk.anchors[arm]) for arm in ARMS})
    assert (status.status == 'accepted') == accepted, status.reason
    if not accepted:
        assert 'tool box' in status.reason


Q_LOWER = tcp_joints([-0.42, 0.0, TABLE + 0.015], TILTED, seed=Q_TILT)  # the TCP 15 mm over the table


def test_a_contact_holds_the_fingers_at_their_opening_too():
    """The TCP 15 mm over the table: the boxes at 32 mm reach 4.6 mm over it and 21 mm from the TCP axis, inside a
    contact's 2 mm clearance and a 40 mm region round the TCP; the whole stroke reaches into the table and 62 mm out."""
    from tasks.dual_arm_crag_skull.contracts.actions import ContactWindow
    arm = YAM.arms['left']
    tcp = arm.tcp(Q_LOWER)[:3, 3]
    low = arm.finger_box_corners(Q_LOWER, OPEN + MARGIN)
    assert 0.002 < low[:, 2].min() - TABLE < 0.010
    below = adapter_module._below(low, TABLE + 0.010)
    assert np.hypot(*(below[:, :2] - tcp[:2]).T).max() < 0.03
    window = ContactWindow(arm='left', kind='grasp', center_xy=tuple(tcp[:2]), fragment_id='frag_b')
    tight = limits(contact_region_radius_m=0.04)
    hold(q=Q_LOWER, contacts=(window,), limits=tight)
    with pytest.raises(Rejected, match='no contact declared'):
        hold(q=Q_LOWER, limits=tight)
    with pytest.raises(Rejected, match='contact centre'):  # fully open: the boxes below the floor reach 62 mm out
        hold(q=Q_LOWER, contacts=(window,), limits=tight, openings={'left': yam_width_range_m()[1], 'right': 0.0})
    with pytest.raises(Rejected, match='fingers are'):  # a wide region: then the whole stroke is in the table
        hold(q=Q_LOWER, contacts=(window,), limits=limits(), openings={'left': yam_width_range_m()[1], 'right': 0.0})
