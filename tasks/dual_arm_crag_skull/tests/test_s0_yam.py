"""S0's YAM linear_4310 tool geometry on the provisional CRX adapter plate."""
import numpy as np
import pytest

from core.robots.linear_4310 import load_linear_4310
from tasks.dual_arm_crag_skull.contracts.actions import ARMS
from tasks.dual_arm_crag_skull.s0.kinematics import (YAM_PROVISIONAL_ADAPTER_THICKNESS_M, yam_flange_T_mount,
                                                     yam_model, yam_tool_geometry, yam_width_range_m)

GRIPPER = load_linear_4310()
THICKNESS = YAM_PROVISIONAL_ADAPTER_THICKNESS_M


def test_the_provisional_adapter_clears_the_back_of_the_gripper():
    """The gripper body reaches 10.5 mm behind its mount plane, so a thinner plate would put it inside the flange."""
    body = GRIPPER.body_points @ yam_flange_T_mount()[:3, :3].T + yam_flange_T_mount()[:3, 3]
    assert body[:, 0].min() == pytest.approx(THICKNESS - 0.0105, abs=1e-6) and body[:, 0].min() > 0


def test_tcp_follows_the_proxy_convention():
    """The TCP frame is the flange frame moved to the grasp point: +x approach, fingers closing along y."""
    flange_T_tcp, _, _ = yam_tool_geometry()
    np.testing.assert_array_equal(flange_T_tcp[:3, :3], np.eye(3))
    np.testing.assert_allclose(flange_T_tcp[:3, 3], [THICKNESS + GRIPPER.mount_T_tcp[2, 3],
                                                     *GRIPPER.mount_T_tcp[:2, 3]], atol=1e-12)
    flange_T_mount = yam_flange_T_mount()
    np.testing.assert_allclose(flange_T_mount[:3, :3] @ GRIPPER.slide_axes['tip_right'], [0, 1, 0], atol=1e-5)
    np.testing.assert_allclose(flange_T_mount[:3, :3] @ GRIPPER.mount_T_tcp[:3, 2], [1, 0, 0], atol=1e-5)


def test_tool_box_parts_split_body_and_adapter_from_the_fingers():
    flange_T_tcp, _, box = yam_tool_geometry()
    body, fingers = box['parts']['body'], box['parts']['fingers']
    np.testing.assert_allclose(body['min'], [0.0, -0.051, -0.0325], atol=1e-6)
    np.testing.assert_allclose(body['max'], [THICKNESS + 0.0635, 0.051, 0.0325], atol=1e-6)
    # Over the whole stroke; the tips reach 0.07 mm past the TCP (2F-85 proxy: 27.8 mm).
    np.testing.assert_allclose(fingers['min'], [THICKNESS + 0.050876, -0.070723, -0.034095], atol=1e-6)
    np.testing.assert_allclose(fingers['max'], [THICKNESS + 0.14472, 0.070723, 0.034094], atol=1e-6)
    assert 0 <= fingers['max'][0] - flange_T_tcp[0, 3] < 1e-4
    np.testing.assert_allclose(box['min'], np.minimum(body['min'], fingers['min']))
    np.testing.assert_allclose(box['max'], np.maximum(body['max'], fingers['max']))


def outside_capsule(points, p0, p1, radius):
    """Signed distance of each point from the capsule surface (> 0 outside)."""
    axis = p1 - p0
    length2 = float(axis @ axis)
    t = np.zeros(len(points)) if length2 == 0 else np.clip((points - p0) @ axis / length2, 0.0, 1.0)
    return np.linalg.norm(points - (p0 + np.outer(t, axis)), axis=1) - radius


def test_capsule_encloses_the_body_adapter_and_both_fingers_over_the_stroke():
    _, capsule, box = yam_tool_geometry()
    flange_T_mount = yam_flange_T_mount()
    clouds = [GRIPPER.body_points] + [points for q in (0.0, GRIPPER.travel_m) for points in GRIPPER.tips_at(q).values()]
    points = np.concatenate(clouds) @ flange_T_mount[:3, :3].T + flange_T_mount[:3, 3]
    body = box['parts']['body']
    adapter = np.array([[x, y, z] for x in (0.0, THICKNESS) for y in (body['min'][1], body['max'][1])
                        for z in (body['min'][2], body['max'][2])])
    distances = outside_capsule(np.concatenate([points, adapter]), np.asarray(capsule['p0']),
                                np.asarray(capsule['p1']), capsule['radius'])
    assert distances.max() <= 1e-9


def test_dual_arm_model_carries_the_yam_tool_on_both_arms():
    model = yam_model()
    flange_T_tcp, _, _ = yam_tool_geometry()
    for arm in ARMS:
        assert model.arms[arm].has_tool_parts
        np.testing.assert_array_equal(model.arms[arm].flange_T_tcp, flange_T_tcp)
    assert yam_width_range_m() == (0.0, GRIPPER.max_width_m)


def test_a_plate_thinner_than_the_gripper_back_is_refused():
    with pytest.raises(ValueError, match='adapter'):
        yam_flange_T_mount(0.01)


# ---------------------------------------------------------------- the finger boxes at an opening

ARM = yam_model().arms['left']
Q = np.array([0.3, -0.4, 0.5, 0.2, 0.6, 0.1])
RNG = np.random.default_rng(7)


def in_flange(corners, q=Q):
    flange = ARM.frames(q)[0]['flange']
    return (np.linalg.inv(flange) @ np.c_[corners, np.ones(len(corners))].T).T[:, :3]


def tips_in_flange(q_finger):
    """Both tips' triangle vertices and random points on their triangles, in the flange frame."""
    flange_T_mount = yam_flange_T_mount()
    points = np.concatenate(list(GRIPPER.tips_at(q_finger).values()))
    triangles = points.reshape(-1, 3, 3)
    weights = RNG.dirichlet(np.ones(3), size=len(triangles))
    surface = np.concatenate([points, np.einsum('tk,tkd->td', weights, triangles)])
    return surface @ flange_T_mount[:3, :3].T + flange_T_mount[:3, 3]


def inside_some(points, lows, highs, tolerance=1e-9):
    return ((points[:, None] >= lows[None] - tolerance) & (points[:, None] <= highs[None] + tolerance)).all(axis=2).any(
        axis=1)


@pytest.mark.parametrize('width', [0.0, 0.012, 0.048, 0.07, GRIPPER.max_width_m])
def test_the_finger_boxes_at_an_opening_hold_the_tips_at_every_opening_up_to_it(width):
    lows, highs = ARM.finger_boxes(width)
    reach = max(GRIPPER.joint_for_width(width), 0.0)
    for q in np.linspace(0.0, reach, 5):
        assert inside_some(tips_in_flange(q), lows, highs).all()
    # and no further than the tips get: inside the box of the tips closed and at the opening
    ends = np.concatenate([tips_in_flange(0.0), tips_in_flange(reach)])
    assert (lows.min(axis=0) >= ends.min(axis=0) - 1e-9).all() and (highs.max(axis=0) <= ends.max(axis=0) + 1e-9).all()


def test_the_boxes_at_the_full_opening_span_the_whole_stroke_box_and_wider_is_the_same():
    whole_low, whole_high = (bound[0] for bound in ARM.finger_boxes())
    lows, highs = ARM.finger_boxes(GRIPPER.max_width_m)
    np.testing.assert_allclose(lows.min(axis=0), whole_low, atol=1e-12)
    np.testing.assert_allclose(highs.max(axis=0), whole_high, atol=1e-12)
    for got, expected in zip(ARM.finger_boxes(0.2), (lows, highs)):
        np.testing.assert_array_equal(got, expected)


def test_the_tip_ends_are_boxed_as_thin_as_they_are():
    """The whole-stroke box is 70.7 mm a side across the fingers, set by the tips' roots; fully open, the last 15 mm
    of each tip reach 53.1 mm out, and the slabs there say so. The slabs are FINGER_SLAB_M long along the approach."""
    from tasks.dual_arm_crag_skull.s0.kinematics import FINGER_SLAB_M
    lows, highs = ARM.finger_boxes(GRIPPER.max_width_m)
    ends = lows[:, 0] > highs[:, 0].max() - 3 * FINGER_SLAB_M - 1e-6
    assert ends.sum() == 6  # three slabs per tip
    assert np.abs(np.concatenate([lows[ends, 1], highs[ends, 1]])).max() == pytest.approx(0.0531, abs=2e-4)
    assert ARM.finger_boxes()[1][0, 1] == pytest.approx(0.0707, abs=2e-4)
    np.testing.assert_allclose(highs[ends, 0] - lows[ends, 0], FINGER_SLAB_M, atol=1e-4)


def test_the_tool_box_at_an_opening_is_the_body_box_and_the_finger_boxes():
    corners = in_flange(ARM.tool_box_corners(Q, 0.03))
    expected = np.concatenate([in_flange(ARM.body_box_corners(Q)), in_flange(ARM.finger_box_corners(Q, 0.03))])
    np.testing.assert_allclose(corners, expected, atol=1e-9)
    assert len(ARM.tool_box_corners(Q)) == 8  # without an opening: the one whole tool box
    lows, highs = ARM.finger_boxes(0.03)
    np.testing.assert_allclose(in_flange(ARM.finger_box_corners(Q, 0.03)).reshape(-1, 8, 3).min(axis=1), lows,
                               atol=1e-9)
    np.testing.assert_allclose(in_flange(ARM.finger_box_corners(Q, 0.03)).reshape(-1, 8, 3).max(axis=1), highs,
                               atol=1e-9)


@pytest.mark.parametrize('width', [-0.001, np.nan, np.inf])
def test_an_opening_that_is_not_a_width_is_refused(width):
    with pytest.raises(ValueError, match='opening'):
        ARM.finger_box_corners(Q, width)


def test_a_gripper_whose_stroke_is_not_known_keeps_the_whole_stroke_boxes():
    from tasks.dual_arm_crag_skull.s0.kinematics import robotiq_proxy_model
    proxy = robotiq_proxy_model().arms['left']
    np.testing.assert_array_equal(proxy.finger_box_corners(Q, 0.01), proxy.finger_box_corners(Q))
    np.testing.assert_array_equal(proxy.tool_box_corners(Q, 0.01), proxy.tool_box_corners(Q))


def test_the_stroke_must_span_the_whole_stroke_box():
    from tasks.dual_arm_crag_skull.s0.kinematics import ArmModel, load_kinematics
    flange_T_tcp, capsule, box = yam_tool_geometry()
    stroke = box['parts']['fingers']['stroke']
    ArmModel('left', load_kinematics(), np.eye(4), flange_T_tcp, capsule, box)
    left = stroke['tips']['tip_left']
    for change in ({'travel_m': stroke['travel_m'] - 0.002}, {'closed_gap_m': np.nan},
                   {'tips': {**stroke['tips'], 'tip_left': {**left, 'slide_axis': [0, 0, 2]}}},
                   {'tips': {**stroke['tips'], 'tip_left': {**left, 'slide_axis': [0, 0, 1]}}}):
        broken = {**box, 'parts': {**box['parts'], 'fingers': {**box['parts']['fingers'],
                                                               'stroke': {**stroke, **change}}}}
        with pytest.raises(ValueError, match='stroke'):
            ArmModel('left', load_kinematics(), np.eye(4), flange_T_tcp, capsule, broken)
