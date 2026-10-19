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
