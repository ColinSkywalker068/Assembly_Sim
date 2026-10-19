import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.robots.kinematics import T, quat_to_R
from tasks.dual_arm_crag_skull.s0.collision import fit_capsule, segment_distances
from tasks.dual_arm_crag_skull.s0.kinematics import (
    CAPSULE_LINKS, ROBOTIQ_PROXY_FLANGE_T_TCP, ArmModel, DualArmModel, load_kinematics, robotiq_proxy_model)

ROOT = Path(__file__).resolve().parents[3]
DATA = load_kinematics()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_kinematic_model_matches_the_repo_assets():
    # Regenerate configs/crx10ial_kinematics.json with s0/export_kinematics.py if this fails.
    provenance = DATA['provenance']
    assert sha256(ROOT / provenance['arm_usd']) == provenance['arm_usd_sha256']
    assert sha256(ROOT / provenance['gripper_usd']) == provenance['gripper_usd_sha256']
    assert provenance['probe_vs_usd_joint_frame_max_abs_diff'] == 0.0
    assert DATA['max_joint_acceleration_rad_s2'] is None


def arm_at_origin():
    return ArmModel('left', DATA, np.eye(4), np.eye(4), None)


def test_fk_reproduces_isaac_link_poses():
    arm = arm_at_origin()
    links = arm.frames(np.array(DATA['fk_check']['q']))[0]
    worst_position = worst_rotation = 0.0
    for name in [*(f'link_{i}' for i in range(1, 7)), 'flange']:
        reference = DATA['fk_check']['links'][f'/World/arm/{name}']
        worst_position = max(worst_position, np.linalg.norm(links[name][:3, 3] - reference['pos']))
        relative = quat_to_R(reference['quat_wxyz']).T @ links[name][:3, :3]
        worst_rotation = max(worst_rotation, Rotation.from_matrix(relative).magnitude())
    # The probe rounds positions and quaternions to 1e-4.
    assert worst_position < 2e-4 and worst_rotation < 5e-4, (worst_position, worst_rotation)


def tcp_vector(arm, q):
    pose = arm.tcp(q)
    return pose[:3, 3], pose[:3, :3]


@pytest.mark.parametrize('seed', range(5))
def test_geometric_jacobian_matches_finite_differences(seed):
    rng = np.random.default_rng(seed)
    arm = ArmModel('left', DATA, T([-.78, 0, .75], [1, 0, 0, 0]), ROBOTIQ_PROXY_FLANGE_T_TCP, None)
    q = rng.uniform(arm.lower * .8, arm.upper * .8)
    position, rotation = tcp_vector(arm, q)
    numeric = np.zeros((6, 6))
    step = 1e-6
    for joint in range(6):
        moved = q.copy()
        moved[joint] += step
        p, r = tcp_vector(arm, moved)
        numeric[:3, joint] = (p - position) / step
        numeric[3:, joint] = Rotation.from_matrix(r @ rotation.T).as_rotvec() / step
    np.testing.assert_allclose(arm.jacobian(q), numeric, atol=2e-5)


def test_right_arm_is_the_left_arm_moved_by_the_preset_bases():
    model = robotiq_proxy_model()
    left, right = model.arms['left'], model.arms['right']
    q = np.array([.2, -.3, -.5, .1, -.9, .4])
    expected = right.world_T_base @ np.linalg.inv(left.world_T_base) @ left.tcp(q)
    np.testing.assert_allclose(right.tcp(q), expected, atol=1e-12)
    np.testing.assert_allclose(left.world_T_base[:3, 3], [-.78, 0, .75])
    np.testing.assert_allclose(right.world_T_base[:3, :3], np.diag([-1., -1., 1.]), atol=1e-12)


def test_tcp_and_tool_geometry_are_never_defaulted():
    with pytest.raises(ValueError, match='both arms'):
        DualArmModel({'left': np.eye(4)}, {'left': None, 'right': None}, data=DATA, tool_box={'left': None, 'right': None})
    with pytest.raises(TypeError):  # the tool box is a required keyword, not a hidden default
        DualArmModel({'left': np.eye(4), 'right': np.eye(4)}, {'left': None, 'right': None}, data=DATA)
    with pytest.raises(ValueError):
        ArmModel('left', DATA, np.eye(4), np.diag([1., 1., 1., 2.]), None)
    with pytest.raises(ValueError, match='no tool box'):
        ArmModel('left', DATA, np.eye(4), np.eye(4), None).tool_box_corners(np.zeros(6))
    with pytest.raises(ValueError, match='min < max'):
        ArmModel('left', DATA, np.eye(4), np.eye(4), None, {'min': [0, 0, 0], 'max': [0.1, 0, 0.1]})


def test_tool_box_corners_follow_the_flange():
    arm = robotiq_proxy_model().arms['left']
    q = np.array([0.3, -0.4, 0.5, 0.2, 0.6, 0.1])
    flange = arm.frames(q)[0]['flange']
    local = (np.linalg.inv(flange) @ np.c_[arm.tool_box_corners(q), np.ones(8)].T).T[:, :3]
    box = DATA['robotiq_proxy_tool_box']
    np.testing.assert_allclose(local.min(axis=0), box['min'], atol=1e-12)
    np.testing.assert_allclose(local.max(axis=0), box['max'], atol=1e-12)
    # Closing the fingers pushes the pads forward: the swept box is longer than the open pose.
    assert box['max'][0] > box['open_pose_max'][0] + 0.01


def test_capsule_set_has_every_link_and_the_tool():
    capsules = robotiq_proxy_model().arms['right'].capsules(np.zeros(6))
    assert capsules.names == tuple(f'right:{name}' for name in (*CAPSULE_LINKS, 'tool'))
    assert (capsules.radius > .03).all() and (capsules.radius < .2).all()


def test_fit_capsule_encloses_points_and_handles_short_sets():
    rng = np.random.default_rng(3)
    for scale in ([.4, .05, .08], [.05, .05, .05]):
        points = rng.uniform(-1, 1, size=(4000, 3)) * scale + [.1, -.2, .3]
        points = points @ Rotation.random(random_state=rng).as_matrix().T
        p0, p1, radius = fit_capsule(points)
        assert (segment_distances(points, points, p0, p1) <= radius + 1e-12).all()  # point-to-axis distance
    p0, p1, radius = fit_capsule(rng.normal(size=(500, 3)) * .01)
    assert radius < .05 and np.linalg.norm(p1 - p0) < 3 * radius


def brute_force(p0, p1, q0, q1):
    s = np.linspace(0, 1, 401)
    a = p0 + s[:, None] * (p1 - p0)
    b = q0 + s[:, None] * (q1 - q0)
    return np.linalg.norm(a[:, None] - b[None], axis=-1).min()


def test_segment_distance_matches_brute_force_including_degenerate_cases():
    rng = np.random.default_rng(7)
    cases = [rng.normal(size=(4, 3)) for _ in range(40)]
    point = rng.normal(size=3)
    cases += [np.array([point, point, *rng.normal(size=(2, 3))]), np.array([*rng.normal(size=(2, 3)), point, point]),
              np.array([point, point, point + .3, point + .3]),
              np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.]])]
    for p0, p1, q0, q1 in cases:
        assert abs(segment_distances(p0, p1, q0, q1) - brute_force(p0, p1, q0, q1)) < 5e-3


def test_capsules_enclose_the_usd_collision_meshes():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    from tasks.dual_arm_crag_skull.s0.export_kinematics import _mesh_points
    stage = Usd.Stage.Open(str(ROOT / DATA['provenance']['arm_usd']))
    root = stage.GetDefaultPrim().GetPath()
    for link in CAPSULE_LINKS:
        frame = stage.GetPrimAtPath(f'{root}/{link}')
        points = _mesh_points(stage, stage.GetPrimAtPath(f'{root}/{link}/collisions'), frame)
        record = DATA['link_capsules'][link]
        distance = segment_distances(points, points, np.array(record['p0']), np.array(record['p1']))
        assert distance.max() <= record['radius'] + 1e-9, link
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0


def test_tool_box_encloses_the_gripper_meshes_over_the_stroke():
    pytest.importorskip('pxr')
    import json
    from pxr import Usd, UsdPhysics
    from core.robots.kinematics import Gripper
    from tasks.dual_arm_crag_skull.s0.export_kinematics import GRIPPER_ROOT, _mesh_points
    probe = Path('/local_data/jz6676/data/fanuc_crag_skull_assembly/04_intermediate/probe/fanuc_probe.json')
    if not probe.exists():
        pytest.skip('probe dump not available on this machine')
    stage = Usd.Stage.Open(str(ROOT / DATA['provenance']['gripper_usd']))
    gripper = Gripper(json.loads(probe.read_text()))
    box = DATA['robotiq_proxy_tool_box']
    for angle in (0.005, 0.135, 0.405, 0.795):  # between the export's 0.01 rad samples on purpose
        for prim in Usd.PrimRange(stage.GetPrimAtPath(GRIPPER_ROOT)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                link = str(prim.GetPath()).split('/')[3]
                points = _mesh_points(stage, prim, stage.GetPrimAtPath(f'{GRIPPER_ROOT}/{link}'))
                transform = gripper.link_T(link, angle)
                moved = points @ transform[:3, :3].T + transform[:3, 3]
                assert (moved >= np.array(box['min']) - 1e-5).all() and (moved <= np.array(box['max']) + 1e-5).all(), (link, angle)
