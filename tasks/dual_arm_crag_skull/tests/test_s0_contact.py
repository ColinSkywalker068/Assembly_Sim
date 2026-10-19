"""S0 contact phase: a chunk declares where an arm will work near the table; only there, only its fingers and
only slowly may that arm come closer to the table than the normal tool clearance.

Hand-checked geometry of the Robotiq 2F-85 proxy (configs/crx10ial_kinematics.json): the fingers reach
x = 0.16283 m in the flange frame over the whole stroke, 27.8 mm past the TCP at x = 0.135 m, and their box is
+-0.0762 m by +-0.0175 m across. Pointing straight down with the TCP 33 mm above the table, the fingertips are
5.2 mm above it (below the 10 mm tool clearance of the test limits, above the 2 mm contact clearance), and each
bottom corner of the finger box is hypot(0.0762, 0.0175) = 78.2 mm from the TCP axis.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.robots.kinematics import T
from tasks.dual_arm_crag_skull.contracts.actions import ARMS, ActionRejected, ContactWindow, encode_chunk
from tasks.dual_arm_crag_skull.contracts.skills import ContactMemory, SkillRequest, contact_windows
from tasks.dual_arm_crag_skull.s0 import adapter as adapter_module
from tasks.dual_arm_crag_skull.s0.adapter import S0Adapter
from tasks.dual_arm_crag_skull.s0.ik import IKOptions, solve_ik
from tasks.dual_arm_crag_skull.s0.kinematics import (
    ROBOTIQ_PROXY_FLANGE_T_TCP, ArmModel, DualArmModel, load_kinematics)
from tasks.dual_arm_crag_skull.tests.test_s0_adapter import HOME, MODEL, expect_rejection, limits, plan, state

DATA = load_kinematics()
TABLE = 0.75  # table_top_z_m of the test limits
LOW = TABLE + 0.033
CORNER = 0.0782  # hypot(0.0762, 0.0175): a bottom finger-box corner's distance from the TCP axis, pointing down
CONTACT = dict(contact_table_clearance_m=0.002, contact_region_radius_m=0.15, contact_max_speed_m_s=0.05)
FAST = dict(CONTACT, max_joint_acceleration_rad_s2=np.full(6, 200.0))  # direct paths: keep the dynamics out of it
DT = 1 / 60  # servo period of the test limits
DOWN = Rotation.from_euler('y', 90, degrees=True).as_matrix()  # tool +x, the approach axis, points at the table
# The gripper on its side: tool x along world +x, tool y down. SIDE_SEED picks the IK branch that reaches it.
SIDEWAYS = np.column_stack([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
SIDE_SEED = np.array([0.651, 1.598, -1.555, 1.556, -2.491, -3.161])
HOLD = np.zeros((4, 3))
LEFT = MODEL.arms['left']


def pose_at(translation):
    pose = np.eye(4)
    pose[:3, 3] = translation
    return pose


def tcp_joints(position, rotation=DOWN, seed=HOME, arm='left', model=MODEL, **ik):
    target = np.eye(4)
    target[:3, :3], target[:3, 3] = rotation, position
    result = solve_ik(model.arms[arm], target, seed, IKOptions(**{'max_seed_jump_rad': 3.0, **ik}))
    assert result.ok, result
    return result.q


Q_LOW = tcp_joints([-.42, 0, LOW])  # fingertips 5.2 mm above the table: needs a contact


def finger_low(q, arm='left'):
    return MODEL.arms[arm].finger_box_corners(q)[:, 2].min() - TABLE


def window(arm='left', kind='grasp', center=(-.42, 0.0), fragment_id='frag_a'):
    return ContactWindow(arm=arm, kind=kind, center_xy=np.array(center, dtype=float), fragment_id=fragment_id)


def chunk_at(q_now, widths_now, moves=HOLD, contacts=(), model=MODEL, chunk_id=3):
    """The left arm translates by `moves` (one waypoint per 0.1 s) from its current TCP; the right arm holds."""
    anchors = {arm: model.arms[arm].tcp(q_now[arm]) for arm in ARMS}
    targets = {arm: np.repeat(anchors[arm][None], len(moves), axis=0) for arm in ARMS}
    targets['left'][:, :3, 3] += moves
    return encode_chunk(request_id='req', chunk_id=chunk_id, observation_time=1.0, valid_until=1.6, anchors=anchors,
                        waypoint_times=1.1 + .1 * np.arange(len(moves)), targets=targets,
                        widths={arm: np.full(len(moves), widths_now[arm]) for arm in ARMS}, active_arms=('left',),
                        contacts=contacts)


def direct(left, *, start=Q_LOW, dt=DT, contacts=(window(),), right=None, right_start=HOME, now=1.0, **overrides):
    """check_joint_path: left-arm joint targets one per `dt` from the measured `start` at `now`; the right arm
    holds `right_start` unless it gets its own targets."""
    left = np.asarray(left)
    right = np.repeat(right_start[None], len(left), axis=0) if right is None else right
    adapter_module.check_joint_path(MODEL, limits(**{**FAST, **overrides}), now=now,
                                    q_now={'left': start, 'right': right_start},
                                    qd_now={arm: np.zeros(6) for arm in ARMS}, times=now + dt * np.arange(1, len(left) + 1),
                                    joints={'left': left, 'right': right}, contacts=contacts)


def proxy_model(tool_box, flange_T_tcp=ROBOTIQ_PROXY_FLANGE_T_TCP):
    return DualArmModel({arm: flange_T_tcp for arm in ARMS}, {arm: DATA['robotiq_proxy_capsule'] for arm in ARMS},
                        data=DATA, tool_box={arm: tool_box for arm in ARMS})


# ---------------------------------------------------------------- the declaration (contracts)

@pytest.mark.parametrize('overrides, message', [
    ({'arm': 'middle'}, 'arm'),
    ({'kind': 'push'}, 'kind'),
    ({'kind': 'retreat'}, 'kind'),  # a retreat keeps the previous grasp or place window instead
    ({'center': (.3, -.2, .8)}, 'center_xy'),
    ({'center': (np.nan, 0.0)}, 'center_xy'),
    ({'fragment_id': None}, 'fragment_id'),
    ({'fragment_id': ''}, 'fragment_id'),
])
def test_malformed_contact_declarations_are_rejected(overrides, message):
    with pytest.raises(ActionRejected, match=message):
        window(**overrides)


def test_a_contact_keeps_its_own_copy_of_the_centre():
    centre = np.array([.3, -.2])
    place = ContactWindow(arm='right', kind='place', center_xy=centre, fragment_id='frag_b')
    centre[0] = 9.0
    assert place.center_xy == (0.3, -0.2)


def test_a_chunk_declares_at_most_one_contact_per_arm():
    q_now, widths_now = state()
    assert chunk_at(q_now, widths_now).contacts == ()
    both = chunk_at(q_now, widths_now, contacts=[window(), window(arm='right', kind='place', center=(.42, 0))])
    assert [w.arm for w in both.contacts] == ['left', 'right'] and isinstance(both.contacts, tuple)
    with pytest.raises(ActionRejected, match='one contact per arm'):
        chunk_at(q_now, widths_now, contacts=(window(), window(center=(0, 0))))
    with pytest.raises(ActionRejected, match='ContactWindow'):
        chunk_at(q_now, widths_now, contacts=({'arm': 'left'},))


def skill(skill_id, arms=('left',), **fields):
    return SkillRequest(request_id='s-1', observation_id='obs', skill_id=skill_id, active_arms=arms,
                        completion_predicate={'type': 'observed'}, timeout_s=10.0, **fields)


def test_contact_windows_follow_the_skill():
    (grasp,) = contact_windows(skill('pick', fragment_id='frag_a', fragment_pose=pose_at([.3, -.2, .78])))
    assert (grasp.arm, grasp.kind, grasp.fragment_id) == ('left', 'grasp', 'frag_a')
    np.testing.assert_allclose(grasp.center_xy, [.3, -.2])
    (support,) = contact_windows(skill('support', arms=('right',), fragment_id='frag_b',
                                       fragment_pose=pose_at([.2, .1, .78])))
    assert (support.arm, support.kind) == ('right', 'grasp')
    (place,) = contact_windows(skill('place', fragment_id='frag_a', target_pose=pose_at([.25, .05, .8])))
    assert (place.kind, place.fragment_id) == ('place', 'frag_a')
    np.testing.assert_allclose(place.center_xy, [.25, .05])
    (align,) = contact_windows(skill('align', fragment_id='frag_a', target_pose=pose_at([.25, .05, .8]),
                                     fragment_pose=pose_at([.3, -.2, .78])))
    assert align.kind == 'place'
    np.testing.assert_allclose(align.center_xy, [.25, .05])
    insert = contact_windows(skill('insert', arms=('left', 'right'), fragment_id='frag_a',
                                   target_pose=pose_at([.25, .05, .8])))
    assert [(w.arm, w.kind) for w in insert] == [('left', 'place'), ('right', 'place')]
    assert contact_windows(skill('reobserve', arms=())) == ()
    with pytest.raises(ValueError, match='fragment_pose'):
        contact_windows(skill('pick', fragment_id='frag_a'))


def test_a_retreat_keeps_each_arms_previous_grasp_or_place_window():
    previous = {'left': window(kind='place', center=(.25, .05)),
                'right': window(arm='right', center=(-.3, .1), fragment_id='frag_b')}
    retreat = skill('retreat', arms=('right', 'left'))
    assert contact_windows(retreat, previous=previous) == (previous['left'], previous['right'])
    assert contact_windows(skill('retreat'), previous={'right': previous['right']}) == ()  # left never had one
    assert contact_windows(skill('retreat')) == ()
    with pytest.raises(ValueError, match='left'):
        contact_windows(skill('retreat'), previous={'left': previous['right']})  # filed under the wrong arm


def test_a_retreat_keeps_only_windows_of_chunks_s0_accepted():
    q_now, widths_now = state()
    memory = ContactMemory()
    pick_left = window(center=(-.4, 0))
    assert memory.windows(skill('retreat')) == ()  # declared or not, nothing was accepted yet
    memory.accepted(chunk_at(q_now, widths_now, contacts=(pick_left,)))
    memory.accepted(chunk_at(q_now, widths_now, contacts=(window(arm='right', center=(.4, 0)),)))
    assert memory.windows(skill('retreat')) == (pick_left,)  # the right arm's chunk did not drop the left window
    insert = (window(kind='place', center=(0, .2)), window(arm='right', kind='place', center=(0, .2)))
    memory.accepted(chunk_at(q_now, widths_now, contacts=insert))
    assert memory.windows(skill('retreat', arms=('left', 'right'))) == insert  # each arm gets its own
    assert memory.windows(skill('pick', fragment_id='frag_c', fragment_pose=pose_at([.3, 0, .78])))[0].center_xy == (.3, 0)
    memory.reset()  # a new episode: an old window may now lie under another fragment
    assert memory.windows(skill('retreat', arms=('left', 'right'))) == ()


# ---------------------------------------------------------------- gripper geometry (kinematics)

def test_tool_box_splits_into_gripper_body_and_fingers():
    q = np.array([0.3, -0.4, 0.5, 0.2, 0.6, 0.1])
    flange = LEFT.frames(q)[0]['flange']

    def local(corners):
        return (np.linalg.inv(flange) @ np.c_[corners, np.ones(8)].T).T[:, :3]

    body, fingers, tool = (local(f(q)) for f in (LEFT.body_box_corners, LEFT.finger_box_corners, LEFT.tool_box_corners))
    # From the 2F-85 collision meshes over finger angles 0..0.8 rad: the body (base_link) ends at x = 89.8 mm and is
    # 75 mm thick; the fingers reach 162.8 mm and are 35 mm thick.
    assert body[:, 0].max() == pytest.approx(0.0898, abs=1e-4) and np.ptp(body[:, 2]) == pytest.approx(0.075, abs=1e-4)
    assert fingers[:, 0].max() == pytest.approx(0.16283, abs=1e-5)
    assert np.ptp(fingers[:, 2]) == pytest.approx(0.035, abs=1e-4)
    np.testing.assert_allclose(np.minimum(body.min(axis=0), fingers.min(axis=0)), tool.min(axis=0), atol=1e-9)
    np.testing.assert_allclose(np.maximum(body.max(axis=0), fingers.max(axis=0)), tool.max(axis=0), atol=1e-9)


def test_tool_parts_must_be_complete_and_span_the_tool_box():
    box = DATA['robotiq_proxy_tool_box']
    parts = box['parts']

    def make(record):
        return ArmModel('left', DATA, np.eye(4), np.eye(4), None, record)

    with pytest.raises(ValueError, match='body and fingers'):
        make({**box, 'parts': {'body': parts['body']}})
    # Each change moves one face of the union away from the tool box: the fingertips (max x), the fingers' outer
    # side (min y) and the body's back at the flange (min x).
    for part, side, axis, value in (('fingers', 'max', 0, 0.15), ('fingers', 'min', 1, -0.07), ('body', 'min', 0, 0.0)):
        bound = list(parts[part][side])
        bound[axis] = value
        with pytest.raises(ValueError, match='tool box'):
            make({**box, 'parts': {**parts, part: {**parts[part], side: bound}}})
    assert make(box).has_tool_parts
    without = make({key: value for key, value in box.items() if key != 'parts'})
    assert not without.has_tool_parts
    with pytest.raises(ValueError, match='finger geometry'):
        without.finger_box_corners(np.zeros(6))


def test_tool_parts_enclose_the_gripper_meshes_over_the_stroke():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdPhysics
    from core.robots.kinematics import Gripper
    from tasks.dual_arm_crag_skull.s0.export_kinematics import GRIPPER_ROOT, _mesh_points
    probe = Path('/local_data/jz6676/data/fanuc_crag_skull_assembly/04_intermediate/probe/fanuc_probe.json')
    if not probe.exists():
        pytest.skip('probe dump not available on this machine')
    root = Path(__file__).resolve().parents[3]
    stage = Usd.Stage.Open(str(root / DATA['provenance']['gripper_usd']))
    gripper = Gripper(json.loads(probe.read_text()))
    parts = DATA['robotiq_proxy_tool_box']['parts']
    for angle in (0.005, 0.135, 0.405, 0.795):  # between the export's samples on purpose
        for prim in Usd.PrimRange(stage.GetPrimAtPath(GRIPPER_ROOT)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                link = str(prim.GetPath()).split('/')[3]
                points = _mesh_points(stage, prim, stage.GetPrimAtPath(f'{GRIPPER_ROOT}/{link}'))
                transform = gripper.link_T(link, angle)
                moved = points @ transform[:3, :3].T + transform[:3, 3]
                part = parts['body' if link == 'base_link' else 'fingers']
                assert (moved >= np.array(part['min']) - 1e-5).all() and (moved <= np.array(part['max']) + 1e-5).all(), (
                    link, angle)


# ---------------------------------------------------------------- limits

def test_contact_parameters_are_all_or_nothing_and_tighter_than_normal():
    assert limits(**CONTACT).contact_enabled and not limits().contact_enabled
    for overrides in ({'contact_table_clearance_m': 0.002},
                      {**CONTACT, 'contact_table_clearance_m': 0.02},  # looser than the 0.01 m table clearance
                      {**CONTACT, 'contact_table_clearance_m': 0.0},  # touching the table under position control
                      {**CONTACT, 'contact_table_clearance_m': -0.001},
                      {**CONTACT, 'contact_region_radius_m': 0.0},
                      {**CONTACT, 'contact_max_speed_m_s': 0.0},
                      {**CONTACT, 'contact_max_speed_m_s': -0.05},
                      {**CONTACT, 'contact_max_speed_m_s': np.inf}):
        with pytest.raises(ValueError):
            limits(**overrides)


# ---------------------------------------------------------------- the rule on chunks (S0Adapter.plan)

def test_holding_close_to_the_table_needs_a_declared_contact():
    q_now, widths_now = state(left=Q_LOW)
    expect_rejection(chunk_at(q_now, widths_now), q_now, widths_now, 'no contact declared for left', **CONTACT)
    status, trajectory = plan(chunk_at(q_now, widths_now, contacts=(window(),)), q_now, widths_now, **CONTACT)
    assert status.status == 'accepted' and not trajectory.hold, status.reason


def test_a_contact_still_keeps_the_fingertips_above_the_contact_clearance():
    q_now, widths_now = state(left=tcp_joints([-.42, 0, TABLE + .029]))  # fingertips 1.2 mm above the table
    expect_rejection(chunk_at(q_now, widths_now, contacts=(window(),)), q_now, widths_now, 'left:fingers are',
                     **CONTACT)


def test_a_contact_applies_only_near_its_centre():
    q_now, widths_now = state(left=Q_LOW)  # the bottom finger corners reach 0.10 + 0.078 m from this centre
    expect_rejection(chunk_at(q_now, widths_now, contacts=(window(center=(-.42, .10)),)), q_now, widths_now,
                     'contact centre', **CONTACT)


def test_the_region_holds_every_low_finger_corner_not_only_the_tcp():
    # The gripper on its side, TCP 82 mm above the table: the fingers' box bottom is 5.8 mm above it and runs from
    # 90.4 mm behind the TCP to 27.8 mm ahead of it.
    q_side = tcp_joints([-.35, .10, TABLE + .082], rotation=SIDEWAYS, seed=SIDE_SEED)
    q_now, widths_now = state(left=q_side)
    status, _ = plan(chunk_at(q_now, widths_now, contacts=(window(center=(-.35, .10)),)), q_now, widths_now, **CONTACT)
    assert status.status == 'accepted', status.reason
    # 70 mm ahead of the TCP: the TCP is inside the 0.15 m radius, the trailing corners 0.161 m away are not.
    expect_rejection(chunk_at(q_now, widths_now, contacts=(window(center=(-.28, .10)),)), q_now, widths_now,
                     'contact centre', **CONTACT)


def test_the_part_below_the_floor_is_spanned_by_its_vertices_for_any_box_pose():
    # Against a dense grid of the finger box: no point below the floor is farther from a centre than the farthest
    # vertex `_below` returns (corners below the floor and edge crossings, whichever edges cross).
    low, high = np.array([0.0446, -0.0762, -0.0175]), np.array([0.1628, 0.0762, 0.0175])
    local = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    grid = np.array(np.meshgrid(*[np.linspace(low[i], high[i], 25) for i in range(3)], indexing='ij')).reshape(3, -1).T
    rng = np.random.default_rng(3)
    for _ in range(60):
        rotation = Rotation.random(random_state=rng).as_matrix()
        corners, points = local @ rotation.T, grid @ rotation.T
        floor = rng.uniform(corners[:, 2].min(), corners[:, 2].max())
        centre = rng.normal(0, 0.05, 2)
        vertices = adapter_module._below(corners, floor)
        inside = points[points[:, 2] < floor]
        reach = np.hypot(*(vertices[:, :2] - centre).T).max() if len(vertices) else 0.0
        assert (np.hypot(*(inside[:, :2] - centre).T) <= reach + 1e-12).all()


def test_the_region_holds_all_of_the_finger_box_below_the_floor_not_only_its_corners():
    # The sideways gripper tilted 3.6 deg, TCP 80.5 mm up: two bottom corners are 2.7 mm above the table and the two
    # trailing ones 10.1 mm, just above the 10 mm floor. The bottom face crosses the floor about 88.7 mm behind the
    # TCP, so from a centre 80 mm ahead of the TCP the box reaches 0.170 m below the floor; its low corners only 0.08.
    tilted = Rotation.from_rotvec(np.radians(3.6) * np.array([0, 1.0, 0])).as_matrix() @ SIDEWAYS
    q = tcp_joints([-.35, .10, TABLE + .0805], rotation=tilted, seed=SIDE_SEED, max_seed_jump_rad=1.0)
    direct([q] * 3, start=q, contacts=(window(center=(-.35, .10)),))
    with pytest.raises(adapter_module.Rejected, match='contact centre'):
        direct([q] * 3, start=q, contacts=(window(center=(-.27, .10)),))


def test_a_contact_applies_only_to_its_own_arm():
    q_now, widths_now = state(left=Q_LOW)
    tcp = LEFT.tcp(Q_LOW)[:2, 3]  # the right arm's contact sits exactly where the left arm works
    expect_rejection(chunk_at(q_now, widths_now, contacts=(window(arm='right', center=tuple(tcp)),)), q_now,
                     widths_now, 'no contact declared for left', **CONTACT)


def test_the_gripper_body_keeps_the_normal_clearance_in_a_contact():
    # A gripper whose body reaches x = 0.16 m, past the TCP: pointing down 33 mm above the table its body is 8 mm
    # above it (needs the 10 mm tool clearance) while the fingertips are 5.2 mm above it (contact clearance 2 mm).
    box = DATA['robotiq_proxy_tool_box']
    parts = {**box['parts'], 'body': {**box['parts']['body'], 'max': [0.16, *box['parts']['body']['max'][1:]]}}
    model = proxy_model({**box, 'parts': parts})
    q_now, widths_now = state(left=Q_LOW)
    status, _ = plan(chunk_at(q_now, widths_now, contacts=(window(),), model=model), q_now, widths_now,
                     adapter=S0Adapter(model, limits(**CONTACT)))
    assert status.status == 'rejected' and 'left:gripper body' in status.reason, status.reason


def test_a_contact_replaces_the_tcp_height_rule_but_keeps_the_tcp_off_the_table():
    q_now, widths_now = state(left=tcp_joints([-.42, 0, TABLE + .05]))  # tool box 22 mm above the table
    expect_rejection(chunk_at(q_now, widths_now), q_now, widths_now, 'left TCP', tcp_min_height_m=.10, **CONTACT)
    status, _ = plan(chunk_at(q_now, widths_now, contacts=(window(),)), q_now, widths_now, tcp_min_height_m=.10,
                     **CONTACT)
    assert status.status == 'accepted', status.reason
    # No finger corner is low here, so the TCP alone must be inside the region.
    expect_rejection(chunk_at(q_now, widths_now, contacts=(window(center=(-.42, .2)),)), q_now, widths_now,
                     'contact centre', tcp_min_height_m=.10, **CONTACT)
    # A TCP defined 35 mm further out than the proxy's, past the fingertips: with the flange 3 mm higher than for Q_LOW
    # the fingertips are 8.2 mm above the table and this TCP 1.0 mm, short of the 2 mm contact clearance.
    model = proxy_model(DATA['robotiq_proxy_tool_box'], flange_T_tcp=T([0.17, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))
    q_now, widths_now = state(left=tcp_joints([-.42, 0, LOW + .003]))
    status, _ = plan(chunk_at(q_now, widths_now, contacts=(window(),), model=model), q_now, widths_now,
                     adapter=S0Adapter(model, limits(**CONTACT)))
    assert status.status == 'rejected' and 'left TCP' in status.reason, status.reason


def test_fingers_near_the_table_must_move_slowly():
    q_now, widths_now = state(left=Q_LOW)
    slide = np.array([[.01 * k, 0, 0] for k in range(1, 5)])  # 0.1 m/s along the table, inside the region
    chunk = chunk_at(q_now, widths_now, moves=slide, contacts=(window(),))
    status, _ = plan(chunk, q_now, widths_now, **CONTACT)
    assert status.status == 'rejected' and 'left:fingers move at' in status.reason, status.reason
    status, _ = plan(chunk, q_now, widths_now, **{**CONTACT, 'contact_max_speed_m_s': 1.0})
    assert status.status == 'accepted', status.reason


def test_a_slow_descent_into_a_contact_needs_the_declaration():
    q_now, widths_now = state(left=tcp_joints([-.42, 0, TABLE + .040]))  # tool box 12.2 mm up: the normal rule holds
    descent = np.array([[0, 0, -.0017 * k] for k in range(1, 5)])  # 6.8 mm down at 17 mm/s, ending 5.4 mm up
    expect_rejection(chunk_at(q_now, widths_now, moves=descent), q_now, widths_now, 'left:tool box', **CONTACT)
    status, _ = plan(chunk_at(q_now, widths_now, moves=descent, contacts=(window(),)), q_now, widths_now, **CONTACT)
    assert status.status == 'accepted', status.reason


def test_a_contact_cannot_move_or_change_kind_within_one_request():
    q_now, widths_now = state(left=Q_LOW)
    adapter = S0Adapter(MODEL, limits(**CONTACT))
    status, _ = plan(chunk_at(q_now, widths_now, contacts=(window(),)), q_now, widths_now, adapter=adapter)
    assert status.status == 'accepted', status.reason
    for chunk_id, changed in ((4, window(center=(-.40, 0))), (5, window(kind='place'))):
        status, _ = plan(chunk_at(q_now, widths_now, contacts=(changed,), chunk_id=chunk_id), q_now, widths_now,
                         adapter=adapter)
        assert status.status == 'rejected' and 'moved within request' in status.reason, status.reason
    status, _ = plan(chunk_at(q_now, widths_now, contacts=(window(),), chunk_id=6), q_now, widths_now, adapter=adapter)
    assert status.status == 'accepted', status.reason


def test_each_arms_contact_stays_fixed_on_its_own():
    q_right = tcp_joints([.42, 0, LOW], arm='right', max_seed_jump_rad=7.0, max_iterations=300)
    q_now, widths_now = state(left=Q_LOW, right=q_right)
    contacts = (window(), window(arm='right', center=tuple(MODEL.arms['right'].tcp(q_right)[:2, 3])))
    adapter = S0Adapter(MODEL, limits(**CONTACT))
    for chunk_id in (3, 4):
        status, _ = plan(chunk_at(q_now, widths_now, contacts=contacts, chunk_id=chunk_id), q_now, widths_now,
                         adapter=adapter)
        assert status.status == 'accepted', status.reason


def test_a_forged_declaration_brakes_instead_of_escaping():
    forged = object.__new__(ContactWindow)  # skips the dataclass validation
    for name, value in (('arm', 'middle'), ('kind', 'grasp'), ('center_xy', (-.42, 0.0)), ('fragment_id', 'frag_a')):
        object.__setattr__(forged, name, value)
    q_now, widths_now = state()  # high above the table: every sample passes, so only the declaration is wrong
    expect_rejection(chunk_at(q_now, widths_now, contacts=(forged,)), q_now, widths_now, 'contact', **CONTACT)
    object.__setattr__(forged, 'arm', 'left')
    object.__setattr__(forged, 'kind', 'push')
    with pytest.raises(adapter_module.Rejected, match='invalid contact'):  # a direct caller catches Rejected only
        direct([Q_LOW] * 3, contacts=(forged,))


def test_declarations_are_consulted_only_where_the_normal_rule_fails():
    q_high, widths_now = state(left=tcp_joints([-.42, 0, TABLE + .10]))
    status, _ = plan(chunk_at(q_high, widths_now, contacts=(window(),)), q_high, widths_now)  # no contact parameters
    assert status.status == 'accepted', status.reason
    q_low, _ = state(left=Q_LOW)
    expect_rejection(chunk_at(q_low, widths_now, contacts=(window(),)), q_low, widths_now,
                     'contact phase is not configured')


def test_a_contact_needs_the_finger_geometry():
    model = proxy_model({key: value for key, value in DATA['robotiq_proxy_tool_box'].items() if key != 'parts'})
    q_now, widths_now = state(left=Q_LOW)
    status, _ = plan(chunk_at(q_now, widths_now, contacts=(window(),), model=model), q_now, widths_now,
                     adapter=S0Adapter(model, limits(**CONTACT)))
    assert status.status == 'rejected' and 'needs the finger geometry' in status.reason, status.reason


# ---------------------------------------------------------------- the rule on direct joint targets (check_joint_path)

def test_direct_joint_targets_get_the_same_contact_rule():
    with pytest.raises(adapter_module.Rejected, match='no contact declared for left'):
        direct([Q_LOW] * 4, contacts=())
    direct([Q_LOW] * 4)


def test_contact_samples_must_be_servo_rate():
    for dt in (.1, 1.5 * DT, 1.1 * DT):  # the path between targets is not checked
        with pytest.raises(adapter_module.Rejected, match='apart'):
            direct([Q_LOW] * 4, dt=dt)
    direct([Q_LOW] * 4, dt=.0167)  # a period rounded to 0.1 ms is still one period
    direct([Q_LOW] * 4, now=2e7)  # a clock 230 days in: its float steps differ from 1/60 s by ~1e-9 s
    up = tcp_joints([-.42, 0, LOW + .008], seed=Q_LOW)
    with pytest.raises(adapter_module.Rejected, match='apart'):  # the interval leaving the contact counts as well
        adapter_module.check_joint_path(MODEL, limits(**FAST), now=1.0, q_now={'left': Q_LOW, 'right': HOME},
                                        qd_now={arm: np.zeros(6) for arm in ARMS}, times=1.0 + np.array([DT, DT + 1.0]),
                                        joints={'left': np.array([Q_LOW, up]), 'right': np.array([HOME, HOME])},
                                        contacts=(window(),))


def test_entering_the_contact_in_the_first_servo_interval_is_rejected():
    up = tcp_joints([-.42, 0, LOW + .006], seed=Q_LOW)
    assert finger_low(up) > .01  # the normal rule holds at the start
    with pytest.raises(adapter_module.Rejected, match='fingers move at'):
        direct([Q_LOW] * 4, start=up)


def test_leaving_the_contact_fast_is_rejected():
    up = tcp_joints([-.42, 0, LOW + .008], seed=Q_LOW)
    assert finger_low(up) > .01  # the normal rule holds after the jump
    with pytest.raises(adapter_module.Rejected, match='fingers move at'):
        direct([Q_LOW, Q_LOW, up, up])


def test_leaving_from_a_measured_state_in_the_contact_counts_too():
    up = tcp_joints([-.42, 0, LOW + .008], seed=Q_LOW)  # no sample needs the contact; only the start does
    with pytest.raises(adapter_module.Rejected, match='fingers move at'):
        direct([up, up, up])
    start = tcp_joints([-.42, 0, TABLE + .050])  # tool box 22 mm up: only the TCP rule (52 mm) needs the contact
    first = tcp_joints([-.42, 0, TABLE + .054], seed=start)  # 4 mm up in one servo period: 0.24 m/s
    with pytest.raises(adapter_module.Rejected, match='fingers move at'):
        direct([first, first, first], start=start, tcp_min_height_m=.052)


def test_every_fast_contact_interval_counts_not_only_the_first():
    slide = [tcp_joints([-.42 + .0015 * k, 0, LOW], seed=Q_LOW) for k in (1, 2)]  # 90 mm/s after a rest
    with pytest.raises(adapter_module.Rejected, match='fingers move at'):
        direct([Q_LOW, Q_LOW, *slide])


def test_spinning_the_gripper_about_its_approach_axis_counts_corner_speed():
    spin = np.array([Q_LOW + np.r_[0, 0, 0, 0, 0, k * DT] for k in range(1, 5)])  # 1 rad/s: corners 78 mm/s, TCP still
    assert np.linalg.norm(LEFT.tcp(spin[-1])[:3, 3] - LEFT.tcp(Q_LOW)[:3, 3]) < 1e-6
    with pytest.raises(adapter_module.Rejected, match='fingers move at'):
        direct(spin)


def test_the_fastest_corner_counts():
    tilt = Q_LOW + np.r_[0, 0, 0, .005, 0, 0]
    moved = np.linalg.norm(LEFT.finger_box_corners(tilt) - LEFT.finger_box_corners(Q_LOW), axis=1) / DT
    assert moved.min() < CONTACT['contact_max_speed_m_s'] < moved.max() and finger_low(tilt) > .002
    with pytest.raises(adapter_module.Rejected, match='fingers move at'):
        direct([tilt, tilt, tilt])


def test_the_right_arm_gets_the_contact_too():
    q_right = tcp_joints([.42, 0, LOW], arm='right', max_seed_jump_rad=7.0, max_iterations=300)
    assert finger_low(q_right, 'right') < .01
    hold, left = np.repeat(q_right[None], 3, axis=0), np.repeat(HOME[None], 3, axis=0)
    with pytest.raises(adapter_module.Rejected, match='no contact declared for right'):
        direct(left, start=HOME, contacts=(), right=hold, right_start=q_right)
    direct(left, start=HOME, contacts=(window(arm='right', center=(.42, 0)),), right=hold, right_start=q_right)


def test_the_region_boundary_is_the_radius():
    direct([Q_LOW] * 3, contact_region_radius_m=CORNER + .001)
    with pytest.raises(adapter_module.Rejected, match='contact centre'):
        direct([Q_LOW] * 3, contact_region_radius_m=CORNER - .001)


def test_check_geometry_refuses_contacts_and_start_states_it_cannot_trust():
    samples = 1.0 + DT * np.arange(1, 4)
    joints = {'left': np.repeat(Q_LOW[None], 3, axis=0), 'right': np.repeat(HOME[None], 3, axis=0)}

    def check(contacts=(window(),), start=(1.0, {'left': Q_LOW, 'right': HOME})):
        adapter_module.check_geometry(MODEL, limits(**CONTACT), samples, joints, contacts=contacts, start=start)

    check()
    forged = SimpleNamespace(arm='left', kind='grasp', center_xy=(np.nan, np.nan), fragment_id='frag_a')
    for contacts, message in (((forged,),  'ContactWindow'), ((window(), window(center=(-.41, 0))), 'one contact')):
        with pytest.raises(adapter_module.Rejected, match=message):
            check(contacts=contacts)
    for start in ((samples[0], {'left': Q_LOW, 'right': HOME}), (np.nan, {'left': Q_LOW, 'right': HOME}),
                  (1.0, {'left': np.full(6, np.nan), 'right': HOME}), (1.0, {'left': Q_LOW}), None):
        with pytest.raises(adapter_module.Rejected, match='start'):
            check(start=start)
    with pytest.raises(adapter_module.Rejected, match='increasing'):  # a backwards sample hides a fast move
        adapter_module.check_geometry(MODEL, limits(**CONTACT), samples[[0, 2, 1]], joints, contacts=(window(),),
                                      start=(1.0, {'left': Q_LOW, 'right': HOME}))
