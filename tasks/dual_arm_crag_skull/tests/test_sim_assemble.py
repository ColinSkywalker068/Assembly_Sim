"""The scripted assembly (sim/assemble.py) on CPU: the voxel occupancy, the gripper's points, the grasp a step needs,
and a whole assembly of the cube fixture on KinematicBackend through S0."""
import dataclasses
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from core.workcell.config import load_workcell_preset
from tasks.dual_arm_crag_skull.baselines.backend import KinematicBackend
from tasks.dual_arm_crag_skull.baselines.task import PlanStep, task_from_layout
from tasks.dual_arm_crag_skull.contracts.actions import ARMS
from tasks.dual_arm_crag_skull.s0.adapter import Rejected, S0Limits
from tasks.dual_arm_crag_skull.s0.kinematics import load_kinematics, yam_model, yam_width_range_m
from tasks.dual_arm_crag_skull.sim import assemble, pick
from tasks.dual_arm_crag_skull.tests.baselines_fixtures import write_layout

MODEL = yam_model()
WORKCELL = load_workcell_preset('dual_arm')
HOME = {arm: np.array(WORKCELL.robot(arm).home_joint_positions, dtype=float) for arm in ARMS}
WIDTHS = {arm: yam_width_range_m() for arm in ARMS}
LIMITS = S0Limits(max_joint_velocity_rad_s=np.asarray(load_kinematics()['max_joint_velocity_rad_s']),
                  max_joint_acceleration_rad_s2=np.full(6, 20.0), gripper_width_range_m=WIDTHS,
                  arm_arm_clearance_m=0.02, table_top_z_m=0.75, table_clearance_m=0.01, tcp_min_height_m=0.02,
                  servo_hz=60, max_observation_age_s=0.5, anchor_position_tolerance_m=0.002,
                  anchor_rotation_tolerance_rad=0.005, contact_table_clearance_m=0.002, contact_region_radius_m=0.12,
                  contact_max_speed_m_s=0.05)


@pytest.fixture
def cubes(tmp_path):
    task = task_from_layout(write_layout(tmp_path / 'layout'), anchor_id='frag_a')
    return task, assemble.load_layout(task)


def test_occupancy_reads_the_layout_cells_in_the_world(cubes):
    task, layout = cubes
    anchor = assemble.Occupancy(task, layout, {'frag_a'})  # its cube: x -0.02..0.02 off the pad centre, on the table
    centre = task.world_T_assembly[:3, 3]
    assert anchor.hits(centre + [0.0, 0.0, 0.02], 0.0)
    assert not anchor.hits(centre + [0.03, 0.0, 0.02], 0.0)  # inside frag_b's cell, not the anchor's
    assert anchor.hits(centre + [0.021, 0.0, 0.02], 0.002) and not anchor.hits(centre + [0.023, 0.0, 0.02], 0.002)
    assert anchor.reach((1.0, 0.0, 0.0)) == pytest.approx(centre[0] + 0.02)
    assert anchor.top() == pytest.approx(0.79)
    assert assemble.Occupancy(task, layout, set()).reach((1.0, 0.0, 0.0)) == -math.inf


def test_occupancy_tells_every_cell_of_a_three_dimensional_grid_apart():
    """On a 2 x 2 x 2 grid (a 3 x 1 x 1 one cannot show that the cell index uses all three axes) each cell centre is a
    hit exactly when its cell is owned; and a margin of half a cell or more is refused, since the corners checked
    could then straddle a cell."""
    pitch, owned = 0.01, {(1, 0, 0), (0, 0, 1), (1, 1, 1)}
    layout = {'processing': {'pitch': pitch}, 'pieces': [{'name': 'f', 'cells': [list(c) for c in sorted(owned)]}],
              'normalization': {'grid_origin': [0.0, 0.0, 0.0], 'grid_shape': [2, 2, 2]}}
    occupancy = assemble.Occupancy(SimpleNamespace(world_T_assembly=np.eye(4)), layout, {'f'})
    for cell in itertools.product(range(2), repeat=3):
        assert occupancy.hits((np.array(cell) + 0.5) * pitch, 0.0) == (cell in owned), cell
    with pytest.raises(ValueError, match='half a cell'):
        occupancy.hits([0.005, 0.005, 0.005], pitch / 2)


def test_the_cached_gripper_points_cannot_be_changed_by_a_caller():
    assert not assemble._gripper_points_cached(MODEL.arms['left'], 0.05).flags.writeable


def test_the_fingertips_end_at_the_tcp_and_open_along_its_closing_axis():
    arm = MODEL.arms['left']
    closed, opened = assemble.gripper_points(arm, 0.02), assemble.gripper_points(arm, 0.08)
    for points in (closed, opened):  # approach is the TCP's +x: nothing of the gripper reaches past the fingertips
        assert points[:, 0].max() == pytest.approx(0.0, abs=2e-4)
    tips = lambda points: points[points[:, 0] > -0.01]  # the last centimetre of the fingers
    span = lambda points: np.ptp(tips(points)[:, 1])
    assert span(opened) - span(closed) == pytest.approx(0.06, abs=1e-3)  # the closing axis is the TCP's y


def test_a_step_takes_a_grasp_whose_fingers_stay_out_of_the_seam(cubes):
    task, layout = cubes
    poses = {key: f.initial_world_T_fragment for key, f in zip(task.fragment_ids, task.fragments)}
    step = task.assembly_plan[0]  # frag_b goes in along -x against the anchor: a finger on that side would be caught
    placement = assemble.plan_step(task, layout, MODEL, LIMITS, key=step.fragment_id,
                                   insertion=step.insertion_direction, poses=poses, placed=['frag_a'], seeds=HOME)
    closing = placement.poses['goal'][:3, 1]
    assert abs(closing[0]) < 0.2  # the fingers close along y at the goal, not along the insertion
    assert placement.rejected  # and grasps closing along x were tried and refused for the seam
    assert set(placement.rejected) <= {'descent_to_begin', 'insertion', 'release', 'back_out'}
    # frag_b shares a face with the anchor: its insertion begins BEGIN_CLEARANCE_M out along +x
    assert placement.begin_offset_m == pytest.approx(assemble.BEGIN_CLEARANCE_M)
    np.testing.assert_allclose(placement.poses['begin'][:3, 3] - placement.poses['goal'][:3, 3],
                               [assemble.BEGIN_CLEARANCE_M, 0.0, 0.0], atol=1e-12)
    assert placement.tip_z == pytest.approx(0.79 - assemble.TOP_PAD_M)  # as high up the cube as the pads still hold


def test_the_fragment_carried_down_to_where_its_insertion_begins_keeps_off_the_fragments_in_place(cubes, monkeypatch):
    """With the insertion made to begin 30 mm inside the anchor, the fragment carried down onto it is refused even
    where the gripper would come down clear (here it has no points at all; on the skull, orange's fingers were clear
    of red while orange came down onto it)."""
    monkeypatch.setattr(assemble, 'BEGIN_CLEARANCE_M', -0.03)
    monkeypatch.setattr(assemble, '_gripper_points_cached', lambda arm_model, width: np.zeros((0, 3)))
    task, layout = cubes
    poses = {key: f.initial_world_T_fragment for key, f in zip(task.fragment_ids, task.fragments)}
    step = task.assembly_plan[0]
    with pytest.raises(ValueError, match='carry_to_begin'):
        assemble.plan_step(task, layout, MODEL, LIMITS, key=step.fragment_id, insertion=step.insertion_direction,
                           poses=poses, placed=['frag_a'], seeds=HOME)


@pytest.mark.parametrize('tcp_m, centre_dx, expected', [
    (0.30, 0.0, (False, None)),  # S0's normal rules hold: no window needed
    (0.015, 0.0, (True, None)),  # under the TCP floor: the window is needed, and its rules hold
    (0.004, 0.0, (True, 'contact_clearance')),  # within S0_MARGIN_M of the contact clearance
    (0.015, 0.2, (True, 'contact_region')),  # the window's centre too far off
])
def test_the_planned_poses_keep_s0s_table_rules_with_a_margin(tcp_m, centre_dx, expected):
    arm_model = MODEL.arms['right']
    pose = pick.down_pose((0.35, 0.0), LIMITS.table_top_z_m + tcp_m, 0.0)
    q, _ = pick.solve_reachable(arm_model, pose, HOME['right'])
    assert assemble.s0_table_use(arm_model, LIMITS, q, (0.35 + centre_dx, 0.0)) == expected


class _Boxes:
    """Stands in for an arm model in s0_table_use: the TCP, and S0's tool, finger and body boxes with their lowest
    corners at the given heights over the table, all around (0.35, 0)."""

    def __init__(self, tcp_m, tool_m, finger_m, body_m):
        self.tcp_m, self.heights = tcp_m, {'tool': tool_m, 'finger': finger_m, 'body': body_m}

    def tcp(self, q):
        pose = np.eye(4)
        pose[:3, 3] = [0.35, 0.0, LIMITS.table_top_z_m + self.tcp_m]
        return pose

    def _box(self, name):
        low = LIMITS.table_top_z_m + self.heights[name]
        offsets = itertools.product((-0.01, 0.01), (-0.01, 0.01), (0.0, 0.02))
        return np.array([[0.35 + dx, dy, low + dz] for dx, dy, dz in offsets])

    def tool_box_corners(self, q):
        return self._box('tool')

    def finger_box_corners(self, q):
        return self._box('finger')

    def body_box_corners(self, q):
        return self._box('body')


@pytest.mark.parametrize('heights, contact, expected', [
    ((0.05, 0.05, 0.05, 0.10), True, (False, None)),
    ((0.015, 0.005, 0.006, 0.012), True, (True, 'body_clearance')),  # the fingers keep their clearance, not the body
    ((0.015, 0.005, 0.006, 0.05), False, (True, 'no_contact_phase')),
])
def test_s0_table_use_tells_each_of_s0s_table_rules_apart(heights, contact, expected):
    limits = LIMITS if contact else dataclasses.replace(LIMITS, contact_table_clearance_m=None,
                                                        contact_region_radius_m=None, contact_max_speed_m_s=None)
    assert assemble.s0_table_use(_Boxes(*heights), limits, None, (0.35, 0.0)) == expected


def _tilted(task, axis, degrees=30.0):
    """The task with frag_b resting turned `degrees` about the world `axis` on its lowest edge, where it was."""
    fragment = task.fragment('frag_b')
    pose = np.array(fragment.initial_world_T_fragment)
    pose[:3, :3] = Rotation.from_euler(axis, degrees, degrees=True).as_matrix()
    corners = np.array([[x, y, z] for x in (-0.02, 0.02) for y in (-0.02, 0.02) for z in (0.0, 0.04)])
    pose[2, 3] = LIMITS.table_top_z_m - (corners @ pose[:3, :3].T)[:, 2].min()
    fragments = tuple(dataclasses.replace(f, initial_world_T_fragment=pose) if f.fragment_id == 'frag_b' else f
                      for f in task.fragments)
    return dataclasses.replace(task, fragments=fragments)


def test_tilted_about_x_the_box_s0_puts_round_the_whole_stroke_refuses_the_pick(cubes, monkeypatch):
    """Turned about x, the fingers (closing along y out of the seam) tilt across their closing axis, and the box S0
    puts round their whole 141 mm stroke dips a corner below its contact clearance: S0's geometry, not the fingers',
    rules the grasp out. (Lines through the middle only: lines beside it change nothing here, but take long.)"""
    monkeypatch.setattr(assemble, 'LATERAL_OFFSETS_M', (0.0,))
    task, layout = cubes
    task = _tilted(task, 'x')
    poses = {key: np.array(f.initial_world_T_fragment) for key, f in zip(task.fragment_ids, task.fragments)}
    step = task.assembly_plan[0]
    with pytest.raises(ValueError, match='s0_pick_contact_clearance'):
        assemble.plan_step(task, layout, MODEL, LIMITS, key=step.fragment_id, insertion=step.insertion_direction,
                           poses=poses, placed=['frag_a'], seeds=HOME)


def test_a_tilted_fragment_goes_in_through_s0_on_the_kinematic_backend(cubes):
    task, layout = cubes
    task = _tilted(task, 'y')
    backend = KinematicBackend(MODEL, home_joints=HOME, width_range_m=WIDTHS, control_hz=60)
    backend.reset(task)
    result = assemble.run_assembly(backend, MODEL, LIMITS, task, layout)
    assert result['error'] is None and result['ending_error'] is None, result['error']
    assert result['steps'][0].placement.family == 'goal_aligned' and result['score'].success


def test_a_fragment_resting_tilted_is_grasped_so_the_gripper_comes_down_upright_at_its_goal(cubes):
    """frag_b rests turned 30 degrees about y (on an edge, say after toppling): the grasp is found with it in its goal
    orientation, so the fingers close on it tilted at the pick and the gripper is upright once it is at its goal. (About
    y the fingers, closing along y out of the seam, tilt about their closing axis, which keeps the box S0 puts round
    their whole stroke off the table.)"""
    task, layout = cubes
    poses = {key: np.array(f.initial_world_T_fragment) for key, f in zip(task.fragment_ids, task.fragments)}
    tilted = poses['frag_b'].copy()
    tilted[:3, :3] = Rotation.from_euler('y', 30, degrees=True).as_matrix()
    corners = np.array([[x, y, z] for x in (-0.02, 0.02) for y in (-0.02, 0.02) for z in (0.0, 0.04)])
    tilted[2, 3] = LIMITS.table_top_z_m - (corners @ tilted[:3, :3].T)[:, 2].min()  # its lowest edge on the table
    poses['frag_b'] = tilted
    step = task.assembly_plan[0]
    placement = assemble.plan_step(task, layout, MODEL, LIMITS, key=step.fragment_id,
                                   insertion=step.insertion_direction, poses=poses, placed=['frag_a'], seeds=HOME)
    assert placement.family == 'goal_aligned'
    np.testing.assert_allclose(placement.poses['goal'][:3, 0], [0.0, 0.0, -1.0], atol=1e-9)  # approach straight down
    assert math.degrees(math.acos(-placement.poses['pick'][2, 0])) == pytest.approx(30.0, abs=1e-6)
    approach = placement.poses['pick'][:3, 0]  # it comes in along its own axis
    np.testing.assert_allclose(placement.poses['pick'][:3, 3] - placement.poses['pre'][:3, 3],
                               pick.APPROACH_HEIGHT_M * approach, atol=1e-9)
    arm_model = MODEL.arms[placement.arm]
    for name, width in (('pick', placement.grasp.chord_m), ('pre', placement.grasp.open_width_m)):
        points = assemble._world(assemble.gripper_points(arm_model, width), placement.poses[name])
        assert points[:, 2].min() >= LIMITS.table_top_z_m + assemble.FINGER_MARGIN_M


def test_the_centre_of_mass_of_a_closed_mesh():
    """An L: a 2 x 1 x 1 box (0..2, 0..1, 0..1) with a unit cube on its x < 1 half (0..1, 0..1, 1..2), volume 3,
    centre of mass (2 (1, 1/2, 1/2) + (1/2, 1/2, 3/2)) / 3 = (5/6, 1/2, 5/6)."""
    import trimesh
    first = trimesh.creation.box(extents=(2, 1, 1))
    first.apply_translation((1.0, 0.5, 0.5))
    second = trimesh.creation.box(extents=(1, 1, 1))
    second.apply_translation((0.5, 0.5, 1.5))
    mesh = trimesh.util.concatenate([first, second])
    np.testing.assert_allclose(assemble._centre_of_mass(mesh.vertices, mesh.faces), (5 / 6, 1 / 2, 5 / 6), atol=1e-9)


def test_the_grasp_line_goes_over_the_centre_of_mass(cubes, monkeypatch):
    """With frag_b's centre of mass taken 8 mm along x from its middle, the fingers (closing along y, out of the seam)
    close on the line over it, not over the middle: gravity then turns it least in the grasp."""
    centre = assemble._centre_of_mass
    monkeypatch.setattr(assemble, '_centre_of_mass', lambda v, f: centre(v, f) + [0.008, 0.0, 0.0])
    task, layout = cubes
    poses = {key: f.initial_world_T_fragment for key, f in zip(task.fragment_ids, task.fragments)}
    step = task.assembly_plan[0]
    placement = assemble.plan_step(task, layout, MODEL, LIMITS, key=step.fragment_id,
                                   insertion=step.insertion_direction, poses=poses, placed=['frag_a'], seeds=HOME)
    mass = assemble._world(assemble._centre_of_mass(*assemble.fragment_mesh(task, step.fragment_id))[None],
                           poses[step.fragment_id])[0]
    assert placement.grasp.center_xy[0] == pytest.approx(mass[0], abs=1e-6)


def test_the_voxel_grid_gives_the_yaws_whose_pads_close_on_faces():
    """A fragment turned 17 degrees about the vertical from its goal has its voxel grid's horizontal axes at 17 and 107
    degrees; tilted 30 degrees about x, only its x axis stays horizontal (closing along world y would meet its tilted
    y faces on an edge); tilted about a diagonal, none does."""
    goal = np.eye(4)
    turned = np.eye(4)
    turned[:3, :3] = Rotation.from_euler('z', 17, degrees=True).as_matrix()
    assert sorted(assemble._grid_yaws_deg(turned, goal)) == pytest.approx([17.0, 107.0])
    tilted = np.eye(4)
    tilted[:3, :3] = Rotation.from_euler('x', 30, degrees=True).as_matrix()
    assert sorted(assemble._grid_yaws_deg(tilted, goal)) == pytest.approx([0.0])
    corner = np.eye(4)
    corner[:3, :3] = Rotation.from_rotvec(np.radians(30) * np.array([1.0, 1.0, 0.0]) / np.sqrt(2)).as_matrix()
    assert assemble._grid_yaws_deg(corner, goal) == ()


def test_grasps_closing_along_the_voxel_grid_come_first():
    """Pads closing across voxel edges grip a strip (in Isaac green slid out of such a grasp); along the grid they
    close on faces. Those come first, the narrowest of them first, then the rest by chord."""
    grasp = lambda yaw, chord: pick.Grasp('f', np.radians(yaw), (0.0, 0.0), chord, chord + 0.02, chord - 0.006, chord)
    found = [grasp(45.0, 0.030), grasp(17.0, 0.050), grasp(107.0, 0.040), grasp(60.0, 0.020)]
    order = [np.degrees(g.yaw_rad) for g in assemble._in_preferred_order(found, (17.0, 107.0))]
    assert order == pytest.approx([107.0, 17.0, 60.0, 45.0])


def test_a_pads_grip_is_how_far_up_it_the_fragment_reaches():
    """A 40 x 30 x 40 mm box gripped across its 30 mm with the fingertips 28 mm up: the box rises 12 mm into each pad
    (it ends 12 mm above the fingertips); with the fingertips 20 mm up, the pads' whole 15 mm."""
    corners = np.array([[x, y, z] for x in (-0.02, 0.02) for y in (-0.015, 0.015) for z in (0.0, 0.04)])
    hull = trimesh.convex.convex_hull(corners)
    surface = pick.surface_points(np.asarray(hull.vertices) + [0.35, 0.0, 0.75], np.asarray(hull.faces))
    high = pick.down_pose((0.35, 0.0), 0.75 + 0.028, np.pi / 2)  # closing along y
    low = pick.down_pose((0.35, 0.0), 0.75 + 0.020, np.pi / 2)
    assert assemble._pad_grip_m(surface, high, 0.03) == pytest.approx(0.012, abs=0.0021)
    assert assemble._pad_grip_m(surface, low, 0.03) == pytest.approx(assemble.PAD_DEPTH_M, abs=0.0021)


@pytest.mark.parametrize('grip, lowered', [(0.0135, True), (0.016, False)])
def test_grasps_whose_pads_both_grip_well_come_first(cubes, monkeypatch, grip, lowered):
    """In Isaac four of five grasps with a pad on only a 4 mm edge of the fragment let it go, and none gripping 12 mm
    on both pads did. Asked for more grip than the cube's top grasp gives, the step grasps lower; asked for more than
    the pads have, it falls back on the grasps it would take otherwise, the top one."""
    task, layout = cubes
    poses = {key: f.initial_world_T_fragment for key, f in zip(task.fragment_ids, task.fragments)}
    step = task.assembly_plan[0]
    monkeypatch.setattr(assemble, 'MIN_PAD_GRIP_M', grip)
    placement = assemble.plan_step(task, layout, MODEL, LIMITS, key=step.fragment_id,
                                   insertion=step.insertion_direction, poses=poses, placed=['frag_a'], seeds=HOME)
    top = 0.79 - assemble.TOP_PAD_M
    assert (placement.tip_z < top - 1e-6) == lowered and placement.rejected['pad_grip'] >= 1


def test_a_grasp_too_thin_to_hold_is_passed_over(cubes, monkeypatch):
    """Pads closing on less than MIN_CHORD_M (here more than the 40 mm cube is across, even corner to corner) would
    grip an edge: none is taken."""
    monkeypatch.setattr(assemble, 'MIN_CHORD_M', 0.1)
    task, layout = cubes
    poses = {key: f.initial_world_T_fragment for key, f in zip(task.fragment_ids, task.fragments)}
    step = task.assembly_plan[0]
    with pytest.raises(ValueError, match="'thin'"):
        assemble.plan_step(task, layout, MODEL, LIMITS, key=step.fragment_id, insertion=step.insertion_direction,
                           poses=poses, placed=['frag_a'], seeds=HOME)


def test_the_fingers_never_release_to_less_than_they_are_open(cubes):
    """If the fragment has turned in the grasp and holds the fingers wider than planned, the release opens beyond
    that, not to the planned release width (which would squeeze it and carry it off)."""
    task, layout = cubes
    backend = _backend(task)
    step = backend.step

    def pried(targets, widths):  # the measured opening is 20 mm more than commanded
        state = step(targets, widths)
        return dataclasses.replace(state, widths={arm: width + 0.02 for arm, width in state.widths.items()})
    backend.step = pried
    seen, commanded = [], []
    runner = assemble.Runner(backend, MODEL, LIMITS, on_step=lambda name, state, targets, contact: seen.append(name))
    poses = {key: pose.world_T_fragment for key, pose in backend.observed_fragment_poses().items()}
    first = task.assembly_plan[0]
    placement = assemble.plan_step(task, layout, MODEL, LIMITS, key=first.fragment_id,
                                   insertion=first.insertion_direction, poses=poses, placed=['frag_a'], seeds=HOME)
    backend.step = lambda targets, widths: commanded.append(dict(widths)) or pried(targets, widths)
    assemble.run_step(runner, task, placement)
    released = [widths[placement.arm] for name, widths in zip(seen[-len(commanded):], commanded) if name == 'release']
    assert released[-1] == pytest.approx(placement.grasp.close_width_m + 0.02 + 2 * assemble.RELEASE_GAP_M)
    assert released[-1] > placement.release_width_m


def test_where_the_line_through_the_middle_will_not_do_a_line_beside_it_is_tried(cubes, monkeypatch):
    """Lines through the middle of the footprint are tried first at each height; with those all refused (as a pocket's
    wall would refuse them), the step still goes, on a line moved across the closing axis by one of the offsets."""
    task, layout = cubes
    poses = {key: f.initial_world_T_fragment for key, f in zip(task.fragment_ids, task.fragments)}
    step = task.assembly_plan[0]
    plain = assemble.plan_step(task, layout, MODEL, LIMITS, key=step.fragment_id, insertion=step.insertion_direction,
                               poses=poses, placed=['frag_a'], seeds=HOME)
    surface = assemble._surface_at(task, step.fragment_id, poses[step.fragment_id])
    middle = (surface[:, :2].min(axis=0) + surface[:, :2].max(axis=0)) / 2  # of the footprint, as pick.py takes it
    try_grasp = assemble._try_grasp

    def refuse_the_middle(*args):
        grasp = args[6]
        across = np.array([math.sin(grasp.yaw_rad), -math.cos(grasp.yaw_rad)])
        if abs((np.asarray(grasp.center_xy) - middle) @ across) < 1e-6:
            args[-1]('middle')
            return None
        return try_grasp(*args)
    monkeypatch.setattr(assemble, '_try_grasp', refuse_the_middle)
    moved = assemble.plan_step(task, layout, MODEL, LIMITS, key=step.fragment_id, insertion=step.insertion_direction,
                               poses=poses, placed=['frag_a'], seeds=HOME)
    across = np.array([math.sin(moved.grasp.yaw_rad), -math.cos(moved.grasp.yaw_rad)])
    offset = (np.asarray(moved.grasp.center_xy) - middle) @ across
    assert min(abs(offset - o) for o in assemble.LATERAL_OFFSETS_M) < 1e-6 and abs(offset) > 1e-6
    assert moved.tip_z == plain.tip_z and moved.rejected['middle'] >= 1


def test_backing_out_counts_the_fragment_just_released(cubes):
    """With nothing in place, frag_b's back-out along +x is all that can refuse a grasp closing along x: the finger
    on its -x side would be dragged back through it."""
    task, layout = cubes
    poses = {key: f.initial_world_T_fragment for key, f in zip(task.fragment_ids, task.fragments)}
    step = task.assembly_plan[0]
    placement = assemble.plan_step(task, layout, MODEL, LIMITS, key=step.fragment_id,
                                   insertion=step.insertion_direction, poses=poses, placed=[], seeds=HOME)
    assert abs(placement.poses['goal'][:3, 1][0]) < 0.2
    assert set(placement.rejected) == {'back_out'}


def test_the_fingers_open_only_enough_to_back_out_past_the_fragment(cubes):
    task, layout = cubes
    poses = {key: f.initial_world_T_fragment for key, f in zip(task.fragment_ids, task.fragments)}
    step = task.assembly_plan[0]
    placement = assemble.plan_step(task, layout, MODEL, LIMITS, key=step.fragment_id,
                                   insertion=step.insertion_direction, poses=poses, placed=['frag_a'], seeds=HOME)
    grasp = placement.grasp
    assert placement.release_width_m == pytest.approx(grasp.clear_width_m + 2 * assemble.RELEASE_GAP_M, abs=1e-4)
    assert grasp.chord_m < placement.release_width_m < grasp.open_width_m


def test_another_order_is_taken_only_if_its_insertions_go_in_straight(cubes):
    task, layout = cubes
    steps = assemble.reordered_plan(task, layout, ['frag_c', 'frag_b'])
    assert [step.fragment_id for step in steps] == ['frag_c', 'frag_b']
    assert steps[0] == task.assembly_plan[1] and steps[1] == task.assembly_plan[0]  # each keeps its direction
    with pytest.raises(ValueError, match='every fragment'):
        assemble.reordered_plan(task, layout, ['frag_b'])
    through = dataclasses.replace(task, assembly_plan=(task.assembly_plan[0], PlanStep('frag_c', (-1.0, 0.0, 0.0))))
    with pytest.raises(ValueError, match='frag_c cannot go in straight'):  # it would leave through the anchor
        assemble.reordered_plan(through, layout, ['frag_b', 'frag_c'])


def _backend(task):
    backend = KinematicBackend(MODEL, home_joints=HOME, width_range_m=WIDTHS, control_hz=60)
    backend.reset(task)
    return backend


def _stop_at(name, count=1):
    """An on_step that raises on the `count`th control period of segment `name`."""
    seen = []

    def on_step(segment, *_):
        if segment == name:
            seen.append(segment)
            if len(seen) == count:
                raise RuntimeError(f'stopped in {name}')
    return on_step


def _moved_in_the_grasp(backend, shift):
    """The kinematic backend, but once an arm has taken a fragment, the fragment sits `shift` (the TCP frame) further
    off in the grasp than where it was taken: a slip the plan did not expect."""
    step, done = backend.step, set()

    def stepped(targets, widths):
        state = step(targets, widths)
        for arm, held in backend.held.items():
            if held is not None and (arm, held[0]) not in done:
                done.add((arm, held[0]))
                moved = np.eye(4)
                moved[:3, 3] = shift
                backend.held[arm] = (held[0], moved @ held[1])
        return state
    backend.step = stepped
    return backend


def test_a_fragment_that_moved_in_the_grasp_goes_in_from_where_it_now_is(cubes):
    """frag_b sits 4 mm off in the grasp after it is taken (across the closing axis, along the cubes' row): read where
    it hangs before it goes over, the step puts the fragment, not the gripper, on its goal, and says how far it had
    moved."""
    task, layout = cubes
    backend = _moved_in_the_grasp(_backend(task), (0.0, 0.0, 0.004))
    result = assemble.run_assembly(backend, MODEL, LIMITS, task, layout)
    assert result['error'] is None and result['score'].success, result['error']
    first = result['steps'][0]
    seen = next(segment for segment in first.segments if segment['name'] == 'observe')
    assert seen['in_hand_mm_deg'][0] == pytest.approx(4.0, abs=0.01) and seen['in_hand_mm_deg'][1] < 0.01
    assert first.drift_mm_deg['frag_b'][0] < 0.1  # S0's IK holds the TCP to 0.1 mm; uncorrected it would be 4 mm


def test_a_fragment_that_moved_on_the_way_is_lined_up_again_where_its_insertion_begins(cubes):
    """frag_b slips 3 mm in the grasp while it comes down to where its insertion begins, after the first reading: read
    again there, the gripper moves to line it up and then goes in straight, and it ends on its goal."""
    task, layout = cubes
    backend, slipped = _backend(task), []

    def on_step(name, *_):
        if name == 'lower' and not slipped:
            arm = next(arm for arm, held in backend.held.items() if held is not None)
            moved = np.eye(4)
            moved[:3, 3] = (0.0, 0.003, 0.0)
            backend.held[arm] = (backend.held[arm][0], moved @ backend.held[arm][1])
            slipped.append(arm)
    result = assemble.run_assembly(backend, MODEL, LIMITS, task, layout, on_step=on_step)
    assert result['error'] is None and result['score'].success, result['error']
    first = result['steps'][0]
    readings = [segment['in_hand_mm_deg'][0] for segment in first.segments if segment['name'] == 'observe']
    assert readings[0] == pytest.approx(0.0, abs=0.15) and readings[1] == pytest.approx(3.0, abs=0.15)  # IK's 0.1 mm
    names = [segment['name'] for segment in first.segments]
    assert names.index('lower') < names.index('align') < names.index('insert')
    assert first.drift_mm_deg['frag_b'][0] < 0.1


def test_a_fragment_that_moved_too_far_in_the_grasp_stops_the_step(cubes):
    task, layout = cubes
    backend = _moved_in_the_grasp(_backend(task), (0.0, 0.0, 0.02))
    result = assemble.run_assembly(backend, MODEL, LIMITS, task, layout)
    assert result['error'] == 'ValueError: frag_b moved 20.0 mm / 0.0 deg in the grasp (at most 10 mm / 10 deg)'


def test_a_run_that_stops_before_the_ending_is_not_settled(cubes):
    """Both fragments are in when the last settle hold breaks off; without the ending the score is not settled, and
    so not a success."""
    task, layout = cubes
    periods = int(np.ceil(assemble.STEP_SETTLE_S * LIMITS.servo_hz))
    result = assemble.run_assembly(_backend(task), MODEL, LIMITS, task, layout, on_step=_stop_at('settle', periods + 1))
    assert result['error'] == 'RuntimeError: stopped in settle' and result['ending_error'] is None
    assert not result['score'].settled and not result['score'].success


def test_an_ending_that_breaks_off_is_reported_and_not_settled(cubes):
    task, layout = cubes
    result = assemble.run_assembly(_backend(task), MODEL, LIMITS, task, layout, on_step=_stop_at('ending'))
    assert result['error'] is None and result['ending_error'] == 'the ending stopped: RuntimeError: stopped in ending'
    assert not result['score'].settled and not result['score'].success


def test_a_failed_step_keeps_the_segments_it_ran(cubes):
    task, layout = cubes
    result = assemble.run_assembly(_backend(task), MODEL, LIMITS, task, layout, on_step=_stop_at('lift'))
    first = result['steps'][0]
    assert first.error == 'RuntimeError: stopped in lift'
    assert [segment['name'] for segment in first.segments] == ['approach', 'descend', 'close', 'squeeze']


def test_s0_names_the_segment_it_refuses(cubes):
    task, _ = cubes
    runner = assemble.Runner(_backend(task), MODEL, LIMITS)
    jump = pick.Segment('jump', (runner.targets['left'] + 1.0)[None], np.full(1, 0.05))  # 1 rad in one period
    with pytest.raises(Rejected, match='^jump: '):
        runner.run('left', jump)


@pytest.mark.parametrize('width', [0.0, 0.03, 0.06, 0.0939])
def test_the_boxes_s0_puts_round_the_gripper_hold_it_at_every_opening(width):
    """What lets the planner leave the fingers' own clearance from the table to S0's rules: S0's finger box holds
    every point of the fingers, and its tool box every point of the gripper, whatever the opening."""
    arm_model = MODEL.arms['right']
    q, _ = pick.solve_reachable(arm_model, pick.down_pose((0.35, 0.0), LIMITS.table_top_z_m + 0.2, 0.4), HOME['right'])
    tcp = arm_model.tcp(q)

    def inside(points, corners):
        low, high = corners.min(axis=0), corners.max(axis=0)  # the boxes are axis-aligned in the TCP frame
        return ((points >= low - 1e-6) & (points <= high + 1e-6)).all()
    in_tcp = lambda corners: (corners - tcp[:3, 3]) @ tcp[:3, :3]
    points = assemble.gripper_points(arm_model, width)
    tips = sum(len(tip) for tip in assemble._gripper_in_mount()[1].values())  # the tips come first
    assert inside(points[:tips], in_tcp(arm_model.finger_box_corners(q)))
    assert inside(points, in_tcp(arm_model.tool_box_corners(q)))


def test_the_fragment_glides_high_enough_held_either_way():
    """Held as at the pick it hangs 50 mm under the TCP, as at its goal 80 mm: the glide height clears the top by
    SAFE_CLEARANCE_M under the deeper of the two."""
    pick_pose, goal_pose = np.eye(4), np.eye(4)
    pick_pose[2, 3] = goal_pose[2, 3] = 0.80
    assert assemble._clearing_height(0.79, pick_pose, goal_pose, (0.75, 0.72)) == pytest.approx(
        0.79 + assemble.SAFE_CLEARANCE_M + 0.08)
    assert assemble._clearing_height(0.79, pick_pose, goal_pose, (0.72, 0.75)) == pytest.approx(
        0.79 + assemble.SAFE_CLEARANCE_M + 0.08)


def test_a_preflight_tries_the_same_segments_the_run_then_sends(cubes):
    """Tried and then run from the same state, a step's segments are the same ones, ending at the same joints (the
    kinematic backend reaches every target) up to the hold where the run reads the fragment in the hand; after it,
    within what that reading corrects (here the 0.1 mm S0's IK leaves): a preflight checks what will be sent."""
    task, layout = cubes
    backend = _backend(task)
    sent = []
    runner = assemble.Runner(backend, MODEL, LIMITS, on_step=lambda name, state, targets, contact: sent.append(
        (name, targets)))
    poses = {key: pose.world_T_fragment for key, pose in backend.observed_fragment_poses().items()}
    step = task.assembly_plan[0]
    placement = assemble.plan_step(task, layout, MODEL, LIMITS, key=step.fragment_id,
                                   insertion=step.insertion_direction, poses=poses, placed=['frag_a'], seeds=HOME)
    trial = assemble.Preflight(runner, poses)
    assemble.run_step(trial, task, placement)
    assemble.run_step(runner, task, placement)
    runs, ends = [], []  # the segments as sent, one per run of periods with the same name
    for k, (name, targets) in enumerate(sent):
        if k == 0 or sent[k - 1][0] != name:
            runs.append(name)
            ends.append(None)
        ends[-1] = targets[placement.arm]
    names = [segment.name for _, segment in trial.segments]
    assert names == runs
    for k, (_, segment) in enumerate(trial.segments):
        read = k > names.index('steady')
        np.testing.assert_allclose(segment.joints[-1], ends[k], atol=1e-3 if read else 1e-6)


def test_a_swept_move_is_checked_wherever_the_gripper_has_moved_a_little():
    """A 10 cm line of the right arm: the poses checked are never more than SWEEP_STEP_M of gripper motion apart, and
    a cloud of points the gripper passes through is hit while one off its path, or out of its reach, is not."""
    arm_model = MODEL.arms['right']
    start = pick.down_pose((0.35, 0.0), LIMITS.table_top_z_m + 0.2, 0.0)
    end = assemble._translated(start, (0.1, 0.0, 0.0))
    q0, _ = pick.solve_reachable(arm_model, start, HOME['right'])
    segment = pick.cartesian_segment('line', arm_model, LIMITS, q0, end, 0.05, LIMITS.servo_hz, max_speed_m_s=0.05)
    poses = assemble._swept_poses(arm_model, segment.joints)
    steps = [np.linalg.norm(b[:3, 3] - a[:3, 3]) for a, b in zip(poses, poses[1:])]
    assert len(poses) >= 0.1 / assemble.SWEEP_STEP_M and max(steps) <= assemble.SWEEP_STEP_M + 1e-3
    gripper = assemble.gripper_points(arm_model, 0.05)

    def cloud(centre):
        points = np.asarray(centre) + np.random.default_rng(0).uniform(-0.002, 0.002, (50, 3))
        return [(cKDTree(points), points.min(axis=0), points.max(axis=0))]
    at = lambda pose: assemble._world(gripper, pose)
    midway = (start[:3, 3] + end[:3, 3]) / 2  # the fingertips pass right through it
    assert assemble._comes_near(at, poses, None, cloud(midway))
    assert not assemble._comes_near(at, poses, None, cloud(midway + [0.0, 0.0, 0.3]))  # well above the gripper
    assert not assemble._comes_near(at, [poses[0]], None, cloud(end[:3, 3] + [0.5, 0.0, 0.0]))  # out of reach


def test_a_preflight_names_the_move_that_would_hit(cubes):
    task, layout = cubes
    backend = _backend(task)
    runner = assemble.Runner(backend, MODEL, LIMITS)
    poses = {key: pose.world_T_fragment for key, pose in backend.observed_fragment_poses().items()}
    step = task.assembly_plan[0]
    placement = assemble.plan_step(task, layout, MODEL, LIMITS, key=step.fragment_id,
                                   insertion=step.insertion_direction, poses=poses, placed=['frag_a'], seeds=HOME)
    assert assemble._preflight(runner, task, placement, poses, {}) is None
    assert assemble._preflight(runner, task, placement, poses, {'park': lambda *_: True}) == 'sweep_park'
    assert backend.time_s == 0.0


def test_each_step_is_tried_through_s0_before_it_is_run(cubes, monkeypatch):
    """S0 asks for a little less arm-arm clearance than the arms have where they wait, which reaching to the pad
    takes away: with the steps tried first the run is refused while planning, before the robot moves; without that,
    S0 refuses it part way through a step."""
    from tasks.dual_arm_crag_skull.s0.collision import capsule_clearance
    task, layout = cubes
    capsules = {arm: MODEL.arms[arm].capsules(HOME[arm]).arrays() for arm in ARMS}
    limits = dataclasses.replace(LIMITS, arm_arm_clearance_m=capsule_clearance(capsules['left'], capsules['right'])[0]
                                 - 0.01)
    backend = _backend(task)
    result = assemble.run_assembly(backend, MODEL, limits, task, layout)
    assert result['error'].startswith(f'ValueError: frag_b: no grasp went in {assemble.MAX_PREFLIGHTS} preflights')
    assert 'preflight_s0_' in result['error'] and backend.time_s == 0.0  # it never moved
    plan_step = assemble.plan_step
    monkeypatch.setattr(assemble, 'plan_step', lambda *args, runner=None, **kwargs: plan_step(*args, **kwargs))
    result = assemble.run_assembly(_backend(task), MODEL, limits, task, layout)
    assert result['error'].startswith('Rejected: ')


def test_the_cubes_go_together_with_s0s_contact_phase_off(cubes):
    """Grasped as high as the pads hold, the cubes never need a contact window, so a run under S0 without its
    contact phase goes through (planning near the table at pick.UNCONFIGURED_CONTACT_SPEED_M_S)."""
    task, layout = cubes
    limits = dataclasses.replace(LIMITS, contact_table_clearance_m=None, contact_region_radius_m=None,
                                 contact_max_speed_m_s=None)
    result = assemble.run_assembly(_backend(task), MODEL, limits, task, layout)
    assert result['error'] is None and result['ending_error'] is None and result['score'].success, result['error']


def test_an_assembly_can_start_part_way_through_the_plan(cubes):
    """With frag_b already at its goal, a run of the rest of the plan does frag_c alone, frag_b counted as in place."""
    task, layout = cubes
    backend = _backend(task)
    backend.poses['frag_b'] = np.array(task.fragment('frag_b').goal_world_T_fragment)
    result = assemble.run_assembly(backend, MODEL, LIMITS, task, layout, plan=task.assembly_plan[1:],
                                   placed=['frag_a', 'frag_b'])
    assert result['error'] is None and [step.fragment_id for step in result['steps']] == ['frag_c']
    assert set(result['steps'][0].drift_mm_deg) == {'frag_b', 'frag_c'} and result['score'].success


def test_the_cube_assembly_goes_together_on_the_kinematic_backend(cubes):
    task, layout = cubes
    backend = KinematicBackend(MODEL, home_joints=HOME, width_range_m=WIDTHS, control_hz=60)
    backend.reset(task)
    seen, commanded, step = [], [], backend.step
    backend.step = lambda targets, widths: commanded.append(dict(widths)) or step(targets, widths)
    result = assemble.run_assembly(backend, MODEL, LIMITS, task, layout, on_step=lambda name, *_: seen.append(name))
    assert result['error'] is None and result['ending_error'] is None, result['error']
    assert [step.fragment_id for step in result['steps']] == ['frag_b', 'frag_c']
    assert result['score'].success
    assert {'descend', 'close', 'lift', 'glide', 'insert', 'release', 'back', 'settle', 'ending'} <= set(seen)
    first, end = result['steps'][0].placement, seen.index('settle')  # its fingers open to the release width only
    assert {commanded[k][first.arm] for k in range(end) if seen[k] in ('back', 'leave', 'park')} == \
        {first.release_width_m}
    for step in result['steps']:
        assert max(max(drift) for drift in step.drift_mm_deg.values()) < 0.5
