"""DexAgent baseline: export, transport, trajectory import and runner. Integration tests need the DexAgent venv
and clone (DEXAGENT_PYTHON, DEXAGENT_SOURCE) and skip without them."""
import base64
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tasks.dual_arm_crag_skull.baselines.budget import BudgetPool
from tasks.dual_arm_crag_skull.baselines.dexagent import dexenv, export, runner, trajectory, transport
from tasks.dual_arm_crag_skull.baselines.dexagent.geometry import asset_mjcf, box_decomposition, load_mesh, voxel_surface
from tasks.dual_arm_crag_skull.baselines.scoring import RunRecord, append_records, pose_errors
from tasks.dual_arm_crag_skull.__main__ import DEFAULT_CONFIG
from tasks.dual_arm_crag_skull.baselines.task import configured_initial_condition, task_from_layout
from tasks.dual_arm_crag_skull.tests.baselines_fixtures import write_layout
from tasks.dual_arm_crag_skull.workspace import REPO_ROOT

PITCH = 0.008


def voxel_mesh(cells, pitch=PITCH, pivot=(0.3, 0.2, 0.0)):
    """The exposed faces of unit cells as separate quads (two triangles each), like the layout npz meshes."""
    cells = {tuple(c) for c in cells}
    vertices, faces = [], []
    for cell in sorted(cells):
        for axis in range(3):
            for side in (0, 1):
                step = [0, 0, 0]
                step[axis] = 1 if side else -1
                if tuple(c + s for c, s in zip(cell, step)) in cells:
                    continue
                u, v = [a for a in range(3) if a != axis]
                corners = []
                for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    corner = list(cell)
                    corner[axis] += side
                    corner[u] += du
                    corner[v] += dv
                    corners.append([(c - p) * pitch for c, p in zip(corner, pivot)])
                base = len(vertices)
                vertices += corners
                faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    return np.array(vertices, np.float32), np.array(faces, np.int32)


L_CELLS = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 2, 0), (0, 0, 1)]


def test_box_decomposition_is_exact_for_a_concave_voxel_mesh(tmp_path):
    vertices, faces = voxel_mesh(L_CELLS)
    np.savez(tmp_path / 'l.npz', v=vertices, f=faces)
    decomposition = box_decomposition(*load_mesh(tmp_path / 'l.npz'))
    assert decomposition.exact and decomposition.pitch_m == pytest.approx(PITCH, abs=1e-7)
    assert len(decomposition.cells) == len(L_CELLS)
    assert decomposition.volume_m3 == pytest.approx(len(L_CELLS) * PITCH ** 3)
    covered = sorted(tuple(int(round(v)) for v in (centre - np.asarray(decomposition.origin_m)) / PITCH - 0.5)
                     for centre in decomposition.cell_centres())
    assert covered == sorted(L_CELLS)
    boxes = decomposition.box_geometry()
    assert sum(float(np.prod(2 * half)) for _, half in boxes) == pytest.approx(len(L_CELLS) * PITCH ** 3)
    notch = np.asarray(decomposition.origin_m) + (np.array([1, 1, 0]) + 0.5) * PITCH  # inside the hull, not the solid
    assert not any(np.all(np.abs(notch - centre) < half) for centre, half in boxes)


def test_the_visual_mesh_is_the_closed_exterior_of_the_solid_where_cells_touch_along_an_edge(tmp_path):
    import trimesh
    cells = [(0, 0, 0), (1, 1, 0), (2, 1, 0), (1, 1, 1)]  # (0,0,0) meets (1,1,0) along one edge only
    vertices, faces = voxel_mesh(cells)
    assert not trimesh.Trimesh(vertices, faces, process=True).is_watertight  # the voxel surface shares it 4 times
    np.savez(tmp_path / 'c.npz', v=vertices, f=faces)
    decomposition = box_decomposition(*load_mesh(tmp_path / 'c.npz'))
    visual = trimesh.Trimesh(*voxel_surface(decomposition), process=True)  # as DexAgent's audit reads a mesh
    assert visual.is_watertight and visual.is_winding_consistent
    assert visual.volume == pytest.approx(len(cells) * PITCH ** 3, rel=1e-4)
    # the 20 exposed cell faces and nothing else: the blocking object_visual_penetration_audit measures a point's
    # depth to the nearest surface sample, which a face inside the solid would cut short
    assert visual.area == pytest.approx(20 * PITCH ** 2, rel=1e-4)
    out = tmp_path / 'assets'
    plan = SimpleNamespace(fragments=[SimpleNamespace(object_id='red_fragment', vertices=vertices, faces=faces,
                                                      decomposition=decomposition, rgb=(1.0, 0.0, 0.0), free=True,
                                                      initial_rig=np.eye(4), goal_rig=np.eye(4))],
                           options=export.ExportOptions())
    export.write_assets(plan, out)
    obj = trimesh.load(out / 'stage2' / 'assets' / 'red_fragment' / 'red_fragment.obj', process=True)
    assert obj.is_watertight and obj.area == pytest.approx(visual.area, rel=1e-6)  # written as the visual mesh


def test_box_decomposition_tolerates_float32_lattice_noise(tmp_path):
    vertices, faces = voxel_mesh(L_CELLS, pivot=(13.37, 7.11, 0.0))
    noisy = vertices + np.float32(3e-8) * np.sign(np.sin(np.arange(vertices.size))).reshape(vertices.shape)
    decomposition = box_decomposition(noisy.astype(float), faces.astype(np.int64))
    assert decomposition.exact and len(decomposition.cells) == len(L_CELLS)


def test_asset_mjcf_follows_the_dexagent_asset_contract():
    vertices, faces = voxel_mesh(L_CELLS)
    decomposition = box_decomposition(vertices.astype(float), faces.astype(np.int64))
    free = asset_mjcf('red_fragment', 'red_fragment.obj', decomposition, (0.4, 0.0, 0.0, 1.0), 650.0, (0.8, 0.02, 0.001))
    fixed = asset_mjcf('red_fragment', 'red_fragment.obj', decomposition, (0.4, 0.0, 0.0, 1.0), 650.0,
                       (0.8, 0.02, 0.001), free=False)
    assert '<body name="red_fragment">' in free and '<freejoint name="red_fragment_free"/>' in free
    assert 'freejoint' not in fixed
    assert 'contype="0" conaffinity="0" group="2" density="0"' in free
    assert free.count('type="box"') == len(decomposition.boxes)
    assert free.count('solref="0.006 1"') == len(decomposition.boxes)  # explicit: no rigid-pair defaults added
    with pytest.raises(ValueError, match='DexAgent-safe'):
        asset_mjcf('Red-fragment', 'x.obj', decomposition, (1, 1, 1, 1), 650.0, (0.8, 0.02, 0.001))


def test_the_collision_boxes_are_inset_as_the_protocol_says(tmp_path):
    """Each box comes in by the inset on every face (its centre kept), so neighbouring fragments' seams open by twice
    it, as in Isaac; the export takes the inset from the protocol."""
    from tasks.dual_arm_crag_skull.baselines.task import canonical_protocol
    vertices, faces = voxel_mesh(L_CELLS)
    decomposition = box_decomposition(vertices.astype(float), faces.astype(np.int64))
    exact = [half for _, half in decomposition.box_geometry()]
    inset = [half for _, half in decomposition.box_geometry(inset_m=0.0005)]
    for full, smaller in zip(exact, inset):
        np.testing.assert_allclose(smaller, np.asarray(full) - 0.0005)
    xml = asset_mjcf('red_fragment', 'red_fragment.obj', decomposition, (0.4, 0.0, 0.0, 1.0), 650.0,
                     (0.8, 0.02, 0.001), inset_m=0.0005)
    sizes = [np.array(chunk.split('"')[0].split(), dtype=float) for chunk in xml.split('type="box" size="')[1:]]
    np.testing.assert_allclose(np.sort(np.concatenate(sizes)), np.sort(np.concatenate(inset)), atol=1e-9)
    along = decomposition.box_geometry(inset_m=(0.00025, 0.00025, 0.0001))
    for full, (_, smaller) in zip(exact, along):
        np.testing.assert_allclose(smaller, np.asarray(full) - [0.00025, 0.00025, 0.0001])
    assert export.ExportOptions().collider_inset_m == canonical_protocol()['collider_inset_m']
    assert export.ExportOptions(collider_inset_m=0.0005).collider_inset_m == {'horizontal': 0.0005, 'vertical': 0.0005}
    for bad in (-0.001, {'horizontal': 0.0005}, {'horizontal': 0.0005, 'vertical': float('nan')}):
        with pytest.raises(ValueError, match='collider_inset_m'):
            export.ExportOptions(collider_inset_m=bad)


@pytest.fixture
def cube_task(tmp_path):
    layout = write_layout(tmp_path / 'layout')
    return task_from_layout(layout, anchor_id='frag_a')


def _exported_half_sizes(plan, out):
    sizes = []
    for asset in export.write_assets(plan, out):
        xml = Path(asset['xml']).read_text()
        chunks = xml.split('type="box" size="')[1:]
        sizes.append([np.array(chunk.split('"')[0].split(), dtype=float) for chunk in chunks])
    return sizes


def test_the_exported_assets_carry_the_protocols_inset(cube_task, tmp_path):
    """Each 40 mm cube is one collision box; written out, its half size is 20 mm less the protocol's inset: its
    horizontal part across the sides, its vertical part across the top and bottom (the cubes end unturned)."""
    from tasks.dual_arm_crag_skull.baselines.task import canonical_protocol
    plan = export.plan_export(cube_task, export.ExportOptions())
    inset = canonical_protocol()['collider_inset_m']
    for sizes in _exported_half_sizes(plan, tmp_path / 'out'):
        assert len(sizes) == 1
        np.testing.assert_allclose(sizes[0], 0.02 - np.array([inset['horizontal'], inset['horizontal'], inset['vertical']]),
                                   atol=1e-9)


def test_the_exported_inset_follows_the_axis_that_is_vertical_at_the_goal(cube_task, tmp_path):
    """A fragment whose goal turns it about x and then about z has its own y axis vertical there: the vertical part
    of the inset comes off across y (a transposed rotation would pick x)."""
    from dataclasses import replace
    turned = np.eye(4)
    turned[:3, :3] = Rotation.from_euler('xz', [90, 90], degrees=True).as_matrix()
    plan = export.plan_export(cube_task, export.ExportOptions(collider_inset_m={'horizontal': 0.0005, 'vertical': 0.0}))
    plan = replace(plan, fragments=tuple(replace(f, goal_rig=f.goal_rig @ turned) for f in plan.fragments))
    for sizes in _exported_half_sizes(plan, tmp_path / 'out'):
        np.testing.assert_allclose(sizes[0], [0.0195, 0.02, 0.0195], atol=1e-9)


def test_rig_transform_puts_the_pad_on_the_dexagent_tabletop():
    world_T_assembly = np.eye(4)
    world_T_assembly[:3, 3] = [0.3, -0.2, 0.75]
    task = SimpleNamespace(world_T_assembly=world_T_assembly, table_top_z_m=0.75)
    rig = export.rig_transform(task, (0.1, 0.4), 30.0)
    centre = np.array([0.3, -0.2, 0.75, 1.0])
    np.testing.assert_allclose(rig @ centre, [0.1, 0.4, 0.0, 1.0], atol=1e-12)
    step = rig @ (centre + [1.0, 0.0, 0.0, 0.0]) - rig @ centre  # world +x turns 30 deg toward rig +y
    np.testing.assert_allclose(step, [np.cos(np.radians(30)), np.sin(np.radians(30)), 0.0, 0.0], atol=1e-12)


def test_plan_maps_ids_frames_hands_and_order(cube_task):
    plan = export.plan_export(cube_task, export.ExportOptions(pad_xy=(0.0, 0.4)))
    ids = plan.id_map
    assert ids == {'red_fragment': 'frag_a', 'blue_fragment': 'frag_b', 'green_fragment': 'frag_c'}
    assert all(export.SAFE_ID.match(key) for key in ids) and len(set(ids.values())) == len(ids)
    assert plan.anchor.object_id == 'red_fragment' and not plan.anchor.free and plan.anchor.free_joint is None
    blue, green = plan.fragment('blue_fragment'), plan.fragment('green_fragment')
    assert (blue.acting_hand, green.acting_hand) == ('left', 'right')  # staged at world x = -0.42 and +0.42
    assert blue.root_body == 'blue_fragment_blue_fragment' and blue.free_joint == 'blue_fragment_blue_fragment_free'
    assert set(plan.order) == {'blue_fragment', 'green_fragment'}
    world_T_rig = np.linalg.inv(plan.rig_T_world)
    for fragment in plan.fragments:
        task_fragment = cube_task.fragment(fragment.fragment_id)
        np.testing.assert_allclose(world_T_rig @ fragment.goal_rig, task_fragment.goal_world_T_fragment, atol=1e-12)
        np.testing.assert_allclose(world_T_rig @ fragment.initial_rig, task_fragment.initial_world_T_fragment,
                                   atol=1e-12)
    assert set(blue.touching) == {'red_fragment'} and set(plan.anchor.touching) == {'blue_fragment', 'green_fragment'}
    windows = [plan.fragment(k).window_s for k in plan.order]
    assert windows[0][0] == plan.options.lead_s and windows[0][1] == windows[1][0]
    assert plan.duration_s == pytest.approx(windows[-1][1] + plan.options.tail_s)


def test_bottom_layer_goes_first_without_a_plan(tmp_path):
    layout = write_layout(tmp_path / 'layout', plan=False)
    data = json.loads(layout.read_text())
    for piece in data['pieces']:
        if piece['name'] == 'frag_b':
            piece['goal_pose']['position'] = [0.0, 0.0, 0.04]  # frag_b now sits on top of the anchor
    layout.write_text(json.dumps(data))
    task = task_from_layout(layout, anchor_id='frag_a')
    plan = export.plan_export(task)
    assert plan.order == ('green_fragment', 'blue_fragment')


def test_the_tasks_plan_sets_the_order_and_each_insertion_in_the_rig(cube_task):
    plan = export.plan_export(cube_task)
    assert plan.order == ('blue_fragment', 'green_fragment')  # frag_b, then frag_c
    np.testing.assert_allclose(plan.fragment('blue_fragment').insertion_rig, (-1, 0, 0), atol=1e-12)
    np.testing.assert_allclose(plan.fragment('green_fragment').insertion_rig, (1, 0, 0), atol=1e-12)
    assert plan.anchor.insertion_rig is None
    yawed = export.plan_export(cube_task, export.ExportOptions(yaw_deg=90.0))  # world -x turns to rig -y
    np.testing.assert_allclose(yawed.fragment('blue_fragment').insertion_rig, (0, -1, 0), atol=1e-12)


def test_the_plans_order_wins_over_the_bottom_layer_rule(tmp_path):
    layout = write_layout(tmp_path / 'layout')
    data = json.loads(layout.read_text())
    data['source']['ownership_repair']['plan'].reverse()  # frag_c first; the bottom-layer rule puts frag_b first
    layout.write_text(json.dumps(data))
    assert export.plan_export(task_from_layout(layout, anchor_id='frag_a')).order == ('green_fragment', 'blue_fragment')


def straight_tail(plan, object_id):
    """Video rig positions and rotations of a fragment over the last part of its window (0.72-0.95 of it)."""
    poses = export.video_poses(plan)
    index = [f.object_id for f in plan.fragments].index(object_id)
    low, high = plan.fragment(object_id).window_s
    frames = [t for t in range(len(poses)) if 0.72 <= (t / plan.options.fps - low) / (high - low) <= 0.95]
    return poses[frames, index, :3], poses[frames, index, 3:]


@pytest.mark.parametrize('object_id, exit_', [('blue_fragment', (1.0, 0.0, 0.0)), ('green_fragment', (-1.0, 0.0, 0.0))])
def test_the_stand_in_video_ends_each_placement_with_its_straight_insertion(cube_task, object_id, exit_):
    plan = export.plan_export(cube_task)
    fragment, exit_ = plan.fragment(object_id), np.asarray(exit_)
    start, above, over, begin, goal = export.waypoints(plan, fragment)
    np.testing.assert_allclose(start, fragment.start_rig[:3, 3])
    np.testing.assert_allclose(goal, fragment.goal_rig[:3, 3])
    np.testing.assert_allclose(np.cross(begin - goal, exit_), 0, atol=1e-12)  # it goes in along the plan's line
    placed = [plan.anchor] + ([plan.fragment('blue_fragment')] if object_id == 'green_fragment' else [])
    at_begin = fragment.points(fragment.goal_rig) + (begin - goal)
    for other in placed:  # where it starts going in it is clear, along that line, of what is in place
        gap = (at_begin @ exit_).min() - (other.points(other.goal_rig) @ exit_).max()
        assert gap == pytest.approx(plan.options.clearance_m) if other is plan.anchor else gap >= 0
    assert above[2] == over[2] and np.allclose(over[:2], begin[:2])  # it comes down outside the skull
    positions, quaternions = straight_tail(plan, object_id)
    assert len(positions) > 10
    np.testing.assert_allclose(np.cross(positions - goal, exit_), 0, atol=1e-9)
    along = (positions - goal) @ exit_
    assert (np.diff(along) <= 1e-12).all() and along.min() >= -1e-12 and along.max() <= (begin - goal) @ exit_ + 1e-12
    np.testing.assert_allclose(np.abs(quaternions @ export.pose_to_wxyz(fragment.goal_rig)[3:]), 1, atol=1e-9)


def test_a_fragment_planned_from_above_comes_straight_down(tmp_path):
    layout = write_layout(tmp_path / 'layout')
    data = json.loads(layout.read_text())
    data['source']['ownership_repair']['plan'][0]['exit_direction'] = [0, 0, 1]
    layout.write_text(json.dumps(data))
    plan = export.plan_export(task_from_layout(layout, anchor_id='frag_a'))
    blue = plan.fragment('blue_fragment')
    np.testing.assert_allclose(blue.insertion_rig, (0, 0, -1), atol=1e-12)
    begin, goal = export.waypoints(plan, blue)[3:]
    assert np.allclose(begin[:2], goal[:2])
    assert begin[2] - goal[2] == pytest.approx(0.04 + plan.options.clearance_m)  # over the 4 cm anchor, clear
    positions, _ = straight_tail(plan, 'blue_fragment')
    np.testing.assert_allclose(positions[:, :2], np.broadcast_to(goal[:2], positions[:, :2].shape), atol=1e-9)
    assert (np.diff(positions[:, 2]) <= 1e-12).all()
    semantics = export.scene_semantics(plan, IDENTITY)
    assert semantics['subgoals'][0]['action'].endswith('with a last straight move downward.')


def test_the_subgoals_and_notes_carry_the_plan(cube_task):
    plan = export.plan_export(cube_task)
    semantics = export.scene_semantics(plan, IDENTITY)
    blue, green = semantics['subgoals']
    assert (blue['object_refs'][0], green['object_refs'][0]) == ('blue_fragment', 'green_fragment')
    assert blue['action'].endswith('with a last straight move along rig -x (toward the left station).')
    assert 'last straight move along rig -x (toward the left station)' in blue['hand_roles']['left']
    assert green['action'].endswith('with a last straight move along rig +x (toward the right station).')
    assert any('assembly plan' in note for note in semantics['notes'])
    note = runner.stage3_note(manifest_for(plan))
    assert ('1. the blue skull fragment, along rig -x (toward the left station); 2. the green skull fragment, along '
            'rig +x (toward the right station).') in note


def test_the_shared_bench_names_directions_by_its_own_landmarks(cube_task):
    plan = export.plan_export(cube_task, export.ExportOptions(bench='crx_yam'))  # rig = the world, shifted down
    blue = export.scene_semantics(plan, IDENTITY)['subgoals'][0]
    assert blue['action'].endswith('with a last straight move along rig -x (toward the left arm).')
    assert export.insertion_text((0, 1, 0), 'crx_yam') == 'along rig +y (away from the third-person camera)'
    manifest = {**manifest_for(plan), 'variant': export.VARIANTS['crx_yam']}
    assert '1. the blue skull fragment, along rig -x (toward the left arm)' in runner.stage3_note(manifest)


def test_without_a_plan_the_note_names_no_order(tmp_path):
    plan = export.plan_export(task_from_layout(write_layout(tmp_path / 'layout', plan=False), anchor_id='frag_a'))
    assert 'assembly plan' not in runner.stage3_note(manifest_for(plan))
    semantics = export.scene_semantics(plan, IDENTITY)
    assert not any('assembly plan' in note for note in semantics['notes'])
    assert all(goal['action'].endswith('against the skull.') for goal in semantics['subgoals'])


IDENTITY = {'path': '/x/video.mp4', 'sha256': 'a' * 64, 'frames': 540, 'fps': 30.0, 'duration_s': 18.0,
            'width': 640, 'height': 480}


def test_acting_hand_prefers_a_reach_witness_on_the_fragments_side(cube_task):
    plan = export.plan_export(cube_task)
    blue = plan.fragment('blue_fragment')  # starts on the left
    assert export.choose_hand(blue, {'reachable_sides': ['left', 'right'], 'best_side': 'right'})[0] == 'left'
    assert export.choose_hand(blue, {'reachable_sides': ['right'], 'best_side': 'right'}) == (
        'right', 'object_reach_check pose witness, other side only')
    assert export.choose_hand(blue, {'reachable_sides': [], 'best_side': 'right'})[0] == 'right'
    assert export.choose_hand(blue, None) == ('left', 'no object_reach_check report; side of the robot it starts on')
    settled = {f.object_id: export.pose_to_wxyz(f.initial_rig) for f in plan.fragments}
    moved = plan.with_settled(settled, {'blue_fragment': {'reachable_sides': ['right']}})
    assert moved.fragment('blue_fragment').acting_hand == 'right' and moved.anchor.acting_hand is None
    assert plan.fragment('blue_fragment').acting_hand == 'left'  # with_settled returns a new plan


def test_scene_semantics_carry_relative_goals_at_the_task_tolerances(cube_task):
    plan = export.plan_export(cube_task)
    semantics = export.scene_semantics(plan, IDENTITY)
    required = ('task_prompt', 'task_summary', 'environment', 'hands', 'objects', 'subgoals', 'task_success')
    assert all(key in semantics for key in required) and semantics['input_identity'] == IDENTITY
    assert [g['index'] for g in semantics['subgoals']] == [1, 2]
    assert semantics['hands']['used'] == ['left', 'right']
    anchor_goal = cube_task.fragment('frag_a').goal_world_T_fragment
    for subgoal in semantics['subgoals']:
        object_id = subgoal['object_refs'][0]
        assert subgoal['object_refs'][1] == 'red_fragment' and subgoal['motion'] == 'place'
        assert 0 <= subgoal['time_window_s'][0] < subgoal['time_window_s'][1] <= IDENTITY['duration_s']
        (predicate,) = subgoal['goal_predicates']
        assert predicate['class'] == 'relative_pose' and predicate['bodies'] == {'A': object_id, 'B': 'red_fragment'}
        params = predicate['params']
        assert (params['pos_tol_m'], params['rot_tol_deg']) == (0.003, 3.0)
        expected = np.linalg.inv(anchor_goal) @ cube_task.fragment(plan.id_map[object_id]).goal_world_T_fragment
        np.testing.assert_allclose(params['rel_pos'], expected[:3, 3], atol=1e-12)
        np.testing.assert_allclose(params['rel_quat'], [1, 0, 0, 0], atol=1e-12)
        assert subgoal['goal'] == f'The {plan.fragment(object_id).name} is released.'
    success = semantics['task_success_predicates']
    assert len(success) == 3 and success[-1]['bodies'] == {'A': 'red_fragment', 'B': 'table'}
    assert success[-1]['params'] == {'reference_frame': 0, 'pos_tol_m': 0.003, 'rot_tol_deg': 3.0}
    text = semantics['task_success'].lower()
    for name in ('blue skull fragment', 'green skull fragment'):
        assert f'the {name} is released' in text
    # The fixture holds the anchor (red) and nothing ever lets go of it: "released" would compile to DexAgent's
    # release_settle, which a fixed body touching nothing fails ("it is falling"); "has not moved" is its invariant.
    assert 'the red skull fragment has not moved' in text and 'the red skull fragment is released' not in text
    assert any('not a human demonstration' in note for note in semantics['notes'])


def test_video_poses_start_settle_and_end_at_the_goals(cube_task):
    plan = export.plan_export(cube_task)
    poses = export.video_poses(plan)
    assert poses.shape == (int(round(plan.duration_s * plan.options.fps)), 3, 7)
    for index, fragment in enumerate(plan.fragments):
        np.testing.assert_allclose(export.pose_matrix(poses[0, index]), fragment.start_rig, atol=1e-9)
        np.testing.assert_allclose(export.pose_matrix(poses[-1, index]), fragment.goal_rig, atol=1e-9)
    anchor = [f.object_id for f in plan.fragments].index('red_fragment')
    assert np.ptp(poses[:, anchor], axis=0).max() < 1e-12
    blue = [f.object_id for f in plan.fragments].index('blue_fragment')
    window = plan.fragment('blue_fragment').window_s
    middle = poses[int((window[0] + window[1]) / 2 * plan.options.fps), blue]
    assert middle[2] > 0.04 + plan.options.clearance_m - 1e-9  # above every fragment top while it glides


def test_project_box_contains_every_projected_point(cube_task):
    plan = export.plan_export(cube_task)
    for fragment in plan.fragments:
        x0, y0, x1, y1 = export.project_box(plan.camera, fragment.points(fragment.start_rig), 0)
        assert 0 <= x0 < x1 <= plan.camera['width'] and 0 <= y0 < y1 <= plan.camera['height']
        assert (x1 - x0) < plan.camera['width'] / 4


def fake_worker(plan, fail=()):
    def run(python, command, spec, log_path, *, env, timeout_s):
        if command == 'stage':
            return {'settled': {f.object_id: export.pose_to_wxyz(f.initial_rig) for f in plan.fragments},
                    'settle': {'simulated_s': 2.0, 'change': {}},
                    'bodies': {f.object_id: [f.root_body] for f in plan.fragments},
                    'reach': {'blue_fragment': {'reachable_sides': ['right'], 'best_side': 'right'},
                              'green_fragment': {'reachable_sides': [], 'best_side': None}},
                    'reach_report': {'ok': False, 'verdict': 'probe'}}
        if command == 'media':
            Path(spec['video']).write_bytes(b'fake mp4')
            return {'identity': IDENTITY, 'scene_checks': {}, 'versions': {'mujoco': '3.10.0'},
                    'objects': {f.object_id: {'iou': 0.9, 'verifiers': {'silhouette_iou_check': {'ok': True}}}
                                for f in plan.fragments}}
        assert command == 'check'
        return {name: {'ok': name not in fail} for name in export.REQUIRED_CHECKS}
    return run


@pytest.mark.parametrize('fail', [(), ('source_reference_provenance_check',)])
def test_export_writes_stage_documents_and_a_manifest(cube_task, tmp_path, monkeypatch, fail):
    plan = export.plan_export(cube_task)
    monkeypatch.setattr(export.dexenv, 'verify_clone', lambda clone, commit: commit)
    monkeypatch.setattr(export.dexenv, 'run_worker', fake_worker(plan, fail))
    out = tmp_path / 'inputs'
    manifest = export.export_task(cube_task, out, clone=tmp_path / 'clone', python='python', env={})
    assert manifest['checks_ok'] is (not fail) and manifest['checks_failed'] == list(fail)
    semantics = json.loads((out / 'stage1' / 'scene_semantics.json').read_text())
    recon = json.loads((out / 'stage2' / 'scene_recon.json').read_text())
    assert semantics['input_identity'] == IDENTITY and recon['scene_semantic'] == '../stage1/scene_semantics.json'
    for entry in recon['objects']:
        assert (out / 'stage2' / entry['asset_xml']).is_file() and entry['mask'] == f'masks/{entry["id"]}.png'
        assert entry['root_body'] == f'{entry["id"]}_{entry["id"]}'
        assert set(entry['properties_covered']) == set(next(o for o in semantics['objects']
                                                            if o['id'] == entry['id'])['properties'])
    assert manifest['dexagent_commit'] == export.DEXAGENT_COMMIT and manifest['video_is_stand_in'] is True
    assert {o['object_id']: o['fragment_id'] for o in manifest['objects']} == plan.id_map
    np.testing.assert_allclose(manifest['rig_T_world'], plan.rig_T_world)
    asset = 'stage2/assets/blue_fragment/blue_fragment.xml'
    assert manifest['files_sha256'][asset] == export.file_sha256(out / asset)
    hands = {o['object_id']: (o['acting_hand'], o['hand_reason']) for o in manifest['objects']}
    assert hands['blue_fragment'] == ('right', 'object_reach_check pose witness, other side only')
    assert hands['green_fragment'][0] == 'right' and 'no object_reach_check' in hands['green_fragment'][1]
    assert {g['acting_hand'] for g in semantics['subgoals']} == set(semantics['hands']['used']) == {'right'}
    assert json.loads((out / 'export_checks.json').read_text())['object_reach_check']['informational'] is True
    with pytest.raises(ValueError, match='not empty'):
        export.export_task(cube_task, out, clone=tmp_path / 'clone', python='python', env={})


def test_export_hands_its_workers_absolute_paths(cube_task, tmp_path, monkeypatch):
    plan = export.plan_export(cube_task)
    monkeypatch.setattr(export.dexenv, 'verify_clone', lambda clone, commit: commit)
    seen, run = [], fake_worker(plan)

    def recording(python, command, spec, log_path, *, env, timeout_s):
        seen.append(Path(log_path))
        return run(python, command, spec, log_path, env=env, timeout_s=timeout_s)

    monkeypatch.setattr(export.dexenv, 'run_worker', recording)
    monkeypatch.chdir(tmp_path)
    manifest = export.export_task(cube_task, 'inputs', clone=tmp_path / 'clone', python='python', env={})
    assert manifest['checks_ok'] and len(seen) == 3 and all(path.is_absolute() for path in seen)  # workers run elsewhere


def test_export_refuses_a_tilted_initial_pose(tmp_path):
    layout = write_layout(tmp_path / 'layout')
    data = json.loads(layout.read_text())
    tilt = np.radians(5)
    data['pieces'][1]['staging_pose']['orientation_wxyz'] = [float(np.cos(tilt)), float(np.sin(tilt)), 0, 0]
    layout.write_text(json.dumps(data))
    task = task_from_layout(layout, anchor_id='frag_a')
    with pytest.raises(ValueError, match='planar'):
        export.plan_export(task)


# ------------------------------------------------------------------ transport


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self.stream))  # the binding control is a beta

    def stream(self, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs)))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeStream(reply)


def message(*blocks, stop_reason='tool_use', input_tokens=1000, output_tokens=100, cache_write=0, cache_read=0,
            **extra):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                            cache_creation_input_tokens=cache_write, cache_read_input_tokens=cache_read)
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, model='claude-opus-5-5', id='msg_1',
                           usage=usage, **extra)


THINK_1 = {'type': 'thinking', 'thinking': 'look at the workspace first', 'signature': 'sig-one'}
CALL_1 = {'type': 'tool_use', 'id': 'toolu_1', 'name': 'list_dir', 'input': {'path': '.'}}
TOOLS = [{'name': name, 'description': name, 'input_schema': {'type': 'object', 'properties': {}}}
         for name in ('list_dir', 'finish', 'report_blocked')]


def request(directory, turn, messages, session='s1', system='stage prompt'):
    value = {'schema': 'dexagent.external.v1', 'session': session, 'id': f'{session}-{turn:04d}', 'turn': turn,
             'system': system, 'messages': messages, 'tools': TOOLS}
    transport.atomic_json(Path(directory) / 'requests' / f'{value["id"]}.json', value)
    return value


def response(directory, value):
    return json.loads((Path(directory) / 'responses' / f'{value["id"]}.json').read_text())


USER_0 = {'role': 'user', 'content': [{'type': 'text', 'text': 'Workspace: /w'}]}


def second_turn_messages(first_answer):
    return [USER_0, {'role': 'assistant', 'content': first_answer},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_1',
                                           'content': [{'type': 'text', 'text': '{"entries": []}'}]}]}]


def responder(tmp_path, client, cap=5.0, who='dexagent:test', **kwargs):
    """A relay spending from the pool in tmp_path/pool (made with `cap` dollars if it is not there yet)."""
    pool = BudgetPool(tmp_path / 'pool') if (tmp_path / 'pool' / 'pool.json').exists() else \
        BudgetPool.create(tmp_path / 'pool', cap, note='test')
    return transport.Responder(tmp_path / 'transport', pool=pool, who=who, client=client, sleep=lambda s: None,
                               **kwargs)


def ledger(tmp_path):
    return [json.loads(line) for line in (tmp_path / 'llm_ledger.jsonl').read_text().splitlines()]


def test_transport_keeps_thinking_and_passes_only_text_and_tool_use(tmp_path):
    client = FakeClient([message(THINK_1, {'type': 'text', 'text': 'listing'}, CALL_1),
                         message({'type': 'tool_use', 'id': 'toolu_2', 'name': 'finish', 'input': {'summary': 'ok'}})])
    agent = responder(tmp_path, client)
    first = request(tmp_path / 'transport', 1, [USER_0])
    assert agent.answer_pending() == 1
    visible = response(tmp_path / 'transport', first)['content']
    assert [b['type'] for b in visible] == ['text', 'tool_use'] and visible[1]['id'] == 'toolu_1'
    second = request(tmp_path / 'transport', 2, second_turn_messages(visible))
    agent.answer_pending()
    sent = client.calls[1]['messages']
    assert sent[1] == {'role': 'assistant', 'content': [THINK_1, {'type': 'text', 'text': 'listing'}, CALL_1]}
    assert sent[2]['content'][0]['type'] == 'tool_result' and sent[2]['content'][0]['tool_use_id'] == 'toolu_1'
    assert response(tmp_path / 'transport', second)['content'][0]['name'] == 'finish'
    first_call = client.calls[0]
    assert first_call['model'] == 'claude-opus-5-5' and first_call['output_config'] == {'effort': 'high'}
    assert first_call['cache_control'] == {'type': 'ephemeral', 'ttl': '1h'} and first_call['max_tokens'] == 64000
    assert first_call['thinking'] == {'type': 'adaptive', 'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}
    assert first_call['betas'] == ['thinking-binding-controls-2026-08-01']
    assert [t['name'] for t in first_call['tools']] == ['list_dir', 'finish', 'report_blocked']
    assert not (tmp_path / 'transport' / 'operator_checkpoint.json').exists()
    assert [r['event'] for r in ledger(tmp_path)] == ['call', 'call'] and agent.calls == 2


def test_transport_logs_the_thinking_blocks_the_api_dropped(tmp_path):
    dropped = SimpleNamespace(type='thinking_dropped', path='messages.1.content.0', reason='prefix_mismatch')
    client = FakeClient([message(THINK_1, CALL_1, input_transformations=[dropped]), message(CALL_1)])
    agent = responder(tmp_path, client)
    first = request(tmp_path / 'transport', 1, [USER_0])
    agent.answer_pending()
    request(tmp_path / 'transport', 2, second_turn_messages(response(tmp_path / 'transport', first)['content']))
    agent.answer_pending()
    events = ledger(tmp_path)
    assert events[0]['input_transformations'] == [{'type': 'thinking_dropped', 'path': 'messages.1.content.0',
                                                   'reason': 'prefix_mismatch'}]
    assert [e['event'] for e in events] == ['call', 'call'] and events[1]['input_transformations'] is None


def assert_stopped(tmp_path, value, kind):
    checkpoint = json.loads((tmp_path / 'transport' / 'operator_checkpoint.json').read_text())
    assert checkpoint['kind'] == 'administrative_checkpoint' and checkpoint['stop'] == kind
    content = response(tmp_path / 'transport', value)['content']
    calls = [b for b in content if b['type'] == 'tool_use']
    assert [c['name'] for c in calls] == ['report_blocked'] and calls[0]['id'].startswith('operator_stop_')
    assert ledger(tmp_path)[-1]['event'] == 'stop' and ledger(tmp_path)[-1]['kind'] == kind
    return content


@pytest.mark.parametrize('tamper', ['assistant', 'earlier_user', 'session', 'system'])
def test_transport_stops_when_dexagent_history_differs(tmp_path, tamper):
    client = FakeClient([message(THINK_1, CALL_1), message(CALL_1)])
    agent = responder(tmp_path, client)
    first = request(tmp_path / 'transport', 1, [USER_0])
    agent.answer_pending()
    messages = second_turn_messages(response(tmp_path / 'transport', first)['content'])
    session, system = 's1', 'stage prompt'
    if tamper == 'assistant':
        messages[1]['content'][0]['input'] = {'path': '/etc'}
    elif tamper == 'earlier_user':
        messages[0] = {'role': 'user', 'content': [{'type': 'text', 'text': 'Workspace: /elsewhere'}]}
    elif tamper == 'session':
        session = 's2'
    else:
        system = 'another prompt'
    second = request(tmp_path / 'transport', 2, messages, session=session, system=system)
    agent.answer_pending()
    assert len(client.calls) == 1
    assert_stopped(tmp_path, second, 'transcript')
    assert agent.summary()['stop']['kind'] == 'transcript'


@pytest.mark.parametrize('stop_reason,kind', [('refusal', 'refusal'), ('max_tokens', 'max_tokens'),
                                              ('model_context_window_exceeded', 'context_window'),
                                              ('pause_turn', 'stop_reason')])
def test_transport_never_passes_tool_calls_of_a_refused_or_truncated_turn(tmp_path, stop_reason, kind):
    client = FakeClient([message(THINK_1, CALL_1, stop_reason=stop_reason,
                                 stop_details=SimpleNamespace(category='general_harms'))])
    agent = responder(tmp_path, client)
    first = request(tmp_path / 'transport', 1, [USER_0])
    agent.answer_pending()
    content = assert_stopped(tmp_path, first, kind)
    assert all(b.get('id') != 'toolu_1' for b in content)
    events = ledger(tmp_path)
    assert events[0]['event'] == 'call' and events[0]['stop_reason'] == stop_reason
    assert agent.spent_usd == pytest.approx((1000 * 4 + 100 * 20) / 1e6)


def test_transport_budget_stop_happens_before_the_call(tmp_path):
    client = FakeClient([message(CALL_1)])
    agent = responder(tmp_path, client, cap=0.05, max_tokens=4000)  # worst case 4000 x $20/MTok = $0.08 > cap
    first = request(tmp_path / 'transport', 1, [USER_0])
    agent.answer_pending()
    assert client.calls == []
    assert_stopped(tmp_path, first, 'budget')
    assert agent.spent_usd == 0.0 and transport.ledger_totals(tmp_path / 'llm_ledger.jsonl') == (0.0, 0)


def test_transport_budget_bound_grows_from_the_measured_input(tmp_path):
    # realistic numbers: the first call's input within its own bound, its output within max_tokens
    client = FakeClient([message(CALL_1, input_tokens=2000, output_tokens=900), message(CALL_1)])
    price = transport.Price.for_model('claude-opus-5-5')
    first_cost = (2000 * 4 + 900 * 20) / 1e6
    second_bound = price.worst_case(2000 + 900 + 500, 1000, '1h')  # plus the new message bytes
    cap = first_cost + second_bound - 1e-6
    agent = responder(tmp_path, client, cap=cap, max_tokens=1000)
    first = request(tmp_path / 'transport', 1, [USER_0])
    agent.answer_pending()
    assert response(tmp_path / 'transport', first)['content'][0]['id'] == 'toolu_1'
    second = request(tmp_path / 'transport', 2, second_turn_messages(response(tmp_path / 'transport', first)['content']))
    agent.answer_pending()
    assert len(client.calls) == 1
    assert_stopped(tmp_path, second, 'budget')
    assert agent.spent_usd == pytest.approx(first_cost) and agent.spent_usd <= cap


@pytest.mark.parametrize('text_bytes,calls', [(200_000, 1), (10, 2)])
def test_transport_budget_bound_counts_the_new_message(tmp_path, text_bytes, calls):
    client = FakeClient([message(CALL_1, input_tokens=1000, output_tokens=100), message(CALL_1)])
    price = transport.Price.for_model('claude-opus-5-5')
    cap = (1000 * 4 + 100 * 20) / 1e6 + price.worst_case(1000 + 100 + 500 + 50_000, 1000, '1h')
    agent = responder(tmp_path, client, cap=cap, max_tokens=1000)
    first = request(tmp_path / 'transport', 1, [USER_0])
    agent.answer_pending()
    messages = second_turn_messages(response(tmp_path / 'transport', first)['content'])
    messages[2]['content'][0]['content'][0]['text'] = 'x' * text_bytes  # a long tool result
    second = request(tmp_path / 'transport', 2, messages)
    agent.answer_pending()
    assert len(client.calls) == calls
    if calls == 1:
        assert_stopped(tmp_path, second, 'budget')


def test_serve_never_pays_for_a_request_left_by_an_exited_process(tmp_path):
    client = FakeClient([message(CALL_1)])
    agent = responder(tmp_path, client)
    request(tmp_path / 'transport', 1, [USER_0])
    assert agent.serve(SimpleNamespace(poll=lambda: 0)) == 'exited' and client.calls == []
    states = iter([None, 0])
    assert agent.serve(SimpleNamespace(poll=lambda: next(states))) == 'exited' and len(client.calls) == 1


def test_ledger_prices_cache_reads_and_writes(tmp_path):
    usage = SimpleNamespace(input_tokens=2000, output_tokens=500, cache_creation_input_tokens=30000,
                            cache_read_input_tokens=100000,
                            cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=10000,
                                                           ephemeral_1h_input_tokens=20000))
    record = transport.usage_record(usage, '1h')
    assert (record['cache_write_5m_tokens'], record['cache_write_1h_tokens']) == (10000, 20000)
    expected = (2000 * 4 + 10000 * 5 + 20000 * 8 + 100000 * 0.2 + 500 * 20) / 1e6
    assert transport.Price.for_model('claude-opus-5-5').cost(record) == pytest.approx(expected)
    no_breakdown = transport.usage_record(SimpleNamespace(input_tokens=0, output_tokens=0,
                                                          cache_creation_input_tokens=300, cache_read_input_tokens=0),
                                          '1h')
    assert no_breakdown['cache_write_1h_tokens'] == 300
    client = FakeClient([message(CALL_1, input_tokens=2000, output_tokens=500, cache_write=0, cache_read=100000)])
    agent = responder(tmp_path, client)
    request(tmp_path / 'transport', 1, [USER_0])
    agent.answer_pending()
    spent, calls = transport.ledger_totals(tmp_path / 'llm_ledger.jsonl')
    assert calls == 1 and spent == pytest.approx((2000 * 4 + 100000 * 0.2 + 500 * 20) / 1e6)
    with pytest.raises(ValueError, match='no price'):
        transport.Price.for_model('claude-unknown')


def test_transport_retries_rate_limits_and_charges_possibly_billed_failures(tmp_path):
    anthropic = pytest.importorskip('anthropic')
    httpx2 = pytest.importorskip('httpx2')
    req = httpx2.Request('POST', 'https://api.anthropic.com/v1/messages')
    rate = anthropic.RateLimitError('rate limited', response=httpx2.Response(429, request=req), body=None)
    overloaded = anthropic.OverloadedError('overloaded', response=httpx2.Response(529, request=req), body=None)
    dropped = anthropic.APIConnectionError(request=req)
    mid_stream = anthropic.APIStatusError('error event', response=httpx2.Response(200, request=req), body=None)
    client = FakeClient([rate, overloaded, dropped, mid_stream, message(CALL_1)])
    agent = responder(tmp_path, client, retries=4, max_tokens=1000)
    first = request(tmp_path / 'transport', 1, [USER_0])
    agent.answer_pending()
    assert response(tmp_path / 'transport', first)['content'][0]['id'] == 'toolu_1'
    events = ledger(tmp_path)
    assert [e['event'] for e in events] == ['api_error'] * 4 + ['call']
    assert [e['charged_usd'] > 0 for e in events[:4]] == [False, False, True, True]
    bad = anthropic.BadRequestError('bad', response=httpx2.Response(400, request=req), body=None)
    other = responder(tmp_path / 'other', FakeClient([bad]))
    second = request(tmp_path / 'other' / 'transport', 1, [USER_0])
    other.answer_pending()
    assert_stopped(tmp_path / 'other', second, 'api_error')


def test_transport_only_reads_images_from_its_own_folder(tmp_path):
    (tmp_path / 'transport' / 'images').mkdir(parents=True)
    png = tmp_path / 'transport' / 'images' / 'abc.png'
    png.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9a'
                                     'wAAAABJRU5ErkJggg=='))
    client = FakeClient([message(CALL_1)])
    agent = responder(tmp_path, client)
    request(tmp_path / 'transport', 1, [{'role': 'user', 'content': [{'type': 'image', 'mime': 'image/png',
                                                                       'path': str(png)}]}])
    agent.answer_pending()
    assert client.calls[0]['messages'][0]['content'][0]['source']['media_type'] == 'image/png'
    secret = tmp_path / 'secret.png'
    secret.write_bytes(png.read_bytes())
    other = responder(tmp_path / 'b', FakeClient([message(CALL_1)]))
    value = request(tmp_path / 'b' / 'transport', 1, [{'role': 'user', 'content': [
        {'type': 'image', 'mime': 'image/png', 'path': str(secret)}]}])
    other.answer_pending()
    assert_stopped(tmp_path / 'b', value, 'transcript')


def test_responder_spends_only_from_the_shared_pool(tmp_path):
    with pytest.raises(TypeError):
        transport.Responder(tmp_path, client=FakeClient([]), who='dexagent:t')  # no pool: no per-trial cap either
    with pytest.raises(ValueError, match='BudgetPool'):
        transport.Responder(tmp_path, pool=5.0, client=FakeClient([]), who='dexagent:t')
    with pytest.raises(ValueError, match='who'):
        transport.Responder(tmp_path, pool=BudgetPool.create(tmp_path / 'pool', 1.0, note='t'), client=FakeClient([]),
                            who='')


def test_two_relays_share_one_pool(tmp_path):
    """DexAgent's and ASPIRE's relays spend one pool: what one spent is not there for the other."""
    probe = responder(tmp_path / 'probe', FakeClient([message(CALL_1)]), cap=100.0, max_tokens=1000)
    request(tmp_path / 'probe' / 'transport', 1, [USER_0])
    probe.answer_pending()
    bound = ledger(tmp_path / 'probe')[0]['bound_usd']  # what the first call of either relay reserves
    first_cost = (1000 * 4 + 100 * 20) / 1e6
    pool = BudgetPool.create(tmp_path / 'pool', bound + first_cost / 2, note='t')  # one call fits, not two
    a = transport.Responder(tmp_path / 'a' / 'transport', pool=pool, who='dexagent:a', max_tokens=1000,
                            client=FakeClient([message(CALL_1)]), sleep=lambda s: None)
    b_client = FakeClient([message(CALL_1)])
    b = transport.Responder(tmp_path / 'b' / 'transport', pool=pool, who='aspire:b', max_tokens=1000, client=b_client,
                            sleep=lambda s: None)
    request(tmp_path / 'a' / 'transport', 1, [USER_0])
    a.answer_pending()
    second = request(tmp_path / 'b' / 'transport', 1, [USER_0])
    b.answer_pending()
    assert b_client.calls == [] and b.summary()['stop']['kind'] == 'budget'
    assert response(tmp_path / 'b' / 'transport', second)['content'][-1]['name'] == 'report_blocked'
    status = pool.status()
    assert status['spent_usd'] == pytest.approx(first_cost) and status['by_who'] == {'dexagent:a': pytest.approx(first_cost)}
    assert status['open'] == 0


def test_the_pool_charges_failed_attempts_as_the_ledger_does(tmp_path):
    anthropic = pytest.importorskip('anthropic')
    httpx2 = pytest.importorskip('httpx2')
    req = httpx2.Request('POST', 'https://api.anthropic.com/v1/messages')
    rate = anthropic.RateLimitError('rate limited', response=httpx2.Response(429, request=req), body=None)
    dropped = anthropic.APIConnectionError(request=req)
    agent = responder(tmp_path, FakeClient([rate, dropped, message(CALL_1)]), retries=4, max_tokens=1000)
    request(tmp_path / 'transport', 1, [USER_0])
    agent.answer_pending()
    pool_log = [json.loads(line) for line in (tmp_path / 'pool' / 'pool.jsonl').read_text().splitlines()]
    settled = [(e['outcome'], e['usd'] > 0) for e in pool_log if e['event'] == 'settle']
    assert settled == [('api_error', False), ('api_error', True), ('call', True)]  # a dropped call may have been billed
    spent, calls = transport.ledger_totals(tmp_path / 'llm_ledger.jsonl')
    assert BudgetPool(tmp_path / 'pool').status()['spent_usd'] == pytest.approx(spent) and calls == 1


def test_a_call_dearer_than_its_bound_is_charged_and_stops_the_run(tmp_path):
    client = FakeClient([message(CALL_1, input_tokens=2_000_000, output_tokens=10), message(CALL_1)])
    agent = responder(tmp_path, client, cap=50.0, max_tokens=1000)
    first = request(tmp_path / 'transport', 1, [USER_0])
    agent.answer_pending()
    content = assert_stopped(tmp_path, first, 'budget')
    assert all(b.get('id') != 'toolu_1' for b in content)  # the turn's tool calls are not run
    assert BudgetPool(tmp_path / 'pool').status()['spent_usd'] == pytest.approx((2_000_000 * 4 + 10 * 20) / 1e6)


# ------------------------------------------------------------------ trajectory


def manifest_for(plan):
    return {'rig_T_world': plan.rig_T_world.tolist(), 'task': plan.task.to_record(),
            'anchor_object_id': plan.anchor.object_id,
            'objects': [{'object_id': f.object_id, 'fragment_id': f.fragment_id, 'free_joint': f.free_joint,
                         'body_T_fragment': np.eye(4).tolist(), 'start_rig_wxyz': export.pose_to_wxyz(f.start_rig),
                         'name': f.name} for f in plan.fragments]}


def write_trajectory(path, plan, final, extra_columns=('robot_left_arm_dof_joint1',)):
    names, last = list(extra_columns), [0.1] * len(extra_columns)
    for fragment in plan.fragments:
        if fragment.free_joint:
            names += [f'{fragment.free_joint}[{k}]' for k in range(7)]
            last += export.pose_to_wxyz(final[fragment.object_id])
    qpos = np.vstack([np.zeros(len(names)), np.array(last)])
    np.savez(path, qpos=qpos, joint_names=np.array(names), fps=np.array(30.0))
    return path


def test_trajectory_import_scores_the_final_poses_in_the_task_frame(cube_task, tmp_path):
    plan = export.plan_export(cube_task)
    manifest = manifest_for(plan)
    goals = {f.object_id: f.goal_rig for f in plan.fragments}
    score = trajectory.score_trajectory(cube_task, manifest, write_trajectory(tmp_path / 'ok.npz', plan, goals),
                                        settled=True)
    assert score.success and score.placed == 2
    shifted = dict(goals, blue_fragment=goals['blue_fragment'] @ np.array(
        [[1, 0, 0, 0.005], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float))
    score = trajectory.score_trajectory(cube_task, manifest, write_trajectory(tmp_path / 'off.npz', plan, shifted),
                                        settled=True)
    blue = next(s for s in score.fragments if s.fragment_id == 'frag_b')
    assert not score.success and blue.position_error_m == pytest.approx(0.005)
    poses = trajectory.world_poses(trajectory.final_rig_poses(tmp_path / 'ok.npz', manifest), manifest)
    np.testing.assert_allclose(poses['frag_c'], cube_task.fragment('frag_c').goal_world_T_fragment, atol=1e-9)
    assert pose_errors(poses['frag_a'], cube_task.fragment('frag_a').goal_world_T_fragment)[0] < 1e-12


def test_frames_compose_with_a_yawed_rig_and_rotated_goals(tmp_path):
    layout = write_layout(tmp_path / 'layout')
    data = json.loads(layout.read_text())
    turn, tilt = np.radians(30) / 2, np.radians(20) / 2
    goals = {'frag_b': [float(np.cos(turn)), 0.0, 0.0, float(np.sin(turn))],
             'frag_c': [float(np.cos(tilt)), float(np.sin(tilt)), 0.0, 0.0]}
    for piece in data['pieces']:
        if piece['name'] in goals:
            piece['goal_pose']['orientation_wxyz'] = goals[piece['name']]
            piece['goal_pose']['position'][2] = 0.01  # the tilted cube's lower edge stays above the table
    layout.write_text(json.dumps(data))
    task = task_from_layout(layout, anchor_id='frag_a')
    plan = export.plan_export(task, export.ExportOptions(yaw_deg=20.0, pad_xy=(0.1, 0.4)))
    anchor_goal = task.fragment('frag_a').goal_world_T_fragment
    for fragment in plan.fragments:
        if fragment is plan.anchor:
            continue
        expected = np.linalg.inv(anchor_goal) @ task.fragment(fragment.fragment_id).goal_world_T_fragment
        params = export.relative_pose_params(plan.anchor.goal_rig, fragment.goal_rig, 0.003, 3.0)
        np.testing.assert_allclose(params['rel_pos'], expected[:3, 3], atol=1e-12)
        w, x, y, z = params['rel_quat']
        np.testing.assert_allclose(Rotation.from_quat([x, y, z, w]).as_matrix(), expected[:3, :3], atol=1e-12)
    manifest = manifest_for(plan)
    final = write_trajectory(tmp_path / 'goal.npz', plan, {f.object_id: f.goal_rig for f in plan.fragments})
    poses = trajectory.world_poses(trajectory.final_rig_poses(final, manifest), manifest)
    for fragment_id, pose in poses.items():
        np.testing.assert_allclose(pose, task.fragment(fragment_id).goal_world_T_fragment, atol=1e-9)
    assert trajectory.score_trajectory(task, manifest, final, settled=True).success


def test_trajectory_import_rejects_missing_or_duplicated_columns(cube_task, tmp_path):
    plan = export.plan_export(cube_task)
    with pytest.raises(ValueError, match='expected exactly one'):
        trajectory.free_joint_columns(['a[0]', 'a[1]'], 'a')
    with pytest.raises(ValueError, match='expected exactly one'):
        trajectory.free_joint_columns([f'a[{k}]' for k in range(7)] + ['a[3]'], 'a')
    manifest = manifest_for(plan)
    other = dict(manifest, task=dict(manifest['task'], task_id='another'))
    goals = {f.object_id: f.goal_rig for f in plan.fragments}
    with pytest.raises(ValueError, match='different task'):
        trajectory.score_trajectory(cube_task, other, write_trajectory(tmp_path / 't.npz', plan, goals),
                                    settled=True)


def test_trajectory_columns_of_the_shipped_dexagent_example():
    path = Path(dexenv.dexagent_source()) / 'examples/default/place_cup_in_bowl/stage3/trajectory.npz'
    if not path.exists():
        pytest.skip(f'no DexAgent clone at {dexenv.dexagent_source()}')
    with np.load(path, allow_pickle=False) as data:
        names, qpos = list(data['joint_names']), data['qpos']
    columns = trajectory.free_joint_columns(names, 'cup_cup_free')
    assert columns == list(range(columns[0], columns[0] + 7))
    final = trajectory._pose_from_qpos(qpos[-1, columns], 'cup')
    np.testing.assert_allclose(final[:3, 3], qpos[-1, columns[:3]])
    assert abs(np.linalg.norm(qpos[-1, columns[3:]]) - 1) < 1e-6


# ------------------------------------------------------------------ runner


def test_run_status_maps_dexagent_and_transport_outcomes():
    status = runner.run_status
    assert status(0, {'finished': True}, None, False, True) == 'completed'
    assert status(0, {'finished': True}, None, False, False) == 'error'
    assert status(1, {'finished': False}, {'kind': 'budget'}, False, False) == 'budget_exhausted'
    assert status(1, {'finished': False}, {'kind': 'refusal'}, False, False) == 'rejected'
    assert status(1, {'finished': False}, {'kind': 'transcript'}, False, False) == 'error'
    assert status(-9, {'status': 'running'}, None, True, False) == 'timeout'
    assert status(1, {'finished': False, 'failure': 'budget exhausted before finish was called'}, None, False,
                  False) == 'budget_exhausted'
    assert status(1, {'finished': False, 'stop_reason': 'wall_time_budget'}, None, False, False) == 'timeout'
    assert status(1, {'finished': False, 'failure': 'blocked: no grasp'}, None, False, False) == 'blocked'
    assert status(1, {'finished': False, 'stop_reason': 'search_stalled'}, None, False, False) == 'blocked'
    mask_gate = {'finished': False, 'stage': 'stage3', 'failure': 'Stage 2 source-reference mask provenance is unverified',
                 'upstream': {'stage': 'stage2', 'evidence': {}}}  # run_stage3.py:512-516
    video_gate = {'finished': False, 'failure': 'Source video and Stage 1 provenance are inconsistent',
                  'upstream': {'stage': 'stage1', 'evidence': {}}}  # agent/input_identity.py:97-99
    assert status(1, mask_gate, None, False, False) == 'error' and status(1, video_gate, None, False, False) == 'error'
    backend = {'finished': False, 'failure': 'blocked: model backend failed: TimeoutError: external model did not answer'}
    assert status(1, backend, None, False, False) == 'error'
    agent_upstream = {'finished': False, 'failure': 'blocked: the tray is too small', 'upstream': {'stage': 'stage2'}}
    assert status(1, agent_upstream, None, False, False) == 'blocked'
    assert status(1, None, None, False, False) == 'error'


def test_trial_paths_are_absolute_even_from_a_relative_trial(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = runner.TrialConfig(Path('runs') / 't1', llm='mock:turns.json')
    assert config.trial_dir == tmp_path / 'runs' / 't1'
    clone = config.trial_dir / 'DexAgent'
    command = runner.stage3_command(config, clone, config.trial_dir / 'inputs')
    paths = [arg.removeprefix('mock:') for arg in command if arg.endswith(('.py', '.yaml', '.json', '.mp4'))]
    assert len(paths) == 6 and all(Path(path).is_absolute() for path in paths)
    yaml = runner.trial_yaml(config, clone, config.trial_dir / 'inputs', 'note')
    assert Path(yaml['agent']['transport_dir']).is_absolute() and Path(yaml['run']['output_dir']).is_absolute()
    with pytest.raises(ValueError, match='finite'):
        runner.TrialConfig(tmp_path, llm='mock:x.json', max_wall_minutes=float('nan'))
    with pytest.raises(FileNotFoundError, match='budget pool'):
        runner.TrialConfig(tmp_path, llm='claude', budget_pool=tmp_path / 'no_pool_here')


def test_a_trial_insets_the_fragment_colliders_as_the_protocol_says(tmp_path):
    with pytest.raises(ValueError, match='inset'):
        runner.TrialConfig(tmp_path / 't', llm='mock:x.json', export=export.ExportOptions(collider_inset_m=0.0005))
    options = export.ExportOptions()
    record = options.to_record()
    record['collider_inset_m']['vertical'] = 1.0  # a record is a copy: the options keep the protocol's inset
    assert options.collider_inset_m['vertical'] == 0.0001


def test_a_trial_holds_the_anchor_by_a_fixture_as_the_protocol_says(tmp_path):
    """A free anchor is a different task; `export` may still write one for diagnostics."""
    with pytest.raises(ValueError, match='fixture'):
        runner.TrialConfig(tmp_path / 't', llm='mock:x.json', export=export.ExportOptions(anchor_mode='free'))
    assert runner.TrialConfig(tmp_path / 't', llm='mock:x.json').export.anchor_mode == 'fixed'


def test_trial_config_and_command(tmp_path):
    with pytest.raises(ValueError, match='budget_pool'):
        runner.TrialConfig(tmp_path, llm='claude')
    assert runner.TrialConfig(tmp_path, llm='claude', scoring_only=True).budget_pool is None  # rescoring calls no model
    with pytest.raises(ValueError, match='stage4_draws'):
        runner.TrialConfig(tmp_path, llm='mock:x.json', stage4_draws=0)
    config = runner.TrialConfig(tmp_path / 't', llm='mock:turns.json', methods=())
    yaml = runner.trial_yaml(config, tmp_path / 'clone', tmp_path / 't' / 'inputs', 'a note')
    assert yaml['agent']['provider'] == 'external'
    assert yaml['agent']['transport_dir'] == str(tmp_path / 't' / 'transport')
    assert yaml['examples']['reuse'] is False and yaml['examples']['save_assets'] is False
    assert yaml['stage3']['methods'] == [] and yaml['stage3']['strict_quality'] is True
    assert yaml['stage3']['stage4_draws'] == 3 and yaml['stage3']['notes'] == 'a note'
    assert yaml['inputs']['setup'] == str(tmp_path / 'clone' / 'setup')
    command = runner.stage3_command(config, tmp_path / 'clone', tmp_path / 't' / 'inputs')
    assert command[2].endswith('stage3_trajectory_generation/run_stage3.py') and '--yes' not in command
    assert command[-2:] == ['--llm', f'mock:{(Path("turns.json")).resolve()}']


def test_dexagent_runs_in_the_sandbox_and_a_live_trial_cannot_opt_out(tmp_path):
    pool = BudgetPool.create(tmp_path / 'pool', 1.0, note='t')
    with pytest.raises(ValueError, match='sandbox'):
        runner.TrialConfig(tmp_path / 'live', llm='claude', budget_pool=tmp_path / 'pool', sandbox=False)
    assert pool.status()['spent_usd'] == 0.0
    config = runner.TrialConfig(tmp_path / 't', llm='mock:turns.json')
    config.trial_dir.mkdir()
    clone, outside = config.trial_dir / 'DexAgent', tmp_path / 'stage2_elsewhere'
    outside.mkdir()
    launch = runner.stage3_launch(config, clone, config.trial_dir / 'inputs', extra_readonly=[outside])
    command = runner.stage3_command(config, clone, config.trial_dir / 'inputs')
    assert launch[0] == 'bwrap' and '--unshare-all' in launch and launch[launch.index('--') + 1:] == command
    binds = [(launch[i], launch[i + 2]) for i in range(len(launch) - 2) if launch[i].endswith('bind')]
    assert [inside for flag, inside in binds if flag == '--bind'] == [str(config.trial_dir)]  # the only writable place
    readonly = {inside for flag, inside in binds if flag == '--ro-bind'}
    venv = Path(config.python).absolute().parent.parent
    assert {str(venv), str(REPO_ROOT / 'core'), str(REPO_ROOT / 'tasks'), str(outside)} <= readonly
    assert not any(inside == str(Path.home()) or inside in (str(REPO_ROOT), str(REPO_ROOT / 'runs')) for _, inside in binds)
    assert launch[launch.index('--chdir') + 1] == str(config.trial_dir)
    turns = tmp_path / 'scripted' / 'turns.json'  # a mock trial's scripted turns, wherever they are: that file alone
    turns.parent.mkdir()
    turns.write_text('{}')
    (turns.parent / 'neighbour.txt').write_text('not for the sandbox')
    mocked = runner.sandbox_paths(runner.TrialConfig(tmp_path / 'm', llm=f'mock:{turns}'))
    assert (turns.resolve(), turns.resolve()) in [entry for entry in mocked['readonly'] if isinstance(entry, tuple)]
    assert not any(str(entry if not isinstance(entry, tuple) else entry[1]) == str(turns.parent)
                   for entry in mocked['readonly'])
    plain = runner.TrialConfig(tmp_path / 'u', llm='mock:turns.json', sandbox=False)
    assert runner.stage3_launch(plain, clone, plain.trial_dir / 'inputs') == runner.stage3_command(plain, clone,
                                                                                                 plain.trial_dir / 'inputs')


def test_simulated_episodes_are_counted_from_the_trace(tmp_path):
    ran = {'ok': True, 'result': {'ok': False, 'subgoals': [{'index': 1, 'ok': False}]}}
    events = [{'event': 'tool', 'name': 'execute_subgoal_plan', 'args': {'plan': 'p.json'}, 'result': ran},
              {'event': 'tool', 'name': 'execute_subgoal_plan', 'args': {'plan': 'p.json', 'dry_run': True},
               'result': {'ok': True, 'result': {'ok': True, 'dry_run': True}}},
              {'event': 'tool', 'name': 'execute_subgoal_plan', 'args': {'plan': 'p.json'},  # search_progress.py:378
               'result': {'ok': False, 'error': 'unchanged_failed_execution', 'execution_performed': False}},
              {'event': 'tool', 'name': 'execute_subgoal_plan', 'args': {}, 'result': {'ok': False, 'error': 'TypeError'}},
              {'event': 'tool', 'name': 'search_subgoal_variants', 'args': {},
               'result': {'ok': True, 'result': {'ok': False, 'evaluations': 12}}},
              {'event': 'tool', 'name': 'library_search', 'args': {}, 'result': {'ok': True, 'result': {}}},
              {'event': 'tool', 'name': 'execute_subgoal_plan', 'args': {'plan': 'q.json', 'resume': True}, 'result': ran}]
    (tmp_path / 'trace.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\nnot json\n')
    assert runner.count_executions(tmp_path, None) == 2 + 12
    assert runner.count_executions(tmp_path / 'missing', None) is None


def test_kill_group_stops_the_whole_process_group():
    child = ('import subprocess, sys, time; '
             'subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]); time.sleep(60)')
    process = subprocess.Popen([sys.executable, '-c', child], start_new_session=True)
    time.sleep(0.5)
    started = time.monotonic()
    code = runner.kill_group(process, grace_s=5)
    assert code != 0 and time.monotonic() - started < 10
    deadline = time.monotonic() + 10  # the orphaned grandchild is reaped by init shortly after its SIGTERM
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail('a process of the killed group is still alive')


def test_heldout_seeds_need_a_flag_a_records_file_and_run_once(cube_task, tmp_path):
    from dataclasses import replace
    heldout = configured_initial_condition(cube_task, 7, json.loads(DEFAULT_CONFIG.read_text()))
    with pytest.raises(ValueError, match='held out'):
        runner.guard_seed(heldout, allow_heldout=False)
    with pytest.raises(ValueError, match='must be recorded'):
        runner.guard_seed(heldout, allow_heldout=True)
    records = tmp_path / 'runs.jsonl'
    runner.guard_seed(heldout, allow_heldout=True, records=records)
    append_records(records, [RunRecord('dexagent', export.VARIANT, heldout.task_id, 7, 'heldout', 'error', None,
                                       backend=runner.BACKEND, test_only=True)])
    with pytest.raises(ValueError, match='already has records'):
        runner.guard_seed(heldout, allow_heldout=True, records=records)
    runner.guard_seed(replace(heldout, seed=8), allow_heldout=True, records=records)
    runner.guard_seed(replace(cube_task, seed=101, split='practice'), allow_heldout=False, records=records)
    done = tmp_path / 'done'
    (done / 'inputs').mkdir(parents=True)
    (done / 'inputs' / 'export_manifest.json').write_text(json.dumps({'schema': export.EXPORT_SCHEMA,
                                                                      'task': heldout.to_record()}))
    with pytest.raises(ValueError, match='already has records'):  # rescoring would count the seed twice
        runner.rescore(done, records=records)
    with pytest.raises(ValueError, match='must be recorded'):  # before any clone, export or model call
        runner.run_trial(heldout, runner.TrialConfig(tmp_path / 'trial', llm='mock:none'), allow_heldout=True)
    with pytest.raises(ValueError, match='not its layout under the canonical protocol'):
        runner.run_trial(replace(cube_task, position_tolerance_m=0.05), runner.TrialConfig(tmp_path / 'trial',
                                                                                           llm='mock:none'))
    assert not (tmp_path / 'trial').exists()


def test_a_failed_heldout_trial_is_recorded_and_its_seed_is_used(cube_task, tmp_path):
    heldout = configured_initial_condition(cube_task, 7, json.loads(DEFAULT_CONFIG.read_text()))
    records = tmp_path / 'runs.jsonl'

    def config(name):  # no DexAgent clone to copy from: the trial fails after its seed was claimed
        return runner.TrialConfig(tmp_path / name, llm='mock:none', source=str(tmp_path / 'no_clone'),
                                  python=str(tmp_path / 'no_venv' / 'bin' / 'python'))

    with pytest.raises(RuntimeError, match='git clone'):
        runner.run_trial(heldout, config('first'), records=records, allow_heldout=True)
    lines = [json.loads(line) for line in records.read_text().splitlines()]
    assert [(line['seed'], line['status'], line['test_only'], line['backend']) for line in lines] == [
        (7, 'error', True, runner.BACKEND)]
    assert 'RuntimeError' in lines[0]['notes'] and lines[0]['artifacts']['heldout_claim']
    with pytest.raises(ValueError, match='already has records'):
        runner.run_trial(heldout, config('second'), records=records, allow_heldout=True)
    with pytest.raises(ValueError, match='already claimed'):
        runner.run_trial(heldout, config('third'), records=tmp_path / 'other.jsonl', allow_heldout=True)
    assert not (tmp_path / 'second').exists() and not (tmp_path / 'third').exists()


def test_scrubbed_environment_carries_no_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'x')
    env = dexenv.scrubbed_env(tmp_path / 'trial', python='/opt/venv/bin/python')
    assert dexenv.credential_keys(env) == []
    assert not any(value in ('sk-ant-test', 'sk-test') for value in env.values())
    assert env['HOME'] == str((tmp_path / 'trial' / 'home').resolve()) and env['MUJOCO_GL'] == 'osmesa'
    assert env['TMPDIR'].startswith(str((tmp_path / 'trial').resolve()))
    with pytest.raises(ValueError, match='credentials'):
        dexenv.run_worker('python', 'check', {}, tmp_path / 'w.log', env={'ANTHROPIC_API_KEY': 'x'})


CANARY = '''
import os, subprocess, sys
sys.path.insert(0, sys.argv[2])
from tasks.dual_arm_crag_skull.baselines.dexagent import dexenv
if sys.argv[1] == '1':
    dexenv.protect_process()
child = ("import sys\\ntry:\\n    print(b'CANARY_KEY_VALUE' in open(f'/proc/{sys.argv[1]}/environ', 'rb').read())\\n"
         "except OSError:\\n    print('denied')")
out = subprocess.run([sys.executable, '-c', child, str(os.getpid())], env={'PATH': '/usr/bin:/bin'},
                     start_new_session=True, capture_output=True, text=True)
print(out.stdout.strip())
'''


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='prctl and /proc are Linux')
def test_protect_process_hides_the_runner_environment_from_children(tmp_path):
    script = tmp_path / 'canary.py'
    script.write_text(CANARY)
    root = str(Path(__file__).resolve().parents[3])
    env = {'PATH': '/usr/bin:/bin', 'CANARY': 'CANARY_KEY_VALUE', 'PYTHONDONTWRITEBYTECODE': '1'}
    seen = {flag: subprocess.run([sys.executable, '-B', str(script), flag, root], env=env, capture_output=True,
                                 text=True, timeout=60).stdout.strip() for flag in ('0', '1')}
    assert seen['0'] in ('True', 'denied')  # 'True' here: a scrubbed child reads its parent's environment
    assert seen['1'] == 'denied'


def test_venv_check_detects_changed_package_files(tmp_path):
    import hashlib
    site = tmp_path / 'venv' / 'lib' / 'python3.11' / 'site-packages'
    (site / 'pkg').mkdir(parents=True)
    (site / 'pkg-1.0.dist-info').mkdir()
    module = site / 'pkg' / '__init__.py'
    module.write_text('x = 1\n')
    digest = base64.urlsafe_b64encode(hashlib.sha256(module.read_bytes()).digest()).rstrip(b'=').decode()
    (site / 'pkg-1.0.dist-info' / 'RECORD').write_text(f'pkg/__init__.py,sha256={digest},6\n'
                                                       'pkg-1.0.dist-info/RECORD,,\n')
    python = tmp_path / 'venv' / 'bin' / 'python'
    assert dexenv.verify_venv(python) == []
    module.write_text('x = 2\n')
    assert dexenv.verify_venv(python) == [str(module.resolve())]


def test_inputs_check_names_changed_or_missing_files(tmp_path):
    for name in ('a.txt', 'b.txt'):
        (tmp_path / name).write_text(name)
    manifest = {'files_sha256': {name: export.file_sha256(tmp_path / name) for name in ('a.txt', 'b.txt')}}
    manifest['files_sha256']['c.txt'] = '0' * 64
    assert runner.verify_inputs(tmp_path, manifest) == ['c.txt']
    (tmp_path / 'a.txt').write_text('changed')
    assert runner.verify_inputs(tmp_path, manifest) == ['a.txt', 'c.txt']


def test_stage3_note_names_the_real_anchor(cube_task):
    plan = export.plan_export(cube_task)
    note = runner.stage3_note(manifest_for(plan))
    assert 'The anchor, the red skull fragment, is fixed at its assembled pose.' in note
    free = manifest_for(plan)
    free['objects'][0]['free_joint'] = 'red_fragment_red_fragment_free'
    assert 'red skull fragment, starts at' in runner.stage3_note(free)


# ------------------------------------------------------------------ integration (DexAgent venv and clone)

AVAILABLE, WHY = dexenv.available()
needs_dexagent = pytest.mark.skipif(not AVAILABLE, reason=WHY)


def dexagent_env(trial):
    return dexenv.scrubbed_env(trial, dexenv.dexagent_python())


@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    if not AVAILABLE:
        pytest.skip(WHY)
    root = tmp_path_factory.mktemp('dexagent_export')
    task = task_from_layout(write_layout(root / 'layout'), anchor_id='frag_a')
    clone = root / 'DexAgent'
    dexenv.fresh_clone(dexenv.dexagent_source(), clone, export.DEXAGENT_COMMIT)
    manifest = export.export_task(task, root / 'inputs', clone=clone, python=dexenv.dexagent_python(),
                                  env=dexagent_env(root))
    return SimpleNamespace(root=root, task=task, clone=clone, manifest=manifest, inputs=root / 'inputs')


@needs_dexagent
def test_integration_export_passes_dexagent_own_checks(exported):
    checks = json.loads((exported.inputs / 'export_checks.json').read_text())
    for name in export.REQUIRED_CHECKS:
        assert checks[name]['ok'] is True, (name, checks[name].get('issues'))
    assert exported.manifest['checks_ok'] is True
    success = checks['goal_language_coverage_check']['clauses']['$.task_success']
    assert all(clause['covered'] for clause in success)  # DexAgent compiles every clause, the anchor's too
    assert sorted(clause['clause'].split(' ', 2)[2] for clause in success) == ['has not moved'] + ['is released'] * (
        len(success) - 1)
    # every visual mesh is closed as DexAgent's blocking object_visual_penetration_audit reads it (MuJoCo's compiled
    # mesh through trimesh with processing), so that audit can run at all
    script = textwrap.dedent('''
        import sys, mujoco, trimesh
        m = mujoco.MjModel.from_xml_path(sys.argv[1])
        for i in range(m.nmesh):
            v = m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]]
            f = m.mesh_face[m.mesh_faceadr[i]:m.mesh_faceadr[i] + m.mesh_facenum[i]]
            if m.mesh(i).name.endswith('_visual'):
                print(m.mesh(i).name, trimesh.Trimesh(v, f, process=True).is_watertight)
    ''')
    closed = subprocess.run([dexenv.dexagent_python(), '-B', '-c', script, str(exported.inputs / 'stage2' / 'scene.xml')],
                            capture_output=True, text=True, timeout=600, env=dexagent_env(exported.root))
    objects = {f"{o['object_id']}_{o['object_id']}_visual" for o in exported.manifest['objects']}
    flags = dict(line.split() for line in closed.stdout.splitlines() if line.split()[0] in objects)
    assert closed.returncode == 0 and set(flags) == objects, closed.stderr[-2000:]
    assert all(flag == 'True' for flag in flags.values()), flags
    issues = checks['scene_recon_manifest_check']['issues']
    assert issues and all('.skill' in issue for issue in issues)  # only the external skill name is flagged
    assert checks['goal_penetration']['exact_boxes']['max_penetration_m'] <= 1e-4
    dexenv.verify_clone(exported.clone, export.DEXAGENT_COMMIT)  # the export never writes into the clone


def run_stage3_mock(exported, name, turns, checkpoint=True, semantic=None, recon=None):
    trial = exported.root / name
    (trial / 'transport').mkdir(parents=True)
    config = runner.TrialConfig(trial, llm=f'mock:{trial / "turns.json"}', max_rounds=3, max_wall_minutes=10)
    (trial / 'turns.json').write_text(json.dumps({'stage3': turns}))
    if checkpoint:
        (trial / 'transport' / 'operator_checkpoint.json').write_text('{"kind": "administrative_checkpoint"}')
    (trial / 'dexagent.yaml').write_text(json.dumps(runner.trial_yaml(config, exported.clone, exported.inputs, 'test')))
    extra = [exported.clone, exported.inputs] + [Path(path).parent for path in (semantic, recon) if path]
    command = runner.stage3_launch(config, exported.clone, exported.inputs, extra_readonly=extra)  # in the sandbox
    assert command[0] == 'bwrap'
    if semantic:
        command[command.index('--semantic') + 1] = str(semantic)
    if recon:
        command[command.index('--recon') + 1] = str(recon)
    result = subprocess.run(command, cwd=trial, env=dexagent_env(trial), capture_output=True, text=True, timeout=1200)
    return trial, result, trajectory.read_run(config.public)


REPORT_BLOCKED = [[{'type': 'tool_use', 'name': 'report_blocked', 'input': {'reason': 'mock stop'}}]]


@needs_dexagent
def test_integration_stage3_accepts_the_exported_inputs(exported):
    trial, result, run = run_stage3_mock(exported, 'mock_ok', REPORT_BLOCKED)
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-2000:]
    assert run['stop_reason'] == 'operator_checkpoint' and run['rounds'] == 1  # the agent loop was reached
    sandbox = runner.sandbox_paths(runner.TrialConfig(trial, llm=f'mock:{trial / "turns.json"}'),
                                   extra_readonly=[exported.clone, exported.inputs])
    directory, reason = trajectory.accepted_trajectory(trial / 'runs' / runner.RUN_NAME / 'stage3', clone=exported.clone,
                                                       python=dexenv.dexagent_python(), env=dexagent_env(trial),
                                                       log_path=trial / 'delivery.log',
                                                       recon=exported.inputs / 'stage2' / 'scene_recon.json',
                                                       sandbox=sandbox)  # the re-check also runs in the sandbox
    assert directory is None and 'has not finished successful verification' in reason
    dexenv.verify_clone(exported.clone, export.DEXAGENT_COMMIT)


@needs_dexagent
def test_integration_stage3_gates_reject_a_changed_mask_or_video_identity(exported):
    tampered = exported.root / 'tampered_stage2'
    import shutil
    shutil.copytree(exported.inputs / 'stage2', tampered)
    (tampered / 'scene_recon.json').write_text((exported.inputs / 'stage2' / 'scene_recon.json').read_text().replace(
        '../stage1/scene_semantics.json', str(exported.inputs / 'stage1' / 'scene_semantics.json')))
    from PIL import Image
    mask = np.array(Image.open(tampered / 'masks' / 'blue_fragment.png'))
    mask[np.argwhere(mask == 0)[0][0], np.argwhere(mask == 0)[0][1]] = 255
    Image.fromarray(mask).save(tampered / 'masks' / 'blue_fragment.png')
    _, _, run = run_stage3_mock(exported, 'mock_mask', REPORT_BLOCKED, recon=tampered / 'scene_recon.json')
    assert run['failure'] == 'Stage 2 source-reference mask provenance is unverified'
    semantics = json.loads((exported.inputs / 'stage1' / 'scene_semantics.json').read_text())
    semantics['input_identity']['sha256'] = '0' * 64
    changed = exported.root / 'changed_semantics.json'
    changed.write_text(json.dumps(semantics))
    _, _, run = run_stage3_mock(exported, 'mock_identity', REPORT_BLOCKED, semantic=changed)
    assert run['failure'] == 'Source video and Stage 1 provenance are inconsistent'


DEX_SIDE = '''
import json, sys
from pathlib import Path
clone, work, cap_stop = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == '1'
sys.path.insert(0, str(clone))
from agent.config import load_config
from agent.external import ExternalLLM
from agent.runner import StageRunner
cfg = load_config(clone / 'dexagent.yaml')
cfg['run']['output_dir'] = str(work / 'runs')
cfg['agent']['provider'] = 'external'
llm = ExternalLLM(work / 'transport', timeout_s=120)
runner = StageRunner(cfg, llm, work / 'ws', 'Test stage: list the workspace, then call finish.',
                     [lambda ws: {'ok': True, 'issues': []}], max_rounds=4, stage='stage3')
out = runner.run('Workspace: ' + str(work / 'ws'))
print(json.dumps({k: out.get(k) for k in ('finished', 'rounds', 'stop_reason', 'tool_calls')}))
'''


@needs_dexagent
@pytest.mark.parametrize('cap,first_input', [(5.0, 1000), (0.17, 15000)])
def test_integration_responder_drives_dexagent_external_transport(exported, tmp_path, cap, first_input):
    script = tmp_path / 'dex_side.py'
    script.write_text(DEX_SIDE)
    client = FakeClient([message(THINK_1, CALL_1, input_tokens=first_input, output_tokens=100),
                         message({'type': 'tool_use', 'id': 'toolu_2', 'name': 'finish', 'input': {'summary': 'ok'}})])
    process = subprocess.Popen([dexenv.dexagent_python(), '-I', '-B', str(script), str(exported.clone), str(tmp_path),
                                '0'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=dexagent_env(tmp_path))
    agent = transport.Responder(tmp_path / 'transport', pool=BudgetPool.create(tmp_path / 'pool', cap, note='t'),
                                who='dexagent:integration', client=client, max_tokens=100,
                                sleep=lambda s: time.sleep(min(s, 0.05)))
    assert agent.serve(process, deadline=time.monotonic() + 300) == 'exited'
    out, err = process.communicate()
    result = json.loads(out.strip().splitlines()[-1])
    if cap == 5.0:
        assert result['finished'] is True and len(client.calls) == 2
        assert client.calls[1]['messages'][1]['content'][0] == THINK_1
    else:  # the first call reserves $0.15 (DexAgent's first request bounds 18.6k input tokens), costs $0.06 on 15k;
        # the next could cost $0.13 with $0.11 left in the pool: DexAgent stops at the checkpoint
        assert result['finished'] is False and result['stop_reason'] == 'operator_checkpoint', err[-2000:]
        assert len(client.calls) == 1 and agent.summary()['stop']['kind'] == 'budget'
