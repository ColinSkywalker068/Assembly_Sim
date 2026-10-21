import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tasks.dual_arm_crag_skull.__main__ import DEFAULT_CONFIG
from tasks.dual_arm_crag_skull.baselines.task import (AssemblyTask, canonical_protocol, check_splits,
                                                      configured_initial_condition, file_sha256, initial_condition,
                                                      pose_from_layout, pose_to_wxyz, require_canonical, split_of,
                                                      success_criterion, task_from_layout, verify_task)
from tasks.dual_arm_crag_skull.tests.baselines_fixtures import BOX_F, BOX_V, GOALS, PAD, STAGING, write_layout

SPLITS = {'practice': [101, 125], 'selection': [126, 130], 'heldout': [1, 100]}
PLACEMENT = dict(max_shift_m=0.03, max_yaw_deg=30.0, gap_m=0.02)
CONFIG = json.loads(DEFAULT_CONFIG.read_text())


@pytest.fixture
def task(tmp_path):
    return task_from_layout(write_layout(tmp_path / 'layout'), anchor_id='frag_a')


# ------------------------------------------------------------------ an independent placement check


def _clip(subject, clipper):
    """Sutherland-Hodgman: the part of convex polygon `subject` inside convex `clipper` (both counter-clockwise)."""
    output = [np.asarray(p, dtype=float) for p in subject]
    for i in range(len(clipper)):
        a, b = np.asarray(clipper[i], dtype=float), np.asarray(clipper[(i + 1) % len(clipper)], dtype=float)
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        points, output = output, []
        for j, current in enumerate(points):
            previous = points[j - 1]
            if side(current) >= 0:
                if side(previous) < 0:
                    output.append(_crossing(previous, current, side))
                output.append(current)
            elif side(previous) >= 0:
                output.append(_crossing(previous, current, side))
        if not output:
            break
    return output


def _crossing(p, q, side):
    s, t = side(p), side(q)
    return p + (q - p) * (s / (s - t))


def overlap_area(first, second):
    points = _clip(first, second)
    if len(points) < 3:
        return 0.0
    x, y = np.array(points).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _point_segment(p, a, b):
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def distance(first, second):
    """Distance between two convex polygons, 0 when they overlap."""
    if overlap_area(first, second) > 1e-12:
        return 0.0
    pairs = [(p, poly) for p, poly in [*((p, second) for p in first), *((p, first) for p in second)]]
    return min(_point_segment(p, poly[i], poly[(i + 1) % len(poly)]) for p, poly in pairs for i in range(len(poly)))


def rectangle(bounds):
    xmin, xmax, ymin, ymax = bounds
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


def assert_valid_placement(base, seeded, *, max_shift_m, max_yaw_deg, gap_m):
    """Every staged fragment of `seeded` is on the table with the gap, off the pad, `gap_m` from the other fragments
    (the anchor included) and the exclusions, and moved within the limits from `base` (yaw only)."""
    footprints = {key: seeded.fragment(key).footprint() for key in seeded.fragment_ids}
    xmin, xmax, ymin, ymax = seeded.table_bounds_m
    for key in seeded.moving_ids:
        polygon = footprints[key]
        assert polygon[:, 0].min() >= xmin + gap_m - 1e-9 and polygon[:, 0].max() <= xmax - gap_m + 1e-9, key
        assert polygon[:, 1].min() >= ymin + gap_m - 1e-9 and polygon[:, 1].max() <= ymax - gap_m + 1e-9, key
        assert overlap_area(polygon, rectangle(seeded.pad_bounds_m)) < 1e-12, f'{key} is on the pad'
        for zone in seeded.exclusions_m:
            assert distance(polygon, rectangle(zone)) >= gap_m - 1e-9, f'{key} is too close to {zone}'
        for other in seeded.fragment_ids:
            if other != key:
                assert distance(polygon, footprints[other]) >= gap_m - 1e-9, f'{key} is too close to {other}'
        before, after = base.fragment(key), seeded.fragment(key)
        shift = np.linalg.norm(after.footprint().mean(axis=0) - before.footprint().mean(axis=0))
        assert shift <= max_shift_m + 1e-12
        start, moved = before.initial_world_T_fragment, after.initial_world_T_fragment
        assert moved[2, 3] == pytest.approx(start[2, 3])
        np.testing.assert_allclose(moved[:3, 2], start[:3, 2], atol=1e-12)  # yaw only: the up axis is kept
        yaw = Rotation.from_matrix(np.array(moved[:3, :3] @ start[:3, :3].T)).as_rotvec()
        assert abs(np.rad2deg(yaw[2])) <= max_yaw_deg + 1e-9
        np.testing.assert_array_equal(after.goal_world_T_fragment, before.goal_world_T_fragment)


def write_crowded_layout(directory):
    """Each staged fragment sits 2.5 cm from one thing it must keep away from: the anchor (whose goal touches the pad
    edge), an exclusion zone, the table edge, the pad far from the anchor. Staged pairs: the next test."""
    directory.mkdir(parents=True)
    goals = {'frag_a': (0.08, 0.0), 'frag_b': (-0.05, 0.05), 'frag_c': (-0.05, -0.05), 'frag_e': (0.05, -0.05),
             'frag_f': (-0.05, 0.0)}
    staging = {'frag_a': (-0.3, 0.3), 'frag_b': (0.145, 0.0), 'frag_c': (0.255, 0.0), 'frag_e': (-0.3, -0.355),
               'frag_f': (-0.125, 0.0)}
    pieces = []
    for name in goals:
        np.savez(directory / f'{name}.npz', v=BOX_V, f=BOX_F)
        pieces.append({'name': name, 'mesh': f'{name}.npz', 'local_bounds': [[-0.02, -0.02, 0.0], [0.02, 0.02, 0.04]],
                       'goal_pose': {'position': [*goals[name], 0.0], 'orientation_wxyz': [1.0, 0.0, 0.0, 0.0]},
                       'staging_pose': {'position': [*staging[name], 0.75], 'orientation_wxyz': [1.0, 0.0, 0.0, 0.0]}})
    path = directory / 'layout.json'
    path.write_text(json.dumps({'schema_version': 2, 'object_id': 'crowded', 'pieces': pieces, 'staging': {
        'pad_bounds': [-0.1, 0.1, -0.1, 0.1], 'table_bounds': [-0.5, 0.5, -0.4, 0.4], 'table_top_z': 0.75,
        'exclusions': [[0.3, 0.45, -0.1, 0.1]]}}))
    return path


# ------------------------------------------------------------------ the task itself


def test_goals_sit_on_the_pad_and_the_anchor_starts_at_its_goal(task, tmp_path):
    assert task.fragment_ids == ('frag_a', 'frag_b', 'frag_c') and task.moving_ids == ('frag_b', 'frag_c')
    np.testing.assert_allclose(task.world_T_assembly[:3, 3], [(PAD[0] + PAD[1]) / 2, (PAD[2] + PAD[3]) / 2, 0.75])
    for key, fragment in zip(task.fragment_ids, task.fragments):
        np.testing.assert_allclose(fragment.goal_world_T_fragment[:3, 3], np.add(GOALS[key], [0, 0, 0.75]))
        expected = fragment.goal_world_T_fragment if key == 'frag_a' else None
        start = fragment.initial_world_T_fragment
        if expected is None:
            np.testing.assert_allclose(start[:3, 3], STAGING[key])
        else:
            np.testing.assert_array_equal(start, expected)
        assert fragment.mesh_sha256 == file_sha256(fragment.mesh)
    layout = tmp_path / 'layout' / 'layout.json'
    assert task.layout_sha256 == file_sha256(layout)
    assert task.task_id == f'cubes-{task.layout_sha256[:12]}-anchor-frag_a'
    assert task.goal_source == 'oracle_layout' and task.seed is None and task.split is None
    criterion = success_criterion(CONFIG)
    assert (task.position_tolerance_m, task.rotation_tolerance_deg, task.settle_s) == (
        criterion['position_tolerance_m'], criterion['rotation_tolerance_deg'], criterion['settle_s'])


def test_assembly_frame_is_the_pad_centre_not_the_origin(tmp_path):
    path = write_layout(tmp_path / 'layout')
    data = json.loads(path.read_text())
    data['staging']['pad_bounds'] = [0.0, 0.2, -0.1, 0.3]
    path.write_text(json.dumps(data))
    task = task_from_layout(path, anchor_id='frag_a')
    np.testing.assert_allclose(task.world_T_assembly[:3, 3], [0.1, 0.1, 0.75])
    np.testing.assert_allclose(task.fragment('frag_b').goal_world_T_fragment[:3, 3], [0.14, 0.1, 0.75])


def test_task_round_trips_through_json_and_is_read_only(task):
    record = json.loads(json.dumps(task.to_record()))
    assert AssemblyTask.from_record(record).to_record() == task.to_record()
    with pytest.raises(ValueError):
        task.fragments[1].goal_world_T_fragment[0, 3] = 1.0
    with pytest.raises(ValueError):
        task.world_T_assembly[0, 3] = 1.0
    copy_ = task.goal_poses()['frag_b']
    copy_[0, 3] = 9.0
    assert task.fragment('frag_b').goal_world_T_fragment[0, 3] != 9.0


@pytest.mark.parametrize('change, message', [
    (lambda t: {'fragments': t.fragments[::-1]}, 'sorted'),
    (lambda t: {'fragments': t.fragments[:1]}, 'at least one'),
    (lambda t: {'anchor_id': 'frag_z'}, 'anchor'),
    (lambda t: {'goal_source': 'guess'}, 'goal_source'),
    (lambda t: {'seed': 3}, 'together'),
    (lambda t: {'seed': 3, 'split': 'train'}, 'split'),
    (lambda t: {'seed': True, 'split': 'heldout'}, 'seed'),
    (lambda t: {'position_tolerance_m': 0.0}, 'position_tolerance_m'),
    (lambda t: {'rotation_tolerance_deg': float('nan')}, 'rotation_tolerance_deg'),
    (lambda t: {'settle_s': -1.0}, 'settle_s'),
    (lambda t: {'pad_bounds_m': (0.2, -0.2, -0.2, 0.2)}, 'pad_bounds_m'),
])
def test_invalid_tasks_are_refused(task, change, message):
    with pytest.raises(ValueError, match=message):
        replace(task, **change(task))


def test_fragment_specs_are_checked(task):
    fragment = task.fragment('frag_b')
    with pytest.raises(ValueError, match='sha256'):
        replace(fragment, mesh_sha256='abc')
    with pytest.raises(ValueError, match='local_bounds'):
        replace(fragment, local_bounds=((0, 0, 0), (0.01, 0.01, 0.0)))
    with pytest.raises(ValueError, match='initial_world_T_fragment'):
        replace(fragment, initial_world_T_fragment=np.diag([2.0, 1.0, 1.0, 1.0]))


def test_layout_schema_and_quaternions_are_checked(tmp_path):
    path = write_layout(tmp_path / 'layout')
    data = json.loads(path.read_text())
    with pytest.raises(ValueError, match='schema_version 2'):
        path.write_text(json.dumps({**data, 'schema_version': 1}))
        task_from_layout(path, anchor_id='frag_a')
    with pytest.raises(ValueError, match='unit quaternion'):
        pose_from_layout({'position': [0, 0, 0], 'orientation_wxyz': [1.0, 0.0, 0.0, 0.1]})


def _edited(data, edit):
    data = copy.deepcopy(data)
    edit(data)
    return data


@pytest.mark.parametrize('edit, message', [
    (lambda d: d['pieces'][1].update(mesh='../elsewhere.npz'), 'outside the layout directory'),
    (lambda d: d['pieces'][1]['goal_pose'].update(position=[5.0, 3.0, 0.0]), 'not over the pad'),
    (lambda d: d['pieces'][1]['goal_pose'].update(position=[0.04, 0.0, -0.3]), 'below the table'),
    (lambda d: d['staging'].update(pad_bounds=[3.0, 3.4, 3.0, 3.4]), 'pad is not on the table'),
    (lambda d: d['staging']['exclusions'].append([0.9, 1.5, 0.0, 0.1]), 'not on the table'),
    (lambda d: d['pieces'][2]['staging_pose'].update(position=[-0.42, 0.11, 0.75]), 'overlaps'),
    (lambda d: d['pieces'][2]['staging_pose'].update(position=[0.1, 0.0, 0.75]), 'on the pad'),
    (lambda d: d['pieces'][2]['staging_pose'].update(position=[0.8, 0.0, 0.75]), 'overlaps'),
    (lambda d: d['pieces'][2]['staging_pose'].update(position=[0.0, 0.7, 0.75]), 'off the table'),
])
def test_malformed_layouts_are_refused(tmp_path, edit, message):
    path = write_layout(tmp_path / 'layout')
    (tmp_path / 'elsewhere.npz').write_bytes((tmp_path / 'layout' / 'frag_b.npz').read_bytes())
    path.write_text(json.dumps(_edited(json.loads(path.read_text()), edit)))
    with pytest.raises(ValueError, match=message):
        task_from_layout(path, anchor_id='frag_a')


def test_layout_quaternions_are_wxyz_with_a_non_negative_scalar():
    pose = pose_from_layout({'position': [1, 2, 3], 'orientation_wxyz': [np.cos(0.25), 0, 0, np.sin(0.25)]})
    np.testing.assert_allclose(Rotation.from_matrix(pose[:3, :3]).as_rotvec(), [0, 0, 0.5], atol=1e-12)
    np.testing.assert_allclose(pose_to_wxyz(pose), [1, 2, 3, np.cos(0.25), 0, 0, np.sin(0.25)], atol=1e-12)
    for rotation in Rotation.random(200, random_state=7):
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        values = pose_to_wxyz(matrix)
        assert values[3] >= 0
        np.testing.assert_allclose(pose_from_layout({'position': values[:3], 'orientation_wxyz': values[3:]}),
                                   matrix, atol=1e-12)


def test_seed_splits_must_be_disjoint_integer_bands():
    assert check_splits(SPLITS) == {'practice': (101, 125), 'selection': (126, 130), 'heldout': (1, 100)}
    assert [split_of(seed, SPLITS) for seed in (1, 100, 101, 125, 126, 130)] == [
        'heldout', 'heldout', 'practice', 'practice', 'selection', 'selection']
    for bad in ({**SPLITS, 'selection': [120, 130]}, {**SPLITS, 'selection': [125, 130]},
                {k: v for k, v in SPLITS.items() if k != 'heldout'},
                {**SPLITS, 'practice': [125, 101]}, {**SPLITS, 'heldout': [True, 100]},
                {**SPLITS, 'extra': [200, 201]}):
        with pytest.raises(ValueError):
            check_splits(bad)
    with pytest.raises(ValueError, match='no seed split'):
        split_of(131, SPLITS)


def test_frozen_task_poses_convert_to_raw_poses(task):
    goal = task.fragment('frag_b').goal_world_T_fragment
    assert not goal.flags.writeable
    np.testing.assert_allclose(pose_to_wxyz(goal), [0.04, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0], atol=1e-12)


# ------------------------------------------------------------------ initial conditions


def test_initial_conditions_are_deterministic_planar_and_valid(task):
    first, again, other = (initial_condition(task, seed, **PLACEMENT) for seed in (101, 101, 102))
    assert first.to_record() == again.to_record() and first.to_record() != other.to_record()
    assert (first.seed, first.split) == (101, 'practice')
    np.testing.assert_array_equal(first.fragment('frag_a').initial_world_T_fragment,
                                  task.fragment('frag_a').initial_world_T_fragment)
    for seed in range(101, 131):
        assert_valid_placement(task, initial_condition(task, seed, **PLACEMENT), **PLACEMENT)


def test_crowded_placements_keep_every_gap(tmp_path):
    base = task_from_layout(write_crowded_layout(tmp_path / 'crowded'), anchor_id='frag_a')
    placement = dict(max_shift_m=0.05, max_yaw_deg=30.0, gap_m=0.02)
    moved = 0
    for seed in range(101, 131):
        seeded = initial_condition(base, seed, **placement)
        assert_valid_placement(base, seeded, **placement)
        moved += seeded.initial_sha256() != base.initial_sha256()
    assert moved == 30


def test_initial_condition_raises_instead_of_placing_fragments_within_the_gap(tmp_path):
    path = write_layout(tmp_path / 'layout')
    data = json.loads(path.read_text())
    data['pieces'][2]['staging_pose']['position'] = [-0.42, 0.15, 0.75]  # 1 cm from frag_b, far from everything else
    path.write_text(json.dumps(data))
    base = task_from_layout(path, anchor_id='frag_a')
    with pytest.raises(ValueError, match='no valid placement for frag_c'):
        initial_condition(base, 101, max_shift_m=0.0, max_yaw_deg=0.0, gap_m=0.02, attempts=5)
    assert_valid_placement(base, initial_condition(base, 101, max_shift_m=0.0, max_yaw_deg=0.0, gap_m=0.005),
                           max_shift_m=0.0, max_yaw_deg=0.0, gap_m=0.005)


def test_initial_condition_refuses_seeded_tasks_and_unsplit_seeds(task):
    seeded = initial_condition(task, 5, **PLACEMENT)
    assert seeded.split == 'heldout'
    with pytest.raises(ValueError, match='base task'):
        initial_condition(seeded, 6, **PLACEMENT)
    with pytest.raises(ValueError, match='no seed split'):
        initial_condition(task, 500, **PLACEMENT)


# ------------------------------------------------------------------ the canonical protocol


def test_configured_values_come_from_jing_json(task):
    place = CONFIG['s2']['completion_defaults']['place']
    assert success_criterion(CONFIG) == {key: place[key] for key in
                                         ('position_tolerance_m', 'rotation_tolerance_deg', 'settle_s')}
    seeded = configured_initial_condition(task, 126, CONFIG)
    assert seeded.split == 'selection'
    assert seeded.to_record() == initial_condition(task, 126, **CONFIG['baselines']['initial_condition']).to_record()
    assert canonical_protocol()['seed_splits'] == check_splits(CONFIG['baselines']['seed_splits'])
    assert canonical_protocol()['anchor'] == 'fixture'
    assert canonical_protocol()['collider_inset_m'] == 0.00025  # every seam open by 0.5 mm (changed 2026-10-21)


@pytest.mark.parametrize('path, value', [
    (('baselines', 'seed_splits', 'heldout'), [1, 99]),
    (('baselines', 'initial_condition', 'max_shift_m'), 0.0),
    (('baselines', 'heldout_claims_dir'), '/tmp/elsewhere'),
    (('s2', 'completion_defaults', 'place', 'position_tolerance_m'), 0.05),
    (('s2', 'completion_defaults', 'place', 'settle_s'), 0.1),
    (('baselines', 'success_tolerances_from'), 's2.completion_defaults.insert'),
    (('baselines', 'anchor'), 'free'),
    (('baselines', 'collider_inset_m'), 0.0),
])
def test_another_protocol_is_refused(task, path, value):
    config = copy.deepcopy(CONFIG)
    section = config
    for key in path[:-1]:
        section = section[key]
    section[path[-1]] = value
    with pytest.raises(ValueError, match='must be the ones in jing.json'):
        require_canonical(config)
    with pytest.raises(ValueError, match='must be the ones in jing.json'):
        configured_initial_condition(task, 101, config)


def test_verify_task_accepts_only_the_layout_under_the_canonical_protocol(task, tmp_path):
    assert verify_task(task) is task
    seeded = configured_initial_condition(task, 7, CONFIG)
    assert verify_task(AssemblyTask.from_record(json.loads(json.dumps(seeded.to_record())))).seed == 7
    loose = initial_condition(task, 7, max_shift_m=0.0, max_yaw_deg=0.0, gap_m=0.02)
    for forged, field in ((replace(task, position_tolerance_m=0.05), 'position_tolerance_m'),
                          (replace(task, settle_s=0.1), 'settle_s'),
                          (replace(task, task_id='renamed'), 'task_id'),
                          (loose, 'fragments'),
                          (replace(configured_initial_condition(task, 101, CONFIG), seed=102), 'fragments')):
        with pytest.raises(ValueError, match=field):
            verify_task(forged)
    relabelled = replace(seeded, split='practice')  # a valid record on its own, but seed 7 is held out
    with pytest.raises(ValueError, match='split'):
        verify_task(relabelled)
    mesh = tmp_path / 'layout' / 'frag_b.npz'
    np.savez(mesh, v=BOX_V * 1.01, f=BOX_F)
    with pytest.raises(ValueError, match='fragments'):
        verify_task(task)


def test_layouts_with_the_same_object_id_get_different_task_ids(tmp_path):
    first = task_from_layout(write_layout(tmp_path / 'one'), anchor_id='frag_a')
    path = write_layout(tmp_path / 'two')
    data = json.loads(path.read_text())
    data['pieces'][1]['staging_pose']['position'] = [-0.42, 0.12, 0.75]
    path.write_text(json.dumps(data))
    second = task_from_layout(path, anchor_id='frag_a')
    assert first.task_id.startswith('cubes-') and second.task_id.startswith('cubes-')
    assert first.task_id != second.task_id and first.layout_sha256 != second.layout_sha256


# ------------------------------------------------------------------ command line


def test_cli_writes_base_and_seeded_task_records(tmp_path, capsys):
    from tasks.dual_arm_crag_skull.baselines.__main__ import main
    layout = write_layout(tmp_path / 'layout')
    base, seeded, heldout = tmp_path / 'base.json', tmp_path / 'seeded.json', tmp_path / 'heldout.json'
    assert main(['task', '--layout', str(layout), '--anchor', 'frag_a', '--output', str(base)]) == 0
    assert main(['task', '--layout', str(layout), '--anchor', 'frag_a', '--seed', '107', '--output', str(seeded)]) == 0
    with pytest.raises(SystemExit):
        main(['task', '--layout', str(layout), '--anchor', 'frag_a', '--seed', '7', '--output', str(heldout)])
    assert 'needs --allow-heldout' in capsys.readouterr().err and not heldout.exists()
    assert main(['task', '--layout', str(layout), '--anchor', 'frag_a', '--seed', '7', '--allow-heldout', '--output',
                 str(heldout)]) == 0
    base_task = AssemblyTask.from_record(json.loads(base.read_text()))
    seeded_task = AssemblyTask.from_record(json.loads(seeded.read_text()))
    assert (base_task.seed, seeded_task.seed, seeded_task.split) == (None, 107, 'practice')
    assert seeded_task.to_record() == configured_initial_condition(base_task, 107, CONFIG).to_record()
    assert verify_task(AssemblyTask.from_record(json.loads(heldout.read_text()))).split == 'heldout'
    with pytest.raises(SystemExit):
        main(['task', '--layout', str(layout), '--anchor', 'frag_a', '--output', str(base)])


# ------------------------------------------------------------------ the assembly plan


def test_the_layouts_plan_is_given_in_order_with_world_insertion_directions(task):
    # frag_b sits on the anchor's +x side and leaves along +x, so it goes in along -x; frag_c the mirror image
    assert [(step.fragment_id, step.insertion_direction) for step in task.assembly_plan] == [
        ('frag_b', (-1.0, 0.0, 0.0)), ('frag_c', (1.0, 0.0, 0.0))]
    record = json.loads(json.dumps(task.to_record()))
    assert record['assembly_plan'] == [{'fragment_id': 'frag_b', 'insertion_direction': [-1.0, 0.0, 0.0]},
                                       {'fragment_id': 'frag_c', 'insertion_direction': [1.0, 0.0, 0.0]}]
    assert AssemblyTask.from_record(record).assembly_plan == task.assembly_plan
    assert configured_initial_condition(task, 101, CONFIG).assembly_plan == task.assembly_plan
    assert canonical_protocol()['assembly_plan'] == 'given'


def test_a_plan_made_for_another_anchor_is_refused(tmp_path):
    with pytest.raises(ValueError, match='plan holds frag_a fixed'):
        task_from_layout(write_layout(tmp_path / 'layout'), anchor_id='frag_b')


def plan_record(*steps):
    return tuple({'fragment_id': name, 'insertion_direction': direction} for name, direction in steps)


@pytest.mark.parametrize('steps, message', [
    ((('frag_b', (-1, 0, 0)),), 'every fragment but the anchor once'),
    ((('frag_b', (-1, 0, 0)), ('frag_b', (-1, 0, 0))), 'every fragment but the anchor once'),
    ((('frag_b', (-1, 0, 0)), ('frag_c', (1, 0, 0)), ('frag_a', (0, 0, -1))), 'every fragment but the anchor once'),
    ((('frag_b', (-1, 0, 0)), ('frag_z', (1, 0, 0))), 'every fragment but the anchor once'),
    ((('frag_b', (-2, 0, 0)), ('frag_c', (1, 0, 0))), 'unit'),
    ((('frag_b', (float('nan'), 0, 0)), ('frag_c', (1, 0, 0))), 'unit'),
    ((('frag_b', (-1, 0)), ('frag_c', (1, 0, 0))), 'unit'),
])
def test_invalid_plans_are_refused(task, steps, message):
    with pytest.raises(ValueError, match=message):
        replace(task, assembly_plan=plan_record(*steps))


def test_a_task_without_its_layouts_plan_is_not_the_canonical_task(task, tmp_path):
    plain = task_from_layout(write_layout(tmp_path / 'plain', plan=False), anchor_id='frag_a')
    assert plain.assembly_plan is None and plain.to_record()['assembly_plan'] is None
    with pytest.raises(ValueError, match='assembly plan'):
        verify_task(plain)
    for forged in (replace(task, assembly_plan=plan_record(('frag_c', (1, 0, 0)), ('frag_b', (-1, 0, 0)))),
                   replace(task, assembly_plan=plan_record(('frag_b', (0, 0, -1)), ('frag_c', (1, 0, 0)))),
                   replace(task, assembly_plan=None)):
        with pytest.raises(ValueError, match='assembly'):
            verify_task(forged)


@pytest.mark.parametrize('edit, message', [
    (lambda plan, pieces: plan[0].update(exit_direction=[-1, 0, 0]), "'frag_b' cannot go in straight"),  # into the anchor
    (lambda plan, pieces: plan[0].update(exit_direction=[1, 1, 0]), 'grid axes'),
    (lambda plan, pieces: plan[0].update(exit_direction=[0, 0, -1]), 'grid axes'),  # down, through the table
    (lambda plan, pieces: [piece.pop('cells') for piece in pieces], 'cells'),
    (lambda plan, pieces: plan[0].pop('exit_direction'), 'exit_direction'),
])
def test_the_layouts_plan_is_checked_on_its_own_cells(tmp_path, edit, message):
    path = write_layout(tmp_path / 'layout')
    data = json.loads(path.read_text())
    edit(data['source']['ownership_repair']['plan'], data['pieces'])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        task_from_layout(path, anchor_id='frag_a')


def test_a_record_from_before_the_plan_loads_without_one(task):
    record = json.loads(json.dumps(task.to_record()))
    record.pop('assembly_plan')
    assert AssemblyTask.from_record(record).assembly_plan is None


def test_a_malformed_plan_step_is_a_value_error(task):
    with pytest.raises(ValueError, match='fragment_id and insertion_direction'):
        replace(task, assembly_plan=({'fragment_id': 'frag_b'}, {'fragment_id': 'frag_c', 'insertion_direction': [1, 0, 0]}))


@pytest.mark.parametrize('value', [-0.0005, float('nan'), '0.5 mm', None, True])
def test_the_collider_inset_is_a_length_of_zero_or_more(value):
    config = copy.deepcopy(CONFIG)
    config['baselines']['collider_inset_m'] = value
    with pytest.raises(ValueError, match='collider_inset_m must be'):
        require_canonical(config)


@pytest.mark.parametrize('value', ['withheld', 'Given', None])
def test_the_plan_protocol_is_given_and_nothing_else(value):
    config = copy.deepcopy(CONFIG)
    config['baselines']['assembly_plan'] = value  # a typo must not switch the requirement off
    with pytest.raises(ValueError, match="assembly_plan must be 'given'"):
        require_canonical(config)
