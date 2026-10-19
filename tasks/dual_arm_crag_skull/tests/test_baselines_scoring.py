import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tasks.dual_arm_crag_skull.baselines import scoring
from tasks.dual_arm_crag_skull.baselines.scoring import (RunRecord, append_records, claim_heldout, recorded_seeds,
                                                         refuse_repeats, score_assembly, summarize, unrecorded_claims,
                                                         wilson_interval)
from tasks.dual_arm_crag_skull.baselines.task import initial_condition, task_from_layout
from tasks.dual_arm_crag_skull.tests.baselines_fixtures import write_layout

PLACEMENT = dict(max_shift_m=0.03, max_yaw_deg=30.0, gap_m=0.02)


@pytest.fixture
def task(tmp_path):
    return task_from_layout(write_layout(tmp_path / 'layout'), anchor_id='frag_a')


def moved(pose, *, shift=(0, 0, 0), rotvec=(0, 0, 0)):
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    delta[:3, 3] = shift
    return delta @ pose


def record(method='aspire', variant='v1', task_id=None, seed=None, split=None, status='completed', score=None, **kw):
    kw.setdefault('backend', 'unit_test')
    kw.setdefault('test_only', False)
    return RunRecord(method, variant, task_id, seed, split, status, score, **kw)


def test_goal_poses_score_as_a_full_assembly(task):
    score = score_assembly(task, task.goal_poses(), settled=True)
    assert score.success and score.placed == 2 and score.placed_fraction == 1.0 and score.anchor_within_tolerance
    assert max(f.position_error_m for f in score.fragments) < 1e-12
    assert (score.layout_sha256, score.goal_source, score.position_tolerance_m, score.rotation_tolerance_deg) == (
        task.layout_sha256, 'oracle_layout', task.position_tolerance_m, task.rotation_tolerance_deg)
    assert score.initial_sha256 == task.initial_sha256() and score.invalid_ids == ()


def test_an_assembly_that_was_not_released_and_settled_is_not_a_success(task):
    score = score_assembly(task, task.goal_poses(), settled=False)
    assert score.placed == 2 and score.anchor_within_tolerance and not score.success and not score.settled
    with pytest.raises(ValueError, match='settled'):
        score_assembly(task, task.goal_poses(), settled=1)


def test_errors_are_judged_relative_to_the_anchor_and_the_anchor_is_bounded(task):
    nudged = {key: moved(pose, shift=(0.002, 0, 0)) for key, pose in task.goal_poses().items()}
    assert score_assembly(task, nudged, settled=True).success
    shoved = {key: moved(pose, shift=(0.01, 0, 0)) for key, pose in task.goal_poses().items()}
    score = score_assembly(task, shoved, settled=True)
    assert score.placed == 2 and not score.anchor_within_tolerance and not score.success
    assert score.fragments[0].world_position_error_m == pytest.approx(0.01)
    assert score.fragments[0].position_error_m == pytest.approx(0.0, abs=1e-12)


def test_the_anchor_rotation_is_bounded_too(task):
    """The anchor turned 5 degrees in place, the others turned with it: placed relative to it, but not a success."""
    anchor = task.fragment('frag_a').goal_world_T_fragment
    turn = anchor @ moved(np.eye(4), rotvec=(0, 0, np.deg2rad(5.0))) @ np.linalg.inv(anchor)
    score = score_assembly(task, {key: turn @ pose for key, pose in task.goal_poses().items()}, settled=True)
    assert score.anchor_position_error_m < 1e-12 and score.anchor_rotation_error_deg == pytest.approx(5.0)
    assert score.placed == 2 and not score.anchor_within_tolerance and not score.success


def test_relative_errors_use_the_anchor_frame_not_the_world_frame(task):
    """The whole assembly turned 2 degrees about the world origin (0.75 m below and beside it): relative errors 0."""
    turn = moved(np.eye(4), rotvec=(0.0, np.deg2rad(2.0), np.deg2rad(2.0)))
    score = score_assembly(task, {key: turn @ pose for key, pose in task.goal_poses().items()}, settled=True)
    assert all(f.position_error_m < 1e-9 and f.rotation_error_deg < 1e-9 for f in score.fragments)
    assert score.placed == 2 and min(f.world_position_error_m for f in score.fragments) > 0.01


@pytest.mark.parametrize('shift, rotvec, within', [
    ((0.0029, 0, 0), (0, 0, 0), True), ((0.0031, 0, 0), (0, 0, 0), False),
    ((0, 0, 0), (0, 0, np.deg2rad(2.9)), True), ((0, 0, 0), (0, 0, np.deg2rad(3.1)), False)])
def test_tolerance_boundaries(task, shift, rotvec, within):
    poses = task.goal_poses()
    poses['frag_b'] = moved(poses['frag_b'], shift=shift, rotvec=rotvec)
    score = score_assembly(task, poses, settled=True)
    assert score.fragments[0].within_tolerance is within and score.success is within
    assert score.fragments[1].within_tolerance


def test_missing_and_invalid_poses_fail_and_serialize(task):
    poses = task.goal_poses()
    del poses['frag_c']
    score = score_assembly(task, poses, settled=True)
    assert not score.success and score.placed == 1
    serialized = json.loads(json.dumps(score.to_record()))
    assert serialized['fragments'][1]['position_error_m'] is None and serialized['placed_fraction'] == 0.5
    no_anchor = task.goal_poses()
    del no_anchor['frag_a']
    score = score_assembly(task, no_anchor, settled=True)
    assert not score.success and score.placed == 2
    assert json.loads(json.dumps(score.to_record()))['anchor_position_error_m'] is None
    exploded = {**task.goal_poses(), 'frag_b': np.full((4, 4), np.nan), 'frag_a': np.diag([2.0, 1, 1, 1])}
    score = score_assembly(task, exploded, settled=True)
    assert score.invalid_ids == ('frag_a', 'frag_b') and not score.anchor_within_tolerance and not score.success
    assert not score.fragments[0].within_tolerance and json.loads(json.dumps(score.to_record()))['invalid_ids'] == [
        'frag_a', 'frag_b']


def test_unknown_fragments_and_candidate_goals_are_refused(task):
    with pytest.raises(ValueError, match='unknown fragments'):
        score_assembly(task, {**task.goal_poses(), 'frag_z': np.eye(4)}, settled=True)
    with pytest.raises(ValueError, match='true assembly'):
        score_assembly(replace(task, goal_source='garf_candidate'), task.goal_poses(), settled=True)


def test_run_records_are_checked(task):
    score = score_assembly(task, task.goal_poses(), settled=True)
    done = record(task_id=task.task_id, score=score, cost_usd=1.5, llm_calls=3)
    serialized = json.loads(json.dumps(done.to_record()))
    assert done.success and serialized['score']['success'] and serialized['backend'] == 'unit_test'
    assert not record(method='dexagent', task_id=task.task_id, status='blocked').success
    for kwargs, message in [({'method': 'gpt'}, 'method'), ({'status': 'ok'}, 'status'),
                            ({'seed': 4, 'split': 'heldout'}, 'different'), ({'seed': 4}, 'together'),
                            ({'seed': None, 'split': 'heldout'}, 'together'),
                            ({'seed': 5, 'split': 'practice', 'score': None}, 'canonical band'),
                            ({'seed': 5, 'split': 'train', 'score': None}, 'canonical band'),
                            ({'cost_usd': float('nan')}, 'cost_usd'), ({'wall_s': -1.0}, 'wall_s'),
                            ({'llm_calls': 1.5}, 'llm_calls'), ({'sim_trials': -2}, 'sim_trials'),
                            ({'backend': ''}, 'backend'), ({'test_only': 'no'}, 'test_only')]:
        args = dict(task_id=task.task_id, score=score)
        args.update(kwargs)
        with pytest.raises(ValueError, match=message):
            record(**args)
    with pytest.raises(TypeError):
        RunRecord('aspire', 'v1', task.task_id, None, None, 'completed', score)  # where it ran must be stated


def test_records_are_strict_json(task, tmp_path):
    append_records(tmp_path / 'runs.jsonl', [record(task_id=task.task_id, status='blocked')])
    line = (tmp_path / 'runs.jsonl').read_text()
    assert 'NaN' not in line and json.loads(line)['test_only'] is False


def test_repeats_are_refused_per_method_variant_task_and_seed(task, tmp_path):
    path = tmp_path / 'runs.jsonl'
    assert recorded_seeds(path, 'aspire', 'v1', task.task_id) == set()
    refuse_repeats(path, 'aspire', 'v1', task.task_id, [1, 2])  # no file yet
    append_records(path, [record(task_id=task.task_id, seed=3, split='heldout', status='blocked'),
                          record(task_id='other_task', seed=4, split='heldout', status='blocked')])
    assert recorded_seeds(path, 'aspire', 'v1', task.task_id) == {3}
    with pytest.raises(ValueError, match=r'already has records .* \[3\]'):
        refuse_repeats(path, 'aspire', 'v1', task.task_id, [2, 3])
    for method, variant, task_id in (('dexagent', 'v1', task.task_id), ('aspire', 'v2', task.task_id),
                                     ('aspire', 'v1', 'third_task')):
        refuse_repeats(path, method, variant, task_id, [3, 4])


def test_heldout_claims_are_exclusive_whatever_the_records_file(task, tmp_path):
    first = claim_heldout('aspire', 'v1', task.task_id, 7, records_path=tmp_path / 'a.jsonl')
    assert first.parent == scoring.claims_dir() and json.loads(first.read_text())['seed'] == 7
    for records_path in (tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'):
        with pytest.raises(ValueError, match='already claimed'):
            claim_heldout('aspire', 'v1', task.task_id, 7, records_path=records_path)
    claim_heldout('aspire', 'v2', task.task_id, 7, records_path=tmp_path / 'a.jsonl')  # another variant: visible
    claim_heldout('dexagent', 'v1', task.task_id, 7, records_path=tmp_path / 'a.jsonl')
    with pytest.raises(ValueError, match='not a held-out seed'):
        claim_heldout('aspire', 'v1', task.task_id, 101, records_path=tmp_path / 'a.jsonl')
    append_records(tmp_path / 'c.jsonl', [record(task_id=task.task_id, seed=8, split='heldout', status='error')])
    with pytest.raises(ValueError, match='already has records'):
        claim_heldout('aspire', 'v1', task.task_id, 8, records_path=tmp_path / 'c.jsonl')
    recorded = [record(task_id=task.task_id, seed=7, split='heldout', status='error').to_record()]
    assert sorted((c['method'], c['variant']) for c in unrecorded_claims(recorded)) == [('aspire', 'v2'),
                                                                                         ('dexagent', 'v1')]


def test_wilson_interval_matches_known_values():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    np.testing.assert_allclose(wilson_interval(0, 10), (0.0, 0.2775), atol=1e-4)
    np.testing.assert_allclose(wilson_interval(5, 10), (0.2366, 0.7634), atol=1e-4)
    np.testing.assert_allclose(wilson_interval(10, 10), (0.7225, 1.0), atol=1e-4)


def seeded_records(task, seeds, *, success=True, method='aspire', variant='v1', backend='unit_test', test_only=False):
    out = []
    for seed in seeds:
        seeded = initial_condition(task, seed, **PLACEMENT)
        poses = seeded.goal_poses() if success else seeded.initial_poses()
        out.append(record(method=method, variant=variant, task_id=seeded.task_id, seed=seed, split=seeded.split,
                          score=score_assembly(seeded, poses, settled=True), backend=backend,
                          test_only=test_only).to_record())
    return out


def test_summary_groups_methods_variants_tasks_backends_and_splits(task, tmp_path):
    good = score_assembly(task, task.goal_poses(), settled=True)
    bad = score_assembly(task, task.initial_poses(), settled=True)
    records = [record(task_id=task.task_id, score=good, cost_usd=1.0),
               record(task_id=task.task_id, score=bad, cost_usd=2.0),
               record(method='dexagent', variant='native', task_id=task.task_id, status='blocked')]
    path = tmp_path / 'runs.jsonl'
    append_records(path, records[:2])
    append_records(path, records[2:])
    loaded = [json.loads(line) for line in path.read_text().splitlines()]
    loaded += seeded_records(task, [1, 2]) + seeded_records(task, [101], success=False)
    loaded += seeded_records(task, [3], backend='other_backend')
    summary = {(row['method'], row['variant'], row['backend'], row['split']): row for row in summarize(loaded)}
    aspire, dexagent = summary[('aspire', 'v1', 'unit_test', None)], summary[('dexagent', 'native', 'unit_test', None)]
    assert (aspire['runs'], aspire['successes'], aspire['success_rate'], aspire['cost_usd_total']) == (2, 1, 0.5, 3.0)
    assert aspire['mean_placed_fraction'] == 0.5 and aspire['statuses'] == {'completed': 2}
    assert (dexagent['successes'], dexagent['mean_placed_fraction'], dexagent['cost_usd_total']) == (0, 0.0, None)
    np.testing.assert_allclose(aspire['wilson95'], wilson_interval(1, 2))
    heldout, practice = summary[('aspire', 'v1', 'unit_test', 'heldout')], summary[('aspire', 'v1', 'unit_test',
                                                                                    'practice')]
    assert (heldout['runs'], heldout['successes'], practice['runs'], practice['successes']) == (2, 2, 1, 0)
    assert summary[('aspire', 'v1', 'other_backend', 'heldout')]['runs'] == 1
    other_task = [dict(row, task_id='another') for row in seeded_records(task, [4])]
    other_task[0]['score']['task_id'] = 'another'
    rows = summarize(seeded_records(task, [5]) + other_task)
    assert sorted(row['task_id'] for row in rows) == sorted([task.task_id, 'another'])


@pytest.mark.parametrize('tamper, message', [
    (lambda rows: rows + rows[:1], 'recorded twice'),
    (lambda rows: [dict(rows[0], success=False)], 'contradicts'),
    (lambda rows: [dict(rows[0], split='practice')], 'canonical band'),
    (lambda rows: [rows[0], {**rows[1], 'score': {**rows[1]['score'], 'position_tolerance_m': 0.05}}],
     'different layouts'),
])
def test_summary_refuses_records_it_cannot_count(task, tamper, message):
    rows = seeded_records(task, [1, 2])
    with pytest.raises(ValueError, match=message):
        summarize(tamper(rows))


def test_test_only_runs_are_refused_unless_included_and_kept_apart(task):
    rows = seeded_records(task, [1], test_only=True, backend='kinematic')
    rows += seeded_records(task, [2], backend='kinematic')
    with pytest.raises(ValueError, match='test-only'):
        summarize(rows)
    groups = summarize(rows, include_test_only=True)
    assert sorted((row['test_only'], row['runs']) for row in groups) == [(False, 1), (True, 1)]


def test_summarize_cli_refuses_a_repeated_file_and_lists_unrecorded_claims(task, tmp_path, capsys):
    from tasks.dual_arm_crag_skull.baselines.__main__ import main
    path = tmp_path / 'runs.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in seeded_records(task, [1])))
    claim_heldout('aspire', 'v1', task.task_id, 9, records_path=path)
    with pytest.raises(SystemExit):
        main(['summarize', str(path), str(tmp_path / '.' / 'runs.jsonl')])
    capsys.readouterr()
    assert main(['summarize', str(path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['groups'][0]['runs'] == 1 and [c['seed'] for c in printed['unrecorded_heldout_claims']] == [9]
