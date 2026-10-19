import json

import numpy as np
import pytest

from tasks.dual_arm_crag_skull.contracts.skills import SkillRejected
from tasks.dual_arm_crag_skull.s2.observation import FragmentState, Observation
from tasks.dual_arm_crag_skull.s2.planner import PlanningContext, ScriptedPlanner
from tasks.dual_arm_crag_skull.tests.s2_fakes import (
    COMPLETIONS, TIMEOUTS, FakeCandidateSource, expected_goal, observation, pose)


def context(obs=None, trained=('pick', 'place'), source=None):
    return PlanningContext(observation=obs or observation(), trained_skills=trained,
                           candidate_source=FakeCandidateSource() if source is None else source,
                           completion_defaults=COMPLETIONS, timeouts_s=TIMEOUTS, new_request_id=lambda: 'req-7')


def submission(**overrides):
    item = {'observation_id': 'obs-1', 'skill_id': 'place', 'active_arms': ['left'], 'fragment_id': 'part_1',
            'support_fragment_id': None, 'target_candidate_id': 'obs-1/call-0/seed-41', 'rationale': 'part_1 is held'}
    item.update(overrides)
    return item


def propose(ctx, seeds=(41, 42)):
    return ctx.run_candidates({'observation_id': 'obs-1', 'fragment_ids': ['part_0', 'part_1'],
                               'anchor_id': 'part_0', 'seeds': list(seeds)})


def test_candidates_are_booked_with_world_goals_from_observed_anchor():
    ctx = context()
    text, is_error = propose(ctx)
    assert not is_error
    summary = json.loads(text)
    assert [c['geometry_status'] for c in summary['candidates']] == ['needs_motion_validation', 'rejected']
    record = ctx.book.record('obs-1/call-0/seed-41')
    np.testing.assert_allclose(record.world_T_goal['part_1'], expected_goal('part_1'), atol=1e-12)
    np.testing.assert_allclose(record.world_T_goal['part_0'], observation().fragments['part_0'].world_T_part,
                               atol=1e-12)
    offsets = summary['candidates'][0]['goal_offsets_from_current_pose']
    assert offsets['part_0'] == {'translation_m': 0.0, 'rotation_deg': 0.0}


def test_resolve_builds_request_with_resolved_pose_and_provenance():
    ctx = context()
    propose(ctx)
    request = ctx.resolve(submission())
    assert request.request_id == 'req-7'
    np.testing.assert_allclose(request.target_pose, expected_goal('part_1'), atol=1e-12)
    assert request.target_provenance['candidate_id'] == 'obs-1/call-0/seed-41'
    assert request.completion_predicate == COMPLETIONS['place']
    assert request.timeout_s == TIMEOUTS['place']


@pytest.mark.parametrize('overrides, message', [
    ({'target_candidate_id': 'obs-1/call-0/seed-42'}, 'geometry screen'),
    ({'target_candidate_id': 'obs-0/call-0/seed-41'}, 'unknown candidate'),
    ({'observation_id': 'obs-0'}, 'stale observation'),
    ({'skill_id': 'insert'}, 'not available'),
    ({'skill_id': 'pick'}, 'takes no target candidate'),
    ({'fragment_id': 'part_9'}, 'no goal for'),
    ({'target_candidate_id': None}, 'needs a target pose'),
])
def test_resolve_rejects_ungrounded_or_unavailable_submissions(overrides, message):
    ctx = context()
    propose(ctx)
    with pytest.raises(SkillRejected, match=message):
        ctx.resolve(submission(**overrides))


def test_resolve_rejects_extra_or_missing_fields():
    ctx = context()
    with pytest.raises(SkillRejected, match='exactly'):
        ctx.resolve({**submission(), 'target_pose': [[1, 0, 0, 0]]})


def test_new_observation_means_new_book():
    ctx = context()
    propose(ctx)
    fresh = context(obs=observation('obs-2'))
    with pytest.raises(SkillRejected, match='unknown candidate'):
        fresh.resolve(submission(observation_id='obs-2'))


def test_unlocalized_anchor_and_failed_garf_are_tool_errors():
    obs = Observation('obs-1', 1.0, {'part_0': FragmentState('part_0', None),
                                     'part_1': FragmentState('part_1', pose([0, 0, 1]), 'icp')})
    text, is_error = propose(context(obs=obs))
    assert is_error and 'not localized' in json.loads(text)['error']

    class Failing:
        def propose(self, request, *, current_observation_id):
            return {'status': 'failed', 'reason': 'timeout after 120 s; runner killed'}
    text, is_error = propose(context(source=Failing()))
    assert is_error and 'timeout' in json.loads(text)['error']


def test_allowed_skills_are_trained_skills_plus_non_motion():
    assert context(trained=('pick',)).allowed_skills == ('pick', 'reobserve')
    tools = context().tools()
    names = [tool['name'] for tool in tools]
    assert names == ['propose_assembly_candidates', 'submit_skill_request']
    for tool in tools:
        assert tool['strict'] is True and tool['input_schema']['additionalProperties'] is False


def test_strict_schemas_avoid_unsupported_keywords():
    forbidden = {'minimum', 'maximum', 'minItems', 'maxItems', 'uniqueItems', 'minLength', 'maxLength'}

    def walk(node):
        if isinstance(node, dict):
            assert not forbidden & set(node), node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    for tool in context().tools():
        walk(tool['input_schema'])


def test_scripted_planner_uses_same_validation():
    ctx = context()
    planner = ScriptedPlanner([lambda c: submission(skill_id='pick', target_candidate_id=None), lambda c: None])
    outcome = planner.plan(ctx)
    assert outcome.status == 'request' and outcome.skill_request.skill_id == 'pick'
    assert planner.plan(ctx).status == 'no_request'
    assert planner.plan(ctx).reason == 'script exhausted'
    bad = ScriptedPlanner([lambda c: submission()]).plan(context())
    assert bad.status == 'error' and 'unknown candidate' in bad.reason


def test_same_seed_with_another_anchor_is_a_new_candidate_not_an_overwrite():
    ctx = context()
    propose(ctx, seeds=(41,))
    first = ctx.book.record('obs-1/call-0/seed-41').world_T_goal['part_1'].copy()
    ctx.run_candidates({'observation_id': 'obs-1', 'fragment_ids': ['part_0', 'part_1'], 'anchor_id': 'part_1',
                        'seeds': [41]})
    np.testing.assert_array_equal(ctx.book.record('obs-1/call-0/seed-41').world_T_goal['part_1'], first)
    assert 'obs-1/call-1/seed-41' in ctx.book


class Mangled(FakeCandidateSource):
    def __init__(self, mutate):
        super().__init__()
        self.mutate = mutate

    def propose(self, request, *, current_observation_id):
        result = super().propose(request, current_observation_id=current_observation_id)
        self.mutate(result['candidates'])
        return result


@pytest.mark.parametrize('mutate, message', [
    (lambda c: c[1].update(seed=None), 'seed must be an integer'),
    (lambda c: c[1].update(seed=41), 'duplicate seeds'),
    (lambda c: c[1]['assembly_T_fragment'].pop('part_1'), 'fragment set differs'),
])
def test_bad_candidate_books_nothing_and_is_a_tool_error(mutate, message):
    ctx = context(source=Mangled(mutate))
    text, is_error = propose(ctx)
    assert is_error and message in json.loads(text)['error']
    assert 'obs-1/call-0/seed-41' not in ctx.book
    with pytest.raises(SkillRejected, match='unknown candidate'):
        ctx.resolve(submission())


def test_pick_carries_the_observed_fragment_pose():
    ctx = context()
    request = ctx.resolve(submission(skill_id='pick', target_candidate_id=None))
    np.testing.assert_array_equal(request.fragment_pose, observation().fragments['part_1'].world_T_part)
    assert request.fragment_pose_provenance == {'source': 'observed_geometry', 'observation_id': 'obs-1',
                                                'pose_source': 'icp'}
    unlocalized = context(obs=observation(part1_localized=False))
    with pytest.raises(SkillRejected, match='needs the observed pose'):
        unlocalized.resolve(submission(skill_id='pick', target_candidate_id=None))


def test_active_arms_are_put_in_canonical_order():
    ctx = context(trained=('pick', 'place', 'align'))
    propose(ctx)
    request = ctx.resolve(submission(skill_id='align', active_arms=['right', 'left']))
    assert request.active_arms == ('left', 'right')
    with pytest.raises(SkillRejected, match='distinct arms'):
        ctx.resolve(submission(skill_id='align', active_arms=['left', 'left']))
