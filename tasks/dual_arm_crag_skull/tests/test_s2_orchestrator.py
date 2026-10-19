import itertools

import numpy as np
import pytest

from tasks.dual_arm_crag_skull.contracts.skills import SkillRequest
from tasks.dual_arm_crag_skull.s2.orchestrator import (
    AssemblyOrchestrator, ExecutionReport, Verification, merge_clusters)
from tasks.dual_arm_crag_skull.s2.planner import PlanOutcome, ScriptedPlanner
from tasks.dual_arm_crag_skull.tests.s2_fakes import COMPLETIONS, TIMEOUTS, FakeCandidateSource, observation


class Observer:
    def __init__(self, fail_at=None):
        self.count, self.fail_at = 0, fail_at

    def observe(self, clusters):
        self.count += 1
        if self.count == self.fail_at:
            raise RuntimeError('camera driver crashed')
        return observation(f'obs-{self.count}', clusters=clusters)


class Executor:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.requests = []

    def execute(self, request, observation):
        self.requests.append((request, observation.observation_id))
        return ExecutionReport(self.statuses.pop(0), reason='scripted')


class Verifier:
    def __init__(self, results):
        self.results = list(results)

    def verify(self, request, before, after):
        return self.results.pop(0)


BOTH = frozenset({'part_0', 'part_1'})


def place_after_proposal(context):
    obs = context.observation.observation_id
    context.run_candidates({'observation_id': obs, 'fragment_ids': ['part_0', 'part_1'], 'anchor_id': 'part_0',
                            'seeds': [41]})
    return {'observation_id': obs, 'skill_id': 'place', 'active_arms': ['left'], 'fragment_id': 'part_1',
            'support_fragment_id': None, 'target_candidate_id': f'{obs}/call-0/seed-41', 'rationale': 'scripted'}


def pick(context):
    return {'observation_id': context.observation.observation_id, 'skill_id': 'pick', 'active_arms': ['left'],
            'fragment_id': 'part_1', 'support_fragment_id': None, 'target_candidate_id': None, 'rationale': 'grab'}


def reobserve(context):
    return {'observation_id': context.observation.observation_id, 'skill_id': 'reobserve', 'active_arms': [],
            'fragment_id': None, 'support_fragment_id': None, 'target_candidate_id': None, 'rationale': 'occluded'}


def orchestrator(planner, executor, verifier, observer=None, **kwargs):
    kwargs.setdefault('max_planning_s', 60.0)
    return AssemblyOrchestrator(observer=observer or Observer(), planner=planner, executor=executor, verifier=verifier,
                                trained_skills=('pick', 'place'), completion_defaults=COMPLETIONS,
                                timeouts_s=TIMEOUTS, candidate_source=FakeCandidateSource(), **kwargs)


def test_successful_place_merges_cluster_and_stops_when_assembled():
    executor = Executor(['completed'])
    run = orchestrator(ScriptedPlanner([place_after_proposal]), executor,
                       Verifier([Verification(True, merged_cluster=BOTH)])).run(5)
    assert run.status == 'assembled'
    assert run.verified_clusters == (BOTH,)
    request, planned_on = executor.requests[0]
    assert request.observation_id == planned_on == 'obs-1'
    assert run.steps[0]['after_observation_id'] == 'obs-2' and run.steps[1]['observation_id'] == 'obs-3'


class Returns:
    """A planner that hands back a fixed outcome, bypassing PlanningContext.resolve."""

    def __init__(self, build):
        self.build = build

    def plan(self, context):
        return PlanOutcome('request', skill_request=self.build(context))


def unvalidated(**overrides):
    def build(context):
        obs = context.observation
        args = dict(request_id='r', observation_id=obs.observation_id, skill_id='pick',
                    active_arms=('left',), fragment_id='part_1', completion_predicate={'type': 'grasp_held'},
                    timeout_s=20.0, fragment_pose=obs.fragments['part_1'].world_T_part,
                    fragment_pose_provenance={'source': 'observed_geometry', 'observation_id': obs.observation_id,
                                              'pose_source': 'icp'})
        args.update(overrides)
        return SkillRequest(**args)
    return build


@pytest.mark.parametrize('overrides, message', [
    ({'observation_id': 'obs-0'}, 'stale observation'),
    ({'skill_id': 'insert', 'completion_predicate': COMPLETIONS['insert']}, 'no trained S1 policy'),
    ({'active_arms': ('left', 'left')}, 'distinct arms'),
    ({'fragment_id': 'part_9'}, 'unknown fragment'),
    ({'timeout_s': 1e9}, 'timeout_s'),
])
def test_unvalidated_requests_never_reach_the_executor(overrides, message):
    executor = Executor([])
    run = orchestrator(Returns(unvalidated(**overrides)), executor, Verifier([]), max_failures_without_success=1).run(3)
    assert executor.requests == [] and run.status == 'too_many_failures'
    assert message in run.steps[0]['orchestrator_rejection']


def test_target_pose_must_equal_the_booked_candidate_goal():
    def tampered(context):
        request = context.resolve(place_after_proposal(context))
        pose = np.array(request.target_pose)
        pose[0, 3] += 0.01
        return SkillRequest(**{**request.__dict__, 'target_pose': pose})
    executor = Executor([])
    run = orchestrator(Returns(tampered), executor, Verifier([]), max_failures_without_success=1).run(2)
    assert executor.requests == []
    assert 'differs from the booked candidate' in run.steps[0]['orchestrator_rejection']


def test_fragment_pose_must_equal_the_current_observation():
    def moved(context):
        request = context.resolve(pick(context))
        pose = np.array(request.fragment_pose)
        pose[2, 3] += 0.005
        return SkillRequest(**{**request.__dict__, 'fragment_pose': pose})
    executor = Executor([])
    run = orchestrator(Returns(moved), executor, Verifier([]), max_failures_without_success=1).run(2)
    assert executor.requests == [] and 'fragment pose differs' in run.steps[0]['orchestrator_rejection']


def test_slow_planning_is_rejected():
    ticks = itertools.count(0.0, 100.0)
    executor = Executor([])
    run = orchestrator(ScriptedPlanner([pick]), executor, Verifier([]), clock=lambda: next(ticks),
                       max_planning_s=30.0, max_failures_without_success=1).run(2)
    assert executor.requests == [] and 'planning took 100.0 s' in run.steps[0]['orchestrator_rejection']


def test_planner_failures_hold_and_stop_without_execution():
    class Refusing:
        def plan(self, context):
            return PlanOutcome('refused', reason='model declined: None')
    executor = Executor([])
    run = orchestrator(Refusing(), executor, Verifier([]), max_failures_without_success=2).run(10)
    assert run.status == 'too_many_failures' and len(run.steps) == 2
    assert all(step['action'] == 'hold' for step in run.steps) and executor.requests == []


def test_failures_interleaved_with_reobserve_still_trip_the_breaker():
    executor = Executor(['failed'] * 5)
    verifier = Verifier([Verification(False, reason='dropped')] * 5)
    run = orchestrator(ScriptedPlanner([pick, reobserve] * 5), executor, verifier,
                       max_failures_without_success=2, max_idle_without_success=10).run(10)
    assert run.status == 'too_many_failures' and len(executor.requests) == 2


def test_reobserve_never_executes_and_repeated_idle_stalls():
    executor = Executor([])
    run = orchestrator(ScriptedPlanner([reobserve] * 3), executor, Verifier([]), max_idle_without_success=3).run(10)
    assert run.status == 'stalled' and executor.requests == []


def test_failed_verification_does_not_merge_and_feeds_back():
    executor = Executor(['completed', 'completed'])
    verifier = Verifier([Verification(False, reason='part_1 slid 6 mm after release'),
                         Verification(True, merged_cluster=BOTH)])
    seen_feedback = []

    def recording(context):
        seen_feedback.append(context.feedback)
        return place_after_proposal(context)
    run = orchestrator(ScriptedPlanner([recording, recording]), executor, verifier).run(5)
    assert run.status == 'assembled'
    assert run.steps[0]['verification']['passed'] is False and 'merged_cluster' not in run.steps[0]
    assert seen_feedback[1][-1]['verification_reason'] == 'part_1 slid 6 mm after release'


def test_failed_execution_never_merges_even_if_verifier_passes():
    executor = Executor(['failed'])
    run = orchestrator(ScriptedPlanner([place_after_proposal]), executor,
                       Verifier([Verification(True, merged_cluster=BOTH)])).run(1)
    assert run.status == 'max_steps' and run.verified_clusters == ()
    assert 'merged_cluster' not in run.steps[0]


@pytest.mark.parametrize('step, cluster, message', [
    (pick, BOTH, 'released_stable'),
    (place_after_proposal, frozenset({'part_0', 'part_9'}), 'does not contain the placed fragment'),
    (place_after_proposal, frozenset({'part_1'}), 'at least two'),
    (place_after_proposal, frozenset({'part_1', 'part_9'}), 'unknown fragments'),
])
def test_invalid_merges_are_refused_without_crashing(step, cluster, message):
    run = orchestrator(ScriptedPlanner([step]), Executor(['completed']),
                       Verifier([Verification(True, merged_cluster=cluster)])).run(1)
    assert run.verified_clusters == () and message in run.steps[0]['merge_rejected']


def test_exception_inside_a_step_keeps_the_log():
    executor = Executor(['completed'])
    run = orchestrator(ScriptedPlanner([pick, pick]), executor, Verifier([Verification(True)]),
                       observer=Observer(fail_at=3)).run(5)
    assert run.status == 'error' and 'camera driver crashed' in run.error
    assert run.steps[0]['execution']['status'] == 'completed' and run.steps[-1]['action'] == 'error'


def test_cluster_merge_unions_overlaps_and_verification_guard():
    clusters = (frozenset({'a', 'b'}), frozenset({'c', 'd'}), frozenset({'e', 'f'}))
    merged = merge_clusters(clusters, {'b', 'c'})
    assert set(merged) == {frozenset({'a', 'b', 'c', 'd'}), frozenset({'e', 'f'})}
    with pytest.raises(ValueError):
        Verification(False, merged_cluster=frozenset({'a'}))


LEFT_ONLY = {('pick', ('left',)), ('place', ('left',))}


def test_untrained_arm_combination_is_refused_by_planner_context_and_recheck():
    def right_pick(context):
        return {**pick(context), 'active_arms': ['right']}
    executor = Executor([])
    run = orchestrator(ScriptedPlanner([right_pick]), executor, Verifier([]), max_failures_without_success=1,
                       trained_combinations=LEFT_ONLY).run(2)
    assert executor.requests == []
    assert "pick with arms ['right'] has no trained S1 policy" in run.steps[0]['plan']['reason']
    run = orchestrator(Returns(unvalidated(active_arms=('right',))), executor, Verifier([]),
                       max_failures_without_success=1, trained_combinations=LEFT_ONLY).run(2)
    assert executor.requests == [] and 'has no trained S1 policy' in run.steps[0]['orchestrator_rejection']
    as_stored = [['pick', ['left']], ['place', ['left']]]  # the checkpoint's JSON form
    for combinations in (LEFT_ONLY, as_stored):
        run = orchestrator(Returns(unvalidated()), Executor(['completed']), Verifier([Verification(True)]),
                           trained_combinations=combinations).run(1)
        assert run.steps[0]['execution']['status'] == 'completed'


def test_orchestrator_refuses_combinations_inconsistent_with_trained_skills():
    with pytest.raises(ValueError, match='trained_skills'):
        orchestrator(ScriptedPlanner([]), Executor([]), Verifier([]), trained_combinations={('pick', ('left',))})
