"""observe -> plan (GARF candidates, screening, skill choice) -> execute -> release and verify -> merge.

Every step plans against a fresh observation and re-validates the planner's request itself before
anything reaches the executor: contract checks against the current observation, a GARF target pose
identical to the booked candidate's goal, and a bound on planning time. A verified cluster is only
recorded after a completed place/insert whose completion test is `released_stable`, never from
GARF output. Failure and idle counters are reset only by a step that executed successfully.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..contracts.skills import SkillRejected, trained_combination_set, validate_skill_request
from .planner import PlanningContext

EXECUTION_STATUSES = ('completed', 'failed', 'rejected', 'timeout')
RUN_STATUSES = ('assembled', 'max_steps', 'too_many_failures', 'stalled', 'error')
MERGING_SKILLS = ('place', 'insert')


@dataclass(frozen=True)
class ExecutionReport:
    status: str
    reason: str = ''
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in EXECUTION_STATUSES:
            raise ValueError(f'unknown execution status {self.status!r}')
        object.__setattr__(self, 'details', MappingProxyType(dict(self.details)))


@dataclass(frozen=True)
class Verification:
    passed: bool
    reason: str = ''
    merged_cluster: frozenset[str] | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.merged_cluster is not None and not self.passed:
            raise ValueError('only a passed verification may merge a cluster')
        if self.merged_cluster is not None:
            object.__setattr__(self, 'merged_cluster', frozenset(self.merged_cluster))
        object.__setattr__(self, 'metrics', MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True)
class RunReport:
    status: str
    steps: tuple[Mapping[str, object], ...]
    verified_clusters: tuple[frozenset[str], ...]
    error: str | None = None


def merge_clusters(clusters, merged):
    """Union `merged` with every existing cluster it overlaps; returns a new tuple."""
    merged = frozenset(merged)
    keep = [cluster for cluster in clusters if not cluster & merged]
    for cluster in clusters:
        if cluster & merged:
            merged = merged | cluster
    return tuple(keep) + (merged,)


def merge_problem(request, cluster, known_fragments):
    """Why `cluster` may not be recorded after `request`, or None."""
    if request.skill_id not in MERGING_SKILLS or request.completion_predicate.get('type') != 'released_stable':
        return 'clusters merge only after a place/insert verified as released_stable'
    if request.fragment_id not in cluster:
        return 'merged cluster does not contain the placed fragment'
    if len(cluster) < 2:
        return 'a cluster needs at least two fragments'
    if not cluster <= set(known_fragments):
        return 'merged cluster names unknown fragments'
    return None


class AssemblyOrchestrator:
    def __init__(self, *, observer, planner, executor, verifier, trained_skills, completion_defaults, timeouts_s,
                 max_planning_s, candidate_source=None, trained_combinations=None, max_failures_without_success=3,
                 max_idle_without_success=3, feedback_window=5, clock=time.monotonic):
        if not np.isfinite(max_planning_s) or max_planning_s <= 0:
            raise ValueError('max_planning_s must be positive')
        self.observer, self.planner, self.executor, self.verifier = observer, planner, executor, verifier
        self.trained_skills = tuple(trained_skills)
        # Pass S1Runtime.trained_combinations so S2 never asks S1 for an arm set it was not trained on.
        self.trained_combinations = (None if trained_combinations is None else
                                     trained_combination_set(trained_combinations, trained_skills=self.trained_skills))
        self.completion_defaults, self.timeouts_s = completion_defaults, timeouts_s
        self.max_planning_s, self.clock = float(max_planning_s), clock
        self.candidate_source = candidate_source
        self.max_failures_without_success = max_failures_without_success
        self.max_idle_without_success = max_idle_without_success
        self.feedback_window = feedback_window
        self.verified_clusters = ()
        self._feedback = []

    def _assembled(self, observation):
        return any(cluster == frozenset(observation.fragments) for cluster in observation.verified_clusters)

    def _recheck(self, request, context, planning_s):
        """The orchestrator's own gate in front of the executor; returns a reason or None."""
        observation = context.observation
        if planning_s > self.max_planning_s:
            return f'planning took {planning_s:.1f} s (limit {self.max_planning_s:.1f} s); observation too old'
        try:
            validate_skill_request(request, trained_skills=self.trained_skills,
                                   known_fragments=tuple(observation.fragments),
                                   current_observation_id=observation.observation_id,
                                   trained_combinations=self.trained_combinations)
        except SkillRejected as exc:
            return f'request failed re-validation: {exc}'
        if request.fragment_pose is not None:
            state = observation.fragments.get(request.fragment_id)
            if state is None or not state.localized or not np.array_equal(state.world_T_part, request.fragment_pose):
                return 'fragment pose differs from the current observation'
        provenance = request.target_provenance
        if provenance is not None and provenance.get('source') == 'garf_candidate':
            try:
                record = context.book.record(provenance['candidate_id'])
            except KeyError as exc:
                return str(exc)
            goal = record.world_T_goal.get(request.fragment_id)
            if goal is None or not np.array_equal(goal, request.target_pose):
                return 'target pose differs from the booked candidate goal'
            if not record.eligible:
                return 'booked candidate did not pass the geometry screen'
        return None

    def step(self, index):
        """One planning event. Returns (record, outcome) with outcome in executed/idle/failed/assembled."""
        observation = self.observer.observe(self.verified_clusters)
        record = {'step': index, 'observation_id': observation.observation_id, 'oracle_poses': observation.oracle}
        if self._assembled(observation):
            return {**record, 'action': 'none', 'reason': 'all fragments in one verified cluster'}, 'assembled'
        context = PlanningContext(observation=observation, trained_skills=self.trained_skills,
                                  candidate_source=self.candidate_source,
                                  completion_defaults=self.completion_defaults, timeouts_s=self.timeouts_s,
                                  feedback=tuple(self._feedback[-self.feedback_window:]),
                                  trained_combinations=self.trained_combinations)
        started = self.clock()
        plan = self.planner.plan(context)
        planning_s = self.clock() - started
        record['plan'] = {'status': plan.status, 'reason': plan.reason, 'planning_s': planning_s,
                          'trace': list(plan.trace)}
        if plan.status != 'request':
            self._feedback.append({'step': index, 'event': f'planner_{plan.status}', 'reason': plan.reason})
            return {**record, 'action': 'hold'}, 'failed' if plan.status in ('error', 'refused') else 'idle'
        request = plan.skill_request
        record['skill_request'] = request.to_record()
        problem = self._recheck(request, context, planning_s)
        if problem:
            self._feedback.append({'step': index, 'event': 'orchestrator_rejected', 'reason': problem})
            return {**record, 'action': 'hold', 'orchestrator_rejection': problem}, 'failed'
        if request.skill_id == 'reobserve':
            self._feedback.append({'step': index, 'event': 'reobserve', 'reason': request.rationale})
            return {**record, 'action': 'reobserve'}, 'idle'
        report = self.executor.execute(request, observation)
        after = self.observer.observe(self.verified_clusters)
        verification = self.verifier.verify(request, observation, after)
        record.update(action='execute', execution={'status': report.status, 'reason': report.reason,
                                                   'details': dict(report.details)},
                      after_observation_id=after.observation_id,
                      verification={'passed': verification.passed, 'reason': verification.reason,
                                    'metrics': dict(verification.metrics)})
        success = report.status == 'completed' and verification.passed
        if success and verification.merged_cluster is not None:
            refusal = merge_problem(request, verification.merged_cluster, observation.fragments)
            if refusal:
                record['merge_rejected'] = refusal
                success = False
            else:
                self.verified_clusters = merge_clusters(self.verified_clusters, verification.merged_cluster)
                record['merged_cluster'] = sorted(verification.merged_cluster)
        self._feedback.append({'step': index, 'event': 'skill', 'skill_id': request.skill_id,
                               'fragment_id': request.fragment_id, 'execution': report.status,
                               'execution_reason': report.reason, 'verified': verification.passed,
                               'verification_reason': record.get('merge_rejected', verification.reason)})
        return record, 'executed' if success else 'failed'

    def run(self, max_steps):
        steps, failures, idle = [], 0, 0
        for index in range(max_steps):
            try:
                record, outcome = self.step(index)
            except Exception as exc:  # noqa: BLE001 - keep the log of every earlier step, then report loudly
                steps.append({'step': index, 'action': 'error', 'error': repr(exc), 'traceback': traceback.format_exc()})
                return RunReport('error', tuple(steps), self.verified_clusters, error=repr(exc))
            steps.append(record)
            if outcome == 'assembled':
                return RunReport('assembled', tuple(steps), self.verified_clusters)
            if outcome == 'executed':
                failures = idle = 0
            elif outcome == 'failed':
                failures += 1
            elif outcome == 'idle':
                idle += 1
            if failures >= self.max_failures_without_success:
                return RunReport('too_many_failures', tuple(steps), self.verified_clusters)
            if idle >= self.max_idle_without_success:
                return RunReport('stalled', tuple(steps), self.verified_clusters)
        return RunReport('max_steps', tuple(steps), self.verified_clusters)
