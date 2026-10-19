"""Planner boundary shared by the Claude planner and the explicit scripted planner.

S2 output is a submission naming a skill, arm(s), fragment and, for goal-directed skills, a
screened GARF candidate ID. The goal pose itself is looked up from the CandidateBook, so the
planner can choose a target but cannot invent or mistype one.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Callable, Mapping

from ..contracts.actions import ARMS
from ..contracts.skills import (
    SKILL_IDS, SKILLS, SkillRejected, SkillRequest, trained_combination_set, validate_skill_request)
from .candidates import CandidateBook

PLAN_STATUSES = ('request', 'no_request', 'refused', 'error')
_NULLABLE_STRING = {'anyOf': [{'type': 'string'}, {'type': 'null'}]}

# Strict tool schemas: no numeric or array-length keywords (unsupported in strict mode);
# every range is checked client-side by the tool and the contracts.
CANDIDATES_TOOL = {
    'name': 'propose_assembly_candidates',
    'description': ('Run GARF on registered complete fragment scans to get unranked assembly hypotheses, each with a '
                    'geometry screen (penetration/contact). Call this before choosing a goal-directed skill '
                    '(place/align/insert) whenever no screened candidate exists for the current observation. '
                    'Use 1-8 distinct non-negative integer seeds. It never moves the robot.'),
    'strict': True,
    'input_schema': {
        'type': 'object', 'additionalProperties': False,
        'required': ['observation_id', 'fragment_ids', 'anchor_id', 'seeds'],
        'properties': {
            'observation_id': {'type': 'string', 'description': 'observation_id from the current observation'},
            'fragment_ids': {'type': 'array', 'items': {'type': 'string'},
                             'description': '2-20 distinct fragment ids, including the anchor'},
            'anchor_id': {'type': 'string', 'description': 'a localized fragment that fixes the world frame'},
            'seeds': {'type': 'array', 'items': {'type': 'integer'}},
        },
    },
}


def submit_tool(allowed_skills):
    return {
        'name': 'submit_skill_request',
        'description': ('Submit exactly one skill request for this observation; this ends your turn. Goal-directed '
                        'skills must cite a candidate_id whose geometry_status is needs_motion_validation. Use '
                        'reobserve (no arms, no fragment, no candidate) when no executable step is justified.'),
        'strict': True,
        'input_schema': {
            'type': 'object', 'additionalProperties': False,
            'required': ['observation_id', 'skill_id', 'active_arms', 'fragment_id', 'support_fragment_id',
                         'target_candidate_id', 'rationale'],
            'properties': {
                'observation_id': {'type': 'string'},
                'skill_id': {'type': 'string', 'enum': list(allowed_skills)},
                'active_arms': {'type': 'array', 'items': {'type': 'string', 'enum': list(ARMS)}},
                'fragment_id': _NULLABLE_STRING,
                'support_fragment_id': _NULLABLE_STRING,
                'target_candidate_id': _NULLABLE_STRING,
                'rationale': {'type': 'string', 'description': 'one or two sentences citing observed facts'},
            },
        },
    }


@dataclass(frozen=True)
class PlanOutcome:
    status: str
    skill_request: SkillRequest | None = None
    reason: str = ''
    retryable: bool = False
    trace: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self):
        if self.status not in PLAN_STATUSES:
            raise ValueError(f'unknown plan status {self.status!r}')
        if (self.status == 'request') != (self.skill_request is not None):
            raise ValueError('a skill request is present exactly when status is "request"')


@dataclass
class PlanningContext:
    """Everything one planning call may use. A context is valid for one observation only."""

    observation: object
    trained_skills: tuple[str, ...]
    candidate_source: object | None
    completion_defaults: Mapping[str, Mapping[str, object]]
    timeouts_s: Mapping[str, float]
    feedback: tuple[Mapping[str, object], ...] = ()
    # (skill_id, arms) pairs from the S1 checkpoint; None when no checkpoint constrains the arms.
    trained_combinations: frozenset | None = None
    new_request_id: Callable[[], str] = field(default=lambda: f's2-{uuid.uuid4().hex[:12]}')

    def __post_init__(self):
        if self.trained_combinations is not None:
            self.trained_combinations = trained_combination_set(self.trained_combinations,
                                                                trained_skills=self.trained_skills)
        self.book = CandidateBook(self.observation)

    @property
    def allowed_skills(self):
        return tuple(skill for skill in SKILL_IDS if skill in self.trained_skills or not SKILLS[skill].moves)

    def tools(self):
        tools = [submit_tool(self.allowed_skills)]
        if self.candidate_source is not None:
            tools.insert(0, CANDIDATES_TOOL)
        return tools

    def run_candidates(self, tool_input):
        """Execute propose_assembly_candidates; returns (json_text, is_error) for a tool_result."""
        if self.candidate_source is None:
            return json.dumps({'error': 'no GARF candidate source configured'}), True
        result = self.candidate_source.propose(dict(tool_input),
                                               current_observation_id=self.observation.observation_id)
        if result.get('status') not in ('candidates', 'incomplete'):
            return json.dumps({'error': result.get('reason', 'GARF call failed'), 'status': result.get('status')}), True
        try:
            records = self.book.add_result(result, tool_input['anchor_id'], tool_input['fragment_ids'])
        except (KeyError, ValueError, TypeError) as exc:
            return json.dumps({'error': str(exc)}), True
        failed = [c.get('reason') for c in result.get('candidates', []) if c.get('status') != 'candidate']
        return json.dumps({'status': result['status'], 'geometry_status': result.get('geometry_status', 'unscreened'),
                           'candidates': [self.book.summary(record) for record in records],
                           'failed_seeds': failed, 'cache_hit': result.get('cache_hit', False)}), False

    def resolve(self, submission):
        """Submission (submit_skill_request input) -> validated SkillRequest, or SkillRejected."""
        expected = set(submit_tool(()).get('input_schema')['required'])
        if set(submission) != expected:
            raise SkillRejected(f'submission fields must be exactly {sorted(expected)}')
        skill_id = submission['skill_id']
        if skill_id not in self.allowed_skills:
            raise SkillRejected(f'skill {skill_id!r} is not available to this planner')
        fragment_id, candidate_id = submission['fragment_id'], submission['target_candidate_id']
        pose = provenance = None
        if candidate_id is not None:
            if not SKILLS[skill_id].needs_target:
                raise SkillRejected(f'{skill_id} takes no target candidate')
            try:
                record = self.book.record(candidate_id)
            except KeyError as exc:
                raise SkillRejected(str(exc)) from exc
            if fragment_id not in record.world_T_goal:
                raise SkillRejected(f'candidate {candidate_id} has no goal for {fragment_id!r}')
            pose, provenance = record.world_T_goal[fragment_id], record.provenance(fragment_id)
        arms = tuple(submission['active_arms'])
        if len(set(arms)) == len(arms):
            arms = tuple(arm for arm in ARMS if arm in arms)  # canonical order: S1 conditions on it
        state = self.observation.fragments.get(fragment_id) if fragment_id is not None else None
        fragment_pose = fragment_provenance = None
        if state is not None and state.localized:
            fragment_pose = state.world_T_part
            fragment_provenance = {'source': 'observed_geometry', 'observation_id': self.observation.observation_id,
                                   'pose_source': state.pose_source}
        request = SkillRequest(
            request_id=self.new_request_id(), observation_id=submission['observation_id'], skill_id=skill_id,
            active_arms=arms, fragment_id=fragment_id,
            support_fragment_id=submission['support_fragment_id'], target_pose=pose, target_provenance=provenance,
            fragment_pose=fragment_pose, fragment_pose_provenance=fragment_provenance,
            completion_predicate=self.completion_defaults[skill_id], timeout_s=self.timeouts_s[skill_id],
            rationale=str(submission['rationale']))
        return validate_skill_request(request, trained_skills=self.trained_skills,
                                      known_fragments=tuple(self.observation.fragments),
                                      current_observation_id=self.observation.observation_id,
                                      trained_combinations=self.trained_combinations)


class ScriptedPlanner:
    """Explicit operator-authored plans for validating S1/S0 without a model service.

    Each step maps a PlanningContext to a submit_skill_request input (or None to stop). It goes
    through the same resolution and validation as model output; it is never a silent fallback.
    """

    def __init__(self, steps):
        self._steps = list(steps)
        self._next = 0

    def plan(self, context):
        if self._next >= len(self._steps):
            return PlanOutcome('no_request', reason='script exhausted')
        step = self._steps[self._next]
        self._next += 1
        submission = step(context)
        if submission is None:
            return PlanOutcome('no_request', reason='script chose no request')
        try:
            request = context.resolve(submission)
        except SkillRejected as exc:
            return PlanOutcome('error', reason=f'scripted submission rejected: {exc}')
        return PlanOutcome('request', skill_request=request, trace=({'planner': 'scripted', 'step': self._next - 1},))
