"""S2 -> S1 skill request. S2 picks a trained skill and names its target; it never commands motion.

Target poses are `world_T_fragment_goal` matrices resolved by the orchestrator from a recorded
source (a screened GARF candidate, observed geometry or an operator), never transcribed by S2.
`fragment_pose` is the fragment's observed current pose (`world_T_fragment`), copied from the
observation so S1 knows which fragment to act on; pick and support cannot run without it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..perception import rigid_transform
from .actions import ARMS, ContactWindow


@dataclass(frozen=True)
class SkillSpec:
    moves: bool
    arm_count: tuple[int, int]
    needs_fragment: bool
    needs_target: bool
    completions: tuple[str, ...]
    allows_support: bool = False
    needs_fragment_pose: bool = False


SKILLS = {
    'pick': SkillSpec(True, (1, 1), True, False, ('grasp_held',), needs_fragment_pose=True),
    'place': SkillSpec(True, (1, 1), True, True, ('pose_within', 'released_stable'), allows_support=True),
    'align': SkillSpec(True, (1, 2), True, True, ('pose_within',), allows_support=True),
    'insert': SkillSpec(True, (1, 2), True, True, ('pose_within', 'released_stable'), allows_support=True),
    'support': SkillSpec(True, (1, 1), True, False, ('grasp_held',), needs_fragment_pose=True),
    'retreat': SkillSpec(True, (1, 2), False, False, ('retreated',)),
    'reobserve': SkillSpec(False, (0, 0), False, False, ('observed',)),
}
SKILL_IDS = tuple(SKILLS)  # Stable order: S1 conditions on the index into this tuple.
# Required numeric parameters and their allowed open-closed ranges.
COMPLETIONS = {
    'grasp_held': {},
    'pose_within': {'position_tolerance_m': (0.0, 0.05), 'rotation_tolerance_deg': (0.0, 30.0)},
    'released_stable': {'position_tolerance_m': (0.0, 0.05), 'rotation_tolerance_deg': (0.0, 30.0),
                        'settle_s': (0.0, 10.0)},
    'retreated': {'clearance_m': (0.0, 0.5)},
    'observed': {},
}
# The contact a moving skill's chunks declare for S0 (contracts.actions.ContactWindow) and the request pose it is
# centred on. A retreat keeps each arm's previous contact instead (contact_windows).
CONTACTS = {'pick': ('grasp', 'fragment_pose'), 'support': ('grasp', 'fragment_pose'),
            'place': ('place', 'target_pose'), 'align': ('place', 'target_pose'), 'insert': ('place', 'target_pose')}
TARGET_SOURCES = ('garf_candidate', 'observed_geometry', 'operator')
# Where an observed fragment pose came from; 'oracle_sim' is simulator truth and marks an oracle run.
POSE_SOURCES = ('icp', 'fused_cluster', 'operator', 'oracle_sim')
MAX_TIMEOUT_S = 120.0


class SkillRejected(ValueError):
    """A request S1 must not receive; the message is the structured rejection reason."""


@dataclass(frozen=True)
class SkillRequest:
    request_id: str
    observation_id: str
    skill_id: str
    active_arms: tuple[str, ...]
    completion_predicate: Mapping[str, object]
    timeout_s: float
    fragment_id: str | None = None
    support_fragment_id: str | None = None
    target_frame: str = 'world'
    target_pose: np.ndarray | None = None
    target_provenance: Mapping[str, object] | None = None
    fragment_pose: np.ndarray | None = None
    fragment_pose_provenance: Mapping[str, object] | None = None
    rationale: str = ''
    extras: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'active_arms', tuple(self.active_arms))
        object.__setattr__(self, 'completion_predicate', MappingProxyType(dict(self.completion_predicate)))
        object.__setattr__(self, 'extras', MappingProxyType(dict(self.extras)))
        for name in ('target_provenance', 'fragment_pose_provenance'):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        for name in ('target_pose', 'fragment_pose'):
            if getattr(self, name) is not None:
                pose = np.array(getattr(self, name), dtype=float, copy=True)
                pose.flags.writeable = False
                object.__setattr__(self, name, pose)

    def to_record(self):
        """JSON-serializable record for episode logs."""
        return {
            'request_id': self.request_id, 'observation_id': self.observation_id, 'skill_id': self.skill_id,
            'active_arms': list(self.active_arms), 'fragment_id': self.fragment_id,
            'support_fragment_id': self.support_fragment_id, 'target_frame': self.target_frame,
            'target_pose': None if self.target_pose is None else np.asarray(self.target_pose).tolist(),
            'target_provenance': None if self.target_provenance is None else dict(self.target_provenance),
            'fragment_pose': None if self.fragment_pose is None else np.asarray(self.fragment_pose).tolist(),
            'fragment_pose_provenance': (None if self.fragment_pose_provenance is None
                                         else dict(self.fragment_pose_provenance)),
            'completion_predicate': dict(self.completion_predicate), 'timeout_s': self.timeout_s,
            'rationale': self.rationale, 'extras': dict(self.extras),
        }


def check_completion(skill_id, predicate):
    """Raise SkillRejected unless `predicate` is an allowed, fully parameterized completion test."""
    spec = SKILLS[skill_id]
    kind = predicate.get('type') if isinstance(predicate, Mapping) else None
    if kind not in spec.completions:
        raise SkillRejected(f'{skill_id} completion must be one of {spec.completions}, got {kind!r}')
    expected = COMPLETIONS[kind]
    if set(predicate) != {'type', *expected}:
        raise SkillRejected(f'{kind} completion needs exactly {sorted(expected)}')
    for name, (low, high) in expected.items():
        value = predicate[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low < float(value) <= high:
            raise SkillRejected(f'{kind}.{name} must be in ({low}, {high}]')


def _check_target(request, spec, current_observation_id):
    if not spec.needs_target:
        if request.target_pose is not None or request.target_provenance is not None:
            raise SkillRejected(f'{request.skill_id} takes no target pose')
        return
    if request.target_pose is None or request.target_provenance is None:
        raise SkillRejected(f'{request.skill_id} needs a target pose with provenance')
    try:
        rigid_transform(request.target_pose)
    except ValueError as exc:
        raise SkillRejected(f'target pose: {exc}') from exc
    provenance = request.target_provenance
    if provenance.get('source') not in TARGET_SOURCES:
        raise SkillRejected(f'target source must be one of {TARGET_SOURCES}')
    if provenance.get('observation_id') != current_observation_id:
        raise SkillRejected('target was resolved against a different observation')
    if 'fragment_id' in provenance and provenance['fragment_id'] != request.fragment_id:
        raise SkillRejected('target provenance names a different fragment')
    if provenance['source'] == 'garf_candidate':
        if provenance.get('geometry_status') != 'needs_motion_validation':
            raise SkillRejected('GARF target lacks a passing geometry screen')
        if not provenance.get('candidate_id') or not provenance.get('checkpoint_sha256'):
            raise SkillRejected('GARF target must record candidate_id and checkpoint_sha256')
        if provenance.get('fragment_id') != request.fragment_id:
            raise SkillRejected('GARF target must record the fragment it was resolved for')


def _check_fragment_pose(request, spec, current_observation_id):
    if request.fragment_pose is None:
        if request.fragment_pose_provenance is not None:
            raise SkillRejected('fragment pose provenance without a fragment pose')
        if spec.needs_fragment_pose:
            raise SkillRejected(f'{request.skill_id} needs the observed pose of the fragment; reobserve')
        return
    if request.fragment_id is None:
        raise SkillRejected(f'{request.skill_id} takes no fragment pose')
    try:
        rigid_transform(request.fragment_pose)
    except ValueError as exc:
        raise SkillRejected(f'fragment pose: {exc}') from exc
    provenance = request.fragment_pose_provenance or {}
    if provenance.get('source') != 'observed_geometry' or provenance.get('pose_source') not in POSE_SOURCES:
        raise SkillRejected('fragment pose must come from observed geometry with a known pose source')
    if provenance.get('observation_id') != current_observation_id:
        raise SkillRejected('fragment pose comes from a different observation')


def trained_combination_set(combinations, *, trained_skills):
    """Validated frozenset of the (skill_id, arms in ARMS order) pairs S1 was trained on.

    The input is typically S1Runtime.trained_combinations; its skills must be exactly `trained_skills`.
    """
    result = frozenset((skill_id, tuple(arms)) for skill_id, arms in combinations)
    for skill_id, arms in result:
        spec = SKILLS.get(skill_id)
        if (spec is None or not spec.moves or arms != tuple(arm for arm in ARMS if arm in arms)
                or not spec.arm_count[0] <= len(arms) <= spec.arm_count[1]):
            raise ValueError(f'invalid trained combination {(skill_id, arms)!r}')
    if {skill_id for skill_id, _ in result} != set(trained_skills):
        raise ValueError('trained_skills must be exactly the skills of the trained combinations')
    return result


def contact_windows(request, previous=None):
    """The ContactWindows each chunk for `request` declares, one per active arm in ARMS order.

    pick and support centre on the fragment's observed pose; place, align and insert on the target pose. A retreat
    declares nothing of its own: each active arm keeps its previous grasp or place window (`previous`, arm ->
    ContactWindow), so a retreat can only leave a place where that arm was already allowed near the table, and an
    arm without one gets none. reobserve declares none.
    """
    if request.skill_id == 'retreat':
        previous = previous or {}
        for arm, window in previous.items():
            if not isinstance(window, ContactWindow) or window.arm != arm:
                raise ValueError(f'previous contact for {arm} must be a ContactWindow of that arm')
        return tuple(previous[arm] for arm in ARMS if arm in request.active_arms and arm in previous)
    if request.skill_id not in CONTACTS:
        return ()
    kind, source = CONTACTS[request.skill_id]
    pose = getattr(request, source)
    if pose is None:
        raise ValueError(f'{request.skill_id} needs its {source} to declare where it works near the table')
    return tuple(ContactWindow(arm=arm, kind=kind, center_xy=np.asarray(pose, dtype=float)[:2, 3],
                               fragment_id=request.fragment_id) for arm in ARMS if arm in request.active_arms)


class ContactMemory:
    """Each arm's last grasp or place window in a chunk S0 accepted: the windows a retreat keeps.

    The executor reports every chunk S0 accepts (`accepted`) and calls `reset()` when an episode starts, because an
    old window may lie under another fragment in a new layout. A window that was only declared (its chunk was
    rejected, or never ran) is not remembered, so a retreat never inherits a place the arm was not allowed to be.
    Until an arm has an accepted window, a retreat declares nothing for it and S0 keeps it off the table.
    """

    def __init__(self):
        self._windows = {}

    def windows(self, request):
        """contact_windows for `request`, with this memory as each arm's previous window."""
        return contact_windows(request, previous=self._windows)

    def accepted(self, chunk):
        self._windows = {**self._windows, **{contact.arm: contact for contact in chunk.contacts}}

    def reset(self):
        self._windows = {}


def validate_skill_request(request, *, trained_skills, known_fragments, current_observation_id,
                           trained_combinations=None):
    """Raise SkillRejected unless S1 has been trained for this request and its target is grounded.

    `trained_combinations` ((skill_id, arms in ARMS order) pairs, as trained_combination_set returns or a
    checkpoint stores them) also limits each moving skill to the arm sets S1 was trained on; None checks
    skills only, for runs without an S1 checkpoint.
    """
    if not isinstance(request.request_id, str) or not request.request_id:
        raise SkillRejected('missing request_id')
    if request.observation_id != current_observation_id:
        raise SkillRejected('stale observation')
    if request.skill_id not in SKILLS:
        raise SkillRejected(f'unknown skill {request.skill_id!r}')
    spec = SKILLS[request.skill_id]
    if spec.moves and request.skill_id not in trained_skills:
        raise SkillRejected(f'skill {request.skill_id!r} has no trained S1 policy')
    arms = tuple(request.active_arms)
    low, high = spec.arm_count
    if len(set(arms)) != len(arms) or not set(arms) <= set(ARMS) or not low <= len(arms) <= high:
        raise SkillRejected(f'{request.skill_id} needs {low}..{high} distinct arms from {ARMS}')
    canonical = tuple(arm for arm in ARMS if arm in arms)
    if spec.moves and trained_combinations is not None and (request.skill_id, canonical) not in {
            (skill_id, tuple(trained_arms)) for skill_id, trained_arms in trained_combinations}:
        raise SkillRejected(f'{request.skill_id} with arms {list(canonical)} has no trained S1 policy')
    fragments = set(known_fragments)
    if spec.needs_fragment != (request.fragment_id is not None):
        raise SkillRejected(f'{request.skill_id} {"needs" if spec.needs_fragment else "takes no"} fragment_id')
    if request.fragment_id is not None and request.fragment_id not in fragments:
        raise SkillRejected(f'unknown fragment {request.fragment_id!r}')
    if request.support_fragment_id is not None:
        if not spec.allows_support:
            raise SkillRejected(f'{request.skill_id} takes no support fragment')
        if request.support_fragment_id not in fragments or request.support_fragment_id == request.fragment_id:
            raise SkillRejected('support fragment must be a different known fragment')
    if request.target_frame != 'world':
        raise SkillRejected('only world-frame targets are implemented')
    _check_target(request, spec, current_observation_id)
    _check_fragment_pose(request, spec, current_observation_id)
    check_completion(request.skill_id, request.completion_predicate)
    timeout = request.timeout_s
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < float(timeout) <= MAX_TIMEOUT_S:
        raise SkillRejected(f'timeout_s must be in (0, {MAX_TIMEOUT_S}]')
    return request
