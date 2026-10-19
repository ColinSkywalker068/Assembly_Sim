"""GARF hypotheses grounded in one observation: world goals, screening status, planner summaries.

GARF poses live in each registered part's local frame. The anchor's observed pose fixes the
gauge: `world_T_goal_i = world_T_anchor @ inv(assembly_T_anchor) @ assembly_T_i`. Records are
bound to the observation that produced them; a new observation needs new goals.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from ..perception import anchor_goals

ELIGIBLE = 'needs_motion_validation'


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    observation_id: str
    seed: int
    anchor_id: str
    world_T_goal: Mapping[str, np.ndarray]
    geometry_status: str
    screening: Mapping[str, object] | None
    checkpoint_sha256: str
    geometry_sha256: str

    @property
    def eligible(self):
        return self.geometry_status == ELIGIBLE

    def provenance(self, fragment_id):
        return {'source': 'garf_candidate', 'observation_id': self.observation_id,
                'candidate_id': self.candidate_id, 'seed': self.seed, 'anchor_id': self.anchor_id,
                'fragment_id': fragment_id, 'checkpoint_sha256': self.checkpoint_sha256,
                'geometry_sha256': self.geometry_sha256, 'geometry_status': self.geometry_status}


def _frozen_poses(poses):
    result = {}
    for key, value in poses.items():
        pose = np.array(value, dtype=float, copy=True)
        pose.flags.writeable = False
        result[key] = pose
    return MappingProxyType(result)


class CandidateBook:
    """Candidates for exactly one observation. Create a new book for each observation."""

    def __init__(self, observation):
        self.observation = observation
        self._records = {}
        self._calls = 0

    def __contains__(self, candidate_id):
        return candidate_id in self._records

    def record(self, candidate_id):
        if candidate_id not in self._records:
            raise KeyError(f'unknown candidate {candidate_id!r} for observation {self.observation.observation_id}')
        return self._records[candidate_id]

    def add_result(self, result, anchor_id, fragment_ids):
        """Book a propose_assembly_candidates result; returns the new records in result order.

        All candidates are checked before any is booked, so a bad entry leaves the book unchanged.
        IDs carry a per-call index: the same seed with another anchor or fragment set is a new candidate.
        """
        anchor = self.observation.fragments.get(anchor_id)
        if anchor is None or not anchor.localized:
            raise ValueError(f'anchor {anchor_id!r} is not localized in {self.observation.observation_id}')
        requested = set(fragment_ids)
        screening = {entry['seed']: entry for entry in result.get('geometry_screening', [])}
        eligible = set(result.get('geometrically_eligible_seeds', []))
        call = self._calls
        pending = []
        for candidate in result.get('candidates', []):
            if candidate.get('status') != 'candidate':
                continue
            if candidate.get('observation_id') != self.observation.observation_id:
                raise ValueError('candidate belongs to a different observation')
            seed = candidate.get('seed')
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f'candidate seed must be an integer, got {seed!r}')
            poses = candidate.get('assembly_T_fragment')
            if not isinstance(poses, Mapping) or set(poses) != requested:
                raise ValueError('candidate fragment set differs from the request')
            goals = anchor_goals(poses, anchor_id, anchor.world_T_part)
            if seed in eligible:
                status = ELIGIBLE
            else:
                status = 'rejected' if seed in screening else 'unscreened'
            pending.append(CandidateRecord(
                candidate_id=f'{self.observation.observation_id}/call-{call}/seed-{seed}',
                observation_id=self.observation.observation_id, seed=seed, anchor_id=anchor_id,
                world_T_goal=_frozen_poses(goals), geometry_status=status,
                screening=None if seed not in screening else MappingProxyType(dict(screening[seed])),
                checkpoint_sha256=str(candidate.get('checkpoint_sha256', '')),
                geometry_sha256=str(candidate.get('geometry_sha256', ''))))
        if len({record.candidate_id for record in pending}) != len(pending):
            raise ValueError('duplicate seeds in one result')
        self._calls += 1
        for record in pending:
            self._records[record.candidate_id] = record
        return pending

    def summary(self, record):
        """Compact planner view: screening outcome and how far each fragment is from its goal."""
        moves = {}
        for fragment_id, goal in sorted(record.world_T_goal.items()):
            state = self.observation.fragments.get(fragment_id)
            if state is None or not state.localized:
                moves[fragment_id] = {'localized': False}
                continue
            delta = np.linalg.inv(state.world_T_part) @ goal
            moves[fragment_id] = {'translation_m': round(float(np.linalg.norm(goal[:3, 3] - state.world_T_part[:3, 3])), 4),
                                  'rotation_deg': round(float(np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude())), 2)}
        entry = {'candidate_id': record.candidate_id, 'seed': record.seed, 'anchor_id': record.anchor_id,
                 'geometry_status': record.geometry_status, 'goal_offsets_from_current_pose': moves}
        if record.screening is not None:
            entry['max_penetration_fraction'] = record.screening.get('max_penetration_fraction')
            entry['contact_components'] = record.screening.get('contact_components')
        return entry
