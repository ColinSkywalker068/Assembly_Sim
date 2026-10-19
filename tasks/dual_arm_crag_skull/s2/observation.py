"""What S2 may see: observed fragment poses, verified clusters, keyframes and execution feedback.

Simulator ground truth is privileged. An observation that contains it must say so (`oracle=True`)
so episodes from oracle ablations can never be mistaken for deployment-style runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..contracts.skills import POSE_SOURCES
from ..perception import rigid_transform

CAMERAS = ('left_wrist', 'right_wrist', 'agent')


@dataclass(frozen=True)
class FragmentState:
    fragment_id: str
    world_T_part: np.ndarray | None
    pose_source: str | None = None
    inlier_rmse_m: float | None = None

    def __post_init__(self):
        if self.world_T_part is None:
            if self.pose_source is not None:
                raise ValueError(f'{self.fragment_id}: unlocalized fragments have no pose source')
            return
        if self.pose_source not in POSE_SOURCES:
            raise ValueError(f'{self.fragment_id}: pose source must be one of {POSE_SOURCES}')
        pose = np.array(rigid_transform(self.world_T_part), copy=True)
        pose.flags.writeable = False
        object.__setattr__(self, 'world_T_part', pose)

    @property
    def localized(self):
        return self.world_T_part is not None


@dataclass(frozen=True)
class Observation:
    observation_id: str
    time: float
    fragments: Mapping[str, FragmentState]
    verified_clusters: tuple[frozenset[str], ...] = ()
    keyframes_png: Mapping[str, bytes] = field(default_factory=dict)
    robot_summary: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.observation_id:
            raise ValueError('observation needs an id')
        if not np.isfinite(self.time):
            raise ValueError('observation time must be finite')
        if any(key != state.fragment_id for key, state in self.fragments.items()):
            raise ValueError('fragment map keys must match fragment ids')
        unknown_cameras = set(self.keyframes_png) - set(CAMERAS)
        if unknown_cameras:
            raise ValueError(f'unknown cameras: {sorted(unknown_cameras)}')
        clusters = tuple(frozenset(cluster) for cluster in self.verified_clusters)
        members = [member for cluster in clusters for member in cluster]
        if len(members) != len(set(members)) or not set(members) <= set(self.fragments):
            raise ValueError('verified clusters must be disjoint sets of known fragments')
        object.__setattr__(self, 'verified_clusters', clusters)
        object.__setattr__(self, 'fragments', MappingProxyType(dict(self.fragments)))
        object.__setattr__(self, 'keyframes_png', MappingProxyType(dict(self.keyframes_png)))
        object.__setattr__(self, 'robot_summary', MappingProxyType(dict(self.robot_summary)))

    @property
    def oracle(self):
        return any(state.pose_source == 'oracle_sim' for state in self.fragments.values())

    def summary(self):
        """JSON-ready text facts for the planner; poses are reported as positions, not matrices."""
        fragments = []
        for fragment_id, state in sorted(self.fragments.items()):
            entry = {'fragment_id': fragment_id, 'localized': state.localized, 'pose_source': state.pose_source}
            if state.localized:
                entry['position_m'] = [round(float(v), 4) for v in state.world_T_part[:3, 3]]
            if state.inlier_rmse_m is not None:
                entry['inlier_rmse_m'] = round(float(state.inlier_rmse_m), 5)
            fragments.append(entry)
        return {'observation_id': self.observation_id, 'time_s': round(float(self.time), 3),
                'oracle_poses_present': self.oracle, 'fragments': fragments,
                'verified_clusters': [sorted(cluster) for cluster in self.verified_clusters],
                'robot': dict(self.robot_summary), 'keyframes': sorted(self.keyframes_png)}
