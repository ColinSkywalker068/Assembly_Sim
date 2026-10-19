"""Shared fakes for S2 tests: a two-fragment observation and a GARF-like candidate source."""
import numpy as np
from scipy.spatial.transform import Rotation

from tasks.dual_arm_crag_skull.s2.observation import FragmentState, Observation

COMPLETIONS = {
    'pick': {'type': 'grasp_held'},
    'place': {'type': 'released_stable', 'position_tolerance_m': .003, 'rotation_tolerance_deg': 3.0, 'settle_s': 1.0},
    'align': {'type': 'pose_within', 'position_tolerance_m': .003, 'rotation_tolerance_deg': 3.0},
    'insert': {'type': 'pose_within', 'position_tolerance_m': .002, 'rotation_tolerance_deg': 2.0},
    'support': {'type': 'grasp_held'},
    'retreat': {'type': 'retreated', 'clearance_m': .1},
    'reobserve': {'type': 'observed'},
}
TIMEOUTS = {'pick': 20.0, 'place': 20.0, 'align': 15.0, 'insert': 15.0, 'support': 20.0, 'retreat': 10.0,
            'reobserve': 5.0}


def pose(translation, rotvec=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    t[:3, 3] = translation
    return t


ANCHOR_WORLD = pose([.0, .1, .80], [0, 0, .4])
PART1_WORLD = pose([.3, -.2, .78], [.5, 0, 0])
ASSEMBLY = {'part_0': pose([.01, 0, 0], [0, .1, 0]), 'part_1': pose([.05, .02, 0], [0, 0, .3])}


def observation(observation_id='obs-1', part1_localized=True, clusters=()):
    fragments = {'part_0': FragmentState('part_0', ANCHOR_WORLD, 'icp', .0004),
                 'part_1': FragmentState('part_1', PART1_WORLD if part1_localized else None,
                                         'icp' if part1_localized else None)}
    return Observation(observation_id, 3.0, fragments, verified_clusters=clusters,
                       keyframes_png={'agent': b'\x89PNG fake'})


def expected_goal(fragment):
    return ANCHOR_WORLD @ np.linalg.inv(ASSEMBLY['part_0']) @ ASSEMBLY[fragment]


class FakeCandidateSource:
    """Seed 41 passes the geometry screen; seed 42 is rejected."""

    def __init__(self):
        self.calls = []

    def propose(self, request, *, current_observation_id):
        self.calls.append((dict(request), current_observation_id))
        if request['observation_id'] != current_observation_id:
            return {'status': 'rejected', 'reason': 'stale observation'}
        candidates = [{'status': 'candidate', 'observation_id': current_observation_id, 'seed': seed,
                       'assembly_T_fragment': {k: v.tolist() for k, v in ASSEMBLY.items()},
                       'checkpoint_sha256': 'cc6512ab', 'geometry_sha256': 'g-hash'} for seed in request['seeds']]
        screening = [{'seed': seed, 'rejected': seed != 41, 'max_penetration_fraction': 0.0 if seed == 41 else .2,
                      'contact_components': 1} for seed in request['seeds']]
        eligible = [seed for seed in request['seeds'] if seed == 41]
        return {'status': 'candidates', 'candidates': candidates, 'geometry_screening': screening,
                'geometry_status': 'needs_motion_validation' if eligible else 'all_candidates_rejected',
                'geometrically_eligible_seeds': eligible}
