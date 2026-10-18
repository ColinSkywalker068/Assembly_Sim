"""Provider-neutral S2 tool boundary. Backends load once, requests do not move robots."""
from __future__ import annotations
import hashlib
import time
import numpy as np
from ..perception import rigid_transform

TOOL_SPEC = {
    'name': 'propose_assembly',
    'description': 'Predict assembly poses from registered complete fragment geometry IDs; does not command motion.',
    'input_schema': {
        'type': 'object', 'additionalProperties': False,
        'required': ['observation_id', 'fragment_ids', 'anchor_id', 'seed'],
        'properties': {
            'observation_id': {'type': 'string'},
            'fragment_ids': {'type': 'array', 'minItems': 2, 'maxItems': 20,
                             'uniqueItems': True, 'items': {'type': 'string'}},
            'anchor_id': {'type': 'string'}, 'seed': {'type': 'integer', 'minimum': 0, 'maximum': 4294967295},
        },
    },
}


def validate_geometry(points, normals):
    p, n = np.asarray(points, dtype=np.float32), np.asarray(normals, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 32 or p.shape != n.shape:
        raise ValueError('each part needs >=32 xyz points and matching normals')
    if not np.isfinite(p).all() or not np.isfinite(n).all():
        raise ValueError('nonfinite geometry')
    lengths = np.linalg.norm(n, axis=1)
    if (lengths < 1e-6).any() or np.max(np.ptp(p, axis=0)) < 1e-6:
        raise ValueError('degenerate points/normals')
    return p.copy(), n / lengths[:, None]


class GarfTool:
    def __init__(self, backend, geometry, checkpoint_sha256):
        # Registry is populated by application, never arbitrary paths from S2.
        self.geometry = {key: validate_geometry(*value) for key, value in geometry.items()}
        self.backend = backend
        self.checkpoint_sha256 = checkpoint_sha256

    def __call__(self, request, *, current_observation_id):
        start = time.monotonic()
        try:
            if set(request) != {'observation_id', 'fragment_ids', 'anchor_id', 'seed'}:
                raise ValueError('unexpected or missing request fields')
            if request['observation_id'] != current_observation_id:
                raise ValueError('stale observation')
            ids = request['fragment_ids']
            if not isinstance(ids, list) or not 2 <= len(ids) <= 20 or not all(isinstance(i, str) for i in ids):
                raise ValueError('invalid fragment IDs')
            if len(set(ids)) != len(ids) or request['anchor_id'] not in ids:
                raise ValueError('duplicate IDs or missing anchor')
            if type(request['seed']) is not int or not 0 <= request['seed'] <= 4294967295:
                raise ValueError('invalid seed')
            parts = [self.geometry[i] for i in ids]
            digest = hashlib.sha256()
            for key, (p, n) in zip(ids, parts):
                digest.update(key.encode()); digest.update(p.tobytes()); digest.update(n.tobytes())
            if hasattr(self.backend, 'predict_registered'):
                poses = self.backend.predict_registered(ids, ids.index(request['anchor_id']), request['seed'])
            else:
                poses = self.backend.predict(parts, ids.index(request['anchor_id']), request['seed'])
            if len(poses) != len(ids):
                raise ValueError('backend returned wrong part count')
            matrices = [rigid_transform(t).tolist() for t in poses]
            return {'status': 'candidate', 'observation_id': current_observation_id,
                    'assembly_T_fragment': dict(zip(ids, matrices)), 'units': 'm',
                    'geometry_sha256': digest.hexdigest(), 'checkpoint_sha256': self.checkpoint_sha256,
                    'backend': dict(getattr(self.backend, 'metadata', {})),
                    'seed': request['seed'], 'elapsed_s': time.monotonic()-start,
                    'requires_geometric_and_motion_validation': True, 'confidence': None}
        except (ValueError, KeyError, TypeError) as exc:
            return {'status': 'rejected', 'reason': str(exc)}


CANDIDATE_TOOL_SPEC = {
    'name': 'propose_assembly_candidates',
    'description': 'Sample up to eight GARF noise seeds with fixed geometry sampling; outputs unranked hypotheses, never motion.',
    'input_schema': {
        'type':'object', 'additionalProperties':False,
        'required':['observation_id','fragment_ids','anchor_id','seeds'],
        'properties': {**{k:v for k,v in TOOL_SPEC['input_schema']['properties'].items() if k!='seed'},
                       'seeds':{'type':'array','minItems':1,'maxItems':8,'uniqueItems':True,
                                'items':{'type':'integer','minimum':0,'maximum':4294967295}}},
    },
}


def propose_candidates(tool, request, *, current_observation_id):
    if set(request) != {'observation_id','fragment_ids','anchor_id','seeds'}:
        return {'status':'rejected','reason':'unexpected candidate request fields'}
    seeds = request['seeds']
    if not isinstance(seeds,list) or not 1 <= len(seeds) <= 8 or any(type(s) is not int or not 0 <= s <= 4294967295 for s in seeds) or len(set(seeds)) != len(seeds):
        return {'status':'rejected','reason':'require 1..8 unique integer seeds'}
    base = {k:v for k,v in request.items() if k!='seeds'}
    results = [tool({**base,'seed':seed},current_observation_id=current_observation_id) for seed in seeds]
    return {'status':'candidates' if all(r['status']=='candidate' for r in results) else 'incomplete',
            'candidates':results, 'ranking':None, 'selected_candidate':None,
            'requires_geometric_and_motion_validation':True}


def fuse_verified_cluster(geometry, cluster_T_members, *, stability_verified):
    """Caller supplies independently observed poses in a rigid cluster frame.

    No GT or GARF prediction may be used to certify stability. Internal fracture
    surfaces are retained: this is an explicit OOD baseline, not a trained feature.
    """
    if stability_verified is not True:
        raise ValueError('cluster requires independently verified rigid stability')
    if len(cluster_T_members)<2:
        raise ValueError('cluster needs at least two parts')
    points, normals = [], []
    transforms = {}
    for key, value in cluster_T_members.items():
        t = rigid_transform(value); p,n = validate_geometry(*geometry[key])
        points.append(p @ t[:3,:3].T+t[:3,3]); normals.append(n @ t[:3,:3].T)
        transforms[key] = t.copy()
    return (np.concatenate(points),np.concatenate(normals)), transforms


def expand_cluster_pose(assembly_T_cluster, cluster_T_members):
    t = rigid_transform(assembly_T_cluster)
    return {key:t @ rigid_transform(value) for key,value in cluster_T_members.items()}
