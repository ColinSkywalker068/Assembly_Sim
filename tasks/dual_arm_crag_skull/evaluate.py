"""Offline oracle pose evaluation. Never used to select deployment actions."""
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .perception import anchor_goals


def evaluate(candidate, truth, anchor):
    poses = anchor_goals(candidate['assembly_T_fragment'], anchor, truth[anchor])
    errors = {}
    for key, pose in poses.items():
        if key == anchor:
            continue
        target = np.asarray(truth[key])
        errors[key] = {
            'translation_error_m':float(np.linalg.norm(pose[:3,3]-target[:3,3])),
            'rotation_error_deg':float(np.rad2deg(Rotation.from_matrix(pose[:3,:3]@target[:3,:3].T).magnitude()))}
    return {'seed':candidate['seed'], 'elapsed_s':candidate['elapsed_s'], 'anchor':anchor,
            'non_anchor_errors':errors,
            'mean_translation_error_m':float(np.mean([v['translation_error_m'] for v in errors.values()])),
            'mean_rotation_error_deg':float(np.mean([v['rotation_error_deg'] for v in errors.values()]))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',required=True)
    parser.add_argument('--truth',required=True)
    parser.add_argument('--anchor',required=True)
    args = parser.parse_args()
    result,truth = json.load(open(args.result)),json.load(open(args.truth))
    candidates = result['candidates'] if 'candidates' in result else [result]
    print(json.dumps([evaluate(c,truth,args.anchor) for c in candidates if c['status']=='candidate'],indent=2))
