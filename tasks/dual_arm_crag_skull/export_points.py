"""Sample complete scan fragments for GARF, with independently randomized local frames."""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from .preparation import load_skull


def export(source, output, target_length=.2):
    assembly = load_skull(Path(source))
    world = []
    for fragment in assembly.fragments:
        mesh = fragment.mesh.copy(); mesh.apply_transform(fragment.ground_truth_transform)
        world.append(mesh)
    scale = target_length / trimesh.util.concatenate(world).extents.max()
    rng = np.random.default_rng(42)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    registry, truth = {}, {}
    for index, f in enumerate(assembly.fragments):
        count = 64 + int((8192-64*len(assembly.fragments))*f.mesh.area/sum(part.mesh.area for part in assembly.fragments))
        points, faces = trimesh.sample.sample_surface(f.mesh, count, seed=index)
        center = points.mean(axis=0)
        rot = Rotation.random(random_state=rng).as_matrix()
        local_points = (points-center) @ rot.T * scale
        normals = f.mesh.face_normals[faces] @ rot.T
        path = output/(f.name+'.npz')
        np.savez(path, points=local_points.astype('float32'), normals=normals.astype('float32'))
        registry[f.name] = str(path.resolve())
        pose = np.eye(4)
        pose[:3,:3] = f.ground_truth_transform[:3,:3] @ rot.T
        pose[:3,3] = (f.ground_truth_transform[:3,:3] @ center + f.ground_truth_transform[:3,3])*scale
        truth[f.name] = pose.tolist()
    (output/'registry.json').write_text(json.dumps(registry,indent=2))
    # Separate privileged file, never passed to GARF.
    (output/'evaluation_gt.json').write_text(json.dumps(truth,indent=2))
    ids = list(registry)
    request = {'current_observation_id':'scan-1', 'request':{'observation_id':'scan-1',
                'fragment_ids':ids, 'anchor_id':max(assembly.fragments, key=lambda f:f.mesh.area).name, 'seed':42}}
    (output/'request.jsonl').write_text(json.dumps(request)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    export(args.input,args.output)
