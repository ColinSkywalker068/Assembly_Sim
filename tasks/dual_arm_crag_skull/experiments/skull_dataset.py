"""Offline CRAG HDF5 audit and specimen-grouped GARF adaptation dataset.

Missing shared_faces is UNKNOWN, never an all-negative training label.
Geometric labels are explicit pseudo-labels; ambiguous samples are ignored.
Raw meshes and cached point samples stay outside the repository.
"""
import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def specimen_id(name):
    match = re.match(r"(AMNH_\d+)(?:_|$)", name)
    if not match:
        raise ValueError(f"Unknown specimen naming convention: {name}")
    return match[1]


def grouped_split(names):
    """Predeclare 70/15/15 split by specimen, before any quality/model scores."""
    groups = sorted({specimen_id(n) for n in names},
                    key=lambda g: hashlib.sha256(('jing-skull-v1/' + g).encode()).hexdigest())
    n = len(groups)
    partitions = {g: ('train' if i < int(.7*n) else 'val' if i < int(.85*n) else 'test')
                  for i, g in enumerate(groups)}
    return {name: partitions[specimen_id(name)] for name in sorted(names)}


def load_meshes(group):
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in group['pieces_names'][:]]
    return names, [trimesh.Trimesh(vertices=group['pieces'][n]['vertices'][:],
                                 faces=group['pieces'][n]['faces'][:], process=True) for n in names]


def contact_labels(meshes, data):
    """Opposing nearest surfaces, two tolerances, uncertainty band; no model scores."""
    p = data['pointclouds_gt'].astype(float)
    normals = data['pointclouds_normals_gt'].astype(float)
    extent = np.ptp(np.concatenate([m.vertices for m in meshes]), axis=0).max()
    positive_tol, negative_tol = extent * .0005, extent * .002
    nearest = np.full(len(p), np.inf)
    dots = np.ones(len(p))
    start = 0
    for i, count in enumerate(data['points_per_part']):
        end = start + count
        for j, mesh in enumerate(meshes):
            if i == j:
                continue
            near = ((p[start:end] >= mesh.bounds[0]-negative_tol).all(1) &
                    (p[start:end] <= mesh.bounds[1]+negative_tol).all(1))
            indices = np.flatnonzero(near) + start
            if not len(indices):
                continue
            _, d, face = trimesh.proximity.closest_point(mesh, p[indices])
            better = d < nearest[indices]
            selected = indices[better]
            nearest[selected] = d[better]
            dots[selected] = np.einsum('ij,ij->i', normals[selected], mesh.face_normals[face[better]])
        start = end
    positive = (nearest <= positive_tol) & (dots <= -.8)
    negative = nearest > negative_tol
    # Near non-opposing surfaces are ambiguous too (thin shells/edges).
    mask = positive | negative
    return positive, mask, nearest, dots, extent


def prepare(args):
    import h5py
    sys.path.insert(0, args.repository)
    from assembly.data.inference import MeshInferenceDataset
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source, 'r') as source:
        inventory = []
        for name, group in source['skull'].items():
            inventory.append(dict(name=name, specimen=specimen_id(name), parts=len(group['pieces']),
                                  labeled_parts=sum('shared_faces' in v for v in group['pieces'].values())))
        eligible = [r['name'] for r in inventory if r['parts'] >= 6 and 'Mandible' not in r['name']]
        split = grouped_split(eligible)
        manifest = dict(source=args.source, source_size=Path(args.source).stat().st_size,
                        split_method='specimen hash order jing-skull-v1; 70/15/15; no score selection',
                        inventory=inventory, split=split,
                        label_type='geometric pseudo-label; positive d<=0.0005*object_extent AND normal_dot<=-0.8; negative d>0.002*extent; otherwise ignored',
                        normalization='official MeshInferenceDataset; 5000 area-weighted points, min20, seed42',
                        physical_scale='experimental uniform 200 mm assembled maximum extent; not source physical units',
                        objects=[])
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        for name in sorted(eligible):
            cached = out/(name+'.npz')
            report_path = out/(name+'.json')
            if report_path.exists():
                report = json.loads(report_path.read_text())
            else:
                names, meshes = load_meshes(source['skull'][name])
                quality = [dict(watertight=bool(m.is_watertight), winding=bool(m.is_winding_consistent),
                                volume=float(m.volume), faces=len(m.faces)) for m in meshes]
                report = dict(name=name, split=split[name], parts=names, mesh_quality=quality)
                if not all(q['watertight'] and q['winding'] and q['volume'] > 0 for q in quality):
                    report.update(status='excluded', reason='mesh closure/winding/volume failed; no silent repair')
                else:
                    with tempfile.TemporaryDirectory() as folder:
                        paths=[]
                        for i, m in enumerate(meshes):
                            path=Path(folder)/f'{i}.ply'; m.export(path); paths.append(str(path))
                        ds=MeshInferenceDataset(name=name, meshes_paths=paths, pad_to_parts=len(meshes),
                            num_points_to_sample=5000, mesh_type='ply', sample_method='weighted',
                            min_points_per_part=20, seed=42, sample_strategy='uniform')
                        original=trimesh.sample.sample_surface
                        rng=np.random.default_rng(42)
                        def sample(*a, **kw):
                            kw.setdefault('seed',rng)
                            return original(*a,**kw)
                        trimesh.sample.sample_surface=sample
                        try: data=ds[0]
                        finally: trimesh.sample.sample_surface=original
                    positive, mask, distances, dots, extent=contact_labels(ds.meshes,data)
                    # An independent check: distance to the provided, decimated full surface.
                    full=source['skull'][name]['full_mesh']
                    full_mesh=trimesh.Trimesh(full['vertices'][:]/ds.scale_max, full['faces'][:], process=True)
                    _, full_distance, _=trimesh.proximity.closest_point(full_mesh,data['pointclouds_gt'])
                    max_roundtrip=0.; max_normal=0.; offset=0
                    for i,count in enumerate(data['points_per_part']):
                        sl=slice(offset,offset+count)
                        r=Rotation.from_quat(data['quaternions'][i][[1,2,3,0]]).as_matrix()
                        reconstructed=(data['pointclouds'][sl]*data['scale'][i])@r.T+data['translations'][i]
                        max_roundtrip=max(max_roundtrip,float(np.max(np.abs(reconstructed-data['pointclouds_gt'][sl]))))
                        max_normal=max(max_normal,float(np.max(np.abs(data['pointclouds_normals'][sl]@r.T-data['pointclouds_normals_gt'][sl]))))
                        offset+=count
                    report.update(status='ready', positive_fraction=float(positive.mean()),
                        supervised_fraction=float(mask.mean()), normal_roundtrip_max=max_normal,
                        point_roundtrip_max=max_roundtrip, positive_normal_dot_median=float(np.median(dots[positive])) if positive.any() else None,
                        full_surface_distance_median_positive=float(np.median(full_distance[positive])) if positive.any() else None,
                        full_surface_distance_median_negative=float(np.median(full_distance[mask&~positive])),
                        extent_model_units=float(extent), model_unit_m=.2/(extent*ds.scale_max))
                    if positive.sum()<20 or max_roundtrip>1e-5 or max_normal>1e-5:
                        report.update(status='excluded', reason='insufficient positives or transform roundtrip failed')
                    else:
                        arrays={k:v for k,v in data.items() if isinstance(v,np.ndarray)}
                        arrays.update(fracture_surface_gt=positive, label_mask=mask, contact_distance=distances,
                                      contact_normal_dot=dots, full_surface_distance=full_distance,
                                      extent=np.array(extent), mesh_scale=np.array(ds.scale_max))
                        np.savez_compressed(cached,**arrays)
                report_path.write_text(json.dumps(report,indent=2))
            manifest['objects'].append(report)
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
            print(name,report['status'],report.get('positive_fraction'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arg in ('source','repository','output'):
        parser.add_argument('--'+arg,required=True)
    prepare(parser.parse_args())
