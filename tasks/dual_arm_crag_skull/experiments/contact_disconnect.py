"""Offline dense diagnostics for isolated reference fragments; never relax gates."""
import argparse,json
from pathlib import Path
import h5py,numpy as np,trimesh
from .skull_dataset import load_meshes


def main():
    p=argparse.ArgumentParser()
    for k in ('source','qa','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args(); rows=[]
    with h5py.File(a.source) as f:
        for item in json.loads(Path(a.qa).read_text()):
            if not item['rejected']:continue
            name=item['object']; names,meshes=load_meshes(f['skull'][name])
            extent=np.ptp(np.concatenate([m.vertices for m in meshes]),axis=0).max()
            for m in meshes:m.apply_scale(.2/extent)
            degree=np.zeros(len(meshes),int)
            for pair in item['pairs']:
                if pair['contact_fraction']>.01:
                    for index in pair['pair']:degree[int(index)]+=1
            for i in np.flatnonzero(degree==0):
                points,faces=trimesh.sample.sample_surface(meshes[i],4096,seed=20261019)
                normal=meshes[i].face_normals[faces]; pairrows=[]
                for j,m in enumerate(meshes):
                    if i==j:continue
                    _,d,face=trimesh.proximity.closest_point(m,points)
                    dot=np.einsum('ij,ij->i',normal,m.face_normals[face])
                    pairrows.append(dict(other=j,min_distance_mm=float(d.min()*1000),
                        distance_quantiles_mm=(np.quantile(d,[.01,.1,.5,.9])*1000).tolist(),
                        close_fraction=float((d<.003).mean()),
                        opposing_contact_fraction=float(((d<.003)&(dot<-.5)).mean()),
                        strict_contact_fraction=float(((d<.0001)&(dot<-.8)).mean())))
                best=min(pairrows,key=lambda r:r['min_distance_mm'])
                result=dict(object=name,isolated_part=int(i),part_name=names[i],samples=4096,
                            part_extents_mm=(meshes[i].extents*1000).tolist(),pairs=pairrows,
                            dense_rule_connects=any(r['opposing_contact_fraction']>.01 for r in pairrows),
                            minimum_surface_gap_mm=best['min_distance_mm'])
                rows.append(result);Path(a.output).write_text(json.dumps(rows,indent=2))
                print(result,flush=True)

if __name__=='__main__':main()
