"""Independently check reference overlap using oriented solid-angle winding.

Unlike ray casting this sums triangle solid angles; disagreement exposes a
containment implementation issue, agreement implicates source solid geometry.
Neither watertightness nor positive total volume proves absence of self-intersection.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import trimesh
from .skull_dataset import load_meshes
from ..tools.candidate_validation import SolidContainment


def winding_number(mesh, points, chunk=16384):
    result=np.zeros(len(points))
    for start in range(0,len(mesh.faces),chunk):
        triangles=mesh.triangles[start:start+chunk]
        for i,p in enumerate(points):
            v=triangles-p
            lengths=np.linalg.norm(v,axis=2)
            a,b,c=v[:,0],v[:,1],v[:,2]
            numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
            denominator=(lengths.prod(1)+np.einsum('ij,ij->i',a,b)*lengths[:,2]+
                         np.einsum('ij,ij->i',b,c)*lengths[:,0]+
                         np.einsum('ij,ij->i',c,a)*lengths[:,1])
            result[i]+=np.sum(2*np.arctan2(numerator,denominator))/(4*np.pi)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True); parser.add_argument('--screening',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(); result=[]
    controls=[r for r in json.loads(Path(args.screening).read_text()) if r['stage']=='reference_control']
    with h5py.File(args.source,'r') as source:
        for control in controls:
            _,meshes=load_meshes(source['skull'][control['object']])
            extent=np.ptp(np.concatenate([m.vertices for m in meshes]),axis=0).max()
            for m in meshes: m.apply_scale(.2/extent)
            worst=max(control['pairs'],key=lambda r:r['penetration_fraction'])
            for first,second in [worst['pair'],worst['pair'][::-1]]:
                i,j=int(first),int(second)
                p,_=trimesh.sample.sample_surface(meshes[i],256,seed=i)
                _,dist,_=trimesh.proximity.closest_point(meshes[j],p)
                inside=meshes[j].contains(p)
                selected=np.flatnonzero(inside&(dist>.001))[:16]
                if not len(selected): continue
                winding=winding_number(meshes[j],p[selected])
                normalized=SolidContainment(meshes[j]).contains(p[selected])
                record=dict(object=control['object'],sample_part=i,container_part=j,
                    ray_inside_fraction=float(inside.mean()),deep_inside_count=int((inside&(dist>.001)).sum()),
                    checked_samples=len(selected),minimum_depth_mm=float(dist[selected].min()*1000),
                    winding_numbers=winding.tolist(),
                    winding_agrees_count=int((np.abs(winding)>.5).sum()),
                    normalized_ray_inside_count=int(normalized.sum()),
                    normalized_ray_winding_agreement=float((normalized==(np.abs(winding)>.5)).mean()))
                result.append(record); print(record,flush=True)
                np.savez_compressed(Path(args.output).with_name(control['object']+'-overlap.npz'),
                                    points=p,inside=inside,depth=dist,checked=selected)
    Path(args.output).write_text(json.dumps(result,indent=2))


if __name__=='__main__': main()
