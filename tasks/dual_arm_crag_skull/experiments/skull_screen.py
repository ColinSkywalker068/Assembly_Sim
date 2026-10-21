"""Screen all pilot validation candidates; GT is a separate diagnostic control."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from .skull_dataset import load_meshes
from ..tools.candidate_validation import score_candidate


def main():
    parser=argparse.ArgumentParser()
    for key in ('source','run','output'): parser.add_argument('--'+key,required=True)
    parser.add_argument('--stages',nargs='+',choices=['baseline','encoder','joint'],default=['baseline','encoder','joint'])
    parser.add_argument('--reference-only',action='store_true')
    parser.add_argument('--dataset',help='For reference-only mode: audit all QA-ready objects, without model evaluation')
    args=parser.parse_args(); run=Path(args.run); out=Path(args.output)
    rows=[]
    with h5py.File(args.source,'r') as source:
        names=json.loads((run/'protocol.json').read_text())['validation']
        if args.dataset:
            if not args.reference_only: raise ValueError('--dataset is only for reference-only mode')
            manifest=json.loads((Path(args.dataset)/'manifest.json').read_text())
            names=[r['name'] for r in manifest['objects'] if r['status']=='ready']
        for name in names:
            _,parts=load_meshes(source['skull'][name])
            extent=np.ptp(np.concatenate([m.vertices for m in parts]),axis=0).max()
            for mesh in parts: mesh.apply_scale(.2/extent)
            meshes={str(i):m for i,m in enumerate(parts)}
            control=score_candidate(meshes,{k:np.eye(4) for k in meshes},samples=256)
            rows.append(dict(object=name,stage='reference_control',seed=None,**control))
            out.write_text(json.dumps(rows,indent=2))
            print('SCREEN',name,'reference',control['rejected'],flush=True)
            if args.reference_only: continue
            for stage in args.stages:
                for seed in (41,42):
                    with np.load(run/f'{stage}--{name}--{seed}.npz') as d: poses=d['poses']
                    result=score_candidate(meshes,{k:poses[int(k)] for k in meshes},samples=256)
                    rows.append(dict(object=name,stage=stage,seed=seed,**result))
                    out.write_text(json.dumps(rows,indent=2))
                    print('SCREEN',name,stage,seed,result['rejected'],flush=True)


if __name__=='__main__': main()
