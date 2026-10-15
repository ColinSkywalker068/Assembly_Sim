"""Oracle cluster ablation, NOT evidence of physical assembly or stability."""
import argparse
import json
from pathlib import Path
import numpy as np
from ..tools.garf import GarfTool, propose_candidates, expand_cluster_pose
from ..tools.garf_backend import GarfBackend
from ..evaluate import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repository','checkpoint','geometry','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(); folder=Path(args.geometry)
    registry=json.loads((folder/'registry.json').read_text())
    geometry={}
    for key,path in registry.items():
        with np.load(path) as v: geometry[key]=(v['points'],v['normals'])
    truth=json.loads((folder/'evaluation_gt.json').read_text())
    anchor=max(geometry,key=lambda k:len(geometry[k][0]))
    # Choose a neighboring part by minimum ground-truth point distance.
    # This uses privileged GT for the experimental condition, never selection in S2.
    from scipy.spatial import cKDTree
    world={k:p@np.asarray(truth[k])[:3,:3].T+np.asarray(truth[k])[:3,3] for k,(p,n) in geometry.items()}
    tree=cKDTree(world[anchor])
    second=min((k for k in geometry if k!=anchor),key=lambda k:tree.query(world[k])[0].min())
    members={k:np.linalg.inv(truth[anchor])@truth[k] for k in (anchor,second)}
    points=[]; normals=[]
    for k,t in members.items():
        p,n=geometry[k]; points.append(p@t[:3,:3].T+t[:3,3]); normals.append(n@t[:3,:3].T)
    fused={k:v for k,v in geometry.items() if k not in members}
    fused['oracle_cluster']=(np.concatenate(points),np.concatenate(normals))
    backend=GarfBackend(args.repository,args.checkpoint)
    tool=GarfTool(backend,fused,backend.checkpoint_sha256)
    candidates=propose_candidates(tool,{'observation_id':'oracle-ablation','fragment_ids':list(fused),
        'anchor_id':'oracle_cluster','seeds':[41,42,43,44]},current_observation_id='oracle-ablation')
    metrics=[]
    for c in candidates['candidates']:
        if c['status']!='candidate': continue
        poses=c['assembly_T_fragment']; t=poses.pop('oracle_cluster')
        poses.update({k:v.tolist() for k,v in expand_cluster_pose(t,members).items()})
        metric=evaluate(c,truth,anchor)
        remaining=[v for k,v in metric['non_anchor_errors'].items() if k not in members]
        metric['remaining_parts_mean_translation_m']=float(np.mean([v['translation_error_m'] for v in remaining]))
        metric['remaining_parts_mean_rotation_deg']=float(np.mean([v['rotation_error_deg'] for v in remaining]))
        metrics.append(metric)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'oracle_cluster':list(members),'physical_stability_verified':False,
        'internal_surfaces_removed':False,'candidates':candidates['candidates'],'metrics':metrics},indent=2))

if __name__=='__main__': main()
