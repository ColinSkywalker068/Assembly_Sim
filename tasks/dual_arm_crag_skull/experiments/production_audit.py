"""Production official preprocessing, noise candidates and fracture-contact audit.
GT is read only after inference, for offline diagnostics; never passed to backend.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from ..export_points import export
from ..tools.official_backend import OfficialGarfBackend
from ..tools.garf import GarfTool, propose_candidates
from ..tools.candidate_validation import screen_result
from ..evaluate import evaluate


def main():
    parser=argparse.ArgumentParser()
    for name in ('repository','checkpoint','source','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--skip-screening',action='store_true',help='Diagnostic control only; explicitly report screening omitted')
    args=parser.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if Path(args.source).is_dir():
        scene=trimesh.Scene()
        for path in sorted(Path(args.source).glob('piece_*.obj')):
            scene.add_geometry(trimesh.load_mesh(path),node_name=path.stem)
        source_path=out/'control.glb'; scene.export(source_path); args.source=str(source_path)
    export(args.source,out/'geometry')
    root=out/'geometry'
    meshes={k:trimesh.load_mesh(v) for k,v in json.loads((root/'mesh_registry.json').read_text()).items()}
    geometry={}
    for k,v in json.loads((root/'registry.json').read_text()).items():
        with np.load(v,allow_pickle=False) as f: geometry[k]=(f['points'],f['normals'])
    source=trimesh.load(args.source,force='scene')
    model_unit_m=.2/source.extents.max()  # explicit experimental source-to-metre scale
    backend=OfficialGarfBackend(args.repository,args.checkpoint,meshes,model_unit_m)
    tool=GarfTool(backend,geometry,backend.checkpoint_sha256)
    ids=list(meshes); anchor=max(ids,key=lambda k:meshes[k].area)
    result=propose_candidates(tool,{'observation_id':'scan-official','fragment_ids':ids,
        'anchor_id':anchor,'seeds':[41,42,43,44]},current_observation_id='scan-official')
    (out/'candidates.json').write_text(json.dumps(result,indent=2))
    print('inference complete',flush=True)
    truth=json.loads((root/'evaluation_gt.json').read_text())
    metrics=[evaluate(c,truth,anchor) for c in result['candidates'] if c['status']=='candidate']
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print([(v['seed'],v['mean_translation_error_m'],v['mean_rotation_error_deg']) for v in metrics],flush=True)
    data=backend.last_diagnostics['data']; prob=backend.last_diagnostics['fracture_probability']
    common=float(data['mesh_scale'])*model_unit_m
    worlds={}
    for key,m in meshes.items():
        worlds[key]=m.copy(); worlds[key].apply_transform(truth[key])
    offset=0; contact=[]; summaries=[]
    for key,count in zip(ids,data['points_per_part']):
        t=np.asarray(truth[key]); p=data['pointclouds_gt'][offset:offset+count]*common
        p=p@t[:3,:3].T+t[:3,3]
        normals=data['pointclouds_normals_gt'][offset:offset+count]@t[:3,:3].T
        distances=np.full(count,np.inf)
        for other,m in worlds.items():
            if key==other: continue
            near=(p>=m.bounds[0]-.001).all(1)&(p<=m.bounds[1]+.001).all(1)
            if not near.any(): continue
            _,d,face=trimesh.proximity.closest_point(m,p[near])
            opposing=np.einsum('ij,ij->i',normals[near],m.face_normals[face])<-.5
            values=np.where(opposing,d,np.inf)
            distances[near]=np.minimum(distances[near],values)
        prediction=prob[offset:offset+count]>.5
        labels=distances<.001; contact.extend(labels.tolist())
        tp=int((prediction&labels).sum()); fp=int((prediction&~labels).sum()); fn=int((~prediction&labels).sum())
        summaries.append({'id':key,'points':int(count),'predicted_fracture_fraction':float(prediction.mean()),
            'contact_proxy_fraction':float(labels.mean()),'tp':tp,'fp':fp,'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None})
        offset+=count
        print('fracture',summaries[-1],flush=True)
    report={'label_type':'GT geometric contact proxy, not annotated fracture GT','distance_m':.001,
        'opposing_normal_dot_max':-.5,'fracture_mask_used_by_pose_model':False,'parts':summaries,
        'counts':data['points_per_part'].tolist(),'model_unit_m':model_unit_m}
    (out/'fracture.json').write_text(json.dumps(report,indent=2))
    np.savez(out/'fracture_diagnostics.npz',probability=prob,contact_proxy=np.asarray(contact))
    # Complete all pairwise checks; no ground truth enters ranking.
    if args.skip_screening:
        (out/'screening_omitted.json').write_text(json.dumps({'reason':'explicit diagnostic control flag'}))
        return
    screened=screen_result(meshes,result,samples=256)
    (out/'screened.json').write_text(json.dumps(screened,indent=2))
    print('screening',screened['geometry_status'],flush=True)

if __name__=='__main__': main()
