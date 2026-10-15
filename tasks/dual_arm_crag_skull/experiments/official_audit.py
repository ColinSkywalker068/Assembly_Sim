"""Use official MeshInferenceDataset and compare scheduler/normal ablations.
GT is used only for the reference anchor and offline metrics, matching upstream.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from ..preparation import load_skull
from ..tools.garf_backend import GarfBackend


def main():
    parser=argparse.ArgumentParser()
    for name in ('repository','checkpoint','source','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--attention', choices=['sdpa','flash'], default='sdpa')
    parser.add_argument('--points', type=int, default=5000)
    parser.add_argument('--minimum-points', type=int, default=20)
    parser.add_argument('--normal-sign', type=int, choices=[-1,1], default=1)
    args=parser.parse_args()
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    backend=GarfBackend(args.repository,args.checkpoint)
    from assembly.data.inference import MeshInferenceDataset
    from assembly.models.denoiser.modules.scheduler import SE3FlowMatchEulerDiscreteScheduler
    paths=[]; meshes=[]
    source=Path(args.source)
    if source.is_dir():
        paths=[str(p) for p in sorted(source.glob('piece_*.obj'))]
        meshes=[trimesh.load_mesh(p) for p in paths]
        mesh_type='obj'
    else:
        assembly=load_skull(source)
        for f in assembly.fragments:
            m=f.mesh.copy(); m.apply_transform(f.ground_truth_transform); meshes.append(m)
            path=output/(f.name+'.ply'); m.export(path); paths.append(str(path))
        mesh_type='ply'
    if len(paths)<2: raise ValueError('need at least two meshes')
    original_extent=trimesh.util.concatenate(meshes).extents.max()
    np.random.seed(42)
    ds=MeshInferenceDataset(name='skull',meshes_paths=paths,pad_to_parts=len(paths),num_points_to_sample=args.points,min_points_per_part=args.minimum_points,
                            mesh_type=mesh_type,sample_method='weighted',seed=42,sample_strategy='uniform')
    # New trimesh uses default_rng(None), which np.random.seed does not seed.
    # Preserve official uniform sampling while fixing its random stream.
    original_sample=trimesh.sample.sample_surface
    sampling_rng=np.random.default_rng(42)
    def seeded_sample(*a, **kw):
        kw.setdefault('seed', sampling_rng)
        return original_sample(*a, **kw)
    trimesh.sample.sample_surface=seeded_sample
    try: data=ds[0]
    finally: trimesh.sample.sample_surface=original_sample
    batch={k:torch.as_tensor(np.ascontiguousarray(v)).unsqueeze(0).cuda().contiguous() for k,v in data.items() if isinstance(v,np.ndarray)}
    valid=batch['points_per_part']!=0; ref=batch['ref_part'][valid]
    truth=torch.cat([batch['translations'][valid],batch['quaternions'][valid]],dim=-1)
    gt=truth.cpu().numpy(); metric_factor=float(data['mesh_scale'])*.2/original_extent
    # Recover every original sampled point using official GT transforms.
    # This checks scale, quaternion order and row/column conventions without inference.
    offset=0; roundtrip=[]
    for i,count in enumerate(data['points_per_part']):
        p=data['pointclouds'][offset:offset+count]*data['scale'][i]
        r=Rotation.from_quat(data['quaternions'][i][[1,2,3,0]]).as_matrix()
        recovered=p@r.T+data['translations'][i]
        target=data['pointclouds_gt'][offset:offset+count]
        roundtrip.append(float(np.max(np.linalg.norm(recovered-target,axis=1))))
        offset+=count
    results=[]; latent_cache={}
    # Identical official input and initialization across each ablation.
    for schedule,one_step,normal_sign in [('exponential',False,args.normal_sign),('linear',True,args.normal_sign)]:
        backend.model.denoiser.use_flash_attn = args.attention=='flash'
        rng=np.random.default_rng(42); generator=torch.Generator('cuda').manual_seed(42)
        x=torch.empty_like(truth).normal_(generator=generator)
        x[:,3:]=torch.as_tensor(Rotation.random(len(x),random_state=rng).as_quat()[:,[3,0,1,2]],device='cuda',dtype=x.dtype)
        x[ref]=truth[ref].to(x.dtype)
        work=dict(batch); work['pointclouds_normals']=batch['pointclouds_normals']*normal_sign
        scheduler=SE3FlowMatchEulerDiscreteScheduler(sigma_schedule=schedule,stochastic_paths=False,stochastic_level=.05,min_stochastic_epsilon=.01)
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
            if normal_sign not in latent_cache:
                torch.manual_seed(42)  # eval-mode PointTransformer still shuffles orders
                latent_cache[normal_sign]=backend.model._extract_features(work)
            latent=latent_cache[normal_sign]
            schedules=([1] if one_step else [])+[20]*6
            for steps in schedules:
                scheduler.set_timesteps(steps)
                for t in scheduler.timesteps:
                    pred=backend.model.denoiser(x=x,timesteps=t.reshape(-1).repeat(len(x)).cuda(),latent=latent,
                                                part_valids=valid,scale=batch['scale'][valid],ref_part=ref)['pred']
                    x=scheduler.step(pred,t,x).prev_sample; x[ref]=truth[ref].to(x.dtype)
        values=x.float().cpu().numpy(); mask=~ref.cpu().numpy()
        rotation_error=np.rad2deg((Rotation.from_quat(values[:,3:][:,[1,2,3,0]])*Rotation.from_quat(gt[:,3:][:,[1,2,3,0]]).inv()).magnitude())
        error=np.linalg.norm(values[:,:3]-gt[:,:3],axis=1)*metric_factor
        result={'schedule':schedule,'one_step_init':one_step,'normal_sign':normal_sign,'attention_backend':args.attention,
                'mean_translation_mm':float(error[mask].mean()*1000),'mean_rotation_deg':float(rotation_error[mask].mean()),
                'per_part_translation_mm':(error*1000).tolist(),'per_part_rotation_deg':rotation_error.tolist()}
        results.append(result); print(json.dumps(result),flush=True)
    (output/'audit.json').write_text(json.dumps({'mesh_scale':float(data['mesh_scale']),'metric_factor':metric_factor,
        'sampling_seed':42,'encoder_seed':42,'features_shared_between_ablations':True,'roundtrip_max_error_m':max(roundtrip)*metric_factor,'counts':data['points_per_part'].tolist(),'scales':data['scale'].tolist(),'results':results},indent=2))

if __name__=='__main__': main()
