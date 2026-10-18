"""Small, explicitly bounded adaptation pilot; no claim of converged training.

Baseline -> encoder Dice supervision -> encoder+denoiser official flow loss.
Validation is fixed, held-out by specimen. Test set remains unopened.
All poses are evaluated in the known-anchor gauge, excluding the anchor.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..tools.official_backend import OfficialGarfBackend
from .fracture_thresholds import ranking_metrics
from .skull_dataset import specimen_id


class PreparedBackend(OfficialGarfBackend):
    def prepare(self, ids):
        return self.prepared_data


def read_data(path):
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def augment(data, rng):
    from assembly.data.transform import recenter_pc, rotate_pc
    d={k:v.copy() for k,v in data.items()}
    offset=0
    for i,n in enumerate(d['points_per_part']):
        sl=slice(offset,offset+n)
        p,t=recenter_pc(d['pointclouds_gt'][sl])
        p,norm,q=rotate_pc(p,d['pointclouds_normals_gt'][sl],rng)
        scale=np.abs(p).max()
        d['pointclouds'][sl]=p/scale
        d['pointclouds_normals'][sl]=norm
        d['quaternions'][i]=q; d['translations'][i]=t; d['scale'][i]=scale
        offset+=n
    return d


def tensor_batch(d, device):
    import torch
    keys=('pointclouds','pointclouds_normals','points_per_part','graph','translations',
          'quaternions','scale','ref_part','fracture_surface_gt','label_mask')
    return {k:torch.as_tensor(np.ascontiguousarray(d[k]),device=device).unsqueeze(0) for k in keys}


def segmentation_loss(output,batch):
    mask=batch['label_mask'].reshape(-1)
    p=output['coarse_seg_pred'].float().reshape(-1)[mask]
    y=batch['fracture_surface_gt'].float().reshape(-1)[mask]
    # Official segmentation objective (Dice), with an explicit uncertainty mask.
    return 1-(2*(p*y).sum()+1e-6)/(p.sum()+y.sum()+1e-6)


def parameter_digest(module):
    h=hashlib.sha256()
    for p in module.parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(backend, dataset, names, stage, out, seeds, metric_dataset=None):
    import torch
    backend.model.eval()
    rows=[]
    for name in names:
        d=dataset[name]; backend.prepared_data=d; backend._feature_cache=None
        count=len(d['points_per_part']); ids=list(map(str,range(count)))
        anchor=int(np.argmax(d['points_per_part']))
        backend.model_unit_m=.2/(float(d['extent'])*float(d['mesh_scale']))
        reference=d if metric_dataset is None else metric_dataset[name]
        point_scale=.2/float(reference['extent'])
        canonical_probability=None
        for seed in seeds:
            poses=backend.predict_registered(ids,anchor,seed)
            errors=[]; clouds=[]; start=0
            # Registered meshes for this offline benchmark use assembled axes.
            # Model sees only independently centered/rotated clouds; one anchor
            # is clamped exactly as in official inference, all other GT is evaluation only.
            for i,(pose,n) in enumerate(zip(poses,reference['points_per_part'])):
                gt=reference['pointclouds_gt'][start:start+n]*point_scale
                pred=gt@pose[:3,:3].T+pose[:3,3]
                clouds.append(pred)
                if i!=anchor:
                    errors.append(dict(part=i,centroid_error_mm=float(np.linalg.norm(pred.mean(0)-gt.mean(0))*1000),
                        rotation_error_deg=float(Rotation.from_matrix(pose[:3,:3]).magnitude()*180/np.pi),
                        point_rmse_mm=float(np.sqrt(np.mean(np.sum((pred-gt)**2,axis=1)))*1000)))
                start+=n
            probability=backend.last_diagnostics['fracture_probability']
            if metric_dataset is not None:
                if canonical_probability is None:
                    device_index=torch.device(backend.device).index or 0
                    with torch.inference_mode(),torch.random.fork_rng(devices=[device_index]),torch.autocast(device_type='cuda',dtype=torch.float16):
                        torch.manual_seed(42)
                        output=backend.model.feature_extractor(tensor_batch(reference,backend.device))
                        canonical_probability=output['coarse_seg_pred'].float().cpu().numpy().reshape(-1)
                probability=canonical_probability
            mask=reference['label_mask']; y=reference['fracture_surface_gt'][mask]; p=probability[mask]
            tp=int(((p>.5)&y).sum()); fp=int(((p>.5)&~y).sum()); fn=int(((p<=.5)&y).sum())
            row=dict(object=name,stage=stage,seed=seed,anchor=anchor,parts=errors,
                     centroid_error_mm=float(np.mean([x['centroid_error_mm'] for x in errors])),
                     rotation_error_deg=float(np.mean([x['rotation_error_deg'] for x in errors])),
                     point_rmse_mm=float(np.mean([x['point_rmse_mm'] for x in errors])),
                     fracture_f1=2*tp/max(1,2*tp+fp+fn),fracture_ranking=ranking_metrics(y,p),
                     input_counts=d['points_per_part'].tolist(),metric_counts=reference['points_per_part'].tolist())
            rows.append(row)
            np.savez_compressed(out/f'{stage}--{name}--{seed}.npz',predicted=np.concatenate(clouds),
                truth=reference['pointclouds_gt']*point_scale,probability=probability,
                labels=reference['fracture_surface_gt'],mask=mask,counts=reference['points_per_part'],
                normals=reference['pointclouds_normals_gt'],poses=np.array(poses))
            print('EVAL',stage,name,seed,row['centroid_error_mm'],row['rotation_error_deg'],flush=True)
        (out/f'{stage}-metrics.json').write_text(json.dumps(rows,indent=2))
    return rows


def train_stage(backend,dataset,names,stage,steps,out,seg_weight=0.,seed=20261018):
    import torch
    model=backend.model
    # Keep small-batch BN statistics frozen, but enable autograd explicitly:
    # upstream DenoiserBase freezes the entire feature extractor in __init__.
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.feature_extractor.parameters(): p.requires_grad_(True)
    if stage=='joint':
        for p in model.denoiser.parameters(): p.requires_grad_(True)
    groups=[dict(params=list(model.feature_extractor.parameters()),lr=1e-5)]
    if stage=='joint': groups.append(dict(params=list(model.denoiser.parameters()),lr=1e-5))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda')
    rng=np.random.default_rng(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    before=dict(encoder=parameter_digest(model.feature_extractor.encoder),denoiser=parameter_digest(model.denoiser))
    history=[]; order=[]; start_time=time.time()
    for step in range(steps):
        if not order: order=list(rng.permutation(names))
        name=order.pop(); data=augment(dataset[name],rng); batch=tensor_batch(data,backend.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            if stage=='encoder':
                loss=segmentation_loss(model.feature_extractor(batch),batch)
                segmentation=loss; pose_loss=None
            else:
                captured={}
                hook=model.feature_extractor.register_forward_hook(lambda module,inputs,output:captured.update(output=output)) if seg_weight else None
                try: prediction=model(batch)
                finally:
                    if hook is not None:hook.remove()
                pose_loss=model._loss(batch,prediction)[0]['vec_mse_loss']
                segmentation=segmentation_loss(captured['output'],batch) if seg_weight else None
                loss=pose_loss+seg_weight*segmentation if seg_weight else pose_loss
        if not torch.isfinite(loss): raise RuntimeError(f'nonfinite {stage} loss at {step}')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        params=[p for p in model.parameters() if p.requires_grad]
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=False)
        old_scale=scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        row=dict(step=step,object=name,loss=float(loss.detach()),
                 flow_loss=float(pose_loss.detach()) if pose_loss is not None else None,
                 dice_loss=float(segmentation.detach()) if segmentation is not None else None,
                 gradient_norm=float(norm) if torch.isfinite(norm) else None,
                 optimizer_step_skipped=scaler.get_scale()<old_scale,seconds=time.time()-start_time)
        history.append(row)
        if step%10==0: print('TRAIN',stage,row,flush=True)
    after=dict(encoder=parameter_digest(model.feature_extractor.encoder),denoiser=parameter_digest(model.denoiser))
    (out/f'{stage}-training.json').write_text(json.dumps(dict(steps=steps,
        optimizer_updates=sum(not r['optimizer_step_skipped'] for r in history),learning_rate=1e-5,
        batch_size=1,segmentation_weight=seg_weight,seed=seed,bn_mode='eval frozen running statistics',before=before,after=after,history=history),indent=2))
    torch.save({'state_dict':model.state_dict()},out/f'{stage}.ckpt')
    backend._feature_cache=None


def main():
    parser=argparse.ArgumentParser()
    for key in ('repository','checkpoint','dataset','output'): parser.add_argument('--'+key,required=True)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--reference-qa',help='Optional stricter next-run gate: exclude failed reference contact/penetration controls')
    args=parser.parse_args(); root=Path(args.dataset); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text())
    if len(manifest['objects'])!=len(manifest['split']): raise ValueError('dataset audit still incomplete')
    ready=[r for r in manifest['objects'] if r['status']=='ready']
    reference_gate=None
    if args.reference_qa:
        reference_path=Path(args.reference_qa)
        controls=json.loads(reference_path.read_text())
        by_name={r['object']:r for r in controls if r['stage']=='reference_control'}
        if any(r['name'] not in by_name for r in ready):
            raise ValueError('reference QA is incomplete; cannot silently drop unaudited objects')
        reference_gate=dict(sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                            excluded=sorted(r['name'] for r in ready if by_name[r['name']]['rejected']))
        ready=[r for r in ready if not by_name[r['name']]['rejected']]
    train=sorted(r['name'] for r in ready if r['split']=='train')
    val=sorted(r['name'] for r in ready if r['split']=='val')
    if not train or not val: raise ValueError('empty train or validation split after QA')
    if {specimen_id(n) for n in train}&{specimen_id(n) for n in val}:
        raise ValueError('specimen leakage across training and validation')
    dataset={n:read_data(root/(n+'.npz')) for n in train+val}
    protocol=dict(train=train,validation=val,test='QA only; not used in training or model evaluation',stages=['baseline','encoder','joint'],
        reference_geometry_gate=reference_gate,
        steps_per_stage=args.steps,seeds=[41,42],inference_steps=20,inference_iterations=6,
        dataset_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        data_sha256={n:hashlib.sha256((root/(n+'.npz')).read_bytes()).hexdigest() for n in train+val},
        stage_objectives={'encoder':'masked Dice with geometry pseudo-labels','joint':'official SE3 flow-matching vec_mse'},
        selection='fixed final step, no validation checkpoint selection; no hyperparameter search',
        limitation='small single-seed sequential pilot, not a convergence or generalization claim')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    backend=PreparedBackend(args.repository,args.checkpoint,{},1.)
    evaluate(backend,dataset,val,'baseline',out,[41,42])
    for stage in ('encoder','joint'):
        train_stage(backend,dataset,train,stage,args.steps,out)
        evaluate(backend,dataset,val,stage,out,[41,42])
    print('COMPLETE',flush=True)


if __name__=='__main__': main()
