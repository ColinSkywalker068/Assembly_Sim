"""Fixed-cohort sampling/loss ablation, independent expanded skull validation.

All controlled arms start from the same new encoder warm-up. Previous joint
weights are a historical comparator, not a compute-matched causal control.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from .skull_dataset import specimen_id,load_meshes
from .skull_finetune import PreparedBackend,read_data,evaluate,train_stage
from ..tools.candidate_validation import score_candidate

ARMS={'flow20':(20,0.),'flow200':(200,0.),'multi200':(200,.2)}


def build(base,out):
    import h5py
    out.mkdir(parents=True,exist_ok=True)
    if (out/'cohort.json').exists():raise ValueError('cohort is immutable; use a new output directory')
    sources=[('skull',base/'skull-adaptation-v1',base/'skull-min200-v2'),
             ('morphosource_v1',base/'morphosource_v1-min20-v2',base/'morphosource_v1-min200-v2'),
             ('morphosource_v2',base/'morphosource_v2-min20-v2',base/'morphosource_v2-min200-v2')]
    prior_controls={r['object']:r for r in json.loads((base/'skull-adaptation-v1/reference_qa.json').read_text())}
    cohort=dict(items=[],excluded=[],test_policy='original specimen test split remains untouched by model training/evaluation',
                physical_scale='experimental 200 mm maximum assembled extent',
                protocol=dict(total_points=5000,minimum_points=[20,200],warmup_steps=100,
                    joint_steps=150,lr=1e-5,seed=20261019,segmentation_weight=.2,
                    inference_steps=20,inference_iterations=6,noise_seeds=[41,42],
                    metric_cloud='same min20 canonical points/labels for every arm; anchor excluded',
                    checkpoint_selection='fixed final step; no outcome-driven validation exclusion'))
    seen_specimens=set()
    for root,low,high in sources:
        lo=json.loads((low/'manifest.json').read_text()); hi=json.loads((high/'manifest.json').read_text())
        for m in [lo,hi]:
            if len(m['objects'])!=len(m['split']):raise ValueError('preprocessing incomplete')
        high_rows={r['name']:r for r in hi['objects']}
        for r in lo['objects']:
            name=r['name']; identifier=root+'__'+name; sid=specimen_id(name)
            if r['split']=='test':continue
            reason=None
            if r['status']!='ready' or high_rows[name]['status']!='ready':reason='mesh/label QA failed in one sampling arm'
            if root=='skull' and name in prior_controls and prior_controls[name]['rejected']:reason='previous reference disconnected; dense diagnostics show real gap'
            if reason:
                cohort['excluded'].append(dict(id=identifier,reason=reason));continue
            if root!='skull' and sid in seen_specimens:
                cohort['excluded'].append(dict(id=identifier,reason='duplicate specimen across source versions'));continue
            source=lo['source']
            with h5py.File(source) as f:_,parts=load_meshes(f[root][name])
            extent=np.ptp(np.concatenate([m.vertices for m in parts]),axis=0).max()
            for m in parts:m.apply_scale(.2/extent)
            meshes={str(i):m for i,m in enumerate(parts)}
            qa=score_candidate(meshes,{k:np.eye(4) for k in meshes},samples=512)
            qa_path=out/(identifier+'-reference.json');qa_path.write_text(json.dumps(qa,indent=2))
            if qa['rejected']:
                cohort['excluded'].append(dict(id=identifier,reason='512-point reference geometry gate failed',qa=qa));continue
            item=dict(id=identifier,name=name,specimen=sid,source=source,root=root,split=r['split'],
                      reference_qa=qa_path.name,parts=len(parts),data_sha256={})
            for minimum,folder in [(20,low),(200,high)]:
                target=out/f'data{minimum}'/(identifier+'.npz');target.parent.mkdir(exist_ok=True)
                shutil.copy2(folder/(name+'.npz'),target)
                item['data_sha256'][str(minimum)]=hashlib.sha256(target.read_bytes()).hexdigest()
            cohort['items'].append(item);seen_specimens.add(sid)
            (out/'cohort-progress.json').write_text(json.dumps(cohort,indent=2))
            print('COHORT',identifier,item['split'],flush=True)
    train={r['specimen'] for r in cohort['items'] if r['split']=='train'}
    val={r['specimen'] for r in cohort['items'] if r['split']=='val'}
    if train&val:raise ValueError('specimen leakage')
    if len(val)<=2:raise ValueError('validation expansion failed')
    (out/'cohort.json').write_text(json.dumps(cohort,indent=2))


def run(a):
    import torch
    out=Path(a.output);root=Path(a.dataset);out.mkdir(parents=True,exist_ok=True)
    cohort=json.loads((root/'cohort.json').read_text())
    train=sorted(r['id'] for r in cohort['items'] if r['split']=='train')
    val=sorted(r['id'] for r in cohort['items'] if r['split']=='val')
    data20={n:read_data(root/'data20'/(n+'.npz')) for n in train+val}
    backend=PreparedBackend(a.repository,a.checkpoint,{},1.)
    if a.arm=='warmup':
        train_stage(backend,data20,train,'encoder',100,out,seed=20261019)
        return
    if a.arm=='previous':
        evaluate(backend,data20,val,'previous',out,[41,42],metric_dataset=data20)
        return
    minimum,weight=ARMS[a.arm]
    data=data20 if minimum==20 else {n:read_data(root/'data200'/(n+'.npz')) for n in train+val}
    # --checkpoint is the identical shared warmup in every controlled arm.
    (out/'protocol.json').write_text(json.dumps(dict(arm=a.arm,minimum=minimum,seg_weight=weight,
        train=train,val=val,start_checkpoint_sha256=backend.checkpoint_sha256,
        cohort_sha256=hashlib.sha256((root/'cohort.json').read_bytes()).hexdigest()),indent=2))
    train_stage(backend,data,train,'joint',150,out,seg_weight=weight,seed=20261019)
    evaluate(backend,data,val,a.arm,out,[41,42],metric_dataset=data20)


def screen(a):
    import h5py
    root=Path(a.dataset);run=Path(a.output);rows=[]
    cohort=json.loads((root/'cohort.json').read_text())
    path=run/(a.arm+'-screening.json')
    if path.exists():rows=json.loads(path.read_text())
    done={(r['object'],r['seed']) for r in rows}
    for item in cohort['items']:
        if item['split']!='val':continue
        with h5py.File(item['source']) as f:_,parts=load_meshes(f[item['root']][item['name']])
        extent=np.ptp(np.concatenate([m.vertices for m in parts]),axis=0).max()
        for m in parts:m.apply_scale(.2/extent)
        meshes={str(i):m for i,m in enumerate(parts)}
        for seed in [41,42]:
            if (item['id'],seed) in done:continue
            artifact=run/f'{a.arm}--{item["id"]}--{seed}.npz'
            if not artifact.exists():continue
            with np.load(artifact) as d:poses=d['poses']
            result=score_candidate(meshes,{k:poses[int(k)] for k in meshes},samples=512)
            rows.append(dict(object=item['id'],seed=seed,arm=a.arm,**result))
            path.write_text(json.dumps(rows,indent=2));print('SCREEN',item['id'],seed,result['rejected'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','run','screen'])
    p.add_argument('--base');p.add_argument('--dataset');p.add_argument('--repository');p.add_argument('--checkpoint')
    p.add_argument('--output',required=True);p.add_argument('--arm',choices=['warmup','previous',*ARMS])
    a=p.parse_args()
    if a.mode=='build':build(Path(a.base),Path(a.output))
    elif a.mode=='run':run(a)
    else:screen(a)
