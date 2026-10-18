"""Render measured QA, predictions and learning curves; no illustrative geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..perception import anchor_goals


def cloud(ax,points,colors,title,limits=None):
    ax.scatter(*points.T,c=colors,s=2,alpha=.85,rasterized=True)
    ax.set_title(title,fontsize=11)
    if limits is None:
        center=(points.min(0)+points.max(0))/2
        radius=np.ptp(points,axis=0).max()*.55
        limits=[(v-radius,v+radius) for v in center]
    ax.set(xlim=limits[0],ylim=limits[1],zlim=limits[2],xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)')
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=25,azim=-65)
    return limits


def transfer_comparison(baseline,adapted,output):
    """Historical Hoolock transfer example, not a new untouched test set."""
    panels=[]
    for root,label in [(baseline,'Original GARF'),(adapted,'Joint fine-tune')]:
        truth=json.loads((root/'geometry/evaluation_gt.json').read_text())
        registry=json.loads((root/'geometry/registry.json').read_text())
        metric=next(r for r in json.loads((root/'metrics.json').read_text()) if r['seed']==41)
        candidate=next(r for r in json.loads((root/'candidates.json').read_text())['candidates'] if r['seed']==41)
        poses=anchor_goals(candidate['assembly_T_fragment'],metric['anchor'],truth[metric['anchor']])
        gt=[]; pred=[]; color=[]
        for i,(key,path) in enumerate(registry.items()):
            with np.load(path) as z: p=z['points']
            t=np.array(truth[key]); pose=np.array(poses[key])
            gt.append((p@t[:3,:3].T+t[:3,3])*1000)
            pred.append((p@pose[:3,:3].T+pose[:3,3])*1000)
            color.extend([plt.get_cmap('tab10')(i)]*len(p))
        if not panels: panels.append((np.concatenate(gt),np.array(color),'Reference'))
        panels.append((np.concatenate(pred),np.array(color),f'{label}: {metric["mean_translation_error_m"]*1000:.1f} mm'))
    joined=np.concatenate([p[0] for p in panels]); center=(joined.min(0)+joined.max(0))/2
    radius=np.ptp(joined,axis=0).max()*.53; limits=[(v-radius,v+radius) for v in center]
    fig=plt.figure(figsize=(15,5.5))
    for i,(p,c,title) in enumerate(panels): cloud(fig.add_subplot(1,3,i+1,projection='3d'),p,c,title,limits)
    fig.suptitle('Original 8-part Hoolock | historical transfer check | noise seed 41 | fixed largest-part anchor')
    fig.text(.5,.02,'These are predicted target poses, NOT a successful physical assembly. Both candidates fail geometry screening.',ha='center')
    fig.tight_layout(rect=[0,.04,1,.94]); fig.savefig(output/'hoolock_transfer.png',dpi=170); plt.close(fig)


def render(dataset,output,run=None):
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((dataset/'manifest.json').read_text())
    ready=[r for r in manifest['objects'] if r['status']=='ready' and r['split']!='test']
    # Deterministic first eligible validation object, otherwise training example.
    examples=sorted(ready,key=lambda r:(r['split']!='val',r['name']))
    if not examples: raise ValueError('No QA-passing sample')
    name=examples[0]['name']
    with np.load(dataset/(name+'.npz')) as z: d=dict(z)
    p=d['pointclouds_gt']*200/float(d['extent'])
    part=np.repeat(np.arange(len(d['points_per_part'])),d['points_per_part'])
    colors=plt.get_cmap('tab10')(part%10)
    exploded=p.copy()
    center=p.mean(0)
    for i in np.unique(part):
        mask=part==i
        exploded[mask]+=.75*(p[mask].mean(0)-center)
    label_colors=np.where(d['fracture_surface_gt'][:,None],np.array([.9,.15,.1]),np.array([.4,.6,.75]))
    label_colors[~d['label_mask']]=[1.,.7,.1]
    fig=plt.figure(figsize=(15,5.5))
    cloud(fig.add_subplot(131,projection='3d'),p,colors,'Reference assembly (part colors)')
    cloud(fig.add_subplot(132,projection='3d'),exploded,colors,'Exploded parts (display offset only)')
    ax=fig.add_subplot(133,projection='3d')
    cloud(ax,exploded,label_colors,'Geometry labels + surface normals')
    chosen=np.flatnonzero(d['fracture_surface_gt'])[::12]
    ax.quiver(*exploded[chosen].T,*d['pointclouds_normals_gt'][chosen].T,length=3,color='black',linewidth=.5)
    fig.suptitle(f'{name} | {len(d["points_per_part"])} parts | experimental size: 200 mm',fontsize=14)
    fig.text(.5,.02,'Red: opposing-contact pseudo-label; blue: separated surface; yellow: ignored uncertainty. Labels are not human annotations.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.94]); fig.savefig(output/'label_qa.png',dpi=170); plt.close(fig)
    fig=plt.figure(figsize=(14,4*((len(d['points_per_part'])+3)//4)))
    for i,count in enumerate(d['points_per_part']):
        ax=fig.add_subplot((len(d['points_per_part'])+3)//4,4,i+1,projection='3d')
        selected=part==i
        cloud(ax,p[selected],label_colors[selected],f'Part {i}: {count} points (own zoom)')
    fig.suptitle(f'{name}: per-part sampling and label audit; each panel uses its own scale')
    fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(output/'part_closeups.png',dpi=170); plt.close(fig)
    if run is None: return
    stages=['baseline','encoder','joint']
    available=[s for s in stages if (run/f'{s}-metrics.json').exists()]
    metrics={s:json.loads((run/f'{s}-metrics.json').read_text()) for s in available}
    if not available: return
    # Same predeclared validation example and same noise seed in every panel.
    candidates=sorted({r['object'] for r in metrics['baseline']})
    for name in candidates:
        fig=plt.figure(figsize=(5*(len(available)+1),5))
        with np.load(run/f'baseline--{name}--41.npz') as z: first=dict(z)
        part=np.repeat(np.arange(len(first['counts'])),first['counts'])
        colors=plt.get_cmap('tab10')(part%10)
        all_points=[first['truth']*1000]
        for s in available:
            with np.load(run/f'{s}--{name}--41.npz') as z: all_points.append(z['predicted']*1000)
        joined=np.concatenate(all_points); center=(joined.min(0)+joined.max(0))/2; radius=np.ptp(joined,axis=0).max()*.53
        limits=[(v-radius,v+radius) for v in center]
        cloud(fig.add_subplot(1,len(all_points),1,projection='3d'),all_points[0],colors,'Reference',limits)
        for i,s in enumerate(available):
            row=next(r for r in metrics[s] if r['object']==name and r['seed']==41)
            cloud(fig.add_subplot(1,len(all_points),i+2,projection='3d'),all_points[i+1],colors,
                  f'{s}: {row["centroid_error_mm"]:.1f} mm / {row["rotation_error_deg"]:.1f} deg',limits)
        fig.suptitle(f'{name}, noise seed 41; same axes; largest part is a FIXED ANCHOR, excluded from scores')
        fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(output/f'assembly_{name}.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,label in zip(axes,['centroid_error_mm','rotation_error_deg','fracture_f1'],['Centroid error (mm), lower is better','Rotation error (degrees), lower is better','Fracture F1, higher is better']):
        for i,s in enumerate(available):
            # Aggregate seeds within specimen first; each dot = one specimen.
            values=[np.mean([r[key] for r in metrics[s] if r['object']==n]) for n in candidates]
            ax.bar(i,np.mean(values),color=['#587b99','#e4aa45','#559675'][i],alpha=.65)
            ax.scatter(np.full(len(values),i),values,color='black',s=24,zorder=3)
        ax.set_xticks(range(len(available)),available); ax.set_title(label,fontsize=10); ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'Validation only: {len(candidates)} unseen specimens, 2 noise seeds; fixed final checkpoints')
    fig.tight_layout(rect=[0,0,1,.92]); fig.savefig(output/'validation_metrics.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,s in zip(axes,['encoder','joint']):
        path=run/f'{s}-training.json'
        if path.exists():
            history=json.loads(path.read_text())['history']
            ax.plot([r['step'] for r in history],[r['loss'] for r in history],alpha=.6)
        ax.set(title=s+' training objective',xlabel='optimizer step',ylabel='loss'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/'training_curves.png',dpi=170); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--transfer',type=Path); parser.add_argument('--baseline',type=Path)
    a=parser.parse_args(); render(a.dataset,a.output,a.run)
    if a.transfer:
        if not a.baseline: parser.error('--transfer requires --baseline')
        transfer_comparison(a.baseline,a.transfer,a.output)
