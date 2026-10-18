"""All-object paired comparisons; no best-candidate or best-object selection."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .skull_visualize import cloud

ARMS=['previous','flow20','flow200','multi200']
LABELS=['Previous joint\n(historical)','20 points min\nflow only','200 points min\nflow only','200 points min\nflow + 0.2 Dice']
COLORS=['#999999','#587b99','#dc9b38','#509971']


def render(dataset,run,out,physics=None,physics_before=None):
    out.mkdir(parents=True,exist_ok=True)
    cohort=json.loads((dataset/'cohort.json').read_text())
    objects=sorted(r['id'] for r in cohort['items'] if r['split']=='val')
    metrics={a:json.loads((run/a/(a+'-metrics.json')).read_text()) for a in ARMS}
    screening={a:json.loads((run/a/(a+'-screening.json')).read_text()) for a in ARMS}
    for a in ARMS:
        expected={(n,s) for n in objects for s in [41,42]}
        if {(r['object'],r['seed']) for r in metrics[a]}!=expected:raise ValueError('incomplete metrics')
        if {(r['object'],r['seed']) for r in screening[a]}!=expected:raise ValueError('incomplete geometry screening')
    summary={}
    for a in ARMS:
        summary[a]={}
        for group,ids in [('all',objects),('original_two',[n for n in objects if n.startswith('skull__')]),
                          ('new_seven',[n for n in objects if not n.startswith('skull__')])]:
            rows=[r for r in metrics[a] if r['object'] in ids]
            summary[a][group]={k:float(np.mean([r[k] for r in rows])) for k in ['centroid_error_mm','rotation_error_deg','point_rmse_mm','fracture_f1']}
        summary[a]['geometry_eligible']=sum(not r['rejected'] for r in screening[a])
        summary[a]['candidate_count']=len(screening[a])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,key,title in zip(axes,['centroid_error_mm','rotation_error_deg','fracture_f1'],
                            ['Position error (mm) - lower is better','Rotation error (degrees) - lower is better','Fracture F1 - higher is better']):
        values=np.array([[np.mean([r[key] for r in metrics[a] if r['object']==n]) for a in ARMS] for n in objects])
        for row in values:ax.plot(range(4),row,color='#777777',alpha=.3,lw=.8)
        ax.bar(range(4),values.mean(0),color=COLORS,alpha=.55)
        for i in range(4):ax.scatter(np.full(len(objects),i),values[:,i],color=COLORS[i],s=25,zorder=3,edgecolor='black',linewidth=.3)
        ax.set_xticks(range(4),LABELS,fontsize=8);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.15)
    fig.suptitle(f'{len(objects)} independent validation skulls, 2 noise seeds each; identical metric points; fixed final weights')
    fig.text(.5,.01,'Dots and connecting lines: specimen means. Previous model is historical; the three new arms share the same starting weights and training steps.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(out/'paired_metrics.png',dpi=180);plt.close(fig)
    examples=[next(n for n in objects if n.startswith('skull__')),next(n for n in objects if not n.startswith('skull__'))]
    for name in examples:
        records={a:next(r for r in metrics[a] if r['object']==name and r['seed']==41) for a in ARMS}
        arrays={}
        for a in ARMS:
            with np.load(run/a/f'{a}--{name}--41.npz') as z:arrays[a]=dict(z)
        first=arrays['previous'];part=np.repeat(np.arange(len(first['counts'])),first['counts'])
        colors=plt.get_cmap('tab10')(part%10);colors[part==records['previous']['anchor']]=[.6,.6,.6,1.]
        points=[first['truth']*1000]+[arrays[a]['predicted']*1000 for a in ARMS]
        joined=np.concatenate(points);center=(joined.min(0)+joined.max(0))/2;radius=np.ptp(joined,axis=0).max()*.53
        limits=[(v-radius,v+radius) for v in center]
        fig=plt.figure(figsize=(15,10));cloud(fig.add_subplot(231,projection='3d'),points[0],colors,'Reference',limits)
        for i,a in enumerate(ARMS):
            r=records[a];s=next(r for r in screening[a] if r['object']==name and r['seed']==41)
            status='geometry rejected' if s['rejected'] else 'geometry passed; motion unverified'
            cloud(fig.add_subplot(2,3,i+2,projection='3d'),points[i+1],colors,
                  f'{a}: {r["centroid_error_mm"]:.1f} mm / {r["rotation_error_deg"]:.1f} deg\n{status}',limits)
        ax=fig.add_subplot(236)
        for i,m in enumerate([20,200]):
            with np.load(dataset/f'data{m}'/(name+'.npz')) as z:counts=z['points_per_part']
            ax.bar(np.arange(len(counts))+(i-.5)*.35,counts,width=.35,label=f'min {m}')
        ax.set(yscale='log',xlabel='fragment index',ylabel='input points (log scale)',title='Same total 5,000 points');ax.legend()
        fig.suptitle(name.split('__',1)[1]+' | noise 41 | gray largest fragment is a FIXED anchor')
        fig.text(.5,.01,'Predicted target geometry, not a physical assembly rollout. Example chosen by name before comparing improvements.',ha='center',fontsize=10)
        fig.subplots_adjust(left=.04,right=.97,bottom=.10,top=.88,hspace=.35,wspace=.16);fig.savefig(out/('assembly_'+name+'.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,a in zip(axes,ARMS[1:]):
        data=json.loads((run/a/'joint-training.json').read_text())['history']
        ax.plot([r['flow_loss'] for r in data],alpha=.7,label='flow loss')
        if a=='multi200':ax.plot([r['dice_loss'] for r in data],alpha=.7,label='Dice loss (unweighted)')
        ax.set(title=a,xlabel='step');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'loss_curves.png',dpi=170);plt.close(fig)
    if physics and (physics/'smoke.json').exists():
        report=json.loads((physics/'smoke.json').read_text())
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        for ax,name in zip(axes,report['cameras']):
            ax.imshow(plt.imread(physics/(name+'.png')));ax.set_title(name);ax.axis('off')
        fig.suptitle('Isaac Sim: measured settling test, Robotiq proxy; no grasp or assembly executed')
        fig.tight_layout();fig.savefig(out/'isaac_three_views.png',dpi=160);plt.close(fig)
        with np.load(physics/'rigid_body_trace.npz') as z:d=dict(z)
        displacement=np.linalg.norm(d['positions']-d['positions'][0],axis=-1)*1000
        speed=np.linalg.norm(d['velocities'][:,:,:3],axis=-1)*1000
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for i in range(displacement.shape[1]):
            axes[0].plot(d['times'],displacement[:,i],label=f'part {i}')
            axes[1].plot(d['times'],speed[:,i])
        axes[0].set(xlabel='simulation seconds',ylabel='displacement from start (mm)')
        axes[1].set(xlabel='simulation seconds',ylabel='linear speed (mm/s)')
        axes[0].legend(ncol=2);fig.suptitle('Actual PhysX body states; settling does not establish assembly correctness')
        fig.tight_layout();fig.savefig(out/'settling_trace.png',dpi=170);plt.close(fig)
        if physics_before:
            before=json.loads((physics_before/'smoke.json').read_text())['settling']['parts']
            after=report['settling']['parts']
            if [r['id'] for r in before]!=[r['id'] for r in after]:raise ValueError('different simulated fragments')
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            for ax,key,factor,threshold,title in [(axes[0],'maximum_linear_speed_m_s',1000,5,'Maximum linear speed (mm/s)'),
                                                 (axes[1],'maximum_angular_speed_rad_s',180/np.pi,5,'Maximum angular speed (degrees/s)')]:
                x=np.arange(len(before))
                ax.bar(x-.18,[r[key]*factor for r in before],width=.36,color='#999999',label='default solver')
                ax.bar(x+.18,[r[key]*factor for r in after],width=.36,color='#509971',label='16 position / 4 velocity iterations')
                ax.axhline(threshold,color='#b3261e',ls='--',label='stability threshold')
                ax.set(title=title,xlabel='fragment index');ax.legend(fontsize=8)
            fig.suptitle('Same fragment geometry, mass and friction; solver precision ablation; no grasp/assembly executed')
            fig.tight_layout();fig.savefig(out/'settling_comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['dataset','run','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--physics',type=Path)
    p.add_argument('--physics-before',type=Path)
    a=p.parse_args();render(a.dataset,a.run,a.output,a.physics,a.physics_before)
