"""Measured rigid-body settling diagnostic, independent of target-pose accuracy."""
import numpy as np


def summarize_stability(times,positions,quaternions,velocities,names,window_s=1.):
    t,p,q,v=[np.asarray(a,dtype=np.float64) for a in (times,positions,quaternions,velocities)]
    n=len(names)
    if (t.ndim!=1 or len(t)<2 or np.any(np.diff(t)<=0) or
        p.shape!=(len(t),n,3) or q.shape!=(len(t),n,4) or v.shape!=(len(t),n,6)):
        raise ValueError('invalid synchronized rigid-body trace')
    if not all(np.isfinite(a).all() for a in (t,p,q,v)):raise ValueError('nonfinite rigid-body trace')
    if t[-1]-t[0]<window_s:raise ValueError('trace too short for stability window')
    if np.any(np.linalg.norm(q,axis=-1)<1e-6):raise ValueError('invalid quaternion')
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    start=max(0,np.searchsorted(t,t[-1]-window_s,side='right')-1)
    result=[]
    for i,name in enumerate(names):
        drift=np.linalg.norm(p[start:,i]-p[start,i],axis=-1).max()
        angle=np.rad2deg(2*np.arccos(np.clip(np.abs(q[start:,i]@q[start,i]),0,1))).max()
        linear=np.linalg.norm(v[start:,i,:3],axis=-1).max()
        angular=np.linalg.norm(v[start:,i,3:],axis=-1).max()
        result.append(dict(id=name,stable=bool(drift<=.001 and angle<=2 and linear<=.005 and angular<=np.deg2rad(5)),
            window_translation_mm=float(drift*1000),window_rotation_deg=float(angle),
            maximum_linear_speed_m_s=float(linear),maximum_angular_speed_rad_s=float(angular),
            initial_to_final_displacement_mm=float(np.linalg.norm(p[-1,i]-p[0,i])*1000)))
    return dict(window_s=float(t[-1]-t[start]),parts=result,all_stable=all(r['stable'] for r in result),
                assembly_success_evaluated=False,grasp_or_release_executed=False,
                thresholds=dict(drift_m=.001,rotation_deg=2,linear_speed_m_s=.005,angular_speed_deg_s=5))
