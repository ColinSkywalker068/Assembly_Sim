import numpy as np
import pytest
from tasks.dual_arm_crag_skull.stability import summarize_stability


def test_stability_checks_motion_inside_window_not_just_matching_endpoints():
    t=np.linspace(0,2,21);p=np.zeros((21,1,3));q=np.tile([1,0,0,0],(21,1,1));v=np.zeros((21,1,6))
    assert summarize_stability(t,p,q,v,['part'])['all_stable']
    p[15,0,0]=.01
    assert not summarize_stability(t,p,q,v,['part'])['all_stable']
    with pytest.raises(ValueError):summarize_stability(t[:5],p[:5],q[:5],v[:5],['part'])


def test_quaternion_sign_does_not_look_like_rotation():
    t=np.linspace(0,2,21);p=np.zeros((21,1,3));q=np.tile([1.,0,0,0],(21,1,1));v=np.zeros((21,1,6))
    q[::2]*=-1
    assert summarize_stability(t,p,q,v,['part'])['all_stable']
    v[-1,0,0]=.01
    assert not summarize_stability(t,p,q,v,['part'])['all_stable']


def test_small_rotations_remain_visible_with_float32_simulator_quaternions():
    t=np.linspace(0,2,21);p=np.zeros((21,1,3));q=np.tile([1.,0,0,0],(21,1,1)).astype('float32');v=np.zeros((21,1,6))
    angle=np.deg2rad(.01)
    q[-1,0]=[np.cos(angle/2),np.sin(angle/2),0,0]
    measured=summarize_stability(t,p,q,v,['part'])['parts'][0]['window_rotation_deg']
    assert abs(measured-.01)<1e-6
