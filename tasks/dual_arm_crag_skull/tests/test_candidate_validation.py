import numpy as np
import trimesh
from tasks.dual_arm_crag_skull.tools.candidate_validation import score_candidate


def test_penetration_and_disconnected_candidates_rejected():
    meshes={k:trimesh.creation.box(extents=(.02,.02,.02)) for k in ('a','b')}
    a=np.eye(4); b=np.eye(4); b[0,3]=.005
    assert score_candidate(meshes,{'a':a,'b':b})['rejected']
    b[0,3]=.02
    assert not score_candidate(meshes,{'a':a,'b':b},samples=1024)['rejected']
    b[0,3]=.1
    assert score_candidate(meshes,{'a':a,'b':b})['contact_components']==2


def test_rejected_set_never_selects_a_candidate():
    from tasks.dual_arm_crag_skull.tools.candidate_validation import screen_result
    meshes={k:trimesh.creation.box(extents=(.02,.02,.02)) for k in ('a','b')}
    candidate={'status':'candidate','seed':1,'assembly_T_fragment':{'a':np.eye(4),'b':np.eye(4)}}
    result=screen_result(meshes,{'candidates':[candidate]})
    assert result['geometry_status']=='all_candidates_rejected'
    assert result['selected_candidate'] is None and not result['motion_authorized']


def test_thin_hollow_shell_contains_wall_but_not_cavity_at_multiple_units():
    from tasks.dual_arm_crag_skull.tools.candidate_validation import SolidContainment
    for scale in (.001,1.,1000.):
        outer=trimesh.creation.box(extents=np.full(3,.2*scale))
        inner=trimesh.creation.box(extents=np.full(3,.198*scale)); inner.invert()
        shell=trimesh.util.concatenate([outer,inner])
        result=SolidContainment(shell).contains(np.array([[0,0,0],[.0995,0,0],[.11,0,0]])*scale)
        np.testing.assert_array_equal(result,[False,True,False])


def test_tiny_fragment_contact_does_not_require_one_percent_of_large_part():
    meshes={'large':trimesh.creation.box(extents=(.2,.2,.2)),
            'small':trimesh.creation.box(extents=(.01,.01,.01))}
    pose=np.eye(4); pose[0,3]=.105
    result=score_candidate(meshes,{'large':np.eye(4),'small':pose},samples=512)
    assert result['contact_components']==1
    assert not result['rejected']
