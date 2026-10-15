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
