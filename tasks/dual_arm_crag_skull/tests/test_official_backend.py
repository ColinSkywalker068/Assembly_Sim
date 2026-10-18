import numpy as np
from scipy.spatial.transform import Rotation
from tasks.dual_arm_crag_skull.tools.official_backend import decode_local_poses


def test_official_decode_restores_arbitrary_local_frames_and_metres():
    rng=np.random.default_rng(9)
    augmentation=Rotation.random(3,random_state=rng)
    center=rng.normal(size=(3,3))
    target=Rotation.random(3,random_state=rng)
    translation=rng.normal(size=(3,3))*.1
    common=.20749
    data={'quaternions':augmentation.inv().as_quat()[:,[3,0,1,2]],'translations':center}
    predicted_rot=target*augmentation.inv()
    predicted_t=translation/common+target.apply(center)
    values=np.concatenate([predicted_t,predicted_rot.as_quat()[:,[3,0,1,2]]],axis=1)
    poses=decode_local_poses(values,data,common)
    for i,t in enumerate(poses):
        np.testing.assert_allclose(t[:3,:3],target[i].as_matrix(),atol=1e-12)
        np.testing.assert_allclose(t[:3,3],translation[i],atol=1e-12)
        p=rng.normal(size=(20,3))*.03
        np.testing.assert_allclose(p@t[:3,:3].T+t[:3,3],target[i].apply(p)+translation[i],atol=1e-12)


def test_fracture_ranking_metrics_handle_ties():
    from tasks.dual_arm_crag_skull.experiments.fracture_thresholds import ranking_metrics
    auc,ap=ranking_metrics(np.array([True,False,True]),np.array([.9,.8,.7]))
    np.testing.assert_allclose([auc,ap],[.5,5/6])
    np.testing.assert_allclose(ranking_metrics(np.array([False,True]),np.array([.5,.5])),[.5,.5])
