import numpy as np
import pytest
import trimesh
from tasks.dual_arm_crag_skull.experiments.skull_dataset import grouped_split, specimen_id, contact_labels


def test_split_keeps_same_specimen_together_and_order_independent():
    names=[f'AMNH_{i}_Cranium' for i in range(20)]+['AMNH_3_Skull','AMNH_10_Skull']
    a=grouped_split(names)
    assert a==grouped_split(names[::-1])
    assert a['AMNH_3_Skull']==a['AMNH_3_Cranium']
    assert a['AMNH_10_Skull']==a['AMNH_10_Cranium']
    assert set(a.values())=={'train','val','test'}
    with pytest.raises(ValueError): specimen_id('unknown')


def test_contact_labels_distinguish_opposing_surface_from_uncertain_gap():
    a=trimesh.creation.box(); b=trimesh.creation.box()
    b.apply_translation([1,0,0])
    d={'pointclouds_gt':np.array([[.5,0,0],[-.5,0,0],[.5,0,0],[1.5,0,0]]),
       'pointclouds_normals_gt':np.array([[1,0,0],[-1,0,0],[-1,0,0],[1,0,0]]),
       'points_per_part':np.array([2,2])}
    positive,mask,*_=contact_labels([a,b],d)
    np.testing.assert_array_equal(positive,[True,False,True,False])
    assert mask.all()
    b.apply_translation([.002,0,0])
    d['pointclouds_gt'][2:,0]+=.002
    positive,mask,*_=contact_labels([a,b],d)
    assert not positive.any()
    np.testing.assert_array_equal(mask,[False,True,False,True])
