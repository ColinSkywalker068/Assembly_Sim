import numpy as np
import pytest
from tasks.dual_arm_crag_skull.perception import depth_to_world, anchor_goals
from tasks.dual_arm_crag_skull.tools.garf import GarfTool


def test_depth_units_and_invalid_pixels():
    depth = np.array([[1000, 0], [np.nan, 2000]])
    k = np.diag([2., 2., 1.])
    transform = np.eye(4); transform[0,3] = 1
    points = depth_to_world(depth, np.ones((2,2)), k, transform, .001)
    np.testing.assert_allclose(points, [[1,0,1],[2,1,2]])


def test_anchor_frame_cancels_gauge():
    a = np.eye(4); a[0,3] = 2
    b = np.eye(4); b[0,3] = 3
    observed = np.eye(4); observed[1,3] = .4
    result = anchor_goals({'a':a, 'b':b}, 'a', observed)
    np.testing.assert_allclose(result['a'], observed)
    np.testing.assert_allclose(result['b'][:3,3], [1,.4,0])


class FakeBackend:
    calls = 0
    def predict(self, parts, anchor, seed):
        self.calls += 1
        return [np.eye(4) for _ in parts]


def test_tool_stale_and_unknown_never_reach_backend():
    backend = FakeBackend()
    rng = np.random.default_rng(4)
    geometry = {key:(rng.normal(size=(40,3)), np.tile([0,0,1], (40,1))) for key in ['a','b']}
    tool = GarfTool(backend, geometry, 'test-only')
    req = {'observation_id':'old', 'fragment_ids':['a','b'], 'anchor_id':'a', 'seed':1}
    assert tool(req,current_observation_id='new')['status'] == 'rejected'
    req['observation_id'] = 'new'
    result = tool(req,current_observation_id='new')
    assert result['status'] == 'candidate'
    assert result['confidence'] is None
    req['fragment_ids'] = ['a','missing']
    assert tool(req,current_observation_id='new')['status'] == 'rejected'
    assert backend.calls == 1


def test_reflection_rejected():
    bad = np.eye(4); bad[0,0] = -1
    with pytest.raises(ValueError, match='SE'):
        anchor_goals({'a':bad}, 'a', np.eye(4))


def test_voxel_bridge_preserves_owned_cells_and_respects_neighbors():
    from tasks.dual_arm_crag_skull.voxel_repair import bridge_one_cell
    cells = [(0,0,0),(2,0,0)]
    result, added = bridge_one_cell(cells, set())
    assert set(cells).issubset(result) and added == [(1,0,0)]
    with pytest.raises(ValueError):
        bridge_one_cell(cells, {(1,0,0)})


def test_partial_icp_recovers_small_displacement():
    from tasks.dual_arm_crag_skull.perception import refine_pose
    model = np.random.default_rng(3).normal(size=(300,3))*.05
    observed = model[:150]+[.001,.002,0]
    result = refine_pose(model,observed,np.eye(4))
    np.testing.assert_allclose(result['world_T_part'][:3,3],[.001,.002,0],atol=1e-6)
    assert result['inlier_rmse_m'] < 1e-6


def test_multi_seed_and_rigid_cluster():
    from tasks.dual_arm_crag_skull.tools.garf import propose_candidates, fuse_verified_cluster, expand_cluster_pose
    p = np.random.default_rng(1).normal(size=(32,3))
    geometry = {'a':(p,np.tile([0,0,1],(32,1))), 'b':(p,np.tile([0,0,1],(32,1)))}
    tool = GarfTool(FakeBackend(),geometry,'test')
    request = {'observation_id':'now','fragment_ids':['a','b'],'anchor_id':'a','seeds':[1,2,3]}
    result = propose_candidates(tool,request,current_observation_id='now')
    assert [r['seed'] for r in result['candidates']] == [1,2,3]
    assert result['selected_candidate'] is None
    b = np.eye(4); b[0,3]=1
    with pytest.raises(ValueError):
        fuse_verified_cluster(geometry,{'a':np.eye(4),'b':b},stability_verified=False)
    merged, mapping = fuse_verified_cluster(geometry,{'a':np.eye(4),'b':b},stability_verified=True)
    np.testing.assert_allclose(merged[0][32:],p+[1,0,0],atol=1e-6)
    np.testing.assert_allclose(expand_cluster_pose(np.eye(4),mapping)['b'],b)
