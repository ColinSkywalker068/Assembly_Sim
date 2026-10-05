import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tasks.stack_sawtooth_shulker_small.scripted import grasp_pose, motion_config
from tasks.stack_sawtooth_shulker_small.conditions import condition_to_task_config
from tasks.stack_sawtooth_shulker_small.dataset import generate_manifest


def test_inverted_grasps_use_world_top_height():
    condition = next(c for c in generate_manifest(42).conditions if c.assembly_order[0] == 'Green')
    config = condition_to_task_config(condition)
    for name, is_base in (('FragmentB', True), ('FragmentA', False)):
        fragment = config.fragment(name)
        pose = grasp_pose(config, fragment, fragment.initial_pose.position, is_base=is_base)
        assert pose[2, 3] == pytest.approx(config.table_top_z + 2*config.unit_size + (.004 if is_base else -.004))


@pytest.mark.parametrize('yaw', [0, 170, 180, -170])
def test_grasp_uses_nearest_equivalent_jaw_orientation(yaw):
    config = motion_config()
    blue = config.fragment('FragmentA')
    canonical = grasp_pose(config, blue, blue.initial_pose.position)
    reference = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ canonical[:3, :3]
    selected = grasp_pose(config, blue, blue.initial_pose.position, reference_rotation=reference)
    angle = Rotation.from_matrix(selected[:3, :3] @ reference.T).magnitude()
    assert angle <= np.deg2rad(10) + 1e-8
    assert selected[:3, 0] == pytest.approx((0, 0, -1))
    assert abs(selected[1, 1]) == pytest.approx(1)


def test_top_down_grasps_and_mating_height():
    config = motion_config()
    blue = config.fragment('FragmentA')
    green = config.fragment('FragmentB')
    for fragment in (blue, green):
        start = grasp_pose(config, fragment, fragment.initial_pose.position)
        goal = grasp_pose(config, fragment, fragment.goal_pose.position)
        assert start[:3, 0] == pytest.approx((0, 0, -1))
        assert np.linalg.det(start[:3, :3]) == pytest.approx(1)
        assert start[:2, 3] == pytest.approx(fragment.initial_pose.position[:2])
        assert goal[:2, 3] == pytest.approx((0, 0))
    assert green.goal_pose.position[2] - blue.goal_pose.position[2] == pytest.approx(.027)
    assert config.density <= 20
    assert config.workcell.physics['static_friction'] >= 1.5
    assert config.workcell.physics['dynamic_friction'] >= 1.0
