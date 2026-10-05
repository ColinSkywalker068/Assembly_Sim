import pytest

from tasks.stack_sawtooth_shulker.config import load_task_config as load_original
from tasks.stack_sawtooth_shulker_small.config import (
    PLACEMENT_SQUARE_SIZE,
    load_task_config,
)
from tasks.stack_sawtooth_shulker_small.geometry import fragment_bounds
from tasks.stack_sawtooth_shulker_small.assets import target_metadata_values


def test_small_dimensions_and_original_isolation():
    original = load_original()
    small = load_task_config(original.workcell)
    assert small.unit_size == pytest.approx(0.027)
    assert small.target_size == pytest.approx((0.185625, 0.185625))
    assert PLACEMENT_SQUARE_SIZE == pytest.approx(0.10125)
    assert original.unit_size == 0.08
    assert original.workcell.environment['assembly_pad']['size'][:2] == [0.55, 0.55]
    assert small.workcell.robots == original.workcell.robots
    assert small.workcell.cameras == original.workcell.cameras
    assert small.workcell.environment['table_size'] == original.workcell.environment['table_size']
    assert small.workcell.environment['assembly_pad']['tape_width'] == pytest.approx(0.00405)
    for fragment, prior in zip(small.fragments, original.fragments):
        lower, upper = fragment_bounds(fragment, small.unit_size)
        assert tuple(b-a for a,b in zip(lower, upper)) == pytest.approx((0.081,0.081,0.054))
        assert fragment.initial_pose.position == prior.initial_pose.position
        assert fragment.initial_pose.orientation_wxyz == prior.initial_pose.orientation_wxyz
        assert fragment.initial_pose.position[2] + lower[2] == pytest.approx(small.table_top_z)
    assert small.fragment('FragmentB').goal_pose.position[2] == pytest.approx(0.777)
    metadata = target_metadata_values(small)
    assert metadata['task_name'] == 'stack_sawtooth_shulker_small'
    assert metadata['placement_square_size'] == pytest.approx(0.10125)
