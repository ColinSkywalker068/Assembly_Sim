"""The CPU parts of experiments/assembly_settle_isaac.py (the Isaac run itself needs Isaac Sim and a GPU)."""
import hashlib

import numpy as np
import pytest

from tasks.dual_arm_crag_skull.baselines.task import task_from_layout
from tasks.dual_arm_crag_skull.experiments.assembly_settle import plan_orders
from tasks.dual_arm_crag_skull.experiments.assembly_settle_isaac import placed_task, prefix_sets
from tasks.dual_arm_crag_skull.tests.baselines_fixtures import write_layout


def test_prefix_sets_are_every_partial_assembly_once():
    assert prefix_sets([('b', 'c'), ('c', 'b')]) == [('b',), ('c',), ('b', 'c')]


def test_a_placed_task_starts_the_placed_fragments_at_their_goals(tmp_path):
    task = task_from_layout(write_layout(tmp_path / 'layout'), anchor_id='frag_a')
    trial = placed_task(task, ('frag_b',))
    np.testing.assert_array_equal(trial.fragment('frag_b').initial_world_T_fragment,
                                  task.fragment('frag_b').goal_world_T_fragment)
    np.testing.assert_array_equal(trial.fragment('frag_c').initial_world_T_fragment,
                                  task.fragment('frag_c').initial_world_T_fragment)
    assert trial.assembly_plan == task.assembly_plan and trial.anchor_id == 'frag_a'
    with pytest.raises(ValueError, match='frag_z'):
        placed_task(task, ('frag_z',))


def test_plan_orders_come_from_the_layout_it_names(tmp_path):
    path = write_layout(tmp_path / 'layout')  # frag_b and frag_c on either side of the anchor: either order
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert plan_orders(path, digest) == [('frag_b', 'frag_c'), ('frag_c', 'frag_b')]
    with pytest.raises(ValueError, match='sha256'):
        plan_orders(path, '0' * 64)
