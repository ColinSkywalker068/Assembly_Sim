from collections import Counter
from dataclasses import replace

import pytest

from tasks.stack_sawtooth_shulker_small.dataset import generate_manifest, validate_manifest, write_manifest, read_manifest
from tasks.stack_sawtooth_shulker_small.conditions import condition_to_task_config
from tasks.stack_sawtooth_shulker_small.preview import preview_guide_specs


def test_constrained_manifest_is_balanced_and_reproducible(tmp_path):
    manifest = generate_manifest(20261005)
    assert manifest == generate_manifest(20261005)
    validate_manifest(manifest)
    assert len(manifest.conditions) == 50
    counts = Counter((c.assembly_order[0], c.high_color) for c in manifest.conditions[2:])
    assert set(counts.values()) == {12}
    for c in manifest.conditions:
        config = condition_to_task_config(c)
        base = config.fragment('FragmentA' if c.assembly_order[0] == 'Blue' else 'FragmentB')
        top = config.fragment('FragmentB' if c.assembly_order[0] == 'Blue' else 'FragmentA')
        assert base.initial_pose.orientation_wxyz == base.goal_pose.orientation_wxyz
        assert top.initial_pose.orientation_wxyz == top.goal_pose.orientation_wxyz
        assert c.blue.pose_label == ('T-up' if c.assembly_order[0] == 'Blue' else 'C-up')
        assert c.green.pose_label == ('C-up' if c.assembly_order[0] == 'Blue' else 'T-up')
        assert all(abs(v) <= .050625 for v in c.base_target_xy)
        square = [s for s in preview_guide_specs(c, .75) if '/PlacementSquare/' in s.path]
        assert len(square) == 4
        assert max(max(s.size[:2]) for s in square) == pytest.approx(.10125)
    path = write_manifest(manifest, tmp_path/'manifest.json')
    assert read_manifest(path) == manifest


def test_manifest_rejects_pose_and_target_changes():
    manifest = generate_manifest(20261005)
    c = manifest.conditions[2]
    for bad in (replace(c, base_target_xy=(.06, 0)),
                replace(c, blue=replace(c.blue, pose_label='C-up'))):
        with pytest.raises(ValueError):
            validate_manifest(replace(manifest, conditions=manifest.conditions[:2]+(bad,)+manifest.conditions[3:]))
