import pytest

from tasks.stack_sawtooth_shulker.assets import (
    fragment_authoring_spec,
    target_metadata_values,
)
from tasks.stack_sawtooth_shulker.config import load_task_config
from tasks.stack_sawtooth_shulker.validation import expected_task_manifest


def test_expected_manifest_names_every_task_and_core_prim():
    config = load_task_config()
    manifest = expected_task_manifest(config.workcell, config)

    assert manifest.task_root == "/World/Task"
    assert manifest.metadata_path == "/World/Task/TargetMetadata"
    assert manifest.fragments_root == "/World/Task/Fragments"
    assert manifest.workcell.robot_paths == ("/World/Robots/Right",)
    assert manifest.workcell.camera_paths == (
        "/World/Cameras/Agent",
        "/World/Robots/Right/flange/WristCamera",
    )
    a, b = manifest.fragments
    assert a.root_path == "/World/Task/Fragments/FragmentA"
    assert b.root_path == "/World/Task/Fragments/FragmentB"
    assert len(a.visual_paths) == len(a.collider_paths) == 14
    assert len(b.visual_paths) == len(b.collider_paths) == 13
    assert a.visual_paths[0] == "/World/Task/Fragments/FragmentA/Visuals/Voxel_000"
    assert a.visual_paths[-1] == "/World/Task/Fragments/FragmentA/Visuals/Voxel_013"
    assert b.collider_paths[0] == "/World/Task/Fragments/FragmentB/Colliders/Voxel_000"
    assert b.collider_paths[-1] == "/World/Task/Fragments/FragmentB/Colliders/Voxel_012"


def test_fragment_authoring_values_preserve_geometry_clearance_and_mass():
    config = load_task_config()
    a, b = config.fragments
    a_spec = fragment_authoring_spec(config, a)
    b_spec = fragment_authoring_spec(config, b)

    assert a_spec.visual_side == pytest.approx(0.08)
    assert a_spec.collider_side == pytest.approx(0.078)
    assert a_spec.material_path == "/World/Looks/StackSawtoothShulker/FragmentA"
    assert b_spec.material_path == "/World/Looks/StackSawtoothShulker/FragmentB"
    assert a_spec.mass == pytest.approx(0.14336)
    assert b_spec.mass == pytest.approx(0.13312)
    assert len(a_spec.centers) == 14
    assert len(b_spec.centers) == 13
    assert a_spec.centers[0] == pytest.approx((-0.08, -0.08, 0.04))
    assert b_spec.centers[0] == pytest.approx((-0.08, -0.08, 0.12))
    assert b_spec.centers[-1] == pytest.approx((0.08, 0.08, 0.12))
    assert (0.0, -0.08, 0.04) in b_spec.centers


def test_target_metadata_values_expose_scale_target_and_goal_transforms():
    config = load_task_config()
    metadata = target_metadata_values(config)

    assert metadata == {
        "task_name": "stack_sawtooth_shulker",
        "unit_size": 0.08,
        "target_size": (0.55, 0.55),
        "fragment_a_goal_position": (0.0, 0.0, 0.75),
        "fragment_a_goal_orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
        "fragment_b_goal_position": (0.0, 0.0, 0.83),
        "fragment_b_goal_orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
    }
