from pathlib import Path

import numpy as np
import pytest
import trimesh

from tasks.dual_arm_breakingbad.domain.model import AssemblyInput, FragmentInput
from tasks.dual_arm_breakingbad.preprocessing.voxelize import (
    VoxelizationConfig,
    choose_vote_owner,
    keep_largest_component,
    process_assembly,
)


def _fragment(name, mesh, transform=None):
    return FragmentInput(
        name=name,
        source_name=f"{name}.obj",
        source_path=Path(f"/{name}.obj"),
        mesh=mesh,
        ground_truth_transform=np.eye(4) if transform is None else transform,
    )


def _assembly(*fragments):
    return AssemblyInput("fixture", "test", Path("/fixture"), tuple(fragments))


def _config(**overrides):
    values = {
        "pitch": 0.05,
        "target_length": 0.4,
        "seed": 7,
        "min_surface_samples": 800,
        "samples_per_pitch_area": 4.0,
    }
    values.update(overrides)
    return VoxelizationConfig(**values)


def test_one_global_transform_preserves_fragment_relationship():
    first = trimesh.creation.box(extents=(0.2, 0.2, 0.1))
    second = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    second_transform = trimesh.transformations.translation_matrix((0.8, 0.1, 0.0))
    assembly = _assembly(
        _fragment("piece_0", first),
        _fragment("piece_1", second, second_transform),
    )

    processed = process_assembly(assembly, _config())

    source_points = np.array([[0.0, 0.0, 0.0, 1.0], [0.8, 0.1, 0.0, 1.0]])
    canonical_points = (processed.source_to_canonical @ source_points.T).T[:, :3]
    singular_values = np.linalg.svd(processed.source_to_canonical[:3, :3], compute_uv=False)
    assert singular_values == pytest.approx([singular_values[0]] * 3)
    assert np.linalg.norm(canonical_points[1] - canonical_points[0]) == pytest.approx(
        np.linalg.norm(source_points[1, :3] - source_points[0, :3]) * singular_values[0]
    )


def test_target_length_and_pitch_are_recorded_exactly():
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 0.5))

    processed = process_assembly(_assembly(_fragment("piece_0", mesh)), _config(pitch=0.04))

    assert processed.config.pitch == 0.04
    assert processed.config.target_length == 0.4
    assert processed.assembly_bounds[1][0] - processed.assembly_bounds[0][0] == pytest.approx(0.4)


def test_all_fragments_share_one_grid():
    left = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    right = left.copy()
    right.apply_translation((0.4, 0.0, 0.0))

    processed = process_assembly(
        _assembly(_fragment("left", left), _fragment("right", right)), _config()
    )

    assert processed.grid_shape == tuple(int(value) for value in processed.grid_shape)
    for fragment in processed.fragments:
        assert fragment.cells
        assert all(
            0 <= cell[axis] < processed.grid_shape[axis]
            for cell in fragment.cells
            for axis in range(3)
        )


def test_cell_ownership_is_disjoint():
    first = trimesh.creation.box(extents=(0.3, 0.2, 0.2))
    second = first.copy()
    second.apply_translation((0.25, 0.0, 0.0))

    processed = process_assembly(
        _assembly(_fragment("piece_0", first), _fragment("piece_1", second)), _config()
    )

    cells = [set(fragment.cells) for fragment in processed.fragments]
    assert cells[0].isdisjoint(cells[1])


def test_highest_vote_ties_follow_fragment_order():
    assert choose_vote_owner({2: 7, 0: 7, 1: 3}) == 0
    assert choose_vote_owner({2: 8, 0: 7, 1: 3}) == 2


def test_cleanup_keeps_only_largest_26_connected_component():
    cells = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (9, 9, 9)]

    cleaned = keep_largest_component(cells)

    assert cleaned == ((0, 0, 0), (1, 1, 1), (2, 2, 2))


def test_empty_fragment_after_cleanup_is_fatal():
    mesh = trimesh.creation.box(extents=(0.4, 0.2, 0.2))
    assembly = _assembly(_fragment("winner", mesh), _fragment("loser", mesh.copy()))

    with pytest.raises(ValueError, match="loser.*no occupied voxels"):
        process_assembly(assembly, _config())


def test_same_seed_repeats_exact_cells():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    assembly = _assembly(_fragment("piece_0", mesh))

    first = process_assembly(assembly, _config(seed=19))
    second = process_assembly(assembly, _config(seed=19))

    assert first.source_to_canonical == pytest.approx(second.source_to_canonical)
    assert first.fragments[0].cells == second.fragments[0].cells


def test_goal_pose_reconstructs_voxel_cells_in_canonical_frame():
    mesh = trimesh.creation.box(extents=(0.4, 0.2, 0.2))

    processed = process_assembly(_assembly(_fragment("piece_0", mesh)), _config())
    fragment = processed.fragments[0]

    expected = np.asarray(processed.grid_origin) + np.asarray(fragment.local_pivot_cells) * 0.05
    assert fragment.goal_pose["position"] == pytest.approx(expected)
