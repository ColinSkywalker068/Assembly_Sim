import numpy as np
import pytest

from tasks.dual_arm_breakingbad.preprocessing.mesh_export import build_voxel_surface


def test_single_voxel_is_exact_cube_surface():
    vertices, faces = build_voxel_surface([(0, 0, 0)], 0.1, (0.0, 0.0, 0.0))

    assert vertices.shape == (24, 3)
    assert faces.shape == (12, 3)
    assert vertices.dtype == np.float32
    assert faces.dtype == np.int32
    assert vertices.min(axis=0) == pytest.approx((0.0, 0.0, 0.0))
    assert vertices.max(axis=0) == pytest.approx((0.1, 0.1, 0.1))


def test_adjacent_voxels_omit_internal_faces():
    vertices, faces = build_voxel_surface(
        [(0, 0, 0), (1, 0, 0)], 0.1, (0.0, 0.0, 0.0)
    )

    assert vertices.shape == (40, 3)
    assert faces.shape == (20, 3)
    assert vertices.min(axis=0) == pytest.approx((0.0, 0.0, 0.0))
    assert vertices.max(axis=0) == pytest.approx((0.2, 0.1, 0.1))


def test_surface_bounds_reconstruct_occupied_cells():
    pitch = 0.016
    pivot = (2.5, 3.5, 0.0)

    vertices, _ = build_voxel_surface([(2, 3, 1), (2, 3, 2)], pitch, pivot)

    assert vertices.min(axis=0) == pytest.approx((-0.5 * pitch, -0.5 * pitch, pitch))
    assert vertices.max(axis=0) == pytest.approx((0.5 * pitch, 0.5 * pitch, 3 * pitch))


def test_export_contains_no_non_cube_vertices():
    pitch = 0.016
    pivot = (1.5, 2.5, 0.0)

    vertices, _ = build_voxel_surface([(1, 2, 0), (1, 2, 1)], pitch, pivot)
    world_grid_coordinates = vertices / pitch + np.asarray(pivot)

    np.testing.assert_allclose(world_grid_coordinates, np.round(world_grid_coordinates))
