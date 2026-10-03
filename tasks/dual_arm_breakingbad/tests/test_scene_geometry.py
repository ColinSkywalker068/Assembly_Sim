from itertools import product

import pytest

from core.workcell.geometry import assembly_pad_geometry
from tasks.dual_arm_breakingbad.scene.assets import _collider_local_geometry
from tasks.dual_arm_breakingbad.scene.geometry import (
    boxes_volume,
    merge_voxel_cells,
    occupied_volume,
)


def covered_cells(box):
    origin = box.min_cell
    size = box.size_cells
    return {
        (origin[0] + x, origin[1] + y, origin[2] + z)
        for x, y, z in product(range(size[0]), range(size[1]), range(size[2]))
    }


def test_solid_block_becomes_one_positive_box():
    cells = list(product(range(3), range(2), range(2)))

    boxes = merge_voxel_cells(cells)

    assert len(boxes) == 1
    assert boxes[0].min_cell == (0, 0, 0)
    assert boxes[0].size_cells == (3, 2, 2)
    assert all(length > 0 for length in boxes[0].size_cells)


def test_l_shape_becomes_two_non_overlapping_boxes():
    cells = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]

    boxes = merge_voxel_cells(cells)
    covered = [covered_cells(box) for box in boxes]

    assert len(boxes) == 2
    assert covered[0].isdisjoint(covered[1])
    assert set().union(*covered) == set(cells)


def test_rejects_duplicate_cells():
    with pytest.raises(ValueError, match="duplicate voxel cell"):
        merge_voxel_cells([(0, 0, 0), (0, 0, 0)])


@pytest.mark.parametrize("cells", [[(0, 0)], [(0, 0, 0.5)], [(0, "1", 0)]])
def test_rejects_malformed_cells(cells):
    with pytest.raises(ValueError, match="integer triplet"):
        merge_voxel_cells(cells)


def test_merged_boxes_preserve_occupied_volume():
    cells = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 2, 2),
    ]
    pitch = 0.016

    boxes = merge_voxel_cells(cells)

    assert boxes_volume(boxes, pitch) == pytest.approx(occupied_volume(cells, pitch))
    assert all(all(length > 0 for length in box.size_cells) for box in boxes)


def test_merged_boxes_cover_every_cell_once():
    cells = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 2, 1)]
    boxes = merge_voxel_cells(cells)
    covered = [cell for box in boxes for cell in covered_cells(box)]

    assert sorted(covered) == sorted(cells)
    assert len(covered) == len(set(covered))


def test_fragment_collider_local_coordinates_match_visual_pivot():
    box = merge_voxel_cells([(2, 3, 1), (2, 3, 2)])[0]

    center, size = _collider_local_geometry(box, 0.1, (2.5, 3.5, 0.0))

    assert center == pytest.approx((0.0, 0.0, 0.2))
    assert size == pytest.approx((0.1, 0.1, 0.2))


def test_assembly_pad_geometry_uses_the_demo_reference_surface_and_collider():
    geometry = assembly_pad_geometry(
        {"center": [0.0, 0.1, 0.75], "size": [0.512, 0.416, 0.0096]}
    )

    assert geometry.center == pytest.approx((0.0, 0.1, 0.75))
    assert geometry.top_z == pytest.approx(0.75)
    assert geometry.collider_center == pytest.approx((0.0, 0.1, 0.7452))
