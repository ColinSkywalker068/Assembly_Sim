import pytest

from tasks.stack_sawtooth_shulker.geometry import (
    COMPLETE_CELLS,
    FRAGMENT_A_CELLS,
    FRAGMENT_B_CELLS,
    cell_center,
    local_cells,
)


BOTTOM_LAYER = frozenset((x, y, 0) for x in range(3) for y in range(3))
MIDDLE_A = frozenset(
    ((0, 0, 1), (0, 2, 1), (2, 0, 1), (2, 2, 1), (1, 1, 1))
)
MIDDLE_B = frozenset(((1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1)))
TOP_LAYER = frozenset((x, y, 2) for x in range(3) for y in range(3))


def test_fragment_cells_encode_the_exact_keyed_partition():
    assert set(FRAGMENT_A_CELLS) == BOTTOM_LAYER | MIDDLE_A
    assert set(FRAGMENT_B_CELLS) == MIDDLE_B | TOP_LAYER
    assert len(FRAGMENT_A_CELLS) == 14
    assert len(FRAGMENT_B_CELLS) == 13
    assert set(FRAGMENT_A_CELLS).isdisjoint(FRAGMENT_B_CELLS)
    assert set(FRAGMENT_A_CELLS) | set(FRAGMENT_B_CELLS) == set(COMPLETE_CELLS)
    assert len(COMPLETE_CELLS) == 27


def test_local_cells_put_each_fragment_on_its_lowest_layer():
    a_offset, a_cells = local_cells(FRAGMENT_A_CELLS)
    b_offset, b_cells = local_cells(FRAGMENT_B_CELLS)

    assert a_offset == 0
    assert b_offset == 1
    assert {cell[2] for cell in a_cells} == {0, 1}
    assert {cell[2] for cell in b_cells} == {0, 1}
    assert set((x, y, z + b_offset) for x, y, z in b_cells) == set(
        FRAGMENT_B_CELLS
    )


def test_cell_centers_are_xy_centered_and_rest_on_z_zero():
    assert cell_center((0, 0, 0), 0.11) == pytest.approx((-0.11, -0.11, 0.055))
    assert cell_center((1, 1, 0), 0.11) == pytest.approx((0.0, 0.0, 0.055))
    assert cell_center((2, 2, 1), 0.11) == pytest.approx((0.11, 0.11, 0.165))


@pytest.mark.parametrize("unit_size", (0.0, -0.1, float("nan"), float("inf")))
def test_cell_center_rejects_nonpositive_or_nonfinite_units(unit_size):
    with pytest.raises(ValueError, match="unit size"):
        cell_center((0, 0, 0), unit_size)
