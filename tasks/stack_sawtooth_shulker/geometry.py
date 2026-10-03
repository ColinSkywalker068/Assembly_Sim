"""Pure analytical geometry for the two-piece keyed cube."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Iterable, Protocol, Sequence


Cell = tuple[int, int, int]


class FragmentGeometry(Protocol):
    local_cells: tuple[Cell, ...]

COMPLETE_CELLS: tuple[Cell, ...] = tuple(
    (x, y, z) for z in range(3) for y in range(3) for x in range(3)
)
FRAGMENT_A_CELLS: tuple[Cell, ...] = tuple(
    sorted(
        {
            *((x, y, 0) for x in range(3) for y in range(3)),
            (0, 0, 1),
            (0, 2, 1),
            (2, 0, 1),
            (2, 2, 1),
            (1, 1, 1),
        }
    )
)
FRAGMENT_B_CELLS: tuple[Cell, ...] = tuple(
    sorted(set(COMPLETE_CELLS) - set(FRAGMENT_A_CELLS))
)


def _cell_tuple(raw: Sequence[int]) -> Cell:
    if len(raw) != 3 or any(
        not isinstance(value, Integral) or isinstance(value, bool) for value in raw
    ):
        raise ValueError(f"voxel cell must be an integer triplet: {raw!r}")
    return tuple(int(value) for value in raw)


def local_cells(cells: Iterable[Sequence[int]]) -> tuple[int, tuple[Cell, ...]]:
    """Move a canonical fragment's lowest Z layer to local Z zero."""

    canonical = tuple(_cell_tuple(cell) for cell in cells)
    if not canonical:
        raise ValueError("fragment must contain at least one voxel cell")
    z_offset = min(cell[2] for cell in canonical)
    localized = tuple(sorted((x, y, z - z_offset) for x, y, z in canonical))
    return z_offset, localized


def cell_center(cell: Sequence[int], unit_size: float) -> tuple[float, float, float]:
    """Return a fragment-local voxel center with a centered 3-by-3 footprint."""

    unit = float(unit_size)
    if not math.isfinite(unit) or unit <= 0:
        raise ValueError("unit size must be finite and positive")
    x, y, z = _cell_tuple(cell)
    return ((x - 1) * unit, (y - 1) * unit, (z + 0.5) * unit)


def fragment_bounds(
    fragment: FragmentGeometry, unit_size: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return local axis-aligned bounds for a fragment's complete voxel cubes."""

    unit = float(unit_size)
    if not math.isfinite(unit) or unit <= 0:
        raise ValueError("unit size must be finite and positive")
    cells = tuple(fragment.local_cells)
    if not cells:
        raise ValueError("fragment must contain at least one local voxel cell")
    centers = tuple(cell_center(cell, unit) for cell in cells)
    half = unit / 2
    return (
        tuple(min(center[axis] for center in centers) - half for axis in range(3)),
        tuple(max(center[axis] for center in centers) + half for axis in range(3)),
    )


def fragment_mass(fragment: FragmentGeometry, unit_size: float, density: float) -> float:
    """Compute mass from nominal occupied voxel volume and material density."""

    unit = float(unit_size)
    material_density = float(density)
    if not math.isfinite(unit) or unit <= 0:
        raise ValueError("unit size must be finite and positive")
    if not math.isfinite(material_density) or material_density <= 0:
        raise ValueError("density must be finite and positive")
    return len(tuple(fragment.local_cells)) * unit**3 * material_density
