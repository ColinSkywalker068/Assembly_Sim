"""Pure geometry helpers for voxel-derived collision shapes."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Sequence


Cell = tuple[int, int, int]


@dataclass(frozen=True)
class VoxelBox:
    min_cell: Cell
    size_cells: Cell

    @property
    def cell_count(self) -> int:
        x, y, z = self.size_cells
        return x * y * z


def _validated_cells(cells: Sequence[Sequence[int]]) -> set[Cell]:
    result: set[Cell] = set()
    for raw in cells:
        if len(raw) != 3 or any(
            not isinstance(value, Integral) or isinstance(value, bool) for value in raw
        ):
            raise ValueError(f"voxel cell must be an integer triplet: {raw!r}")
        cell = tuple(int(value) for value in raw)
        if cell in result:
            raise ValueError(f"duplicate voxel cell: {cell}")
        result.add(cell)
    return result


def merge_voxel_cells(cells: Sequence[Sequence[int]]) -> list[VoxelBox]:
    """Greedily merge occupied cells into deterministic, disjoint boxes."""

    remaining = _validated_cells(cells)
    boxes: list[VoxelBox] = []
    while remaining:
        x0, y0, z0 = min(remaining, key=lambda cell: (cell[2], cell[1], cell[0]))
        size_x = 1
        while (x0 + size_x, y0, z0) in remaining:
            size_x += 1
        size_y = 1
        while all(
            (x, y0 + size_y, z0) in remaining
            for x in range(x0, x0 + size_x)
        ):
            size_y += 1
        size_z = 1
        while all(
            (x, y, z0 + size_z) in remaining
            for y in range(y0, y0 + size_y)
            for x in range(x0, x0 + size_x)
        ):
            size_z += 1
        box = VoxelBox((x0, y0, z0), (size_x, size_y, size_z))
        for z in range(z0, z0 + size_z):
            for y in range(y0, y0 + size_y):
                for x in range(x0, x0 + size_x):
                    remaining.remove((x, y, z))
        boxes.append(box)
    return boxes


def occupied_volume(cells: Sequence[Sequence[int]], pitch: float) -> float:
    if pitch <= 0:
        raise ValueError("voxel pitch must be positive")
    return len(_validated_cells(cells)) * float(pitch) ** 3


def boxes_volume(boxes: Iterable[VoxelBox], pitch: float) -> float:
    if pitch <= 0:
        raise ValueError("voxel pitch must be positive")
    total = 0
    for box in boxes:
        if any(length <= 0 for length in box.size_cells):
            raise ValueError(f"voxel box must have positive size: {box}")
        total += box.cell_count
    return total * float(pitch) ** 3
