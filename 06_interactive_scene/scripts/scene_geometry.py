"""Pure geometry helpers for voxel-derived collision shapes."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Mapping, Sequence


Cell = tuple[int, int, int]


@dataclass(frozen=True)
class VoxelBox:
    min_cell: Cell
    size_cells: Cell

    @property
    def cell_count(self) -> int:
        x, y, z = self.size_cells
        return x * y * z


@dataclass(frozen=True)
class PlateGeometry:
    offset_xy: tuple[float, float]
    size: tuple[float, float, float]

    @property
    def local_surface_center(self) -> tuple[float, float, float]:
        return (
            self.offset_xy[0] + self.size[0] / 2,
            self.offset_xy[1] + self.size[1] / 2,
            0.0,
        )

    @property
    def collider_local_center(self) -> tuple[float, float, float]:
        x, y, _ = self.local_surface_center
        return (x, y, -self.size[2] / 2)

    def world_surface_center(
        self, origin: Sequence[float]
    ) -> tuple[float, float, float]:
        local = self.local_surface_center
        return tuple(float(origin[index]) + local[index] for index in range(3))


@dataclass(frozen=True)
class AssemblyPadGeometry:
    center: tuple[float, float, float]
    size: tuple[float, float, float]

    @property
    def top_z(self) -> float:
        return self.center[2]

    @property
    def collider_center(self) -> tuple[float, float, float]:
        return (self.center[0], self.center[1], self.top_z - self.size[2] / 2)


def plate_geometry(spec: Mapping[str, Sequence[float]]) -> PlateGeometry:
    return PlateGeometry(
        offset_xy=tuple(float(value) for value in spec["offset"]),
        size=tuple(float(value) for value in spec["size"]),
    )


def assembly_pad_geometry(spec: Mapping[str, Sequence[float]]) -> AssemblyPadGeometry:
    center = tuple(float(value) for value in spec["center"])
    size = tuple(float(value) for value in spec["size"])
    if len(center) != 3 or len(size) != 3 or any(value <= 0 for value in size):
        raise ValueError("assembly pad center and size must be finite positive 3D values")
    return AssemblyPadGeometry(center=center, size=size)


def _validated_cells(cells: Sequence[Sequence[int]]) -> set[Cell]:
    result: set[Cell] = set()
    for raw in cells:
        if len(raw) != 3 or any(not isinstance(value, Integral) for value in raw):
            raise ValueError(f"voxel cell must be an integer triplet: {raw!r}")
        cell = tuple(int(value) for value in raw)
        if cell in result:
            raise ValueError(f"duplicate voxel cell: {cell}")
        result.add(cell)
    return result


def merge_voxel_cells(cells: Sequence[Sequence[int]]) -> list[VoxelBox]:
    """Greedily merge occupied cells into deterministic non-overlapping boxes."""

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
    total_cells = 0
    for box in boxes:
        if any(length <= 0 for length in box.size_cells):
            raise ValueError(f"voxel box must have positive size: {box}")
        total_cells += box.cell_count
    return total_cells * float(pitch) ** 3
