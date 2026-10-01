"""Plain voxel-cube visual mesh construction."""

from __future__ import annotations

from typing import Sequence

import numpy as np


Cell = tuple[int, int, int]
_TRIANGLES = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)


def build_voxel_surface(
    cells: Sequence[Sequence[int]],
    pitch: float,
    pivot_cells: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build exposed cube faces for exactly the supplied occupied cells."""

    if pitch <= 0:
        raise ValueError("pitch must be positive")
    occupied: set[Cell] = set()
    for raw in cells:
        if len(raw) != 3 or any(not isinstance(value, (int, np.integer)) for value in raw):
            raise ValueError("voxel cells must be integer triplets")
        cell = tuple(int(value) for value in raw)
        if cell in occupied:
            raise ValueError(f"duplicate voxel cell: {cell}")
        occupied.add(cell)
    if not occupied:
        raise ValueError("cannot build a visual mesh without occupied cells")
    pivot = np.asarray(tuple(float(value) for value in pivot_cells), dtype=float)
    if pivot.shape != (3,):
        raise ValueError("pivot_cells must contain three values")

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    face_specs = (
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        ((-1, 0, 0), ((0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1))),
        ((0, 1, 0), ((1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1))),
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
        ((0, 0, -1), ((0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0))),
    )
    for i, j, k in sorted(occupied):
        for (di, dj, dk), corners in face_specs:
            if (i + di, j + dj, k + dk) in occupied:
                continue
            base = len(vertices)
            for x, y, z in corners:
                cell_corner = np.asarray((i + x, j + y, k + z), dtype=float)
                local = (cell_corner - pivot) * pitch
                vertices.append(tuple(float(value) for value in local))
            faces.extend(tuple(int(value) for value in row + base) for row in _TRIANGLES)
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)
