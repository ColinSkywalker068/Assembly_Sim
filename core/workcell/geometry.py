"""Pure geometry contracts for the invariant workcell."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


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


def assembly_pad_geometry(spec: Mapping[str, Sequence[float]]) -> AssemblyPadGeometry:
    center = tuple(float(value) for value in spec["center"])
    size = tuple(float(value) for value in spec["size"])
    if (
        len(center) != 3
        or len(size) != 3
        or not all(math.isfinite(value) for value in (*center, *size))
        or any(value <= 0 for value in size)
    ):
        raise ValueError("assembly pad center and size must be finite positive 3D values")
    return AssemblyPadGeometry(center=center, size=size)
