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


@dataclass(frozen=True)
class TapeSegment:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


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


def assembly_tape_segments(spec: Mapping[str, Sequence[float] | float]) -> tuple[TapeSegment, ...]:
    geometry = assembly_pad_geometry(spec)
    width = float(spec["tape_width"])
    x_size, y_size, thickness = geometry.size
    if not math.isfinite(width) or width <= 0 or 2 * width >= min(x_size, y_size):
        raise ValueError("assembly tape width must fit inside the workspace footprint")

    x, y, z = geometry.center
    tape_z = z + thickness / 2
    x_offset = (x_size - width) / 2
    y_offset = (y_size - width) / 2
    return (
        TapeSegment("Top", (x, y + y_offset, tape_z), (x_size, width, thickness)),
        TapeSegment("Bottom", (x, y - y_offset, tape_z), (x_size, width, thickness)),
        TapeSegment("Left", (x - x_offset, y, tape_z), (width, y_size - 2 * width, thickness)),
        TapeSegment("Right", (x + x_offset, y, tape_z), (width, y_size - 2 * width, thickness)),
    )
