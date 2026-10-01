"""Deterministic AABB staging for voxel fragments on a worktable."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class AABB2D:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @classmethod
    def from_center_extent(cls, center: Sequence[float], extent: Sequence[float]) -> "AABB2D":
        return cls(
            float(center[0]) - float(extent[0]) / 2,
            float(center[0]) + float(extent[0]) / 2,
            float(center[1]) - float(extent[1]) / 2,
            float(center[1]) + float(extent[1]) / 2,
        )

    @property
    def center(self) -> tuple[float, float]:
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    def expanded(self, amount: float) -> "AABB2D":
        return AABB2D(
            self.min_x - amount,
            self.max_x + amount,
            self.min_y - amount,
            self.max_y + amount,
        )

    def overlaps(self, other: "AABB2D") -> bool:
        tolerance = 1e-12
        return not (
            self.max_x <= other.min_x + tolerance
            or other.max_x <= self.min_x + tolerance
            or self.max_y <= other.min_y + tolerance
            or other.max_y <= self.min_y + tolerance
        )


@dataclass(frozen=True)
class TableBounds(AABB2D):
    table_top_z: float

    def contains(self, other: AABB2D) -> bool:
        return (
            self.min_x <= other.min_x
            and other.max_x <= self.max_x
            and self.min_y <= other.min_y
            and other.max_y <= self.max_y
        )


@dataclass(frozen=True)
class PieceBounds:
    local_min: tuple[float, float, float]
    local_max: tuple[float, float, float]

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(high - low for low, high in zip(self.local_min, self.local_max))

    def world_aabb(self, pose: Pose) -> AABB2D:
        return AABB2D(
            pose.position[0] + self.local_min[0],
            pose.position[0] + self.local_max[0],
            pose.position[1] + self.local_min[1],
            pose.position[1] + self.local_max[1],
        )


@dataclass(frozen=True)
class StagingSpec:
    table: TableBounds
    exclusions: tuple[AABB2D, ...] = ()
    gap: float = 0.04
    grid_step: float = 0.01

    def __post_init__(self) -> None:
        if self.gap < 0:
            raise ValueError("staging gap must be non-negative")
        if self.grid_step <= 0:
            raise ValueError("staging grid_step must be positive")


def _positions(start: float, stop: float, step: float) -> list[float]:
    if stop < start:
        return []
    count = max(0, int(math.floor((stop - start) / step + 1e-9)))
    values = [start + index * step for index in range(count + 1)]
    if values and stop - values[-1] > 1e-9:
        values.append(stop)
    return values


def compute_staging_poses(
    piece_bounds: Mapping[str, PieceBounds], spec: StagingSpec
) -> dict[str, Pose]:
    """Pack fragment AABBs deterministically without intersections."""

    anchor = spec.exclusions[0].center if spec.exclusions else spec.table.center
    occupied: list[AABB2D] = []
    placed: dict[str, Pose] = {}
    order = sorted(
        piece_bounds,
        key=lambda name: (
            -(piece_bounds[name].size[0] * piece_bounds[name].size[1]),
            name,
        ),
    )
    for name in order:
        bounds = piece_bounds[name]
        width, depth, _ = bounds.size
        candidates = []
        for min_y in _positions(spec.table.min_y, spec.table.max_y - depth, spec.grid_step):
            for min_x in _positions(spec.table.min_x, spec.table.max_x - width, spec.grid_step):
                candidate = AABB2D(min_x, min_x + width, min_y, min_y + depth)
                cx, cy = candidate.center
                candidates.append(((cx - anchor[0]) ** 2 + (cy - anchor[1]) ** 2, cy, cx, candidate))
        candidates.sort(key=lambda item: item[:3])
        selected = None
        for _, _, _, candidate in candidates:
            if any(candidate.expanded(spec.gap).overlaps(blocked) for blocked in spec.exclusions):
                continue
            if any(
                candidate.expanded(spec.gap / 2).overlaps(other.expanded(spec.gap / 2))
                for other in occupied
            ):
                continue
            selected = candidate
            break
        if selected is None:
            raise ValueError(
                f"could not place {name}: table capacity exhausted with {spec.gap:.3f} m gap"
            )
        placed[name] = Pose(
            (
                selected.min_x - bounds.local_min[0],
                selected.min_y - bounds.local_min[1],
                spec.table.table_top_z - bounds.local_min[2],
            )
        )
        occupied.append(selected)
    return placed


PieceExtent = PieceBounds
