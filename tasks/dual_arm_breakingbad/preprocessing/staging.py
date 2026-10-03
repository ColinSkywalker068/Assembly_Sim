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
        tolerance = 1e-12
        return (
            self.min_x <= other.min_x + tolerance
            and other.max_x <= self.max_x + tolerance
            and self.min_y <= other.min_y + tolerance
            and other.max_y <= self.max_y + tolerance
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
    pad: AABB2D | None = None
    lane_centers_x: tuple[float, float] | None = None
    gap: float = 0.04
    grid_step: float = 0.01

    def __post_init__(self) -> None:
        if self.gap < 0:
            raise ValueError("staging gap must be non-negative")
        if self.grid_step <= 0:
            raise ValueError("staging grid_step must be positive")
        if (self.pad is None) != (self.lane_centers_x is None):
            raise ValueError("bilateral staging requires both pad and lane_centers_x")
        if self.lane_centers_x is not None:
            if len(self.lane_centers_x) != 2 or not all(
                math.isfinite(value) for value in self.lane_centers_x
            ):
                raise ValueError("lane_centers_x must contain two finite values")
            if self.lane_centers_x[0] >= self.lane_centers_x[1]:
                raise ValueError("lane_centers_x must be ordered left then right")


def _positions(start: float, stop: float, step: float) -> list[float]:
    if stop < start:
        return []
    count = max(0, int(math.floor((stop - start) / step + 1e-9)))
    values = [start + index * step for index in range(count + 1)]
    if values and stop - values[-1] > 1e-9:
        values.append(stop)
    return values


def resolved_lane_centers(
    piece_bounds: Mapping[str, PieceBounds], spec: StagingSpec
) -> tuple[float, float]:
    """Keep the demo lanes unless fragment width would enter the marked region."""

    if spec.pad is None or spec.lane_centers_x is None:
        raise ValueError("bilateral staging is not configured")
    names = sorted(piece_bounds)
    split = (len(names) + 1) // 2
    left_width = max((piece_bounds[name].size[0] for name in names[:split]), default=0.0)
    right_width = max((piece_bounds[name].size[0] for name in names[split:]), default=0.0)
    return (
        min(spec.lane_centers_x[0], spec.pad.min_x - left_width / 2),
        max(spec.lane_centers_x[1], spec.pad.max_x + right_width / 2),
    )


def _bilateral_staging_poses(
    piece_bounds: Mapping[str, PieceBounds], spec: StagingSpec
) -> dict[str, Pose]:
    assert spec.pad is not None
    assert spec.lane_centers_x is not None
    names = sorted(piece_bounds)
    split = (len(names) + 1) // 2
    lane_centers = resolved_lane_centers(piece_bounds, spec)
    lanes = (("left", lane_centers[0], names[:split]), ("right", lane_centers[1], names[split:]))
    occupied: list[AABB2D] = []
    placed: dict[str, Pose] = {}
    for side, lane_x, lane_names in lanes:
        total_depth = sum(piece_bounds[name].size[1] for name in lane_names)
        total_depth += spec.gap * max(0, len(lane_names) - 1)
        centred_start = spec.table.center[1] - total_depth / 2
        starts = _positions(spec.table.min_y, spec.table.max_y - total_depth, spec.grid_step)
        starts = sorted(
            {centred_start, *starts},
            key=lambda start: (abs(start + total_depth / 2 - spec.table.center[1]), start),
        )
        selected_lane = None
        for lane_start in starts:
            cursor = lane_start
            candidates = []
            for name in lane_names:
                bounds = piece_bounds[name]
                width, depth, _ = bounds.size
                candidate = AABB2D(lane_x - width / 2, lane_x + width / 2, cursor, cursor + depth)
                if not spec.table.contains(candidate):
                    break
                if candidate.overlaps(spec.pad):
                    break
                if any(
                    candidate.expanded(spec.gap).overlaps(blocked)
                    for blocked in spec.exclusions
                ):
                    break
                if any(
                    candidate.expanded(spec.gap / 2).overlaps(other.expanded(spec.gap / 2))
                    for other in (*occupied, *candidates)
                ):
                    break
                candidates.append(candidate)
                cursor += depth + spec.gap
            if len(candidates) == len(lane_names):
                selected_lane = candidates
                break
        if selected_lane is None:
            name = lane_names[-1]
            raise ValueError(f"could not place {name}: {side} lane capacity exhausted")
        for name, selected in zip(lane_names, selected_lane):
            bounds = piece_bounds[name]
            placed[name] = Pose(
                (
                    selected.min_x - bounds.local_min[0],
                    selected.min_y - bounds.local_min[1],
                    spec.table.table_top_z - bounds.local_min[2],
                )
            )
            occupied.append(selected)
    return placed


def compute_staging_poses(
    piece_bounds: Mapping[str, PieceBounds], spec: StagingSpec
) -> dict[str, Pose]:
    """Pack fragment AABBs deterministically without intersections."""

    if spec.pad is not None:
        return _bilateral_staging_poses(piece_bounds, spec)

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
