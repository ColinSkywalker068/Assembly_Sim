"""Immutable configuration for the deterministic initial task scene."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Iterable, Sequence

from core.workcell.config import WorkcellConfig, load_workcell_preset

from .geometry import (
    COMPLETE_CELLS,
    FRAGMENT_A_CELLS,
    FRAGMENT_B_CELLS,
    Cell,
    local_cells,
)


IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)
X_FLIP_WXYZ = (0.0, 1.0, 0.0, 0.0)
SUPPORTED_ORIENTATIONS = (IDENTITY_WXYZ, X_FLIP_WXYZ)
SCALE = 0.027 / 0.08
DEFAULT_UNIT_SIZE = 0.027
PLACEMENT_SQUARE_SIZE = 0.30 * SCALE
DEFAULT_COLLIDER_CLEARANCE = 0.002
DEFAULT_DENSITY = 20.0
MINIMUM_STATIC_CLEARANCE = 0.04

# Measured from the checked-in CRX-10iA/L base_link bounds after applying the
# right preset's 180-degree yaw. This is the nearest physical base edge to the
# target, not the deliberately conservative staging exclusion used elsewhere.
RIGHT_BASE_NEAR_X = 0.685


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float] = IDENTITY_WXYZ


@dataclass(frozen=True)
class FragmentConfig:
    name: str
    color: tuple[float, float, float]
    canonical_cells: tuple[Cell, ...]
    local_cells: tuple[Cell, ...]
    canonical_z_offset: int
    initial_pose: Pose
    goal_pose: Pose


@dataclass(frozen=True)
class StackSawtoothConfig:
    workcell: WorkcellConfig
    unit_size: float
    collider_clearance: float
    density: float
    target_center: tuple[float, float, float]
    target_size: tuple[float, float]
    table_bounds: tuple[float, float, float, float]
    table_top_z: float
    robot_base_near_x: float
    minimum_static_clearance: float
    fragments: tuple[FragmentConfig, ...]

    def fragment(self, name: str) -> FragmentConfig:
        for fragment in self.fragments:
            if fragment.name == name:
                return fragment
        raise KeyError(f"unknown fragment: {name}")


def _fragment(
    name: str,
    color: tuple[float, float, float],
    canonical_cells: tuple[Cell, ...],
    initial_position: tuple[float, float, float],
    goal_position: tuple[float, float, float],
) -> FragmentConfig:
    z_offset, localized = local_cells(canonical_cells)
    return FragmentConfig(
        name=name,
        color=color,
        canonical_cells=canonical_cells,
        local_cells=localized,
        canonical_z_offset=z_offset,
        initial_pose=Pose(initial_position),
        goal_pose=Pose(goal_position),
    )


def load_task_config(workcell: WorkcellConfig | None = None) -> StackSawtoothConfig:
    """Load the task on top of the immutable single-right-arm workcell."""

    workcell = workcell or load_workcell_preset("single_arm_right")
    # Isolate the scaled task target from the shared workcell and original task.
    data = deepcopy(workcell.data)
    data["render"].update(multi_gpu=False, max_gpu_count=1)
    scaled_pad = data["environment"]["assembly_pad"]
    scaled_pad["size"] = [value * SCALE for value in scaled_pad["size"]]
    for key in ("tape_width", "grid_step"):
        scaled_pad[key] *= SCALE
    scaled_pad["lane_centers_x"] = [value * SCALE for value in scaled_pad["lane_centers_x"]]
    workcell = replace(workcell, data=data)
    pad = workcell.environment["assembly_pad"]
    target_center = tuple(float(value) for value in pad["center"])
    target_size = tuple(float(value) for value in pad["size"][:2])
    table_position = tuple(float(value) for value in workcell.environment["table_position"])
    table_size = tuple(float(value) for value in workcell.environment["table_size"])
    margin = float(workcell.environment.get("workspace_margin", 0.0))
    table_top = table_position[2] + table_size[2] / 2
    table_bounds = (
        table_position[0] - table_size[0] / 2 + margin,
        table_position[0] + table_size[0] / 2 - margin,
        table_position[1] - table_size[1] / 2 + margin,
        table_position[1] + table_size[1] / 2 - margin,
    )
    unit = DEFAULT_UNIT_SIZE
    config = StackSawtoothConfig(
        workcell=workcell,
        unit_size=unit,
        collider_clearance=DEFAULT_COLLIDER_CLEARANCE,
        density=DEFAULT_DENSITY,
        target_center=target_center,
        target_size=target_size,
        table_bounds=table_bounds,
        table_top_z=table_top,
        robot_base_near_x=RIGHT_BASE_NEAR_X,
        minimum_static_clearance=MINIMUM_STATIC_CLEARANCE,
        fragments=(
            _fragment(
                "FragmentA",
                (0.08, 0.28, 0.85),
                FRAGMENT_A_CELLS,
                (0.48, -0.185, table_top),
                (target_center[0], target_center[1], table_top),
            ),
            _fragment(
                "FragmentB",
                (0.12, 0.65, 0.25),
                FRAGMENT_B_CELLS,
                (0.48, 0.185, table_top),
                (target_center[0], target_center[1], table_top + unit),
            ),
        ),
    )
    validate_task_config(config)
    return config


def _finite_tuple(values: Sequence[float], length: int, field: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field} must contain {length} finite values")
    return result


def _validated_cells(values: Iterable[Sequence[int]], name: str) -> tuple[Cell, ...]:
    cells: list[Cell] = []
    for raw in values:
        if len(raw) != 3 or any(
            not isinstance(value, Integral) or isinstance(value, bool) for value in raw
        ):
            raise ValueError(f"fragment {name} cell must be an integer triplet")
        cell = tuple(int(value) for value in raw)
        if any(value not in range(3) for value in cell):
            raise ValueError(f"fragment {name} cell is outside the 3-by-3-by-3 grid: {cell}")
        cells.append(cell)
    if not cells:
        raise ValueError(f"fragment {name} cells must not be empty")
    if len(cells) != len(set(cells)):
        raise ValueError(f"fragment {name} contains a duplicate cell")
    return tuple(cells)


def _validate_pose(pose: Pose, field: str) -> None:
    _finite_tuple(pose.position, 3, f"{field} position")
    quaternion = _finite_tuple(pose.orientation_wxyz, 4, f"{field} orientation")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isclose(norm, 1.0, abs_tol=1e-9):
        raise ValueError(f"{field} orientation must be a unit quaternion")
    if not any(
        all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(quaternion, allowed))
        for allowed in SUPPORTED_ORIENTATIONS
    ):
        raise ValueError(f"{field} orientation must be identity or the local-X flip")


def _world_bounds(fragment: FragmentConfig, unit_size: float, pose: Pose):
    from tasks.stack_sawtooth_shulker.conditions import oriented_fragment_bounds

    local_min, local_max = oriented_fragment_bounds(
        fragment, unit_size, pose.orientation_wxyz
    )
    x, y, z = pose.position
    return (
        (x + local_min[0], y + local_min[1], z + local_min[2]),
        (x + local_max[0], y + local_max[1], z + local_max[2]),
    )


def _xy_bounds(bounds) -> tuple[float, float, float, float]:
    lower, upper = bounds
    return (
        lower[0],
        upper[0],
        lower[1],
        upper[1],
    )


def _overlaps(first, second, tolerance: float = 1e-12) -> bool:
    return not (
        first[1] <= second[0] + tolerance
        or second[1] <= first[0] + tolerance
        or first[3] <= second[2] + tolerance
        or second[3] <= first[2] + tolerance
    )


def validate_task_config(config: StackSawtoothConfig) -> None:
    """Reject scene geometry that violates the deterministic task contract."""

    unit = float(config.unit_size)
    if not math.isfinite(unit) or unit <= 0:
        raise ValueError("unit size must be finite and positive")
    clearance = float(config.collider_clearance)
    if not math.isfinite(clearance) or clearance < 0 or clearance >= unit:
        raise ValueError("collider clearance must be finite and satisfy 0 <= clearance < unit size")
    density = float(config.density)
    if not math.isfinite(density) or density <= 0:
        raise ValueError("density must be finite and positive")
    target = _finite_tuple(config.target_size, 2, "target size")
    if not math.isclose(target[0], target[1], abs_tol=1e-12):
        raise ValueError("target must be square")
    if target[0] < 3 * unit - 1e-12:
        raise ValueError("target side must contain the assembled three-unit cube")
    _finite_tuple(config.target_center, 3, "target center")
    table = _finite_tuple(config.table_bounds, 4, "table bounds")
    if not (table[0] < table[1] and table[2] < table[3]):
        raise ValueError("table bounds must be increasing")
    if len(config.fragments) != 2:
        raise ValueError("task requires exactly two fragments")

    owners: dict[Cell, str] = {}
    for fragment in config.fragments:
        if not fragment.name:
            raise ValueError("fragment name must not be empty")
        cells = _validated_cells(fragment.canonical_cells, fragment.name)
        expected_offset, expected_local = local_cells(cells)
        if (
            fragment.canonical_z_offset != expected_offset
            or tuple(fragment.local_cells) != expected_local
        ):
            raise ValueError(f"fragment {fragment.name} local mapping is inconsistent")
        color = _finite_tuple(fragment.color, 3, f"fragment {fragment.name} color")
        if any(value < 0 or value > 1 for value in color):
            raise ValueError(f"fragment {fragment.name} color must lie in [0, 1]")
        for cell in cells:
            if cell in owners:
                raise ValueError(
                    f"fragment cells overlap at {cell}: {owners[cell]}, {fragment.name}"
                )
            owners[cell] = fragment.name
        _validate_pose(fragment.initial_pose, f"fragment {fragment.name} initial pose")
        _validate_pose(fragment.goal_pose, f"fragment {fragment.name} goal pose")
    if set(owners) != set(COMPLETE_CELLS):
        raise ValueError("fragment cell partition does not reconstruct the complete cube")

    target_bounds = (
        config.target_center[0] - target[0] / 2,
        config.target_center[0] + target[0] / 2,
        config.target_center[1] - target[1] / 2,
        config.target_center[1] + target[1] / 2,
    )
    world_bounds = {}
    for fragment in config.fragments:
        bounds_3d = _world_bounds(fragment, unit, fragment.initial_pose)
        if not math.isclose(bounds_3d[0][2], config.table_top_z, abs_tol=1e-9):
            raise ValueError(f"fragment {fragment.name} must rest on the table")
        bounds = _xy_bounds(bounds_3d)
        world_bounds[fragment.name] = bounds
        if not (
            table[0] <= bounds[0]
            and bounds[1] <= table[1]
            and table[2] <= bounds[2]
            and bounds[3] <= table[3]
        ):
            raise ValueError(f"fragment {fragment.name} lies outside the usable table")
        if _overlaps(bounds, target_bounds):
            raise ValueError(f"fragment {fragment.name} overlaps the target region")
        if bounds[0] - target_bounds[1] < config.minimum_static_clearance - 1e-12:
            raise ValueError(f"fragment {fragment.name} lacks target clearance")
        if config.robot_base_near_x - bounds[1] < config.minimum_static_clearance - 1e-12:
            raise ValueError(f"fragment {fragment.name} lacks robot-base clearance")
    first, second = (world_bounds[fragment.name] for fragment in config.fragments)
    if _overlaps(first, second):
        raise ValueError("initial fragment placements overlap")
    y_gap = max(second[2] - first[3], first[2] - second[3])
    if y_gap < config.minimum_static_clearance - 1e-12:
        raise ValueError("initial fragment placements lack required separation")

    by_name = {fragment.name: fragment for fragment in config.fragments}
    a_goal = by_name["FragmentA"].goal_pose
    b_goal = by_name["FragmentB"].goal_pose
    target_x, target_y = a_goal.position[:2]
    if not (
        math.isclose(b_goal.position[0], target_x, abs_tol=1e-9)
        and math.isclose(b_goal.position[1], target_y, abs_tol=1e-9)
    ):
        raise ValueError("fragment goal poses must share the assembled cube center")
    canonical = (
        a_goal.orientation_wxyz == IDENTITY_WXYZ
        and b_goal.orientation_wxyz == IDENTITY_WXYZ
        and math.isclose(a_goal.position[2], config.table_top_z, abs_tol=1e-9)
        and math.isclose(b_goal.position[2], config.table_top_z + unit, abs_tol=1e-9)
    )
    inverted = (
        a_goal.orientation_wxyz == X_FLIP_WXYZ
        and b_goal.orientation_wxyz == X_FLIP_WXYZ
        and math.isclose(a_goal.position[2], config.table_top_z + 3 * unit, abs_tol=1e-9)
        and math.isclose(b_goal.position[2], config.table_top_z + 2 * unit, abs_tol=1e-9)
    )
    if not (canonical or inverted):
        raise ValueError("fragment goal poses must reconstruct the canonical or inverted cube")
    half_cube = 1.5 * unit
    if not (
        target_bounds[0] <= target_x - half_cube
        and target_x + half_cube <= target_bounds[1]
        and target_bounds[2] <= target_y - half_cube
        and target_y + half_cube <= target_bounds[3]
    ):
        raise ValueError("assembled cube goal lies outside the target region")
