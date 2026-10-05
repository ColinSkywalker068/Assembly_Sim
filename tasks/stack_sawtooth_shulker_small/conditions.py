"""Dataset scene conditions and their conversion into task configurations."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import product

from .config import (
    IDENTITY_WXYZ,
    X_FLIP_WXYZ,
    FragmentConfig,
    Pose,
    StackSawtoothConfig,
    load_task_config,
    validate_task_config,
)
from .geometry import cell_center


COLORS = ("Blue", "Green")
POSE_LABELS = ("C-up", "T-up")
COLOR_TO_FRAGMENT = {"Blue": "FragmentA", "Green": "FragmentB"}
REFERENCE_POSE = {"Blue": "T-up", "Green": "C-up"}


@dataclass(frozen=True)
class FragmentCondition:
    color: str
    pose_label: str
    initial_pose: Pose


@dataclass(frozen=True)
class DemoCondition:
    demonstration_id: str
    demo_seed: int
    kind: str
    group: int | None
    assembly_order: tuple[str, str]
    blue: FragmentCondition
    green: FragmentCondition
    high_color: str
    base_target_xy: tuple[float, float]


def _validated_quaternion(values) -> tuple[float, float, float, float]:
    quaternion = tuple(float(value) for value in values)
    if len(quaternion) != 4 or not all(math.isfinite(value) for value in quaternion):
        raise ValueError("quaternion must contain four finite values")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isclose(norm, 1.0, abs_tol=1e-9):
        raise ValueError("quaternion must have unit length")
    return quaternion


def _rotate_point(
    point: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = orientation_wxyz
    matrix = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    return tuple(sum(row[index] * point[index] for index in range(3)) for row in matrix)


def oriented_fragment_bounds(
    fragment: FragmentConfig,
    unit_size: float,
    orientation_wxyz: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return fragment-local bounds after applying an orientation about its root."""

    unit = float(unit_size)
    if not math.isfinite(unit) or unit <= 0:
        raise ValueError("unit size must be finite and positive")
    quaternion = _validated_quaternion(orientation_wxyz)
    half = unit / 2
    points = []
    for cell in fragment.local_cells:
        center = cell_center(cell, unit)
        for offsets in product((-half, half), repeat=3):
            corner = tuple(center[axis] + offsets[axis] for axis in range(3))
            points.append(_rotate_point(corner, quaternion))
    return (
        tuple(min(point[axis] for point in points) for axis in range(3)),
        tuple(max(point[axis] for point in points) for axis in range(3)),
    )


def _fragment_for_color(config: StackSawtoothConfig, color: str) -> FragmentConfig:
    try:
        return config.fragment(COLOR_TO_FRAGMENT[color])
    except KeyError as exc:
        raise ValueError(f"unknown fragment color: {color}") from exc


def fragment_pose_for_label(
    config: StackSawtoothConfig,
    color: str,
    label: str,
    x: float,
    y: float,
) -> Pose:
    """Return the approved orientation at X/Y with geometry resting on the table."""

    fragment = _fragment_for_color(config, color)
    if label not in POSE_LABELS:
        raise ValueError(f"unknown pose label: {label}")
    orientation = IDENTITY_WXYZ if label == REFERENCE_POSE[color] else X_FLIP_WXYZ
    lower, _ = oriented_fragment_bounds(fragment, config.unit_size, orientation)
    return Pose((float(x), float(y), config.table_top_z - lower[2]), orientation)


def _pose_close(first: Pose, second: Pose, tolerance: float = 1e-9) -> bool:
    return all(
        math.isclose(a, b, abs_tol=tolerance)
        for a, b in zip(
            (*first.position, *first.orientation_wxyz),
            (*second.position, *second.orientation_wxyz),
        )
    )


def _validated_fragment_condition(
    config: StackSawtoothConfig, condition: FragmentCondition
) -> FragmentConfig:
    fragment = _fragment_for_color(config, condition.color)
    x, y, _ = condition.initial_pose.position
    expected = fragment_pose_for_label(config, condition.color, condition.pose_label, x, y)
    if not _pose_close(condition.initial_pose, expected):
        raise ValueError(f"{condition.color} initial pose does not match its pose label")
    return replace(fragment, initial_pose=expected)


def condition_to_task_config(
    condition: DemoCondition,
    base: StackSawtoothConfig | None = None,
) -> StackSawtoothConfig:
    """Convert one immutable dataset condition into a validated scene config."""

    config = base or load_task_config()
    if condition.assembly_order not in (("Blue", "Green"), ("Green", "Blue")):
        raise ValueError(f"invalid assembly order: {condition.assembly_order}")
    expected_labels = ("T-up", "C-up") if condition.assembly_order[0] == "Blue" else ("C-up", "T-up")
    if (condition.blue.pose_label, condition.green.pose_label) != expected_labels:
        raise ValueError("fragments must start in the mating orientations for the chosen base")
    if condition.high_color not in COLORS:
        raise ValueError(f"invalid high color: {condition.high_color}")
    if condition.blue.color != "Blue" or condition.green.color != "Green":
        raise ValueError("fragment condition colors must be Blue and Green")
    target_x, target_y = (float(value) for value in condition.base_target_xy)
    if not all(math.isfinite(value) for value in (target_x, target_y)):
        raise ValueError("base target coordinates must be finite")

    blue = _validated_fragment_condition(config, condition.blue)
    green = _validated_fragment_condition(config, condition.green)
    unit = config.unit_size
    table = config.table_top_z
    if condition.assembly_order[0] == "Blue":
        blue_goal = Pose((target_x, target_y, table), IDENTITY_WXYZ)
        green_goal = Pose((target_x, target_y, table + unit), IDENTITY_WXYZ)
    else:
        green_goal = Pose((target_x, target_y, table + 2 * unit), X_FLIP_WXYZ)
        blue_goal = Pose((target_x, target_y, table + 3 * unit), X_FLIP_WXYZ)
    converted = replace(
        config,
        fragments=(
            replace(blue, goal_pose=blue_goal),
            replace(green, goal_pose=green_goal),
        ),
    )
    validate_task_config(converted)
    return converted
