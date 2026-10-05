"""Deterministic dataset manifests for the Shulker assembly task."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .conditions import (
    COLORS,
    POSE_LABELS,
    DemoCondition,
    FragmentCondition,
    condition_to_task_config,
    fragment_pose_for_label,
)
from .config import Pose, load_task_config, PLACEMENT_SQUARE_SIZE


SCHEMA_VERSION = 1
GENERATOR_VERSION = "stack-sawtooth-small-mating-scenes-v1"
TASK_NAME = "stack_sawtooth_shulker_small"
UINT64_MAX = 2**64 - 1

GROUPS = (
    (1, ("Blue", "Green"), "T-up", "C-up"),
    (2, ("Green", "Blue"), "C-up", "T-up"),
)


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: int
    generator_version: str
    task_name: str
    dataset_seed: int
    unit_size: float
    target_center_xy: tuple[float, float]
    target_size: tuple[float, float]
    placement_square_size: float
    initial_x: float
    high_y_range: tuple[float, float]
    low_y_range: tuple[float, float]
    conditions: tuple[DemoCondition, ...]


def _validated_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= UINT64_MAX:
        raise ValueError("dataset seed must be an unsigned 64-bit integer")
    return seed


def _digest_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def derive_demo_seed(dataset_seed: int, demo_id: str) -> int:
    seed = _validated_seed(dataset_seed)
    if not isinstance(demo_id, str) or not demo_id:
        raise ValueError("demonstration ID must be a nonempty string")
    return _digest_seed(f"stack_sawtooth_shulker_small-dataset-v1:{seed}:{demo_id}")


def _condition(
    demo_id: str,
    dataset_seed: int,
    kind: str,
    group: int | None,
    order: tuple[str, str],
    blue_label: str,
    green_label: str,
    high_color: str,
    blue_y: float,
    green_y: float,
    target_xy: tuple[float, float],
) -> DemoCondition:
    config = load_task_config()
    return DemoCondition(
        demonstration_id=demo_id,
        demo_seed=derive_demo_seed(dataset_seed, demo_id),
        kind=kind,
        group=group,
        assembly_order=order,
        blue=FragmentCondition(
            "Blue",
            blue_label,
            fragment_pose_for_label(config, "Blue", blue_label, 0.48, blue_y),
        ),
        green=FragmentCondition(
            "Green",
            green_label,
            fragment_pose_for_label(config, "Green", green_label, 0.48, green_y),
        ),
        high_color=high_color,
        base_target_xy=target_xy,
    )


def generate_manifest(dataset_seed: int) -> DatasetManifest:
    seed = _validated_seed(dataset_seed)
    config = load_task_config()
    conditions = [
        _condition(
            f"demo_{index:03d}",
            seed,
            "standard",
            None,
            ("Blue", "Green"),
            "T-up",
            "C-up",
            "Green",
            -0.185,
            0.185,
            (0.0, 0.0),
        )
        for index in range(2)
    ]
    next_index = 2
    for group, order, blue_label, green_label in GROUPS:
        high_colors = ["Blue"] * 12 + ["Green"] * 12
        group_seed = _digest_seed(f"stack_sawtooth_shulker_small-group-v1:{seed}:{group}")
        random.Random(group_seed).shuffle(high_colors)
        for high_color in high_colors:
            demo_id = f"demo_{next_index:03d}"
            rng = random.Random(derive_demo_seed(seed, demo_id))
            high_y = rng.uniform(0.185, 0.335)
            low_y = rng.uniform(-0.335, -0.185)
            blue_y, green_y = (
                (high_y, low_y) if high_color == "Blue" else (low_y, high_y)
            )
            while True:
                target_xy = (
                    rng.uniform(0.0, PLACEMENT_SQUARE_SIZE / 2) - rng.uniform(0.0, PLACEMENT_SQUARE_SIZE / 2),
                    rng.uniform(0.0, PLACEMENT_SQUARE_SIZE / 2) - rng.uniform(0.0, PLACEMENT_SQUARE_SIZE / 2),
                )
                if target_xy != (0.0, 0.0):
                    break
            conditions.append(
                _condition(
                    demo_id,
                    seed,
                    "stratified",
                    group,
                    order,
                    blue_label,
                    green_label,
                    high_color,
                    blue_y,
                    green_y,
                    target_xy,
                )
            )
            next_index += 1
    return DatasetManifest(
        SCHEMA_VERSION,
        GENERATOR_VERSION,
        TASK_NAME,
        seed,
        config.unit_size,
        tuple(config.target_center[:2]),
        config.target_size,
        PLACEMENT_SQUARE_SIZE,
        0.48,
        (0.185, 0.335),
        (-0.335, -0.185),
        tuple(conditions),
    )


def validate_manifest(manifest: DatasetManifest) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {manifest.schema_version}")
    if manifest.generator_version != GENERATOR_VERSION:
        raise ValueError(f"unsupported generator version: {manifest.generator_version}")
    if manifest.task_name != TASK_NAME:
        raise ValueError(f"unexpected task name: {manifest.task_name}")
    seed = _validated_seed(manifest.dataset_seed)
    expected_constants = (
        (manifest.unit_size, 0.027, "unit size"),
        (manifest.placement_square_size, PLACEMENT_SQUARE_SIZE, "placement square size"),
        (manifest.initial_x, 0.48, "initial X"),
    )
    for actual, expected, field in expected_constants:
        if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=1e-12):
            raise ValueError(f"invalid {field}: {actual}")
    if manifest.target_center_xy != (0.0, 0.0) or manifest.target_size != load_task_config().target_size:
        raise ValueError("invalid target geometry")
    if manifest.high_y_range != (0.185, 0.335):
        raise ValueError("invalid high Y range")
    if manifest.low_y_range != (-0.335, -0.185):
        raise ValueError("invalid low Y range")
    if len(manifest.conditions) != 50:
        raise ValueError("manifest must contain exactly 50 demonstrations")
    expected_ids = tuple(f"demo_{index:03d}" for index in range(50))
    actual_ids = tuple(entry.demonstration_id for entry in manifest.conditions)
    if actual_ids != expected_ids:
        raise ValueError("demonstration IDs must be unique and sequential")
    seeds = tuple(entry.demo_seed for entry in manifest.conditions)
    if len(set(seeds)) != len(seeds):
        raise ValueError("demonstration seeds must be unique")
    for entry in manifest.conditions:
        expected_seed = derive_demo_seed(seed, entry.demonstration_id)
        if entry.demo_seed != expected_seed:
            raise ValueError(
                f"demonstration seed mismatch for {entry.demonstration_id}"
            )

    for index, entry in enumerate(manifest.conditions[:2]):
        expected = _condition(
            f"demo_{index:03d}",
            seed,
            "standard",
            None,
            ("Blue", "Green"),
            "T-up",
            "C-up",
            "Green",
            -0.185,
            0.185,
            (0.0, 0.0),
        )
        if entry != expected:
            raise ValueError(f"canonical standard condition changed: {entry.demonstration_id}")

    group_specs = {group: (order, blue, green) for group, order, blue, green in GROUPS}
    group_counts = Counter()
    group_high_counts: dict[int, Counter] = {group: Counter() for group in group_specs}
    global_counts = {
        "order": Counter(),
        "blue_pose": Counter(),
        "green_pose": Counter(),
        "high_color": Counter(),
    }
    for entry in manifest.conditions[2:]:
        if entry.kind != "stratified":
            raise ValueError(f"invalid condition kind for {entry.demonstration_id}")
        if entry.group not in group_specs:
            raise ValueError(f"invalid group for {entry.demonstration_id}: {entry.group}")
        expected_order, expected_blue, expected_green = group_specs[entry.group]
        if (
            entry.assembly_order != expected_order
            or entry.blue.pose_label != expected_blue
            or entry.green.pose_label != expected_green
        ):
            raise ValueError(f"condition does not match group {entry.group}")
        if entry.blue.color != "Blue" or entry.green.color != "Green":
            raise ValueError("fragment colors must be Blue and Green")
        if entry.blue.pose_label not in POSE_LABELS or entry.green.pose_label not in POSE_LABELS:
            raise ValueError("invalid pose label")
        if entry.high_color not in COLORS:
            raise ValueError(f"invalid high color: {entry.high_color}")
        blue_x, blue_y, _ = entry.blue.initial_pose.position
        green_x, green_y, _ = entry.green.initial_pose.position
        if not (
            math.isclose(blue_x, manifest.initial_x, abs_tol=1e-12)
            and math.isclose(green_x, manifest.initial_x, abs_tol=1e-12)
        ):
            raise ValueError("fragment initial X must remain fixed")
        high_y = blue_y if entry.high_color == "Blue" else green_y
        low_y = green_y if entry.high_color == "Blue" else blue_y
        if not manifest.high_y_range[0] <= high_y <= manifest.high_y_range[1]:
            raise ValueError(f"high Y is outside its range for {entry.demonstration_id}")
        if not manifest.low_y_range[0] <= low_y <= manifest.low_y_range[1]:
            raise ValueError(f"low Y is outside its range for {entry.demonstration_id}")
        if high_y - low_y < 0.37 - 1e-12:
            raise ValueError("fragment Y separation is below the standard separation")
        if len(entry.base_target_xy) != 2 or not all(
            math.isfinite(value) for value in entry.base_target_xy
        ):
            raise ValueError("base target must contain two finite coordinates")
        if not all(-PLACEMENT_SQUARE_SIZE / 2 <= value <= PLACEMENT_SQUARE_SIZE / 2 for value in entry.base_target_xy):
            raise ValueError("base target lies outside the placement square")
        if entry.base_target_xy == (0.0, 0.0):
            raise ValueError("stratified base target must not equal the target center")
        condition_to_task_config(entry)
        group_counts[entry.group] += 1
        group_high_counts[entry.group][entry.high_color] += 1
        global_counts["order"][entry.assembly_order[0]] += 1
        global_counts["blue_pose"][entry.blue.pose_label] += 1
        global_counts["green_pose"][entry.green.pose_label] += 1
        global_counts["high_color"][entry.high_color] += 1
    for group in group_specs:
        if group_counts[group] != 24:
            raise ValueError(f"group {group} must contain exactly 24 demonstrations")
        if group_high_counts[group] != Counter({"Blue": 12, "Green": 12}):
            raise ValueError(f"group {group} must have exact high-color balance")
    for field in ("order", "high_color"):
        counts = global_counts[field]
        if sorted(counts.values()) != [24, 24]:
            raise ValueError(f"global {field} balance must be 24/24")


def condition_to_dict(condition: DemoCondition) -> dict:
    return asdict(condition)


def manifest_to_dict(manifest: DatasetManifest) -> dict:
    return asdict(manifest)


def _exact_keys(value: dict, expected: set[str], field: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{field} fields mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _pose_from_dict(value: dict, field: str) -> Pose:
    _exact_keys(value, {"position", "orientation_wxyz"}, field)
    try:
        return Pose(tuple(value["position"]), tuple(value["orientation_wxyz"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}") from exc


def _fragment_from_dict(value: dict, field: str) -> FragmentCondition:
    _exact_keys(value, {"color", "pose_label", "initial_pose"}, field)
    return FragmentCondition(
        value["color"],
        value["pose_label"],
        _pose_from_dict(value["initial_pose"], f"{field}.initial_pose"),
    )


def _condition_from_dict(value: dict, index: int) -> DemoCondition:
    field = f"conditions[{index}]"
    _exact_keys(
        value,
        {
            "demonstration_id",
            "demo_seed",
            "kind",
            "group",
            "assembly_order",
            "blue",
            "green",
            "high_color",
            "base_target_xy",
        },
        field,
    )
    return DemoCondition(
        value["demonstration_id"],
        value["demo_seed"],
        value["kind"],
        value["group"],
        tuple(value["assembly_order"]),
        _fragment_from_dict(value["blue"], f"{field}.blue"),
        _fragment_from_dict(value["green"], f"{field}.green"),
        value["high_color"],
        tuple(value["base_target_xy"]),
    )


def _manifest_from_dict(value: dict) -> DatasetManifest:
    if not isinstance(value, dict):
        raise ValueError("manifest JSON root must be an object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {value.get('schema_version')}")
    expected = {
        "schema_version",
        "generator_version",
        "task_name",
        "dataset_seed",
        "unit_size",
        "target_center_xy",
        "target_size",
        "placement_square_size",
        "initial_x",
        "high_y_range",
        "low_y_range",
        "conditions",
    }
    _exact_keys(value, expected, "manifest")
    try:
        conditions = tuple(
            _condition_from_dict(entry, index)
            for index, entry in enumerate(value["conditions"])
        )
        return DatasetManifest(
            value["schema_version"],
            value["generator_version"],
            value["task_name"],
            value["dataset_seed"],
            value["unit_size"],
            tuple(value["target_center_xy"]),
            tuple(value["target_size"]),
            value["placement_square_size"],
            value["initial_x"],
            tuple(value["high_y_range"]),
            tuple(value["low_y_range"]),
            conditions,
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("invalid manifest JSON structure") from exc


def write_manifest(manifest: DatasetManifest, path: Path) -> Path:
    validate_manifest(manifest)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"manifest output already exists: {destination}")
    payload = json.dumps(
        manifest_to_dict(manifest), sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"manifest output already exists: {destination}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_manifest(path: Path) -> DatasetManifest:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid manifest JSON: {exc.msg}") from exc
    manifest = _manifest_from_dict(value)
    validate_manifest(manifest)
    return manifest
