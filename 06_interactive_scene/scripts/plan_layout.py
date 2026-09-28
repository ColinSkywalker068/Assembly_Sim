"""Deterministic staging and offline reachability checks for one FANUC arm."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from scene_config import SceneConfig


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
        return not (
            self.max_x <= other.min_x
            or other.max_x <= self.min_x
            or self.max_y <= other.min_y
            or other.max_y <= self.min_y
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
class PieceExtent:
    local_min: tuple[float, float, float]
    local_max: tuple[float, float, float]

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(high - low for low, high in zip(self.local_min, self.local_max))

    def world_aabb(self, pose: Pose) -> AABB2D:
        x, y, _ = pose.position
        return AABB2D(
            x + self.local_min[0],
            x + self.local_max[0],
            y + self.local_min[1],
            y + self.local_max[1],
        )


@dataclass(frozen=True)
class ReachabilityTarget:
    name: str
    reachable: bool
    joints: tuple[float, ...]
    position_error: float
    rotation_error: float
    joint_margin: float


@dataclass(frozen=True)
class ReachabilityReport:
    targets: Mapping[str, ReachabilityTarget]
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]
    probe_path: str

    @property
    def all_reachable(self) -> bool:
        return bool(self.targets) and all(target.reachable for target in self.targets.values())


def _layout(config: SceneConfig) -> dict:
    return json.loads(config.resolve_repo_path("layout_json").read_text(encoding="utf-8"))


def load_piece_extents(config: SceneConfig) -> dict[str, PieceExtent]:
    layout = _layout(config)
    pitch = float(layout["pitch"])
    result: dict[str, PieceExtent] = {}
    for piece in layout["pieces"]:
        cells = [tuple(int(value) for value in cell) for cell in piece["cells"]]
        min_cell = tuple(min(cell[axis] for cell in cells) for axis in range(3))
        max_cell = tuple(max(cell[axis] for cell in cells) for axis in range(3))
        fp_x, fp_y = (float(value) for value in piece["fp_cell"])
        local_min = (
            (min_cell[0] - fp_x) * pitch,
            (min_cell[1] - fp_y) * pitch,
            min_cell[2] * pitch,
        )
        local_max = (
            (max_cell[0] + 1 - fp_x) * pitch,
            (max_cell[1] + 1 - fp_y) * pitch,
            (max_cell[2] + 1) * pitch,
        )
        result[str(piece["name"])] = PieceExtent(local_min, local_max)
    return result


def table_bounds(config: SceneConfig) -> TableBounds:
    environment = config.data["environment"]
    position = [float(value) for value in environment["table_position"]]
    size = [float(value) for value in environment["table_size"]]
    margin = float(environment.get("workspace_margin", 0.0))
    return TableBounds(
        position[0] - size[0] / 2 + margin,
        position[0] + size[0] / 2 - margin,
        position[1] - size[1] / 2 + margin,
        position[1] + size[1] / 2 - margin,
        position[2] + size[2] / 2,
    )


def scene_exclusions(config: SceneConfig) -> list[AABB2D]:
    robot_position = config.data["robot"]["base_position"]
    robot = AABB2D.from_center_extent(robot_position[:2], (0.34, 0.34))
    layout = _layout(config)
    plate_position = config.data["environment"]["plate_position"]
    plate = AABB2D.from_center_extent(plate_position[:2], layout["plate"]["size"][:2])
    return [robot, plate]


def _positions(start: float, stop: float, step: float) -> list[float]:
    count = max(0, int(math.floor((stop - start) / step)))
    return [start + index * step for index in range(count + 1)]


def compute_staging_poses(
    piece_extents: Mapping[str, PieceExtent],
    bounds: TableBounds,
    exclusions: Sequence[AABB2D],
    gap: float,
) -> dict[str, Pose]:
    """Pack pieces near the robot exclusion while keeping collision-safe gaps."""

    if gap < 0:
        raise ValueError("staging gap must be non-negative")
    anchor = exclusions[0].center if exclusions else (bounds.min_x, bounds.min_y)
    placed: dict[str, Pose] = {}
    occupied: list[AABB2D] = []
    order = sorted(
        piece_extents,
        key=lambda name: (
            -(piece_extents[name].size[0] * piece_extents[name].size[1]),
            name,
        ),
    )
    step = 0.01
    for name in order:
        extent = piece_extents[name]
        sx, sy, _ = extent.size
        candidates: list[tuple[float, float, AABB2D]] = []
        for min_y in _positions(bounds.min_y, bounds.max_y - sy, step):
            for min_x in _positions(bounds.min_x, bounds.max_x - sx, step):
                candidate = AABB2D(min_x, min_x + sx, min_y, min_y + sy)
                cx, cy = candidate.center
                candidates.append(((cx - anchor[0]) ** 2 + (cy - anchor[1]) ** 2, cy, cx, candidate))
        candidates.sort(key=lambda item: item[:3])
        selected = None
        for _, _, _, candidate in candidates:
            if any(candidate.expanded(gap).overlaps(blocked) for blocked in exclusions):
                continue
            if any(candidate.expanded(gap / 2).overlaps(other.expanded(gap / 2)) for other in occupied):
                continue
            selected = candidate
            break
        if selected is None:
            raise ValueError(f"could not place {name} on table with {gap:.3f} m gap")
        root_x = selected.min_x - extent.local_min[0]
        root_y = selected.min_y - extent.local_min[1]
        root_z = bounds.table_top_z - extent.local_min[2]
        placed[name] = Pose((root_x, root_y, root_z))
        occupied.append(selected)
    return placed


def _tcp_target(position: Sequence[float], closing_degrees: float) -> np.ndarray:
    angle = math.radians(closing_degrees)
    y_axis = np.array([math.cos(angle), math.sin(angle), 0.0])
    x_axis = np.array([0.0, 0.0, -1.0])
    z_axis = np.cross(x_axis, y_axis)
    tcp = np.eye(4)
    tcp[:3, 0] = x_axis
    tcp[:3, 1] = y_axis
    tcp[:3, 2] = z_axis
    tcp[:3, 3] = position
    flange = tcp.copy()
    flange[:3, 3] = tcp[:3, 3] - 0.135 * tcp[:3, 0]
    return flange


def _load_kinematics(config: SceneConfig):
    import importlib.util

    scripts = config.repo_root / "03_scripts"
    source = scripts / "asm_kin.py"
    if not source.is_file():
        source = Path(__file__).resolve().parents[2] / "03_scripts" / "asm_kin.py"
    spec = importlib.util.spec_from_file_location("interactive_scene_layout_asm_kin", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy kinematics helper: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    arm = module.Arm(str(config.resolve_repo_path("probe_json")))
    robot = config.data["robot"]
    arm.base = module.T(robot["base_position"], robot["base_orientation_wxyz"])
    return arm


def _best_ik(arm, targets: Sequence[np.ndarray], home: np.ndarray) -> ReachabilityTarget:
    seeds = [home]
    for j2 in (-0.9, -0.3, 0.3, 0.9):
        for j3 in (-1.4, -0.6, 0.2, 1.0):
            seed = home.copy()
            seed[1], seed[2] = j2, j3
            seeds.append(seed)
    best = None
    for target in targets:
        for seed in seeds:
            joints, position_error, rotation_error = arm.ik(
                target, seed, iters=140, damping=0.03, step=0.8
            )
            margin = float(np.min(np.minimum(joints - arm.lo, arm.hi - joints)))
            score = position_error + 0.1 * rotation_error - 0.001 * margin
            candidate = (score, joints, position_error, rotation_error, margin)
            if best is None or candidate[0] < best[0]:
                best = candidate
    assert best is not None
    _, joints, position_error, rotation_error, margin = best
    return ReachabilityTarget(
        name="",
        reachable=bool(position_error < 0.002 and rotation_error < 0.02),
        joints=tuple(float(value) for value in joints),
        position_error=float(position_error),
        rotation_error=float(rotation_error),
        joint_margin=margin,
    )


def check_reachability(config: SceneConfig, poses: Mapping[str, Pose]) -> ReachabilityReport:
    layout = _layout(config)
    pieces = {piece["name"]: piece for piece in layout["pieces"]}
    arm = _load_kinematics(config)
    home = np.asarray(config.data["robot"]["home_joint_positions"], dtype=float)
    approach_height = float(config.data["placement"]["approach_height"])
    results: dict[str, ReachabilityTarget] = {}
    for name in config.fragment_names:
        piece = pieces[name]
        pose = poses[name]
        grasp = piece["grasp"]
        target_position = (
            pose.position[0] + float(grasp["xy"][0]),
            pose.position[1] + float(grasp["xy"][1]),
            pose.position[2] + float(grasp["z_top"]) - 0.012 + approach_height,
        )
        angle = float(grasp["angle_deg"])
        result = _best_ik(
            arm,
            [_tcp_target(target_position, angle), _tcp_target(target_position, angle + 180.0)],
            home,
        )
        results[name] = ReachabilityTarget(name=name, **{key: value for key, value in result.__dict__.items() if key != "name"})

    plate = config.data["environment"]["plate_position"]
    plate_position = (float(plate[0]), float(plate[1]), float(plate[2]) + approach_height + 0.12)
    plate_result = _best_ik(
        arm,
        [_tcp_target(plate_position, 90.0), _tcp_target(plate_position, 270.0)],
        home,
    )
    results["plate_center"] = ReachabilityTarget(
        name="plate_center",
        **{key: value for key, value in plate_result.__dict__.items() if key != "name"},
    )
    return ReachabilityReport(
        targets=results,
        lower_limits=tuple(float(value) for value in arm.lo),
        upper_limits=tuple(float(value) for value in arm.hi),
        probe_path=str(config.resolve_repo_path("probe_json")),
    )


def freeze_layout(config_path: Path) -> ReachabilityReport:
    """Choose the best reachable configured base yaw and persist stable poses."""

    source = SceneConfig.load(config_path)
    extents = load_piece_extents(source)
    poses = compute_staging_poses(
        extents,
        table_bounds(source),
        scene_exclusions(source),
        float(source.data["environment"]["staging_gap"]),
    )
    candidates: list[tuple[float, SceneConfig, ReachabilityReport]] = []
    for yaw_degrees in source.data["placement"]["robot_yaw_candidates_degrees"]:
        data = deepcopy(source.data)
        half_angle = math.radians(float(yaw_degrees)) / 2
        data["robot"]["base_orientation_wxyz"] = [
            math.cos(half_angle),
            0.0,
            0.0,
            math.sin(half_angle),
        ]
        candidate_config = SceneConfig(source.config_path, source.repo_root, data)
        report = check_reachability(candidate_config, poses)
        if report.all_reachable:
            minimum_margin = min(target.joint_margin for target in report.targets.values())
            candidates.append((minimum_margin, candidate_config, report))
    if not candidates:
        raise RuntimeError("no configured robot base yaw reaches all fragment and plate targets")
    _, selected, report = max(candidates, key=lambda candidate: candidate[0])
    output = deepcopy(selected.data)
    output["fragments"]["initial_poses"] = {
        name: {
            "position": list(poses[name].position),
            "orientation_wxyz": list(poses[name].orientation_wxyz),
        }
        for name in selected.fragment_names
    }
    output["placement"]["reachability"] = {
        "all_reachable": report.all_reachable,
        "targets": {
            name: {
                "position_error": target.position_error,
                "rotation_error": target.rotation_error,
                "joint_margin": target.joint_margin,
                "joints": list(target.joints),
            }
            for name, target in report.targets.items()
        },
    }
    source.config_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    final_report = freeze_layout(arguments.config)
    print(
        json.dumps(
            {
                "all_reachable": final_report.all_reachable,
                "targets": {
                    name: {
                        "reachable": target.reachable,
                        "position_error": target.position_error,
                        "rotation_error": target.rotation_error,
                        "joint_margin": target.joint_margin,
                    }
                    for name, target in final_report.targets.items()
                },
            },
            indent=2,
        )
    )
