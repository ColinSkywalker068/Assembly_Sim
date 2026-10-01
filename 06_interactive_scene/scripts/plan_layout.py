"""Deterministic staging and offline reachability checks for both FANUC arms."""

from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from scene_config import SceneConfig
from scene_geometry import plate_geometry

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_SCRIPTS = REPO_ROOT / "03_scripts"
if str(SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS))

from fragment_assembly.staging import (  # noqa: E402
    AABB2D,
    PieceBounds as PieceExtent,
    Pose,
    StagingSpec,
    TableBounds,
    compute_staging_poses as _compute_staging_poses,
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
class RobotReachability:
    targets: Mapping[str, ReachabilityTarget]
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]


@dataclass(frozen=True)
class ReachabilityReport:
    robots: Mapping[str, RobotReachability]
    probe_path: str

    @property
    def all_reachable(self) -> bool:
        if not self.robots:
            return False
        target_names = next(iter(self.robots.values())).targets
        return bool(target_names) and all(self.reachable_robots(name) for name in target_names)

    def reachable_robots(self, target_name: str) -> tuple[str, ...]:
        return tuple(
            name
            for name, report in self.robots.items()
            if report.targets[target_name].reachable
        )


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
    robots = [
        AABB2D.from_center_extent(config.robot_spec(name)["base_position"][:2], (0.34, 0.34))
        for name in config.robot_names
    ]
    layout = _layout(config)
    geometry = plate_geometry(layout["plate"])
    plate_center = geometry.world_surface_center(
        config.data["environment"]["plate_position"]
    )
    plate = AABB2D.from_center_extent(plate_center[:2], geometry.size[:2])
    return [*robots, plate]


def compute_staging_poses(
    piece_extents: Mapping[str, PieceExtent],
    bounds: TableBounds,
    exclusions: Sequence[AABB2D],
    gap: float,
) -> dict[str, Pose]:
    """Compatibility wrapper around the shared staging implementation."""

    return _compute_staging_poses(
        piece_extents,
        StagingSpec(table=bounds, exclusions=tuple(exclusions), gap=gap, grid_step=0.01),
    )


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


def _load_kinematics(config: SceneConfig, robot_name: str):
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
    robot = config.robot_spec(robot_name)
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
    approach_height = float(config.data["placement"]["approach_height"])
    robot_reports: dict[str, RobotReachability] = {}
    for robot_name in config.robot_names:
        arm = _load_kinematics(config, robot_name)
        home = np.asarray(config.robot_spec(robot_name)["home_joint_positions"], dtype=float)
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
            results[name] = ReachabilityTarget(
                name=name,
                **{key: value for key, value in result.__dict__.items() if key != "name"},
            )

        plate = plate_geometry(layout["plate"]).world_surface_center(
            config.data["environment"]["plate_position"]
        )
        plate_position = (
            float(plate[0]),
            float(plate[1]),
            float(plate[2]) + approach_height + 0.12,
        )
        plate_result = _best_ik(
            arm,
            [_tcp_target(plate_position, 90.0), _tcp_target(plate_position, 270.0)],
            home,
        )
        results["plate_center"] = ReachabilityTarget(
            name="plate_center",
            **{key: value for key, value in plate_result.__dict__.items() if key != "name"},
        )
        robot_reports[robot_name] = RobotReachability(
            targets=results,
            lower_limits=tuple(float(value) for value in arm.lo),
            upper_limits=tuple(float(value) for value in arm.hi),
        )
    return ReachabilityReport(
        robots=robot_reports,
        probe_path=str(config.resolve_repo_path("probe_json")),
    )


def freeze_layout(config_path: Path) -> ReachabilityReport:
    """Validate and persist reachability for the approved demo-exact poses."""

    source = SceneConfig.load(config_path)
    poses = {
        name: Pose(
            tuple(source.data["fragments"]["initial_poses"][name]["position"]),
            tuple(source.data["fragments"]["initial_poses"][name]["orientation_wxyz"]),
        )
        for name in source.fragment_names
    }
    report = check_reachability(source, poses)
    if not report.all_reachable:
        raise RuntimeError("one or more demo layout targets are unreachable by both robots")
    output = deepcopy(source.data)
    output["placement"]["reachability"] = {
        "all_reachable": report.all_reachable,
        "robots": {
            robot_name: {
                "targets": {
                    name: {
                        "reachable": target.reachable,
                        "position_error": target.position_error,
                        "rotation_error": target.rotation_error,
                        "joint_margin": target.joint_margin,
                        "joints": list(target.joints),
                    }
                    for name, target in robot.targets.items()
                }
            }
            for robot_name, robot in report.robots.items()
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
                "robots": {
                    robot_name: {
                        "targets": {
                            name: {
                                "reachable": target.reachable,
                                "position_error": target.position_error,
                                "rotation_error": target.rotation_error,
                                "joint_margin": target.joint_margin,
                            }
                            for name, target in robot.targets.items()
                        }
                    }
                    for robot_name, robot in final_report.robots.items()
                },
            },
            indent=2,
        )
    )
