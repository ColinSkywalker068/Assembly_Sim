"""Pure stage-manifest contracts for core workcells."""

from __future__ import annotations

from dataclasses import dataclass

from core.workcell.cameras import camera_specs
from core.workcell.config import WorkcellConfig


@dataclass(frozen=True)
class WorkcellManifest:
    floor_path: str
    table_path: str
    assembly_pad_path: str
    robot_paths: tuple[str, ...]
    camera_paths: tuple[str, ...]


def expected_workcell_manifest(config: WorkcellConfig) -> WorkcellManifest:
    robot_paths = tuple(config.robot(name).prim_path for name in config.robot_names)
    camera_paths = tuple(spec.prim_path for spec in camera_specs(config).values())
    if len(set(robot_paths)) != len(robot_paths):
        raise ValueError("robot root paths must be unique")
    if len(set(camera_paths)) != len(camera_paths):
        raise ValueError("camera paths must be unique")
    return WorkcellManifest(
        floor_path="/World/Environment/Floor",
        table_path="/World/Environment/Table",
        assembly_pad_path="/World/Environment/AssemblyPad",
        robot_paths=robot_paths,
        camera_paths=camera_paths,
    )


def validate_workcell_stage(handles) -> tuple[str, ...]:
    manifest = expected_workcell_manifest(handles.config)
    required = (
        manifest.floor_path,
        manifest.table_path,
        manifest.assembly_pad_path,
        *manifest.robot_paths,
        *manifest.camera_paths,
    )
    return tuple(path for path in required if not handles.stage.GetPrimAtPath(path).IsValid())
