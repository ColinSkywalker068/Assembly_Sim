"""Build the physics-enabled interactive skull assembly stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scene_assets import FragmentHandle, StageManifest, author_environment, author_fragment, expected_stage_manifest
from scene_config import SceneConfig
from scene_controls import RobotHandle, SceneController, compose_robot


@dataclass
class SceneHandles:
    config: SceneConfig
    app: Any
    world: Any
    stage: Any
    manifest: StageManifest
    fragments: tuple[FragmentHandle, ...]
    robot: RobotHandle | None = None


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    fragment_count: int
    missing_paths: tuple[str, ...]
    extra_fragment_paths: tuple[str, ...]


def create_world(config: SceneConfig, headless: bool):
    # The Windows pip distribution can discover torch from several extension
    # worker threads at once.  Importing it once on the main thread avoids a
    # c10.dll initialization race seen with newer torch releases.
    import torch  # noqa: F401

    from isaacsim import SimulationApp

    width, height = config.camera_resolution
    app = SimulationApp(
        {
            "headless": bool(headless),
            "width": width,
            "height": height,
            "renderer": config.data["render"]["renderer"],
        }
    )
    from isaacsim.core.api import World

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=config.physics_dt,
        rendering_dt=config.render_dt,
    )
    world.get_physics_context().set_gravity(float(config.data["physics"]["gravity"][2]))
    return app, world


def build_stage(config: SceneConfig, headless: bool = True) -> SceneHandles:
    config.validate_inputs()
    app, world = create_world(config, headless)
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Fragments")
    author_environment(stage, config)
    fragments = tuple(author_fragment(stage, config, name) for name in config.fragment_names)
    robot = compose_robot(stage, world, config)
    world.reset()
    robot.initialize_dofs()
    handles = SceneHandles(
        config, app, world, stage, expected_stage_manifest(config), fragments, robot
    )
    SceneController(handles).reset()
    for _ in range(4):
        world.step(render=False)
    return handles


def validate_manifest(handles: SceneHandles) -> ValidationReport:
    required = (
        handles.manifest.floor_path,
        handles.manifest.table_path,
        handles.manifest.plate_path,
        handles.manifest.robot_path,
        *handles.manifest.fragment_paths,
    )
    missing = tuple(path for path in required if not handles.stage.GetPrimAtPath(path).IsValid())
    fragment_root = handles.stage.GetPrimAtPath("/World/Fragments")
    actual_fragments = {
        child.GetPath().pathString
        for child in fragment_root.GetChildren()
        if child.GetTypeName() == "Xform"
    }
    expected_fragments = set(handles.manifest.fragment_paths)
    extra = tuple(sorted(actual_fragments - expected_fragments))
    return ValidationReport(
        ok=not missing and not extra and len(handles.fragments) == len(expected_fragments),
        fragment_count=len(handles.fragments),
        missing_paths=missing,
        extra_fragment_paths=extra,
    )
