"""Build the invariant Isaac Sim workcell selected by a named preset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.robots.builder import compose_robot
from core.robots.controls import RobotHandle, WorkcellController
from core.workcell.cameras import CameraHandles, author_cameras
from core.workcell.config import WorkcellConfig
from core.workcell.environment import author_environment


@dataclass
class WorkcellHandles:
    config: WorkcellConfig
    app: Any
    world: Any
    stage: Any
    robots: Mapping[str, RobotHandle]
    cameras: CameraHandles
    controller: WorkcellController | None = None


def simulation_launch_config(
    config: WorkcellConfig, headless: bool, stream: bool = False
) -> dict[str, Any]:
    width, height = config.camera_resolution
    launch = {
        "headless": bool(headless),
        "width": width,
        "height": height,
        "renderer": config.render["renderer"],
    }
    if stream:
        launch.update(
            {"headless": True, "hide_ui": False, "multi_gpu": False, "max_gpu_count": 1}
        )
    return launch


def create_world(config: WorkcellConfig, headless: bool, stream: bool = False):
    import torch  # noqa: F401
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_launch_config(config, headless, stream))
    if stream:
        from isaacsim.core.utils.extensions import enable_extension

        app.set_setting("/app/window/drawMouse", True)
        enable_extension("omni.kit.livestream.webrtc")
    from isaacsim.core.api import World

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=config.physics_dt,
        rendering_dt=config.render_dt,
    )
    world.get_physics_context().set_gravity(float(config.physics["gravity"][2]))
    return app, world


def build_workcell(
    config: WorkcellConfig, headless: bool = True, stream: bool = False
) -> WorkcellHandles:
    app, world = create_world(config, headless, stream)
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    author_environment(stage, config)
    robots = {
        name: compose_robot(stage, world, config, name) for name in config.robot_names
    }
    cameras = author_cameras(stage, config, robots)
    world.reset()
    for robot in robots.values():
        robot.initialize_dofs()
    handles = WorkcellHandles(config, app, world, stage, robots, cameras)
    handles.controller = WorkcellController(handles)
    handles.controller.reset()
    for _ in range(4):
        world.step(render=False)
    return handles
