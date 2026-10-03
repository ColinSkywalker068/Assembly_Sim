"""Compose the analytical two-fragment task onto the shared workcell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.scene.builder import WorkcellHandles, build_workcell

from .assets import (
    FragmentHandle,
    author_fragment,
    author_target_metadata,
)
from .config import StackSawtoothConfig, load_task_config
from .validation import TaskSceneManifest, expected_task_manifest


@dataclass
class StackSawtoothHandles:
    config: StackSawtoothConfig
    workcell: WorkcellHandles
    manifest: TaskSceneManifest
    fragments: tuple[FragmentHandle, ...]

    @property
    def app(self) -> Any:
        return self.workcell.app

    @property
    def world(self) -> Any:
        return self.workcell.world

    @property
    def stage(self) -> Any:
        return self.workcell.stage

    @property
    def cameras(self):
        return self.workcell.cameras


def define_task_roots(stage: Any) -> None:
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, "/World/Task")
    UsdGeom.Xform.Define(stage, "/World/Task/Fragments")


def stabilize_initial_fragments(stage: Any, config: StackSawtoothConfig) -> None:
    """Restore and sleep task bodies after their first PhysX registration step."""

    import omni.physx
    from pxr import Gf, PhysicsSchemaTools, Sdf, UsdGeom, UsdPhysics, UsdUtils

    from core.workcell.environment import set_transform

    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    interface = omni.physx.get_physx_simulation_interface()
    for fragment in config.fragments:
        path = f"/World/Task/Fragments/{fragment.name}"
        prim = stage.GetPrimAtPath(path)
        set_transform(
            UsdGeom.Xformable(prim),
            fragment.initial_pose.position,
            fragment.initial_pose.orientation_wxyz,
        )
        body = UsdPhysics.RigidBodyAPI(prim)
        body.CreateVelocityAttr(Gf.Vec3f(0.0))
        body.CreateAngularVelocityAttr(Gf.Vec3f(0.0))
        encoded_path = PhysicsSchemaTools.sdfPathToInt(Sdf.Path(path))
        interface.put_to_sleep(stage_id, encoded_path)


def build_stack_sawtooth_scene(
    headless: bool = True, stream: bool = False
) -> StackSawtoothHandles:
    config = load_task_config()
    workcell = build_workcell(config.workcell, headless=headless, stream=stream)
    try:
        define_task_roots(workcell.stage)
        author_target_metadata(workcell.stage, config)
        fragments = tuple(
            author_fragment(workcell.stage, workcell.config, config, fragment)
            for fragment in config.fragments
        )
        handles = StackSawtoothHandles(
            config,
            workcell,
            expected_task_manifest(workcell.config, config),
            fragments,
        )
        workcell.world.reset()
        for robot in workcell.robots.values():
            robot.initialize_dofs()
        if workcell.controller is None:
            raise RuntimeError("core workcell controller is not initialized")
        workcell.controller.reset()
        workcell.world.step(render=False)
        stabilize_initial_fragments(workcell.stage, config)
        for _ in range(3):
            workcell.world.step(render=False)
        return handles
    except BaseException:
        workcell.app.close()
        raise
