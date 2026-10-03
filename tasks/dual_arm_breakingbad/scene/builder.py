"""Compose generated voxel fragments onto the reusable dual-arm workcell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.scene.builder import WorkcellHandles, build_workcell

from ..domain.layout import AssemblyLayout
from .assets import AssemblyManifest, FragmentHandle, author_fragment, expected_assembly_manifest
from .config import AssemblySceneConfig, load_assembly_scene_config
from .controller import AssemblySceneController


@dataclass
class AssemblySceneHandles:
    config: AssemblySceneConfig
    layout: AssemblyLayout
    workcell: WorkcellHandles
    manifest: AssemblyManifest
    fragments: tuple[FragmentHandle, ...]
    controller: AssemblySceneController | None = None

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


def build_assembly_scene(
    assembly: Path, headless: bool = True, stream: bool = False
) -> AssemblySceneHandles:
    config = load_assembly_scene_config(assembly)
    workcell = build_workcell(config.workcell, headless=headless, stream=stream)
    from pxr import UsdGeom

    UsdGeom.Xform.Define(workcell.stage, "/World/Fragments")
    fragments = tuple(
        author_fragment(workcell.stage, workcell.config, config.layout, name)
        for name in config.layout.fragment_names
    )
    handles = AssemblySceneHandles(
        config,
        config.layout,
        workcell,
        expected_assembly_manifest(workcell.config, config.layout),
        fragments,
    )
    handles.controller = AssemblySceneController(handles)
    workcell.world.reset()
    for robot in workcell.robots.values():
        robot.initialize_dofs()
    handles.controller.reset()
    for _ in range(4):
        workcell.world.step(render=False)
    return handles
