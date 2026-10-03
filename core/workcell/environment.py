"""USD authoring for the shared floor, table, pad, lighting, and materials."""

from __future__ import annotations

from typing import Any, Sequence

from core.workcell.config import WorkcellConfig
from core.workcell.geometry import assembly_tape_segments
from core.workcell.physics import author_physics_material, bind_physics


def set_transform(
    xformable: Any,
    position: Sequence[float],
    orientation_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    scale: Sequence[float] | None = None,
) -> None:
    from pxr import Gf

    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in position]))
    xformable.AddOrientOp().Set(
        Gf.Quatf(float(orientation_wxyz[0]), *[float(value) for value in orientation_wxyz[1:]])
    )
    if scale is not None:
        xformable.AddScaleOp().Set(Gf.Vec3d(*[float(value) for value in scale]))


def author_visual_material(stage: Any, path: str, color: Sequence[float]):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*[float(value) for value in color])
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.48)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_visual(prim: Any, material: Any) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _author_static_box(stage, path, position, size, visual_material, physics_material):
    from pxr import UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    set_transform(cube, position, scale=size)
    bind_visual(cube.GetPrim(), visual_material)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    bind_physics(cube.GetPrim(), physics_material)
    return cube


def author_environment(stage: Any, config: WorkcellConfig) -> None:
    from pxr import Gf, UsdGeom, UsdLux

    environment = config.environment
    colors = environment["colors"]
    UsdGeom.Xform.Define(stage, "/World/Environment")
    UsdGeom.Xform.Define(stage, "/World/Looks")
    floor_material = author_visual_material(
        stage, "/World/Looks/Floor", colors["floor"]
    )
    table_material = author_visual_material(
        stage, "/World/Looks/Table", colors["table"]
    )
    contact = author_physics_material(stage, "/World/Looks/ContactMaterial", config)
    _author_static_box(
        stage,
        "/World/Environment/Floor",
        environment["floor_position"],
        environment["floor_size"],
        floor_material,
        contact,
    )
    _author_static_box(
        stage,
        "/World/Environment/Table",
        environment["table_position"],
        environment["table_size"],
        table_material,
        contact,
    )

    dome = UsdLux.DomeLight.Define(stage, "/World/Environment/DomeLight")
    dome.CreateIntensityAttr(700.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/Environment/KeyLight")
    distant.CreateIntensityAttr(2500.0)
    UsdGeom.Xformable(distant).AddRotateXYZOp().Set(Gf.Vec3f(315.0, 35.0, 0.0))

    path = "/World/Environment/AssemblyPad"
    UsdGeom.Xform.Define(stage, path)
    tape_material = author_visual_material(
        stage, "/World/Looks/AssemblyTape", colors["assembly_tape"]
    )
    for segment in assembly_tape_segments(environment["assembly_pad"]):
        tape = UsdGeom.Cube.Define(stage, f"{path}/{segment.name}")
        tape.CreateSizeAttr(1.0)
        set_transform(tape, segment.center, scale=segment.size)
        bind_visual(tape.GetPrim(), tape_material)
