"""USD authoring for the shared floor, table, pad, lighting, and materials."""

from __future__ import annotations

from typing import Any, Sequence

from core.workcell.config import WorkcellConfig
from core.workcell.geometry import assembly_pad_geometry
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
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

    environment = config.environment
    UsdGeom.Xform.Define(stage, "/World/Environment")
    UsdGeom.Xform.Define(stage, "/World/Looks")
    support = author_visual_material(stage, "/World/Looks/Support", (0.72, 0.72, 0.76))
    contact = author_physics_material(stage, "/World/Looks/ContactMaterial", config)
    _author_static_box(
        stage,
        "/World/Environment/Floor",
        environment["floor_position"],
        environment["floor_size"],
        support,
        contact,
    )
    _author_static_box(
        stage,
        "/World/Environment/Table",
        environment["table_position"],
        environment["table_size"],
        support,
        contact,
    )

    dome = UsdLux.DomeLight.Define(stage, "/World/Environment/DomeLight")
    dome.CreateIntensityAttr(700.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/Environment/KeyLight")
    distant.CreateIntensityAttr(2500.0)
    UsdGeom.Xformable(distant).AddRotateXYZOp().Set(Gf.Vec3f(315.0, 35.0, 0.0))

    path = "/World/Environment/AssemblyPad"
    geometry = assembly_pad_geometry(environment["assembly_pad"])
    UsdGeom.Xform.Define(stage, path)
    pad_material = author_visual_material(stage, "/World/Looks/AssemblyPad", (0.68, 0.70, 0.74))
    grid_material = author_visual_material(
        stage, "/World/Looks/AssemblyPadGrid", (0.34, 0.38, 0.45)
    )
    visual = UsdGeom.Cube.Define(stage, f"{path}/Visual")
    visual.CreateSizeAttr(1.0)
    set_transform(visual, geometry.collider_center, scale=geometry.size)
    bind_visual(visual.GetPrim(), pad_material)

    grid_step = float(environment["assembly_pad"]["grid_step"])
    x_half, y_half = geometry.size[0] / 2, geometry.size[1] / 2
    grid_z = geometry.top_z + 0.0005
    grid_root = UsdGeom.Xform.Define(stage, f"{path}/Grid")
    for axis, half in (("X", y_half), ("Y", x_half)):
        count = int(round((2 * half) / grid_step))
        for index in range(count + 1):
            offset = -half + min(index * grid_step, 2 * half)
            line = UsdGeom.Cube.Define(stage, f"{grid_root.GetPath()}/{axis}_{index:02d}")
            line.CreateSizeAttr(1.0)
            if axis == "X":
                position = (geometry.center[0], geometry.center[1] + offset, grid_z)
                scale = (2 * x_half, 0.001, 0.001)
            else:
                position = (geometry.center[0] + offset, geometry.center[1], grid_z)
                scale = (0.001, 2 * y_half, 0.001)
            set_transform(line, position, scale=scale)
            bind_visual(line.GetPrim(), grid_material)

    collider = UsdGeom.Cube.Define(stage, f"{path}/Collider")
    collider.CreateSizeAttr(1.0)
    set_transform(collider, geometry.collider_center, scale=geometry.size)
    collider.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
    bind_physics(collider.GetPrim(), contact)
