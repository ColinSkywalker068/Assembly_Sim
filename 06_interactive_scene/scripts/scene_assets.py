"""USD asset authoring for the interactive assembly scene.

Isaac Sim imports stay inside authoring functions so pure contract tests can
run in an ordinary Python environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scene_config import SceneConfig
from scene_geometry import VoxelBox, assembly_pad_geometry, merge_voxel_cells, plate_geometry


@dataclass(frozen=True)
class StageManifest:
    floor_path: str
    table_path: str
    plate_path: str | None
    assembly_pad_path: str | None
    robot_paths: tuple[str, str]
    fragment_paths: tuple[str, ...]
    camera_paths: tuple[str, str, str]


@dataclass(frozen=True)
class FragmentHandle:
    name: str
    root_path: str
    visual_path: str
    collider_paths: tuple[str, ...]


def expected_stage_manifest(config: SceneConfig) -> StageManifest:
    cameras = config.data["cameras"]
    robot_paths = tuple(
        str(config.robot_spec(name)["prim_path"]) for name in config.robot_names
    )
    camera_paths = tuple(
        str(cameras[name]["prim_path"])
        for name in ("agent", "left_wrist", "right_wrist")
    )
    if len(set(robot_paths)) != len(robot_paths):
        raise ValueError("robot root paths must be unique")
    if len(set(camera_paths)) != len(camera_paths):
        raise ValueError("camera paths must be unique")
    return StageManifest(
        floor_path="/World/Environment/Floor",
        table_path="/World/Environment/Table",
        plate_path="/World/Environment/Plate" if config.has_support_surface else None,
        assembly_pad_path=(
            "/World/Environment/AssemblyPad" if config.has_generic_assembly_pad else None
        ),
        robot_paths=robot_paths,
        fragment_paths=tuple(f"/World/Fragments/{name}" for name in config.fragment_names),
        camera_paths=camera_paths,
    )


def _set_transform(xformable: Any, position, orientation_wxyz=(1.0, 0.0, 0.0, 0.0), scale=None):
    from pxr import Gf, UsdGeom

    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in position]))
    xformable.AddOrientOp().Set(Gf.Quatf(float(orientation_wxyz[0]), *[float(v) for v in orientation_wxyz[1:]]))
    if scale is not None:
        xformable.AddScaleOp().Set(Gf.Vec3d(*[float(value) for value in scale]))


def _visual_material(stage, path: str, color):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.48)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _physics_material(stage, path: str, config: SceneConfig):
    from pxr import UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics = config.data["physics"]
    api.CreateStaticFrictionAttr(float(physics["static_friction"]))
    api.CreateDynamicFrictionAttr(float(physics["dynamic_friction"]))
    api.CreateRestitutionAttr(float(physics["restitution"]))
    return material


def _bind_visual(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _bind_physics(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def _author_static_box(stage, path: str, position, size, color, visual_material, physics_material):
    from pxr import UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_transform(cube, position, scale=size)
    _bind_visual(cube.GetPrim(), visual_material)
    collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    collision.CreateCollisionEnabledAttr(True)
    _bind_physics(cube.GetPrim(), physics_material)
    return cube


def author_environment(stage, config: SceneConfig) -> None:
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, Vt

    manifest = expected_stage_manifest(config)
    UsdGeom.Xform.Define(stage, "/World/Environment")
    UsdGeom.Xform.Define(stage, "/World/Looks")
    support_visual = _visual_material(stage, "/World/Looks/Support", (0.72, 0.72, 0.76))
    physics_material = _physics_material(stage, "/World/Looks/ContactMaterial", config)
    environment = config.data["environment"]
    _author_static_box(
        stage,
        manifest.floor_path,
        environment["floor_position"],
        environment["floor_size"],
        (0.72, 0.72, 0.76),
        support_visual,
        physics_material,
    )
    dome = UsdLux.DomeLight.Define(stage, "/World/Environment/DomeLight")
    dome.CreateIntensityAttr(700.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/Environment/KeyLight")
    distant.CreateIntensityAttr(2500.0)
    UsdGeom.Xformable(distant).AddRotateXYZOp().Set(Gf.Vec3f(315.0, 35.0, 0.0))
    _author_static_box(
        stage,
        manifest.table_path,
        environment["table_position"],
        environment["table_size"],
        (0.88, 0.87, 0.90),
        support_visual,
        physics_material,
    )

    if manifest.assembly_pad_path is not None:
        geometry = assembly_pad_geometry(config.assembly_pad_spec)
        pad_root = UsdGeom.Xform.Define(stage, manifest.assembly_pad_path)
        pad_visual = _visual_material(stage, "/World/Looks/AssemblyPad", (0.68, 0.70, 0.74))
        grid_visual = _visual_material(stage, "/World/Looks/AssemblyPadGrid", (0.34, 0.38, 0.45))
        visual = UsdGeom.Cube.Define(stage, f"{manifest.assembly_pad_path}/Visual")
        visual.CreateSizeAttr(1.0)
        _set_transform(visual, geometry.collider_center, scale=geometry.size)
        _bind_visual(visual.GetPrim(), pad_visual)
        grid_step = float(config.assembly_pad_spec["grid_step"])
        x_half, y_half = geometry.size[0] / 2, geometry.size[1] / 2
        grid_z = geometry.top_z + 0.0005
        grid_root = UsdGeom.Xform.Define(stage, f"{manifest.assembly_pad_path}/Grid")
        for axis, half, other_half in (("X", x_half, y_half), ("Y", y_half, x_half)):
            count = int(round((2 * half) / grid_step))
            for index in range(count + 1):
                offset = -half + min(index * grid_step, 2 * half)
                line = UsdGeom.Cube.Define(stage, f"{grid_root.GetPath()}/{axis}_{index:02d}")
                line.CreateSizeAttr(1.0)
                position = (
                    geometry.center[0],
                    geometry.center[1] + offset,
                    grid_z,
                ) if axis == "X" else (
                    geometry.center[0] + offset,
                    geometry.center[1],
                    grid_z,
                )
                scale = (2 * x_half, 0.001, 0.001) if axis == "X" else (0.001, 2 * y_half, 0.001)
                _set_transform(line, position, scale=scale)
                _bind_visual(line.GetPrim(), grid_visual)
        collider = UsdGeom.Cube.Define(stage, f"{manifest.assembly_pad_path}/Collider")
        collider.CreateSizeAttr(1.0)
        _set_transform(collider, geometry.collider_center, scale=geometry.size)
        collider.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
        _bind_physics(collider.GetPrim(), physics_material)

    if manifest.plate_path is None:
        return
    plate_visual = _visual_material(stage, "/World/Looks/Plate", (0.78, 0.79, 0.82))
    plate_data = np.load(config.resolve_repo_path("bricks_dir") / "plate.npz")
    if not {"v", "f"}.issubset(plate_data.files) or not len(plate_data["v"]) or not len(plate_data["f"]):
        raise ValueError("malformed or empty plate mesh")
    plate_root = UsdGeom.Xform.Define(stage, manifest.plate_path)
    _set_transform(plate_root, environment["plate_position"])
    mesh = UsdGeom.Mesh.Define(stage, f"{manifest.plate_path}/Visual")
    vertices = plate_data["v"].astype(np.float32)
    faces = plate_data["f"].astype(np.int32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr("none")
    _bind_visual(mesh.GetPrim(), plate_visual)
    geometry = plate_geometry(config.assembly.data["plate"])
    collider = UsdGeom.Cube.Define(stage, f"{manifest.plate_path}/Collider")
    collider.CreateSizeAttr(1.0)
    _set_transform(collider, geometry.collider_local_center, scale=geometry.size)
    collider.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
    _bind_physics(collider.GetPrim(), physics_material)


def _collider_local_geometry(box: VoxelBox, pitch: float, fp_cell) -> tuple[tuple[float, ...], tuple[float, ...]]:
    local_min = (
        (box.min_cell[0] - float(fp_cell[0])) * pitch,
        (box.min_cell[1] - float(fp_cell[1])) * pitch,
        box.min_cell[2] * pitch,
    )
    size = tuple(float(cells) * pitch for cells in box.size_cells)
    center = tuple(low + length / 2 for low, length in zip(local_min, size))
    return center, size


def author_fragment(stage, config: SceneConfig, name: str) -> FragmentHandle:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt

    if name not in config.fragment_names:
        raise ValueError(f"unknown fragment: {name}")
    piece = config.fragment_spec(name)
    mesh_path = config.fragment_mesh_path(name)
    mesh_data = np.load(mesh_path)
    if not {"v", "f"}.issubset(mesh_data.files) or not len(mesh_data["v"]) or not len(mesh_data["f"]):
        raise ValueError(f"malformed or empty fragment mesh: {name} ({mesh_path})")
    root_path = f"/World/Fragments/{name}"
    root = UsdGeom.Xform.Define(stage, root_path)
    pose = config.fragment_initial_pose(name)
    if not pose:
        raise ValueError(f"fragment has no frozen initial pose: {name}")
    _set_transform(root, pose["position"], pose["orientation_wxyz"])

    visual_path = f"{root_path}/Visual"
    visual = UsdGeom.Mesh.Define(stage, visual_path)
    vertices = mesh_data["v"].astype(np.float32)
    faces = mesh_data["f"].astype(np.int32)
    visual.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    visual.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    visual.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    visual.CreateSubdivisionSchemeAttr("none")
    material = _visual_material(stage, f"/World/Looks/{name}", tuple(piece["color"]))
    _bind_visual(visual.GetPrim(), material)

    physics_material = stage.GetPrimAtPath("/World/Looks/ContactMaterial")
    physics_material = UsdShade.Material(physics_material)
    collider_paths = []
    pitch = float(config.assembly.pitch)
    pivot_cells = piece.get("local_pivot_cells", piece.get("fp_cell"))
    if pivot_cells is None:
        raise ValueError(f"fragment has no local voxel pivot: {name}")
    for index, box in enumerate(merge_voxel_cells(piece["cells"])):
        collider_path = f"{root_path}/Colliders/Box_{index:03d}"
        collider = UsdGeom.Cube.Define(stage, collider_path)
        collider.CreateSizeAttr(1.0)
        center, size = _collider_local_geometry(box, pitch, pivot_cells)
        _set_transform(collider, center, scale=size)
        collider.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
        _bind_physics(collider.GetPrim(), physics_material)
        collider_paths.append(collider_path)

    cells_array = np.asarray(piece["cells"], dtype=float)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid_body.CreateRigidBodyEnabledAttr(True)
    # Preserve the exact occupied-voxel collision volume.  Sparse fragments
    # begin asleep for inspection and wake normally when contacted or driven.
    rigid_body.CreateStartsAsleepAttr(True)
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateDensityAttr(float(config.data["physics"]["fragment_density"]))
    center_of_mass = np.mean(cells_array + 0.5, axis=0) * pitch
    center_of_mass[0] -= float(pivot_cells[0]) * pitch
    center_of_mass[1] -= float(pivot_cells[1]) * pitch
    center_of_mass[2] -= float(pivot_cells[2]) * pitch
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*[float(value) for value in center_of_mass]))
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    physx_body.CreateLinearDampingAttr(float(config.data["physics"]["linear_damping"]))
    physx_body.CreateAngularDampingAttr(float(config.data["physics"]["angular_damping"]))
    return FragmentHandle(name, root_path, visual_path, tuple(collider_paths))
