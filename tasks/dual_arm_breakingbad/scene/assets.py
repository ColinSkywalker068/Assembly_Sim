"""Author task-owned voxel fragments into a shared core workcell stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.scene.validation import WorkcellManifest, expected_workcell_manifest
from core.workcell.environment import author_visual_material, bind_visual, set_transform
from core.workcell.physics import bind_physics

from ..domain.layout import AssemblyLayout
from .geometry import VoxelBox, merge_voxel_cells


@dataclass(frozen=True)
class AssemblyManifest:
    workcell: WorkcellManifest
    fragment_paths: tuple[str, ...]


@dataclass(frozen=True)
class FragmentHandle:
    name: str
    root_path: str
    visual_path: str
    collider_paths: tuple[str, ...]


def expected_assembly_manifest(workcell, layout: AssemblyLayout) -> AssemblyManifest:
    return AssemblyManifest(
        expected_workcell_manifest(workcell),
        tuple(f"/World/Fragments/{name}" for name in layout.fragment_names),
    )


def _collider_local_geometry(
    box: VoxelBox, pitch: float, pivot_cells
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    local_min = (
        (box.min_cell[0] - float(pivot_cells[0])) * pitch,
        (box.min_cell[1] - float(pivot_cells[1])) * pitch,
        (box.min_cell[2] - float(pivot_cells[2])) * pitch,
    )
    size = tuple(float(cells) * pitch for cells in box.size_cells)
    return tuple(low + length / 2 for low, length in zip(local_min, size)), size


def author_fragment(stage: Any, workcell, layout: AssemblyLayout, name: str) -> FragmentHandle:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt

    piece = layout.fragment(name)
    mesh_path = layout.mesh_path(name)
    with np.load(mesh_path) as mesh_data:
        vertices = mesh_data["v"].astype(np.float32)
        faces = mesh_data["f"].astype(np.int32)
    if not len(vertices) or not len(faces):
        raise ValueError(f"malformed or empty fragment mesh: {name} ({mesh_path})")

    root_path = f"/World/Fragments/{name}"
    root = UsdGeom.Xform.Define(stage, root_path)
    pose = layout.initial_pose(name)
    set_transform(root, pose["position"], pose["orientation_wxyz"])
    visual_path = f"{root_path}/Visual"
    visual = UsdGeom.Mesh.Define(stage, visual_path)
    visual.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    visual.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32))
    )
    visual.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    visual.CreateSubdivisionSchemeAttr("none")
    material = author_visual_material(stage, f"/World/Looks/{name}", piece["color"])
    bind_visual(visual.GetPrim(), material)

    physics_material = UsdShade.Material(
        stage.GetPrimAtPath("/World/Looks/ContactMaterial")
    )
    pivot = piece["local_pivot_cells"]
    collider_paths = []
    for index, box in enumerate(merge_voxel_cells(piece["cells"])):
        path = f"{root_path}/Colliders/Box_{index:03d}"
        collider = UsdGeom.Cube.Define(stage, path)
        collider.CreateSizeAttr(1.0)
        center, size = _collider_local_geometry(box, layout.pitch, pivot)
        set_transform(collider, center, scale=size)
        collider.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
        bind_physics(collider.GetPrim(), physics_material)
        collider_paths.append(path)

    cells = np.asarray(piece["cells"], dtype=float)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateStartsAsleepAttr(True)
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateDensityAttr(float(workcell.physics["fragment_density"]))
    center_of_mass = (np.mean(cells + 0.5, axis=0) - np.asarray(pivot)) * layout.pitch
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*[float(value) for value in center_of_mass]))
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    body.CreateLinearDampingAttr(float(workcell.physics["linear_damping"]))
    body.CreateAngularDampingAttr(float(workcell.physics["angular_damping"]))
    return FragmentHandle(name, root_path, visual_path, tuple(collider_paths))
