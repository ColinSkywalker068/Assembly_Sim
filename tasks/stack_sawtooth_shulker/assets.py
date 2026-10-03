"""Procedural USD authoring for the task-owned voxel fragments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.workcell.environment import author_visual_material, bind_visual, set_transform
from core.workcell.physics import bind_physics

from .config import FragmentConfig, StackSawtoothConfig
from .geometry import cell_center, fragment_mass


@dataclass(frozen=True)
class FragmentAuthoringSpec:
    centers: tuple[tuple[float, float, float], ...]
    visual_side: float
    collider_side: float
    material_path: str
    mass: float


@dataclass(frozen=True)
class FragmentHandle:
    name: str
    root_path: str
    visual_paths: tuple[str, ...]
    collider_paths: tuple[str, ...]
    mass: float


def fragment_authoring_spec(
    config: StackSawtoothConfig, fragment: FragmentConfig
) -> FragmentAuthoringSpec:
    return FragmentAuthoringSpec(
        centers=tuple(
            cell_center(cell, config.unit_size) for cell in sorted(fragment.local_cells)
        ),
        visual_side=config.unit_size,
        collider_side=config.unit_size - config.collider_clearance,
        material_path=f"/World/Looks/StackSawtoothShulker/{fragment.name}",
        mass=fragment_mass(fragment, config.unit_size, config.density),
    )


def target_metadata_values(config: StackSawtoothConfig) -> Mapping[str, object]:
    a = config.fragment("FragmentA")
    b = config.fragment("FragmentB")
    return {
        "task_name": "stack_sawtooth_shulker",
        "unit_size": config.unit_size,
        "target_size": config.target_size,
        "fragment_a_goal_position": a.goal_pose.position,
        "fragment_a_goal_orientation_wxyz": a.goal_pose.orientation_wxyz,
        "fragment_b_goal_position": b.goal_pose.position,
        "fragment_b_goal_orientation_wxyz": b.goal_pose.orientation_wxyz,
    }


def author_target_metadata(stage: Any, config: StackSawtoothConfig) -> str:
    from pxr import Gf, Sdf, UsdGeom

    path = "/World/Task/TargetMetadata"
    prim = UsdGeom.Scope.Define(stage, path).GetPrim()
    values = target_metadata_values(config)
    attributes = (
        ("taskName", Sdf.ValueTypeNames.String, values["task_name"]),
        ("unitSize", Sdf.ValueTypeNames.Double, values["unit_size"]),
        ("targetSize", Sdf.ValueTypeNames.Double2, Gf.Vec2d(*values["target_size"])),
        (
            "fragmentAGoalPosition",
            Sdf.ValueTypeNames.Double3,
            Gf.Vec3d(*values["fragment_a_goal_position"]),
        ),
        (
            "fragmentAGoalOrientationWxyz",
            Sdf.ValueTypeNames.Double4,
            Gf.Vec4d(*values["fragment_a_goal_orientation_wxyz"]),
        ),
        (
            "fragmentBGoalPosition",
            Sdf.ValueTypeNames.Double3,
            Gf.Vec3d(*values["fragment_b_goal_position"]),
        ),
        (
            "fragmentBGoalOrientationWxyz",
            Sdf.ValueTypeNames.Double4,
            Gf.Vec4d(*values["fragment_b_goal_orientation_wxyz"]),
        ),
    )
    for name, value_type, value in attributes:
        prim.CreateAttribute(name, value_type, custom=True).Set(value)
    return path


def author_fragment(
    stage: Any,
    workcell,
    config: StackSawtoothConfig,
    fragment: FragmentConfig,
) -> FragmentHandle:
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    spec = fragment_authoring_spec(config, fragment)
    root_path = f"/World/Task/Fragments/{fragment.name}"
    root = UsdGeom.Xform.Define(stage, root_path)
    set_transform(
        root,
        fragment.initial_pose.position,
        fragment.initial_pose.orientation_wxyz,
    )
    visuals_root = f"{root_path}/Visuals"
    colliders_root = f"{root_path}/Colliders"
    UsdGeom.Xform.Define(stage, visuals_root)
    UsdGeom.Xform.Define(stage, colliders_root)
    material = author_visual_material(stage, spec.material_path, fragment.color)
    physics_material = UsdShade.Material(
        stage.GetPrimAtPath("/World/Looks/ContactMaterial")
    )

    visual_paths = []
    collider_paths = []
    for index, center in enumerate(spec.centers):
        visual_path = f"{visuals_root}/Voxel_{index:03d}"
        visual = UsdGeom.Cube.Define(stage, visual_path)
        visual.CreateSizeAttr(1.0)
        set_transform(visual, center, scale=(spec.visual_side,) * 3)
        bind_visual(visual.GetPrim(), material)
        visual_paths.append(visual_path)

        collider_path = f"{colliders_root}/Voxel_{index:03d}"
        collider = UsdGeom.Cube.Define(stage, collider_path)
        collider.CreateSizeAttr(1.0)
        set_transform(collider, center, scale=(spec.collider_side,) * 3)
        collider.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
        bind_physics(collider.GetPrim(), physics_material)
        collider_paths.append(collider_path)

    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateStartsAsleepAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(spec.mass)
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    body.CreateLinearDampingAttr(float(workcell.physics["linear_damping"]))
    body.CreateAngularDampingAttr(float(workcell.physics["angular_damping"]))
    return FragmentHandle(
        fragment.name,
        root_path,
        tuple(visual_paths),
        tuple(collider_paths),
        spec.mass,
    )
