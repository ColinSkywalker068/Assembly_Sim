"""Compose FANUC and Robotiq assets into one articulation."""

from __future__ import annotations

from typing import Any

from core.robots.controls import RobotHandle, robot_prim_paths
from core.workcell.config import WorkcellConfig


def compose_robot(
    stage: Any, world: Any, config: WorkcellConfig, name: str
) -> RobotHandle:
    import numpy as np
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.prims import XFormPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdPhysics

    settings = config.robot(name)
    paths = robot_prim_paths(config, name)
    add_reference_to_stage(str(config.asset_path("arm_usd")), paths.root)
    XFormPrim(paths.root).set_world_poses(
        positions=np.asarray([settings.base_position], dtype=float),
        orientations=np.asarray([settings.base_orientation_wxyz], dtype=float),
    )
    if not stage.GetPrimAtPath(paths.flange).IsValid():
        raise RuntimeError(f"FANUC asset is missing expected flange prim: {paths.flange}")

    add_reference_to_stage(str(config.asset_path("gripper_usd")), paths.gripper_container)
    gripper_root = stage.GetPrimAtPath(paths.gripper_root)
    if not gripper_root.IsValid() or not stage.GetPrimAtPath(paths.gripper_base).IsValid():
        raise RuntimeError(
            f"Robotiq asset is missing expected hierarchy: {paths.gripper_root}, {paths.gripper_base}"
        )
    source_prefix = Sdf.Path("/World/Robotiq_2F_85")
    target_prefix = Sdf.Path(paths.gripper_root)
    for prim in Usd.PrimRange(gripper_root):
        for relationship in prim.GetRelationships():
            targets = relationship.GetTargets()
            remapped = [
                target.ReplacePrefix(source_prefix, target_prefix)
                if target.HasPrefix(source_prefix)
                else target
                for target in targets
            ]
            if remapped != targets:
                relationship.SetTargets(remapped)
    if gripper_root.HasAPI(UsdPhysics.ArticulationRootAPI):
        gripper_root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if gripper_root.HasAPI(PhysxSchema.PhysxArticulationAPI):
        gripper_root.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

    mount = UsdPhysics.FixedJoint.Define(stage, paths.mount_joint)
    mount.CreateBody0Rel().SetTargets([Sdf.Path(paths.flange)])
    mount.CreateBody1Rel().SetTargets([Sdf.Path(paths.gripper_base)])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    mount.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    mount.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    mount.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    robot = world.scene.add(Robot(prim_path=paths.root, name=f"fanuc_robotiq_{name}"))
    return RobotHandle(
        name=name,
        robot=robot,
        root_path=paths.root,
        flange_path=paths.flange,
        gripper_path=paths.gripper_root,
        arm_dof_names=settings.arm_dof_names,
        gripper_dof_names=settings.gripper_dof_names,
    )
