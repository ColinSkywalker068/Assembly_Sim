"""Robot composition and small, bounded operator controls.

Isaac imports are deliberately local so the target-generation contract can be
tested without starting Omniverse Kit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scene_config import SceneConfig


@dataclass(frozen=True)
class CommandResult:
    accepted: bool
    clamped: bool
    message: str


@dataclass
class RobotHandle:
    robot: Any
    root_path: str
    flange_path: str
    gripper_path: str
    arm_dof_names: tuple[str, ...]
    gripper_dof_names: tuple[str, ...]
    arm_dof_indices: tuple[int, ...] = ()
    gripper_dof_indices: tuple[int, ...] = ()

    def initialize_dofs(self) -> None:
        names = tuple(str(name) for name in self.robot.dof_names)
        missing_arm = tuple(name for name in self.arm_dof_names if name not in names)
        missing_gripper = tuple(name for name in self.gripper_dof_names if name not in names)
        if missing_arm or missing_gripper:
            raise RuntimeError(
                "robot articulation is missing configured DOFs: "
                f"arm={missing_arm or 'none'}, gripper={missing_gripper or 'none'}; exposed={names}"
            )
        self.arm_dof_indices = tuple(names.index(name) for name in self.arm_dof_names)
        self.gripper_dof_indices = tuple(names.index(name) for name in self.gripper_dof_names)


def clamped_joint_target(
    current: Sequence[float],
    joint_index: int,
    delta_rad: float,
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[list[float], bool]:
    if not (len(current) == len(lower) == len(upper)):
        raise ValueError("current positions and joint limits must have equal lengths")
    if joint_index < 0 or joint_index >= len(current):
        raise IndexError(f"joint index {joint_index} is outside 0..{len(current) - 1}")
    requested = float(current[joint_index]) + float(delta_rad)
    bounded = min(max(requested, float(lower[joint_index])), float(upper[joint_index]))
    target = [float(value) for value in current]
    target[joint_index] = bounded
    return target, bounded != requested


def gripper_dof_targets(
    dof_names: Sequence[str],
    open_fraction: float,
    open_radians: float,
    closed_radians: float,
    mimic: Mapping[str, float],
) -> dict[str, float]:
    if not 0.0 <= float(open_fraction) <= 1.0:
        raise ValueError(f"open fraction must be in [0, 1], got {open_fraction}")
    missing = tuple(name for name in dof_names if name not in mimic)
    if missing:
        raise KeyError(f"gripper mimic map is missing DOFs: {', '.join(missing)}")
    finger_angle = float(closed_radians) + float(open_fraction) * (
        float(open_radians) - float(closed_radians)
    )
    return {name: finger_angle * float(mimic[name]) for name in dof_names}


def _probe_arm_limits(config: SceneConfig) -> tuple[tuple[float, ...], tuple[float, ...]]:
    probe = json.loads(config.resolve_repo_path("probe_json").read_text(encoding="utf-8"))
    lower, upper = probe["arm"]["limits"]
    return tuple(float(value) for value in lower), tuple(float(value) for value in upper)


def _mimic_2f85() -> Mapping[str, float]:
    import importlib.util

    source = Path(__file__).resolve().parents[2] / "03_scripts" / "asm_kin.py"
    spec = importlib.util.spec_from_file_location("interactive_scene_asm_kin", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load gripper mimic model: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MIMIC_2F85


def _apply_named_positions(robot_handle: RobotHandle, indices: Sequence[int], values: Sequence[float]) -> None:
    import numpy as np
    from isaacsim.core.utils.types import ArticulationAction

    robot_handle.robot.get_articulation_controller().apply_action(
        ArticulationAction(
            joint_positions=np.asarray(values, dtype=float),
            joint_indices=np.asarray(indices, dtype=int),
        )
    )


class ArmController:
    def __init__(self, robot: RobotHandle, config: SceneConfig):
        self.robot = robot
        self.config = config
        self.lower, self.upper = _probe_arm_limits(config)

    def jog(self, joint_index: int, delta_rad: float) -> CommandResult:
        if joint_index < 0 or joint_index >= len(self.robot.arm_dof_names):
            return CommandResult(False, False, f"invalid arm joint index {joint_index}")
        measured = self.robot.robot.get_joint_positions()
        current = [float(measured[index]) for index in self.robot.arm_dof_indices]
        targets, was_clamped = clamped_joint_target(
            current, joint_index, delta_rad, self.lower, self.upper
        )
        _apply_named_positions(self.robot, self.robot.arm_dof_indices, targets)
        suffix = " (clamped at limit)" if was_clamped else ""
        return CommandResult(True, was_clamped, f"jogged {self.robot.arm_dof_names[joint_index]}{suffix}")

    def home(self) -> CommandResult:
        targets = tuple(float(value) for value in self.config.data["robot"]["home_joint_positions"])
        if len(targets) != len(self.robot.arm_dof_names):
            return CommandResult(False, False, "home target count does not match arm DOFs")
        bounded = [min(max(value, low), high) for value, low, high in zip(targets, self.lower, self.upper)]
        was_clamped = bounded != list(targets)
        _apply_named_positions(self.robot, self.robot.arm_dof_indices, bounded)
        return CommandResult(True, was_clamped, "commanded arm home")


class GripperController:
    def __init__(self, robot: RobotHandle, config: SceneConfig):
        self.robot = robot
        self.config = config
        self.mimic = _mimic_2f85()

    def command(self, open_fraction: float) -> CommandResult:
        settings = self.config.data["robot"]
        try:
            targets = gripper_dof_targets(
                self.robot.gripper_dof_names,
                open_fraction,
                settings["gripper_open_radians"],
                settings["gripper_closed_radians"],
                self.mimic,
            )
        except (ValueError, KeyError) as exc:
            return CommandResult(False, False, str(exc))
        _apply_named_positions(
            self.robot,
            self.robot.gripper_dof_indices,
            [targets[name] for name in self.robot.gripper_dof_names],
        )
        return CommandResult(True, False, f"commanded gripper open fraction {open_fraction:.2f}")


class SceneController:
    def __init__(self, handles: Any):
        if handles.robot is None:
            raise ValueError("scene has no robot handle")
        self.handles = handles
        self.arm = ArmController(handles.robot, handles.config)
        self.gripper = GripperController(handles.robot, handles.config)

    def reset(self) -> None:
        import numpy as np
        from isaacsim.core.prims import XFormPrim

        self.arm.home()
        self.gripper.command(1.0)
        poses = self.handles.config.data["fragments"]["initial_poses"]
        for fragment in self.handles.fragments:
            pose = poses[fragment.name]
            prim = XFormPrim(fragment.root_path)
            prim.set_world_poses(
                positions=np.asarray([pose["position"]], dtype=float),
                orientations=np.asarray([pose["orientation_wxyz"]], dtype=float),
            )


def compose_robot(stage: Any, world: Any, config: SceneConfig) -> RobotHandle:
    """Reference and join the arm and gripper into one articulation."""
    import numpy as np
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdPhysics
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.prims import XFormPrim
    from isaacsim.core.utils.stage import add_reference_to_stage

    settings = config.data["robot"]
    root_path = str(settings["prim_path"])
    flange_path = f"{root_path}/flange"
    # Rigid bodies may not be nested beneath the rigid flange prim.  Keep the
    # gripper as a sibling in the USD namespace and connect it physically with
    # the fixed joint below.
    gripper_container_path = "/World/Gripper"
    gripper_path = f"{gripper_container_path}/Robotiq_2F_85"
    add_reference_to_stage(str(config.resolve_repo_path("arm_usd")), root_path)
    XFormPrim(root_path).set_world_poses(
        positions=np.asarray([settings["base_position"]], dtype=float),
        orientations=np.asarray([settings["base_orientation_wxyz"]], dtype=float),
    )
    if not stage.GetPrimAtPath(flange_path).IsValid():
        raise RuntimeError(f"FANUC asset is missing expected flange prim: {flange_path}")

    add_reference_to_stage(str(config.resolve_repo_path("gripper_usd")), gripper_container_path)
    gripper_root = stage.GetPrimAtPath(gripper_path)
    gripper_base_path = f"{gripper_path}/base_link"
    if not gripper_root.IsValid() or not stage.GetPrimAtPath(gripper_base_path).IsValid():
        raise RuntimeError(
            f"Robotiq asset is missing expected hierarchy: {gripper_path}, {gripper_base_path}"
        )
    # This flattened Robotiq asset stores absolute /World/Robotiq_2F_85
    # relationship targets.  A nested reference therefore needs explicit
    # namespace remapping for joint bodies and the PhysX mimic relationship.
    source_prefix = Sdf.Path("/World/Robotiq_2F_85")
    target_prefix = Sdf.Path(gripper_path)
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

    mount_path = f"{flange_path}/GripperMountJoint"
    mount = UsdPhysics.FixedJoint.Define(stage, mount_path)
    mount.CreateBody0Rel().SetTargets([Sdf.Path(flange_path)])
    mount.CreateBody1Rel().SetTargets([Sdf.Path(gripper_base_path)])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    mount.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    mount.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    mount.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    robot = world.scene.add(Robot(prim_path=root_path, name="fanuc_robotiq"))
    return RobotHandle(
        robot=robot,
        root_path=root_path,
        flange_path=flange_path,
        gripper_path=gripper_path,
        arm_dof_names=tuple(str(name) for name in settings["arm_dof_names"]),
        gripper_dof_names=tuple(str(name) for name in settings["gripper_dof_names"]),
    )
