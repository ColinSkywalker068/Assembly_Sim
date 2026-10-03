"""Robot composition contracts and bounded operator controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.robots.kinematics import MIMIC_2F85
from core.workcell.config import WorkcellConfig


@dataclass(frozen=True)
class CommandResult:
    accepted: bool
    clamped: bool
    message: str


@dataclass(frozen=True)
class RobotPrimPaths:
    root: str
    flange: str
    gripper_container: str
    gripper_root: str
    gripper_base: str
    mount_joint: str

    def __iter__(self):
        return iter(
            (
                self.root,
                self.flange,
                self.gripper_container,
                self.gripper_root,
                self.gripper_base,
                self.mount_joint,
            )
        )


@dataclass(frozen=True)
class RobotResetTargets:
    arm: tuple[float, ...]
    gripper: tuple[float, ...]


@dataclass
class RobotHandle:
    name: str
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
        missing = tuple(
            name
            for name in (*self.arm_dof_names, *self.gripper_dof_names)
            if name not in names
        )
        if missing:
            raise RuntimeError(f"robot articulation is missing configured DOFs: {missing}")
        self.arm_dof_indices = tuple(names.index(name) for name in self.arm_dof_names)
        self.gripper_dof_indices = tuple(names.index(name) for name in self.gripper_dof_names)


def robot_prim_paths(config: WorkcellConfig, name: str) -> RobotPrimPaths:
    robot = config.robot(name)
    container = robot.gripper_path.rsplit("/", 1)[0]
    flange = f"{robot.prim_path}/flange"
    return RobotPrimPaths(
        root=robot.prim_path,
        flange=flange,
        gripper_container=container,
        gripper_root=robot.gripper_path,
        gripper_base=f"{robot.gripper_path}/base_link",
        mount_joint=f"{flange}/GripperMountJoint",
    )


def clamped_joint_target(current, joint_index, delta_rad, lower, upper):
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


def robot_reset_targets(config: WorkcellConfig) -> Mapping[str, RobotResetTargets]:
    result = {}
    for name in config.robot_names:
        robot = config.robot(name)
        gripper = gripper_dof_targets(
            robot.gripper_dof_names,
            1.0,
            robot.gripper_open_radians,
            robot.gripper_closed_radians,
            MIMIC_2F85,
        )
        result[name] = RobotResetTargets(
            arm=robot.home_joint_positions,
            gripper=tuple(gripper[dof] for dof in robot.gripper_dof_names),
        )
    return result


def _apply_named_positions(handle, indices, values) -> None:
    import numpy as np
    from isaacsim.core.utils.types import ArticulationAction

    handle.robot.get_articulation_controller().apply_action(
        ArticulationAction(
            joint_positions=np.asarray(values, dtype=float),
            joint_indices=np.asarray(indices, dtype=int),
        )
    )


class ArmController:
    def __init__(self, robot: RobotHandle, config: WorkcellConfig):
        self.robot = robot
        self.settings = config.robot(robot.name)

    def jog(self, joint_index: int, delta_rad: float) -> CommandResult:
        if joint_index < 0 or joint_index >= len(self.robot.arm_dof_names):
            return CommandResult(False, False, f"invalid arm joint index {joint_index}")
        measured = self.robot.robot.get_joint_positions()
        current = [float(measured[index]) for index in self.robot.arm_dof_indices]
        targets, was_clamped = clamped_joint_target(
            current,
            joint_index,
            delta_rad,
            self.settings.arm_lower_limits,
            self.settings.arm_upper_limits,
        )
        _apply_named_positions(self.robot, self.robot.arm_dof_indices, targets)
        suffix = " (clamped at limit)" if was_clamped else ""
        return CommandResult(True, was_clamped, f"jogged {self.robot.arm_dof_names[joint_index]}{suffix}")

    def home(self) -> CommandResult:
        targets = [
            min(max(value, low), high)
            for value, low, high in zip(
                self.settings.home_joint_positions,
                self.settings.arm_lower_limits,
                self.settings.arm_upper_limits,
            )
        ]
        was_clamped = targets != list(self.settings.home_joint_positions)
        _apply_named_positions(self.robot, self.robot.arm_dof_indices, targets)
        return CommandResult(True, was_clamped, "commanded arm home")


class GripperController:
    def __init__(self, robot: RobotHandle, config: WorkcellConfig):
        self.robot = robot
        self.settings = config.robot(robot.name)

    def command(self, open_fraction: float) -> CommandResult:
        try:
            targets = gripper_dof_targets(
                self.robot.gripper_dof_names,
                open_fraction,
                self.settings.gripper_open_radians,
                self.settings.gripper_closed_radians,
                MIMIC_2F85,
            )
        except (ValueError, KeyError) as exc:
            return CommandResult(False, False, str(exc))
        _apply_named_positions(
            self.robot,
            self.robot.gripper_dof_indices,
            [targets[name] for name in self.robot.gripper_dof_names],
        )
        return CommandResult(True, False, f"commanded gripper open fraction {open_fraction:.2f}")


class WorkcellController:
    def __init__(self, handles, arm_factory=ArmController, gripper_factory=GripperController):
        if not handles.robots:
            raise ValueError("workcell has no robot handles")
        self.handles = handles
        self.arms = {
            name: arm_factory(handles.robots[name], handles.config)
            for name in handles.config.robot_names
        }
        self.grippers = {
            name: gripper_factory(handles.robots[name], handles.config)
            for name in handles.config.robot_names
        }
        self.active_robot_name = handles.config.robot_names[0]

    def toggle_robot(self) -> str:
        names = self.handles.config.robot_names
        index = (names.index(self.active_robot_name) + 1) % len(names)
        self.active_robot_name = names[index]
        return self.active_robot_name

    def jog(self, joint_index: int, delta_rad: float):
        return self.arms[self.active_robot_name].jog(joint_index, delta_rad)

    def home(self):
        return self.arms[self.active_robot_name].home()

    def command_gripper(self, open_fraction: float):
        return self.grippers[self.active_robot_name].command(open_fraction)

    def reset(self) -> None:
        import numpy as np

        targets = robot_reset_targets(self.handles.config)
        for name, handle in self.handles.robots.items():
            target = targets[name]
            handle.robot.set_joint_positions(
                np.asarray(target.arm, dtype=float),
                joint_indices=np.asarray(handle.arm_dof_indices, dtype=int),
            )
            handle.robot.set_joint_positions(
                np.asarray(target.gripper, dtype=float),
                joint_indices=np.asarray(handle.gripper_dof_indices, dtype=int),
            )
            indices = np.asarray(handle.arm_dof_indices + handle.gripper_dof_indices, dtype=int)
            handle.robot.set_joint_velocities(np.zeros(len(indices)), joint_indices=indices)
