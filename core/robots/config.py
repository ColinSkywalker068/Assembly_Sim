"""Typed robot configuration shared by all workcell presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _float_tuple(values, length: int, field: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{field} must contain {length} values")
    return result


@dataclass(frozen=True)
class RobotConfig:
    """One canonical robot placement combined with the shared robot model."""

    name: str
    prim_path: str
    gripper_path: str
    base_position: tuple[float, float, float]
    base_orientation_wxyz: tuple[float, float, float, float]
    home_joint_positions: tuple[float, ...]
    arm_lower_limits: tuple[float, ...]
    arm_upper_limits: tuple[float, ...]
    joint_jog_radians: float
    arm_dof_names: tuple[str, ...]
    gripper_dof_names: tuple[str, ...]
    gripper_open_radians: float
    gripper_closed_radians: float

    @classmethod
    def from_data(
        cls,
        name: str,
        model: Mapping[str, Any],
        placement: Mapping[str, Any],
    ) -> "RobotConfig":
        title = name.capitalize()
        return cls(
            name=name,
            prim_path=f"/World/Robots/{title}",
            gripper_path=f"/World/Grippers/{title}/Robotiq_2F_85",
            base_position=_float_tuple(placement["base_position"], 3, "base_position"),
            base_orientation_wxyz=_float_tuple(
                placement["base_orientation_wxyz"], 4, "base_orientation_wxyz"
            ),
            home_joint_positions=tuple(float(value) for value in model["home_joint_positions"]),
            arm_lower_limits=tuple(float(value) for value in model["arm_lower_limits"]),
            arm_upper_limits=tuple(float(value) for value in model["arm_upper_limits"]),
            joint_jog_radians=float(model["joint_jog_radians"]),
            arm_dof_names=tuple(str(value) for value in model["arm_dof_names"]),
            gripper_dof_names=tuple(str(value) for value in model["gripper_dof_names"]),
            gripper_open_radians=float(model["gripper_open_radians"]),
            gripper_closed_radians=float(model["gripper_closed_radians"]),
        )
