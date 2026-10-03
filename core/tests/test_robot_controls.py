import math

import pytest

from core.robots.controls import (
    clamped_joint_target,
    gripper_dof_targets,
    robot_prim_paths,
)
from core.workcell.config import load_workcell_preset


ARM_LOWER = (-math.pi, -math.pi, -1.5 * math.pi, -3.3161, -math.pi, -3.3161)
ARM_UPPER = (math.pi, math.pi, 1.5 * math.pi, 3.3161, math.pi, 3.3161)
MIMIC = {
    "finger_joint": 1.0,
    "right_outer_knuckle_joint": 1.0,
    "left_outer_finger_joint": 0.0,
    "right_outer_finger_joint": 0.0,
    "left_inner_finger_joint": -1.0,
    "right_inner_finger_joint": 1.0,
    "left_inner_finger_knuckle_joint": -1.0,
    "right_inner_finger_knuckle_joint": -1.0,
}


def test_joint_target_rejects_invalid_index():
    with pytest.raises(IndexError, match="joint index"):
        clamped_joint_target([0.0] * 6, 6, 0.1, ARM_LOWER, ARM_UPPER)


@pytest.mark.parametrize("joint_index", range(6))
def test_joint_target_clamps_probe_limits(joint_index):
    current = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    high, high_clamped = clamped_joint_target(
        current, joint_index, 100.0, ARM_LOWER, ARM_UPPER
    )
    low, low_clamped = clamped_joint_target(
        current, joint_index, -100.0, ARM_LOWER, ARM_UPPER
    )

    assert high[joint_index] == pytest.approx(ARM_UPPER[joint_index])
    assert low[joint_index] == pytest.approx(ARM_LOWER[joint_index])
    assert high_clamped and low_clamped


def test_gripper_targets_cover_every_configured_dof():
    targets = gripper_dof_targets(tuple(MIMIC), 0.0, 0.0, 0.8, MIMIC)

    assert tuple(targets) == tuple(MIMIC)
    for name, multiplier in MIMIC.items():
        assert targets[name] == pytest.approx(0.8 * multiplier)


def test_robot_prim_namespaces_are_disjoint():
    config = load_workcell_preset("dual_arm")
    left = robot_prim_paths(config, "left")
    right = robot_prim_paths(config, "right")

    assert left.root == "/World/Robots/Left"
    assert right.root == "/World/Robots/Right"
    assert left.gripper_root == "/World/Grippers/Left/Robotiq_2F_85"
    assert right.gripper_root == "/World/Grippers/Right/Robotiq_2F_85"
    assert not set(left).intersection(right)


def test_robot_config_contains_canonical_joint_limits():
    robot = load_workcell_preset("dual_arm").robot("left")

    assert robot.arm_lower_limits == pytest.approx(ARM_LOWER)
    assert robot.arm_upper_limits == pytest.approx(ARM_UPPER)
