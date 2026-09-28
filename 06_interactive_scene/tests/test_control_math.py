import math
import sys
from pathlib import Path

import pytest


SCENE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCENE_ROOT / "scripts"))

from scene_controls import clamped_joint_target, gripper_dof_targets


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


def test_joint_index_is_rejected():
    with pytest.raises(IndexError, match="joint index"):
        clamped_joint_target([0.0] * 6, 6, 0.1, ARM_LOWER, ARM_UPPER)


@pytest.mark.parametrize("joint_index", range(6))
def test_joint_target_clamps_each_probe_limit_and_preserves_other_joints(joint_index):
    current = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    high, was_clamped = clamped_joint_target(current, joint_index, 100.0, ARM_LOWER, ARM_UPPER)
    low, low_was_clamped = clamped_joint_target(current, joint_index, -100.0, ARM_LOWER, ARM_UPPER)

    assert high[joint_index] == pytest.approx(ARM_UPPER[joint_index])
    assert low[joint_index] == pytest.approx(ARM_LOWER[joint_index])
    assert was_clamped and low_was_clamped
    assert high[:joint_index] + high[joint_index + 1 :] == pytest.approx(
        current[:joint_index] + current[joint_index + 1 :]
    )


def test_unclamped_joint_target_reports_false():
    target, was_clamped = clamped_joint_target([0.0] * 6, 2, 0.25, ARM_LOWER, ARM_UPPER)
    assert target == pytest.approx([0.0, 0.0, 0.25, 0.0, 0.0, 0.0])
    assert not was_clamped


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_gripper_fraction_outside_unit_interval_is_rejected(fraction):
    with pytest.raises(ValueError, match="open fraction"):
        gripper_dof_targets(tuple(MIMIC), fraction, 0.0, 0.8, MIMIC)


@pytest.mark.parametrize("fraction,finger_angle", [(0.0, 0.8), (1.0, 0.0)])
def test_gripper_targets_cover_every_configured_dof(fraction, finger_angle):
    targets = gripper_dof_targets(tuple(MIMIC), fraction, 0.0, 0.8, MIMIC)
    assert tuple(targets) == tuple(MIMIC)
    for name, multiplier in MIMIC.items():
        assert targets[name] == pytest.approx(finger_angle * multiplier)


def test_gripper_targets_name_missing_mimic_dof():
    with pytest.raises(KeyError, match="unexpected_joint"):
        gripper_dof_targets(("unexpected_joint",), 0.5, 0.0, 0.8, MIMIC)
