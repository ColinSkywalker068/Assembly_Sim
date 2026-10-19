"""Arm and Gripper accept a probe JSON path or an already-loaded mapping (live-stage joints)."""

import json

import numpy as np

from core.robots.kinematics import Arm, Gripper


def _joint(name, parent, child, pos0, axis="X"):
    return {"path": f"/World/arm/{parent}/{name}", "body0": [f"/World/arm/{parent}"], "body1": [f"/World/arm/{child}"],
            "pos0": pos0, "rot0": [0.0, 0.0, 0.0, 1.0], "pos1": [0.0, 0.0, 0.0], "rot1": [0.0, 0.0, 0.0, 1.0],
            "axis": axis}


ARM = {
    "dof_names": ["joint_1", "joint_2"],
    "limits": [[-3.0, -3.0], [3.0, 3.0]],
    "joints": [
        _joint("joint_1", "base_link", "link_1", [0.0, 0.0, 0.2], "Z"),
        _joint("joint_2", "link_1", "link_2", [0.0, 0.0, 0.5], "Y"),
        _joint("joint_6_flange", "link_2", "flange", [0.4, 0.0, 0.0]),
    ],
}
GRIP = {
    "dof_names": ["left_inner_finger_joint"],
    "joints": [{**_joint("left_inner_finger_joint", "base_link", "left_inner_finger", [0.0, 0.04, 0.0]),
                "path": "/World/grip/base_link/left_inner_finger_joint",
                "body0": ["/World/grip/base_link"], "body1": ["/World/grip/left_inner_finger"]}],
    "links": [{"path": "/World/grip/base_link", "pos": [0.0, 0.0, 0.0]},
              {"path": "/World/grip/left_inner_finger", "pos": [0.0, 0.04, 0.0],
               "bounds": [[0.10, 0.03, -0.01], [0.14, 0.05, 0.01]]}],
}


def test_arm_mapping_matches_json_path(tmp_path):
    path = tmp_path / "probe.json"
    path.write_text(json.dumps({"arm": ARM}))
    q = np.array([0.3, -0.7])
    from_path, from_mapping = Arm(str(path)), Arm({"arm": ARM})
    np.testing.assert_allclose(from_mapping.flange(q), from_path.flange(q))
    np.testing.assert_allclose(from_mapping.lo, [-3.0, -3.0])


def test_arm_accepts_teleop_style_tuple_limits():
    # tasks/stack_sawtooth_shulker_small/teleop.py builds this shape from a live stage.
    arm = Arm({"arm": dict(joints=ARM["joints"], dof_names=("joint_1", "joint_2"), limits=((-1.0, -2.0), (1.0, 2.0)))})
    flange = arm.flange(np.zeros(2))
    np.testing.assert_allclose(flange[:3, 3], [0.4, 0.0, 0.7])
    np.testing.assert_allclose(arm.hi, [1.0, 2.0])


def test_gripper_mapping_matches_json_path(tmp_path):
    path = tmp_path / "probe.json"
    path.write_text(json.dumps({"grip": GRIP}))
    np.testing.assert_allclose(Gripper({"grip": GRIP}).gap, Gripper(str(path)).gap)
