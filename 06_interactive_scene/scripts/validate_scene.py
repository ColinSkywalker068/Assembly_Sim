"""Headless validation entry point for the interactive scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from build_scene import build_stage, validate_manifest
from scene_config import SceneConfig
from scene_controls import SceneController


def _step(world, count: int) -> None:
    for _ in range(count):
        world.step(render=False)


def validate_robot_motion(handles) -> dict:
    import numpy as np
    from pxr import UsdGeom

    robot = handles.robot
    if robot is None:
        return {"ok": False, "error": "scene has no robot"}
    controller = SceneController(handles)
    controller.reset()
    _step(handles.world, 30)
    before = np.asarray(robot.robot.get_joint_positions(), dtype=float)
    arm_before = before[list(robot.arm_dof_indices)]
    jog = controller.arm.jog(0, float(handles.config.data["robot"]["joint_jog_radians"]))
    _step(handles.world, 60)
    after = np.asarray(robot.robot.get_joint_positions(), dtype=float)
    arm_motion = float(abs(after[robot.arm_dof_indices[0]] - arm_before[0]))

    finger_path = f"{robot.gripper_path}/left_inner_finger"
    finger_prim = handles.stage.GetPrimAtPath(finger_path)
    if not finger_prim.IsValid():
        return {"ok": False, "error": f"missing gripper finger prim: {finger_path}"}
    controller.gripper.command(1.0)
    _step(handles.world, 45)
    open_position = np.asarray(
        UsdGeom.XformCache().GetLocalToWorldTransform(finger_prim).ExtractTranslation(), dtype=float
    )
    close = controller.gripper.command(0.0)
    _step(handles.world, 90)
    closed_position = np.asarray(
        UsdGeom.XformCache().GetLocalToWorldTransform(finger_prim).ExtractTranslation(), dtype=float
    )
    finger_motion = float(np.linalg.norm(closed_position - open_position))

    controller.reset()
    _step(handles.world, 90)
    reset_positions = np.asarray(robot.robot.get_joint_positions(), dtype=float)
    reset_arm = reset_positions[list(robot.arm_dof_indices)]
    expected_home = np.asarray(handles.config.data["robot"]["home_joint_positions"], dtype=float)
    reset_error = float(np.max(np.abs(reset_arm - expected_home)))
    arm_names = tuple(robot.arm_dof_names)
    gripper_names = tuple(robot.gripper_dof_names)
    expected_arm = tuple(str(name) for name in handles.config.data["robot"]["arm_dof_names"])
    expected_gripper = tuple(
        str(name) for name in handles.config.data["robot"]["gripper_dof_names"]
    )
    ok = (
        jog.accepted
        and close.accepted
        and arm_names == expected_arm
        and gripper_names == expected_gripper
        and arm_motion > 1e-4
        and finger_motion > 1e-5
        and reset_error < 0.03
    )
    return {
        "ok": bool(ok),
        "arm_dof_names": arm_names,
        "gripper_dof_names": gripper_names,
        "arm_motion_radians": arm_motion,
        "finger_motion_meters": finger_motion,
        "reset_max_error_radians": reset_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest-only", action="store_true")
    mode.add_argument("--robot-motion", action="store_true")
    arguments = parser.parse_args()
    config = SceneConfig.load(arguments.config)
    handles = build_stage(config, headless=True)
    try:
        if arguments.robot_motion:
            result = validate_robot_motion(handles)
        else:
            report = validate_manifest(handles)
            result = {
                "ok": report.ok,
                "fragment_count": report.fragment_count,
                "missing_paths": report.missing_paths,
                "extra_fragment_paths": report.extra_fragment_paths,
            }
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    finally:
        handles.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
