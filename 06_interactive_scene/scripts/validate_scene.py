"""Headless validation entry point for the interactive scene."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from build_scene import build_stage, validate_manifest
from scene_config import SceneConfig
from scene_controls import SceneController
from scene_cameras import capture_rgb, save_portable_stage, validate_stability, validated_scene_output
from scene_geometry import merge_voxel_cells


def _step(world, count: int) -> None:
    for _ in range(count):
        world.step(render=False)


def validate_robot_motion(handles) -> dict:
    import numpy as np
    from pxr import UsdGeom
    from isaacsim.core.prims import RigidPrim

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

    fragment = handles.fragments[0]
    fragment_body = RigidPrim(fragment.root_path, reset_xform_properties=False)
    fragment_body.set_velocities(np.asarray([[0.35, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=float))
    _step(handles.world, 8)
    controller.reset()
    _step(handles.world, 30)
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
    fragment_position, fragment_orientation = fragment_body.get_world_poses()
    fragment_velocity = np.asarray(fragment_body.get_velocities(), dtype=float)
    expected_fragment_pose = handles.config.data["fragments"]["initial_poses"][fragment.name]
    fragment_position_error = float(
        np.max(
            np.abs(
                np.asarray(fragment_position[0], dtype=float)
                - np.asarray(expected_fragment_pose["position"], dtype=float)
            )
        )
    )
    expected_orientation = np.asarray(expected_fragment_pose["orientation_wxyz"], dtype=float)
    actual_orientation = np.asarray(fragment_orientation[0], dtype=float)
    fragment_orientation_error = float(
        min(
            np.max(np.abs(actual_orientation - expected_orientation)),
            np.max(np.abs(actual_orientation + expected_orientation)),
        )
    )
    fragment_speed = float(np.max(np.abs(fragment_velocity)))
    ok = (
        jog.accepted
        and close.accepted
        and arm_names == expected_arm
        and gripper_names == expected_gripper
        and arm_motion > 1e-4
        and finger_motion > 1e-5
        and reset_error < 0.03
        and fragment_position_error < 1e-4
        and fragment_orientation_error < 1e-4
        and fragment_speed < 1e-4
    )
    return {
        "ok": bool(ok),
        "arm_dof_names": arm_names,
        "gripper_dof_names": gripper_names,
        "arm_motion_radians": arm_motion,
        "finger_motion_meters": finger_motion,
        "reset_max_error_radians": reset_error,
        "fragment_reset_position_error_meters": fragment_position_error,
        "fragment_reset_orientation_error": fragment_orientation_error,
        "fragment_reset_max_speed": fragment_speed,
    }


def validate_fragment_collision_contract(handles) -> dict:
    with handles.config.resolve_repo_path("layout_json").open("r", encoding="utf-8") as stream:
        layout = json.load(stream)
    pieces = {piece["name"]: piece for piece in layout["pieces"]}
    failures = []
    for fragment in handles.fragments:
        expected_count = len(merge_voxel_cells(pieces[fragment.name]["cells"]))
        expected_paths = tuple(
            f"{fragment.root_path}/Colliders/Box_{index:03d}" for index in range(expected_count)
        )
        if fragment.collider_paths != expected_paths:
            failures.append(
                {
                    "fragment": fragment.name,
                    "expected_paths": expected_paths,
                    "actual_paths": fragment.collider_paths,
                }
            )
    return {"ok": not failures, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest-only", action="store_true")
    mode.add_argument("--robot-motion", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--save-stage", action="store_true")
    parser.add_argument("--capture", action="store_true")
    arguments = parser.parse_args()
    config = SceneConfig.load(arguments.config)
    handles = build_stage(config, headless=True)
    try:
        if arguments.robot_motion:
            result = validate_robot_motion(handles)
        elif arguments.full:
            manifest = validate_manifest(handles)
            collision_geometry = validate_fragment_collision_contract(handles)
            stability = validate_stability(handles, seconds=3.0)
            saved = None
            captures = {}
            if arguments.save_stage:
                saved = validated_scene_output(config, "generated_usd")
                save_portable_stage(handles.stage, saved)
                from pxr import Usd

                reopened = Usd.Stage.Open(str(saved))
                expected = (
                    handles.manifest.floor_path,
                    handles.manifest.table_path,
                    handles.manifest.plate_path,
                    handles.manifest.robot_path,
                    *handles.manifest.fragment_paths,
                    *handles.manifest.camera_paths,
                )
                missing_after_reopen = tuple(
                    path for path in expected if not reopened.GetPrimAtPath(path).IsValid()
                )
            else:
                missing_after_reopen = ()
            if arguments.capture:
                if handles.cameras is None:
                    raise RuntimeError("scene has no camera handles")
                for name, camera_path, key in (
                    ("agent", handles.cameras.agent_path, "agent_image"),
                    ("wrist", handles.cameras.wrist_path, "wrist_image"),
                ):
                    capture = capture_rgb(
                        camera_path,
                        validated_scene_output(config, key),
                        resolution=config.camera_resolution,
                    )
                    captures[name] = {
                        "path": str(capture.path),
                        "shape": capture.shape,
                        "attempts": capture.attempts,
                    }
            result = {
                "ok": bool(
                    manifest.ok
                    and collision_geometry["ok"]
                    and stability.ok
                    and not missing_after_reopen
                    and (not arguments.save_stage or saved is not None)
                    and (not arguments.capture or len(captures) == 2)
                ),
                "fragment_count": manifest.fragment_count,
                "collision_geometry": collision_geometry,
                "stability": {
                    "ok": stability.ok,
                    "frames": stability.frames,
                    "max_linear_speed": stability.max_linear_speed,
                    "max_angular_speed": stability.max_angular_speed,
                    "failures": stability.failures,
                },
                "saved_stage": None if saved is None else str(saved),
                "missing_after_reopen": missing_after_reopen,
                "captures": captures,
            }
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
    try:
        process_exit_code = main()
    except BaseException:
        traceback.print_exc()
        process_exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Kit can intercept SystemExit on Windows and report a successful process even
    # when validation failed. main() has already closed SimulationApp, so terminate
    # directly to preserve the validator's command-line contract.
    os._exit(process_exit_code)
