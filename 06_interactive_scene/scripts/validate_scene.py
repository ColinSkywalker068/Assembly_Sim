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

    if not handles.robots:
        return {"ok": False, "error": "scene has no robots"}
    controller = SceneController(handles)
    controller.reset()
    _step(handles.world, 30)
    robot_results = {}
    for name in handles.config.robot_names:
        controller.select_robot(name)
        robot = handles.robots[name]
        other_name = next(value for value in handles.config.robot_names if value != name)
        other = handles.robots[other_name]
        before = np.asarray(robot.robot.get_joint_positions(), dtype=float)
        other_before = np.asarray(other.robot.get_joint_positions(), dtype=float)
        jog = controller.jog(
            0, float(handles.config.robot_spec(name)["joint_jog_radians"])
        )
        _step(handles.world, 60)
        after = np.asarray(robot.robot.get_joint_positions(), dtype=float)
        other_after = np.asarray(other.robot.get_joint_positions(), dtype=float)
        arm_motion = float(abs(after[robot.arm_dof_indices[0]] - before[robot.arm_dof_indices[0]]))
        other_drift = float(
            np.max(
                np.abs(
                    other_after[list(other.arm_dof_indices)]
                    - other_before[list(other.arm_dof_indices)]
                )
            )
        )

        finger_path = f"{robot.gripper_path}/left_inner_finger"
        finger_prim = handles.stage.GetPrimAtPath(finger_path)
        if not finger_prim.IsValid():
            return {"ok": False, "error": f"missing gripper finger prim: {finger_path}"}
        controller.command_gripper(1.0)
        _step(handles.world, 45)
        open_position = np.asarray(
            UsdGeom.XformCache().GetLocalToWorldTransform(finger_prim).ExtractTranslation(),
            dtype=float,
        )
        close = controller.command_gripper(0.0)
        _step(handles.world, 90)
        closed_position = np.asarray(
            UsdGeom.XformCache().GetLocalToWorldTransform(finger_prim).ExtractTranslation(),
            dtype=float,
        )
        finger_motion = float(np.linalg.norm(closed_position - open_position))
        robot_results[name] = {
            "jog_accepted": jog.accepted,
            "close_accepted": close.accepted,
            "arm_dof_names": tuple(robot.arm_dof_names),
            "gripper_dof_names": tuple(robot.gripper_dof_names),
            "arm_motion_radians": arm_motion,
            "inactive_arm_drift_radians": other_drift,
            "finger_motion_meters": finger_motion,
        }

    fragment = handles.fragments[0]
    fragment_body = RigidPrim(fragment.root_path, reset_xform_properties=False)
    fragment_body.set_velocities(np.asarray([[0.35, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=float))
    _step(handles.world, 8)
    controller.reset()
    _step(handles.world, 30)
    reset_errors = {}
    for name, robot in handles.robots.items():
        reset_positions = np.asarray(robot.robot.get_joint_positions(), dtype=float)
        reset_arm = reset_positions[list(robot.arm_dof_indices)]
        expected_home = np.asarray(
            handles.config.robot_spec(name)["home_joint_positions"], dtype=float
        )
        reset_errors[name] = float(np.max(np.abs(reset_arm - expected_home)))
    fragment_position, fragment_orientation = fragment_body.get_world_poses()
    fragment_velocity = np.asarray(fragment_body.get_velocities(), dtype=float)
    expected_fragment_pose = handles.config.fragment_initial_pose(fragment.name)
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
    robots_ok = all(
        result["jog_accepted"]
        and result["close_accepted"]
        and result["arm_dof_names"]
        == tuple(handles.config.robot_spec(name)["arm_dof_names"])
        and result["gripper_dof_names"]
        == tuple(handles.config.robot_spec(name)["gripper_dof_names"])
        and result["arm_motion_radians"] > 1e-4
        and result["inactive_arm_drift_radians"] < 0.01
        and result["finger_motion_meters"] > 1e-5
        and reset_errors[name] < 0.03
        for name, result in robot_results.items()
    )
    ok = (
        robots_ok
        and fragment_position_error < 1e-4
        and fragment_orientation_error < 1e-4
        and fragment_speed < 1e-4
    )
    return {
        "ok": bool(ok),
        "robots": robot_results,
        "reset_max_error_radians": reset_errors,
        "fragment_reset_position_error_meters": fragment_position_error,
        "fragment_reset_orientation_error": fragment_orientation_error,
        "fragment_reset_max_speed": fragment_speed,
    }


def validate_fragment_collision_contract(handles) -> dict:
    failures = []
    for fragment in handles.fragments:
        piece = handles.config.fragment_spec(fragment.name)
        expected_count = len(merge_voxel_cells(piece["cells"]))
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assembly", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest-only", action="store_true")
    mode.add_argument("--robot-motion", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--save-stage", action="store_true")
    parser.add_argument("--capture", action="store_true")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    config = SceneConfig.load(arguments.config, assembly_path=arguments.assembly)
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
                expected = tuple(path for path in (
                    handles.manifest.floor_path,
                    handles.manifest.table_path,
                    handles.manifest.plate_path,
                    handles.manifest.assembly_pad_path,
                    *handles.manifest.robot_paths,
                    *handles.manifest.fragment_paths,
                    *handles.manifest.camera_paths,
                ) if path is not None)
                missing_after_reopen = tuple(
                    path for path in expected if not reopened.GetPrimAtPath(path).IsValid()
                )
            else:
                missing_after_reopen = ()
            if arguments.capture:
                if handles.cameras is None:
                    raise RuntimeError("scene has no camera handles")
                from run_scene import capture_requests

                for name, camera_path, key in capture_requests(handles.cameras):
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
                    and (not arguments.capture or len(captures) == 3)
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
