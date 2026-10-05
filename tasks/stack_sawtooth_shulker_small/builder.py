"""Compose the analytical two-fragment task onto the shared workcell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.scene.builder import WorkcellHandles, build_workcell
from core.workcell.environment import set_transform

from .assets import (
    FragmentHandle,
    author_fragment,
    author_target_metadata,
)
from .config import StackSawtoothConfig, load_task_config
from .validation import TaskSceneManifest, expected_task_manifest


@dataclass
class StackSawtoothHandles:
    config: StackSawtoothConfig
    workcell: WorkcellHandles
    manifest: TaskSceneManifest
    fragments: tuple[FragmentHandle, ...]

    @property
    def app(self) -> Any:
        return self.workcell.app

    @property
    def world(self) -> Any:
        return self.workcell.world

    @property
    def stage(self) -> Any:
        return self.workcell.stage

    @property
    def cameras(self):
        return self.workcell.cameras


def define_task_roots(stage: Any) -> None:
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, "/World/Task")
    UsdGeom.Xform.Define(stage, "/World/Task/Fragments")


def reset_fragment_rigid_bodies(view: Any, config: StackSawtoothConfig) -> None:
    """Set manifest poses through PhysX and clear all fragment velocities."""

    positions = np.asarray(
        [fragment.initial_pose.position for fragment in config.fragments], dtype=float
    )
    orientations = np.asarray(
        [fragment.initial_pose.orientation_wxyz for fragment in config.fragments],
        dtype=float,
    )
    view.set_world_poses(positions=positions, orientations=orientations)
    view.set_velocities(np.zeros((len(config.fragments), 6), dtype=float))


def synchronize_fragment_usd_poses(stage: Any, config: StackSawtoothConfig) -> None:
    """Author the requested fragment poses into USD for the renderer."""

    from pxr import UsdGeom

    for fragment in config.fragments:
        path = f"/World/Task/Fragments/{fragment.name}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"fragment root is missing from USD stage: {path}")
        set_transform(
            UsdGeom.Xform(prim),
            fragment.initial_pose.position,
            fragment.initial_pose.orientation_wxyz,
        )


def _numpy_array(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def verify_fragment_rigid_body_poses(
    view: Any,
    config: StackSawtoothConfig,
    position_tolerance: float = 1e-4,
    orientation_tolerance: float = 1e-5,
) -> None:
    """Reject stale PhysX poses before a condition is rendered or recorded."""

    positions, orientations = view.get_world_poses()
    actual_positions = _numpy_array(positions)
    actual_orientations = _numpy_array(orientations)
    expected_shape = (len(config.fragments), 3)
    if actual_positions.shape != expected_shape:
        raise RuntimeError(
            f"fragment rigid-body positions have shape {actual_positions.shape}, "
            f"expected {expected_shape}"
        )
    if actual_orientations.shape != (len(config.fragments), 4):
        raise RuntimeError("fragment rigid-body orientations have an invalid shape")
    for index, fragment in enumerate(config.fragments):
        expected_position = np.asarray(fragment.initial_pose.position, dtype=float)
        if not np.allclose(
            actual_positions[index], expected_position, rtol=0.0, atol=position_tolerance
        ):
            raise RuntimeError(
                f"{fragment.name} PhysX position does not match its requested pose: "
                f"actual={actual_positions[index].tolist()}, "
                f"expected={expected_position.tolist()}"
            )
        expected_orientation = np.asarray(
            fragment.initial_pose.orientation_wxyz, dtype=float
        )
        alignment = abs(float(np.dot(actual_orientations[index], expected_orientation)))
        if not np.isclose(alignment, 1.0, rtol=0.0, atol=orientation_tolerance):
            raise RuntimeError(
                f"{fragment.name} PhysX orientation does not match its requested pose: "
                f"actual={actual_orientations[index].tolist()}, "
                f"expected={expected_orientation.tolist()}"
            )


def verify_fragment_usd_poses(
    stage: Any,
    config: StackSawtoothConfig,
    position_tolerance: float = 1e-4,
    orientation_tolerance: float = 1e-5,
) -> None:
    """Reject renderer-visible fragment transforms that differ from the condition."""

    from pxr import UsdGeom

    cache = UsdGeom.XformCache()
    for fragment in config.fragments:
        path = f"/World/Task/Fragments/{fragment.name}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"fragment root is missing from USD stage: {path}")
        matrix = cache.GetLocalToWorldTransform(prim)
        actual_position = np.asarray(matrix.ExtractTranslation(), dtype=float)
        expected_position = np.asarray(fragment.initial_pose.position, dtype=float)
        if not np.allclose(
            actual_position, expected_position, rtol=0.0, atol=position_tolerance
        ):
            raise RuntimeError(
                f"{fragment.name} USD position does not match its requested pose: "
                f"actual={actual_position.tolist()}, expected={expected_position.tolist()}"
            )
        quaternion = matrix.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        actual_orientation = np.asarray(
            (quaternion.GetReal(), imaginary[0], imaginary[1], imaginary[2]), dtype=float
        )
        expected_orientation = np.asarray(
            fragment.initial_pose.orientation_wxyz, dtype=float
        )
        alignment = abs(float(np.dot(actual_orientation, expected_orientation)))
        if not np.isclose(alignment, 1.0, rtol=0.0, atol=orientation_tolerance):
            raise RuntimeError(
                f"{fragment.name} USD orientation does not match its requested pose: "
                f"actual={actual_orientation.tolist()}, "
                f"expected={expected_orientation.tolist()}"
            )


def stabilize_initial_fragments(stage: Any, config: StackSawtoothConfig) -> Any:
    """Restore task bodies through PhysX, clear velocity, and put them to sleep."""

    import omni.physx
    from isaacsim.core.prims import RigidPrim
    from pxr import PhysicsSchemaTools, Sdf, UsdUtils

    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    interface = omni.physx.get_physx_simulation_interface()
    paths = [f"/World/Task/Fragments/{fragment.name}" for fragment in config.fragments]
    view = RigidPrim(prim_paths_expr=paths, name="stack_sawtooth_fragments")
    view.initialize()
    reset_fragment_rigid_bodies(view, config)
    for fragment in config.fragments:
        path = f"/World/Task/Fragments/{fragment.name}"
        encoded_path = PhysicsSchemaTools.sdfPathToInt(Sdf.Path(path))
        interface.put_to_sleep(stage_id, encoded_path)
    return view


def build_stack_sawtooth_scene(
    config: StackSawtoothConfig | None = None,
    headless: bool = True,
    stream: bool = False,
    stage_setup: Any = None,
) -> StackSawtoothHandles:
    config = config or load_task_config()
    workcell = build_workcell(config.workcell, headless=headless, stream=stream)
    try:
        define_task_roots(workcell.stage)
        author_target_metadata(workcell.stage, config)
        fragments = tuple(
            author_fragment(workcell.stage, workcell.config, config, fragment)
            for fragment in config.fragments
        )
        if stage_setup is not None:
            stage_setup(workcell.stage, config)
        handles = StackSawtoothHandles(
            config,
            workcell,
            expected_task_manifest(workcell.config, config),
            fragments,
        )
        workcell.world.reset()
        for robot in workcell.robots.values():
            robot.initialize_dofs()
        if workcell.controller is None:
            raise RuntimeError("core workcell controller is not initialized")
        workcell.controller.reset()
        workcell.world.step(render=False)
        synchronize_fragment_usd_poses(workcell.stage, config)
        fragment_view = stabilize_initial_fragments(workcell.stage, config)
        for _ in range(3):
            workcell.world.step(render=False)
        verify_fragment_usd_poses(workcell.stage, config)
        verify_fragment_rigid_body_poses(fragment_view, config)
        return handles
    except BaseException:
        workcell.app.close()
        raise


def apply_stack_sawtooth_config(
    handles: StackSawtoothHandles, config: StackSawtoothConfig
) -> None:
    """Apply another validated condition to task-owned state in a live scene."""

    author_target_metadata(handles.stage, config)
    synchronize_fragment_usd_poses(handles.stage, config)
    fragment_view = stabilize_initial_fragments(handles.stage, config)
    handles.world.step(render=False)
    verify_fragment_usd_poses(handles.stage, config)
    verify_fragment_rigid_body_poses(fragment_view, config)
    handles.config = config
