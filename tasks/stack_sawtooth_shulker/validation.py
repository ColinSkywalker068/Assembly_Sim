"""Pure manifests and live validation contracts for the task scene."""

from __future__ import annotations

from dataclasses import dataclass
import math

from core.scene.validation import (
    WorkcellManifest,
    expected_workcell_manifest,
    validate_workcell_stage,
)
from core.workcell.config import WorkcellConfig

from .config import StackSawtoothConfig


@dataclass(frozen=True)
class FragmentManifest:
    name: str
    root_path: str
    visuals_root: str
    colliders_root: str
    visual_paths: tuple[str, ...]
    collider_paths: tuple[str, ...]


@dataclass(frozen=True)
class TaskSceneManifest:
    workcell: WorkcellManifest
    task_root: str
    metadata_path: str
    fragments_root: str
    fragments: tuple[FragmentManifest, ...]

    def fragment(self, name: str) -> FragmentManifest:
        for fragment in self.fragments:
            if fragment.name == name:
                return fragment
        raise KeyError(f"unknown fragment manifest: {name}")


def expected_task_manifest(
    workcell: WorkcellConfig, config: StackSawtoothConfig
) -> TaskSceneManifest:
    fragments = []
    all_paths: list[str] = []
    for fragment in config.fragments:
        root = f"/World/Task/Fragments/{fragment.name}"
        visuals_root = f"{root}/Visuals"
        colliders_root = f"{root}/Colliders"
        visual_paths = tuple(
            f"{visuals_root}/Voxel_{index:03d}"
            for index, _ in enumerate(sorted(fragment.local_cells))
        )
        collider_paths = tuple(
            f"{colliders_root}/Voxel_{index:03d}"
            for index, _ in enumerate(sorted(fragment.local_cells))
        )
        fragments.append(
            FragmentManifest(
                fragment.name,
                root,
                visuals_root,
                colliders_root,
                visual_paths,
                collider_paths,
            )
        )
        all_paths.extend((root, visuals_root, colliders_root, *visual_paths, *collider_paths))
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("task manifest paths must be unique")
    return TaskSceneManifest(
        workcell=expected_workcell_manifest(workcell),
        task_root="/World/Task",
        metadata_path="/World/Task/TargetMetadata",
        fragments_root="/World/Task/Fragments",
        fragments=tuple(fragments),
    )


@dataclass(frozen=True)
class SceneValidationReport:
    ok: bool
    missing_paths: tuple[str, ...]
    extra_fragment_paths: tuple[str, ...]
    contract_failures: tuple[str, ...]


def _required_task_paths(manifest: TaskSceneManifest) -> tuple[str, ...]:
    result = [manifest.task_root, manifest.metadata_path, manifest.fragments_root]
    for fragment in manifest.fragments:
        result.extend(
            (
                fragment.root_path,
                fragment.visuals_root,
                fragment.colliders_root,
                *fragment.visual_paths,
                *fragment.collider_paths,
            )
        )
    return tuple(result)


def _live_contract_failures(handles) -> tuple[str, ...]:
    from pxr import UsdGeom, UsdPhysics

    failures = []
    cache = UsdGeom.XformCache()
    for handle in handles.fragments:
        fragment = handles.config.fragment(handle.name)
        prim = handles.stage.GetPrimAtPath(handle.root_path)
        if not prim.IsValid():
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            failures.append(f"{handle.name} is missing RigidBodyAPI")
            continue
        body = UsdPhysics.RigidBodyAPI(prim)
        if body.GetRigidBodyEnabledAttr().Get() is not True:
            failures.append(f"{handle.name} rigid body is not enabled")
        if body.GetStartsAsleepAttr().Get() is not True:
            failures.append(f"{handle.name} does not start asleep")
        mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        if mass is None or not math.isclose(float(mass), handle.mass, abs_tol=1e-7):
            failures.append(f"{handle.name} mass mismatch")
        transform = cache.GetLocalToWorldTransform(prim)
        position = tuple(float(value) for value in transform.ExtractTranslation())
        expected_position = fragment.initial_pose.position
        if any(
            not math.isclose(actual, expected, abs_tol=1e-6)
            for actual, expected in zip(position, expected_position)
        ):
            failures.append(f"{handle.name} initial position mismatch")
        quaternion = transform.ExtractRotationQuat()
        orientation = (
            float(quaternion.GetReal()),
            *(float(value) for value in quaternion.GetImaginary()),
        )
        expected_orientation = fragment.initial_pose.orientation_wxyz
        if all(
            math.isclose(actual, -expected, abs_tol=1e-6)
            for actual, expected in zip(orientation, expected_orientation)
        ):
            orientation = tuple(-value for value in orientation)
        if any(
            not math.isclose(actual, expected, abs_tol=1e-6)
            for actual, expected in zip(orientation, expected_orientation)
        ):
            failures.append(f"{handle.name} initial orientation mismatch")
    return tuple(failures)


def validate_stack_sawtooth_scene(handles) -> SceneValidationReport:
    missing = list(validate_workcell_stage(handles.workcell))
    missing.extend(
        path
        for path in _required_task_paths(handles.manifest)
        if not handles.stage.GetPrimAtPath(path).IsValid()
    )
    root = handles.stage.GetPrimAtPath(handles.manifest.fragments_root)
    actual = (
        {
            child.GetPath().pathString
            for child in root.GetChildren()
            if child.GetTypeName() == "Xform"
        }
        if root.IsValid()
        else set()
    )
    expected = {fragment.root_path for fragment in handles.manifest.fragments}
    extra = tuple(sorted(actual - expected))
    contract_failures = list(_live_contract_failures(handles))
    for handle, manifest in zip(handles.fragments, handles.manifest.fragments):
        if handle.name != manifest.name:
            contract_failures.append(
                f"fragment handle order mismatch: {handle.name}, {manifest.name}"
            )
        if handle.visual_paths != manifest.visual_paths:
            contract_failures.append(f"{handle.name} visual path contract mismatch")
        if handle.collider_paths != manifest.collider_paths:
            contract_failures.append(f"{handle.name} collider path contract mismatch")
    return SceneValidationReport(
        ok=not missing and not extra and not contract_failures,
        missing_paths=tuple(missing),
        extra_fragment_paths=extra,
        contract_failures=tuple(contract_failures),
    )
