"""Validation contracts for task-owned generated assembly content."""

from __future__ import annotations

from dataclasses import dataclass

from core.scene.validation import validate_workcell_stage

from .geometry import merge_voxel_cells


@dataclass(frozen=True)
class AssemblyValidationReport:
    ok: bool
    fragment_count: int
    missing_paths: tuple[str, ...]
    extra_fragment_paths: tuple[str, ...]


def validate_fragment_collision_contract(handles) -> dict:
    failures = []
    for fragment in handles.fragments:
        piece = handles.layout.fragment(fragment.name)
        expected = tuple(
            f"{fragment.root_path}/Colliders/Box_{index:03d}"
            for index, _ in enumerate(merge_voxel_cells(piece["cells"]))
        )
        if tuple(fragment.collider_paths) != expected:
            failures.append(
                {
                    "fragment": fragment.name,
                    "expected": list(expected),
                    "actual": list(fragment.collider_paths),
                }
            )
    return {"ok": not failures, "failures": failures}


def validate_assembly_stage(handles) -> AssemblyValidationReport:
    missing = list(validate_workcell_stage(handles.workcell))
    missing.extend(
        path
        for path in handles.manifest.fragment_paths
        if not handles.stage.GetPrimAtPath(path).IsValid()
    )
    root = handles.stage.GetPrimAtPath("/World/Fragments")
    actual = {
        child.GetPath().pathString
        for child in root.GetChildren()
        if child.GetTypeName() == "Xform"
    }
    expected = set(handles.manifest.fragment_paths)
    extra = tuple(sorted(actual - expected))
    collision = validate_fragment_collision_contract(handles)
    return AssemblyValidationReport(
        not missing and not extra and collision["ok"],
        len(handles.fragments),
        tuple(missing),
        extra,
    )


validate_assembly_scene = validate_assembly_stage
