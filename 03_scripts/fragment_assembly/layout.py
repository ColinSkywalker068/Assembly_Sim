"""Schema-versioned generated assembly metadata."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def _pose(value: Any, field: str) -> dict[str, list[float]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a pose mapping")
    try:
        position = [float(item) for item in value["position"]]
        orientation = [float(item) for item in value["orientation_wxyz"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain numeric position and orientation_wxyz") from exc
    if len(position) != 3 or len(orientation) != 4 or not np.isfinite(position + orientation).all():
        raise ValueError(f"{field} has invalid dimensions or non-finite values")
    return {"position": position, "orientation_wxyz": orientation}


def _cells(value: Any, name: str) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"fragment {name} voxel cells must be a non-empty list")
    result = []
    for raw in value:
        if (
            not isinstance(raw, list)
            or len(raw) != 3
            or any(not isinstance(item, int) or isinstance(item, bool) for item in raw)
        ):
            raise ValueError(f"fragment {name} voxel cell must be an integer triplet")
        result.append(tuple(raw))
    if len(result) != len(set(result)):
        raise ValueError(f"fragment {name} contains a duplicate voxel cell")
    return tuple(result)


@dataclass(frozen=True)
class AssemblyLayout:
    path: Path
    data: Mapping[str, Any]
    _legacy_initial_poses: Mapping[str, Any] | None = None

    @property
    def schema_version(self) -> int:
        return int(self.data.get("schema_version", 1))

    @property
    def object_id(self) -> str:
        return str(self.data.get("object_id", self.path.parent.name))

    @property
    def source_path(self) -> Path:
        source = self.data.get("source", {})
        value = source.get("path", self.data.get("source", "")) if isinstance(source, Mapping) else source
        return Path(str(value))

    @property
    def fragment_names(self) -> tuple[str, ...]:
        return tuple(str(piece["name"]) for piece in self.data["pieces"])

    @property
    def pitch(self) -> float:
        if self.schema_version == 2:
            return float(self.data["processing"]["pitch"])
        return float(self.data["pitch"])

    @property
    def has_support_surface(self) -> bool:
        return self.schema_version == 1 and "plate" in self.data

    def fragment(self, name: str) -> Mapping[str, Any]:
        for piece in self.data["pieces"]:
            if piece["name"] == name:
                return piece
        raise KeyError(f"unknown fragment: {name}")

    def mesh_path(self, name: str) -> Path:
        piece = self.fragment(name)
        value = piece.get("mesh", piece.get("npz", f"{name}.npz"))
        path = Path(str(value))
        return (path if path.is_absolute() else self.path.parent / path).resolve()

    def initial_pose(self, name: str) -> Mapping[str, Any]:
        if self.schema_version == 2:
            return self.fragment(name)["staging_pose"]
        if self._legacy_initial_poses is None or name not in self._legacy_initial_poses:
            raise KeyError(f"legacy fragment has no initial pose: {name}")
        return self._legacy_initial_poses[name]

    def goal_pose(self, name: str) -> Mapping[str, Any]:
        if self.schema_version == 2:
            return self.fragment(name)["goal_pose"]
        piece = self.fragment(name)
        return {
            "position": [*piece.get("assembled_xy", (0.0, 0.0)), 0.0],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }


def _validate_v2(data: Mapping[str, Any], path: Path, validate_assets: bool) -> None:
    if not isinstance(data.get("object_id"), str) or not data["object_id"]:
        raise ValueError("layout object_id must be a non-empty string")
    normalization = data.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("layout normalization is required")
    transform = np.asarray(normalization.get("source_to_canonical"), dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("normalization source_to_canonical must be a finite 4-by-4 matrix")
    grid_shape = normalization.get("grid_shape")
    if not isinstance(grid_shape, list) or len(grid_shape) != 3 or any(
        not isinstance(item, int) or item <= 0 for item in grid_shape
    ):
        raise ValueError("normalization grid_shape must contain three positive integers")
    processing = data.get("processing")
    if not isinstance(processing, Mapping) or float(processing.get("pitch", 0.0)) <= 0:
        raise ValueError("processing pitch must be positive")
    pieces = data.get("pieces")
    if not isinstance(pieces, list) or not pieces:
        raise ValueError("layout pieces must be a non-empty list")
    if data.get("fragment_count") != len(pieces):
        raise ValueError("fragment_count does not match pieces")
    seen_names = set()
    cell_owners: dict[tuple[int, int, int], str] = {}
    for piece in pieces:
        name = piece.get("name")
        if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
            raise ValueError(f"invalid fragment name: {name!r}")
        if name in seen_names:
            raise ValueError(f"duplicate fragment name: {name}")
        seen_names.add(name)
        cells = _cells(piece.get("cells"), name)
        if piece.get("voxel_count") != len(cells):
            raise ValueError(f"fragment {name} voxel_count does not match cells")
        for cell in cells:
            if any(value < 0 or value >= grid_shape[axis] for axis, value in enumerate(cell)):
                raise ValueError(f"fragment {name} voxel cell {cell} is outside common grid")
            previous = cell_owners.get(cell)
            if previous is not None:
                raise ValueError(
                    f"voxel cell {cell} is owned by multiple fragments: {previous}, {name}"
                )
            cell_owners[cell] = name
        pivot = np.asarray(piece.get("local_pivot_cells"), dtype=float)
        if pivot.shape != (3,) or not np.isfinite(pivot).all():
            raise ValueError(f"fragment {name} local_pivot_cells must contain three finite values")
        bounds = np.asarray(piece.get("local_bounds"), dtype=float)
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or not np.all(bounds[1] > bounds[0])
        ):
            raise ValueError(f"fragment {name} local_bounds must be finite increasing 3D bounds")
        color = np.asarray(piece.get("color"), dtype=float)
        if color.shape != (3,) or not np.isfinite(color).all():
            raise ValueError(f"fragment {name} color must contain three finite values")
        for field in ("goal_pose", "staging_pose"):
            if field not in piece:
                raise ValueError(f"fragment {name} is missing {field}")
            _pose(piece[field], f"fragment {name} {field}")
        mesh_value = piece.get("mesh")
        if not isinstance(mesh_value, str) or not mesh_value:
            raise ValueError(f"fragment {name} mesh path is required")
        mesh_relative = Path(mesh_value)
        mesh_path = (path.parent / mesh_relative).resolve()
        if mesh_relative.is_absolute() or not mesh_path.is_relative_to(path.parent.resolve()):
            raise ValueError(
                f"fragment {name} mesh path must be relative to the layout directory"
            )
        if validate_assets:
            if not mesh_path.is_file():
                raise FileNotFoundError(f"fragment {name} mesh not found: {mesh_path}")
            with np.load(mesh_path) as mesh:
                if not {"v", "f"}.issubset(mesh.files) or not len(mesh["v"]) or not len(mesh["f"]):
                    raise ValueError(f"fragment {name} has an empty or malformed visual mesh")


def load_layout(path: Path, validate_assets: bool = True) -> AssemblyLayout:
    layout_path = Path(path).expanduser().resolve()
    if not layout_path.is_file():
        raise FileNotFoundError(f"assembly layout not found: {layout_path}")
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    version = int(data.get("schema_version", 1))
    if version != 2:
        raise ValueError(f"unsupported generated layout schema_version: {version}")
    _validate_v2(data, layout_path, validate_assets)
    return AssemblyLayout(layout_path, data)


def load_legacy_layout(
    path: Path,
    fragment_names: Sequence[str],
    initial_poses: Mapping[str, Any],
    validate_assets: bool = True,
) -> AssemblyLayout:
    layout_path = Path(path).expanduser().resolve()
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    pieces = {piece["name"]: piece for piece in data.get("pieces", ())}
    if tuple(fragment_names) != tuple(piece["name"] for piece in data.get("pieces", ())):
        raise ValueError("legacy scene fragment names do not match layout pieces")
    for name in fragment_names:
        _cells(pieces[name].get("cells"), name)
        _pose(initial_poses[name], f"legacy fragment {name} initial pose")
        if validate_assets:
            mesh_path = layout_path.parent / pieces[name].get("npz", f"{name}.npz")
            if not mesh_path.is_file():
                raise FileNotFoundError(f"legacy fragment mesh not found: {mesh_path.resolve()}")
    return AssemblyLayout(layout_path, data, dict(initial_poses))
