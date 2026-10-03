"""Typed, portable configuration for the interactive Isaac Sim scene."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


EXPECTED_ROBOTS = ("left", "right")
INPUT_PATH_KEYS = ("arm_usd", "gripper_usd", "probe_json", "layout_json", "bricks_dir")

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_SCRIPTS = REPO_ROOT / "03_scripts"
if str(SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS))

from fragment_assembly.layout import AssemblyLayout, load_layout, load_legacy_layout  # noqa: E402


@dataclass(frozen=True)
class SceneConfig:
    """Validated scene configuration whose paths are anchored to the repository."""

    config_path: Path
    repo_root: Path
    data: Mapping[str, Any]
    assembly: AssemblyLayout | None = None

    @classmethod
    def load(cls, path: Path, assembly_path: Path | None = None) -> "SceneConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"scene config not found: {config_path}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        repo_root = cls._find_repo_root(config_path.parent)
        if assembly_path is not None:
            assembly = load_layout(assembly_path, validate_assets=False)
        else:
            layout_value = Path(data["paths"]["layout_json"])
            legacy_path = (
                layout_value if layout_value.is_absolute() else repo_root / layout_value
            ).resolve()
            assembly = (
                load_legacy_layout(
                    legacy_path,
                    data["fragments"]["names"],
                    data["fragments"]["initial_poses"],
                    validate_assets=False,
                )
                if legacy_path.is_file()
                else None
            )
        config = cls(config_path=config_path, repo_root=repo_root, data=data, assembly=assembly)
        config._validate_contract()
        return config

    @staticmethod
    def _find_repo_root(start: Path) -> Path:
        for candidate in (start, *start.parents):
            if (candidate / "README.md").is_file() and (candidate / "02_robot_assets").is_dir():
                return candidate.resolve()
        raise ValueError(f"could not locate repository root above config: {start.resolve()}")

    def _validate_contract(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported scene schema_version: {self.schema_version}")
        if not self.fragment_names:
            raise ValueError("interactive scene requires at least one fragment")
        if self.robot_names != EXPECTED_ROBOTS:
            raise ValueError("interactive scene requires exactly left and right robots")
        if self.physics_dt <= 0 or self.render_dt <= 0:
            raise ValueError("physics_dt and render_dt must be positive")
        width, height = self.camera_resolution
        if width <= 0 or height <= 0:
            raise ValueError("camera resolution must contain positive dimensions")

    @property
    def schema_version(self) -> int:
        return int(self.data["schema_version"])

    @property
    def physics_dt(self) -> float:
        return float(self.data["physics"]["dt"])

    @property
    def render_dt(self) -> float:
        return float(self.data["render"]["dt"])

    @property
    def camera_resolution(self) -> tuple[int, int]:
        value = self.data["render"]["camera_resolution"]
        return int(value[0]), int(value[1])

    @property
    def fragment_names(self) -> tuple[str, ...]:
        if self.assembly is not None:
            return self.assembly.fragment_names
        return tuple(str(name) for name in self.data["fragments"]["names"])

    @property
    def has_support_surface(self) -> bool:
        return self.assembly.has_support_surface if self.assembly is not None else True

    @property
    def has_generic_assembly_pad(self) -> bool:
        return (
            self.assembly is not None
            and not self.assembly.has_support_surface
            and "assembly_pad" in self.data["environment"]
        )

    @property
    def assembly_pad_spec(self) -> Mapping[str, Any]:
        if not self.has_generic_assembly_pad:
            raise RuntimeError("generic assembly pad is unavailable for this scene")
        return self.data["environment"]["assembly_pad"]

    def fragment_spec(self, name: str) -> Mapping[str, Any]:
        if self.assembly is None:
            raise RuntimeError("fragment layout is unavailable; validate scene inputs first")
        return self.assembly.fragment(name)

    def fragment_mesh_path(self, name: str) -> Path:
        if self.assembly is None:
            return (self.resolve_repo_path("bricks_dir") / f"{name}.npz").resolve()
        return self.assembly.mesh_path(name)

    def fragment_initial_pose(self, name: str) -> Mapping[str, Any]:
        if self.assembly is None:
            return self.data["fragments"]["initial_poses"][name]
        return self.assembly.initial_pose(name)

    @property
    def robot_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.data["robots"])

    def robot_spec(self, name: str) -> Mapping[str, Any]:
        if name not in EXPECTED_ROBOTS:
            raise KeyError(f"unknown robot: {name}")
        return self.data["robots"][name]

    def resolve_repo_path(self, key: str) -> Path:
        if self.assembly is not None and key == "layout_json":
            return self.assembly.path
        if self.assembly is not None and key == "bricks_dir":
            return self.assembly.path.parent
        try:
            value = self.data["paths"][key]
        except KeyError as exc:
            raise KeyError(f"unknown scene path field: {key}") from exc
        path = Path(value)
        return (path if path.is_absolute() else self.repo_root / path).resolve()

    def validate_inputs(self) -> None:
        path_keys = INPUT_PATH_KEYS if self.assembly is None else INPUT_PATH_KEYS[:3]
        for key in path_keys:
            resolved = self.resolve_repo_path(key)
            if not resolved.exists():
                raise FileNotFoundError(f"missing scene input '{key}': {resolved}")
        for name in self.fragment_names:
            mesh = self.fragment_mesh_path(name)
            if not mesh.is_file():
                raise FileNotFoundError(f"missing fragment mesh '{name}': {mesh.resolve()}")
