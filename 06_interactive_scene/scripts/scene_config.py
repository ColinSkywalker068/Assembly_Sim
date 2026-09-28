"""Typed, portable configuration for the interactive Isaac Sim scene."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


EXPECTED_FRAGMENTS = tuple(f"piece_{index}" for index in range(8))
INPUT_PATH_KEYS = ("arm_usd", "gripper_usd", "probe_json", "layout_json", "bricks_dir")


@dataclass(frozen=True)
class SceneConfig:
    """Validated scene configuration whose paths are anchored to the repository."""

    config_path: Path
    repo_root: Path
    data: Mapping[str, Any]

    @classmethod
    def load(cls, path: Path) -> "SceneConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"scene config not found: {config_path}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        repo_root = cls._find_repo_root(config_path.parent)
        config = cls(config_path=config_path, repo_root=repo_root, data=data)
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
        if self.fragment_names != EXPECTED_FRAGMENTS:
            raise ValueError("interactive scene requires exactly piece_0 through piece_7")
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
        return tuple(str(name) for name in self.data["fragments"]["names"])

    def resolve_repo_path(self, key: str) -> Path:
        try:
            value = self.data["paths"][key]
        except KeyError as exc:
            raise KeyError(f"unknown scene path field: {key}") from exc
        path = Path(value)
        return (path if path.is_absolute() else self.repo_root / path).resolve()

    def validate_inputs(self) -> None:
        for key in INPUT_PATH_KEYS:
            resolved = self.resolve_repo_path(key)
            if not resolved.exists():
                raise FileNotFoundError(f"missing scene input '{key}': {resolved}")
        bricks_dir = self.resolve_repo_path("bricks_dir")
        for name in self.fragment_names:
            mesh = bricks_dir / f"{name}.npz"
            if not mesh.is_file():
                raise FileNotFoundError(f"missing fragment mesh '{name}': {mesh.resolve()}")
