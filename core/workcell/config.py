"""Typed loading for invariant workcell data and named robot presets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.robots.config import RobotConfig


PRESET_NAMES = ("dual_arm", "single_arm_left", "single_arm_right")
FORBIDDEN_TASK_KEYS = frozenset(("environment", "physics", "render", "robots", "cameras"))


@dataclass(frozen=True)
class WorkcellConfig:
    """Resolved invariant workcell configuration for one named robot preset."""

    repo_root: Path
    preset_name: str
    data: Mapping[str, Any]
    robots: Mapping[str, RobotConfig]

    @property
    def physics_dt(self) -> float:
        return float(self.physics["dt"])

    @property
    def render_dt(self) -> float:
        return float(self.render["dt"])

    @property
    def camera_resolution(self) -> tuple[int, int]:
        width, height = self.render["camera_resolution"]
        return int(width), int(height)

    @property
    def environment(self) -> Mapping[str, Any]:
        return self.data["environment"]

    @property
    def physics(self) -> Mapping[str, Any]:
        return self.data["physics"]

    @property
    def render(self) -> Mapping[str, Any]:
        return self.data["render"]

    @property
    def cameras(self) -> Mapping[str, Any]:
        return self.data["cameras"]

    @property
    def robot_names(self) -> tuple[str, ...]:
        return tuple(self.robots)

    def robot(self, name: str) -> RobotConfig:
        try:
            return self.robots[name]
        except KeyError as exc:
            raise KeyError(f"robot {name!r} is not enabled in preset {self.preset_name!r}") from exc

    def asset_path(self, name: str) -> Path:
        try:
            value = self.data["assets"][name]
        except KeyError as exc:
            raise KeyError(f"unknown core asset: {name}") from exc
        path = Path(value)
        return (path if path.is_absolute() else self.repo_root / path).resolve()


def available_presets() -> tuple[str, ...]:
    return PRESET_NAMES


def validate_task_workcell_boundary(data: Mapping[str, Any]) -> None:
    conflicts = sorted(FORBIDDEN_TASK_KEYS.intersection(data))
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(f"task configuration cannot override core workcell field(s): {joined}")


def load_workcell_preset(
    name: str,
    repo_root: Path | None = None,
) -> WorkcellConfig:
    if name not in PRESET_NAMES:
        choices = ", ".join(PRESET_NAMES)
        raise ValueError(f"unknown workcell preset {name!r}; choose one of: {choices}")

    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    config_dir = root / "core" / "config"
    data = json.loads((config_dir / "workcell.json").read_text(encoding="utf-8"))
    preset = json.loads(
        (config_dir / "presets" / f"{name}.json").read_text(encoding="utf-8")
    )
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("unsupported workcell schema_version")
    if int(preset.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported preset schema_version: {name}")

    model = data["robot_model"]
    robots = {
        robot_name: RobotConfig.from_data(robot_name, model, placement)
        for robot_name, placement in preset["robots"].items()
    }
    if not robots:
        raise ValueError(f"workcell preset {name!r} must enable at least one robot")
    return WorkcellConfig(root, name, data, robots)
