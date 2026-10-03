"""Composition of generated assembly metadata with an immutable core preset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.workcell.config import WorkcellConfig, load_workcell_preset

from ..domain.layout import AssemblyLayout, load_layout


@dataclass(frozen=True)
class AssemblySceneConfig:
    layout: AssemblyLayout
    workcell: WorkcellConfig


def load_assembly_scene_config(path: Path) -> AssemblySceneConfig:
    layout = load_layout(path)
    metadata = layout.data.get("workcell", {})
    preset = str(metadata.get("preset", "dual_arm"))
    workcell = load_workcell_preset(preset)
    if preset != "dual_arm":
        raise ValueError(
            f"dual_arm_breakingbad requires the dual_arm preset, got {preset!r}"
        )
    return AssemblySceneConfig(layout, workcell)
