"""Stage and atomically publish a processed fragment assembly."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.workcell.config import WorkcellConfig

from ..domain.layout import load_layout
from ..paths import require_external_output
from .staging import (
    AABB2D,
    PieceBounds,
    StagingSpec,
    TableBounds,
    compute_staging_poses,
    resolved_lane_centers,
)
from .voxelize import ProcessedAssembly


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _staging_spec(workcell: WorkcellConfig) -> StagingSpec:
    environment = workcell.environment
    position = [float(value) for value in environment["table_position"]]
    size = [float(value) for value in environment["table_size"]]
    margin = float(environment.get("workspace_margin", 0.0))
    table = TableBounds(
        position[0] - size[0] / 2 + margin,
        position[0] + size[0] / 2 - margin,
        position[1] - size[1] / 2 + margin,
        position[1] + size[1] / 2 - margin,
        position[2] + size[2] / 2,
    )
    exclusions = tuple(
        AABB2D.from_center_extent(robot.base_position[:2], (0.34, 0.34))
        for robot in (workcell.robot(name) for name in workcell.robot_names)
    )
    pad_data = environment.get("assembly_pad")
    pad = None
    lane_centers_x = None
    if pad_data is not None:
        center = [float(value) for value in pad_data["center"]]
        size = [float(value) for value in pad_data["size"]]
        if len(center) != 3 or len(size) != 3:
            raise ValueError("environment assembly_pad center and size must be 3D")
        pad = AABB2D.from_center_extent(center[:2], size[:2])
        lane_centers_x = tuple(float(value) for value in pad_data["lane_centers_x"])
    return StagingSpec(
        table=table,
        exclusions=exclusions,
        pad=pad,
        lane_centers_x=lane_centers_x,
        gap=float(environment.get("staging_gap", 0.04)),
        grid_step=float(environment.get("assembly_pad", {}).get("grid_step", 0.01)),
    )


def _layout_data(processed: ProcessedAssembly, workcell: WorkcellConfig, staging) -> dict:
    spec = _staging_spec(workcell)
    bilateral = spec.pad is not None
    bounds = {fragment.name: PieceBounds(*fragment.local_bounds) for fragment in processed.fragments}
    lane_centers = resolved_lane_centers(bounds, spec) if bilateral else None
    names = sorted(fragment.name for fragment in processed.fragments)
    split = (len(names) + 1) // 2
    assignments = {
        name: "left" if index < split else "right" for index, name in enumerate(names)
    }
    pieces = []
    for fragment in processed.fragments:
        pose = staging[fragment.name]
        pieces.append(
            {
                "name": fragment.name,
                "source_name": fragment.source_name,
                "mesh": f"{fragment.name}.npz",
                "color": list(fragment.color),
                "cells": [list(cell) for cell in fragment.cells],
                "voxel_count": len(fragment.cells),
                "local_pivot_cells": list(fragment.local_pivot_cells),
                "local_bounds": [list(row) for row in fragment.local_bounds],
                "goal_pose": _json_value(fragment.goal_pose),
                "staging_pose": {
                    "position": list(pose.position),
                    "orientation_wxyz": list(pose.orientation_wxyz),
                },
                "grasp": _json_value(fragment.grasp),
            }
        )
    return {
        "schema_version": 2,
        "object_id": processed.object_id,
        "source": _json_value(processed.source),
        "processing": {
            "pitch": processed.config.pitch,
            "target_length": processed.config.target_length,
            "seed": processed.config.seed,
            "min_surface_samples": processed.config.min_surface_samples,
            "samples_per_pitch_area": processed.config.samples_per_pitch_area,
            "cleanup_connectivity": 26,
        },
        "normalization": {
            "source_to_canonical": processed.source_to_canonical.tolist(),
            "grid_origin": list(processed.grid_origin),
            "grid_shape": list(processed.grid_shape),
            "assembly_bounds": [list(row) for row in processed.assembly_bounds],
        },
        "fragment_count": len(pieces),
        "pieces": pieces,
        "workcell": {
            "preset": workcell.preset_name,
            "schema_version": int(workcell.data["schema_version"]),
        },
        "staging": {
            "algorithm": "bilateral_lanes_v1" if bilateral else "deterministic_aabb_grid_v1",
            "table_bounds": [spec.table.min_x, spec.table.max_x, spec.table.min_y, spec.table.max_y],
            "table_top_z": spec.table.table_top_z,
            "exclusions": [
                [item.min_x, item.max_x, item.min_y, item.max_y] for item in spec.exclusions
            ],
            "gap": spec.gap,
            "grid_step": spec.grid_step,
            **(
                {
                    "pad_bounds": [
                        round(spec.pad.min_x, 12),
                        round(spec.pad.max_x, 12),
                        round(spec.pad.min_y, 12),
                        round(spec.pad.max_y, 12),
                    ],
                    "lane_centers_x": list(lane_centers),
                    "assignments": assignments,
                }
                if bilateral
                else {}
            ),
        },
    }


def write_processed_assembly(
    processed: ProcessedAssembly,
    output_dir: Path,
    workcell: WorkcellConfig,
    overwrite: bool = False,
) -> Path:
    """Write validated assets to a temporary sibling, then publish atomically."""

    output = require_external_output(output_dir)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty; pass --overwrite: {output}")
    bounds = {fragment.name: PieceBounds(*fragment.local_bounds) for fragment in processed.fragments}
    staging = compute_staging_poses(bounds, _staging_spec(workcell))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    backup = output.parent / f".{output.name}.backup"
    try:
        for fragment in processed.fragments:
            if not len(fragment.vertices) or not len(fragment.faces):
                raise ValueError(f"fragment {fragment.name} has an empty visual mesh")
            np.savez(
                temporary / f"{fragment.name}.npz",
                v=np.asarray(fragment.vertices, dtype=np.float32),
                f=np.asarray(fragment.faces, dtype=np.int32),
            )
        data = _layout_data(processed, workcell, staging)
        layout_path = temporary / "layout.json"
        layout_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        load_layout(layout_path)

        if backup.exists():
            shutil.rmtree(backup)
        if output.exists():
            os.replace(output, backup)
        try:
            os.replace(temporary, output)
        except BaseException:
            if backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return output / "layout.json"
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
