import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from assembly_pipeline import (
    DEFAULT_SCENE_CONFIG,
    _load_workcell,
    build_launch_command,
    build_parser,
    default_object_id,
    default_output_path,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "assembly_pipeline.py"


def _minimal_layout(root):
    root.mkdir()
    np.savez(
        root / "visual.npz",
        v=np.asarray([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0.1]], dtype=np.float32),
        f=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    data = {
        "schema_version": 2,
        "object_id": "offline",
        "source": {"loader": "breaking_bad", "path": str(root / "deleted-source")},
        "processing": {"pitch": 0.1},
        "normalization": {
            "source_to_canonical": np.eye(4).tolist(),
            "grid_origin": [0, 0, 0],
            "grid_shape": [1, 1, 1],
            "assembly_bounds": [[0, 0, 0], [0.1, 0.1, 0.1]],
        },
        "fragment_count": 1,
        "pieces": [
            {
                "name": "fragment",
                "source_name": "piece_0.obj",
                "mesh": "visual.npz",
                "color": [0.8, 0.2, 0.1],
                "cells": [[0, 0, 0]],
                "voxel_count": 1,
                "local_pivot_cells": [0.5, 0.5, 0],
                "local_bounds": [[-0.05, -0.05, 0], [0.05, 0.05, 0.1]],
                "goal_pose": {"position": [0.05, 0.05, 0], "orientation_wxyz": [1, 0, 0, 0]},
                "staging_pose": {"position": [0, 0, 0.75], "orientation_wxyz": [1, 0, 0, 0]},
                "grasp": None,
            }
        ],
        "staging": {"algorithm": "test"},
    }
    path = root / "layout.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_prepare_requires_dataset_and_input():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "prepare"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "--dataset" in result.stderr
    assert "--input" in result.stderr


def test_prepare_defaults_output_from_object_id():
    source = Path("/datasets/example_data/artifact/39087_sf/fractured_0")

    object_id = default_object_id("breaking-bad", source)

    assert object_id == "39087_sf_fractured_0"
    assert default_output_path(object_id) == REPO_ROOT / "04_intermediate/assemblies/39087_sf_fractured_0"


def test_prepare_exposes_pitch_length_seed_scene_and_overwrite():
    arguments = build_parser().parse_args(
        [
            "prepare",
            "--dataset",
            "breaking-bad",
            "--input",
            "/data/sample",
            "--pitch",
            "0.02",
            "--target-length",
            "0.5",
            "--seed",
            "11",
            "--scene-config",
            "/repo/scene.json",
            "--overwrite",
        ]
    )

    assert arguments.pitch == 0.02
    assert arguments.target_length == 0.5
    assert arguments.seed == 11
    assert arguments.scene_config == Path("/repo/scene.json")
    assert arguments.overwrite is True


def test_stock_workcell_config_exposes_bilateral_grid_pad_staging():
    workcell = _load_workcell(DEFAULT_SCENE_CONFIG)

    pad = workcell["environment"]["assembly_pad"]
    assert pad["center"] == [0.0, 0.1, 0.75]
    assert pad["size"] == [0.512, 0.416, 0.0096]
    assert pad["lane_centers_x"] == [-0.42, 0.42]


def test_validate_succeeds_after_source_directory_is_unavailable(tmp_path):
    layout_path = _minimal_layout(tmp_path / "generated")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "validate", "--assembly", str(layout_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"fragment_count": 1' in result.stdout
    assert not (layout_path.parent / "deleted-source").exists()


def test_launch_delegates_config_and_assembly_arguments(tmp_path):
    config = tmp_path / "scene.json"
    assembly = tmp_path / "layout.json"

    command = build_launch_command(config, assembly, smoke_frames=10)

    assert command == [
        sys.executable,
        str(REPO_ROOT / "06_interactive_scene/scripts/run_scene.py"),
        "--config",
        str(config.resolve()),
        "--assembly",
        str(assembly.resolve()),
        "--smoke-frames",
        "10",
    ]


def test_launch_streaming_delegates_stream_flag(tmp_path):
    config = tmp_path / "scene.json"
    assembly = tmp_path / "layout.json"

    command = build_launch_command(config, assembly, stream=True)

    assert command[-1] == "--stream"


def test_launch_parser_accepts_streaming_mode():
    arguments = build_parser().parse_args(
        ["launch", "--assembly", "generated/layout.json", "--stream"]
    )

    assert arguments.stream is True
