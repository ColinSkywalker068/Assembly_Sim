import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from core.workcell.config import load_workcell_preset
from tasks.dual_arm_breakingbad.__main__ import build_parser, default_object_id


REPO_ROOT = Path(__file__).resolve().parents[3]


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
        "workcell": {"preset": "dual_arm", "schema_version": 1},
        "staging": {"algorithm": "test"},
    }
    path = root / "layout.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_prepare_requires_input_and_output():
    result = subprocess.run(
        [sys.executable, "-m", "tasks.dual_arm_breakingbad", "prepare"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "--input" in result.stderr
    assert "--output" in result.stderr


def test_object_id_is_derived_from_breaking_bad_sample_path():
    source = Path("/datasets/example_data/artifact/39087_sf/fractured_0")

    assert default_object_id(source) == "39087_sf_fractured_0"


def test_prepare_exposes_processing_parameters():
    arguments = build_parser().parse_args(
        [
            "prepare", "--input", "/data/sample", "--output", "/data/processed",
            "--pitch", "0.02", "--target-length", "0.5", "--seed", "11",
            "--overwrite",
        ]
    )

    assert arguments.pitch == 0.02
    assert arguments.target_length == 0.5
    assert arguments.seed == 11
    assert arguments.overwrite is True


def test_stock_workcell_exposes_bilateral_grid_pad_staging():
    pad = load_workcell_preset("dual_arm").environment["assembly_pad"]

    assert pad["center"] == [0.0, 0.0, 0.75]
    assert pad["size"] == [0.55, 0.55, 0.0006]
    assert pad["tape_width"] == 0.012
    assert pad["lane_centers_x"] == [-0.42, 0.42]


def test_validate_succeeds_after_source_directory_is_unavailable(tmp_path):
    layout_path = _minimal_layout(tmp_path / "generated")
    result = subprocess.run(
        [
            sys.executable, "-m", "tasks.dual_arm_breakingbad", "validate",
            "--assembly", str(layout_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"fragment_count": 1' in result.stdout
    assert not (layout_path.parent / "deleted-source").exists()


def test_build_and_launch_parsers_use_generated_assembly_without_legacy_config():
    build = build_parser().parse_args(
        [
            "build", "--assembly", "layout.json", "--output-usd", "/tmp/scene.usda",
            "--capture-directory", "/tmp/captures",
        ]
    )
    launch = build_parser().parse_args(
        ["launch", "--assembly", "layout.json", "--stream"]
    )

    assert build.output == Path("/tmp/scene.usda")
    assert build.capture_directory == Path("/tmp/captures")
    assert launch.stream is True
