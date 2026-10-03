import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tasks.dual_arm_breakingbad.preprocessing.staging import PieceBounds, Pose


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_breaking_bad_sample_prepares_three_plain_staged_voxel_fragments(tmp_path):
    source_value = os.environ.get("BREAKING_BAD_SAMPLE")
    if not source_value:
        pytest.skip("BREAKING_BAD_SAMPLE is not configured")
    output = tmp_path / "processed"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tasks.dual_arm_breakingbad",
            "prepare",
            "--input",
            source_value,
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr

    layout_path = output / "layout.json"
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    assert data["fragment_count"] == 3
    assert [piece["name"] for piece in data["pieces"]] == ["piece_0", "piece_1", "piece_2"]
    assert len(data["normalization"]["grid_shape"]) == 3
    assert all(value > 0 for value in data["normalization"]["grid_shape"])
    assert sorted(path.name for path in output.iterdir()) == [
        "layout.json",
        "piece_0.npz",
        "piece_1.npz",
        "piece_2.npz",
    ]

    occupied = []
    for piece in data["pieces"]:
        mesh = np.load(output / piece["mesh"])
        assert len(mesh["v"]) > 0
        assert len(mesh["f"]) > 0
        cells = {tuple(cell) for cell in piece["cells"]}
        assert len(cells) == piece["voxel_count"]
        occupied.append(cells)
    for index, first in enumerate(occupied):
        for second in occupied[index + 1 :]:
            assert first.isdisjoint(second)

    from tasks.dual_arm_breakingbad.scene.geometry import merge_voxel_cells

    for piece, cells in zip(data["pieces"], occupied):
        boxes = merge_voxel_cells(piece["cells"])
        assert sum(box.cell_count for box in boxes) == len(cells)

    gap = float(data["staging"]["gap"])
    table_top = float(data["staging"]["table_top_z"])
    aabbs = []
    for piece in data["pieces"]:
        bounds = PieceBounds(tuple(piece["local_bounds"][0]), tuple(piece["local_bounds"][1]))
        staging = piece["staging_pose"]
        pose = Pose(tuple(staging["position"]), tuple(staging["orientation_wxyz"]))
        assert pose.position[2] + bounds.local_min[2] == pytest.approx(table_top)
        aabbs.append(bounds.world_aabb(pose))
    for index, first in enumerate(aabbs):
        for second in aabbs[index + 1 :]:
            assert not first.expanded(gap / 2).overlaps(second.expanded(gap / 2))
