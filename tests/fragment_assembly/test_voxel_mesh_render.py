import json

import numpy as np

from fragment_assembly.voxel_mesh_render import load_voxel_assembly, render_voxel_assembly


def _write_layout(root):
    root.mkdir()
    np.savez(
        root / "piece_0.npz",
        v=np.asarray([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=np.float32),
        f=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    data = {
        "schema_version": 2,
        "object_id": "voxel-test",
        "source": {"loader": "breaking_bad", "path": "/unavailable/source"},
        "processing": {"pitch": 0.1},
        "normalization": {
            "source_to_canonical": np.eye(4).tolist(),
            "grid_origin": [0, 0, 0],
            "grid_shape": [2, 2, 2],
            "assembly_bounds": [[0, 0, 0], [0.2, 0.2, 0.2]],
        },
        "fragment_count": 1,
        "pieces": [
            {
                "name": "piece_0",
                "source_name": "piece_0.obj",
                "mesh": "piece_0.npz",
                "color": [0.2, 0.4, 0.8],
                "cells": [[0, 0, 0]],
                "voxel_count": 1,
                "local_pivot_cells": [0.5, 0.5, 0.0],
                "local_bounds": [[-0.05, -0.05, 0], [0.05, 0.05, 0.1]],
                "goal_pose": {
                    "position": [1.0, 2.0, 3.0],
                    "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "staging_pose": {
                    "position": [0.0, 0.0, 0.75],
                    "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "grasp": None,
            }
        ],
        "staging": {"algorithm": "test"},
    }
    path = root / "layout.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_voxel_assembly_applies_goal_pose_to_exported_visual_mesh(tmp_path):
    layout_path = _write_layout(tmp_path / "assembly")

    assembly = load_voxel_assembly(layout_path)

    assert assembly.object_id == "voxel-test"
    assert np.allclose(assembly.fragments[0].mesh.vertices[0], [1.0, 2.0, 3.0])
    assert assembly.fragments[0].mesh.faces.tolist() == [[0, 1, 2]]


def test_voxel_renderer_writes_six_views_and_contact_sheet(tmp_path):
    layout_path = _write_layout(tmp_path / "assembly")

    result = render_voxel_assembly(layout_path, tmp_path / "renders", image_size=160)

    assert len(result.view_paths) == 6
    assert all(path.is_file() and path.stat().st_size > 0 for path in result.view_paths)
    assert result.contact_sheet.is_file() and result.contact_sheet.stat().st_size > 0
