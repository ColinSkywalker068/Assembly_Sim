from pathlib import Path

import numpy as np
import pytest

from fragment_assembly.preprocess import write_processed_assembly
from fragment_assembly.voxelize import ProcessedAssembly, ProcessedFragment, VoxelizationConfig


def _fragment(name="piece_custom", valid=True):
    return ProcessedFragment(
        name=name,
        source_name=f"{name}.obj",
        cells=((0, 0, 0),),
        local_pivot_cells=(0.5, 0.5, 0.0),
        local_bounds=((-0.05, -0.05, 0.0), (0.05, 0.05, 0.1)),
        vertices=(
            np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.1]], dtype=np.float32)
            if valid
            else np.empty((0, 3), dtype=np.float32)
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        goal_pose={"position": (0.05, 0.05, 0.0), "orientation_wxyz": (1.0, 0.0, 0.0, 0.0)},
        color=(0.8, 0.2, 0.1),
        grasp=None,
    )


def _processed(fragment=None):
    return ProcessedAssembly(
        object_id="fixture",
        source={"loader": "test", "path": "/raw/missing", "fragment_names": ("source.obj",)},
        config=VoxelizationConfig(pitch=0.1),
        source_to_canonical=np.eye(4),
        grid_origin=(0.0, 0.0, 0.0),
        grid_shape=(1, 1, 1),
        assembly_bounds=((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)),
        fragments=(fragment or _fragment(),),
    )


def _workcell():
    return {
        "config_path": "/repo/scene.json",
        "environment": {
            "table_position": [0.0, 0.0, 0.725],
            "table_size": [2.1, 1.3, 0.05],
            "workspace_margin": 0.08,
            "staging_gap": 0.04,
        },
        "robots": {
            "left": {"base_position": [-0.78, 0.0, 0.75]},
            "right": {"base_position": [0.78, 0.0, 0.75]},
        },
    }


def test_export_writes_one_npz_per_fragment_without_plate_or_ghost(tmp_path):
    output = tmp_path / "generated"

    layout_path = write_processed_assembly(_processed(), output, _workcell())

    assert layout_path == output / "layout.json"
    assert sorted(path.name for path in output.iterdir()) == ["layout.json", "piece_custom.npz"]
    mesh = np.load(output / "piece_custom.npz")
    assert set(mesh.files) == {"v", "f"}


def test_failed_export_does_not_publish_partial_directory(tmp_path):
    output = tmp_path / "generated"

    with pytest.raises(ValueError, match="empty visual mesh"):
        write_processed_assembly(_processed(_fragment(valid=False)), output, _workcell())

    assert not output.exists()


def test_nonempty_output_requires_overwrite(tmp_path):
    output = tmp_path / "generated"
    output.mkdir()
    (output / "user-file.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        write_processed_assembly(_processed(), output, _workcell())

    assert (output / "user-file.txt").read_text(encoding="utf-8") == "keep"
