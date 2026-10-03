import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.workcell.config import load_workcell_preset
from tasks.dual_arm_breakingbad.paths import require_external_output
from tasks.dual_arm_breakingbad.preprocessing.preprocess import write_processed_assembly
from tasks.dual_arm_breakingbad.preprocessing.voxelize import (
    ProcessedAssembly,
    ProcessedFragment,
    VoxelizationConfig,
)


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
    return load_workcell_preset("dual_arm")


def _three_processed():
    fragments = tuple(
        replace(
            _fragment(f"piece_{index}"),
            cells=((index, 0, 0),),
            local_pivot_cells=(index + 0.5, 0.5, 0.0),
        )
        for index in range(3)
    )
    return replace(
        _processed(),
        grid_shape=(3, 1, 1),
        assembly_bounds=((0.0, 0.0, 0.0), (0.3, 0.1, 0.1)),
        fragments=fragments,
    )


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


def test_export_records_bilateral_lanes_and_a_two_plus_one_assignment(tmp_path):
    output = tmp_path / "generated"

    layout_path = write_processed_assembly(_three_processed(), output, _workcell())

    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    assert layout["staging"]["algorithm"] == "bilateral_lanes_v1"
    assert layout["staging"]["pad_bounds"] == [-0.256, 0.256, -0.108, 0.308]
    assert layout["staging"]["lane_centers_x"] == [-0.42, 0.42]
    assert layout["staging"]["assignments"] == {
        "piece_0": "left",
        "piece_1": "left",
        "piece_2": "right",
    }
    poses = {piece["name"]: piece["staging_pose"]["position"] for piece in layout["pieces"]}
    assert poses["piece_0"][0] == pytest.approx(-0.42)
    assert poses["piece_1"][0] == pytest.approx(-0.42)
    assert poses["piece_2"][0] == pytest.approx(0.42)


def test_export_records_core_preset_identity_not_an_absolute_config_path(tmp_path):
    layout_path = write_processed_assembly(_processed(), tmp_path / "generated", _workcell())
    data = json.loads(layout_path.read_text(encoding="utf-8"))

    assert data["workcell"] == {"preset": "dual_arm", "schema_version": 1}
    assert "workcell_config" not in data["staging"]


def test_processed_output_inside_repository_is_rejected():
    repository_output = Path(__file__).resolve().parents[3] / "generated"

    with pytest.raises(ValueError, match="outside the repository"):
        require_external_output(repository_output)
