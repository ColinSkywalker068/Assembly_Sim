import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from core.workcell.config import load_workcell_preset
from tasks.dual_arm_breakingbad.domain.layout import load_layout
from tasks.dual_arm_breakingbad.preprocessing.preprocess import write_processed_assembly
from tasks.dual_arm_breakingbad.preprocessing.voxelize import (
    ProcessedAssembly,
    ProcessedFragment,
    VoxelizationConfig,
)


def _processed(source_path="/raw/source/sample"):
    fragment = ProcessedFragment(
        name="odd_fragment",
        source_name="piece_7.obj",
        cells=((1, 2, 0), (1, 2, 1)),
        local_pivot_cells=(1.5, 2.5, 0.0),
        local_bounds=((-0.05, -0.05, 0.0), (0.05, 0.05, 0.2)),
        vertices=np.asarray(
            [[-0.05, -0.05, 0.0], [0.05, -0.05, 0.0], [0.05, 0.05, 0.2]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        goal_pose={"position": (0.15, 0.25, 0.0), "orientation_wxyz": (1.0, 0.0, 0.0, 0.0)},
        color=(0.8, 0.2, 0.1),
        grasp=None,
    )
    return ProcessedAssembly(
        object_id="object_17",
        source={"loader": "breaking_bad", "path": source_path, "fragment_names": ("piece_7.obj",)},
        config=VoxelizationConfig(pitch=0.1, target_length=0.4, seed=3),
        source_to_canonical=np.diag((0.4, 0.4, 0.4, 1.0)),
        grid_origin=(-0.1, -0.2, 0.0),
        grid_shape=(4, 5, 2),
        assembly_bounds=((-0.1, -0.2, 0.0), (0.3, 0.3, 0.2)),
        fragments=(fragment,),
    )


def _workcell():
    return load_workcell_preset("dual_arm")


def _export(tmp_path, source_path="/raw/source/sample"):
    layout_path = write_processed_assembly(_processed(source_path), tmp_path / "out", _workcell())
    return layout_path, load_layout(layout_path)


def test_schema_v2_round_trip_contains_required_object_state(tmp_path):
    layout_path, layout = _export(tmp_path)
    data = json.loads(layout_path.read_text(encoding="utf-8"))

    assert layout.schema_version == 2
    assert layout.object_id == "object_17"
    assert layout.fragment_names == ("odd_fragment",)
    assert data["fragment_count"] == 1
    assert data["processing"]["pitch"] == 0.1
    assert data["normalization"]["grid_shape"] == [4, 5, 2]
    assert data["pieces"][0]["goal_pose"]["position"] == [0.15, 0.25, 0.0]
    assert "staging_pose" in data["pieces"][0]


def test_mesh_paths_resolve_relative_to_layout(tmp_path):
    layout_path, layout = _export(tmp_path)

    assert layout.mesh_path("odd_fragment") == layout_path.parent / "odd_fragment.npz"


def test_processed_directory_remains_loadable_after_move(tmp_path):
    layout_path, _ = _export(tmp_path)
    moved = tmp_path / "relocated" / "object"
    moved.parent.mkdir()
    layout_path.parent.rename(moved)

    layout = load_layout(moved / "layout.json")

    assert layout.mesh_path("odd_fragment") == moved / "odd_fragment.npz"
    assert layout.source_path == Path("/raw/source/sample")


@pytest.mark.parametrize(
    "cells",
    [
        [[1, 2, 0], [1, 2, 0]],
        [[1, 2]],
        [[1, 2, 0.5]],
    ],
)
def test_validator_rejects_duplicate_or_malformed_cells(tmp_path, cells):
    layout_path, _ = _export(tmp_path)
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    data["pieces"][0]["cells"] = cells
    layout_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="voxel cell|duplicate"):
        load_layout(layout_path, validate_assets=False)


@pytest.mark.parametrize("field", ["goal_pose", "staging_pose"])
def test_validator_rejects_missing_goal_or_staging_pose(tmp_path, field):
    layout_path, _ = _export(tmp_path)
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    del data["pieces"][0][field]
    layout_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_layout(layout_path, validate_assets=False)


def test_validator_does_not_consult_source_dataset(tmp_path):
    absent_source = tmp_path / "deleted-source"
    layout_path, _ = _export(tmp_path, str(absent_source))

    layout = load_layout(layout_path)

    assert layout.source_path == absent_source


def test_validator_rejects_cells_owned_by_multiple_fragments(tmp_path):
    layout_path, _ = _export(tmp_path)
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    duplicate = deepcopy(data["pieces"][0])
    duplicate["name"] = "second_fragment"
    duplicate["mesh"] = "second_fragment.npz"
    np.savez(
        layout_path.parent / duplicate["mesh"],
        v=np.asarray([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0.1]], dtype=np.float32),
        f=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    data["pieces"].append(duplicate)
    data["fragment_count"] = 2
    layout_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="owned by multiple fragments"):
        load_layout(layout_path, validate_assets=False)


def test_validator_rejects_cells_outside_common_grid(tmp_path):
    layout_path, _ = _export(tmp_path)
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    data["pieces"][0]["cells"] = [[4, 2, 0]]
    data["pieces"][0]["voxel_count"] = 1
    layout_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="outside common grid"):
        load_layout(layout_path, validate_assets=False)


@pytest.mark.parametrize("mesh_path", ["/tmp/external.npz", "../external.npz"])
def test_validator_rejects_mesh_paths_outside_layout_directory(tmp_path, mesh_path):
    layout_path, _ = _export(tmp_path)
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    data["pieces"][0]["mesh"] = mesh_path
    layout_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="relative to the layout directory"):
        load_layout(layout_path, validate_assets=False)


@pytest.mark.parametrize("field", ["local_pivot_cells", "local_bounds", "color"])
def test_validator_requires_fragment_geometry_metadata(tmp_path, field):
    layout_path, _ = _export(tmp_path)
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    del data["pieces"][0][field]
    layout_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_layout(layout_path, validate_assets=False)
