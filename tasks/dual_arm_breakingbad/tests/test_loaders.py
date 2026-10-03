from pathlib import Path

import numpy as np
import pytest
import trimesh

from tasks.dual_arm_breakingbad.preprocessing.loaders import load_breaking_bad, load_crag_glb


def _write_obj(path: Path, offset=(0.0, 0.0, 0.0)) -> None:
    mesh = trimesh.creation.box(extents=(0.2, 0.3, 0.4))
    mesh.apply_translation(offset)
    mesh.export(path)


def test_breaking_bad_discovers_arbitrary_piece_count_in_numeric_order(tmp_path):
    for name in ("piece_10.obj", "piece_2.obj", "piece_0.obj"):
        _write_obj(tmp_path / name)
    _write_obj(tmp_path / "unrelated.obj")

    assembly = load_breaking_bad(tmp_path)

    assert assembly.object_id == tmp_path.name
    assert tuple(fragment.name for fragment in assembly.fragments) == (
        "piece_0",
        "piece_2",
        "piece_10",
    )
    assert tuple(fragment.source_name for fragment in assembly.fragments) == (
        "piece_0.obj",
        "piece_2.obj",
        "piece_10.obj",
    )


def test_breaking_bad_uses_identity_ground_truth_transforms(tmp_path):
    _write_obj(tmp_path / "piece_0.obj", offset=(1.0, 2.0, 3.0))

    fragment = load_breaking_bad(tmp_path, object_id="sample").fragments[0]

    assert np.array_equal(fragment.ground_truth_transform, np.eye(4))
    assert fragment.mesh.centroid == pytest.approx((1.0, 2.0, 3.0))


def test_loader_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no piece_<integer>.*OBJ"):
        load_breaking_bad(tmp_path)


def test_loader_rejects_duplicate_numeric_piece_ids(tmp_path):
    _write_obj(tmp_path / "piece_1.obj")
    _write_obj(tmp_path / "piece_01.obj")

    with pytest.raises(ValueError, match="duplicate piece id 1"):
        load_breaking_bad(tmp_path)


def test_loader_rejects_non_triangle_or_empty_mesh(tmp_path):
    (tmp_path / "piece_0.obj").write_text("# no geometry\n", encoding="utf-8")

    with pytest.raises(ValueError, match="piece_0.obj.*empty triangle mesh"):
        load_breaking_bad(tmp_path)


def test_crag_loader_preserves_scene_node_transforms(tmp_path):
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(extents=(0.1, 0.2, 0.3)),
        node_name="fragment_node",
        geom_name="fragment_geometry",
        transform=trimesh.transformations.translation_matrix((1.0, 2.0, 3.0)),
    )
    path = tmp_path / "assembly.glb"
    scene.export(path)

    assembly = load_crag_glb(path)

    assert len(assembly.fragments) == 1
    assert assembly.fragments[0].source_name == "fragment_geometry"
    np.testing.assert_allclose(
        assembly.fragments[0].ground_truth_transform,
        np.asarray([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    )
