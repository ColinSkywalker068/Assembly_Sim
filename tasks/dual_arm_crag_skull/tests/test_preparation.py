from pathlib import Path
import numpy as np
import pytest
import trimesh
from tasks.dual_arm_crag_skull.preparation import load_skull, require_connected, prepare
from tasks.dual_arm_breakingbad.domain.layout import load_layout


def scene_file(tmp_path, count=6):
    scene = trimesh.Scene()
    for i in range(count):
        transform = np.eye(4)
        transform[0, 3] = i * 1.2
        scene.add_geometry(trimesh.creation.box(), node_name=f'node_{i}', transform=transform)
    path = tmp_path / 'synthetic.glb'
    scene.export(path)
    return path


def test_loader_preserves_transform_and_identity(tmp_path):
    assembly = load_skull(scene_file(tmp_path))
    assert len(assembly.fragments) == 6
    assert len({f.name for f in assembly.fragments}) == 6
    assert assembly.fragments[1].ground_truth_transform[0, 3] == pytest.approx(1.2)


def test_reject_small_scene_and_corner_connection(tmp_path):
    with pytest.raises(ValueError, match='six mesh'):
        load_skull(scene_file(tmp_path, 5))
    with pytest.raises(ValueError, match='six-connected'):
        require_connected([(0, 0, 0), (1, 1, 1)])
    require_connected([(0, 0, 0), (1, 0, 0)])


def test_prepare_roundtrip(tmp_path):
    source = scene_file(tmp_path)
    output = tmp_path / 'prepared'
    layout_path = prepare(source, output, pitch=.005, target_length=.2)
    layout = load_layout(layout_path)
    assert len(layout.fragment_names) == 6
    assert len(list(output.glob('*.stl'))) == 6
    assert all(p['grasp'] is None for p in layout.data['pieces'])
    assert layout.data['processing']['preserve_all_components']
    with pytest.raises(FileExistsError):
        prepare(source, output, pitch=.005, target_length=.2)


def test_reject_nonmanifold_before_publish(tmp_path):
    source = scene_file(tmp_path)
    with pytest.raises(ValueError, match='non-manifold'):
        prepare(source, tmp_path / 'bad', pitch=.01, target_length=.2)
    assert not (tmp_path / 'bad').exists()
