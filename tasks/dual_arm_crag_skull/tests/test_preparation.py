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


def test_simulation_only_never_exports_failed_print_geometry(tmp_path):
    import json
    source = scene_file(tmp_path)
    output = tmp_path/'simulation'
    prepare(source,output,pitch=.01,target_length=.2,simulation_only=True)
    manifest = json.loads((output/'geometry_manifest.json').read_text())
    assert manifest['simulation_only'] and not manifest['manufacturing_validated']
    assert not list(output.glob('*.stl'))
    assert not all(manifest['topology_checks'].values())


def test_straight_assembly_repair_is_recorded_and_needs_a_real_anchor(tmp_path):
    import json
    source = scene_file(tmp_path)
    anchor = load_skull(source).fragments[0].name
    with pytest.raises(ValueError, match='anchor'):
        prepare(source, tmp_path / 'bad', pitch=.01, target_length=.2, simulation_only=True,
                straight_assembly_anchor='fragment_unknown')
    assert not (tmp_path / 'bad').exists()
    output = tmp_path / 'straight'
    layout = load_layout(prepare(source, output, pitch=.01, target_length=.2, simulation_only=True,
                                 straight_assembly_anchor=anchor))
    manifest = json.loads((output / 'geometry_manifest.json').read_text())
    assert manifest['ownership_repair']['anchor'] == anchor and manifest['ownership_repair']['flipped'] == 0
    assert layout.data['source']['ownership_repair'] == manifest['ownership_repair']
    prepare(source, tmp_path / 'plain', pitch=.01, target_length=.2, simulation_only=True)
    assert json.loads((tmp_path / 'plain' / 'geometry_manifest.json').read_text())['ownership_repair'] is None


def test_a_given_assembly_order_is_the_layouts_plan(tmp_path):
    import json
    from tasks.dual_arm_crag_skull.interlock_repair import insertion_orders
    source = scene_file(tmp_path)
    names = [fragment.name for fragment in load_skull(source).fragments]
    anchor = names[0]
    with pytest.raises(ValueError, match='straight-assembly anchor'):
        prepare(source, tmp_path / 'no_anchor', pitch=.01, target_length=.2, simulation_only=True,
                assembly_order=names[1:])
    with pytest.raises(ValueError, match='once'):
        prepare(source, tmp_path / 'short', pitch=.01, target_length=.2, simulation_only=True,
                straight_assembly_anchor=anchor, assembly_order=names[2:])
    with pytest.raises(ValueError, match='cannot go in straight'):  # the 10 mm cells join the boxes: some must go first
        prepare(source, tmp_path / 'stuck', pitch=.01, target_length=.2, simulation_only=True,
                straight_assembly_anchor=anchor, assembly_order=names[:0:-1])
    assert not any((tmp_path / name).exists() for name in ('no_anchor', 'short', 'stuck'))
    default = load_layout(prepare(source, tmp_path / 'default', pitch=.01, target_length=.2, simulation_only=True,
                                  straight_assembly_anchor=anchor))
    own = default.data['source']['ownership_repair']
    assert own['plan_source'] == 'repair'
    owner_of = {tuple(cell): piece['name'] for piece in default.data['pieces'] for cell in piece['cells']}
    orders = insertion_orders(owner_of, {step['fragment']: step['exit_direction'] for step in own['plan']}, anchor)
    other = next(list(order) for order in orders if list(order) != [step['fragment'] for step in own['plan']])
    layout = load_layout(prepare(source, tmp_path / 'given', pitch=.01, target_length=.2, simulation_only=True,
                                 straight_assembly_anchor=anchor, assembly_order=other))
    repair = layout.data['source']['ownership_repair']
    assert [step['fragment'] for step in repair['plan']] == other and repair['plan_source'] == 'given'
    assert {step['fragment']: step['exit_direction'] for step in repair['plan']} == {
        step['fragment']: step['exit_direction'] for step in own['plan']}
    assert [piece['cells'] for piece in layout.data['pieces']] == [piece['cells'] for piece in default.data['pieces']]
    assert json.loads((tmp_path / 'given' / 'geometry_manifest.json').read_text())['ownership_repair'] == repair
