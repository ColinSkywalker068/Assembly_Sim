"""The gravity check of experiments/assembly_settle.py; needs MuJoCo (the DexAgent venv), else it skips:
envs/dexagent/bin/python -B -m pytest -p no:cacheprovider tasks/dual_arm_crag_skull/tests/test_assembly_settle.py"""
import json

import pytest

mujoco = pytest.importorskip('mujoco')

from tasks.dual_arm_crag_skull.experiments.assembly_settle import drift, orders_of, settle_orders  # noqa: E402

SCENE = """<mujoco><option timestep="0.002"/><worldbody>
<geom name="floor" type="plane" size="1 1 0.1"/>
<body name="a"><freejoint name="a_free"/><geom type="box" size="0.02 0.02 0.02" density="650"/></body>
<body name="b"><freejoint name="b_free"/><geom type="box" size="0.02 0.02 0.02" density="650"/></body>
</worldbody></mujoco>"""
JOINTS = {'a': 'a_free', 'b': 'b_free'}
GOALS = {'a': [0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0], 'b': [0.1, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]}  # b floats 0.28 m up


def test_a_box_on_the_floor_stays_and_one_in_the_air_falls():
    model = mujoco.MjModel.from_xml_string(SCENE)
    resting = drift(model, JOINTS, GOALS, {'a'}, 0.5)
    # mm and deg (MuJoCo's soft contact lets a resting box sink 0.2 mm); b, not placed, falls far away
    assert set(resting) == {'a'} and max(resting['a']) < 0.5
    both = drift(model, JOINTS, GOALS, {'a', 'b'}, 0.5)
    assert both['b'][0] > 250 and max(both['a']) < 0.5


def test_each_orders_worst_step_is_read_against_the_tolerances(tmp_path):
    (tmp_path / 'inputs' / 'stage2').mkdir(parents=True)
    (tmp_path / 'inputs' / 'stage2' / 'scene.xml').write_text(SCENE)
    manifest = {'objects': [{'object_id': name, 'free_joint': joint, 'goal_rig_wxyz': GOALS[name]}
                            for name, joint in JOINTS.items()],
                'task': {'position_tolerance_m': 0.003, 'rotation_tolerance_deg': 3.0}}
    (tmp_path / 'inputs' / 'export_manifest.json').write_text(json.dumps(manifest))
    report = settle_orders(tmp_path, [['b', 'a'], ['a', 'b']], 0.5, 2)
    rows = {tuple(row['order']): row for row in report['orders']}
    keys = [row['worst_over_tolerance'] for row in report['orders']]
    assert keys == sorted(keys) and sorted(rows) == [('a', 'b'), ('b', 'a')]
    assert rows['a', 'b']['worst_mm'] > 250 and rows['a', 'b']['worst_over_tolerance'] > 80
    assert [step['placed'] for step in rows['a', 'b']['steps']] == [['a'], ['a', 'b']]
    assert max(rows['a', 'b']['steps'][0]['drift_mm_deg']['a']) < 0.5
    assert sorted(report['sets']) == ['a', 'a,b', 'b']


def test_a_fragment_not_placed_is_moved_out_of_the_way():
    model = mujoco.MjModel.from_xml_string(SCENE)
    same_place = {'a': GOALS['a'], 'b': GOALS['a']}  # b's goal is a's place: left there, it would burst a away
    assert max(drift(model, JOINTS, same_place, {'a'}, 0.5)['a']) < 0.5


ARM = SCENE.replace('<body name="a">', """<body name="robot_arm" pos="-0.12 0 0.07">
<joint name="robot_hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.01" density="8000"/></body>
<body name="a">""").replace('</mujoco>',
                         '<keyframe><key qpos="0  0 0 0.02 1 0 0 0  0.1 0 0.3 1 0 0 0"/></keyframe></mujoco>')


def test_the_robot_is_held_at_its_keyframe():
    # a heavy arm hinged 2 cm above a: let go, it would swing down onto a
    model = mujoco.MjModel.from_xml_string(ARM)
    assert max(drift(model, JOINTS, GOALS, {'a'}, 0.5)['a']) < 0.5


def test_orders_come_from_the_export_or_from_its_layouts_plan(tmp_path):
    import hashlib
    cup = [[x, y, 0] for x in range(3) for y in range(3)] + [[0, 1, 1], [2, 1, 1]]
    layout = {'pieces': [{'name': 'c', 'cells': cup}, {'name': 'p', 'cells': [[1, 1, 1]]},
                         {'name': 'q', 'cells': [[3, 1, 0]]}],
              'source': {'ownership_repair': {'anchor': 'c', 'plan': [
                  {'fragment': 'q', 'exit_direction': [1, 0, 0]}, {'fragment': 'p', 'exit_direction': [0, 0, 1]}]}}}
    path = tmp_path / 'layout.json'
    path.write_text(json.dumps(layout))
    manifest = {'order': ['blue', 'green'],
                'objects': [{'fragment_id': 'p', 'object_id': 'green'}, {'fragment_id': 'q', 'object_id': 'blue'}],
                'task': {'layout': str(path), 'layout_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
    assert orders_of(manifest, False) == [['blue', 'green']]
    assert orders_of(manifest, True) == [['green', 'blue'], ['blue', 'green']]  # p and q go in either way
    manifest['task']['layout_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='sha256'):
        orders_of(manifest, True)
