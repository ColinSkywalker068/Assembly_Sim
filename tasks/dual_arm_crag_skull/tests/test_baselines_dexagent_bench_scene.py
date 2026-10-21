"""DexAgent's CRX + YAM bench scene and setup.yaml (baselines/dexagent/bench/scene.py). MuJoCo checks need the DexAgent
venv and clone (DEXAGENT_PYTHON, DEXAGENT_SOURCE) and skip without them."""
import json
import math
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.workcell.config import load_workcell_preset
from tasks.dual_arm_crag_skull.baselines.dexagent import dexenv
from tasks.dual_arm_crag_skull.baselines.dexagent.bench import crx, scene, yam
from tasks.dual_arm_crag_skull.s0.kinematics import yam_model

AVAILABLE, WHY = dexenv.available()
needs_dexagent = pytest.mark.skipif(not AVAILABLE, reason=WHY)
WORKCELL = load_workcell_preset('dual_arm')
REPORT = Path(__file__).resolve().parents[1] / 'reports' / '2026-10-18_camera_intrinsics.json'
# S0's link rotations are off orthonormal by about 3e-8 (float32 probe quaternions, test_baselines_dexagent_bench)
S0_PRECISION = 1e-7


S0_LIMITS = {'max_joint_velocity_rad_s': [2.0943951, 2.0943951, 3.1415927, 3.1415927, 3.1415927, 3.1415927],
             'max_joint_acceleration_rad_s2': 20.0, 'servo_hz': 60}


@pytest.fixture(scope='module')
def bench(tmp_path_factory):
    out = tmp_path_factory.mktemp('bench')
    return out, scene.write_bench(out, S0_LIMITS)


def test_the_runtime_arm_is_a_stable_critically_damped_servo_planned_within_s0():
    from tasks.dual_arm_crag_skull.sim.yam_usd import FingerDrive
    runtime = scene.runtime_robot(yam.yam_hand(), S0_LIMITS)
    armature = np.asarray(runtime['arm_armature'])
    kp, kv = np.asarray(runtime['arm_kp']), np.asarray(runtime['arm_kv'])
    np.testing.assert_allclose(kv, 2 * np.sqrt(kp * armature))  # critical on the armature alone
    assert (np.sqrt(kp / armature) * 0.002 <= 0.6 + 1e-12).all()  # explicit stiffness at MuJoCo's 2 ms step
    vmax, amax = np.asarray(runtime['arm_vmax_rad_s']), np.asarray(runtime['arm_amax_rad_s2'])
    assert vmax.shape == amax.shape == (6,)
    assert (vmax <= 0.8 * np.asarray(S0_LIMITS['max_joint_velocity_rad_s']) + 1e-12).all()
    # S0 reads DexAgent's 30 fps ramp at its 60 Hz servo rate: a frame's velocity change falls in one servo period,
    # twice the planned acceleration (executor.py), so the plan keeps to 0.8 x 30/60 of S0's
    assert (vmax <= math.radians(90) + 1e-12).all() and (amax <= 0.8 * 20.0 * 30 / 60 + 1e-12).all()
    assert amax == pytest.approx([8.0] * 6)
    drive = FingerDrive()
    assert (runtime['finger_force_n'], runtime['finger_speed_m_s']) == (drive.max_force_n, drive.max_speed_m_s)
    assert runtime['arm_control'] == 'joint_position' and runtime['finger_armature'] is None
    np.testing.assert_allclose(runtime['arm_vmax_limit_rad_s'], S0_LIMITS['max_joint_velocity_rad_s'])  # S0's own,
    assert runtime['arm_amax_limit_rad_s2'] == [20.0] * 6                                                # for the audits
    slow = scene.runtime_robot(yam.yam_hand(), {**S0_LIMITS, 'max_joint_acceleration_rad_s2': 5.0})
    assert slow['arm_amax_rad_s2'] == pytest.approx([2.0] * 6)  # follows a stricter S0
    slower = scene.runtime_robot(yam.yam_hand(), {**S0_LIMITS, 'max_joint_velocity_rad_s': [1.0] * 6})
    assert slower['arm_vmax_rad_s'] == pytest.approx([0.8] * 6)  # in speed too (the CRX's own leave 90 deg/s binding)


def test_the_runtime_carries_s0s_contact_band_for_the_planner_and_the_delivery_check():
    from tasks.dual_arm_crag_skull.s0.kinematics import yam_tool_geometry
    contact = {'table_clearance_m': 0.01, 'tcp_min_height_m': 0.02, 'contact_max_speed_m_s': 0.05}
    runtime = scene.runtime_robot(yam.yam_hand(), {**S0_LIMITS, **contact})
    band = runtime['s0_band']
    assert runtime['s0_check'] is True and band['table_top_z_m'] == 0.0  # the rig's tabletop
    assert {key: band[key] for key in contact} == contact and band['plan_share'] == 0.8
    flange_T_tcp, _, box = yam_tool_geometry()  # S0's, in the flange frame (the runtime arm's last body)
    assert band['finger_box_min'] == pytest.approx(box['parts']['fingers']['min'], abs=1e-12)
    assert band['finger_box_max'] == pytest.approx(box['parts']['fingers']['max'], abs=1e-12)
    np.testing.assert_allclose(band['flange_tcp'], flange_T_tcp[:3, 3], atol=1e-12)
    assert 's0_band' not in scene.runtime_robot(yam.yam_hand(), S0_LIMITS)  # no contact phase in S0: no band


def test_the_rig_is_the_world_shifted_down_to_the_tabletop():
    rig = scene.rig_T_world(WORKCELL)
    assert scene.tabletop_z(WORKCELL) == pytest.approx(0.75)
    np.testing.assert_allclose(rig, np.diag([1.0, 1, 1, 1]) + np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -0.75],
                                                                     [0, 0, 0, 0]]))


def test_the_ego_camera_is_the_workcells_agent_camera_as_isaac_renders_it():
    """Against the renderer's own projection and view (reports/2026-10-18_camera_intrinsics.json)."""
    measured = json.loads(REPORT.read_text())['smoke']['cameras']['agent']
    pose, fovy = scene.cameras(WORKCELL, yam.yam_hand().flange_T_mount)['ego']
    width, height = WORKCELL.camera_resolution
    fy = measured['intrinsics'][1][1]
    assert fovy == pytest.approx(math.degrees(2 * math.atan(height / 2 / fy)), abs=1e-6)
    optical = np.asarray(measured['world_T_camera_optical'])  # OpenCV: z forward, y down
    world = np.linalg.inv(scene.rig_T_world(WORKCELL)) @ pose  # MuJoCo: z backward, y up
    np.testing.assert_allclose(world[:3, 3], optical[:3, 3], atol=1e-9)
    np.testing.assert_allclose(world[:3, :3] @ np.diag([1.0, -1.0, -1.0]), optical[:3, :3], atol=1e-9)


def test_wrist_cameras_ride_on_the_wrist_where_the_workcells_are_on_the_flange():
    hand = yam.yam_hand()
    views = scene.cameras(WORKCELL, hand.flange_T_mount)
    for arm in ('left', 'right'):
        mount_T_camera, fovy = views[f'wrist_{arm}']
        flange_T_camera = hand.flange_T_mount @ mount_T_camera
        settings = WORKCELL.cameras[f'{arm}_wrist']
        np.testing.assert_allclose(flange_T_camera[:3, 3], settings['position'], atol=1e-12)
        look = np.asarray(settings['look_at_local']) - np.asarray(settings['position'])
        np.testing.assert_allclose(-flange_T_camera[:3, 2], look / np.linalg.norm(look), atol=1e-12)
        assert fovy == pytest.approx(math.degrees(2 * math.atan(720 / (18 / 20.955 * 1920))))


@pytest.mark.parametrize('value, text', [(1e-05, '1.0e-05'), (2.5e-17, '2.5e-17'), (0.75, '0.75'), (3, '3'),
                                         (True, 'true'), ('a "b"', '"a \\"b\\""'), (np.float64(-0.0), '-0.0'),
                                         (None, 'null')])
def test_setup_yaml_scalars_read_back_as_written(value, text):
    assert scene._scalar(value) == text


def test_setup_yaml_refuses_what_it_cannot_write():
    with pytest.raises(ValueError):
        scene._scalar(float('nan'))
    with pytest.raises(TypeError):
        scene._scalar(object())


def _floats(text):
    return np.array([float(v) for v in text.split()])


def test_the_scene_names_what_dexagent_looks_up(bench):
    _, path = bench
    root = ET.parse(path).getroot()
    bodies = {body.get('name') for body in root.iter('body')}
    for arm in ('left', 'right'):
        assert {f'robot_{arm}_arm_{name}' for name in ('root', 'base', 'link1', 'link5', 'gripper')} <= bodies
        assert {f'robot_{arm}_{arm}_{name}' for name in ('base_tx', 'base_yaw', 'hand_C_MC', 'thumb_pad')} <= bodies
    assert root.find(".//camera[@name='ego']") is not None and root.find(".//geom[@name='g_table']") is not None
    assert [key.get('name') for key in root.iter('key')] == ['home']
    hand_geoms = {geom.get('name') for geom in root.iter('geom') if 'collision_hand_' in (geom.get('name') or '')}
    pairs = {(pair.get('geom1'), pair.get('geom2')) for pair in root.iter('pair')}
    assert pairs == {('g_table', name) for name in hand_geoms} and len(hand_geoms) > 20
    welds = list(root.iter('weld'))
    assert [(w.get('body1'), w.get('body2'), w.get('torquescale')) for w in welds] == [
        ('robot_left_arm_gripper', 'robot_left_left_base_yaw', '0'),
        ('robot_right_arm_gripper', 'robot_right_right_base_yaw', '0')]
    coupled = [e for e in root.find('equality') if e.tag == 'joint']
    assert [(e.get('name'), e.get('joint1'), e.get('joint2'), e.get('polycoef')) for e in coupled] == [
        (f'robot_{arm}_{arm}_fingers_coupled', f'robot_{arm}_{arm}_index_FE', f'robot_{arm}_{arm}_thumb_FE',
         '0 1 0 0 0') for arm in ('left', 'right')]
    assert all({key: e.get(key) for key in ('solref', 'solimp')} == yam.COUPLING for e in coupled)


def test_the_home_keyframe_puts_each_wrist_on_its_flange_at_the_mount(bench):
    """The key's floating-wrist values against S0's FK of the key's arm joints, and the weld's relative pose."""
    _, path = bench
    root = ET.parse(path).getroot()
    key = _floats(root.find('.//key').get('qpos'))
    hand = yam.yam_hand()
    model = yam_model(workcell=WORKCELL)
    rig = scene.rig_T_world(WORKCELL)
    relpose = _floats(root.find('.//weld').get('relpose'))
    weld = np.eye(4)
    weld[:3, 3] = relpose[:3]
    weld[:3, :3] = Rotation.from_quat(np.r_[relpose[4:], relpose[3]]).as_matrix()
    np.testing.assert_allclose(weld, hand.flange_T_mount, atol=1e-12)
    for index, arm in enumerate(('left', 'right')):
        wrist, fingers, joints = key[14 * index:14 * index + 6], key[14 * index + 6:14 * index + 8], \
            key[14 * index + 8:14 * index + 14]
        np.testing.assert_allclose(joints, WORKCELL.robot(arm).home_joint_positions)
        assert fingers.tolist() == [0.0, 0.0]  # fully open
        pose = np.eye(4)
        pose[:3, 3] = wrist[:3]
        pose[:3, :3] = Rotation.from_euler('XYZ', wrist[3:]).as_matrix()
        expected = rig @ model.arms[arm].frames(joints)[0]['flange'] @ hand.flange_T_mount
        np.testing.assert_allclose(pose, expected, atol=S0_PRECISION)


def test_the_arm_bases_are_the_workcells_in_the_rig(bench):
    _, path = bench
    root = ET.parse(path).getroot()
    for arm in ('left', 'right'):
        base = root.find(f".//body[@name='robot_{arm}_arm_root']")
        settings = WORKCELL.robot(arm)
        np.testing.assert_allclose(_floats(base.get('pos')), np.asarray(settings.base_position) - [0, 0, 0.75],
                                   atol=1e-12)
        w, x, y, z = _floats(base.get('quat'))
        expected = np.asarray(settings.base_orientation_wxyz)
        assert abs(np.dot([w, x, y, z], expected)) == pytest.approx(1.0, abs=1e-12)


@needs_dexagent
def test_mujoco_loads_the_bench_and_its_home_satisfies_the_welds(bench, tmp_path):
    """MuJoCo: the scene compiles, setup.yaml reads back, each wrist sits exactly where its weld wants it at home,
    nothing touches the table, and DexAgent reads the flange-to-hand mounts and each hand from it."""
    out, path = bench
    script = tmp_path / 'load.py'
    script.write_text(textwrap.dedent('''
        import json, sys
        sys.path.insert(0, sys.argv[2])
        import mujoco, numpy as np, yaml
        from tool_library.common import hand as H, rigid_robot as RR
        model = mujoco.MjModel.from_xml_path(sys.argv[1] + '/scene.xml')
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        setup = yaml.safe_load(open(sys.argv[1] + '/setup.yaml'))
        mounts = RR.trusted_mounts(sys.argv[1] + '/scene.xml')
        rigs = {side: sorted(H.HandRig.build(model, f'robot_{side}_', side).finger_act) for side in ('left', 'right')}
        weld = [float(np.abs(data.efc_pos[data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]).max())]
        table = model.geom('g_table').id
        print(json.dumps({'nq': model.nq, 'weld_violation': weld[0], 'contacts': int(data.ncon),
                          'mounts': {s: [p.tolist(), q.tolist()] for s, (p, q) in mounts.items()}, 'rigs': rigs,
                          'yaml_stations': {s: v['base_pos'] for s, v in setup['stations'].items()},
                          'yaml_scene': setup['scene'], 'ego_fovy': float(model.cam_fovy[model.camera('ego').id])}))
    '''))
    result = subprocess.run([dexenv.dexagent_python(), '-I', '-B', str(script), str(out), dexenv.dexagent_source()],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    loaded = json.loads(result.stdout)
    assert loaded['nq'] == 28 and loaded['weld_violation'] < S0_PRECISION and loaded['contacts'] == 0
    hand = yam.yam_hand()
    for side in ('left', 'right'):
        p, q = loaded['mounts'][side]
        np.testing.assert_allclose(p, hand.flange_T_mount[:3, 3], atol=S0_PRECISION)
        assert loaded['rigs'][side] == ['index_FE', 'thumb_FE']
        np.testing.assert_allclose(loaded['yaml_stations'][side],
                                   np.asarray(WORKCELL.robot(side).base_position) - [0, 0, 0.75], atol=1e-12)
    assert loaded['yaml_scene'] == 'scene.xml'
    assert loaded['ego_fovy'] == pytest.approx(scene.cameras(WORKCELL, hand.flange_T_mount)['ego'][1], abs=1e-9)
