"""DexAgent's CRX + YAM bench: the YAM linear_4310 as a DexAgent hand (baselines/dexagent/bench/yam.py). MuJoCo checks
need the DexAgent venv and clone (DEXAGENT_PYTHON, DEXAGENT_SOURCE) and skip without them."""
import json
import subprocess
import textwrap
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.robots.linear_4310 import TIPS, load_linear_4310
from tasks.dual_arm_crag_skull.baselines.dexagent import dexenv
from tasks.dual_arm_crag_skull.baselines.dexagent.bench import yam
from tasks.dual_arm_crag_skull.sim.yam_usd import FingerDrive, tip_points_in_link

AVAILABLE, WHY = dexenv.available()
needs_dexagent = pytest.mark.skipif(not AVAILABLE, reason=WHY)
GRIPPER = load_linear_4310()


@pytest.fixture(scope='module')
def hand():
    return yam.yam_hand()


@pytest.fixture(scope='module')
def hand_xml(hand, tmp_path_factory):
    out = tmp_path_factory.mktemp('yam')
    return out, {side: yam.write_hand(out, side, hand) for side in ('left', 'right')}


def _floats(text):
    return np.array([float(v) for v in text.split()])


def _pose(element):
    pose = np.eye(4)
    pose[:3, 3] = _floats(element.get('pos', '0 0 0'))
    w, x, y, z = _floats(element.get('quat', '1 0 0 0'))
    pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    return pose


def mount_poses(root, side, closing):
    """Wrist (mount) frame poses of the gripper body, the tips and the pads, both slides at `closing`."""
    body = root.find(f".//body[@name='{side}_hand_C_MC']")
    poses = {'gripper': _pose(body)}
    for tip in TIPS:
        link = body.find(f"body[@name='{side}_{yam.FINGERS[tip]}_link']")
        slide = np.eye(4)
        slide[:3, 3] = _floats(link.find('joint').get('axis')) * closing
        poses[tip] = poses['gripper'] @ _pose(link) @ slide
        poses[f'{tip}_pad'] = poses[tip] @ _pose(link.find(f"body[@name='{side}_{yam.FINGERS[tip]}_pad']"))
    return poses


def test_the_hand_follows_dexagents_naming(hand_xml, hand):
    _, paths = hand_xml
    for side, path in paths.items():
        root = ET.parse(path).getroot()
        joints = [joint.get('name') for joint in root.find('worldbody').iter('joint')]
        wrist = [f'{side}_{name}' for name in ('pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z')]
        assert joints == [*wrist, f'{side}_thumb_FE', f'{side}_index_FE']
        actuators = {element.get('joint'): element.get('name') for element in root.find('actuator')}
        assert actuators == {joint: f'{joint}_position' for joint in joints}
        bodies = [body.get('name') for body in root.iter('body')]
        assert bodies == [f'{side}_{name}' for name in ('base_tx', 'base_ty', 'base_tz', 'base_roll', 'base_pitch',
                                                         'base_yaw', 'hand_C_MC', 'thumb_link', 'thumb_pad',
                                                         'index_link', 'index_pad')]
        sites = {site.get('name') for site in root.iter('site')}
        assert sites == {f'{side}_tcp', f'{side}_thumb_tip', f'{side}_index_tip', f'{side}_thumb_pip',
                         f'{side}_index_pip'}
        collision = sorted(geom.get('name') for geom in root.iter('geom') if geom.get('name'))
        expected = [f'collision_hand_{side}_hand_C_MC_hull']
        expected += [f'collision_hand_{side}_thumb_{k}' for k in range(len(hand.pieces['tip_left']))]
        expected += [f'collision_hand_{side}_index_{k}' for k in range(len(hand.pieces['tip_right']))]
        assert collision == sorted(expected)
        for geom in root.iter('geom'):  # nothing collides but through explicit pairs
            assert (geom.get('contype'), geom.get('conaffinity')) == ('0', '0')
            assert geom.get('name') is None or geom.get('group') == yam.COLLISION


def test_every_piece_is_drawn_and_collides_and_the_official_meshes_are_hidden(hand_xml, hand):
    """DexAgent checks a robot against objects by the hulls of its visible geoms: those are the pieces."""
    _, paths = hand_xml
    root = ET.parse(paths['left']).getroot()
    visible = sorted(geom.get('mesh') for geom in root.iter('geom') if geom.get('group') == yam.VISIBLE)
    collision = sorted(geom.get('mesh') for geom in root.iter('geom') if geom.get('group') == yam.COLLISION)
    assert visible == collision and len(visible) == 1 + sum(len(pieces) for pieces in hand.pieces.values())
    hidden = sorted(geom.get('mesh') for geom in root.iter('geom') if geom.get('group') == yam.HIDDEN)
    assert hidden == ['left_gripper_mesh', 'left_tip_left_mesh', 'left_tip_right_mesh']


def test_only_the_pad_pieces_ride_on_the_pad_body(hand_xml, hand):
    _, paths = hand_xml
    root = ET.parse(paths['right']).getroot()
    for tip in TIPS:
        finger = yam.FINGERS[tip]
        pad = root.find(f".//body[@name='right_{finger}_pad']")
        on_pad = [geom.get('name') for geom in pad.findall('geom') if geom.get('name')]
        assert on_pad == [f'collision_hand_right_{finger}_{k}' for k in hand.pad_pieces(tip)]
        assert [hand.pieces[tip][k].heights_m for k in hand.pad_pieces(tip)][-1][1] <= yam.PAD_HEIGHT_M + 0.002
        assert pad.find('site').get('name') == f'right_{finger}_tip' and pad.find('joint') is None


@pytest.mark.parametrize('closing', [0.0, 0.01, 0.03])
def test_the_slides_close_the_fingers_as_isaacs_joints_open_them(hand_xml, closing):
    """This hand's q (closing from fully open) is Isaac's travel_m - q: the tips sit where Isaac's do."""
    _, paths = hand_xml
    poses = mount_poses(ET.parse(paths['left']).getroot(), 'left', closing)
    for tip in TIPS:
        expected = GRIPPER.link_frames[tip].copy()
        expected[:3, 3] += GRIPPER.slide_axes[tip] * (GRIPPER.travel_m - closing)
        np.testing.assert_allclose(poses[tip], expected, atol=1e-9)
        np.testing.assert_allclose(poses[f'{tip}_pad'], expected, atol=1e-12)


def test_the_slides_end_where_the_pads_touch(hand_xml, hand):
    _, paths = hand_xml
    root = ET.parse(paths['right']).getroot()
    for joint in ('right_thumb_FE', 'right_index_FE'):
        low, high = _floats(root.find(f".//joint[@name='{joint}']").get('range'))
        assert low == 0.0 and high == pytest.approx(GRIPPER.travel_m - GRIPPER.contact_joint_m, abs=1e-12)
    assert GRIPPER.width_between(GRIPPER.contact_joint_m, GRIPPER.contact_joint_m) == pytest.approx(0.0, abs=1e-12)


def test_tip_sites_are_a_pad_radius_behind_the_middle_of_each_pad_face(hand_xml):
    """Where DexAgent puts the site of a pad without a capsule (its pinch standoffs and the pads check assume it)."""
    _, paths = hand_xml
    root = ET.parse(paths['left']).getroot()
    for tip in TIPS:
        finger = yam.FINGERS[tip]
        site = _floats(root.find(f".//site[@name='left_{finger}_tip']").get('pos'))  # in the tip's link frame
        approach, _, lateral = yam.tip_axes(GRIPPER, tip)
        inward = -GRIPPER.link_frames[tip][:3, :3].T @ GRIPPER.slide_axes[tip]
        points = tip_points_in_link(GRIPPER, tip, 0.0003)
        end = (points @ approach).max()
        assert end - site @ approach == pytest.approx(yam.PAD_HEIGHT_M / 2, abs=1e-6)
        near = points[np.abs(points @ approach - site @ approach) <= 0.0005]
        assert (near @ inward).max() - site @ inward == pytest.approx(yam.PAD_RADIUS_M, abs=2e-4)
        across = near[(near @ inward) > (near @ inward).max() - 0.0005] @ lateral  # the face
        assert across.min() + 0.004 < site @ lateral < across.max() - 0.004
        knuckle = _floats(root.find(f".//site[@name='left_{finger}_pip']").get('pos'))
        np.testing.assert_allclose(site - knuckle, (yam.KNUCKLE_HEIGHT_M - yam.PAD_HEIGHT_M / 2) * approach,
                                   atol=1e-12)


def test_the_fingers_have_isaacs_drive(hand_xml):
    _, paths = hand_xml
    drive = FingerDrive()
    root = ET.parse(paths['left']).getroot()
    for finger in ('thumb', 'index'):
        actuator = root.find(f".//general[@name='left_{finger}_FE_position']")
        assert _floats(actuator.get('gainprm'))[0] == drive.stiffness_n_m
        np.testing.assert_allclose(_floats(actuator.get('biasprm')), [0.0, -drive.stiffness_n_m, -drive.damping_n_s_m])
        assert actuator.get('forcelimited') == 'true'
        np.testing.assert_allclose(_floats(actuator.get('forcerange')), [-drive.max_force_n, drive.max_force_n])


def test_the_index_follows_the_thumb_as_stiff_as_the_contacts(hand_xml):
    """The official MJCF's coupling of its two slides (polycoef 0 1 0 0 0), tip_right following tip_left as Isaac's
    mimic joint does, at the bench's contact softness rather than MuJoCo's default."""
    _, paths = hand_xml
    for side, path in paths.items():
        joints = list(ET.parse(path).getroot().find('equality'))
        assert [(e.tag, e.get('name'), e.get('joint1'), e.get('joint2'), e.get('polycoef')) for e in joints] == [
            ('joint', f'{side}_fingers_coupled', f'{side}_index_FE', f'{side}_thumb_FE', '0 1 0 0 0')]
        assert {key: joints[0].get(key) for key in ('solref', 'solimp')} == yam.COUPLING
    assert yam.FINGERS == {'tip_left': 'thumb', 'tip_right': 'index'}


COUPLING_PROBE = textwrap.dedent('''
    """Close the hand, its wrist rigid, on a fixed box between the pads, the references ramped at the drive\'s speed cap
    as DexAgent\'s runtime keeps them: argv hand.xml box_centre(3) box_axes(9, columns) box_half(3) coupling(none |
    default | the hand\'s) -> the two slides at rest."""
    import json, sys
    import xml.etree.ElementTree as ET
    import mujoco
    root = ET.parse(sys.argv[1]).getroot()
    root.find('compiler').set('meshdir', sys.argv[1].rsplit('/robot/', 1)[0] + '/')
    ET.SubElement(root, 'option', timestep='0.002', integrator='implicitfast', impratio='5', gravity='0 0 0')
    for body in root.iter('body'):
        for joint in [j for j in body.findall('joint') if '_pos_' in j.get('name') or '_rot_' in j.get('name')]:
            body.remove(joint)
    actuator = root.find('actuator')
    for servo in [a for a in actuator if '_pos_' in a.get('name') or '_rot_' in a.get('name')]:
        actuator.remove(servo)
    coupled = root.find('equality')[0]
    if sys.argv[5] == 'none':
        root.remove(root.find('equality'))
    elif sys.argv[5] == 'default':
        del coupled.attrib['solref'], coupled.attrib['solimp']
    ET.SubElement(root.find('worldbody'), 'geom', name='box', type='box', pos=sys.argv[2], size=sys.argv[4],
                  quat=sys.argv[3], contype='0', conaffinity='0')
    contact = ET.SubElement(root, 'contact')
    for geom in root.iter('geom'):
        name = geom.get('name') or ''
        if name.startswith('collision_hand_left_') and 'hand_C_MC' not in name:
            ET.SubElement(contact, 'pair', geom1='box', geom2=name, solref='0.005', solimp='0.99 0.999 0.0005',
                          friction='0.8 0.8 0.005 0.0001 0.0001')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    fingers = ('thumb', 'index')
    closed = {f: model.jnt_range[model.joint(f'left_{f}_FE').id][1] for f in fingers}
    for _ in range(750):
        for f in fingers:
            data.ctrl[model.actuator(f'left_{f}_FE_position').id] = min(closed[f], 0.1 * data.time)
        mujoco.mj_step(model, data)
    print(json.dumps({f: float(data.qpos[model.joint(f'left_{f}_FE').qposadr[0]]) for f in fingers}))
''')


@needs_dexagent
def test_the_coupled_fingers_stop_together_on_a_box_off_the_middle(hand_xml, hand, tmp_path):
    """A box 5 mm off the middle towards the thumb: coupled, the thumb touches first and both stop there, 10 mm wider
    than the box (check_yam.predicted_stop_width, as Isaac measures it); uncoupled, or coupled at MuJoCo\'s default
    softness, the index servo closes on the box\'s other face."""
    from tasks.dual_arm_crag_skull.sim.check_yam import BOX_SIDE_M, predicted_stop_width
    from core.robots.kinematics import R_to_quat
    _, paths = hand_xml
    script = tmp_path / 'probe.py'
    script.write_text(COUPLING_PROBE)
    gripper, depth, width, shift = hand.gripper, 0.005, 0.040, 0.005
    closing = gripper.slide_axes['tip_left'] / np.linalg.norm(gripper.slide_axes['tip_left'])
    approach = gripper.mount_T_tcp[:3, 2]
    centre = gripper.mount_T_tcp[:3, 3] - approach * depth + closing * shift
    axes = np.stack([closing, np.cross(approach, closing), approach], axis=1)
    text = lambda values: ' '.join(format(float(v), '.17g') for v in values)
    spans = {}
    for coupling in ('hand', 'none', 'default'):
        result = subprocess.run([dexenv.dexagent_python(), '-I', '-B', str(script), str(paths['left']), text(centre),
                                 text(R_to_quat(axes)), text([width / 2, BOX_SIDE_M / 2, BOX_SIDE_M / 2]), coupling],
                                capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr[-3000:]
        slides = json.loads(result.stdout)
        opening = [gripper.travel_m - slides[yam.FINGERS[tip]] for tip in TIPS]
        spans[coupling] = (gripper.width_between(*opening), slides['thumb'] - slides['index'])
    predicted = predicted_stop_width(gripper, depth, width, BOX_SIDE_M / 2, shift)
    assert predicted == pytest.approx(width + 2 * shift, abs=1e-4)
    assert spans['hand'][0] == pytest.approx(predicted, abs=5e-4) and abs(spans['hand'][1]) < 3e-4
    for loose in ('none', 'default'):
        assert spans[loose][0] == pytest.approx(width, abs=5e-4) and abs(spans[loose][1]) > 0.009


def test_the_wrist_is_the_mount_frame_with_z_the_approach(hand_xml, hand):
    """DexAgent's hands face +z of the wrist: the gripper hangs in its mount frame, the TCP straight down +z."""
    _, paths = hand_xml
    root = ET.parse(paths['left']).getroot()
    body = root.find(".//body[@name='left_hand_C_MC']")
    np.testing.assert_allclose(_pose(body), GRIPPER.link_frames['gripper'], atol=1e-12)
    tcp = _pose(body) @ np.r_[_floats(root.find(".//site[@name='left_tcp']").get('pos')), 1.0]
    np.testing.assert_allclose(tcp[:3], GRIPPER.mount_T_tcp[:3, 3], atol=1e-12)
    assert np.allclose(GRIPPER.mount_T_tcp[:3, 2], [0, 0, 1]) and tcp[2] > 0.14


@needs_dexagent
def test_dexagents_own_hand_rig_reads_the_hand(hand_xml, hand, tmp_path):
    """DexAgent's HandRig.build: the two fingers, their tips, the palm, every collision piece, and as tip geoms only
    the pad's."""
    _, paths = hand_xml
    script = tmp_path / 'rig.py'
    script.write_text(textwrap.dedent('''
        import json, sys
        sys.path.insert(0, sys.argv[2])
        import mujoco
        from tool_library.common import hand as H
        model = mujoco.MjModel.from_xml_path(sys.argv[1])
        rig = H.HandRig.build(model, '', 'left')
        print(json.dumps({'fingers': sorted(rig.finger_act), 'tips': sorted(rig.fingertip_sites),
                          'palm': model.body(rig.palm_body).name, 'collision': len(rig.collision_gids),
                          'roles': {name: H.joint_role(name) for name in rig.finger_act},
                          'closed': rig.finger_targets(1.0), 'open': rig.finger_targets(0.0),
                          'tip_geoms': {f: sorted(model.geom(g).name for g in rig.tip_geoms(f)) for f in ('thumb', 'index')},
                          'wrist_body': model.body(H.wrist_body(rig)).name}))
    '''))
    result = subprocess.run([dexenv.dexagent_python(), '-I', '-B', str(script), str(paths['left']),
                             dexenv.dexagent_source()], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    out = json.loads(result.stdout)
    assert out['fingers'] == ['index_FE', 'thumb_FE'] and out['tips'] == ['index', 'thumb']
    assert out['palm'] == 'left_hand_C_MC' and set(out['roles'].values()) == {'flex'}
    assert out['wrist_body'] == 'left_base_yaw'
    assert out['collision'] == 1 + sum(len(pieces) for pieces in hand.pieces.values())
    assert out['open'] == {'thumb_FE': 0.0, 'index_FE': 0.0}
    assert out['closed'] == pytest.approx({name: hand.closing_range_m[1] for name in ('thumb_FE', 'index_FE')})
    for tip in TIPS:
        finger = yam.FINGERS[tip]
        assert out['tip_geoms'][finger] == sorted(f'collision_hand_left_{finger}_{k}' for k in hand.pad_pieces(tip))
