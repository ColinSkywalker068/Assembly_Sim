"""DexAgent's bench for the hardware every method shares: two FANUC CRX-10iA/L with I2RT YAM linear_4310 grippers on
the Isaac workcell's table (core/config/presets/dual_arm.json), as a DexAgent setup folder: setup.yaml, scene.xml,
robot/ (the arm and each hand on their own) and assets/. It replaces DexAgent's own setup/ in each trial clone: its
tools and gates read that folder, several of them always DexAgent's own.

Frame (DexAgent's rig): the midpoint of the two arm bases on the tabletop, x toward the right station, z up, so rig =
world shifted down by the tabletop height (RIG_T_WORLD); the layout is not moved. The scene follows DexAgent's bench
(its setup/scene.xml) with these parts:
- `table` / `g_table`, the workcell's table;
- per side, the arm (`robot_<side>_arm_root` at the workcell's base pose, bench/crx.py) and the floating hand DexAgent
  plans with (`robot_<side>_<side>_base_tx` .. `_base_yaw`, bench/yam.py) held to the flange by a position-only weld
  at the mount (flange_T_mount), with a wrist camera `wrist_<side>` where the workcell's wrist camera is, and its
  fingers coupled (`robot_<side>_<side>_fingers_coupled`);
- `ego`, the workcell's agent camera;
- explicit table pairs for every hand collision geom (the arms' come with DexAgent's runtime);
- keyframe `home`: both arms at the workcell's home joints, each hand on its flange, the grippers fully open.
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from core.robots.kinematics import R_to_quat, T
from core.workcell.config import load_workcell_preset

from ....contracts.actions import ARMS
from ....s0.kinematics import yam_model
from . import crx, yam

PRESET = 'dual_arm'
SETUP_YAML, SCENE_XML = 'setup.yaml', 'scene.xml'
FILM_WIDTH_MM = 20.955  # Isaac's default horizontal aperture; rendered pixels are square (reports/2026-10-18_camera_intrinsics.json)
ARM_JOINT_DAMPING = 5.0  # the proposal scene's passive arm joints, as DexAgent's bench has them
TABLE_PAIR = {'solref': '0.01', 'friction': '0.8 0.8 0.005 0.0001 0.0001'}  # the workcell's static friction
TABLE_RGBA = '0.22 0.1 0.04 1'
# The runtime arm (DexAgent's compile_rigid, patched): Isaac's CRX drives are effectively rigid (stiffness 1e15 per
# degree), which an explicit MuJoCo servo cannot be at a 2 ms step, so each joint is a position servo critically
# damped at ARM_SERVO_RAD_S on a reflected inertia ARM_ARMATURE_KG_M2, gravity compensated. PROVISIONAL: neither is
# the real arm's; ARM_SERVO_RAD_S x 2 ms = 0.6 keeps the explicit stiffness stable.
ARM_ARMATURE_KG_M2 = (2.0, 2.0, 1.0, 0.3, 0.3, 0.1)
ARM_SERVO_RAD_S = 300.0
# DexAgent plans its arm moves at 90 deg/s and 12 rad/s^2 (armplan.DEFAULTS); on this bench never above this share of
# S0's limits either (S0 checks the executed delivery). Its runtime commands a linear ramp between frames at
# DEXAGENT_FPS (grasp_robust.FPS and the tools' defaults), which the executor samples at S0's servo rate: a frame's
# whole velocity change falls in one servo period, so S0 reads servo_hz / DEXAGENT_FPS times the planned acceleration
# (twice at 30 fps and 60 Hz), and the plan keeps to DEXAGENT_FPS / servo_hz of the share.
PLAN_SHARE_OF_S0 = 0.8
DEXAGENT_FPS = 30.0
# DexAgent caps its floating planning wrist (40 N, 3 N m) to stand for its compliant arm; the CRX is stiff, so the caps
# stand far above any contact force here (only the servo gains bound the wrist).
RIGID_WRIST_FORCE_N, RIGID_WRIST_TORQUE_NM = 2000.0, 200.0
# Jaw-object contacts (sliding, torsional and rolling, MuJoCo's lengths for the last two): the workcell's static
# friction (core/config/workcell.json), which holds a grasp in PhysX, where friction sticks below it; MuJoCo's soft
# friction creeps under a sustained load at any coefficient, so its grasps are the more pessimistic. A torsional patch
# of a few millimetres for a 12 mm pad (DexAgent's own is 2 cm, for soft fingertips). PROVISIONAL until calibrated.
HAND_PAIR_FRICTION = (0.8, 0.005, 0.0001)
# Tools withheld from DexAgent's agent on this bench (the patch filters them out of its library index): the stations
# never move on the shared hardware, and these need a five-finger hand (a palm to wrap or press with, an opposable
# thumb, curled or spare fingers, human-hand retargeting) or crash on a two-jaw hand.
HIDDEN_TOOLS = ('place_arm_stations', 'attach_arms', 'refit_to_real_rig',
                'add_wrist_cameras', 'flat_hand_pose', 'flat_hand_pose_seated', 'reachable_contact_search',
                'slide_contact_pose', 'pin_pose', 'push_pose_from_point', 'hook_grasp_pose', 'overhang_pinch_pose',
                'plate_edge_pinch_pose', 'second_grasp_pose', 'finger_press_targets', 'finger_press_in_hand_test',
                'grasp_prior_bodex', 'grasp_prior_dexgraspnet', 'human_grasp_pose_prior', 'human_grasp_seeds',
                'retarget_hand_motion', 'physics_informed_retargeting', 'register_camera_hand_overlay',
                'thin_grasp_diagnosis', 'generate_flat_pinches', 'grasp_through_hole_test', 'multi_object_hold_test',
                'combined_hold_state_test', 'flat_press_drag_test', 'grasp_slide_pull_test_preshaped')
DEXAGENT_PLAN_LIMITS = {'vmax_rad_s': math.radians(90.0), 'amax_rad_s2': 12.0, 'flange_amax_m_s2': 4.8}


def _values(values):
    return ' '.join(format(float(v), '.17g') for v in np.asarray(values, dtype=float).ravel())


def tabletop_z(workcell):
    environment = workcell.environment
    return float(environment['table_position'][2]) + float(environment['table_size'][2]) / 2


def rig_T_world(workcell):
    """The rig frame from the world: the world shifted down by the tabletop height (the arm bases stand on it)."""
    bases = [np.asarray(workcell.robot(arm).base_position, dtype=float) for arm in ARMS]
    middle = (bases[0] + bases[1]) / 2
    if not (np.allclose(middle[:2], 0.0) and np.allclose([b[2] for b in bases], tabletop_z(workcell))):
        raise ValueError('the arm bases must stand on the tabletop either side of the world origin')
    return T([0.0, 0.0, -tabletop_z(workcell)], [1.0, 0.0, 0.0, 0.0])


def fovy_deg(focal_length_mm, width_px, height_px):
    """Vertical field of view of an Isaac camera with square pixels and the default film width."""
    focal_px = focal_length_mm / FILM_WIDTH_MM * width_px
    return math.degrees(2 * math.atan(height_px / 2 / focal_px))


def look_at(position, target, up=(0.0, 0.0, 1.0)):
    """Pose of a camera at `position` looking at `target` (MuJoCo and USD alike: looking along -z, y up)."""
    z = np.asarray(position, dtype=float) - np.asarray(target, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, dtype=float), z)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    pose[:3, 3] = position
    return pose


def cameras(workcell, flange_T_mount):
    """{'ego': (rig_T_camera, fovy), 'wrist_<side>': (mount_T_camera, fovy)} from the workcell's cameras."""
    width, height = workcell.camera_resolution
    agent = workcell.cameras['agent']
    out = {'ego': (rig_T_world(workcell) @ look_at(agent['position'], agent['look_at']),
                   fovy_deg(float(agent['focal_length']), width, height))}
    for arm in ARMS:
        wrist = workcell.cameras[f'{arm}_wrist']
        flange_T_camera = look_at(wrist['position'], wrist['look_at_local'])
        out[f'wrist_{arm}'] = (np.linalg.inv(flange_T_mount) @ flange_T_camera,
                               fovy_deg(float(wrist['focal_length']), width, height))
    return out


def home(workcell, hand):
    """Per arm: (home joints, rig_T_wrist), the wrist on the flange at the arm's home."""
    model = yam_model(workcell=workcell)
    out = {}
    for arm in ARMS:
        joints = np.asarray(workcell.robot(arm).home_joint_positions, dtype=float)
        flange = model.arms[arm].frames(joints)[0]['flange']
        out[arm] = joints, rig_T_world(workcell) @ flange @ hand.flange_T_mount
    return out


def wrist_joints(rig_T_wrist):
    """The floating wrist's joint values for a wrist pose: position, then intrinsic XYZ angles (DexAgent's)."""
    return np.r_[rig_T_wrist[:3, 3], Rotation.from_matrix(rig_T_wrist[:3, :3]).as_euler('XYZ')]


def _pose(element, pose):
    element.set('pos', _values(pose[:3, 3]))
    element.set('quat', _values(R_to_quat(pose[:3, :3])))
    return element


def _arm(links, arm, root_pose):
    root = crx.arm_element(links, prefix=f'robot_{arm}_arm_', root_pose=root_pose)
    for joint in root.iter('joint'):
        joint.set('damping', _values([ARM_JOINT_DAMPING]))
    return root


def scene_xml(workcell, links, hand):
    """The bench scene (module docstring) as an ElementTree root, and its keyframe's joint values in order."""
    root = ET.Element('mujoco', model='crx_yam_bench')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', impratio='5', integrator='implicitfast')
    width, height = workcell.camera_resolution
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, 'headlight', ambient='0.35 0.35 0.35', diffuse='0.45 0.45 0.45')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', solref='0.005', solimp='0.99 0.999 0.0005')
    asset = ET.SubElement(root, 'asset')
    files = crx.mesh_files(links)
    for arm in ARMS:
        for link in links:
            ET.SubElement(asset, 'mesh', name=f'robot_{arm}_arm_{link.name}', file=f'{crx.MESH_DIR}/{files[link.name]}')
        asset.extend(yam.assets(hand, arm, prefix=f'robot_{arm}_'))
    world = ET.SubElement(root, 'worldbody')
    views = cameras(workcell, hand.flange_T_mount)
    _pose(ET.SubElement(world, 'camera', name='ego', fovy=_values([views['ego'][1]])), views['ego'][0])
    size = np.asarray(workcell.environment['table_size'], dtype=float)
    table = ET.SubElement(world, 'body', name='table', pos=_values([0.0, 0.0, -size[2] / 2]))
    ET.SubElement(table, 'geom', name='g_table', type='box', size=_values(size / 2), rgba=TABLE_RGBA)
    rig = rig_T_world(workcell)
    poses = home(workcell, hand)
    key = []
    for arm in ARMS:
        settings = workcell.robot(arm)
        world.append(yam.floating_hand(hand, arm, prefix=f'robot_{arm}_',
                                       camera=(f'wrist_{arm}', *views[f'wrist_{arm}'])))
        key += [*wrist_joints(poses[arm][1]), 0.0, 0.0]
        world.append(_arm(links, arm, rig @ T(settings.base_position, settings.base_orientation_wxyz)))
        key += list(poses[arm][0])
    contact = ET.SubElement(root, 'contact')
    for arm in ARMS:
        for geom in world.iter('geom'):
            name = geom.get('name') or ''
            if name.startswith(f'robot_{arm}_collision_hand_{arm}_'):
                ET.SubElement(contact, 'pair', name=f'{name}_g_table', geom1='g_table', geom2=name, **TABLE_PAIR)
    equality = ET.SubElement(root, 'equality')
    relpose = np.r_[hand.flange_T_mount[:3, 3], R_to_quat(hand.flange_T_mount[:3, :3])]
    for arm in ARMS:
        ET.SubElement(equality, 'weld', body1=f'robot_{arm}_arm_gripper', body2=f'robot_{arm}_{arm}_base_yaw',
                      anchor='0 0 0', relpose=_values(relpose), torquescale='0')
        equality.append(yam.coupling(arm, prefix=f'robot_{arm}_'))
    actuator = ET.SubElement(root, 'actuator')
    for arm in ARMS:
        actuator.extend(yam.actuators(hand, arm, prefix=f'robot_{arm}_'))
    keyframe = ET.SubElement(root, 'keyframe')
    ET.SubElement(keyframe, 'key', name='home', qpos=_values(key))
    ET.indent(root)
    return root, np.asarray(key)


def runtime_robot(hand, s0_limits):
    """The bench patch's runtime settings: the arm servos, the planning limits and the finger drive."""
    armature = np.asarray(ARM_ARMATURE_KG_M2, dtype=float)
    def per_joint(key):
        return np.broadcast_to(np.asarray(s0_limits[key], dtype=float), armature.shape)

    vmax = np.minimum(DEXAGENT_PLAN_LIMITS['vmax_rad_s'], PLAN_SHARE_OF_S0 * per_joint('max_joint_velocity_rad_s'))
    frames = DEXAGENT_FPS / float(s0_limits['servo_hz'])
    amax = np.minimum(DEXAGENT_PLAN_LIMITS['amax_rad_s2'],
                      PLAN_SHARE_OF_S0 * frames * per_joint('max_joint_acceleration_rad_s2'))
    return {'schema': 'dexagent.rigid_robot.v1', 'arm_control': 'joint_position', 'arm_inertia_xml': crx.ARM_XML,
            'arm_kp': (ARM_SERVO_RAD_S ** 2 * armature).tolist(), 'arm_kv': (2 * ARM_SERVO_RAD_S * armature).tolist(),
            'arm_armature': armature.tolist(), 'arm_damping': [0.0] * len(armature), 'arm_gravcomp': True,
            'arm_vmax_rad_s': vmax.tolist(), 'arm_amax_rad_s2': amax.tolist(),
            'arm_vmax_limit_rad_s': per_joint('max_joint_velocity_rad_s').tolist(),
            'arm_amax_limit_rad_s2': per_joint('max_joint_acceleration_rad_s2').tolist(),
            'flange_amax_m_s2': DEXAGENT_PLAN_LIMITS['flange_amax_m_s2'],
            'finger_force_n': hand.drive.max_force_n, 'finger_speed_m_s': hand.drive.max_speed_m_s,
            'finger_armature': None, 'wrist_force_n': RIGID_WRIST_FORCE_N, 'wrist_torque_nm': RIGID_WRIST_TORQUE_NM,
            'hand_pair_friction': list(HAND_PAIR_FRICTION), 'fixed_stations': True, 'hidden_tools': list(HIDDEN_TOOLS),
            **s0_band(hand, s0_limits)}


def s0_band(hand, s0_limits):
    """S0's contact band for the planner (rig frame: the table top at z = 0), and the S0 check of every delivery:
    {'s0_check': True, 's0_band': {...}} when S0's limits include the contact phase, else {}. The finger box and the
    TCP are S0's (s0.kinematics.yam_tool_geometry), in the flange frame, which is the runtime arm's last body."""
    keys = ('table_clearance_m', 'tcp_min_height_m', 'contact_max_speed_m_s')
    if not all(s0_limits.get(key) is not None for key in keys):
        return {}
    from ....s0.kinematics import yam_tool_geometry
    flange_T_tcp, _, box = yam_tool_geometry(float(hand.flange_T_mount[0, 3]), hand.gripper)
    fingers = box['parts']['fingers']
    return {'s0_check': True, 's0_band': {
        'table_top_z_m': 0.0, **{key: float(s0_limits[key]) for key in keys}, 'plan_share': PLAN_SHARE_OF_S0,
        'finger_box_min': list(fingers['min']), 'finger_box_max': list(fingers['max']),
        'flange_tcp': flange_T_tcp[:3, 3].tolist()}}


def setup_yaml(workcell, links, hand, s0_limits):
    """setup.yaml for the bench: the keys DexAgent's code reads (scene, stations' base body and pose, table, home
    keyframe, ego camera, physics) and the bench patch's `runtime_robot`, plus the others as documentation."""
    rig = rig_T_world(workcell)
    size = np.asarray(workcell.environment['table_size'], dtype=float)
    width, height = workcell.camera_resolution
    views = cameras(workcell, hand.flange_T_mount)
    poses = home(workcell, hand)
    stations = {}
    for arm in ARMS:
        settings = workcell.robot(arm)
        base = rig @ T(settings.base_position, settings.base_orientation_wxyz)
        stations[arm] = {
            'arm': 'FANUC CRX-10iA/L, 6 DOF', 'hand': 'I2RT YAM linear_4310 parallel gripper (two slide fingers)',
            'arm_xml': crx.ARM_XML, 'base_body': f'robot_{arm}_arm_root', 'base_pos': base[:3, 3].tolist(),
            'base_quat_wxyz': R_to_quat(base[:3, :3]).tolist(),
            'arm_joints': [f'robot_{arm}_arm_{joint}' for joint in crx.JOINTS],
            'flange_body': f'robot_{arm}_arm_gripper', 'hand_root_body': f'robot_{arm}_{arm}_base_yaw',
            'hand_xml': f'robot/yam_{arm}.xml',
            'wrist_joints': [f'robot_{arm}_{arm}_{name}' for _, name, _, _ in yam.WRIST],
            'finger_joint_prefix': f'robot_{arm}_{arm}_', 'home_arm_rad': poses[arm][0].tolist(),
            'hand_mount': {'pos': hand.flange_T_mount[:3, 3].tolist(),
                           'quat_wxyz': R_to_quat(hand.flange_T_mount[:3, :3]).tolist()}}
    ego, ego_fovy = views['ego']
    focal = height / 2 / math.tan(math.radians(ego_fovy) / 2)
    data = {
        'name': 'crx_yam_bench',
        'scene': SCENE_XML,
        'frame': {'origin': 'midpoint of the two arm bases on the tabletop (the Isaac workcell world shifted down by '
                            f'{tabletop_z(workcell)} m)', 'x': 'toward the right station', 'z': 'up',
                  'tabletop_z': 0.0, 'world_z_m': tabletop_z(workcell)},
        'table': {'body': 'table', 'geom': 'g_table', 'center': [0.0, 0.0, -size[2] / 2],
                  'half_extents': (size / 2).tolist()},
        'stations': stations,
        'home': {'keyframe': 'home', 'description': 'both arms at the workcell home joints, grippers fully open'},
        'cameras': {'ego': {'width': int(width), 'height': int(height), 'fovy_deg': ego_fovy, 'focal_px': focal,
                            'pos': ego[:3, 3].tolist(), 'quat_wxyz': R_to_quat(ego[:3, :3]).tolist()},
                    **{f'wrist_{arm}': {'parent_body': f'robot_{arm}_{arm}_base_yaw',
                                        'pos': views[f'wrist_{arm}'][0][:3, 3].tolist(),
                                        'quat_wxyz': R_to_quat(views[f'wrist_{arm}'][0][:3, :3]).tolist(),
                                        'fovy_deg': views[f'wrist_{arm}'][1]} for arm in ARMS}},
        'physics': {'timestep': 0.002, 'integrator': 'implicitfast', 'impratio': 5},
        'runtime_robot': runtime_robot(hand, s0_limits),
        'robot': {'body_prefix': 'robot_', 'hand_root_frame': 'the gripper mount frame (+z the approach), on the flange '
                                                             'at hand_mount'},
    }
    return data


def _scalar(value):
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError(f'{value} has no place in setup.yaml')
        text = repr(float(value))
        mantissa, _, exponent = text.partition('e')
        # YAML 1.1 (PyYAML) reads 1e-05 as a string: its floats need a point in the mantissa
        return f'{mantissa}.0e{exponent}' if exponent and '.' not in mantissa else text
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f'cannot write {type(value).__name__} to setup.yaml')


def yaml_lines(data, indent=0):
    """`data` (nested dicts of scalars and flat lists) as block YAML, lists in flow style."""
    lines, pad = [], '  ' * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines += [f'{pad}{key}:', *yaml_lines(value, indent + 1)]
        elif isinstance(value, (list, tuple, np.ndarray)):
            lines.append(f"{pad}{key}: [{', '.join(_scalar(item) for item in value)}]")
        else:
            lines.append(f'{pad}{key}: {_scalar(value)}')
    return lines


def write_bench(setup_dir, s0_limits, workcell=None, links=None, hand=None):
    """Write the bench under `setup_dir` (a DexAgent setup folder) and return the scene's path. `s0_limits` has
    S0's max_joint_velocity_rad_s and max_joint_acceleration_rad_s2 (per joint, or one for all)."""
    setup_dir = Path(setup_dir)
    workcell = load_workcell_preset(PRESET) if workcell is None else workcell
    links = crx.crx_links() if links is None else links
    hand = yam.yam_hand() if hand is None else hand
    crx.write_arm(setup_dir, links)
    for arm in ARMS:
        yam.write_hand(setup_dir, arm, hand)
    root, _ = scene_xml(workcell, links, hand)
    path = setup_dir / SCENE_XML
    path.write_text(ET.tostring(root, encoding='unicode') + '\n')
    (setup_dir / SETUP_YAML).write_text('\n'.join(yaml_lines(setup_yaml(workcell, links, hand, s0_limits))) + '\n')
    (setup_dir / 'bench.json').write_text(json.dumps({'schema': 'assembly_sim.dexagent_bench.v1', 'preset': PRESET,
                                                      'adapter_thickness_m': float(hand.flange_T_mount[0, 3])},
                                                     indent=1) + '\n')
    return path


BENCH_README = ('# CRX + YAM bench (generated)\n\nWritten by tasks/dual_arm_crag_skull/baselines/dexagent/bench/scene.py: two '
                'FANUC CRX-10iA/L with I2RT YAM linear_4310 grippers on the Isaac workcell table. Do not edit.\n')


def install_bench(clone, s0_limits):
    """Replace a DexAgent clone's setup/ with this bench (its tools and gates read that folder, several of them always
    the clone's own)."""
    import shutil
    clone = Path(clone)
    if not (clone / 'tool_library').is_dir() or not (clone / 'setup' / 'setup.yaml').is_file():
        raise ValueError(f'{clone} is not a DexAgent clone with a setup/ folder')
    shutil.rmtree(clone / 'setup')
    (clone / 'setup').mkdir()
    path = write_bench(clone / 'setup', s0_limits)
    (clone / 'setup' / 'README.md').write_text(BENCH_README)
    return path
