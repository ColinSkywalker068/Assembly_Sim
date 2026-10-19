"""The I2RT YAM linear_4310 as a DexAgent hand, with the Isaac gripper's geometry, masses and drive (sim/yam_usd.py).

DexAgent drives any hand through a naming convention (its tool_library/common/hand.py). This one has two fingers,
thumb (tip_left) and index (tip_right):
- the wrist frame, `<side>_base_yaw` (the last body of the floating wrist `<side>_pos_x` .. `<side>_rot_z` with
  `_position` servos, which DexAgent plans with), is the gripper's mount frame: its +z is the approach, the way
  DexAgent takes a hand's palm to face (a hand retreats along -z); on the arm it sits on the CRX flange at
  flange_T_mount (s0.kinematics);
- `<side>_hand_C_MC` is the gripper body (DexAgent's palm) and `<side>_<finger>_link` each tip, on one slide joint
  `<side>_<finger>_FE` pointing inward: 0 is fully open, so DexAgent's grip fraction (low = open) closes the hand, and
  the range ends where the pads touch. Isaac's joint is the opening instead (contact_joint_m to travel_m): Isaac's q =
  travel_m - this q;
- each pad (the solid last 15 mm of a tip) is a child body `<side>_<finger>_pad` without a joint, carrying the tip
  site: DexAgent treats every collision geom of the tip site's body as the part meant to touch, and the rails above
  the pad are not;
- tip sites `<side>_<finger>_tip` PAD_RADIUS_M behind the middle of each pad face, where DexAgent puts the site of a
  pad it finds no capsule on (hand.pad_radius); `<side>_<finger>_pip` as far behind the finger 40 mm up; `<side>_tcp`;
- a position servo `<joint>_position` per finger with Isaac's stiffness, damping and force cap. The two fingers get
  the same target and nothing couples them, as in Isaac.

Geometry (yam_pieces): each tip's merged convex pieces are both its visible geoms and its collision geoms
`collision_hand_<side>_<finger>_<k>`. DexAgent checks robots against objects by the convex hulls of their visible
geoms, and the hull of an official tip mesh would fill the window between the rails. The body's convex hull is
`collision_hand_<side>_hand_C_MC_hull`, the name under which DexAgent's handgeom would otherwise add a hull of its
own. The official meshes are kept, hidden (group 4). PhysX's speed cap (maxJointVelocity) has no MuJoCo counterpart:
DexAgent's runtime keeps the finger references within it (the bench patch).
"""
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.robots.kinematics import R_to_quat
from core.robots.linear_4310 import ASSET_DIR, TIPS, load_linear_4310

from ....s0.kinematics import YAM_PROVISIONAL_ADAPTER_THICKNESS_M, yam_flange_T_mount
from ....sim.yam_usd import FingerDrive, tip_points_in_link
from .yam_pieces import body_vertices, tip_axes, tip_pieces

FINGERS = {'tip_left': 'thumb', 'tip_right': 'index'}
PAD_HEIGHT_M = 0.015  # the solid end of each tip; above it the window between the rails
PAD_RADIUS_M = 0.0075  # DexAgent's pad radius for a pad without a capsule: TIP_STANDOFF_M / 2 (hand.py:1198, 1526)
KNUCKLE_HEIGHT_M = 0.04
MESH_DIR = 'assets/yam_gripper'
WRIST_KP = 1000.0
WRIST_MASS_KG, WRIST_INERTIA_KG_M2 = 0.1, 0.01  # each floating-wrist body, as DexAgent's own hand files have it
VISIBLE, COLLISION, HIDDEN = '1', '3', '4'
RGBA = '0.2 0.2 0.22 1'
WRIST = (('base_tx', 'pos_x', 'slide', '1 0 0'), ('base_ty', 'pos_y', 'slide', '0 1 0'),
         ('base_tz', 'pos_z', 'slide', '0 0 1'), ('base_roll', 'rot_x', 'hinge', '1 0 0'),
         ('base_pitch', 'rot_y', 'hinge', '0 1 0'), ('base_yaw', 'rot_z', 'hinge', '0 0 1'))
WRIST_RANGE = {'slide': '-5 5', 'hinge': '-6.28 6.28'}


def _values(values):
    return ' '.join(format(float(v), '.17g') for v in np.asarray(values, dtype=float).ravel())


def pad_face(gripper, tip):
    """(middle of the pad's face, inward unit vector) in the tip's link frame: PAD_HEIGHT_M / 2 up from the fingertip
    end, on the pad's inner face, halfway across it."""
    approach, _, lateral = tip_axes(gripper, tip)
    inward = -gripper.link_frames[tip][:3, :3].T @ gripper.slide_axes[tip]
    points = tip_points_in_link(gripper, tip, 0.0005)
    end = (points @ approach).max()
    band = points[np.abs(end - points @ approach - PAD_HEIGHT_M / 2) <= 0.0005]
    across = band @ lateral
    middle = (across.min() + across.max()) / 2
    return (end - PAD_HEIGHT_M / 2) * approach + (band @ inward).max() * inward + middle * lateral, inward


def tip_site(gripper, tip):
    """The tip site in the tip's link frame: PAD_RADIUS_M behind the middle of the pad face."""
    face, inward = pad_face(gripper, tip)
    return face - PAD_RADIUS_M * inward


def knuckle_site(gripper, tip):
    """A point up the finger, KNUCKLE_HEIGHT_M from its end, as far behind the pad face's plane as the tip site."""
    approach, _, _ = tip_axes(gripper, tip)
    return tip_site(gripper, tip) - (KNUCKLE_HEIGHT_M - PAD_HEIGHT_M / 2) * approach


@dataclass(frozen=True)
class YamHand:
    gripper: object
    flange_T_mount: np.ndarray
    drive: FingerDrive
    pieces: dict  # tip -> [TipPiece], from the fingertip end up
    body: np.ndarray  # the body's convex hull vertices

    @property
    def closing_range_m(self):
        """The slide joints' range: fully open to the pads touching."""
        return 0.0, float(self.gripper.travel_m - self.gripper.contact_joint_m)

    def pad_pieces(self, tip):
        """Indices of the pieces on the pad body: those starting below PAD_HEIGHT_M."""
        return [k for k, piece in enumerate(self.pieces[tip]) if piece.heights_m[0] < PAD_HEIGHT_M]


def yam_hand(adapter_thickness_m=YAM_PROVISIONAL_ADAPTER_THICKNESS_M, drive=FingerDrive(), gripper=None):
    gripper = load_linear_4310() if gripper is None else gripper
    return YamHand(gripper, yam_flange_T_mount(adapter_thickness_m, gripper), drive,
                   {tip: tip_pieces(gripper, tip) for tip in TIPS}, body_vertices(gripper))


def _pose(element, pose):
    element.set('pos', _values(pose[:3, 3]))
    element.set('quat', _values(R_to_quat(pose[:3, :3])))
    return element


def _inertial(body, inertial):
    ET.SubElement(body, 'inertial', pos=_values(inertial.center_of_mass_m), quat=_values(inertial.principal_axes_wxyz),
                  mass=format(inertial.mass_kg, '.17g'), diaginertia=_values(inertial.diagonal_inertia_kg_m2))


def assets(hand, side, prefix=''):
    """The hand's meshes: the official ones (files under MESH_DIR) and the convex pieces (inline vertices)."""
    head = f'{prefix}{side}_'
    elements = [ET.Element('mesh', name=f'{head}{link}_mesh', file=f'{MESH_DIR}/{link}.stl')
                for link in ('gripper', *TIPS)]
    elements.append(ET.Element('mesh', name=f'{head}hand_C_MC_hull', vertex=_values(hand.body)))
    for tip, pieces in hand.pieces.items():
        for k, piece in enumerate(pieces):
            elements.append(ET.Element('mesh', name=f'{head}{FINGERS[tip]}_hull_{k}', vertex=_values(piece.vertices)))
    return elements


def _geoms(body, mesh, collision):
    """A visible geom of `mesh` and a collision geom `collision` of it (its contacts come from explicit pairs)."""
    ET.SubElement(body, 'geom', type='mesh', mesh=mesh, contype='0', conaffinity='0', group=VISIBLE, density='0',
                  rgba=RGBA)
    ET.SubElement(body, 'geom', name=collision, type='mesh', mesh=mesh, contype='0', conaffinity='0',
                  group=COLLISION, density='0')


def _official_mesh(body, hand, side, prefix, link):
    _pose(ET.SubElement(body, 'geom', type='mesh', mesh=f'{prefix}{side}_{link}_mesh', contype='0', conaffinity='0',
                        group=HIDDEN, density='0', rgba=RGBA), hand.gripper.mesh_poses[link])


def hand_body(hand, side, prefix=''):
    """`<prefix><side>_hand_C_MC` with its tips, posed in the wrist (mount) frame."""
    gripper, head = hand.gripper, f'{prefix}{side}_'
    mount_T_body = gripper.link_frames['gripper']
    body = _pose(ET.Element('body', name=f'{head}hand_C_MC'), mount_T_body)
    _inertial(body, gripper.inertials['gripper'])
    _official_mesh(body, hand, side, prefix, 'gripper')
    _geoms(body, f'{head}hand_C_MC_hull', f'{prefix}collision_hand_{side}_hand_C_MC_hull')
    body_T_mount = np.linalg.inv(mount_T_body)
    ET.SubElement(body, 'site', name=f'{head}tcp', pos=_values((body_T_mount @ gripper.mount_T_tcp)[:3, 3]),
                  size='0.003', group='3')
    for tip in TIPS:
        finger = FINGERS[tip]
        opened = gripper.link_frames[tip].copy()
        opened[:3, 3] += gripper.slide_axes[tip] * gripper.travel_m
        link = _pose(ET.SubElement(body, 'body', name=f'{head}{finger}_link'), body_T_mount @ opened)
        _inertial(link, gripper.inertials[tip])
        inward = -gripper.link_frames[tip][:3, :3].T @ gripper.slide_axes[tip]
        ET.SubElement(link, 'joint', name=f'{head}{finger}_FE', type='slide', axis=_values(inward),
                      range=_values(hand.closing_range_m))
        _official_mesh(link, hand, side, prefix, tip)
        pad = ET.SubElement(link, 'body', name=f'{head}{finger}_pad')
        on_pad = hand.pad_pieces(tip)
        for k in range(len(hand.pieces[tip])):
            _geoms(pad if k in on_pad else link, f'{head}{finger}_hull_{k}', f'{prefix}collision_hand_{side}_{finger}_{k}')
        ET.SubElement(pad, 'site', name=f'{head}{finger}_tip', pos=_values(tip_site(gripper, tip)), size='0.003',
                      group='3')
        ET.SubElement(link, 'site', name=f'{head}{finger}_pip', pos=_values(knuckle_site(gripper, tip)), size='0.003',
                      group='3')
    return body


def floating_hand(hand, side, prefix='', camera=None):
    """The floating wrist `<prefix><side>_base_tx` .. `_base_yaw` (DexAgent's planning hand) carrying the gripper;
    `camera` (name, wrist_T_camera in MuJoCo's camera convention, vertical field of view in degrees) rides on the
    wrist."""
    head = f'{prefix}{side}_'
    root = parent = None
    for body_name, joint_name, kind, axis in WRIST:
        body = ET.Element('body', name=f'{head}{body_name}')
        ET.SubElement(body, 'inertial', pos='0 0 0', mass=_values([WRIST_MASS_KG]),
                      diaginertia=_values([WRIST_INERTIA_KG_M2] * 3))
        ET.SubElement(body, 'joint', name=f'{head}{joint_name}', type=kind, axis=axis, range=WRIST_RANGE[kind])
        if parent is None:
            root = body
        else:
            parent.append(body)
        parent = body
    if camera is not None:
        name, pose, fovy_deg = camera
        _pose(ET.SubElement(parent, 'camera', name=name, fovy=_values([fovy_deg])), pose)
    parent.append(hand_body(hand, side, prefix))
    return root


def actuators(hand, side, prefix=''):
    """Position servos: the six wrist joints (planning only) and the two fingers (Isaac's drive)."""
    head, elements = f'{prefix}{side}_', []
    mass = sum(inertial.mass_kg for inertial in hand.gripper.inertials.values())
    damping = 2.0 * np.sqrt(WRIST_KP * mass)
    for _, joint_name, kind, _ in WRIST:
        elements.append(ET.Element('general', name=f'{head}{joint_name}_position', joint=f'{head}{joint_name}',
                                   ctrlrange=WRIST_RANGE[kind], biastype='affine', gainprm=_values([WRIST_KP]),
                                   biasprm=_values([0.0, -WRIST_KP, -damping])))
    drive = hand.drive
    for tip in TIPS:
        joint = f'{head}{FINGERS[tip]}_FE'
        elements.append(ET.Element('general', name=f'{joint}_position', joint=joint,
                                   ctrlrange=_values(hand.closing_range_m), biastype='affine',
                                   gainprm=_values([drive.stiffness_n_m]),
                                   biasprm=_values([0.0, -drive.stiffness_n_m, -drive.damping_n_s_m]),
                                   forcelimited='true', forcerange=_values([-drive.max_force_n, drive.max_force_n])))
    return elements


def copy_meshes(setup_dir):
    """Copy the official meshes (and their licence) under `setup_dir`/MESH_DIR."""
    target = Path(setup_dir) / MESH_DIR
    target.mkdir(parents=True, exist_ok=True)
    for link in ('gripper', *TIPS):
        shutil.copyfile(Path(ASSET_DIR) / 'assets' / f'{link}.stl', target / f'{link}.stl')
    shutil.copyfile(Path(ASSET_DIR) / 'LICENSE', target / 'LICENSE')
    return target


def write_hand(setup_dir, side, hand=None):
    """The hand on its own (robot/yam_<side>.xml under `setup_dir`), its wrist at the origin, as DexAgent's grasp
    tools load a hand."""
    hand = yam_hand() if hand is None else hand
    setup_dir = Path(setup_dir)
    copy_meshes(setup_dir)
    root = ET.Element('mujoco', model=f'yam_{side}')
    ET.SubElement(root, 'compiler', angle='radian', meshdir='../')
    ET.SubElement(root, 'asset').extend(assets(hand, side))
    ET.SubElement(root, 'worldbody').append(floating_hand(hand, side))
    ET.SubElement(root, 'actuator').extend(actuators(hand, side))
    ET.indent(root)
    path = setup_dir / 'robot' / f'yam_{side}.xml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ET.tostring(root, encoding='unicode') + '\n')
    return path
