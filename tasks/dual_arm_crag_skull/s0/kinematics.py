"""Offline CRX-10iA/L kinematics for S0: FK, TCP, geometric Jacobian and collision capsules.

Joint frames come from configs/crx10ial_kinematics.json (Isaac Sim 4.5 probe of the repo USD,
sha256-pinned; regenerate with s0/export_kinematics.py). The TCP is always the flange pose times an
explicit `flange_T_tcp`. `yam_model` carries the YAM linear_4310 gripper from its official geometry on a
provisional adapter plate; `robotiq_proxy_model` the earlier Robotiq 2F-85 stand-in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from core.robots.kinematics import Arm, T, rot_axis
from core.robots.linear_4310 import load_linear_4310
from core.workcell.config import load_workcell_preset

from ..contracts.actions import ARMS
from ..perception import rigid_transform
from .collision import fit_capsule

KINEMATICS_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'crx10ial_kinematics.json'
CAPSULE_LINKS = ('base_link', 'link_1', 'link_2', 'link_3', 'link_4', 'link_5', 'link_6')
# The 12 edges of a box whose 8 corners are listed x-major, as ArmModel._box_corners lists them.
BOX_EDGES = tuple((i, j) for i in range(8) for j in range(i + 1, 8) if i ^ j in (1, 2, 4))
AXES = {'X': np.array([1.0, 0.0, 0.0]), 'Y': np.array([0.0, 1.0, 0.0]), 'Z': np.array([0.0, 0.0, 1.0])}

# Robotiq 2F-85 stand-in for the unknown YAM gripper. TCP offset copied from
# tasks/stack_sawtooth_shulker_small/teleop.py (TCP_OFFSET: grasp centre on the base's +X approach
# axis); the gripper base frame equals the flange frame because core/robots/builder.py mounts it
# with identity local poses. 0.085 m is the 2F-85 nominal stroke. Proxy values, not YAM values.
ROBOTIQ_PROXY_FLANGE_T_TCP = T([0.135, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
ROBOTIQ_PROXY_MAX_WIDTH_M = 0.085

# I2RT linear_4310, the YAM gripper (core/robots/linear_4310.py), on the CRX flange through an adapter plate that is
# not designed yet. PROVISIONAL, not measured: the plate thickness, and the clocking (mount x, y, z -> flange y, z, x)
# that keeps the proxy's TCP convention, +x approach and fingers closing along y. The gripper body reaches 10.5 mm
# behind its mount plane, so the plate must be thicker than that or recessed.
YAM_PROVISIONAL_ADAPTER_THICKNESS_M = 0.015
YAM_MOUNT_ROTATION = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def load_kinematics(path=KINEMATICS_PATH):
    data = json.loads(Path(path).read_text())
    if data.get('schema_version') != 1:
        raise ValueError('unsupported kinematics schema')
    return data


@dataclass(frozen=True)
class Capsules:
    """Capsule set in the world frame: p0/p1 (K, 3), radius (K,), names (K,)."""

    p0: np.ndarray
    p1: np.ndarray
    radius: np.ndarray
    names: tuple[str, ...]

    def arrays(self):
        return self.p0, self.p1, self.radius


def _capsule_tuple(record):
    return np.asarray(record['p0'], dtype=float), np.asarray(record['p1'], dtype=float), float(record['radius'])


def _tool_box_tuple(record):
    low, high = np.asarray(record['min'], dtype=float), np.asarray(record['max'], dtype=float)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([*low, *high]).all() or (low >= high).any():
        raise ValueError('tool box needs finite min < max in the flange frame')
    return low, high


def _tool_parts_tuple(record, box):
    """(body, fingers) flange-frame boxes from a tool box record's `parts`, or None if it has none."""
    parts = record.get('parts')
    if parts is None:
        return None
    if not isinstance(parts, Mapping) or set(parts) != {'body', 'fingers'}:
        raise ValueError('tool box parts must be exactly the gripper body and fingers')
    body, fingers = _tool_box_tuple(parts['body']), _tool_box_tuple(parts['fingers'])
    low, high = box
    if (np.abs(np.minimum(body[0], fingers[0]) - low).max() > 1e-9
            or np.abs(np.maximum(body[1], fingers[1]) - high).max() > 1e-9):
        raise ValueError('the gripper body and fingers must together span exactly the tool box')
    return body, fingers


class ArmModel:
    """One arm: world_T_base, the probed joint chain, flange_T_tcp, per-link capsules and tool box."""

    def __init__(self, name, data, world_T_base, flange_T_tcp, tool_capsule, tool_box=None):
        self.name = name
        self._arm = Arm({'arm': data['arm']})
        self._arm.base = rigid_transform(world_T_base).copy()
        self.world_T_base = self._arm.base.copy()
        self.flange_T_tcp = rigid_transform(flange_T_tcp).copy()
        self.lower, self.upper = np.array(self._arm.lo, dtype=float), np.array(self._arm.hi, dtype=float)
        self.max_velocity = np.asarray(data['max_joint_velocity_rad_s'], dtype=float)
        self.dof = len(self._arm.dof)
        self._link_capsules = {link: _capsule_tuple(data['link_capsules'][link]) for link in CAPSULE_LINKS}
        self._tool_capsule = None if tool_capsule is None else _capsule_tuple(tool_capsule)
        self._tool_box = None if tool_box is None else _tool_box_tuple(tool_box)
        self._tool_parts = None if tool_box is None else _tool_parts_tuple(tool_box, self._tool_box)

    @property
    def has_tool_box(self):
        return self._tool_box is not None

    @property
    def has_tool_parts(self):
        """True when the tool box is split into the gripper body and fingers, as S0's contact phase needs."""
        return self._tool_parts is not None

    def _box_corners(self, box, q):
        low, high = box
        corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
        flange = self.frames(q)[0]['flange']
        return corners @ flange[:3, :3].T + flange[:3, 3]

    def _part(self, index):
        if self._tool_parts is None:
            raise ValueError(f'{self.name}: no finger geometry (tool box parts); the contact phase needs it')
        return self._tool_parts[index]

    def tool_box_corners(self, q):
        """World positions (8, 3) of the tool's flange-frame bounding box corners at joints q."""
        if self._tool_box is None:
            raise ValueError(f'{self.name}: no tool box; tool-table checks need the tool geometry')
        return self._box_corners(self._tool_box, q)

    def body_box_corners(self, q):
        """World corners (8, 3) of the box around the gripper body, the part that does not move with the fingers."""
        return self._box_corners(self._part(0), q)

    def finger_box_corners(self, q):
        """World corners (8, 3) of the box around the fingers over their whole stroke."""
        return self._box_corners(self._part(1), q)

    def _check(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.dof,) or not np.isfinite(q).all():
            raise ValueError(f'{self.name}: expected {self.dof} finite joint positions')
        return q

    def frames(self, q):
        """World poses of link_1..link_6 (child frames), flange and TCP, plus joint axes/origins."""
        q = self._check(q)
        transform = self._arm.base.copy()
        links, axes, origins = {}, [], []
        for index, (axis, before, after) in enumerate(self._arm.chain):
            joint = transform @ before
            axes.append(joint[:3, :3] @ AXES[axis])
            origins.append(joint[:3, 3].copy())
            transform = joint @ rot_axis(axis, q[index]) @ after
            links[f'link_{index + 1}'] = transform
        flange = transform @ self._arm.T_flange
        links['base_link'] = self._arm.base.copy()
        links['flange'] = flange
        links['tcp'] = flange @ self.flange_T_tcp
        return links, np.array(axes), np.array(origins)

    def tcp(self, q):
        return self.frames(q)[0]['tcp']

    def jacobian(self, q):
        """Geometric Jacobian of the TCP in the world frame: rows (v, omega), columns joints."""
        links, axes, origins = self.frames(q)
        point = links['tcp'][:3, 3]
        linear = np.cross(axes, point - origins)
        return np.vstack([linear.T, axes.T])

    def capsules(self, q):
        links = self.frames(q)[0]
        names, p0, p1, radius = [], [], [], []
        items = [(link, links[link], self._link_capsules[link]) for link in CAPSULE_LINKS]
        if self._tool_capsule is not None:
            items.append(('tool', links['flange'], self._tool_capsule))
        for name, frame, (a, b, r) in items:
            names.append(f'{self.name}:{name}')
            p0.append(frame[:3, :3] @ a + frame[:3, 3])
            p1.append(frame[:3, :3] @ b + frame[:3, 3])
            radius.append(r)
        return Capsules(np.array(p0), np.array(p1), np.array(radius), tuple(names))


class DualArmModel:
    """Both arms placed by the core `dual_arm` preset; TCP and tool geometry passed explicitly."""

    def __init__(self, flange_T_tcp: Mapping[str, np.ndarray], tool_capsule: Mapping[str, dict | None],
                 data=None, workcell=None, *, tool_box: Mapping[str, dict | None]):
        data = load_kinematics() if data is None else data
        workcell = load_workcell_preset('dual_arm') if workcell is None else workcell
        if set(flange_T_tcp) != set(ARMS) or set(tool_capsule) != set(ARMS) or set(tool_box) != set(ARMS):
            raise ValueError('flange_T_tcp, tool_capsule and tool_box must be given for both arms (None = unknown)')
        self.data = data
        self.arms = {}
        for arm in ARMS:
            robot = workcell.robot(arm)
            if tuple(robot.arm_dof_names) != tuple(data['arm']['dof_names']):
                raise ValueError(f'{arm}: workcell DOF names differ from the kinematic model')
            base = T(robot.base_position, robot.base_orientation_wxyz)
            self.arms[arm] = ArmModel(arm, data, base, flange_T_tcp[arm], tool_capsule[arm], tool_box[arm])


def robotiq_proxy_model(data=None, workcell=None):
    """Dual-arm model with the labelled Robotiq 2F-85 proxy TCP and capsule on both arms."""
    data = load_kinematics() if data is None else data
    return DualArmModel({arm: ROBOTIQ_PROXY_FLANGE_T_TCP for arm in ARMS},
                        {arm: data['robotiq_proxy_capsule'] for arm in ARMS}, data=data, workcell=workcell,
                        tool_box={arm: data['robotiq_proxy_tool_box'] for arm in ARMS})


@lru_cache(maxsize=1)
def _yam():
    return load_linear_4310()


def yam_flange_T_mount(thickness_m=YAM_PROVISIONAL_ADAPTER_THICKNESS_M, gripper=None):
    """The linear_4310 mount frame on the CRX flange: the adapter plate's thickness along the flange +x, and the
    clocking that puts the gripper's approach axis on the flange +x and its finger slide axis on the flange +y."""
    gripper = _yam() if gripper is None else gripper
    back = float(gripper.body_points[:, 2].min())
    if not thickness_m + back > 0:
        raise ValueError(f'the adapter plate must be thicker than the {-back * 1000:.1f} mm the gripper body reaches '
                         'behind its mount plane')
    flange_T_mount = np.eye(4)
    flange_T_mount[:3, :3] = YAM_MOUNT_ROTATION
    flange_T_mount[0, 3] = thickness_m
    return flange_T_mount


def yam_tool_geometry(thickness_m=YAM_PROVISIONAL_ADAPTER_THICKNESS_M, gripper=None):
    """(flange_T_tcp, tool capsule, tool box) of linear_4310 on the adapter plate, in the flange frame.

    The TCP is the grasp site with the flange's orientation (the proxy's convention). Box parts: `body` is the gripper
    body plus the plate (a block over the body's cross-section), `fingers` both tips over the whole stroke; the tips
    move linearly, so the two ends of the stroke bound them."""
    gripper = _yam() if gripper is None else gripper
    flange_T_mount = yam_flange_T_mount(thickness_m, gripper)

    def to_flange(points):
        return points @ flange_T_mount[:3, :3].T + flange_T_mount[:3, 3]

    body = to_flange(gripper.body_points)
    low, high = body.min(axis=0), body.max(axis=0)
    plate = np.array([[x, y, z] for x in (0.0, thickness_m) for y in (low[1], high[1]) for z in (low[2], high[2])])
    body = np.concatenate([body, plate])
    fingers = to_flange(np.concatenate([points for q in (0.0, gripper.travel_m)
                                        for points in gripper.tips_at(q).values()]))
    parts = {'body': {'min': body.min(axis=0).tolist(), 'max': body.max(axis=0).tolist(),
                      'links': ['adapter_plate', 'gripper']},
             'fingers': {'min': fingers.min(axis=0).tolist(), 'max': fingers.max(axis=0).tolist(),
                         'links': ['tip_left', 'tip_right']}}
    box = {'min': np.minimum(body.min(axis=0), fingers.min(axis=0)).tolist(),
           'max': np.maximum(body.max(axis=0), fingers.max(axis=0)).tolist(), 'parts': parts, 'frame': 'CRX flange',
           'source': 'I2RT linear_4310 meshes (core/assets/robots/i2rt_linear_4310) over the whole stroke, on a '
                     f'PROVISIONAL {thickness_m * 1000:g} mm adapter plate'}
    p0, p1, radius = fit_capsule(np.concatenate([body, fingers]))
    capsule = {'p0': p0.tolist(), 'p1': p1.tolist(), 'radius': radius, 'vertices': int(len(body) + len(fingers)),
               'source': box['source']}
    return T((flange_T_mount @ gripper.mount_T_tcp)[:3, 3], [1.0, 0.0, 0.0, 0.0]), capsule, box


def yam_width_range_m(gripper=None):
    """(0, widest gap between the pad faces) from the official meshes; the real gripper is not measured yet."""
    gripper = _yam() if gripper is None else gripper
    return 0.0, gripper.max_width_m


def yam_model(data=None, workcell=None, thickness_m=YAM_PROVISIONAL_ADAPTER_THICKNESS_M):
    """Dual-arm model with linear_4310 on the provisional adapter plate on both arms."""
    flange_T_tcp, capsule, box = yam_tool_geometry(thickness_m)
    return DualArmModel({arm: flange_T_tcp for arm in ARMS}, {arm: capsule for arm in ARMS}, data=data,
                        workcell=workcell, tool_box={arm: box for arm in ARMS})
