"""Scripted robot assembly of a task's assembly plan through S0, on any AssemblyBackend: our own reference that the plan
can be carried out with this robot and gripper, not a baseline (it reads the fragments' observed poses, which every
method is given, and their meshes). sim/check_assembly.py runs it on Isaac; the tests run it on KinematicBackend.

Planning a step (`plan_step`): an arm (the one on the fragment's side first) and a chord grasp (sim/pick.py) with the
fingertips at any height up the fragment, the highest the pads still hold first (the least of the fingers goes down
into a pocket), on a line through the middle of its footprint or, failing that, on one moved across the closing axis
by LATERAL_OFFSETS_M (a fragment in a pocket may leave room for the fingers only off its middle). The grasp is found
with the fragment as if it stood on the table in its goal orientation, so that the gripper comes down onto the goal
upright (goal_aligned: for a fragment resting tilted, after toppling say, the fingers then come in tilted at the
pick), and failing that where it rests, the fingers coming straight down (top_down); a fragment upright as in its goal
has the two the same. It must be such that
- with the fragment where it rests, the open fingers come down around it at least STAGED_MARGIN_M from the other
  fragments still on the table;
- with the fragment carried to its goal (the grasp held: the TCP keeps its pose relative to the fragment), the fingers
  and the gripper body stay FINGER_MARGIN_M clear of the cells of the fragments already in place (the layout's voxel
  grid) while the fragment comes down to where its insertion begins and all along the plan's straight insertion;
- once the fingers open at the goal, only to RELEASE_GAP_M a side beyond the width at which they rise past the
  fragment (a pocket has no room for more), and while they back out and rise, they stay as clear of those cells and
  of the fragment's own;
- the arm reaches the poses on the way (S0's IK), and where the gripper is lowest (picking, where the insertion
  begins, at the goal, backed out) S0's table rules hold with S0_MARGIN_M to spare: the normal ones, or those of the
  contact window the step declares there (s0_table_use; a fragment turned on its way tilts the gripper, and the
  box S0 puts round the fingers' whole stroke then dips a corner). That box holds the fingers at every opening, so
  this keeps them off the table too, coming in along their own axis to the pick included.
Where the insertion begins: backing out of the goal against the insertion direction, BEGIN_CLEARANCE_M beyond the
furthest the placed cells reach that way, so the fragment comes down to it outside them (the plan guarantees the
straight insertion itself).

Running a step (`run_step`), pick.py's segments, each checked by S0 (check_joint_path, with the contact window it
needs) from the measured state before it is stepped: approach above the fragment, descend, close, squeeze, lift, rise
until the fragment clears everything by SAFE_CLEARANCE_M, glide over the point where the insertion begins (turning to
the goal orientation), come down to it, go in at INSERT_SPEED_M_S, open to the release width, back out BACK_M, rise
and go back to where the arm started (out of the other arm's way); the other arm holds still. Near the table
(fingertips within LOW_M of it, or wherever the plan found S0 needs it) the segment gets the contact window (grasp:
centred on the fragment's observed pose; place: on its goal) and S0's contact speed. Then both arms hold for
STEP_SETTLE_S and the drift of every placed fragment from its goal is read.
`run_assembly` does every step, then the ending every method gets (both grippers open, held the task's settle time
through S0) and scores the result.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from functools import lru_cache
from types import SimpleNamespace
from typing import Callable, Mapping

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..baselines.backend import RobotSnapshot
from ..contracts.actions import ARMS, ContactWindow
from ..s0.adapter import Rejected, check_joint_path
from ..s0.ik import IKOptions, solve_ik
from . import pick

FINGER_MARGIN_M = 0.002
STAGED_MARGIN_M = 0.005
BEGIN_CLEARANCE_M = 0.02
SAFE_CLEARANCE_M = 0.03
BACK_M = 0.04
RELEASE_GAP_M = 0.003  # a side, beyond the clear width: more than FINGER_MARGIN_M off the fragment's own cells
HEIGHT_STEP_M = 0.006
LATERAL_OFFSETS_M = (0.0, -0.004, 0.004, -0.008, 0.008, -0.012, 0.012, -0.016, 0.016)  # the middle line first
MAX_PREFLIGHTS = 16  # candidates a step tries through its Runner before it gives up
PATH_STEP_M = 0.003
THIN_CELL_M = 0.001  # the tips' points, one per 1 mm cell: none more than 1.7 mm from the mesh, inside FINGER_MARGIN_M
TOP_PAD_M = 0.012  # the pads are 15 mm tall: grasping higher than this below the top grips nothing
LOW_M = 0.03
S0_MARGIN_M = 0.003
TILT_TOLERANCE_DEG = 1.0  # a fragment resting tilted less than this from its goal orientation is upright
SWEEP_STEP_M = 0.004  # a swept move is checked wherever some point of the gripper has moved this far
GRIPPER_REACH_M = 0.2  # no point of the gripper is further than this from its TCP
TRANSIT_SPEED_M_S = 0.12
INSERT_SPEED_M_S = 0.02
STEP_SETTLE_S = 0.5


def _world(points, pose):
    pose = np.asarray(pose, dtype=float)
    return np.asarray(points, dtype=float) @ pose[:3, :3].T + pose[:3, 3]


def fragment_mesh(task, key):
    with np.load(task.fragment(key).mesh) as data:
        return np.asarray(data['v'], dtype=float), np.asarray(data['f'], dtype=int)


class Occupancy:
    """The cells of the layout's voxel grid that the fragments `keys` own, in the world (the assembly frame is the
    world shifted, as task_from_layout puts it)."""

    def __init__(self, task, layout, keys):
        if not np.allclose(np.asarray(task.world_T_assembly)[:3, :3], np.eye(3)):
            raise ValueError('the assembly frame must be the world shifted, unrotated')
        normal = layout['normalization']
        self.pitch = float(layout['processing']['pitch'])
        self.origin = np.asarray(task.world_T_assembly)[:3, 3] + np.asarray(normal['grid_origin'], dtype=float)
        self.shape = np.asarray(normal['grid_shape'], dtype=int)
        keys = set(keys)
        self.cells = np.array([cell for piece in layout['pieces'] if piece['name'] in keys for cell in piece['cells']],
                              dtype=int).reshape(-1, 3)
        self.codes = np.unique(self._code(self.cells))
        if len(self.cells):
            self.low = self.origin + self.cells.min(axis=0) * self.pitch
            self.high = self.origin + (self.cells.max(axis=0) + 1) * self.pitch

    def _code(self, index):
        return (index[:, 0] * self.shape[1] + index[:, 1]) * self.shape[2] + index[:, 2]

    def hits(self, points, margin_m):
        """Whether any point comes within `margin_m` (each axis; under half a cell, so that a cell the box round a point
        meets holds one of its corners) of an owned cell."""
        if not 0 <= margin_m < self.pitch / 2:
            raise ValueError(f'the margin {margin_m} m must be under half a cell ({self.pitch / 2} m)')
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not len(self.codes) or not len(points):
            return False
        points = points[((points >= self.low - margin_m) & (points <= self.high + margin_m)).all(axis=1)]
        for offset in itertools.product((-margin_m, margin_m), repeat=3):
            index = np.floor((points + offset - self.origin) / self.pitch).astype(int)
            inside = ((index >= 0) & (index < self.shape)).all(axis=1)
            if np.isin(self._code(index[inside]), self.codes).any():
                return True
        return False

    def reach(self, direction):
        """The furthest any owned cell reaches along `direction`; -inf if none."""
        if not len(self.cells):
            return -np.inf
        corners = np.array(list(itertools.product((0, 1), repeat=3)))
        points = self.origin + (self.cells[:, None, :] + corners[None]) * self.pitch
        return float((points.reshape(-1, 3) @ np.asarray(direction, dtype=float)).max())

    def top(self):
        return self.reach((0.0, 0.0, 1.0))


def _box_surface(low, high, spacing_m):
    axes = [np.linspace(a, b, max(2, int(np.ceil((b - a) / spacing_m)) + 1)) for a, b in zip(low, high)]
    faces = []
    for axis in range(3):
        others = [axes[k] for k in range(3) if k != axis]
        grid = np.stack(np.meshgrid(*others, indexing='ij'), -1).reshape(-1, 2)
        for value in (low[axis], high[axis]):
            faces.append(np.insert(grid, axis, value, axis=1))
    return np.concatenate(faces)


def _thin(points, cell_m):
    """One of `points` per `cell_m` cube they fall in."""
    _, first = np.unique(np.floor(points / cell_m).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(first)]


@lru_cache(maxsize=1)
def _gripper_in_mount():
    from core.robots.linear_4310 import load_linear_4310

    from .yam_usd import tip_points_in_link
    gripper = load_linear_4310()
    tips = {}
    for tip in ('tip_left', 'tip_right'):
        frame = gripper.link_frames[tip]
        tips[tip] = _thin(tip_points_in_link(gripper, tip, 0.002) @ frame[:3, :3].T + frame[:3, 3], THIN_CELL_M)
    body = _box_surface(gripper.body_points.min(axis=0), gripper.body_points.max(axis=0), 0.004)
    return gripper, tips, body


@lru_cache(maxsize=512)
def _gripper_points_cached(arm_model, width_m):
    points = gripper_points(arm_model, width_m)
    points.setflags(write=False)  # shared by every caller
    return points


def gripper_points(arm_model, width_m):
    """Points on both tips at the opening `width_m` and on the gripper body's bounding box, in the TCP frame."""
    from ..s0.kinematics import yam_flange_T_mount
    gripper, tips, body = _gripper_in_mount()
    q = gripper.joint_for_width(min(max(float(width_m), 0.0), gripper.max_width_m))
    mount = np.concatenate([tips[tip] + q * np.asarray(gripper.slide_axes[tip]) for tip in tips] + [body])
    tcp_T_mount = np.linalg.inv(np.asarray(arm_model.flange_T_tcp, dtype=float)) @ yam_flange_T_mount()
    return mount @ tcp_T_mount[:3, :3].T + tcp_T_mount[:3, 3]


def _translated(pose, offset):
    out = np.array(pose, dtype=float, copy=True)
    out[:3, 3] += offset
    return out


def _sweep(points, rotation, positions):
    """`points` (TCP frame) at every position of the TCP with `rotation`, as one (len(positions) * N, 3) array."""
    turned = np.asarray(points) @ np.asarray(rotation).T
    return (np.asarray(positions)[:, None, :] + turned[None]).reshape(-1, 3)


def _line(start, end, step_m=PATH_STEP_M):
    count = max(1, int(np.ceil(np.linalg.norm(np.asarray(end) - np.asarray(start)) / step_m)))
    return np.asarray(start) + np.outer(np.linspace(0.0, 1.0, count + 1), np.asarray(end) - np.asarray(start))


def s0_table_use(arm_model, limits, q, centre_xy):
    """How the arm at joints q stands to S0's table rules (s0/adapter.py), each with S0_MARGIN_M to spare: (whether
    it needs a contact window, already within the margin of the normal floor; None, or which of that window's rules,
    centred at `centre_xy`, fails: no_contact_phase, contact_clearance, body_clearance or contact_region)."""
    from ..s0.adapter import _below
    table = limits.table_top_z_m
    tcp = arm_model.tcp(q)
    height = tcp[2, 3] - table
    if (arm_model.tool_box_corners(q)[:, 2].min() - table >= limits.table_clearance_m + S0_MARGIN_M
            and height >= limits.tcp_min_height_m + S0_MARGIN_M):
        return False, None
    if not limits.contact_enabled:
        return True, 'no_contact_phase'
    fingers = arm_model.finger_box_corners(q)
    if min(fingers[:, 2].min() - table, height) < limits.contact_table_clearance_m + S0_MARGIN_M:
        return True, 'contact_clearance'
    if arm_model.body_box_corners(q)[:, 2].min() - table < limits.table_clearance_m + S0_MARGIN_M:
        return True, 'body_clearance'
    points = np.vstack([tcp[:3, 3], _below(fingers, table + limits.table_clearance_m + S0_MARGIN_M)])
    reach = np.hypot(points[:, 0] - centre_xy[0], points[:, 1] - centre_xy[1]).max()
    if reach > limits.contact_region_radius_m - S0_MARGIN_M:
        return True, 'contact_region'
    return True, None


@dataclass(frozen=True)
class Placement:
    """A planned step: the arm, the grasp (found as `family` says, with the TCP fingertips at tip_z in that frame), the
    TCP poses on the way (pre, pick, up, over, begin, goal, back, leave), the joints S0's IK found for pre, the width
    the fingers open to at the goal and which of pick, begin, goal and back need their contact window
    (s0_table_use)."""

    fragment_id: str
    arm: str
    grasp: pick.Grasp
    tip_z: float
    poses: Mapping[str, np.ndarray]
    q_pre: np.ndarray
    insertion: tuple
    begin_offset_m: float
    release_width_m: float
    family: str = 'top_down'
    windows: Mapping[str, bool] = field(default_factory=dict)
    rejected: Mapping[str, int] = field(default_factory=dict)

    def to_record(self):
        return {'fragment_id': self.fragment_id, 'arm': self.arm, 'family': self.family, 'tip_z': self.tip_z,
                'grasp': {'yaw_deg': float(np.degrees(self.grasp.yaw_rad)), 'chord_m': self.grasp.chord_m,
                          'open_width_m': self.grasp.open_width_m, 'close_width_m': self.grasp.close_width_m,
                          'clear_width_m': self.grasp.clear_width_m},
                'insertion': list(self.insertion), 'begin_offset_m': self.begin_offset_m,
                'release_width_m': self.release_width_m, 'windows': dict(self.windows),
                'rejected': dict(self.rejected), 'poses': {k: v.tolist() for k, v in self.poses.items()}}


def _surface_at(task, key, pose):
    vertices, faces = fragment_mesh(task, key)
    return pick.surface_points(_world(vertices, pose), faces)


def _side_arms(start):
    own = 'left' if start[0, 3] < 0 else 'right'
    return (own, next(arm for arm in ARMS if arm != own))


def _grasp_frames(vertices, start, goal, table):
    """[(family, frame)] in the order tried: the fragment's pose at which grasp candidates are worked out as if it
    stood there on the table, a grasp found there carried to where it rests by start @ inv(frame). goal_aligned is its
    goal orientation, turned about the vertical as near as can be to how it rests, at its resting place; top_down is
    where it rests; only top_down if it rests upright."""
    up, up_goal = start[:3, :3].T @ (0.0, 0.0, 1.0), np.asarray(goal)[:3, :3].T @ (0.0, 0.0, 1.0)
    if np.degrees(np.arccos(np.clip(up @ up_goal, -1.0, 1.0))) <= TILT_TOLERANCE_DEG:
        return [('top_down', start)]
    # turned about the vertical to be nearest to how it rests (any turn keeps the gripper upright at the goal)
    rest = Rotation.from_matrix(np.array(start[:3, :3]))
    yaws = Rotation.from_euler('z', np.radians(np.arange(0.0, 360.0, 1.0)))
    turns = yaws * Rotation.from_matrix(np.array(goal)[:3, :3])
    rotation = turns[int(np.argmin((turns * rest.inv()).magnitude()))].as_matrix()
    upright, resting = vertices @ rotation.T, _world(vertices, start)
    frame = np.eye(4)
    frame[:3, :3] = rotation
    frame[:2, 3] = ((resting[:, :2].min(axis=0) + resting[:, :2].max(axis=0))
                    - (upright[:, :2].min(axis=0) + upright[:, :2].max(axis=0))) / 2
    frame[2, 3] = table - upright[:, 2].min()
    return [('goal_aligned', frame), ('top_down', start)]


def plan_step(task, layout, model, limits, *, key, insertion, poses, placed, seeds, runner=None):
    """The Placement of `key` (module docstring), the fragments `placed` at their goals and the rest where `poses`
    (world_T_fragment) has them; `seeds` {arm: joints} start the IK. Given the Runner that will carry it out, each
    candidate is first tried through it (_preflight: S0 on every segment from where the robot is, and the joint moves
    and a tilted lift swept against the fragments) and passed over if it would not go, up to MAX_PREFLIGHTS of them.
    ValueError, counting why candidates failed, if none works."""
    start, goal = np.asarray(poses[key], dtype=float), task.fragment(key).goal_world_T_fragment
    vertices, faces = fragment_mesh(task, key)
    points = pick.surface_points(_world(vertices, start), faces)
    out = -np.asarray(insertion, dtype=float)
    occupancy = Occupancy(task, layout, placed)
    released = Occupancy(task, layout, [*placed, key])  # once the fingers open, the fragment is one of them
    at_goal = _world(vertices, goal)
    offset = max(occupancy.reach(out) - float((at_goal @ out).min()), 0.0) + BEGIN_CLEARANCE_M
    staged = [_surface_at(task, other, poses[other]) for other in task.fragment_ids
              if other not in placed and other != key]
    staged_tree = cKDTree(np.concatenate(staged)) if staged else None
    neighbours = [(cKDTree(surface), surface.min(axis=0), surface.max(axis=0)) for surface in staged]
    itself = (cKDTree(points), points.min(axis=0), points.max(axis=0))
    tops = [occupancy.top()] + [float(_world(fragment_mesh(task, other)[0], poses[other])[:, 2].max())
                                for other in task.fragment_ids if other not in placed and other != key]
    top = max(tops)
    table = limits.table_top_z_m
    drops = (float(points[:, 2].min()), float(at_goal[:, 2].min()))  # the fragment's lowest, resting and at its goal
    rejected, tried = {}, 0

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    for family, frame in _grasp_frames(vertices, start, goal, table):
        seen = points if family == 'top_down' else pick.surface_points(_world(vertices, frame), faces)
        highest = float(seen[:, 2].max())
        carry = start @ np.linalg.inv(frame)
        for arm in _side_arms(start):
            arm_model = model.arms[arm]
            for tip_z, offset in itertools.product(
                    np.arange(highest - TOP_PAD_M, table + pick.FINGERTIP_HEIGHT_M - 1e-9, -HEIGHT_STEP_M),
                    LATERAL_OFFSETS_M):
                candidates = pick.grasp_candidates(key, seen, tip_z - pick.FINGERTIP_HEIGHT_M,
                                                   limits.gripper_width_range_m[arm][1], offsets_m=(offset,))
                for grasp in sorted(candidates, key=lambda g: g.chord_m):
                    placement = _try_grasp(task, model, limits, arm, arm_model, key, grasp, float(tip_z), carry,
                                           start, goal, out, offset, occupancy, released, staged_tree, top, drops,
                                           seeds, reject)
                    if placement is None:
                        continue
                    candidate = Placement(key, arm, grasp, float(tip_z), placement['poses'], placement['q_pre'],
                                          tuple(float(v) for v in insertion), offset, placement['release_width'],
                                          family=family, windows=placement['windows'], rejected=dict(rejected))
                    if runner is None:
                        return candidate
                    sweeps = _sweeps(candidate, points, occupancy, released, neighbours, itself)
                    problem = _preflight(runner, task, candidate, poses, sweeps)
                    if problem is None:
                        return candidate
                    reject(f'preflight_{problem}')
                    tried += 1
                    if tried >= MAX_PREFLIGHTS:
                        raise ValueError(f'{key}: no grasp went in {tried} preflights ({rejected})')
    raise ValueError(f'{key}: no grasp works ({rejected or "no candidates"})')


def _sweeps(placement, points, occupancy, released, neighbours, itself):
    """{segment name: check} for _preflight, the moves the plan's checks leave out: the open gripper's joint move to
    above the fragment (against the placed cells, the fragments on the table and this one) and its move back to where
    the arm waits (against the placed cells, this one among them now, and the fragments still on the table); and, for
    a fragment grasped tilted, the fragment itself lifted along the tilted approach axis (against the others on the
    table: it moves sideways as it rises)."""
    def gripper(cells, trees):
        def check(arm_model, segment):
            points_tcp = _gripper_points_cached(arm_model, round(float(segment.widths[0]), 4))
            return _comes_near(lambda pose: _world(points_tcp, pose), _swept_poses(arm_model, segment.joints), cells,
                               trees)
        return check

    sweeps = {'approach': gripper(occupancy, [*neighbours, itself]), 'park': gripper(released, neighbours)}
    if placement.family == 'goal_aligned':
        held = _world(points, np.linalg.inv(placement.poses['pick']))  # the fragment in the TCP frame, as grasped

        def lift(arm_model, segment):
            return _comes_near(lambda pose: _world(held, pose), _swept_poses(arm_model, segment.joints), None,
                               neighbours)
        sweeps['lift'] = lift
    return sweeps


def _clearing_height(top, pick_pose, tcp_goal, drops):
    """The TCP height at which the fragment clears `top` by SAFE_CLEARANCE_M held either way it is on the way: as at
    the pick, and as at its goal (it turns from one to the other while it glides); `drops` are its lowest points
    resting and at its goal."""
    return top + SAFE_CLEARANCE_M + max(pick_pose[2, 3] - drops[0], tcp_goal[2, 3] - drops[1])


def _try_grasp(task, model, limits, arm, arm_model, key, grasp, tip_z, carry, start, goal, out, offset, occupancy,
               released, staged_tree, top, drops, seeds, reject):
    # The geometry first (the fingers are symmetric, so it does not matter which way round IK turns the TCP), then
    # the neighbours on the table, then IK.
    pick_pose = carry @ pick.down_pose(grasp.center_xy, tip_z, grasp.yaw_rad)
    pre = _translated(pick_pose, -pick.APPROACH_HEIGHT_M * pick_pose[:3, 0])  # back along the approach axis
    opened = _gripper_points_cached(arm_model, round(grasp.open_width_m, 4))
    closed = _gripper_points_cached(arm_model, round(grasp.chord_m, 4))
    release_width = round(min(grasp.open_width_m, grasp.clear_width_m + 2 * RELEASE_GAP_M), 4)
    loose = _gripper_points_cached(arm_model, release_width)
    tcp_goal = goal @ np.linalg.inv(start) @ pick_pose
    tcp_begin = _translated(tcp_goal, offset * out)
    rise = _clearing_height(top, pick_pose, tcp_goal, drops)
    safe_z = max(rise, pre[2, 3], tcp_begin[2, 3])
    over = _translated(tcp_begin, (0, 0, safe_z - tcp_begin[2, 3]))
    back = _translated(tcp_goal, BACK_M * out)
    leave = _translated(back, (0, 0, max(safe_z - back[2, 3], 0.0)))
    checks = (('release', loose, released, tcp_goal[:3, 3][None]),  # the cheapest first
              ('insertion', closed, occupancy, _line(tcp_begin[:3, 3], tcp_goal[:3, 3])),
              ('back_out', loose, released,
               np.vstack([_line(tcp_goal[:3, 3], back[:3, 3]), _line(back[:3, 3], leave[:3, 3])])),
              ('descent_to_begin', closed, occupancy, _line(over[:3, 3], tcp_begin[:3, 3])))
    for reason, gripper, cells, path in checks:
        if cells.hits(_sweep(gripper, tcp_goal[:3, :3], path), FINGER_MARGIN_M):
            reject(reason)
            return None
    if staged_tree is not None:
        down = _sweep(opened, pick_pose[:3, :3], _line(pick_pose[:3, 3], pre[:3, 3], 0.005))
        if staged_tree.query(down, distance_upper_bound=STAGED_MARGIN_M)[0].min() < STAGED_MARGIN_M:
            reject('staged_neighbour')
            return None
    try:
        q_pre, pre = pick.solve_reachable(arm_model, pre, seeds[arm])
    except ValueError:
        reject('ik_pick')
        return None
    if not np.allclose(pre[:3, :3], pick_pose[:3, :3]):  # IK took the TCP turned half way round: the same grasp
        pick_pose = _translated(pre, pick.APPROACH_HEIGHT_M * pre[:3, 0])
        tcp_goal = goal @ np.linalg.inv(start) @ pick_pose
        tcp_begin, back = _translated(tcp_goal, offset * out), _translated(tcp_goal, BACK_M * out)
        over = _translated(tcp_begin, (0, 0, safe_z - tcp_begin[2, 3]))
        leave = _translated(back, (0, 0, max(safe_z - back[2, 3], 0.0)))
    options = IKOptions(max_seed_jump_rad=3.5)
    windows = {}
    for name, pose, centre in (('pick', pick_pose, start[:2, 3]), ('goal', tcp_goal, goal[:2, 3]),
                               ('begin', tcp_begin, goal[:2, 3]), ('back', back, goal[:2, 3]), ('over', over, None),
                               ('leave', leave, None)):  # the low ones first, each with S0's table rules
        solution = solve_ik(arm_model, pose, q_pre, options)
        if not solution.ok:
            reject(f'ik_{name}')
            return None
        if centre is not None:
            windows[name], problem = s0_table_use(arm_model, limits, solution.q, centre)
            if problem is not None:
                reject(f's0_{name}_{problem}')
                return None
    up = _translated(pre, (0, 0, max(safe_z - pre[2, 3], 0.0)))
    return {'q_pre': q_pre, 'release_width': release_width, 'windows': windows,
            'poses': {'pre': pre, 'pick': pick_pose, 'up': up, 'over': over, 'begin': tcp_begin, 'goal': tcp_goal,
                      'back': back, 'leave': leave}}


@dataclass
class StepRecord:
    fragment_id: str
    placement: Placement | None = None
    segments: list = field(default_factory=list)
    drift_mm_deg: dict = field(default_factory=dict)
    error: str | None = None

    def to_record(self):
        return {'fragment_id': self.fragment_id, 'error': self.error, 'segments': self.segments,
                'drift_mm_deg': self.drift_mm_deg,
                'placement': None if self.placement is None else self.placement.to_record()}


def _drift(task, poses, keys):
    drift = {}
    for key in keys:
        goal, now = np.array(task.fragment(key).goal_world_T_fragment), np.array(poses[key])  # scipy wants writable
        turn = (Rotation.from_matrix(now[:3, :3]) * Rotation.from_matrix(goal[:3, :3]).inv()).magnitude()
        drift[key] = [1000 * float(np.linalg.norm(now[:3, 3] - goal[:3, 3])), float(np.degrees(turn))]
    return drift


class Runner:
    """Steps segments through S0 on `backend` with the other arm held; `on_step(name, state, targets, contact)` sees
    each control period (contact: the segment's ContactWindow, or None)."""

    def __init__(self, backend, model, limits, on_step=None):
        self.backend, self.model, self.limits, self.on_step = backend, model, limits, on_step
        self.servo_hz = float(limits.servo_hz)
        self.state = backend.robot_state()
        self.targets = {arm: np.asarray(self.state.joints[arm], dtype=float) for arm in ARMS}
        self.park = {arm: target.copy() for arm, target in self.targets.items()}  # where each arm waits its turn
        self.widths = {arm: limits.gripper_width_range_m[arm][1] for arm in ARMS}

    def _check(self, arm, segment):
        """S0's check of `segment` from the state now, the other arm held where it is; the sample times. A refusal
        names the segment."""
        other = next(name for name in ARMS if name != arm)
        count = len(segment.joints)
        times = self.state.time_s + np.arange(1, count + 1) / self.servo_hz
        joints = {arm: segment.joints, other: np.repeat(self.targets[other][None], count, axis=0)}
        try:
            check_joint_path(self.model, self.limits, now=self.state.time_s, q_now=self.state.joints,
                             qd_now=self.state.joint_velocities, qd_window_s=self.state.velocity_window_s,
                             times=times, joints=joints, contacts=() if segment.contact is None else (segment.contact,))
        except Rejected as exc:
            raise Rejected(f'{segment.name}: {exc}') from exc
        return times

    @staticmethod
    def _record(arm, segment):
        return {'name': segment.name, 'arm': arm, 'count': len(segment.joints),
                'window': None if segment.contact is None else segment.contact.kind}

    def run(self, arm, segment):
        self._check(arm, segment)
        for k in range(len(segment.joints)):
            self.targets[arm] = np.asarray(segment.joints[k], dtype=float)
            self.widths[arm] = float(segment.widths[k])
            self.state = self.backend.step(dict(self.targets), dict(self.widths))
            if self.on_step is not None:
                self.on_step(segment.name, self.state, dict(self.targets), segment.contact)
        return self._record(arm, segment)

    def hold(self, arm, seconds, name='hold'):
        count = max(1, int(np.ceil(seconds * self.servo_hz - 1e-9)))
        segment = pick.Segment(name, np.repeat(self.targets[arm][None], count, axis=0),
                               np.full(count, self.widths[arm]))
        return self.run(arm, segment)


class Preflight(Runner):
    """A Runner that only tries: each segment is checked by S0 as Runner.run would send it, then the arm is taken to
    stand at rest where the segment ends (every segment ends at rest), and nothing is stepped. It starts where `runner`
    is and sees the fragments where `poses` (world_T_fragment) has them; `segments` keeps what it was given."""

    def __init__(self, runner, poses):
        observed = {key: SimpleNamespace(world_T_fragment=np.asarray(pose, dtype=float)) for key, pose in poses.items()}
        self.backend = SimpleNamespace(observed_fragment_poses=lambda: observed)
        self.model, self.limits, self.on_step, self.servo_hz = runner.model, runner.limits, None, runner.servo_hz
        self.state, self.park = runner.state, runner.park
        self.targets = {arm: np.array(q, dtype=float) for arm, q in runner.targets.items()}
        self.widths = dict(runner.widths)
        self.segments = []

    def run(self, arm, segment):
        times = self._check(arm, segment)
        self.targets[arm] = np.asarray(segment.joints[-1], dtype=float)
        self.widths[arm] = float(segment.widths[-1])
        joints = {name: self.targets[name].copy() for name in ARMS}
        tcps = {name: self.model.arms[name].tcp(q) for name, q in joints.items()}
        self.state = RobotSnapshot(float(times[-1]), joints, {name: np.zeros_like(q) for name, q in joints.items()},
                                   dict(self.widths), tcps, self.state.velocity_window_s)
        self.segments.append((arm, segment))
        return self._record(arm, segment)


def _box_distance(point, low, high):
    return float(np.linalg.norm(np.maximum(np.maximum(np.asarray(low) - point, 0.0), np.asarray(point) - high)))


def _swept_poses(arm_model, joints):
    """The TCP poses along `joints` (N, dof) a sweep is checked at: the first, the last, and wherever some point of
    the gripper may have moved SWEEP_STEP_M since the last one taken."""
    tcps = [arm_model.tcp(q) for q in joints]
    taken, moved = [tcps[0]], 0.0
    for before, after in zip(tcps, tcps[1:]):
        turn = Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).magnitude()
        moved += float(np.linalg.norm(after[:3, 3] - before[:3, 3])) + GRIPPER_REACH_M * turn
        if moved >= SWEEP_STEP_M:
            taken.append(after)
            moved = 0.0
    if taken[-1] is not tcps[-1]:
        taken.append(tcps[-1])
    return taken


def _comes_near(points_at, poses, cells, trees):
    """Whether the points `points_at(pose)` (world), at any of the TCP `poses`, come within FINGER_MARGIN_M of the
    Occupancy `cells` (None: none) or within STAGED_MARGIN_M of the points of one of `trees` [(cKDTree, low, high)];
    poses whose reach (GRIPPER_REACH_M) stays clear of them all are skipped."""
    for pose in poses:
        centre = pose[:3, 3]
        near_cells = (cells is not None and len(cells.codes) > 0
                      and _box_distance(centre, cells.low, cells.high) <= GRIPPER_REACH_M + FINGER_MARGIN_M)
        near = [tree for tree, low, high in trees
                if _box_distance(centre, low, high) <= GRIPPER_REACH_M + STAGED_MARGIN_M]
        if not near_cells and not near:
            continue
        world = points_at(pose)
        if near_cells and cells.hits(world, FINGER_MARGIN_M):
            return True
        if any(tree.query(world, distance_upper_bound=STAGED_MARGIN_M)[0].min() < STAGED_MARGIN_M for tree in near):
            return True
    return False


def _preflight(runner, task, placement, poses, sweeps):
    """None if the step would go: its segments, made as run_step makes them from where the robot is now and checked
    by S0 (Preflight), and the moves the plan's own checks leave out keep off the fragments: `sweeps` {segment name:
    check(arm_model, segment) -> True if it hits}. Else why not, naming the segment."""
    trial = Preflight(runner, poses)
    try:
        run_step(trial, task, placement)
    except Rejected as exc:
        return f's0_{str(exc).split(":", 1)[0]}'
    except ValueError as exc:  # a line IK cannot follow, or a move that cannot be timed
        return f'segment_{str(exc).split(":", 1)[0]}'
    arm_model = runner.model.arms[placement.arm]
    for _, segment in trial.segments:
        check = sweeps.get(segment.name)
        if check is not None and check(arm_model, segment):
            return f'sweep_{segment.name}'
    return None


def run_step(runner, task, placement, records=None):
    """Carry out a Placement (module docstring), appending each segment's record to `records` as it is done (so a step
    that fails part way keeps those it ran); returns them. Raises Rejected or ValueError."""
    arm, grasp, poses = placement.arm, placement.grasp, placement.poses
    limits, model = runner.limits, runner.model
    arm_model, servo_hz = model.arms[arm], runner.servo_hz
    table = limits.table_top_z_m
    observed = runner.backend.observed_fragment_poses()[placement.fragment_id].world_T_fragment
    goal = task.fragment(placement.fragment_id).goal_world_T_fragment
    grasp_window = ContactWindow(arm, 'grasp', tuple(observed[:2, 3]), placement.fragment_id)
    place_window = ContactWindow(arm, 'place', tuple(goal[:2, 3]), placement.fragment_id)
    contact_speed = (pick.CONTACT_SPEED_FRACTION * limits.contact_max_speed_m_s if limits.contact_enabled
                     else pick.UNCONFIGURED_CONTACT_SPEED_M_S)  # S0 then refuses whatever needs a contact itself
    records = [] if records is None else records

    def low(name):
        return poses[name][2, 3] < table + LOW_M or placement.windows.get(name, False)

    def line(name, target, width, window, speed):
        segment = pick.cartesian_segment(name, arm_model, limits, runner.state.joints[arm], target, width, servo_hz,
                                         max_speed_m_s=speed, contact=window)
        records.append(runner.run(arm, segment))

    def width(name, start, end, seconds, window):
        segment = pick.width_segment(name, runner.state.joints[arm], start, end, seconds, servo_hz, window)
        records.append(runner.run(arm, segment))

    records.append(runner.run(arm, pick.joint_segment('approach', limits, runner.state.joints[arm], placement.q_pre,
                                                      grasp.open_width_m, servo_hz)))
    pick_window = grasp_window if low('pick') else None
    line('descend', poses['pick'], grasp.open_width_m, pick_window, contact_speed)
    width('close', grasp.open_width_m, grasp.close_width_m, pick.CLOSE_S, pick_window)
    width('squeeze', grasp.close_width_m, grasp.close_width_m, pick.SQUEEZE_S, pick_window)
    line('lift', poses['pre'], grasp.close_width_m, pick_window, contact_speed)
    if poses['up'][2, 3] > poses['pre'][2, 3] + 1e-6:
        line('rise', poses['up'], grasp.close_width_m, None, TRANSIT_SPEED_M_S)
    line('glide', poses['over'], grasp.close_width_m, None, TRANSIT_SPEED_M_S)
    near = place_window if low('begin') or low('goal') or low('back') else None
    line('lower', poses['begin'], grasp.close_width_m, near, contact_speed if near else TRANSIT_SPEED_M_S / 2)
    line('insert', poses['goal'], grasp.close_width_m, near, min(INSERT_SPEED_M_S, contact_speed))
    loose = placement.release_width_m
    width('release', grasp.close_width_m, loose, pick.CLOSE_S, near)
    line('back', poses['back'], loose, near, contact_speed if near else INSERT_SPEED_M_S)
    if poses['leave'][2, 3] > poses['back'][2, 3] + 1e-6:
        line('leave', poses['leave'], loose, near if low('back') else None,
             contact_speed if near else TRANSIT_SPEED_M_S)
    records.append(runner.run(arm, pick.joint_segment('park', limits, runner.state.joints[arm], runner.park[arm],
                                                      loose, servo_hz)))  # out of the other arm's way
    return records


def reordered_plan(task, layout, order):
    """The task's plan steps in `order` (fragment ids, every one once), each with its own insertion, checked to go in
    straight on the layout's cells as the plan is (interlock_repair.blocked): for trying an order the robot may manage
    where the plan's fails, not the plan every method is given."""
    from ..interlock_repair import blocked
    if not np.allclose(np.asarray(task.world_T_assembly)[:3, :3], np.eye(3)):
        raise ValueError('the assembly frame must be the world shifted, unrotated')
    steps = {step.fragment_id: step for step in task.assembly_plan}
    if sorted(order) != sorted(steps):
        raise ValueError(f'the order must name every fragment of the plan once: {sorted(steps)}')
    owner_of = {tuple(cell): piece['name'] for piece in layout['pieces'] for cell in piece['cells']}
    exits = {key: [int(round(-v)) for v in step.insertion_direction] for key, step in steps.items()}
    stuck = blocked(owner_of, [{'fragment': key, 'direction': exits[key]} for key in reversed(order)], task.anchor_id)
    if stuck is not None:
        raise ValueError(f'{stuck} cannot go in straight in the order {list(order)}')
    return tuple(steps[key] for key in order)


def run_assembly(backend, model, limits, task, layout, *, on_step: Callable | None = None, settle_s=STEP_SETTLE_S,
                 plan=None):
    """Every step of the task's plan (or of `plan`, reordered_plan's), each planned and run (plan_step, run_step) and
    followed by a hold of `settle_s`, then the ending every method gets and the score. Stops at the first step that
    cannot be planned or that S0 (or the backend) refuses. Returns {'steps': [StepRecord], 'error', 'ending_error',
    'score'}: a run that stopped, or whose ending did not finish, is not settled."""
    from ..baselines.scoring import score_assembly
    plan = task.assembly_plan if plan is None else plan
    if plan is None:
        raise ValueError('the task has no assembly plan')
    runner = Runner(backend, model, limits, on_step)
    steps, placed, error = [], [task.anchor_id], None
    for step in plan:
        record = StepRecord(step.fragment_id)
        steps.append(record)
        try:
            poses = {key: pose.world_T_fragment for key, pose in backend.observed_fragment_poses().items()}
            record.placement = plan_step(task, layout, model, limits, key=step.fragment_id,
                                         insertion=step.insertion_direction, poses=poses, placed=placed,
                                         seeds=runner.targets, runner=runner)
            run_step(runner, task, record.placement, record.segments)
            placed.append(step.fragment_id)
            record.segments.append(runner.hold(record.placement.arm, settle_s, 'settle'))
            record.drift_mm_deg = _drift(task, backend.privileged_fragment_poses(),
                                         [key for key in placed if key != task.anchor_id])
        except Exception as exc:  # planning, S0 or the backend: the run stops here, recorded
            record.error = f'{type(exc).__name__}: {exc}'
            error = record.error
            break
    ending = _ending(runner, task) if error is None else None
    score = score_assembly(task, backend.privileged_fragment_poses(), settled=error is None and ending is None)
    return {'steps': steps, 'error': error, 'ending_error': ending, 'score': score}


def _ending(runner, task):
    """Both grippers fully open, the arms held at their targets for the task's settle time, checked by S0 as one path;
    None, or why it was refused or stopped."""
    limits = runner.limits
    period = 1.0 / runner.servo_hz
    periods = max(1, int(np.ceil(task.settle_s / period - 1e-9)))
    state = runner.state
    hold = {arm: np.repeat(runner.targets[arm][None], periods, axis=0) for arm in ARMS}
    widths = {arm: limits.gripper_width_range_m[arm][1] for arm in ARMS}
    try:
        check_joint_path(runner.model, limits, now=state.time_s, q_now=state.joints, qd_now=state.joint_velocities,
                         qd_window_s=state.velocity_window_s, times=state.time_s + period * np.arange(1, periods + 1),
                         joints=hold)
    except Rejected as exc:
        return f'S0 refused the ending: {exc}'
    try:
        for k in range(periods):
            runner.state = runner.backend.step({arm: hold[arm][k] for arm in ARMS}, widths)
            if runner.on_step is not None:
                runner.on_step('ending', runner.state, {arm: hold[arm][k] for arm in ARMS}, None)
    except Exception as exc:  # the backend (a protective stop, say) or whoever watches the steps
        return f'the ending stopped: {type(exc).__name__}: {exc}'
    return None


def load_layout(task):
    with open(task.layout) as handle:
        return json.load(handle)
