"""Scripted top-down pick and place of one fragment through S0, on any AssemblyBackend (sim/check_grasp.py runs it on
Isaac, the tests on KinematicBackend). Planning and execution only; no Isaac import.

Grasp (choose_grasp): the fingers come straight down (TCP +x along world -z), fingertips FINGERTIP_HEIGHT_M above
the table, and close along a horizontal axis at some yaw. The fragment is measured on points covering its surface
every SURFACE_SPACING_M (surface_points), not on its mesh vertices, and the fingers on their meshes (sim/fingers.py:
solid pads over the last 15 mm, a window between two rails above): for each yaw, the chord is the width at which
the closing pads first meet the fragment, with the TCP moved along the closing axis so that both meet it at once,
and the open fingers must clear everything they pass on the way down by OPEN_MARGIN_M a side, within the gripper's
range. The narrowest chord wins. The pads then close to SQUEEZE_M below the chord, so each finger's drive presses
with its stiffness times SQUEEZE_M / 2.

Motion: every segment is planned from the measured state when it starts, checked by S0 (check_joint_path, with the
contact window it needs) and only then stepped at the servo rate. Joint moves and Cartesian lines follow a
raised-cosine profile within SPEED_FRACTION of S0's joint velocity and acceleration limits, and Cartesian lines that
touch the contact band stay under CONTACT_SPEED_FRACTION of S0's contact speed. The windows follow the S1 contract:
the pick window is centred on the fragment's observed pose for the descent, closing and lift; the place window on
its target for the lowering, release and retreat; the other arm holds still.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..contracts.actions import ARMS, ContactWindow
from ..s0.adapter import Rejected, check_joint_path
from ..s0.ik import IKOptions, solve_ik

FINGERTIP_HEIGHT_M = 0.003
OPEN_MARGIN_M = 0.01
SQUEEZE_M = 0.006
APPROACH_HEIGHT_M = 0.10
LIFT_M = APPROACH_HEIGHT_M  # the lift returns to the approach pose
CLOSE_S = 0.6
SQUEEZE_S = 0.4
HOLD_S = 2.0
SPEED_FRACTION = 0.5
CONTACT_SPEED_FRACTION = 0.6
# Planning speed near the table when S0's contact phase is not configured; S0 then refuses those segments itself.
UNCONFIGURED_CONTACT_SPEED_M_S = 0.03
YAWS_DEG = tuple(range(0, 180, 5))


SURFACE_SPACING_M = 0.002


def surface_points(vertices, faces, spacing_m=SURFACE_SPACING_M):
    """Points covering every triangle no further than `spacing_m` apart: what choose_grasp measures the fragment
    with. Its vertices alone would miss faces: a voxel mesh has one only every pitch, so a pad band crossing a face
    between them would see the fragment narrower than it is."""
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)
    corners = vertices[faces]
    edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2).max(axis=1)
    counts = np.maximum(1, np.ceil(edges / spacing_m)).astype(int)
    points = []
    for count in np.unique(counts):
        grid = np.array([(i, j) for i in range(count + 1) for j in range(count + 1 - i)], dtype=float) / count
        weights = np.column_stack([1 - grid.sum(axis=1), grid])
        group = corners[counts == count]
        points.append(np.einsum('pk,tkd->tpd', weights, group).reshape(-1, 3))
    return np.vstack(points)


@dataclass(frozen=True)
class Grasp:
    """A top-down grasp: TCP xy, closing-axis yaw, the chord the pads meet, and the open and closing widths."""

    fragment_id: str
    yaw_rad: float
    center_xy: tuple
    chord_m: float
    open_width_m: float
    close_width_m: float


def choose_grasp(fragment_id, points, table_z, max_width_m, *, fingers=None, yaws_deg=YAWS_DEG):
    """The narrowest top-down grasp of the fragment the open fingers can come down around; ValueError if none.
    `points` (N, 3) must cover its surface densely (surface_points of its mesh in the world); `fingers` is a
    FingerReach (the YAM's by default)."""
    from .fingers import yam_finger_reach
    fingers = yam_finger_reach() if fingers is None else fingers
    plus, minus = (next(tip for tip, side in fingers.side.items() if side == sign) for sign in (1.0, -1.0))
    points = np.asarray(points, dtype=float)
    centre = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
    heights = points[:, 2] - (table_z + FINGERTIP_HEIGHT_M)
    best, met = None, False
    for yaw in np.radians(yaws_deg):
        # the TCP's closing axis (its y) and the axis across it (its z, approach x closing) for this yaw
        closing, across = np.array([math.cos(yaw), math.sin(yaw)]), np.array([math.sin(yaw), -math.cos(yaw)])
        offsets = points[:, :2] - centre
        along, lateral = offsets @ closing, offsets @ across
        stops = fingers.stop_joints(heights, lateral, along)
        if not (np.isfinite(stops[plus]) and np.isfinite(stops[minus])):
            continue
        met = True
        shift = (stops[plus] - stops[minus]) / 2  # along the closing axis, to where both tips stop at the same joint
        meet = max((stops[plus] + stops[minus]) / 2, fingers.gripper.contact_joint_m)
        clear = fingers.clear_joints(heights, lateral, along - shift)
        opened = max(clear.values()) + OPEN_MARGIN_M
        if not opened <= fingers.gripper.travel_m:
            continue
        chord, open_width = fingers.gripper.width(meet), fingers.gripper.width(opened)
        if open_width > max_width_m or (best is not None and chord >= best.chord_m):
            continue
        middle = centre + closing * shift
        best = Grasp(fragment_id, float(yaw), tuple(float(v) for v in middle), chord, open_width,
                     max(0.0, chord - SQUEEZE_M))
    if best is None:
        if not met:
            raise ValueError(f'{fragment_id}: the pads meet the fragment at no yaw (nothing between them above the '
                             'fingertips)')
        raise ValueError(f'{fragment_id}: no top-down grasp fits inside a {max_width_m * 1000:.1f} mm opening')
    return best


def down_pose(xy, z, yaw):
    """TCP pose pointing straight down (TCP +x along world -z) with the fingers closing along the yaw direction."""
    approach = np.array([0.0, 0.0, -1.0])
    closing = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([approach, closing, np.cross(approach, closing)])
    pose[:3, 3] = [xy[0], xy[1], z]
    return pose


@dataclass(frozen=True)
class Segment:
    """Targets of the active arm at the servo rate: joints (N, 6), widths (N,), and the contact window they need."""

    name: str
    joints: np.ndarray
    widths: np.ndarray
    contact: ContactWindow | None = None


def raised_cosine(count):
    """Path fractions at the `count` servo instants of a raised-cosine move, ending at exactly 1."""
    return (1 - np.cos(np.pi * np.arange(1, count + 1) / count)) / 2


def _samples(duration_s, servo_hz):
    return max(1, math.ceil(duration_s * servo_hz - 1e-9))


def _joint_rates(q0, path, servo_hz):
    """Peak joint speeds and accelerations as check_joint_path measures them from rest: the first segment's
    acceleration over half a period, later ones over a full period."""
    velocity = np.diff(np.vstack([q0, path]), axis=0) * servo_hz
    spacing = np.full(len(path), 1.0 / servo_hz)
    spacing[0] /= 2
    acceleration = np.diff(np.vstack([np.zeros_like(q0), velocity]), axis=0) / spacing[:, None]
    return np.abs(velocity).max(axis=0), np.abs(acceleration).max(axis=0)


def _retime_factor(q0, path, limits, servo_hz):
    """How much longer a move must take for its joint rates to stay within SPEED_FRACTION of S0's limits."""
    speed, accel = _joint_rates(q0, path, servo_hz)
    v_max = SPEED_FRACTION * np.asarray(limits.max_joint_velocity_rad_s, dtype=float)
    a_max = SPEED_FRACTION * np.asarray(limits.max_joint_acceleration_rad_s2, dtype=float)
    return max(1.0, float(np.max(speed / v_max)), float(np.sqrt(np.max(accel / a_max))))


def joint_segment(name, limits, q0, q1, width, servo_hz, contact=None):
    """Raised-cosine joint-space move from q0 to q1, timed within SPEED_FRACTION of the joint limits."""
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    duration = 0.2
    for _ in range(8):
        path = q0 + np.outer(raised_cosine(_samples(duration, servo_hz)), q1 - q0)
        factor = _retime_factor(q0, path, limits, servo_hz)
        if factor <= 1.0:
            return Segment(name, path, np.full(len(path), float(width)), contact)
        duration *= factor * 1.05
    raise ValueError(f'{name}: could not time the joint move within the limits')


def cartesian_segment(name, arm_model, limits, q0, target, width, servo_hz, *, max_speed_m_s, contact=None):
    """Straight TCP line (slerped orientation) from the TCP at q0 to `target`, solved by S0's IK sample by sample."""
    q0 = np.asarray(q0, dtype=float)
    start = arm_model.tcp(q0)
    distance = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
    turn = Slerp([0.0, 1.0], Rotation.from_matrix([start[:3, :3], target[:3, :3]]))
    duration = max(0.2, math.pi / 2 * distance / max_speed_m_s)
    options = IKOptions(max_seed_jump_rad=0.2)
    for _ in range(8):
        fractions = raised_cosine(_samples(duration, servo_hz))
        path, seed = [], q0
        for fraction in fractions:
            pose = np.eye(4)
            pose[:3, :3] = turn(fraction).as_matrix()
            pose[:3, 3] = start[:3, 3] + fraction * (target[:3, 3] - start[:3, 3])
            solution = solve_ik(arm_model, pose, seed, options)
            if not solution.ok:
                raise ValueError(f'{name}: IK failed ({solution.reason}) at {fraction:.2f} of the line')
            path.append(solution.q)
            seed = solution.q
        path = np.array(path)
        factor = _retime_factor(q0, path, limits, servo_hz)
        if factor <= 1.0:
            return Segment(name, path, np.full(len(path), float(width)), contact)
        duration *= factor * 1.05
    raise ValueError(f'{name}: could not time the line within the limits')


def width_segment(name, q, start_width, end_width, duration_s, servo_hz, contact=None):
    """Arm held at q while the width ramps linearly from start_width to end_width."""
    count = _samples(duration_s, servo_hz)
    widths = start_width + (end_width - start_width) * np.arange(1, count + 1) / count
    return Segment(name, np.repeat(np.asarray(q, dtype=float)[None], count, axis=0), widths, contact)


@dataclass(frozen=True)
class SegmentRecord:
    """One executed segment: its name, how many servo steps, its contact window, and its first and last width."""

    name: str
    count: int
    window: ContactWindow | None
    first_width_m: float
    last_width_m: float

    def to_record(self):
        window = None if self.window is None else {'kind': self.window.kind, 'center_xy': list(self.window.center_xy),
                                                   'fragment_id': self.window.fragment_id}
        return {'name': self.name, 'count': self.count, 'window': window, 'first_width_m': self.first_width_m,
                'last_width_m': self.last_width_m}


@dataclass(frozen=True)
class PickRecord:
    """What run_pick_and_place did: the grasp, the place target, and each segment (SegmentRecord)."""

    arm: str
    grasp: Grasp
    place_xy: tuple
    segments: tuple


def solve_reachable(arm_model, pose, seed, *, yaw_flip=True):
    """IK for `pose` from far away (seed jumps allowed), trying the 180-degree turned pose too: the fingers are
    symmetric. Returns (q, pose) or raises ValueError."""
    options = IKOptions(max_seed_jump_rad=3.5)
    candidates = [pose]
    if yaw_flip:
        flipped = pose.copy()
        flipped[:3, 1:3] *= -1
        candidates.append(flipped)
    reasons = []
    for candidate in candidates:
        solution = solve_ik(arm_model, candidate, seed, options)
        if solution.ok:
            return solution.q, candidate
        reasons.append(solution.reason)
    raise ValueError(f'no IK solution for the pre-grasp pose ({", ".join(reasons)})')


def run_pick_and_place(backend, model, limits, *, arm, fragment_id, points, carry_xy, tip_reach_m=0.0,
                       on_step: Callable | None = None):
    """Pick `fragment_id` (world mesh points `points` at its observed pose) with `arm`, lift it, carry it by
    `carry_xy`, put it down, release it, retreat and hold for the task's settle time. Raises Rejected when S0
    refuses a segment and ValueError when planning fails; `on_step(segment_name, state)` sees every state."""
    if arm not in ARMS:
        raise ValueError(f'arm must be one of {ARMS}')
    other = next(name for name in ARMS if name != arm)
    arm_model, servo_hz = model.arms[arm], float(limits.servo_hz)
    table = limits.table_top_z_m
    max_width = limits.gripper_width_range_m[arm][1]
    grasp = choose_grasp(fragment_id, points, table, max_width)
    observed = backend.observed_fragment_poses()[fragment_id].world_T_fragment
    pick = ContactWindow(arm, 'grasp', tuple(observed[:2, 3]), fragment_id)
    place_xy = tuple(float(v) for v in observed[:2, 3] + np.asarray(carry_xy, dtype=float))
    place = ContactWindow(arm, 'place', place_xy, fragment_id)
    grasp_z = table + FINGERTIP_HEIGHT_M + tip_reach_m
    contact_speed = (CONTACT_SPEED_FRACTION * limits.contact_max_speed_m_s if limits.contact_enabled
                     else UNCONFIGURED_CONTACT_SPEED_M_S)
    state = backend.robot_state()
    hold = np.asarray(state.joints[other], dtype=float)
    q_pre, pre = solve_reachable(arm_model, down_pose(grasp.center_xy, grasp_z + APPROACH_HEIGHT_M, grasp.yaw_rad),
                                 state.joints[arm])
    at = {'grasp': pre.copy(), 'carry': pre.copy()}
    at['grasp'][2, 3] = grasp_z
    at['carry'][:2, 3] += np.asarray(carry_xy, dtype=float)
    at['lower'] = at['carry'].copy()
    at['lower'][2, 3] = grasp_z
    at['retreat'] = at['carry'].copy()
    log = []

    def run(segment):
        nonlocal state
        times = state.time_s + np.arange(1, len(segment.joints) + 1) / servo_hz
        joints = {arm: segment.joints, other: np.repeat(hold[None], len(segment.joints), axis=0)}
        check_joint_path(model, limits, now=state.time_s, q_now=state.joints, qd_now=state.joint_velocities,
                         times=times, joints=joints, contacts=() if segment.contact is None else (segment.contact,))
        for k in range(len(segment.joints)):
            state = backend.step({arm: segment.joints[k], other: hold},
                                 {arm: float(segment.widths[k]), other: limits.gripper_width_range_m[other][1]})
            if on_step is not None:
                on_step(segment.name, state)
        log.append(SegmentRecord(segment.name, len(segment.joints), segment.contact, float(segment.widths[0]),
                                 float(segment.widths[-1])))

    def line(name, target, width, contact, speed):
        return cartesian_segment(name, arm_model, limits, state.joints[arm], target, width, servo_hz,
                                 max_speed_m_s=speed, contact=contact)

    run(joint_segment('approach', limits, state.joints[arm], q_pre, grasp.open_width_m, servo_hz))
    run(line('descend', at['grasp'], grasp.open_width_m, pick, contact_speed))
    run(width_segment('close', state.joints[arm], grasp.open_width_m, grasp.close_width_m, CLOSE_S, servo_hz, pick))
    run(width_segment('squeeze', state.joints[arm], grasp.close_width_m, grasp.close_width_m, SQUEEZE_S, servo_hz,
                      pick))
    run(line('lift', pre, grasp.close_width_m, pick, contact_speed))
    run(width_segment('hold', state.joints[arm], grasp.close_width_m, grasp.close_width_m, HOLD_S, servo_hz))
    run(line('carry', at['carry'], grasp.close_width_m, None, contact_speed))
    run(line('lower', at['lower'], grasp.close_width_m, place, contact_speed))
    run(width_segment('release', state.joints[arm], grasp.close_width_m, grasp.open_width_m, CLOSE_S, servo_hz, place))
    run(line('retreat', at['retreat'], grasp.open_width_m, place, contact_speed))
    settle = max(1.0 / servo_hz, float(getattr(backend.task, 'settle_s', 1.0)))
    run(width_segment('settle', state.joints[arm], grasp.open_width_m, grasp.open_width_m, settle, servo_hz))
    return PickRecord(arm, grasp, place_xy, tuple(log))
