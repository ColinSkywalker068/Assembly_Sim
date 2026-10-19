"""S1 -> S0 action chunk: dual-TCP waypoints relative to a per-hand anchor fixed for the chunk.

Each waypoint is 20 numbers: left(position3, rotation6, width1), right(position3, rotation6,
width1). rotation6 is the first, then the second column of the rotation matrix. A hand's
target is `T_world_target = T_world_anchor @ T_anchor_target`; widths are absolute metres.
Times are seconds on the episode monotonic clock (the clock observations are stamped with).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..perception import rigid_transform

ARMS = ('left', 'right')
HAND_DIM = 10
ACTION_DIM = len(ARMS) * HAND_DIM
CONTROL_MODES = ('tcp_relative_to_anchor',)
S0_STATUSES = ('accepted', 'rejected', 'holding', 'completed', 'failed')
CONTACT_KINDS = ('grasp', 'place')
MIN_COLUMN_NORM = 1e-6
# |a1 x a2| / (|a1| |a2|) below this is treated as collinear (about 0.06 degrees).
MIN_COLUMN_SINE = 1e-3


class ActionRejected(ValueError):
    """A chunk that S0 must not execute; the message is the structured rejection reason."""


def _frozen(value, dtype=float):
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def rotation6_from_matrix(rotation):
    """(..., 3, 3) rotation matrices -> (..., 6) first and second columns."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError('expected (..., 3, 3) rotation matrices')
    return np.concatenate([rotation[..., :, 0], rotation[..., :, 1]], axis=-1)


def matrix_from_rotation6(values):
    """(..., 6) -> (..., 3, 3) by Gram-Schmidt; rejects instead of inventing an orientation."""
    values = np.asarray(values, dtype=float)
    if values.shape[-1] != 6:
        raise ValueError('expected (..., 6) rotation6 values')
    if not np.isfinite(values).all():
        raise ActionRejected('nonfinite rotation6')
    a1, a2 = values[..., :3], values[..., 3:]
    n1, n2 = np.linalg.norm(a1, axis=-1), np.linalg.norm(a2, axis=-1)
    if (n1 < MIN_COLUMN_NORM).any() or (n2 < MIN_COLUMN_NORM).any():
        raise ActionRejected('zero-length rotation6 column')
    if (np.linalg.norm(np.cross(a1, a2), axis=-1) / (n1 * n2) < MIN_COLUMN_SINE).any():
        raise ActionRejected('near-collinear rotation6 columns')
    b1 = a1 / n1[..., None]
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


def hand_vector(anchor_T_target, width):
    """(..., 4, 4) relative poses and (...) widths -> (..., 10) hand vectors."""
    anchor_T_target = np.asarray(anchor_T_target, dtype=float)
    width = np.asarray(width, dtype=float)
    return np.concatenate([anchor_T_target[..., :3, 3], rotation6_from_matrix(anchor_T_target[..., :3, :3]),
                           width[..., None]], axis=-1)


def hand_pose(vector):
    """(..., 10) hand vectors -> ((..., 4, 4) relative poses, (...) widths)."""
    vector = np.asarray(vector, dtype=float)
    if vector.shape[-1] != HAND_DIM:
        raise ValueError(f'expected (..., {HAND_DIM}) hand vectors')
    pose = np.zeros(vector.shape[:-1] + (4, 4))
    pose[..., :3, :3] = matrix_from_rotation6(vector[..., 3:9])
    pose[..., :3, 3] = vector[..., :3]
    pose[..., 3, 3] = 1.0
    return pose, vector[..., 9].copy()


def hand_slice(arm):
    index = ARMS.index(arm)
    return slice(index * HAND_DIM, (index + 1) * HAND_DIM)


@dataclass(frozen=True)
class ContactWindow:
    """A planned grasp or place contact a chunk declares for one arm; S0 decides whether to allow it (s0/adapter.py).

    Within S0's contact radius of `center_xy` (world x, y in metres), S0 lets the arm's fingers come closer to
    the table than the normal tool clearance, at limited speed, and replaces the TCP-height rule by the contact
    clearance; the gripper body, the links and the other arm keep their normal limits. `fragment_id` names the
    fragment grasped or placed. A retreat declares no contact of its own and keeps the arm's previous one
    (contracts.skills.contact_windows).
    """

    arm: str
    kind: str
    center_xy: tuple[float, float]
    fragment_id: str

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ActionRejected(f'contact arm must be one of {ARMS}')
        if self.kind not in CONTACT_KINDS:
            raise ActionRejected(f'contact kind must be one of {CONTACT_KINDS}')
        try:
            center = np.array(self.center_xy, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ActionRejected(f'contact center_xy: {exc}') from exc
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ActionRejected('contact center_xy must be two finite world coordinates (m)')
        if not isinstance(self.fragment_id, str) or not self.fragment_id:
            raise ActionRejected(f'a {self.kind} contact needs the fragment_id it acts on')
        object.__setattr__(self, 'center_xy', (float(center[0]), float(center[1])))


@dataclass(frozen=True)
class ActionChunk:
    request_id: str
    chunk_id: int
    observation_time: float
    valid_until: float
    anchors: Mapping[str, np.ndarray]
    waypoint_times: np.ndarray
    waypoints: np.ndarray
    active_arms: tuple[str, ...]
    control_mode: str = 'tcp_relative_to_anchor'
    contacts: tuple[ContactWindow, ...] = ()

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ActionRejected('missing request_id')
        if isinstance(self.chunk_id, bool) or not isinstance(self.chunk_id, int) or self.chunk_id < 0:
            raise ActionRejected('chunk_id must be a non-negative integer')
        if self.control_mode not in CONTROL_MODES:
            raise ActionRejected(f'unsupported control mode {self.control_mode!r}')
        if set(self.anchors) != set(ARMS):
            raise ActionRejected('both arms need an anchor, active or not')
        active = tuple(self.active_arms)
        if len(set(active)) != len(active) or not set(active) <= set(ARMS):
            raise ActionRejected('active_arms must be distinct members of left/right')
        anchors = {}
        for arm in ARMS:
            try:
                anchors[arm] = _frozen(rigid_transform(self.anchors[arm]))
            except ValueError as exc:
                raise ActionRejected(f'{arm} anchor: {exc}') from exc
        times = _frozen(self.waypoint_times)
        points = _frozen(self.waypoints)
        observed, valid_until = float(self.observation_time), float(self.valid_until)
        if not np.isfinite([observed, valid_until]).all() or valid_until < observed:
            raise ActionRejected('observation_time/valid_until must be finite and ordered')
        if times.ndim != 1 or len(times) < 1 or points.shape != (len(times), ACTION_DIM):
            raise ActionRejected(f'expected N>=1 times and an (N, {ACTION_DIM}) waypoint array')
        if not np.isfinite(times).all() or not np.isfinite(points).all():
            raise ActionRejected('nonfinite waypoint times or values')
        if (np.diff(times) <= 0).any() or times[0] <= observed:
            raise ActionRejected('waypoint times must be strictly increasing and after observation_time')
        widths = points[:, [hand_slice(arm).stop - 1 for arm in ARMS]]
        if (widths < 0).any():
            raise ActionRejected('gripper widths must be non-negative metres')
        for arm in ARMS:
            matrix_from_rotation6(points[:, hand_slice(arm)][:, 3:9])
        try:
            contacts = tuple(self.contacts)
        except TypeError as exc:
            raise ActionRejected('contacts must be a sequence of ContactWindow declarations') from exc
        if not all(isinstance(contact, ContactWindow) for contact in contacts):
            raise ActionRejected('contacts must be ContactWindow declarations')
        if len({contact.arm for contact in contacts}) != len(contacts):
            raise ActionRejected('a chunk declares at most one contact per arm')
        object.__setattr__(self, 'contacts', contacts)
        object.__setattr__(self, 'anchors', MappingProxyType(anchors))
        object.__setattr__(self, 'waypoint_times', times)
        object.__setattr__(self, 'waypoints', points)
        object.__setattr__(self, 'active_arms', active)
        object.__setattr__(self, 'observation_time', observed)
        object.__setattr__(self, 'valid_until', valid_until)


def encode_chunk(*, request_id, chunk_id, observation_time, valid_until, anchors, waypoint_times,
                 targets, widths, active_arms, contacts=()):
    """Absolute world targets per arm ((N, 4, 4) and (N,)) -> ActionChunk relative to anchors."""
    columns = []
    for arm in ARMS:
        anchor = rigid_transform(anchors[arm])
        world_T_target = np.asarray(targets[arm], dtype=float)
        if world_T_target.ndim != 3 or world_T_target.shape[1:] != (4, 4):
            raise ActionRejected(f'{arm} targets must be (N, 4, 4)')
        try:
            for target in world_T_target:
                rigid_transform(target)
        except ValueError as exc:
            raise ActionRejected(f'{arm} target: {exc}') from exc
        relative = np.linalg.inv(anchor) @ world_T_target
        columns.append(hand_vector(relative, widths[arm]))
    return ActionChunk(request_id=request_id, chunk_id=chunk_id, observation_time=observation_time,
                       valid_until=valid_until, anchors=anchors, waypoint_times=waypoint_times,
                       waypoints=np.concatenate(columns, axis=-1), active_arms=tuple(active_arms),
                       contacts=contacts)


def decode_chunk(chunk):
    """ActionChunk -> {arm: ((N, 4, 4) world targets, (N,) widths)}."""
    result = {}
    for arm in ARMS:
        relative, width = hand_pose(chunk.waypoints[:, hand_slice(arm)])
        result[arm] = (chunk.anchors[arm] @ relative, width)
    return result


def inactive_arm_violations(chunk, *, position_tolerance_m=1e-4, rotation_tolerance_rad=1e-3,
                            width_tolerance_m=1e-4):
    """Inactive arms must hold their anchor pose and keep a constant width."""
    violations = []
    for arm in ARMS:
        if arm in chunk.active_arms:
            continue
        relative, width = hand_pose(chunk.waypoints[:, hand_slice(arm)])
        translation = np.linalg.norm(relative[:, :3, 3], axis=1).max()
        cosine = np.clip((np.trace(relative[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)
        rotation = np.arccos(cosine).max()
        if translation > position_tolerance_m or rotation > rotation_tolerance_rad or np.ptp(width) > width_tolerance_m:
            violations.append(arm)
    return violations


def executable_reason(chunk, now, *, max_observation_age_s=None):
    """None when the chunk may start at `now`, otherwise why S0 must reject it.

    `valid_until` is S1's own claim; `max_observation_age_s` is S0's independent limit on how old
    the observation behind a chunk may be when execution starts.
    """
    if not np.isfinite(now):
        return 'nonfinite clock'
    if now < chunk.observation_time:
        return 'clock is earlier than observation_time (different clock domains?)'
    if max_observation_age_s is not None and now - chunk.observation_time > max_observation_age_s:
        return f'observation is {now - chunk.observation_time:.3f} s old (limit {max_observation_age_s} s)'
    if now > chunk.valid_until:
        return 'stale chunk: valid_until has passed'
    if now >= chunk.waypoint_times[-1]:
        return 'every waypoint is already in the past'
    moved = inactive_arm_violations(chunk)
    if moved:
        return f'inactive arm moves: {", ".join(moved)}'
    return None


@dataclass(frozen=True)
class S0Status:
    """What S0 reports back. A completed chunk is not an assembly success; an evaluator decides that."""

    request_id: str
    chunk_id: int
    status: str
    timestamp: float
    reason: str | None = None
    measured_tcp: Mapping[str, np.ndarray] | None = None
    measured_width: Mapping[str, float] | None = None
    tracking_error: Mapping[str, Mapping[str, float]] | None = None

    def __post_init__(self):
        if self.status not in S0_STATUSES:
            raise ValueError(f'unknown S0 status {self.status!r}')
        if self.status in ('rejected', 'failed') and not self.reason:
            raise ValueError('rejected and failed statuses need a reason')
