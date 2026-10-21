"""S0 motion adapter: an S1 ActionChunk becomes a servo-rate dual-arm joint trajectory, or a rejection.

Every check rejects the whole chunk; nothing is clipped, slowed down or re-timed into a different
motion. A rejection comes with a braking command: each arm's joints decelerate within the
acceleration limits so that they stop together, and the gripper widths stay at their last
COMMANDED values, so stopping never releases a fragment the robot is holding or supporting. For a
position-controlled gripper the command, not the measured opening, sets the squeeze: a gripper
commanded to 0.020 m that a fragment blocks at 0.030 m loses its grip if it is "held" at 0.030 m.
Measured widths set no command; they bound how far the fingers reach (below). An accepted chunk
must also end in a state from which that same braking stop stays inside the joint limits.

The collision checks are coarse: bounding capsules for the links (s0/collision.py) and
flange-frame boxes round the gripper, evaluated at the servo samples only: one round the body and
the fingers over their whole stroke or, with `finger_opening_margin_m` set, the body's box and the
fingers' boxes (slabs of each tip, s0/kinematics.py) at every opening up to the widest the gripper
can have in the motion checked (widest_opening: the measured opening, the last command and every
new one) plus that margin. That bound is read once, at the chunk's start: fingers pushed wider while it
runs (a fragment turning in the grip) are the margin's to cover. They catch
gross arm-arm, arm-table and tool-table interference; they are not continuous collision checking
against meshes, they do not check self-collision within one arm, braking paths are not
collision-checked, and fragments are not collision objects at all.

Planned contact (grasping or placing close to the table) is a contact phase the chunk declares
(contracts.actions.ContactWindow, one per arm). Only where the normal tool-table or TCP-height rule
fails does S0 consult it, and then it lets the sample through only if
  - the TCP and every finger-box corner below the normal floor lie within `contact_region_radius_m`
    of the declared centre (x, y);
  - the fingers' box and the TCP stay `contact_table_clearance_m` above the table, while the gripper
    body's box keeps the normal `table_clearance_m`;
  - over every servo interval next to such a sample (and the first interval when the measured state
    itself needs the contact), samples are at most one servo period apart and every finger-box
    corner moves slower than `contact_max_speed_m_s`.
"Every corner below the normal floor" includes the points where the box's edges cross the floor, so
the whole part of the finger box below it is held to the region. S0Adapter also keeps each
request's contact where its first accepted chunk declared it. There are grasp and place contacts
only: a retreat keeps the arm's previous one (contracts.skills.contact_windows). Links, the other
arm and arm-arm clearance keep their normal rules. Speed outside the contact band is not limited
beyond the joint limits: an approach decelerates as planned, and tracking overshoot is the
controller's to bound.
"""
from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from ..contracts.actions import ARMS, ContactWindow, S0Status, decode_chunk, executable_reason
from ..perception import rigid_transform
from .collision import capsule_clearance, lowest_point_z
from .ik import IKOptions, solve_ik
from .kinematics import BOX_EDGES

# base_link sits on the table and link_1 only spins about the vertical axis (its height is
# constant), so neither can approach the table; the gripper is covered by the tool box rule.
TABLE_CHECKED_LINKS = ('link_2', 'link_3', 'link_4', 'link_5', 'link_6')
CHECK_OVERSAMPLE = 4
# Samples next to a contact may be this much longer than one servo period: a period rounded to 0.1 ms, or float
# steps of a clock that has run for months, are still one period.
CONTACT_SPACING_SLACK = 1.01


@dataclass(frozen=True)
class S0Limits:
    """Every physical limit is explicit; nothing about the YAM gripper or CRX dynamics is assumed.

    The three contact_* values configure the contact phase (all three or none): without them a declared
    contact never relaxes anything, and a chunk that needs one is rejected. finger_opening_margin_m: None, S0's boxes
    hold the fingers over their whole stroke; a length, at every opening up to the widest the gripper can have in the
    motion checked plus this much (the gripper's overshoot past its command and its width calibration), and then every
    check needs each gripper's widest opening (check_geometry).
    """

    max_joint_velocity_rad_s: np.ndarray
    max_joint_acceleration_rad_s2: np.ndarray
    gripper_width_range_m: Mapping[str, tuple[float, float]]
    arm_arm_clearance_m: float
    table_top_z_m: float
    table_clearance_m: float
    tcp_min_height_m: float
    servo_hz: int
    max_observation_age_s: float
    anchor_position_tolerance_m: float
    anchor_rotation_tolerance_rad: float
    inactive_width_tolerance_m: float = 1e-4
    contact_table_clearance_m: float | None = None
    contact_region_radius_m: float | None = None
    contact_max_speed_m_s: float | None = None
    finger_opening_margin_m: float | None = None

    @property
    def contact_enabled(self):
        return self.contact_table_clearance_m is not None

    def __post_init__(self):
        for name in ('max_joint_velocity_rad_s', 'max_joint_acceleration_rad_s2'):
            value = np.array(getattr(self, name), dtype=float)
            if value.shape != (6,) or not np.isfinite(value).all() or (value <= 0).any():
                raise ValueError(f'{name} must be six positive finite values')
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if set(self.gripper_width_range_m) != set(ARMS):
            raise ValueError('gripper width range needed for both arms')
        for arm, (low, high) in self.gripper_width_range_m.items():
            if not 0 <= low < high:
                raise ValueError(f'{arm} gripper width range must satisfy 0 <= min < max')
        if isinstance(self.servo_hz, bool) or not isinstance(self.servo_hz, int) or self.servo_hz <= 0:
            raise ValueError('servo_hz must be a positive integer')
        scalars = (self.arm_arm_clearance_m, self.table_clearance_m, self.tcp_min_height_m,
                   self.inactive_width_tolerance_m)
        if not np.isfinite([self.table_top_z_m, *scalars]).all() or min(scalars) < 0:
            raise ValueError('clearances and tolerances must be finite and non-negative')
        positive = (self.max_observation_age_s, self.anchor_position_tolerance_m, self.anchor_rotation_tolerance_rad)
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError('observation age and anchor tolerances must be finite and positive')
        contact = (self.contact_table_clearance_m, self.contact_region_radius_m, self.contact_max_speed_m_s)
        if any(value is not None for value in contact):
            if any(value is None for value in contact) or not np.isfinite(contact).all():
                raise ValueError('the contact phase needs a finite clearance, region radius and speed limit, or none')
            clearance, radius, speed = contact
            if not 0 < clearance <= self.table_clearance_m or not radius > 0 or not speed > 0:
                raise ValueError('contact clearance must be in (0, table_clearance_m]; region radius and speed '
                                 'limit must be positive')
        margin = self.finger_opening_margin_m
        if margin is not None and (isinstance(margin, bool) or not isinstance(margin, numbers.Real)
                                   or not np.isfinite(margin) or margin < 0):
            raise ValueError('finger_opening_margin_m must be None or a finite length >= 0')


@dataclass(frozen=True)
class JointTrajectory:
    """Joint and width targets to reach at `times` (seconds, episode monotonic clock).

    `hold` marks a braking command after a rejection; `note` reports a braking stop that would
    itself leave the joint limits (nothing better than braking is available in that case).
    """

    request_id: str
    chunk_id: int
    times: np.ndarray
    joints: Mapping[str, np.ndarray]
    widths: Mapping[str, np.ndarray]
    hold: bool
    note: str | None = None


class Rejected(Exception):
    pass


def braking_profile(q0, qd0, max_acceleration, elapsed):
    """Joints at `elapsed` seconds while every joint decelerates so that all stop at the same time.

    Each joint i uses the constant deceleration qd0_i / T with T = max_i |qd0_i| / a_i, which never
    exceeds a_i. Returns (positions (len(elapsed), dof), stop time T, stop position q0 + qd0 T / 2).
    """
    q0, qd0 = np.asarray(q0, dtype=float), np.asarray(qd0, dtype=float)
    duration = float(np.max(np.abs(qd0) / np.asarray(max_acceleration, dtype=float)))
    t = np.clip(np.asarray(elapsed, dtype=float), 0.0, duration)[:, None]
    if duration == 0.0:
        return np.repeat(q0[None], len(t), axis=0), 0.0, q0.copy()
    deceleration = qd0 / duration
    return q0 + qd0 * t - 0.5 * deceleration * t * t, duration, q0 + 0.5 * qd0 * duration


def _frozen(value):
    array = np.array(value, dtype=float, copy=True)
    array.flags.writeable = False
    return array


class S0Adapter:
    def __init__(self, model, limits: S0Limits, ik_options: IKOptions = IKOptions()):
        missing = [arm for arm in ARMS if not model.arms[arm].has_tool_box]
        if missing:
            raise ValueError(f'S0 needs the tool box of every arm for tool-table checks; missing: {missing}')
        self.model, self.limits, self.ik_options = model, limits, ik_options
        self._last_chunk = {}
        self._contacts = {}  # (request_id, arm) -> (kind, center_xy) of the first accepted chunk that declared one

    def plan(self, chunk, *, now, joints, joint_velocities, widths, commanded_widths, tcp_at_observation):
        """Return (S0Status, JointTrajectory).

        `widths` are the measured openings as far as the fingers reach (RobotSnapshot.spans): they set no
        command, and bound the boxes with limits.finger_opening_margin_m. `commanded_widths[arm]` is the last width
        S0 commanded/accepted for that gripper (from S0's own command history): holds and brakes keep
        it, the inactive-arm rule compares the chunk against it and active-arm width motion starts from
        it. `tcp_at_observation[arm]` is the TCP the caller measured at `chunk.observation_time` (from
        its state history); each anchor must match it. A rejection returns a braking command. Invalid
        measured state, commanded widths or clock returns ('failed', None): there is no valid state to
        brake from, so the caller must stop the robot by other means (e-stop).
        """
        try:
            q_now, qd_now, w_measured, w_command = self._measured(now, joints, joint_velocities, widths,
                                                                  commanded_widths)
        except (KeyError, TypeError, ValueError) as exc:
            return S0Status(chunk.request_id, chunk.chunk_id, 'failed', float(now), reason=str(exc)), None
        now = float(now)
        measured = {arm: self.model.arms[arm].tcp(q_now[arm]) for arm in ARMS}
        try:
            reason = executable_reason(chunk, now, max_observation_age_s=self.limits.max_observation_age_s)
            if reason:
                raise Rejected(reason)
            self._check_order(chunk)
            self._check_contacts_fixed(chunk)
            self._check_anchors(chunk, tcp_at_observation)
            trajectory = self._trajectory(chunk, now, q_now, qd_now, w_command, w_measured)
        except (Rejected, ValueError) as exc:  # ValueError covers ActionRejected from the contracts
            brake = self._brake(chunk, now, q_now, qd_now, w_command)
            reason = str(exc) if brake.note is None else f'{exc}; {brake.note}'
            return S0Status(chunk.request_id, chunk.chunk_id, 'rejected', now, reason=reason, measured_tcp=measured,
                            measured_width=dict(w_measured)), brake
        self._last_chunk[chunk.request_id] = chunk.chunk_id
        for contact in chunk.contacts:
            self._contacts.setdefault((chunk.request_id, contact.arm), (contact.kind, contact.center_xy))
        return S0Status(chunk.request_id, chunk.chunk_id, 'accepted', now, measured_tcp=measured,
                        measured_width=dict(w_measured)), trajectory

    def _measured(self, now, joints, joint_velocities, widths, commanded_widths):
        if not np.isfinite(now):
            raise ValueError('clock is not finite')
        q_now, qd_now, w_measured, w_command = {}, {}, {}, {}
        for arm in ARMS:
            model = self.model.arms[arm]
            q = np.asarray(joints[arm], dtype=float)
            qd = np.asarray(joint_velocities[arm], dtype=float)
            if q.shape != (model.dof,) or qd.shape != (model.dof,) or not np.isfinite([*q, *qd]).all():
                raise ValueError(f'{arm}: measured joint state must be finite with {model.dof} values')
            if (q < model.lower - 1e-6).any() or (q > model.upper + 1e-6).any():
                raise ValueError(f'{arm}: measured joints are outside the model limits')
            if not np.isfinite(widths[arm]):
                raise ValueError(f'{arm}: measured width is not finite')
            command = float(commanded_widths[arm])
            low, high = self.limits.gripper_width_range_m[arm]
            if not np.isfinite(command) or not low <= command <= high:
                raise ValueError(f'{arm}: commanded width {command} is not inside the gripper range [{low}, {high}] m')
            q_now[arm], qd_now[arm], w_measured[arm], w_command[arm] = q, qd, float(widths[arm]), command
        return q_now, qd_now, w_measured, w_command

    def _check_order(self, chunk):
        last = self._last_chunk.get(chunk.request_id)
        if last is not None and chunk.chunk_id <= last:
            raise Rejected(f'chunk_id {chunk.chunk_id} of {chunk.request_id} is not after the last accepted chunk {last}')

    def _check_contacts_fixed(self, chunk):
        """A request's contact stays where its first accepted chunk declared it, so chunk by chunk it cannot creep."""
        for contact in chunk.contacts:
            first = self._contacts.get((chunk.request_id, contact.arm))
            if first is not None and first != (contact.kind, contact.center_xy):
                raise Rejected(f'the {contact.arm} contact moved within request {chunk.request_id}: {first} became '
                               f'{(contact.kind, contact.center_xy)}')

    def _check_anchors(self, chunk, tcp_at_observation):
        limits = self.limits
        for arm in ARMS:
            try:
                reference = rigid_transform(tcp_at_observation[arm])
            except (KeyError, TypeError, ValueError) as exc:
                raise Rejected(f'{arm} tcp_at_observation is not a valid pose: {exc}') from exc
            anchor = chunk.anchors[arm]
            position = float(np.linalg.norm(anchor[:3, 3] - reference[:3, 3]))
            rotation = float(Rotation.from_matrix(anchor[:3, :3] @ reference[:3, :3].T).magnitude())
            if position > limits.anchor_position_tolerance_m or rotation > limits.anchor_rotation_tolerance_rad:
                raise Rejected(f'{arm} anchor is {position * 1000:.1f} mm / {np.degrees(rotation):.2f} deg from the TCP '
                               f'measured at observation_time')

    def _servo_times(self, now, end):
        step = 1.0 / self.limits.servo_hz
        count = int(np.floor((end - now) / step + 1e-9))
        times = now + step * np.arange(1, count + 1)
        if not len(times) or times[-1] < end - 1e-9:
            times = np.append(times, end)
        return times

    def _outside_limits(self, arm, positions):
        model = self.model.arms[arm]
        excess = np.maximum(model.lower - positions, positions - model.upper)
        if (excess > 0).any():
            joint = int(np.argmax(excess.max(axis=0) if excess.ndim > 1 else excess))
            return joint
        return None

    def _brake(self, chunk, now, q_now, qd_now, w_command):
        acceleration = self.limits.max_joint_acceleration_rad_s2
        durations = {arm: braking_profile(q_now[arm], qd_now[arm], acceleration, [0.0])[1] for arm in ARMS}
        longest = max(durations.values())
        times = np.array([now]) if longest == 0.0 else self._servo_times(now, now + longest)
        joints, notes = {}, []
        for arm in ARMS:
            positions, _, stop = braking_profile(q_now[arm], qd_now[arm], acceleration, times - now)
            joints[arm] = _frozen(positions)
            joint = self._outside_limits(arm, stop)
            if joint is not None:
                notes.append(f'braking {arm} would pass the joint_{joint + 1} limit (stop at {stop[joint]:.3f} rad)')
        widths = {arm: _frozen(np.full(len(times), w_command[arm])) for arm in ARMS}
        return JointTrajectory(chunk.request_id, chunk.chunk_id, _frozen(times), joints, widths, hold=True,
                               note='; '.join(notes) or None)

    def _trajectory(self, chunk, now, q_now, qd_now, w_command, w_measured):
        decoded = decode_chunk(chunk)
        keep = chunk.waypoint_times > now
        knots = np.concatenate([[now], chunk.waypoint_times[keep]])
        samples = self._servo_times(now, knots[-1])
        acceleration = self.limits.max_joint_acceleration_rad_s2
        joints, widths = {}, {}
        for arm in ARMS:
            targets, chunk_widths = decoded[arm][0][keep], decoded[arm][1][keep]
            if arm not in chunk.active_arms:
                if np.abs(chunk_widths - w_command[arm]).max() > self.limits.inactive_width_tolerance_m:
                    raise Rejected(f'inactive arm {arm}: chunk width differs from the last commanded width')
                positions, _, stop = braking_profile(q_now[arm], qd_now[arm], acceleration, samples - now)
                joint = self._outside_limits(arm, stop)
                if joint is not None:
                    raise Rejected(f'inactive arm {arm} cannot stop inside the joint_{joint + 1} limit')
                joints[arm] = positions
                widths[arm] = np.full(len(samples), w_command[arm])
                continue
            low, high = self.limits.gripper_width_range_m[arm]
            outside = (chunk_widths < low) | (chunk_widths > high)
            if outside.any():
                raise Rejected(f'{arm} width {chunk_widths[outside][0]:.4f} m outside [{low}, {high}] m')
            path = [q_now[arm]]
            for index, target in enumerate(targets):
                result = solve_ik(self.model.arms[arm], target, path[-1], self.ik_options)
                if not result.ok:
                    raise Rejected(f'{arm} IK {result.reason} at waypoint {index} (t={knots[index + 1]:.3f} s, '
                                   f'position error {result.position_error_m:.4f} m)')
                path.append(result.q)
            spline = CubicSpline(knots, np.array(path), bc_type=((1, qd_now[arm]), (2, np.zeros(6))))
            self._check_dynamics(arm, spline, knots)
            _, _, stop = braking_profile(spline(knots[-1]), spline(knots[-1], 1), acceleration, [0.0])
            joint = self._outside_limits(arm, stop)
            if joint is not None:
                raise Rejected(f'{arm} cannot brake to a stop inside the joint_{joint + 1} limit after the chunk '
                               f'(stop at {stop[joint]:.3f} rad)')
            joints[arm] = spline(samples)
            widths[arm] = np.interp(samples, knots, np.concatenate([[w_command[arm]], chunk_widths]))
        openings = {arm: widest_opening(w_measured[arm], w_command[arm], widths[arm]) for arm in ARMS}
        check_geometry(self.model, self.limits, samples, joints, contacts=chunk.contacts, start=(now, q_now),
                       openings=openings)
        return JointTrajectory(chunk.request_id, chunk.chunk_id, _frozen(samples),
                               {arm: _frozen(value) for arm, value in joints.items()},
                               {arm: _frozen(value) for arm, value in widths.items()}, hold=False)

    def _check_dynamics(self, arm, spline, knots):
        model = self.model.arms[arm]
        grid = np.linspace(knots[0], knots[-1], max(2, int(np.ceil((knots[-1] - knots[0]) * self.limits.servo_hz
                                                                         * CHECK_OVERSAMPLE)) + 1))
        grid = np.union1d(grid, knots)
        for order, limit, label in ((0, None, 'joint limit'), (1, self.limits.max_joint_velocity_rad_s, 'velocity'),
                                    (2, self.limits.max_joint_acceleration_rad_s2, 'acceleration')):
            values = spline(grid, order)
            if order == 0:
                excess = np.maximum(model.lower - values, values - model.upper)
            else:
                excess = np.abs(values) - limit
            if (excess > 0).any():
                sample, joint = np.unravel_index(np.argmax(excess), excess.shape)
                raise Rejected(f'{arm} {label} exceeded on joint_{joint + 1} at t={grid[sample]:.3f} s '
                               f'(value {values[sample, joint]:.3f})')


def check_geometry(model, limits, samples, joints, *, contacts=(), start=None, openings=None):
    """Raise Rejected unless at every sample the arms keep apart, the links and the tool box keep off the
    table and each TCP stays above `tcp_min_height_m`. `joints[arm]` is (len(samples), dof).

    Where an arm breaks the tool box or TCP rule, its ContactWindow among `contacts` may still let the sample
    through (module docstring). The per-sample parts of that rule are checked with each sample; the spacing and
    speed parts afterwards, from `start` = (time, {arm: joints}), the measured state the samples continue from,
    which is required whenever `contacts` is not empty. `openings` {arm: m}: the widest each gripper can be open
    at these samples (widest_opening), required when limits.finger_opening_margin_m is set: the tool and finger
    boxes are then taken at that opening plus the margin, the same at every sample. Without the margin the boxes
    hold the fingers over their whole stroke and `openings` is not used.
    """
    table = limits.table_top_z_m
    floor = table + limits.table_clearance_m
    widths = _box_widths(limits, openings)
    declared = _declared_contacts(contacts)
    start = _measured_start(model, start, samples) if declared else None
    used = {arm: [] for arm in ARMS}
    for index, time in enumerate(samples):
        capsules = {arm: model.arms[arm].capsules(joints[arm][index]) for arm in ARMS}
        clearance, i, j = capsule_clearance(capsules['left'].arrays(), capsules['right'].arrays())
        if clearance < limits.arm_arm_clearance_m:
            raise Rejected(f'arm-arm clearance {clearance:.3f} m < {limits.arm_arm_clearance_m} m between '
                           f'{capsules["left"].names[i]} and {capsules["right"].names[j]} at t={time:.3f} s')
        for arm in ARMS:
            arm_model = model.arms[arm]
            q = joints[arm][index]
            names = capsules[arm].names
            lowest = lowest_point_z(capsules[arm].arrays())
            for k, name in enumerate(names):
                if name.split(':')[1] in TABLE_CHECKED_LINKS and lowest[k] < floor:
                    raise Rejected(f'{name} is {lowest[k] - table:.3f} m from the table at t={time:.3f} s')
            problem, tcp = _table_problem(arm_model, limits, q, time, widths[arm])
            if problem is None:
                continue
            refusal = _contact_refusal(arm_model, limits, declared.get(arm), q, tcp, widths[arm])
            if refusal:
                raise Rejected(f'{problem}; {refusal})')
            _check_contact_clearances(arm_model, limits, declared[arm], q, tcp, time, widths[arm])
            used[arm].append(index)
    for arm in declared:
        _check_contact_motion(model.arms[arm], limits, samples, joints[arm], start, used[arm], widths[arm])


def widest_opening(measured, commanded, targets=()):
    """The widest a position-controlled gripper can be open while it is sent width `targets` (m): it moves between
    where it is (`measured`, which what it grips may hold wider than its command), where it was last sent
    (`commanded`: it may still be on its way there) and the new targets. Its overshoot past them is
    finger_opening_margin_m's to cover."""
    try:
        values = np.concatenate([[float(measured), float(commanded)], np.ravel(np.asarray(targets, dtype=float))])
    except (TypeError, ValueError) as exc:
        raise Rejected(f'gripper widths must be numbers: {exc}') from exc
    if not np.isfinite(values).all():
        raise Rejected('gripper widths must be finite')
    return float(values.max())


def _box_widths(limits, openings):
    """{arm: the opening S0 takes the tool and finger boxes at, or None for the whole stroke}."""
    margin = limits.finger_opening_margin_m
    if margin is None:
        return {arm: None for arm in ARMS}
    widths = {}
    for arm in ARMS:
        try:
            value = float(openings[arm])
        except (KeyError, TypeError, ValueError) as exc:
            raise Rejected(f'S0 takes the fingers at each gripper\'s widest opening in the motion, and {arm}\'s is '
                           f'missing or not a number: {exc!r}') from exc
        if not (np.isfinite(value) and value >= 0):
            raise Rejected(f'{arm}: the widest opening must be a finite width >= 0, got {value}')
        widths[arm] = value + margin
    return widths


def _declared_contacts(contacts):
    declared = {}
    for contact in contacts:
        if not isinstance(contact, ContactWindow):
            raise Rejected('contacts must be ContactWindow declarations (contracts.actions)')
        try:  # rebuilt as the base class, so a subclass or an instance that skipped validation is checked too
            contact = ContactWindow(arm=contact.arm, kind=contact.kind, center_xy=contact.center_xy,
                                    fragment_id=contact.fragment_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise Rejected(f'invalid contact declaration: {exc}') from exc
        if contact.arm in declared:
            raise Rejected(f'one contact per arm: {contact.arm} has two')
        declared[contact.arm] = contact
    return declared


def _measured_start(model, start, samples):
    """The checked (time, {arm: joints}) of the measured state the samples continue from; the samples after it
    must be strictly increasing."""
    try:
        time, joints = start
        time = float('nan') if isinstance(time, bool) else float(time)
        joints = {arm: np.asarray(joints[arm], dtype=float) for arm in ARMS}
    except (KeyError, TypeError, ValueError) as exc:
        raise Rejected(f'a contact needs the measured start state (time, {{arm: joints}}) of both arms: {exc!r}') from exc
    if not np.isfinite(time) or not len(samples) or not time < samples[0]:
        raise Rejected('the measured start state needs a finite time before the first sample')
    if not (np.diff(np.asarray(samples, dtype=float)) > 0).all():
        raise Rejected('sample times next to a contact must be strictly increasing')
    for arm in ARMS:
        if joints[arm].shape != (model.arms[arm].dof,) or not np.isfinite(joints[arm]).all():
            raise Rejected(f'the measured start state of {arm} must be {model.arms[arm].dof} finite joint positions')
    return time, joints


def needs_contact(arm_model, limits, q, opening=None):
    """Whether the arm at joints q breaks S0's normal tool-box or TCP-height rule, so that it needs a contact window
    there: the boxes at `opening` (the widest the gripper is open, widest_opening) plus the margin when the limits
    take openings, else over the whole stroke."""
    margin = limits.finger_opening_margin_m
    if margin is not None and opening is None:
        raise ValueError('S0 takes its boxes at the opening: give the widest the gripper is open')
    return _table_problem(arm_model, limits, q, 0.0, None if margin is None else float(opening) + margin)[0] is not None


def _table_problem(arm_model, limits, q, time, width=None):
    """(Why q breaks the tool box or TCP-height rule, or None; the TCP pose.) `width`: the opening the boxes are
    taken at (None: the whole stroke)."""
    table, arm = limits.table_top_z_m, arm_model.name
    tcp = arm_model.tcp(q)
    tool_low = float(arm_model.tool_box_corners(q, width)[:, 2].min())
    if not tool_low >= table + limits.table_clearance_m:
        return (f'{arm}:tool box is {tool_low - table:.3f} m from the table at t={time:.3f} s '
                f'(needs {limits.table_clearance_m} m'), tcp
    height = tcp[2, 3] - table
    if not height >= limits.tcp_min_height_m:
        return f'{arm} TCP {height:.3f} m above the table at t={time:.3f} s (needs {limits.tcp_min_height_m} m', tcp
    return None, tcp


def _below(corners, floor):
    """The vertices of the parts of boxes below the plane z = floor, from their corners (8 a box, box after box, as
    ArmModel lists them): each box's corners below it and the points where its edges cross it (each part is convex,
    so the farthest of these from any centre is the farthest point of them all)."""
    parts = []
    for box in np.asarray(corners, dtype=float).reshape(-1, 8, 3):
        low = box[:, 2] < floor
        parts.append(box[low])
        parts.extend(box[i] + (floor - box[i, 2]) / (box[j, 2] - box[i, 2]) * (box[j] - box[i])
                     for i, j in BOX_EDGES if low[i] != low[j])
    return np.vstack(parts)


def _contact_refusal(arm_model, limits, contact, q, tcp, width=None):
    """None when `contact` may let this sample through, otherwise why it cannot. The TCP and all of the finger boxes
    below the normal floor must lie within the contact radius of the centre (x, y)."""
    arm = arm_model.name
    if contact is None:
        return f'no contact declared for {arm}'
    if not limits.contact_enabled:
        return f'a {contact.kind} contact is declared but the contact phase is not configured'
    if not arm_model.has_tool_parts:
        return f'a {contact.kind} contact needs the finger geometry of the {arm} gripper'
    below = _below(arm_model.finger_box_corners(q, width), limits.table_top_z_m + limits.table_clearance_m)
    points = np.vstack([tcp[:3, 3], below])
    reach = float(np.hypot(points[:, 0] - contact.center_xy[0], points[:, 1] - contact.center_xy[1]).max())
    if not reach <= limits.contact_region_radius_m:
        return (f'the TCP or fingers reach {reach:.3f} m from the {contact.kind} contact centre, outside its '
                f'{limits.contact_region_radius_m} m radius')
    return None


def _check_contact_clearances(arm_model, limits, contact, q, tcp, time, width=None):
    """In a contact the gripper body keeps the normal clearance; the fingers and the TCP the contact clearance."""
    table, arm = limits.table_top_z_m, arm_model.name
    body_low = float(arm_model.body_box_corners(q)[:, 2].min())
    if not body_low >= table + limits.table_clearance_m:
        raise Rejected(f'{arm}:gripper body is {body_low - table:.3f} m from the table at t={time:.3f} s (needs '
                       f'{limits.table_clearance_m} m, also in a {contact.kind} contact)')
    finger_low = float(arm_model.finger_box_corners(q, width)[:, 2].min())
    if not finger_low >= table + limits.contact_table_clearance_m:
        raise Rejected(f'{arm}:fingers are {finger_low - table:.4f} m from the table at t={time:.3f} s (needs '
                       f'{limits.contact_table_clearance_m} m in a {contact.kind} contact)')
    height = tcp[2, 3] - table
    if not height >= limits.contact_table_clearance_m:
        raise Rejected(f'{arm} TCP {height:.3f} m above the table at t={time:.3f} s (needs '
                       f'{limits.contact_table_clearance_m} m in a {contact.kind} contact)')


def _check_contact_motion(arm_model, limits, samples, path, start, used, width=None):
    """The rules over time for one arm's contact.

    Every servo interval next to a sample that used the contact (`used`), and the first interval when the
    measured start itself needs it, must be at most one servo period long and keep every finger-box corner
    slower than contact_max_speed_m_s; interval k ends at sample k and starts at sample k - 1, or at the start
    for k = 0.
    """
    if not limits.contact_enabled or not arm_model.has_tool_parts:
        return  # nothing was relaxed: every sample that needed this contact was rejected
    arm = arm_model.name
    start_time, start_joints = start
    start_q = start_joints[arm]
    starts_in_contact = _table_problem(arm_model, limits, start_q, start_time, width)[0] is not None
    near = {*used, *(i + 1 for i in used if i + 1 < len(samples)), *([0] if starts_in_contact else [])}
    if not near:
        return
    times = np.concatenate([[start_time], np.asarray(samples, dtype=float)])
    corners = np.array([arm_model.finger_box_corners(q, width) for q in (start_q, *path)])
    steps = np.diff(times)
    speeds = np.linalg.norm(np.diff(corners, axis=0), axis=2).max(axis=1) / steps
    period = 1.0 / limits.servo_hz
    for k in sorted(near):
        if not steps[k] <= CONTACT_SPACING_SLACK * period:
            raise Rejected(f'{arm}: samples next to a contact must be at most one servo period ({period:.4f} s) '
                           f'apart, got {steps[k]:.4f} s before t={samples[k]:.3f} s')
        if not speeds[k] <= limits.contact_max_speed_m_s:
            raise Rejected(f'{arm}:fingers move at {speeds[k]:.3f} m/s near the table at t={samples[k]:.3f} s '
                           f'(limit {limits.contact_max_speed_m_s} m/s in a contact)')


def check_joint_path(model, limits, *, now, q_now, qd_now, qd_window_s, times, joints, contacts=(), openings=None):
    """Raise Rejected unless joint targets sampled at `times` keep both arms inside the joint position,
    velocity and acceleration limits, can brake to a stop inside the joint limits after the last
    target, and pass check_geometry.

    For callers that command joints directly (the ASPIRE adapter) instead of sending ActionChunks. The
    targets are read as reached at their times. At every knot the acceleration is the change of segment
    velocity over the mean of the two spans it joins; at the first, the span before `now` is the one the
    measured `qd_now` averages over, `qd_window_s` (RobotSnapshot.velocity_window_s: the backends report
    each control period's velocity, so a path's first step reads as every later one, and a motion cut into
    several paths reads as it does whole). An instantaneous velocity (window 0) leaves half the first
    segment, 2 (v_1 - qd_now) / dt_1: constant acceleration from `now`. `q_now`, `qd_now`, the clock and the
    window must be finite (the window non-negative). Braking from the last segment's velocity uses S0's
    synchronized profile. Nothing is clipped or re-timed. `contacts` are ContactWindows, applied by
    check_geometry as for a chunk; targets next to a contact must then be at most one servo period apart,
    because nothing between targets is checked. Away from a contact, long intervals are not checked in
    between either: a caller whose targets are further apart than a servo period must check the path
    between them itself, as the ASPIRE adapter does. Unlike S0Adapter, this function keeps no memory, so
    a caller that issues several paths for one request must keep its contacts fixed itself. `openings` {arm: m}:
    each gripper's widest opening over the path (widest_opening), which check_geometry needs when
    limits.finger_opening_margin_m is set.
    """
    try:
        now = float('nan') if isinstance(now, bool) else float(now)
    except (TypeError, ValueError):
        now = float('nan')
    if not np.isfinite(now):
        raise Rejected('the measured clock is not finite')
    if (isinstance(qd_window_s, bool) or not isinstance(qd_window_s, numbers.Real)
            or not (np.isfinite(qd_window_s) and qd_window_s >= 0)):
        raise Rejected('the measured joint velocity must say what it means: qd_window_s, the time it averages over '
                       '(0 for an instantaneous velocity), finite and non-negative')
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or times[0] <= now or (np.diff(times) <= 0).any():
        raise Rejected('sample times must be finite, strictly increasing and after now')
    steps = np.diff(np.concatenate([[now], times]))
    samples = {}
    for arm in ARMS:
        arm_model = model.arms[arm]
        if arm not in q_now or arm not in qd_now or arm not in joints:
            raise Rejected(f'{arm}: measured state and joint targets are required for both arms')
        q0, qd0 = np.asarray(q_now[arm], dtype=float), np.asarray(qd_now[arm], dtype=float)
        if q0.shape != (arm_model.dof,) or qd0.shape != (arm_model.dof,) or not np.isfinite([*q0, *qd0]).all():
            raise Rejected(f'{arm}: the measured joint state must be {arm_model.dof} finite positions and velocities')
        path = np.asarray(joints[arm], dtype=float)
        if path.shape != (len(times), arm_model.dof) or not np.isfinite(path).all():
            raise Rejected(f'{arm}: expected {len(times)} x {arm_model.dof} finite joint targets')
        excess = np.maximum(arm_model.lower - path, path - arm_model.upper)
        if (excess > 0).any():
            sample, joint = np.unravel_index(np.argmax(excess), excess.shape)
            raise Rejected(f'{arm} joint limit exceeded on joint_{joint + 1} at t={times[sample]:.3f} s '
                           f'(value {path[sample, joint]:.3f})')
        velocity = np.diff(np.vstack([q0, path]), axis=0) / steps[:, None]
        spacing = np.concatenate([[(float(qd_window_s) + steps[0]) / 2], (steps[:-1] + steps[1:]) / 2])
        acceleration = np.diff(np.vstack([qd0, velocity]), axis=0) / spacing[:, None]
        for values, limit, label in ((velocity, limits.max_joint_velocity_rad_s, 'velocity'),
                                     (acceleration, limits.max_joint_acceleration_rad_s2, 'acceleration')):
            excess = np.abs(values) - limit
            if (excess > 0).any():
                sample, joint = np.unravel_index(np.argmax(excess), excess.shape)
                raise Rejected(f'{arm} {label} exceeded on joint_{joint + 1} at t={times[sample]:.3f} s '
                               f'(value {values[sample, joint]:.3f})')
        _, _, stop = braking_profile(path[-1], velocity[-1], limits.max_joint_acceleration_rad_s2, [0.0])
        beyond = np.maximum(arm_model.lower - stop, stop - arm_model.upper)
        if (beyond > 0).any():
            joint = int(np.argmax(beyond))
            raise Rejected(f'{arm} cannot brake to a stop inside the joint_{joint + 1} limit after the last target '
                           f'(stop at {stop[joint]:.3f} rad)')
        samples[arm] = path
    check_geometry(model, limits, times, samples, contacts=contacts,
                   start=(now, {arm: np.asarray(q_now[arm], dtype=float) for arm in ARMS}), openings=openings)
