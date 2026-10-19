"""DexAgent's delivery executed on the hardware every method shares (README "Port to the shared hardware", step 5).

A delivery (DexAgent's accepted trajectory.npz) gives, per arm, the commanded joint reference (`reference`: the
runtime's targets, not its recorded states) and the jaw targets at DexAgent's frame rate, with a phase label per
frame, and the recorded state it starts from (`qpos` at frame 0: the first target may already be on its way).
DexAgent's runtime ramps every command linearly from one frame to the next (rigid_robot.servo_rollout), so the
commands here are that ramp sampled at S0's servo rate. A jaw target is DexAgent's closing coordinate; the shared
gripper's finger joint is the opening (travel_m minus it), and its width command the pad gap of the two.

Contact windows (contracts.actions.ContactWindow) are declared the way an ASPIRE program declares them: just before an
arm's commands enter S0's contact band (the tool box below the table clearance or the TCP below its minimum height;
s0.adapter), kept while it stays in the band, cleared once it is out. Each arm's phase labels say which fragment it
works on and how: approaching, closing on, gripping or lifting a fragment opens a window on the fragment where it is
observed; carrying, setting it down, letting go of it or settling after it opens one on its goal when the fragment is
within S0's contact radius of its goal (setting it down there), else on the fragment (setting it down where it is,
which S0 treats as a grasp contact); any other phase (a retreat, a return home) keeps the arm's last fragment. The
fixture-held anchor is never a contact. The commands run through S0 as ASPIRE's do, in chunks each checked as one path
from the measured state (s0.adapter.check_joint_path, and the width range) before any of it is stepped. S0 reads a
chunk's first segment from the measured velocity, as a constant acceleration twice the velocity change over one period,
so a chunk starts where the commanded velocity changes least (on DexAgent's 30 fps ramp, halfway between frames, where
nothing changes): one starts at, or one command before, each command where an arm enters the band, whichever changes
less, and its window is declared then, from the label of the arm's first command in the band, on what is observed;
the others are at most CHUNK_S long and end in their second half. A chunk S0 rejects changes nothing: S0
rejects, never re-times, and the execution ends where it stands. Then, rejected or not, the ending every method gets:
both grippers fully open and the arms held for the task's settle time with the windows they had, checked the same
way, and score_assembly on the backend's privileged poses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ...contracts.actions import ARMS, ContactWindow
from ...s0.adapter import Rejected, check_joint_path
from ..scoring import score_assembly

CHUNK_S = 1.0
# A held arm is not exactly at its target (Isaac: 1.2e-4 rad off home after its settle, measured 2026-10-19); a
# delivery planned from another start is off by far more. 1e-3 rad is 1 mm at a metre.
START_TOLERANCE_RAD = 1e-3
GRASP_WORDS = ('reach for', 'reach to hold', 'approach', 'descend on', 'close on', 'grip', 'hold', 'lift', 'regrasp',
               'realign on')
PLACE_WORDS = ('carry', 'set down', 'let go of', 'let go', 'settle after release of', 'ease off', 'press')
_LABEL_TAIL = re.compile(r'\s*#\d+(\.\d+)?\s*$')


def split_label(label):
    """{side: phase} of a frame label: '<side>: <phase> #k' or '<side>: <phase> | <side>: <phase> #k'."""
    out = {}
    for part in _LABEL_TAIL.sub('', str(label)).split(' | '):
        side, sep, phase = part.partition(':')
        if sep and side.strip() in ARMS:
            out[side.strip()] = phase.strip()
    return out


def phase_contact(phase, objects):
    """(kind, fragment_id) of a phase on one of `objects` ({DexAgent body name: fragment id}), or None when the phase
    names none of them or is neither a grasp nor a place (the arm keeps its window)."""
    found = [(name, fragment) for name, fragment in objects.items() if re.search(rf'\b{re.escape(name)}\b', phase)]
    if not found:
        return None
    fragment = max(found, key=lambda item: len(item[0]))[1]
    words = phase.lower()
    if any(words.startswith(word) for word in PLACE_WORDS):
        return 'place', fragment
    if any(words.startswith(word) for word in GRASP_WORDS):
        return 'grasp', fragment
    return None


@dataclass(frozen=True)
class Delivery:
    time_s: np.ndarray  # (T,) frame times, from 0
    joints: Mapping[str, np.ndarray]  # arm -> (T, 6) commanded joints
    jaws: Mapping[str, np.ndarray]  # arm -> (T, 2) DexAgent's jaw targets (thumb, index)
    labels: tuple
    start: Mapping[str, np.ndarray] | None = None  # arm -> the recorded joints at frame 0 (default: the first target)


def read_delivery(trajectory):
    """The commanded reference of an accepted trajectory.npz (path or loaded mapping)."""
    data = np.load(trajectory, allow_pickle=False) if not isinstance(trajectory, Mapping) else trajectory
    names = [str(name) for name in data['joint_names']]
    reference = np.asarray(data['reference'], dtype=float)
    time_s = np.asarray(data['time_s'], dtype=float)
    labels = tuple(str(label) for label in data['labels'])
    if reference.shape[0] != len(time_s) or len(labels) != len(time_s) or not np.isfinite(reference).all():
        raise ValueError('the trajectory\'s reference, times and labels disagree or are not finite')
    if (np.diff(time_s) <= 0).any() or abs(time_s[0]) > 1e-9:
        raise ValueError('frame times must start at 0 and increase')

    def columns(keys):
        missing = [key for key in keys if key not in names]
        if missing:
            raise ValueError(f'the trajectory has no columns {missing}')
        return reference[:, [names.index(key) for key in keys]]

    joints = {arm: columns([f'robot_{arm}_arm_dof_joint{i}' for i in range(1, 7)]) for arm in ARMS}
    jaws = {arm: columns([f'robot_{arm}_{arm}_thumb_FE', f'robot_{arm}_{arm}_index_FE']) for arm in ARMS}
    state = np.asarray(data['qpos'], dtype=float)[0]
    if state.shape != reference.shape[1:] or not np.isfinite(state).all():
        raise ValueError('the trajectory\'s recorded state at frame 0 is not one finite row of its columns')
    start = {arm: state[[names.index(f'robot_{arm}_arm_dof_joint{i}') for i in range(1, 7)]] for arm in ARMS}
    return Delivery(time_s, joints, jaws, labels, start)


@dataclass(frozen=True)
class Commands:
    period_s: float
    start: Mapping[str, np.ndarray]  # arm -> the joints the delivery starts from (its frame 0)
    time_s: np.ndarray  # (N,) sample times after the start, one servo period apart
    joints: Mapping[str, np.ndarray]  # arm -> (N, 6)
    widths: Mapping[str, np.ndarray]  # arm -> (N,)
    contacts: Mapping[str, tuple]  # arm -> per sample (kind, fragment_id) or None


def jaw_width(gripper, jaws):
    """The pad gap of DexAgent's jaw targets (thumb, index): Isaac's finger joints are travel_m minus them."""
    jaws = np.asarray(jaws, dtype=float)
    closing = gripper.travel_m - gripper.contact_joint_m
    if not ((jaws >= -1e-9) & (jaws <= closing + 1e-9)).all():
        raise ValueError(f'jaw targets outside DexAgent\'s range [0, {closing:.5f}] m')
    joints = gripper.travel_m - np.clip(jaws, 0.0, closing)
    return np.maximum(0.0, joints[..., 0] + joints[..., 1] + gripper.closed_gap_m)


def servo_commands(delivery, gripper, servo_hz, objects):
    """The delivery's command ramp sampled every servo period up to its last frame, with each arm's contact."""
    period = 1.0 / float(servo_hz)
    times = np.arange(1, int(np.floor(delivery.time_s[-1] / period + 1e-9)) + 1) * period
    joints = {arm: np.column_stack([np.interp(times, delivery.time_s, delivery.joints[arm][:, k]) for k in range(6)])
              for arm in ARMS}
    widths = {arm: np.interp(times, delivery.time_s, jaw_width(gripper, delivery.jaws[arm])) for arm in ARMS}
    frame = np.clip(np.searchsorted(delivery.time_s, times, side='left'), 0, len(delivery.time_s) - 1)
    contacts = {}
    for arm in ARMS:
        current, per_frame = None, []
        for label in delivery.labels:
            phase = split_label(label).get(arm)
            found = None if phase is None else phase_contact(phase, objects)
            current = found if found is not None else current
            per_frame.append(current)
        contacts[arm] = tuple(per_frame[i] for i in frame)
    start = delivery.start if delivery.start is not None else {arm: delivery.joints[arm][0] for arm in ARMS}
    return Commands(period, {arm: np.array(start[arm], dtype=float) for arm in ARMS}, times, joints, widths, contacts)


@dataclass
class Execution:
    status: str  # 'completed' | 'rejected' | 'refused'
    steps: int
    reason: str | None = None
    windows: list | None = None  # the windows declared, in order: (time, arm, kind, fragment, centre)
    score: object | None = None
    settled: bool = False
    settle_error: str | None = None


def band_samples(model, limits, commands):
    """{arm: (N,) bool}: which of its commands are in S0's contact band; and whether its start is."""
    inside = {arm: np.array([in_band(model, limits, arm, q) for q in commands.joints[arm]], dtype=bool) for arm in ARMS}
    return inside, {arm: in_band(model, limits, arm, commands.start[arm]) for arm in ARMS}


def band_entries(model, limits, commands, samples=None):
    """Samples where an arm's commands enter S0's contact band (in it, the previous command or the start not)."""
    inside, start = samples or band_samples(model, limits, commands)
    entries = set()
    for arm in ARMS:
        before = np.concatenate([[start[arm]], inside[arm][:-1]])
        entries.update(int(k) for k in np.flatnonzero(inside[arm] & ~before))
    return sorted(entries)


def velocity_change(commands):
    """(N,) largest change of commanded joint velocity, per period, at the start of a chunk beginning at each sample:
    between the segment into the last stepped command and the one out of it (0 for the first sample)."""
    change = np.zeros(len(commands.time_s))
    for arm in ARMS:
        path = np.vstack([commands.start[arm], commands.joints[arm]])
        second = np.abs(path[2:] - 2.0 * path[1:-1] + path[:-2]).max(axis=1)
        change[1:] = np.maximum(change[1:], second)
    return change


def _chunks(commands, chunk_s, entries=()):
    """(start, stop) sample ranges: a new one at each of `entries` or the sample before it, whichever the commanded
    velocity changes less at; the others at most chunk_s long and ending, in their second half, where it changes least
    (the latest such sample)."""
    n, size = len(commands.time_s), max(1, int(round(chunk_s / commands.period_s)))
    change = velocity_change(commands)
    forced = sorted({k - 1 if k > 1 and change[k - 1] < change[k] else k for k in entries if 0 < k < n} | {n})
    start = 0
    while start < n:
        limit = next(k for k in forced if k > start)
        if limit - start <= size:
            stop = limit
        else:
            candidates = np.arange(start + (size + 1) // 2, start + size + 1)
            stop = int(candidates[np.flatnonzero(change[candidates] == change[candidates].min())[-1]])
        yield start, stop
        start = stop


def in_band(model, limits, arm, joints):
    """Whether S0's normal table rules fail at any of `joints` (N, 6) of `arm`: its contact band."""
    arm_model, table = model.arms[arm], limits.table_top_z_m
    return any(arm_model.tool_box_corners(q)[:, 2].min() < table + limits.table_clearance_m
               or arm_model.tcp(q)[2, 3] < table + limits.tcp_min_height_m for q in np.atleast_2d(joints))


def contact_window(backend, task, limits, arm, kind, fragment):
    """The window an arm declares for a labelled contact (module docstring)."""
    observed = np.asarray(backend.observed_fragment_poses()[fragment].world_T_fragment, dtype=float)[:2, 3]
    if kind == 'place':
        goal = np.asarray(task.fragment(fragment).goal_world_T_fragment, dtype=float)[:2, 3]
        if np.linalg.norm(goal - observed) <= limits.contact_region_radius_m:
            return ContactWindow(arm=arm, kind='place', center_xy=tuple(goal), fragment_id=fragment)
    return ContactWindow(arm=arm, kind='grasp', center_xy=tuple(observed), fragment_id=fragment)


def _check_widths(limits, widths):
    for arm in ARMS:
        low, high = limits.gripper_width_range_m[arm]
        values = np.asarray(widths[arm], dtype=float)
        if not ((values >= low) & (values <= high)).all():
            raise Rejected(f'{arm} width outside [{low}, {high}] m')


def execute(backend, task, model, limits, commands, chunk_s=CHUNK_S):
    """Run `commands` on a reset backend through S0, then release, settle and score (module docstring)."""
    state = backend.robot_state()
    for arm in ARMS:
        gap = float(np.max(np.abs(np.asarray(state.joints[arm], dtype=float) - commands.start[arm])))
        if gap > START_TOLERANCE_RAD:
            return Execution('refused', 0, f'the delivery does not start where the {arm} arm stands ({gap:.2e} rad)')
    windows, declared, steps, rejection = [], {}, 0, None
    targets = {arm: np.asarray(state.joints[arm], dtype=float).copy() for arm in ARMS}
    samples = band_samples(model, limits, commands)
    for start, stop in _chunks(commands, chunk_s, band_entries(model, limits, commands, samples)):
        state = backend.robot_state()
        path = {arm: commands.joints[arm][start:stop] for arm in ARMS}
        pending, new = dict(declared), []  # the chunk's windows, kept only if S0 accepts it
        for arm in ARMS:
            first = np.flatnonzero(samples[0][arm][start:stop])
            if not len(first) and not in_band(model, limits, arm, state.joints[arm]):
                pending.pop(arm, None)  # out of the band: no window needed
                continue
            label = commands.contacts[arm][start + int(first[0]) if len(first) else start]
            if arm in pending or label is None:
                continue  # a window stays while the arm is in the band; without a labelled contact S0 decides
            window = pending[arm] = contact_window(backend, task, limits, arm, *label)
            new.append((float(state.time_s), arm, window.kind, window.fragment_id, window.center_xy))
        times = state.time_s + (commands.time_s[start:stop] - (commands.time_s[start - 1] if start else 0.0))
        widths = {arm: commands.widths[arm][start:stop] for arm in ARMS}
        try:
            _check_widths(limits, widths)
            check_joint_path(model, limits, now=state.time_s, q_now=state.joints, qd_now=state.joint_velocities,
                             times=times, joints=path, contacts=tuple(pending.values()))
        except Rejected as exc:
            rejection = f'S0 rejected the delivery at t={state.time_s:.3f} s: {exc}'
            break
        declared = pending
        windows.extend(new)
        for k in range(stop - start):
            targets = {arm: path[arm][k] for arm in ARMS}
            backend.step(targets, {arm: float(widths[arm][k]) for arm in ARMS})
            steps += 1
    settled, settle_error = release_and_settle(backend, task, model, limits, targets, declared)
    score = score_assembly(task, backend.privileged_fragment_poses(), settled=settled)
    return Execution('rejected' if rejection else 'completed', steps, rejection, windows, score, settled, settle_error)


def release_and_settle(backend, task, model, limits, targets, contacts=None):
    """Both grippers fully open and the arms held at their last targets for the task's settle time (at least one
    period), checked as one path with the arms' windows; (settled, why not)."""
    state = backend.robot_state()
    period = 1.0 / limits.servo_hz
    periods = max(1, int(np.ceil(task.settle_s / period - 1e-9)))
    hold = {arm: np.repeat(np.asarray(targets[arm], dtype=float)[None], periods, axis=0) for arm in ARMS}
    widths = {arm: limits.gripper_width_range_m[arm][1] for arm in ARMS}
    times = state.time_s + period * np.arange(1, periods + 1)
    try:
        check_joint_path(model, limits, now=state.time_s, q_now=state.joints, qd_now=state.joint_velocities,
                         times=times, joints=hold, contacts=tuple((contacts or {}).values()))
    except Rejected as exc:
        return False, f'S0 rejected the release-and-settle hold: {exc}'
    for k in range(periods):
        backend.step({arm: hold[arm][k] for arm in ARMS}, widths)
    return True, None
