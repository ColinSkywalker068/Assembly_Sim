"""ASPIRE EnvironmentAdapter over an AssemblyBackend: the dual CRX workcell as the 6-DoF "fanuc_crx" embodiment.

Frames: ASPIRE's base frame "robot_base" is the task world frame (metres, z up); every pose a program reads or sends
is in it, quaternions wxyz with w >= 0. Arms are "left" and "right" (contracts.actions.ARMS); "primary" and
"secondary", the defaults of ASPIRE's shared signatures, resolve to left and right through `resolve_arm`. Gripper
positions are normalized linearly over S0's configured width range: 0 is the minimum width (closed), 1 the maximum.

A program receives what our S2 receives (`get_task_context().metadata`): fragment ids, the anchor, goal poses with
their source, fragment-frame bounds, the placement tolerances and the fragment poses the backend reports as
observable, with their source, plus RGB-D frames and the robot state. Privileged simulator poses are read only by
`check_success`, after the program has finished.

Every joint command passes S0's check_joint_path (joint position, velocity and acceleration limits from the measured
state, braking inside the joint limits, arm-arm clearance, table clearance, TCP height) before the backend steps, with
the geometry also checked between control periods at S0's servo rate. A rejected command moves nothing and returns
ASPIRE's typed failure (`safety_interlock`); nothing is clipped or re-timed. A trajectory, like the interpolated move of
the `go_home` hook, is checked as a whole, including one hold period at its last waypoint that ends the commanded
motion at rest, before its first step.

`reset(task_ref, seed)` takes the base task (or the path of its JSON record), checks it against its layout under the
canonical protocol (baselines.task.verify_task) and draws the seed's initial condition. The adapter's mode guards the
seed before anything else happens: 'practice' runs practice and selection seeds, 'evaluation' runs only held-out
seeds, and the configuration must carry the protocol of configs/jing.json.

Contacts (S0's contact phase, the same rule our S1 chunks follow): an arm may work closer to the table than S0's
normal clearances only inside a declared window, with its fingers only and slowly. A program declares one per arm with
the extension tool `fanuc_crx.declare_contact(kind, fragment_id, arm=...)`: 'grasp' centres the window on the
fragment's observed pose, 'place' on its goal, so a program names the fragment but never the coordinates.
`fanuc_crx.clear_contact(arm=...)` removes it. Neither is allowed while the arm is inside the contact band (lift
first), which keeps a window where it was declared, as S0 keeps a request's window. While any window is declared the
joint speed bounds ASPIRE's planner reads (`action_bounds`) shrink so that no finger corner exceeds
CONTACT_SPEED_FRACTION of S0's contact speed.

`check_success` ends the trial the same way for every method (baselines.backend): both grippers open fully and the
arms hold their last targets for the task's settle time, checked like any command but outside the program's step
budget; only then are the privileged poses scored. A program that ends holding a fragment is judged where that
fragment comes to rest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from cap_harness.contracts import (ArmCommand, CameraObservation, ExecutionResult, Observation, Pose, RobotAction,
                                   RobotState, StepResult, SynchronizedTrajectory, TaskContext, Trajectory)
from cap_harness.errors import ApiError, ErrorCode

from ...contracts.actions import ARMS, ContactWindow
from ...s0.adapter import Rejected, check_geometry, check_joint_path, needs_contact, reach_bounds, widest_opening
from ..backend import ProtectiveStop
from ..scoring import score_assembly
from ..task import (AssemblyTask, canonical_split, configured_initial_condition, pose_to_wxyz, require_canonical,
                    verify_task)

EMBODIMENT = 'fanuc_crx'
BASE_FRAME = 'robot_base'
ARM_ALIASES = MappingProxyType({'primary': 'left', 'secondary': 'right'})
MODE_SPLITS = MappingProxyType({'practice': ('practice', 'selection'), 'evaluation': ('heldout',)})
BENCHMARK = 'dual_arm_crag_skull'
SUITE = 'fragment_assembly'
FAMILY = 'bimanual_assembly'
DEFAULT_CAMERA = 'agent'
# ASPIRE's joint interpolation starts and stops within one control period. S0 reads that step from rest over half the
# period plus the time the measured velocity averages over (RobotSnapshot.velocity_window_s; the backends report each
# period's velocity, so a_max dt; an instantaneous velocity would allow a_max dt / 2), so its speed bound is this
# fraction of min(v_max, a_max (window + dt) / 2) and its own trajectories pass the S0 checks with a margin.
INTERPOLATION_FRACTION = 0.9
CONTACT_SPEED_FRACTION = 0.8
CONTACT_KINDS = ('grasp', 'place')
TRAJECTORY_TOLERANCE_RAD = 0.01
GRIPPER_TOLERANCE_M = 1e-4
# A measured opening this fraction of the stroke outside the configured range still reads as 0 or 1; further out,
# the configured range is wrong and the adapter raises.
MEASURED_WIDTH_SLACK = 0.05


def guard_seed(seed, mode, config):
    """The split of `seed` if `mode` may run it, else ValueError: 'practice' runs practice and selection seeds,
    'evaluation' only held-out seeds. `config` must carry the canonical protocol (configs/jing.json), so a held-out
    seed cannot be relabelled, or judged differently, by passing another configuration."""
    if mode not in MODE_SPLITS:
        raise ValueError(f'mode must be one of {tuple(MODE_SPLITS)}')
    require_canonical(config)
    split = canonical_split(seed)
    if split not in MODE_SPLITS[mode]:
        raise ValueError(f'seed {seed} is a {split} seed; {mode} mode runs only {"/".join(MODE_SPLITS[mode])} seeds')
    return split


def load_task(task_ref):
    """The base AssemblyTask from an AssemblyTask or the path of its JSON record (AssemblyTask.to_record())."""
    if isinstance(task_ref, AssemblyTask):
        task = task_ref
    elif isinstance(task_ref, (str, Path)):
        task = AssemblyTask.from_record(json.loads(Path(task_ref).read_text()))
    else:
        raise ValueError('task_ref must be an AssemblyTask or the path of its JSON record')
    if task.seed is not None:
        raise ValueError('task_ref must be the base task (seed None); the adapter draws the initial condition')
    return task


def as_pose(world_T):
    """ASPIRE Pose in robot_base of a rigid 4x4, rigidity judged with our tolerance (baselines.task.pose_to_wxyz)."""
    values = pose_to_wxyz(world_T)
    return Pose(position=values[:3], quaternion_wxyz=values[3:], frame=BASE_FRAME)


def pose_mapping(world_T):
    """ASPIRE's mapping pose (position, quaternion_wxyz, frame) of a 4x4 in robot_base, accepted by move_to_pose."""
    values = pose_to_wxyz(world_T)
    return {'position': tuple(values[:3]), 'quaternion_wxyz': tuple(values[3:]), 'frame': BASE_FRAME}


def _frozen(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_frozen(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _failure(code, message, **details):
    return ApiError(code=code, message=message, details=details)


@dataclass(frozen=True)
class Executed:
    """Outcome of a checked command sequence: periods stepped, last observation and the typed error if any."""

    ok: bool
    steps: int
    observation: Observation | None
    error: ApiError | None = None


class SkullAssemblyAdapter:
    """ASPIRE EnvironmentAdapter for the skull task (see module docstring)."""

    def __init__(self, backend, model, limits, *, config, mode, run_observer=None, gripper_max_steps=30):
        if mode not in MODE_SPLITS:
            raise ValueError(f'mode must be one of {tuple(MODE_SPLITS)}')
        missing = [arm for arm in ARMS if not model.arms[arm].has_tool_box]
        if missing:
            raise ValueError(f'the S0 checks need the tool box of every arm; missing: {missing}')
        if isinstance(gripper_max_steps, bool) or not isinstance(gripper_max_steps, int) or gripper_max_steps < 1:
            raise ValueError('gripper_max_steps must be a positive integer')
        require_canonical(config)
        self.backend, self.model, self.limits, self.config, self.mode = backend, model, limits, config, mode
        self.gripper_max_steps = gripper_max_steps
        self.control_frequency = float(backend.control_hz)
        if not np.isfinite(self.control_frequency) or self.control_frequency <= 0:
            raise ValueError('backend control_hz must be positive and finite')
        self.control_period_s = 1.0 / self.control_frequency
        self.joint_names = tuple(model.data['arm']['dof_names'])
        self._run_observer = run_observer
        self.task = None
        self.last_score = None
        self.settle_error = None
        self.rejected_commands = 0
        self._contacts = {}
        self._closed = False
        self.protective_stop = None  # backend.ProtectiveStop that ended the trial, if one did

    @property
    def embodiment(self):
        return EMBODIMENT

    @property
    def action_bounds(self):
        """Joint speed bounds (rad/s) ASPIRE's joint interpolation divides by control_frequency to size its steps; while
        a contact is declared, low enough that no finger corner of an arm with one exceeds CONTACT_SPEED_FRACTION of
        S0's contact speed (both arms slow down: ASPIRE reads one bound for every arm)."""
        span = (float(self.backend.velocity_window_s) + self.control_period_s) / 2
        bound = INTERPOLATION_FRACTION * np.minimum(
            self.limits.max_joint_velocity_rad_s, self.limits.max_joint_acceleration_rad_s2 * span)
        if self._contacts:
            lever = max(self.contact_lever_m(arm) for arm in self._contacts)
            bound = np.minimum(bound, CONTACT_SPEED_FRACTION * self.limits.contact_max_speed_m_s / lever)
        return -bound, bound.copy()

    @property
    def contacts(self):
        """arm -> the ContactWindow it has declared."""
        return MappingProxyType(dict(self._contacts))

    def contact_lever_m(self, arm):
        """Sum over the joints of the distance from each joint axis to the arm's farthest finger corner, at the
        measured joints: a finger corner moves at most this times the largest joint speed."""
        q = np.asarray(self.backend.robot_state().joints[arm], dtype=float)
        arm_model = self.model.arms[arm]
        _, axes, origins = arm_model.frames(q)
        corners = arm_model.finger_box_corners(q)
        lever = 0.0
        for axis, origin in zip(axes, origins):
            offsets = corners - origin
            lever += float(np.linalg.norm(offsets - np.outer(offsets @ axis, axis), axis=1).max())
        return lever

    def _in_contact_band(self, arm):
        """Whether the arm's measured pose already needs a contact window (S0's normal table rules fail for it, with
        S0's boxes at the gripper's opening when its limits take openings)."""
        snapshot = self.backend.robot_state()
        q = np.asarray(snapshot.joints[arm], dtype=float)
        opening = widest_opening(snapshot.spans[arm], self._widths[arm])
        return needs_contact(self.model.arms[arm], self.limits, q, opening)

    def declare_contact(self, kind, fragment_id, *, arm='primary'):
        """Declare where `arm` may work close to the table: 'grasp' a fragment (the window is centred on its observed
        pose) or 'place' it (centred on its goal). Refused inside the contact band."""
        task = self._require_task()
        arm = self.resolve_arm(arm)
        if arm not in ARMS:
            return ExecutionResult(ok=False, steps_executed=0,
                                   error=_failure(ErrorCode.NOT_FOUND, f'unknown arm {arm!r}; arms are {ARMS}'))
        if not self.limits.contact_enabled:
            return ExecutionResult(ok=False, steps_executed=0, error=_failure(
                ErrorCode.UNSUPPORTED, "S0's contact phase is not configured; nothing may come close to the table"))
        if kind not in CONTACT_KINDS:
            return ExecutionResult(ok=False, steps_executed=0, error=_failure(
                ErrorCode.INVALID_REQUEST, f'contact kind must be one of {CONTACT_KINDS}'))
        if fragment_id not in task.fragment_ids:
            return ExecutionResult(ok=False, steps_executed=0,
                                   error=_failure(ErrorCode.NOT_FOUND, f'unknown fragment {fragment_id!r}'))
        if fragment_id == task.anchor_id:
            return ExecutionResult(ok=False, steps_executed=0, error=_failure(
                ErrorCode.INVALID_REQUEST, f'{fragment_id} is the anchor, held by its fixture: not grasped or placed'))
        pose = (self.backend.observed_fragment_poses()[fragment_id].world_T_fragment if kind == 'grasp'
                else task.fragment(fragment_id).goal_world_T_fragment)
        window = ContactWindow(arm=arm, kind=kind, center_xy=tuple(float(v) for v in pose[:2, 3]),
                               fragment_id=fragment_id)
        if self._contacts.get(arm) != window and self._in_contact_band(arm):
            return ExecutionResult(ok=False, steps_executed=0, error=_failure(
                ErrorCode.SAFETY_INTERLOCK, f'{arm} is inside the contact band: lift it out before declaring another '
                                            'contact'))
        self._contacts[arm] = window
        return ExecutionResult(ok=True, steps_executed=0, diagnostics={
            'contact': {'arm': arm, 'kind': kind, 'fragment_id': fragment_id, 'center_xy': window.center_xy,
                        'region_radius_m': self.limits.contact_region_radius_m}})

    def clear_contact(self, *, arm='primary'):
        """Remove `arm`'s declared contact window; refused inside the contact band."""
        self._require_task()
        arm = self.resolve_arm(arm)
        if arm not in ARMS:
            return ExecutionResult(ok=False, steps_executed=0,
                                   error=_failure(ErrorCode.NOT_FOUND, f'unknown arm {arm!r}; arms are {ARMS}'))
        if arm in self._contacts and self._in_contact_band(arm):
            return ExecutionResult(ok=False, steps_executed=0, error=_failure(
                ErrorCode.SAFETY_INTERLOCK, f'{arm} is inside the contact band: lift it out before clearing its contact'))
        self._contacts.pop(arm, None)
        return ExecutionResult(ok=True, steps_executed=0)

    def resolve_arm(self, arm):
        return ARM_ALIASES.get(arm, arm)

    def reset(self, task_ref, seed):
        guard_seed(seed, self.mode, self.config)
        if self._closed:
            raise RuntimeError('the adapter is closed')
        task = configured_initial_condition(verify_task(load_task(task_ref)), seed, self.config)
        if abs(task.table_top_z_m - self.limits.table_top_z_m) > 1e-9:
            raise ValueError(f'S0 limits put the table at {self.limits.table_top_z_m} m, the task at '
                             f'{task.table_top_z_m} m')
        snapshot = self.backend.reset(task)
        self.task, self.last_score, self.settle_error, self.rejected_commands = task, None, None, 0
        self.protective_stop = None
        self._contacts = {}
        self._home = {arm: np.array(snapshot.joints[arm], dtype=float) for arm in ARMS}
        self._targets = {arm: np.array(snapshot.joints[arm], dtype=float) for arm in ARMS}
        self._widths = {arm: float(np.clip(snapshot.widths[arm], *self.limits.gripper_width_range_m[arm]))
                        for arm in ARMS}
        self._static = self._static_metadata(task)
        observation = self.get_observation()
        if self._run_observer is not None:
            self._run_observer.on_reset(observation, {
                'task_id': task.task_id, 'layout_sha256': task.layout_sha256, 'anchor_id': task.anchor_id,
                'goal_source': task.goal_source, 'seed': task.seed, 'split': task.split, 'mode': self.mode,
                'backend': type(self.backend).__name__})
        return observation

    def _require_task(self):
        if self._closed:
            raise RuntimeError('the adapter is closed')
        if self.task is None:
            raise RuntimeError('reset() must be called before using the skull assembly adapter')
        return self.task

    def _static_metadata(self, task):
        return {
            'embodiment': EMBODIMENT, 'base_frame': BASE_FRAME,
            'frame_convention': 'robot_base is the task world frame: metres, z up, quaternion_wxyz',
            'arms': ARMS, 'arm_aliases': dict(ARM_ALIASES), 'joint_names': {arm: self.joint_names for arm in ARMS},
            'fragment_ids': task.fragment_ids, 'anchor_id': task.anchor_id, 'moving_ids': task.moving_ids,
            'anchor_fixture': 'held at its goal pose by a fixture for the whole trial; it cannot move',
            'goal_source': task.goal_source,
            'goal_poses': {f.fragment_id: pose_mapping(f.goal_world_T_fragment) for f in task.fragments},
            'assembly_plan': None if task.assembly_plan is None else [step.to_record() for step in task.assembly_plan],
            'fragment_local_bounds': {f.fragment_id: {'min': f.local_bounds[0], 'max': f.local_bounds[1]}
                                      for f in task.fragments},
            'placement_tolerance': {'position_m': task.position_tolerance_m,
                                    'rotation_deg': task.rotation_tolerance_deg,
                                    'judged': 'each fragment relative to the final anchor pose; the anchor must '
                                              'stay within the same tolerance of its goal'},
            'table_top_z_m': task.table_top_z_m,
            'contact_rule': self._contact_rule(),
            'controller': {
                'control_frequency_hz': self.control_frequency, 'control_period_s': self.control_period_s,
                'joint_lower_rad': {arm: self.model.arms[arm].lower for arm in ARMS},
                'joint_upper_rad': {arm: self.model.arms[arm].upper for arm in ARMS},
                'max_joint_velocity_rad_s': self.limits.max_joint_velocity_rad_s,
                'max_joint_acceleration_rad_s2': self.limits.max_joint_acceleration_rad_s2,
                'gripper_width_range_m': {arm: self.limits.gripper_width_range_m[arm] for arm in ARMS},
                'gripper_semantics': '0 = minimum width (closed), 1 = maximum width (open), linear in width',
                'arm_arm_clearance_m': self.limits.arm_arm_clearance_m,
                'table_clearance_m': self.limits.table_clearance_m, 'tcp_min_height_m': self.limits.tcp_min_height_m,
                'command_checks': 'every joint command is checked from the measured state against these limits, '
                                  'arm-arm clearance, tool and link clearance above the table and TCP height; a '
                                  'violating command is rejected (safety_interlock) and nothing moves; joint targets '
                                  'are reached in one control period, so large moves need plan_motion, move_to_pose '
                                  'or go_home',
            },
        }

    def _contact_rule(self):
        limits = self.limits
        if not limits.contact_enabled:
            return {'configured': False, 'meaning': "S0's contact phase is not configured: nothing may come closer to "
                                                    'the table than table_clearance_m'}
        return {'configured': True,
                'declare': 'fanuc_crx.declare_contact(kind, fragment_id, arm=...)',
                'clear': 'fanuc_crx.clear_contact(arm=...)',
                'kinds': {'grasp': "the window is centred on the fragment's observed pose",
                          'place': "the window is centred on the fragment's goal pose"},
                'when': 'before any motion that brings an arm closer to the table than table_clearance_m (tool) or '
                        'tcp_min_height_m (TCP); not while that arm is already that close (lift it out first)',
                'inside': 'only the fingers may go lower, never closer than fingers_clearance_m to the table, within '
                          'region_radius_m of the window centre, every finger corner at most max_speed_m_s; the '
                          'planner slows down by itself while a window is declared',
                'region_radius_m': limits.contact_region_radius_m,
                'fingers_clearance_m': limits.contact_table_clearance_m,
                'max_speed_m_s': limits.contact_max_speed_m_s}

    def get_task_context(self):
        task = self._require_task()
        observed = self.backend.observed_fragment_poses()
        metadata = {**self._static, 'observed_at_s': self.backend.robot_state().time_s,
                    'observed_fragment_poses': {key: pose_mapping(pose.world_T_fragment)
                                                for key, pose in observed.items()},
                    'observed_pose_sources': {key: pose.source for key, pose in observed.items()},
                    'observed_pose_times_s': {key: pose.time_s for key, pose in observed.items()}}
        tolerance = f'{task.position_tolerance_m * 1000:g} mm and {task.rotation_tolerance_deg:g} deg'
        language = (f'Assemble the fragments: the anchor {task.anchor_id} is held at its goal by a fixture; place '
                    f'{", ".join(task.moving_ids)} at their goal poses (metadata goal_poses, frame robot_base) within '
                    f'{tolerance} relative to the anchor. Current fragment poses are in metadata '
                    f'observed_fragment_poses. To grasp or place close to the table, first declare the contact with '
                    f'fanuc_crx.declare_contact (metadata contact_rule).')
        if task.assembly_plan is not None:
            language += (' The fragments go in in the order of metadata assembly_plan: with the anchor and the '
                         'fragments before it in place, each one goes into its goal by a straight last move along its '
                         'insertion_direction (frame robot_base) without colliding.')
        return TaskContext(suite=SUITE, task_id=0, task_name=task.task_id, language=language, family=FAMILY,
                           metadata=_frozen(metadata))

    def _normalized(self, arm, width):
        low, high = self.limits.gripper_width_range_m[arm]
        slack = MEASURED_WIDTH_SLACK * (high - low)
        if not low - slack <= width <= high + slack:
            raise ValueError(f'{arm}: gripper width {width} m is outside the configured range [{low}, {high}] m')
        return float(np.clip((width - low) / (high - low), 0.0, 1.0))

    def _width(self, arm, position):
        """The width of a normalized position in [0, 1]; the clip only removes rounding at the range ends."""
        low, high = self.limits.gripper_width_range_m[arm]
        return float(np.clip(low + float(position) * (high - low), low, high))

    def _robot_state(self, snapshot):
        return RobotState(
            joint_positions=dict(snapshot.joints), joint_velocities=dict(snapshot.joint_velocities),
            end_effector_poses={arm: as_pose(self.model.arms[arm].tcp(snapshot.joints[arm])) for arm in ARMS},
            gripper_positions={arm: self._normalized(arm, snapshot.widths[arm]) for arm in ARMS},
            base_frame=BASE_FRAME, joint_names={arm: self.joint_names for arm in ARMS}, timestamp_s=snapshot.time_s,
            embodiment=EMBODIMENT)

    def get_robot_state(self):
        self._require_task()
        return self._robot_state(self.backend.robot_state())

    def get_observation(self):
        self._require_task()
        snapshot = self.backend.robot_state()
        cameras = {name: CameraObservation(rgb=frame.rgb, depth_m=frame.depth_m, intrinsics=frame.intrinsics,
                                           frame=f'camera/{name}', camera_pose=as_pose(frame.world_T_camera),
                                           timestamp_s=frame.time_s)
                   for name, frame in self.backend.camera_frames().items()}
        return Observation(cameras=cameras, robot_state=self._robot_state(snapshot),
                           task_context=self.get_task_context(), timestamp_s=snapshot.time_s)

    def _action(self, joints, widths):
        return RobotAction({arm: ArmCommand('joint_position', joints[arm], self._normalized(arm, widths[arm]),
                                            embodiment=EMBODIMENT) for arm in ARMS})

    def _check(self, snapshot, times, path, samples):
        """Raise Rejected unless the path passes check_joint_path and, between control periods, the geometry checks
        at S0's servo rate (joints interpolated linearly from one target to the next), with each gripper's widest
        opening over the path (S0's boxes at the openings, when its limits take them); then bound the fingers' reach
        on the backend for the path (AssemblyBackend.bound_reach)."""
        for _, widths in samples:
            for arm in ARMS:
                low, high = self.limits.gripper_width_range_m[arm]
                if not low <= widths[arm] <= high:
                    raise Rejected(f'{arm} width {widths[arm]:.4f} m outside [{low}, {high}] m')
        contacts = tuple(self._contacts.values())
        openings = {arm: widest_opening(snapshot.spans[arm], self._widths[arm], [w[arm] for _, w in samples])
                    for arm in ARMS}
        check_joint_path(self.model, self.limits, now=snapshot.time_s, q_now=snapshot.joints,
                         qd_now=snapshot.joint_velocities, qd_window_s=snapshot.velocity_window_s, times=times,
                         joints=path, contacts=contacts, openings=openings)
        factor = int(np.ceil(self.limits.servo_hz * self.control_period_s - 1e-9))
        if factor > 1:
            fractions = np.arange(1, factor) / factor
            starts = np.concatenate([[snapshot.time_s], times[:-1]])
            between = {}
            for arm in ARMS:
                points = np.vstack([snapshot.joints[arm], path[arm]])
                steps = np.diff(points, axis=0)
                between[arm] = (points[:-1, None] + fractions[None, :, None] * steps[:, None]).reshape(
                    -1, points.shape[1])
            check_geometry(self.model, self.limits, (starts[:, None] + fractions * self.control_period_s).ravel(),
                           between, contacts=contacts, openings=openings,
                           start=(snapshot.time_s, {arm: np.asarray(snapshot.joints[arm], dtype=float) for arm in ARMS}))
        self.backend.bound_reach(reach_bounds(self.limits, openings))

    def _run(self, samples):
        """Check the (joints, widths) samples, one per control period, as one path from the measured state; step
        through them only if the whole path passes."""
        if self.protective_stop is not None:
            return Executed(False, 0, None, _failure(ErrorCode.SAFETY_INTERLOCK, 'the robot stopped (protective stop: '
                                                     f'{self.protective_stop.reason}); the trial is over'))
        snapshot = self.backend.robot_state()
        times = snapshot.time_s + self.control_period_s * np.arange(1, len(samples) + 1)
        path = {arm: np.array([joints[arm] for joints, _ in samples], dtype=float) for arm in ARMS}
        try:
            self._check(snapshot, times, path, samples)
        except Rejected as exc:
            self.rejected_commands += 1
            return Executed(False, 0, None, _failure(ErrorCode.SAFETY_INTERLOCK,
                                                     f'S0 rejected the command, nothing moved: {exc}', reason=str(exc)))
        observation = None
        for index, (joints, widths) in enumerate(samples):
            action = self._action(joints, widths)
            if self._run_observer is not None:
                self._run_observer.before_step(action)
            try:
                self.backend.step(joints, widths)
            except ProtectiveStop as halt:  # the robot stopped (configs/jing.json s0.protective_stop): the trial ends
                self.protective_stop = halt
                observation = self.get_observation()
                if self._run_observer is not None:  # the environment ended the episode at this step (ASPIRE's record
                    # reads termination from executed steps only)
                    self._run_observer.after_step(action, StepResult(ok=True, observation=observation, terminated=True,
                                                                     diagnostics={'protective_stop': halt.reason}))
                return Executed(False, index, observation,
                                _failure(ErrorCode.SAFETY_INTERLOCK, f'protective stop: {halt.reason}'))
            except Exception as exc:  # a backend failure reaches the program as a typed result, not a traceback
                result = StepResult(ok=False, error=_failure(ErrorCode.ADAPTER_FAILED, f'backend step failed: {exc}',
                                                             exception_type=type(exc).__name__))
                if self._run_observer is not None:
                    self._run_observer.after_step(action, result)
                return Executed(False, index, observation, result.error)
            self._targets = {arm: np.array(joints[arm], dtype=float) for arm in ARMS}
            self._widths = dict(widths)
            observation = self.get_observation()
            if self._run_observer is not None:
                self._run_observer.after_step(action, StepResult(ok=True, observation=observation))
        return Executed(True, len(samples), observation)

    def _finish(self, executed, targets):
        """ExecutionResult of a checked motion that should end at `targets` (arm -> joints)."""
        if not executed.ok:
            return ExecutionResult(ok=False, steps_executed=executed.steps, final_observation=executed.observation,
                                   error=executed.error)
        final = executed.observation.robot_state
        errors = {arm: float(np.max(np.abs(final.joint_positions[arm] - target))) for arm, target in targets.items()}
        if max(errors.values()) > TRAJECTORY_TOLERANCE_RAD:
            return ExecutionResult(ok=False, steps_executed=executed.steps, final_observation=executed.observation,
                                   error=_failure(ErrorCode.TIMEOUT, 'the final joint target did not converge',
                                                  joint_error_rad=errors), final_errors=errors)
        return ExecutionResult(ok=True, steps_executed=executed.steps, final_observation=executed.observation,
                               final_errors=errors, diagnostics={'joint_error_rad': errors})

    def step(self, action):
        self._require_task()
        if not isinstance(action, RobotAction) or action.embodiment != EMBODIMENT:
            return StepResult(ok=False, error=_failure(ErrorCode.INVALID_REQUEST,
                                                       f'step needs a RobotAction for the {EMBODIMENT} embodiment'))
        unknown = sorted(set(action.arms) - set(ARMS))
        if unknown:
            return StepResult(ok=False, error=_failure(ErrorCode.NOT_FOUND, f'unknown arms {unknown}; arms are {ARMS}'))
        joints = {arm: action.arms[arm].target if arm in action.arms else self._targets[arm] for arm in ARMS}
        widths = dict(self._widths)
        for arm, command in action.arms.items():
            if command.gripper_position is not None:
                widths[arm] = self._width(arm, command.gripper_position)
        executed = self._run([(joints, widths)])
        if not executed.ok:
            return StepResult(ok=False, observation=executed.observation, error=executed.error,
                              terminated=self.protective_stop is not None)
        return StepResult(ok=True, observation=executed.observation,
                          diagnostics={'control_period_s': self.control_period_s})

    def execute_trajectory(self, trajectory):
        self._require_task()
        if not isinstance(trajectory, (Trajectory, SynchronizedTrajectory)) or trajectory.embodiment != EMBODIMENT:
            raise ValueError(f'expected a Trajectory or SynchronizedTrajectory for the {EMBODIMENT} embodiment')
        if not np.isclose(trajectory.dt_s, self.control_period_s, rtol=0.0, atol=1e-9):
            raise ValueError(f'trajectory dt_s {trajectory.dt_s} must equal the control period {self.control_period_s}')
        if isinstance(trajectory, Trajectory):
            positions, names = {trajectory.arm: trajectory.joint_positions}, {trajectory.arm: trajectory.joint_names}
            grippers = None if trajectory.gripper_positions is None else {trajectory.arm: trajectory.gripper_positions}
        else:
            positions, names = trajectory.joint_positions, trajectory.joint_names
            grippers = trajectory.gripper_positions
            if set(positions) != set(ARMS):
                raise ValueError(f'a synchronized trajectory must name every arm {ARMS}')
        if not set(positions) <= set(ARMS):
            raise ValueError(f'trajectory arms {sorted(positions)} are not in {ARMS}')
        if any(tuple(value) != self.joint_names for value in names.values()):
            raise ValueError(f'trajectory joint names must be {self.joint_names}')
        samples = []
        for index in range(len(next(iter(positions.values())))):
            joints = {arm: positions[arm][index] if arm in positions else self._targets[arm] for arm in ARMS}
            widths = dict(self._widths)
            for arm, values in (grippers or {}).items():
                widths[arm] = self._width(arm, values[index])
            samples.append((joints, widths))
        samples.append(samples[-1])  # one hold period at the last waypoint: the commanded motion ends at rest
        return self._finish(self._run(samples), {arm: values[-1] for arm, values in positions.items()})

    def go_home(self, *, arm='primary'):
        """Move `arm` to its joints at reset along a straight joint path sized by `action_bounds`, plus the hold period,
        through the same S0 checks (CapApi's own fallback commands the home target in one period, which they reject)."""
        self._require_task()
        arm = self.resolve_arm(arm)
        if arm not in ARMS:
            return ExecutionResult(ok=False, steps_executed=0,
                                   error=_failure(ErrorCode.NOT_FOUND, f'unknown arm {arm!r}; arms are {ARMS}'))
        start, home = np.array(self.backend.robot_state().joints[arm]), self._home[arm]
        count = max(1, int(np.ceil(np.max(np.abs(home - start) / (self.action_bounds[1] * self.control_period_s)))))
        samples = [({**self._targets, arm: home if k == count else start + (home - start) * k / count},
                    dict(self._widths)) for k in range(1, count + 1)]
        return self._finish(self._run([*samples, samples[-1]]), {arm: home})

    def set_gripper(self, position, *, arm='primary'):
        """Command one gripper width, holding the joints, until the opening is reached or stops changing.

        A command refused before anything moved raises its ApiError instead of returning it: CapApi keeps a
        set_gripper/set_grippers target as the commanded opening of later motions even when the call failed, unless
        the call raises, and reports the raise as a failed ExecutionResult."""
        self._require_task()
        arm = self.resolve_arm(arm)
        if arm not in ARMS:
            return ExecutionResult(ok=False, steps_executed=0,
                                   error=_failure(ErrorCode.NOT_FOUND, f'unknown arm {arm!r}; arms are {ARMS}'))
        numeric = float(position)
        if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            return ExecutionResult(ok=False, steps_executed=0,
                                   error=_failure(ErrorCode.INVALID_REQUEST, 'gripper position must be in [0, 1]'))
        target = self._width(arm, numeric)
        steps, observation = 0, None
        width = self.backend.robot_state().widths[arm]
        for _ in range(self.gripper_max_steps):
            before = width
            executed = self._run([(dict(self._targets), {**self._widths, arm: target})])
            if not executed.ok:
                if steps == 0:
                    raise executed.error
                return ExecutionResult(ok=False, steps_executed=steps, final_observation=observation,
                                       error=executed.error)
            steps, observation = steps + 1, executed.observation
            width = self.backend.robot_state().widths[arm]
            reached, stalled = abs(width - target) <= GRIPPER_TOLERANCE_M, abs(width - before) <= GRIPPER_TOLERANCE_M
            if reached or (stalled and steps > 1):
                diagnostics = {'completion': 'reached' if reached else 'stalled', 'width_m': width,
                               'target_width_m': target}
                return ExecutionResult(ok=True, steps_executed=steps, final_observation=observation,
                                       diagnostics=diagnostics)
        return ExecutionResult(ok=False, steps_executed=steps, final_observation=observation,
                               error=_failure(ErrorCode.TIMEOUT, f'{arm} gripper still moving after {steps} periods',
                                              width_m=width, target_width_m=target))

    def set_grippers(self, positions):
        """Command both gripper widths in one shared control period; a refused command raises (see set_gripper)."""
        self._require_task()
        if not isinstance(positions, Mapping) or set(positions) != set(ARMS):
            return ExecutionResult(ok=False, steps_executed=0,
                                   error=_failure(ErrorCode.INVALID_REQUEST, f'positions must name every arm {ARMS}'))
        widths = {}
        for arm in ARMS:
            numeric = float(positions[arm])
            if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                return ExecutionResult(ok=False, steps_executed=0, error=_failure(
                    ErrorCode.INVALID_REQUEST, f'gripper position for {arm!r} must be finite and in [0, 1]'))
            widths[arm] = self._width(arm, numeric)
        executed = self._run([(dict(self._targets), widths)])
        if not executed.ok and executed.steps == 0:
            raise executed.error
        return ExecutionResult(ok=executed.ok, steps_executed=executed.steps, final_observation=executed.observation,
                               error=executed.error)

    def check_success(self):
        """Release and settle (module docstring), then score_assembly on the backend's privileged poses; the score is
        kept in `last_score`, and why settling was refused, if it was, in `settle_error` (the score is then not
        settled and not a success)."""
        task = self._require_task()
        if self.protective_stop is None:
            try:
                settled = self._release_and_settle(task)
            except ProtectiveStop as halt:
                self.protective_stop = halt
        if self.protective_stop is not None:  # the trial ended where the robot stopped: a failure, unscored
            self.last_score = None
            return False
        self.last_score = score_assembly(task, self.backend.privileged_fragment_poses(), settled=settled)
        return self.last_score.success

    def _release_and_settle(self, task):
        """Open both grippers fully and hold the last joint targets for the task's settle time (at least one period),
        checked as one path; stepped on the backend directly, so it uses none of the program's step budget."""
        widths = {arm: self.limits.gripper_width_range_m[arm][1] for arm in ARMS}
        periods = max(1, int(np.ceil(task.settle_s * self.control_frequency - 1e-9)))
        samples = [(dict(self._targets), widths)] * periods
        snapshot = self.backend.robot_state()
        times = snapshot.time_s + self.control_period_s * np.arange(1, periods + 1)
        try:
            self._check(snapshot, times, {arm: np.array([self._targets[arm]] * periods) for arm in ARMS}, samples)
        except Rejected as exc:
            self.settle_error = f'S0 rejected the release-and-settle hold: {exc}'
            return False
        for joints, opening in samples:
            self.backend.step(joints, opening)
        self._widths = dict(widths)
        self.settle_error = None
        return True

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.backend.close()
