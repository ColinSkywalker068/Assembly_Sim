"""The simulator boundary every method executes through, and a physics-free double for tests.

AssemblyBackend is what an executor needs from a simulator: reset to an AssemblyTask, the robot's
joint state, the fragment poses a method may observe (with their source), camera frames, one control
period of joint-position and gripper-width targets, and the privileged final poses used only for
scoring. Checks on what is commanded (limits, collisions) belong to the caller (S0, the ASPIRE
adapter), not to the backend. The Isaac Sim implementation is sim/backend.py (IsaacAssemblyBackend); it
starts a trial from where the fragments came to rest after the reset, not exactly from the task's
initial poses (one staged skull fragment tips 18 deg while settling).

End of a trial, for every method: the executor opens both grippers fully and holds the arms for the
task's `settle_s` (at least one control period), through its usual command checks, and only then
reads `privileged_fragment_poses()` for scoring (`score_assembly(..., settled=True)`), so a fragment
still held is judged where it fell. There is no rest check at that point: a fragment still sliding
after `settle_s` is judged where it is then.

KinematicBackend implements the protocol without physics so adapters can be tested end to end: joints
reach their targets exactly, a closing gripper whose TCP is within `capture_radius_m` of a free
fragment's box centre carries that fragment rigidly (never the anchor, which the fixture holds), an
opening gripper lets it go, and a released
fragment stays where it was let go (no gravity, no contact, no collisions). Its results are never
reported as baseline performance.
"""
from __future__ import annotations

import numbers
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from ..contracts.actions import ARMS
from ..contracts.skills import POSE_SOURCES
from ..perception import rigid_transform


def _frozen(value, dtype=float):
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _finite_time(value, name):
    if isinstance(value, bool) or not np.isfinite(float(value)):
        raise ValueError(f'{name} must be a finite time in seconds')
    return float(value)


def _joint_vector(value, name):
    array = _frozen(value)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite joint vector')
    return array


@dataclass(frozen=True)
class RobotSnapshot:
    """Measured robot state: joints (rad), joint velocities (rad/s), gripper openings (m) and TCPs (world).
    `velocity_window_s` says what the velocities are: their average over that time up to `time_s` (0 for an
    instantaneous velocity). S0 reads a path's first step with it (s0.adapter.check_joint_path)."""

    time_s: float
    joints: Mapping[str, np.ndarray]
    joint_velocities: Mapping[str, np.ndarray]
    widths: Mapping[str, float]
    tcp: Mapping[str, np.ndarray]
    velocity_window_s: float

    def __post_init__(self):
        object.__setattr__(self, 'time_s', _finite_time(self.time_s, 'time_s'))
        window = self.velocity_window_s
        if isinstance(window, bool) or not isinstance(window, numbers.Real) or not (np.isfinite(window) and window >= 0):
            raise ValueError('velocity_window_s must be a finite, non-negative time in seconds')
        object.__setattr__(self, 'velocity_window_s', float(window))
        for name in ('joints', 'joint_velocities', 'tcp', 'widths'):
            if set(getattr(self, name)) != set(ARMS):
                raise ValueError(f'{name} must cover {ARMS}')
        joints = {arm: _joint_vector(self.joints[arm], f'joints[{arm}]') for arm in ARMS}
        velocities = {arm: _joint_vector(self.joint_velocities[arm], f'joint_velocities[{arm}]') for arm in ARMS}
        if any(joints[arm].shape != velocities[arm].shape for arm in ARMS):
            raise ValueError('joints and joint_velocities must have the same shape per arm')
        tcp = {arm: _frozen(rigid_transform(self.tcp[arm])) for arm in ARMS}
        widths = {arm: float(self.widths[arm]) for arm in ARMS}
        if not all(np.isfinite(width) and width >= 0 for width in widths.values()):
            raise ValueError('widths must be finite and non-negative')
        for name, value in (('joints', joints), ('joint_velocities', velocities), ('tcp', tcp), ('widths', widths)):
            object.__setattr__(self, name, MappingProxyType(value))


@dataclass(frozen=True)
class ObservedPose:
    """A fragment pose a method may use: `source` is one of contracts.skills.POSE_SOURCES, `time_s` when it was
    observed (the clock of RobotSnapshot.time_s)."""

    world_T_fragment: np.ndarray
    source: str
    time_s: float

    def __post_init__(self):
        if self.source not in POSE_SOURCES:
            raise ValueError(f'pose source must be one of {POSE_SOURCES}')
        object.__setattr__(self, 'world_T_fragment', _frozen(rigid_transform(self.world_T_fragment)))
        object.__setattr__(self, 'time_s', _finite_time(self.time_s, 'time_s'))


@dataclass(frozen=True)
class CameraFrame:
    """RGB (H, W, 3) uint8, depth (H, W) metres, K (OpenCV pixel-centre convention), world_T_camera with
    OpenCV axes (x right, y down, z forward), and `time_s`, the sample (exposure) time on the clock of
    RobotSnapshot.time_s, which may be older than the latest robot state. `synthetic` marks frames that are
    not renders."""

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    world_T_camera: np.ndarray
    time_s: float
    synthetic: bool = False

    def __post_init__(self):
        rgb, depth = np.asarray(self.rgb), np.asarray(self.depth_m)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or depth.shape != rgb.shape[:2]:
            raise ValueError('camera frame needs uint8 RGB (H, W, 3) and depth (H, W)')
        if isinstance(self.time_s, bool) or not np.isfinite(float(self.time_s)):
            raise ValueError('camera frame needs a finite sample time time_s')
        k = np.asarray(self.intrinsics, dtype=float)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or k[0, 1] != 0
                or k[1, 0] != 0 or not np.allclose(k[2], [0, 0, 1])):
            raise ValueError('camera frame needs rectified zero-skew 3x3 intrinsics with fx, fy > 0')
        object.__setattr__(self, 'time_s', float(self.time_s))
        object.__setattr__(self, 'rgb', _frozen(rgb, np.uint8))
        object.__setattr__(self, 'depth_m', _frozen(depth, np.float32))
        object.__setattr__(self, 'intrinsics', _frozen(k))
        object.__setattr__(self, 'world_T_camera', _frozen(rigid_transform(self.world_T_camera)))


def checked_targets(model, width_range_m, joints, widths):
    """{arm: joint target vector}, after the checks every backend applies before anything moves: both arms given,
    finite targets of the model's shape inside its joint limits, widths inside `width_range_m` (arm -> (low, high)).
    The vectors are read-only copies, so a caller reusing its buffer cannot change a target afterwards."""
    if set(joints) != set(ARMS) or set(widths) != set(ARMS):
        raise ValueError(f'joints and widths must cover {ARMS}')
    targets = {}
    for arm in ARMS:
        target = _frozen(joints[arm])
        arm_model = model.arms[arm]
        if target.shape != (arm_model.dof,) or not np.isfinite(target).all():
            raise ValueError(f'{arm}: expected {arm_model.dof} finite joint targets')
        if (target < arm_model.lower).any() or (target > arm_model.upper).any():
            raise ValueError(f'{arm}: joint targets outside the model limits')
        low, high = width_range_m[arm]
        if not low <= float(widths[arm]) <= high:
            raise ValueError(f'{arm}: width {widths[arm]} outside [{low}, {high}] m')
        targets[arm] = target
    return targets


@runtime_checkable
class AssemblyBackend(Protocol):
    control_hz: float
    velocity_window_s: float  # what its snapshots' joint velocities average over (RobotSnapshot)

    def reset(self, task) -> RobotSnapshot:
        """Load `task`'s fragments at their initial poses, the anchor held there by a fixture (the protocol's
        `anchor`: nothing can move it), and the robot at home; return the robot state. A physics backend lets the
        free fragments settle first, so the trial starts where they came to rest."""
        ...

    def robot_state(self) -> RobotSnapshot:
        ...

    def observed_fragment_poses(self) -> Mapping[str, ObservedPose]:
        """Fragment poses a method may use, each labelled with where it came from."""
        ...

    def camera_frames(self) -> Mapping[str, CameraFrame]:
        ...

    def step(self, joints: Mapping[str, np.ndarray], widths: Mapping[str, float]) -> RobotSnapshot:
        """Track these joint and width targets for one control period (1 / control_hz seconds)."""
        ...

    def privileged_fragment_poses(self) -> Mapping[str, np.ndarray]:
        """Simulator truth, for scoring after the method has finished; never handed to a method."""
        ...

    def close(self) -> None:
        ...


class KinematicBackend:
    """Physics-free AssemblyBackend for tests (see module docstring); fragment poses observed as oracle_sim."""

    def __init__(self, model, *, home_joints: Mapping[str, np.ndarray], width_range_m: Mapping[str, tuple],
                 control_hz=20.0, capture_radius_m=0.02):
        if set(home_joints) != set(ARMS) or set(width_range_m) != set(ARMS):
            raise ValueError(f'home_joints and width_range_m must cover {ARMS}')
        for name, value in (('control_hz', control_hz), ('capture_radius_m', capture_radius_m)):
            if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        self.model = model
        self.home = {arm: _joint_vector(home_joints[arm], f'home_joints[{arm}]') for arm in ARMS}
        self.width_range = {arm: (float(low), float(high)) for arm, (low, high) in width_range_m.items()}
        if not all(np.isfinite(pair).all() and 0 <= pair[0] < pair[1] for pair in self.width_range.values()):
            raise ValueError('width_range_m must be finite (low, high) with 0 <= low < high')
        self.control_hz = float(control_hz)
        self.velocity_window_s = 1.0 / self.control_hz  # each step's (target - previous) * control_hz
        self.capture_radius_m = float(capture_radius_m)
        self.task = None

    def reset(self, task):
        self.task = task
        self.time_s = 0.0
        self.joints = {arm: self.home[arm].copy() for arm in ARMS}
        self.velocities = {arm: np.zeros_like(self.home[arm]) for arm in ARMS}
        self.widths = {arm: self.width_range[arm][1] for arm in ARMS}
        self.poses = task.initial_poses()
        self.held = {arm: None for arm in ARMS}  # arm -> (fragment_id, tcp_T_fragment)
        return self.robot_state()

    def _require_reset(self):
        if self.task is None:
            raise RuntimeError('reset the backend first')

    def robot_state(self):
        self._require_reset()
        return RobotSnapshot(self.time_s, self.joints, self.velocities, self.widths,
                             {arm: self.model.arms[arm].tcp(self.joints[arm]) for arm in ARMS}, self.velocity_window_s)

    def observed_fragment_poses(self):
        self._require_reset()
        return {key: ObservedPose(pose, 'oracle_sim', self.time_s) for key, pose in self.poses.items()}

    def privileged_fragment_poses(self):
        self._require_reset()
        return {key: pose.copy() for key, pose in self.poses.items()}

    def camera_frames(self):
        self._require_reset()
        camera = np.eye(4)
        camera[:3, :3] = np.diag([1.0, -1.0, -1.0])  # looking straight down
        camera[:3, 3] = [*self.task.world_T_assembly[:2, 3], self.task.table_top_z_m + 1.0]
        intrinsics = np.array([[8.0, 0.0, 3.5], [0.0, 8.0, 3.5], [0.0, 0.0, 1.0]])
        return {'agent': CameraFrame(np.zeros((8, 8, 3), np.uint8), np.ones((8, 8), np.float32), intrinsics,
                                     camera, self.time_s, synthetic=True)}

    def _box_centre(self, key):
        fragment = self.task.fragment(key)
        low, high = (np.asarray(corner) for corner in fragment.local_bounds)
        return self.poses[key][:3, :3] @ ((low + high) / 2) + self.poses[key][:3, 3]

    def step(self, joints, widths):
        self._require_reset()
        targets = checked_targets(self.model, self.width_range, joints, widths)
        for arm in ARMS:
            width = float(widths[arm])
            tcp = self.model.arms[arm].tcp(self.joints[arm])
            if width > self.widths[arm] + 1e-9:
                self.held[arm] = None
            elif width < self.widths[arm] - 1e-9 and self.held[arm] is None:
                taken = {held[0] for held in self.held.values() if held is not None}
                free = [key for key in self.task.moving_ids if key not in taken]
                distances = {key: np.linalg.norm(self._box_centre(key) - tcp[:3, 3]) for key in free}
                nearest = min(distances, key=distances.get, default=None)
                if nearest is not None and distances[nearest] <= self.capture_radius_m:
                    self.held[arm] = (nearest, np.linalg.inv(tcp) @ self.poses[nearest])
            self.velocities[arm] = (targets[arm] - self.joints[arm]) * self.control_hz
            self.joints[arm] = targets[arm]
            self.widths[arm] = width
            if self.held[arm] is not None:
                key, tcp_T_fragment = self.held[arm]
                self.poses[key] = self.model.arms[arm].tcp(self.joints[arm]) @ tcp_T_fragment
        self.time_s += 1.0 / self.control_hz
        return self.robot_state()

    def holding(self):
        """Test helper: arm -> held fragment id (privileged; a real backend has no such oracle)."""
        return {arm: None if held is None else held[0] for arm, held in self.held.items()}

    def close(self):
        self.task = None
