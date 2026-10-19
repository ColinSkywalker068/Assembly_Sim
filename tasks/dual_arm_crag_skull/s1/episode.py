"""Episode records for S1 (conditional Diffusion Policy): streams, skill segments, persistence. numpy only.

Conventions follow INTERFACES.md: metres, seconds, radians, `T_A_B` maps B coordinates to A, every
stream stamped on the episode monotonic clock. Ground truth that the policy must not see (object
poses, latent demo modes) lives in `privileged.npz`, which the loaders never open. Windowed
samples, the split and the normalizer are in `dataset.py`.

Timing semantics:
  * pose streams are stamped at measurement time; observations use the latest sample at or
    before the observation time (strictly causal), supervision interpolates future samples;
  * camera frames carry both the exposure time and the receive time; a frame is usable at t only
    once it has been received (receive_time <= t), and its age is measured from the exposure;
  * commanded streams are stamped at the time the command is meant to be reached.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..contracts.actions import ARMS, rotation6_from_matrix
from ..contracts.skills import POSE_SOURCES, SKILLS, TARGET_SOURCES

SCHEMA_VERSION = 2
SOURCES = ('sim_expert', 'umi', 'robot_teleop')
CAMERAS = ('left_wrist', 'right_wrist', 'agent')
CLOCK_DOMAIN = 'episode_monotonic'
SUPERVISION = ('achieved', 'commanded')
POLICY_HZ = 10
OBS_STEPS = 2
PRED_STEPS = 16
EXEC_STEPS = 4
IMAGE_MAX_AGE_S = 0.15
STATE_MAX_AGE_S = 0.1
MAX_GAP_S = 0.1
# Explicit label for simulator ground-truth goals; refused unless a dataset opts into oracle runs.
ORACLE_TARGET_SOURCE = 'oracle_ground_truth'
ORACLE_POSE_SOURCE = 'oracle_sim'
_TIME_EPS = 1e-9


def _frozen(value, dtype=float):
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _check_times(times, name, minimum=2):
    if times.ndim != 1 or len(times) < minimum or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError(f'{name}: need >= {minimum} finite, strictly increasing times')


def _check_poses(poses, name):
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError(f'{name}: expected finite (T, 4, 4) poses')
    rotation = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(np.swapaxes(rotation, 1, 2) @ rotation, np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError(f'{name}: poses must be rigid SE(3)')


def invert(poses):
    """Inverse of (..., 4, 4) rigid transforms."""
    poses = np.asarray(poses, dtype=float)
    result = np.zeros_like(poses)
    transposed = np.swapaxes(poses[..., :3, :3], -1, -2)
    result[..., :3, :3] = transposed
    result[..., :3, 3] = -(transposed @ poses[..., :3, 3, None])[..., 0]
    result[..., 3, 3] = 1.0
    return result


def pose9(relative):
    """(..., 4, 4) -> (..., 9): translation then rotation6 (first, second column)."""
    relative = np.asarray(relative, dtype=float)
    return np.concatenate([relative[..., :3, 3], rotation6_from_matrix(relative[..., :3, :3])], axis=-1)


@dataclass(frozen=True)
class PoseStream:
    """Sampled `world_T_tcp` poses and absolute gripper widths in metres."""

    times: np.ndarray
    poses: np.ndarray
    widths: np.ndarray

    def __post_init__(self):
        times, poses, widths = _frozen(self.times), _frozen(self.poses), _frozen(self.widths)
        _check_times(times, 'pose stream', minimum=1)  # one sample suffices for causal state, not for resample
        _check_poses(poses, 'pose stream')
        if len(poses) != len(times) or widths.shape != times.shape:
            raise ValueError('pose stream: poses and widths must match times')
        if not np.isfinite(widths).all() or (widths < 0).any():
            raise ValueError('pose stream: widths must be finite and non-negative')
        object.__setattr__(self, 'times', times)
        object.__setattr__(self, 'poses', poses)
        object.__setattr__(self, 'widths', widths)


def resample(stream, times):
    """Linear positions and widths, SLERP rotations. Refuses to extrapolate past the samples."""
    times = np.asarray(times, dtype=float)
    if len(stream.times) < 2:
        raise ValueError('resampling needs at least two pose samples')
    start, end = float(stream.times[0]), float(stream.times[-1])
    if times.size and (times.min() < start - _TIME_EPS or times.max() > end + _TIME_EPS):
        raise ValueError(f'refusing to extrapolate outside the recorded span [{start:.6f}, {end:.6f}] s')
    clipped = np.clip(times, start, end)
    poses = np.zeros(times.shape + (4, 4))
    # scipy's Cython memoryviews reject read-only buffers, so hand it writable copies.
    rotations = Rotation.from_matrix(np.array(stream.poses[:, :3, :3]))
    poses[..., :3, :3] = Slerp(np.array(stream.times), rotations)(clipped).as_matrix()
    for axis in range(3):
        poses[..., axis, 3] = np.interp(clipped, stream.times, stream.poses[:, axis, 3])
    poses[..., 3, 3] = 1.0
    return poses, np.interp(clipped, stream.times, stream.widths)


def interpolation_gaps(stream, times):
    """Length of the sample interval each time is interpolated across (0 on a sample, inf outside)."""
    times = np.asarray(times, dtype=float)
    after = np.searchsorted(stream.times, times - _TIME_EPS, side='left')
    gaps = np.full(times.shape, np.inf)
    inside = (after > 0) & (after < len(stream.times))
    gaps[inside] = stream.times[after[inside]] - stream.times[after[inside] - 1]
    on_sample = (after < len(stream.times)) & (np.abs(stream.times[np.minimum(after, len(stream.times) - 1)] - times) <= _TIME_EPS)
    gaps[on_sample] = 0.0
    return gaps


def latest_index(stream, times, *, max_age_s):
    """Index of the latest sample at or before each time and at most `max_age_s` old (-1 if none)."""
    times = np.asarray(times, dtype=float)
    index = np.searchsorted(stream.times, times + _TIME_EPS, side='right') - 1
    ok = index >= 0
    ok &= times - stream.times[np.maximum(index, 0)] <= max_age_s + _TIME_EPS
    return np.where(ok, index, -1)


def causal_state(stream, times, *, max_age_s=STATE_MAX_AGE_S):
    """Latest measured pose and width at or before each time; raises if a sample is missing or too old."""
    index = latest_index(stream, times, max_age_s=max_age_s)
    if (index < 0).any():
        bad = float(np.asarray(times, dtype=float)[np.argmax(index < 0)])
        raise ValueError(f'no pose sample at or before t={bad:.6f} s within {max_age_s} s')
    return stream.poses[index].copy(), stream.widths[index].copy()


@dataclass(frozen=True)
class CameraStream:
    """RGB frames (F, H, W, 3) uint8 with exposure `times` and `receive_times` on one clock domain."""

    times: np.ndarray
    frames: np.ndarray
    receive_times: np.ndarray
    clock_domain: str = CLOCK_DOMAIN

    def __post_init__(self):
        frames = np.asarray(self.frames)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError('camera stream: frames must be (F, H, W, 3) uint8')
        times, received = _frozen(self.times), _frozen(self.receive_times)
        _check_times(times, 'camera stream', minimum=1)
        if len(times) != len(frames) or received.shape != times.shape:
            raise ValueError('camera stream: one exposure and one receive time per frame')
        if not np.isfinite(received).all() or (received < times - _TIME_EPS).any() or (np.diff(received) < 0).any():
            raise ValueError('camera stream: receive times must be finite, non-decreasing and not before exposure')
        if self.clock_domain != CLOCK_DOMAIN:
            raise ValueError(f'camera stream: clock domain must be {CLOCK_DOMAIN!r}')
        object.__setattr__(self, 'times', times)
        object.__setattr__(self, 'receive_times', received)
        object.__setattr__(self, 'frames', _frozen(frames, np.uint8))


def align_frames(stream, times, max_age_s=IMAGE_MAX_AGE_S):
    """Index of the latest frame already received at each time whose exposure is at most max_age_s old.

    Returns (indices, available); unavailable entries have index -1 and must be masked, never
    filled with a later frame or a synthesized view.
    """
    times = np.asarray(times, dtype=float)
    if stream is None:
        return np.full(times.shape, -1), np.zeros(times.shape, dtype=bool)
    index = np.searchsorted(stream.receive_times, times + _TIME_EPS, side='right') - 1
    available = index >= 0
    age = times - stream.times[np.maximum(index, 0)]
    available &= age <= max_age_s + _TIME_EPS
    return np.where(available, index, -1), available


def canonical_arms(arms):
    """Distinct arms in ARMS order; the conditioning and the trained-combination record depend on it."""
    arms = tuple(arms)
    if len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError(f'arms must be distinct members of {ARMS}, got {arms}')
    return tuple(arm for arm in ARMS if arm in arms)


def _check_pose_provenance(provenance, name):
    if provenance is None or provenance.get('source') != 'observed_geometry' \
            or provenance.get('pose_source') not in POSE_SOURCES:
        raise ValueError(f'{name}: fragment pose needs observed_geometry provenance with a pose source in {POSE_SOURCES}')


@dataclass(frozen=True)
class SkillSegment:
    """One executed skill request: the same conditioning the S1 runtime receives for it.

    `target_pose` is world_T_fragment_goal; `fragment_pose` is the observed world_T_fragment the
    request carried. Both need provenance; simulator truth must be labelled as oracle.
    """

    start: float
    end: float
    skill_id: str
    active_arms: tuple[str, ...]
    request_id: str
    fragment_id: str | None = None
    target_pose: np.ndarray | None = None
    target_provenance: Mapping[str, object] | None = None
    fragment_pose: np.ndarray | None = None
    fragment_pose_provenance: Mapping[str, object] | None = None

    def __post_init__(self):
        if not np.isfinite([self.start, self.end]).all() or not self.start < self.end:
            raise ValueError('segment: need finite start < end')
        if self.skill_id not in SKILLS:
            raise ValueError(f'segment: unknown skill {self.skill_id!r}')
        spec = SKILLS[self.skill_id]
        arms = canonical_arms(self.active_arms)
        low, high = spec.arm_count
        if not low <= len(arms) <= high:
            raise ValueError(f'segment: {self.skill_id} needs {low}..{high} distinct arms')
        if spec.needs_fragment != (self.fragment_id is not None):
            raise ValueError(f'segment: {self.skill_id} {"needs" if spec.needs_fragment else "takes no"} fragment_id')
        if spec.needs_target != (self.target_pose is not None):
            raise ValueError(f'segment: {self.skill_id} {"needs" if spec.needs_target else "takes no"} target pose')
        if self.target_pose is not None:
            source = (self.target_provenance or {}).get('source')
            if source not in TARGET_SOURCES + (ORACLE_TARGET_SOURCE,):
                raise ValueError(f'segment: target provenance source must be one of {TARGET_SOURCES} '
                                 f'or {ORACLE_TARGET_SOURCE!r}, got {source!r}')
            object.__setattr__(self, 'target_pose', self._pose(self.target_pose, 'segment target'))
        elif self.target_provenance is not None:
            raise ValueError('segment: target provenance without a target pose')
        if spec.needs_fragment_pose and self.fragment_pose is None:
            raise ValueError(f'segment: {self.skill_id} needs the observed fragment pose')
        if self.fragment_pose is not None:
            if self.fragment_id is None:
                raise ValueError(f'segment: {self.skill_id} takes no fragment pose')
            _check_pose_provenance(self.fragment_pose_provenance, 'segment')
            object.__setattr__(self, 'fragment_pose', self._pose(self.fragment_pose, 'segment fragment pose'))
        elif self.fragment_pose_provenance is not None:
            raise ValueError('segment: fragment pose provenance without a fragment pose')
        for name in ('target_provenance', 'fragment_pose_provenance'):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(self, 'active_arms', arms)
        object.__setattr__(self, 'start', float(self.start))
        object.__setattr__(self, 'end', float(self.end))

    @staticmethod
    def _pose(value, name):
        pose = _frozen(value)
        _check_poses(pose[None], name)
        return pose

    @property
    def oracle(self):
        """True when the conditioning comes from simulator ground truth (an explicit oracle ablation)."""
        target = (self.target_provenance or {}).get('source') == ORACLE_TARGET_SOURCE
        pose = (self.fragment_pose_provenance or {}).get('pose_source') == ORACLE_POSE_SOURCE
        return target or pose


def _arm_map(streams, name):
    if streams is None:
        return None
    if set(streams) != set(ARMS) or not all(isinstance(s, PoseStream) for s in streams.values()):
        raise ValueError(f'{name}: need a PoseStream for each of {ARMS}')
    return MappingProxyType({arm: streams[arm] for arm in ARMS})


@dataclass(frozen=True)
class Episode:
    episode_id: str
    source: str
    fragment_ids: tuple[str, ...]
    geometry_hash: str
    calibration_version: str
    tcp: Mapping[str, PoseStream]
    segments: tuple[SkillSegment, ...]
    cameras: Mapping[str, CameraStream] = field(default_factory=dict)
    commanded: Mapping[str, PoseStream] | None = None
    accepted: Mapping[str, PoseStream] | None = None
    seed: int | None = None
    task_instruction: str = ''
    success: bool | None = None
    failure_reason: str | None = None
    initial_condition_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError('episode: missing episode_id')
        if self.source not in SOURCES:
            raise ValueError(f'episode: source must be one of {SOURCES}')
        object.__setattr__(self, 'tcp', _arm_map(self.tcp, 'episode.tcp'))
        object.__setattr__(self, 'commanded', _arm_map(self.commanded, 'episode.commanded'))
        object.__setattr__(self, 'accepted', _arm_map(self.accepted, 'episode.accepted'))
        unknown = set(self.cameras) - set(CAMERAS)
        if unknown or not all(isinstance(c, CameraStream) for c in self.cameras.values()):
            raise ValueError(f'episode: unknown cameras {sorted(unknown)} or non-CameraStream values')
        if len({c.frames.shape[1:] for c in self.cameras.values()}) > 1:
            raise ValueError('episode: all cameras must share one frame shape')
        object.__setattr__(self, 'cameras', MappingProxyType(dict(self.cameras)))
        fragments = tuple(self.fragment_ids)
        object.__setattr__(self, 'fragment_ids', fragments)
        segments = tuple(self.segments)
        if not segments or not all(isinstance(s, SkillSegment) for s in segments):
            raise ValueError('episode: need at least one SkillSegment')
        start, end = self.span()
        for previous, current in zip(segments, segments[1:]):
            if current.start < previous.end - _TIME_EPS:
                raise ValueError('episode: segments must be sorted and non-overlapping')
        if segments[0].start < start - _TIME_EPS or segments[-1].end > end + _TIME_EPS:
            raise ValueError('episode: segments must lie inside the recorded TCP span')
        for segment in segments:
            if segment.fragment_id is not None and segment.fragment_id not in fragments:
                raise ValueError(f'episode: segment fragment {segment.fragment_id!r} not in fragment_ids')
        object.__setattr__(self, 'segments', segments)

    def span(self, streams=None):
        streams = self.tcp if streams is None else streams
        return (max(float(s.times[0]) for s in streams.values()), min(float(s.times[-1]) for s in streams.values()))


def buffers_until(episode, time):
    """What a runtime buffer holds at `time`: TCP samples at or before it, frames received by then."""
    tcp = {}
    for arm in ARMS:
        stream = episode.tcp[arm]
        keep = stream.times <= time + _TIME_EPS
        if not keep.any():
            raise ValueError(f'{arm}: no TCP sample at or before t={time:.6f} s')
        tcp[arm] = PoseStream(stream.times[keep], stream.poses[keep], stream.widths[keep])
    cameras = {}
    for name, camera in episode.cameras.items():
        keep = camera.receive_times <= time + _TIME_EPS
        if keep.any():
            cameras[name] = CameraStream(camera.times[keep], camera.frames[keep], camera.receive_times[keep],
                                         clock_domain=camera.clock_domain)
    return tcp, cameras


def initial_condition_key(episode):
    """Grouping key for splits: retries of one initial layout must land in the same split."""
    if episode.initial_condition_id:
        return f'ic:{episode.initial_condition_id}'
    if episode.seed is not None:
        return f'geometry_seed:{episode.geometry_hash}:{episode.seed}'
    return f'episode:{episode.episode_id}'


# ---------------------------------------------------------------- persistence

def _stream_arrays(prefix, streams, arrays):
    for arm in ARMS:
        arrays[f'{prefix}__{arm}__times'] = streams[arm].times
        arrays[f'{prefix}__{arm}__poses'] = streams[arm].poses
        arrays[f'{prefix}__{arm}__widths'] = streams[arm].widths


def _pose_or_none(value):
    return None if value is None else np.asarray(value).tolist()


def _dict_or_none(value):
    return None if value is None else dict(value)


def _segment_record(segment):
    return {'start': segment.start, 'end': segment.end, 'skill_id': segment.skill_id,
            'active_arms': list(segment.active_arms), 'request_id': segment.request_id,
            'fragment_id': segment.fragment_id, 'target_pose': _pose_or_none(segment.target_pose),
            'target_provenance': _dict_or_none(segment.target_provenance),
            'fragment_pose': _pose_or_none(segment.fragment_pose),
            'fragment_pose_provenance': _dict_or_none(segment.fragment_pose_provenance)}


def save_episode(episode, directory, *, privileged=None):
    """Write meta.json + streams.npz + camera_<name>.npz (+ privileged.npz). Refuses to overwrite."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {}
    _stream_arrays('tcp', episode.tcp, arrays)
    for name, streams in (('commanded', episode.commanded), ('accepted', episode.accepted)):
        if streams is not None:
            _stream_arrays(name, streams, arrays)
    np.savez_compressed(directory / 'streams.npz', **arrays)
    for name, camera in episode.cameras.items():
        np.savez_compressed(directory / f'camera_{name}.npz', times=camera.times, frames=camera.frames,
                            receive_times=camera.receive_times)
    meta = {'schema_version': SCHEMA_VERSION, 'episode_id': episode.episode_id, 'source': episode.source,
            'fragment_ids': list(episode.fragment_ids), 'geometry_hash': episode.geometry_hash,
            'calibration_version': episode.calibration_version, 'seed': episode.seed,
            'initial_condition_id': episode.initial_condition_id,
            'task_instruction': episode.task_instruction, 'success': episode.success,
            'failure_reason': episode.failure_reason, 'cameras': sorted(episode.cameras),
            'camera_clock_domains': {name: c.clock_domain for name, c in episode.cameras.items()},
            'has_commanded': episode.commanded is not None, 'has_accepted': episode.accepted is not None,
            'segments': [_segment_record(s) for s in episode.segments],
            'privileged_file': None}
    if privileged is not None:
        np.savez_compressed(directory / 'privileged.npz', **privileged)
        meta['privileged_file'] = 'privileged.npz'
    (directory / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    return directory


def _load_streams(arrays, prefix):
    return {arm: PoseStream(arrays[f'{prefix}__{arm}__times'], arrays[f'{prefix}__{arm}__poses'],
                            arrays[f'{prefix}__{arm}__widths']) for arm in ARMS}


def _load_segment(record):
    arrays = {key: None if record[key] is None else np.asarray(record[key]) for key in ('target_pose', 'fragment_pose')}
    return SkillSegment(start=record['start'], end=record['end'], skill_id=record['skill_id'],
                        active_arms=tuple(record['active_arms']), request_id=record['request_id'],
                        fragment_id=record['fragment_id'], target_provenance=record['target_provenance'],
                        fragment_pose_provenance=record['fragment_pose_provenance'], **arrays)


def load_episode(directory):
    """Load everything a policy may use. privileged.npz is deliberately never opened here."""
    directory = Path(directory)
    meta = json.loads((directory / 'meta.json').read_text())
    if meta.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'unsupported episode schema {meta.get("schema_version")!r} (need {SCHEMA_VERSION})')
    with np.load(directory / 'streams.npz', allow_pickle=False) as arrays:
        tcp = _load_streams(arrays, 'tcp')
        commanded = _load_streams(arrays, 'commanded') if meta['has_commanded'] else None
        accepted = _load_streams(arrays, 'accepted') if meta['has_accepted'] else None
    cameras = {}
    for name in meta['cameras']:
        with np.load(directory / f'camera_{name}.npz', allow_pickle=False) as data:
            cameras[name] = CameraStream(data['times'], data['frames'], data['receive_times'],
                                         clock_domain=meta['camera_clock_domains'][name])
    return Episode(episode_id=meta['episode_id'], source=meta['source'], fragment_ids=tuple(meta['fragment_ids']),
                   geometry_hash=meta['geometry_hash'], calibration_version=meta['calibration_version'],
                   tcp=tcp, segments=tuple(_load_segment(s) for s in meta['segments']), cameras=cameras,
                   commanded=commanded, accepted=accepted, seed=meta['seed'],
                   task_instruction=meta['task_instruction'], success=meta['success'],
                   failure_reason=meta['failure_reason'], initial_condition_id=meta['initial_condition_id'])


def load_privileged(directory):
    """Explicit opt-in for experts and offline evaluation; never part of S1/S2 observations."""
    path = Path(directory) / 'privileged.npz'
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}
