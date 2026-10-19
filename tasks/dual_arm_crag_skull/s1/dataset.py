"""Windowed S1 samples, the train/val/test split and the normalizer. numpy only.

A sample at 10 Hz grid step i observes steps i-1 and i and supervises the 16 future steps i+1..i+16.
Observations are strictly causal (latest measured sample at or before each step) and may reach
back into the previous skill segment, exactly as the runtime sees the robot. Supervision never
crosses the segment end: future steps past it hold the segment's last pose and width (Diffusion
Policy's pad_after), so the final part of every skill is observed in training. Each future pose
is expressed relative to that hand's TCP at step i (the anchor) with `contracts.actions.hand_vector`.

Conditioning (layout version CONDITIONING_VERSION): skill index, active-arm mask in ARMS order,
the goal pose and the observed fragment pose, both relative to the anchor of the first active arm
(each with a presence mask). Anchors and other absolute world poses never reach the network.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..contracts.actions import ACTION_DIM, ARMS, hand_vector
from ..contracts.skills import SKILL_IDS, SKILLS
from .episode import (CAMERAS, IMAGE_MAX_AGE_S, MAX_GAP_S, OBS_STEPS, POLICY_HZ, PRED_STEPS, SCHEMA_VERSION,
                      STATE_MAX_AGE_S, SUPERVISION, _TIME_EPS, _frozen, align_frames, canonical_arms, causal_state,
                      initial_condition_key, interpolation_gaps, invert, latest_index, pose9, resample)

CONDITIONING_VERSION = 2
POSE9 = 9
# Per observation step: each hand's pose relative to its own current pose, both widths, left_T_right.
LOWDIM_DIM = len(ARMS) * POSE9 + len(ARMS) + POSE9
TARGET_DIM = FRAGMENT_DIM = POSE9
ARM_MASK_DIM = len(ARMS)
# Keys of a sample that may reach the network.
POLICY_INPUTS = ('images', 'image_mask', 'lowdim', 'skill_index', 'arm_mask', 'target', 'target_mask',
                 'fragment', 'fragment_mask')
LAYOUT = {'conditioning_version': CONDITIONING_VERSION, 'skill_ids': list(SKILL_IDS), 'arms': list(ARMS),
          'lowdim_dim': LOWDIM_DIM, 'policy_inputs': list(POLICY_INPUTS), 'action_dim': ACTION_DIM}
MIN_WINDOWS_PER_COMBINATION = 16


def _relative(anchor, pose):
    if pose is None:
        return np.zeros(POSE9), np.zeros(1)
    return pose9(invert(anchor) @ np.asarray(pose, dtype=float)), np.ones(1)


def encode_observation(history_poses, history_widths, *, skill_id, active_arms, target_pose, fragment_pose):
    """Low-dim observation and conditioning shared by training windows and live inference.

    history_poses: {arm: (S, 4, 4) world_T_tcp}, oldest first, last = observation time.
    """
    arms = canonical_arms(active_arms)
    anchors = {arm: history_poses[arm][-1] for arm in ARMS}
    own = [pose9(invert(anchors[arm]) @ history_poses[arm]) for arm in ARMS]
    between = pose9(invert(history_poses['left']) @ history_poses['right'])
    widths = np.stack([history_widths[arm] for arm in ARMS], axis=-1)
    reference = anchors[arms[0]] if arms else anchors[ARMS[0]]
    target, target_mask = _relative(reference, target_pose)
    fragment, fragment_mask = _relative(reference, fragment_pose)
    return {'lowdim': np.concatenate([*own, widths, between], axis=-1), 'skill_index': SKILL_IDS.index(skill_id),
            'arm_mask': np.array([float(arm in arms) for arm in ARMS]), 'target': target, 'target_mask': target_mask,
            'fragment': fragment, 'fragment_mask': fragment_mask,
            'anchors': np.stack([anchors[arm] for arm in ARMS]),
            'widths_now': np.array([history_widths[arm][-1] for arm in ARMS]), 'active_arms': arms}


def _image_history(cameras, camera_names, times, image_shape, max_age_s):
    images = np.zeros((len(times), len(camera_names)) + tuple(image_shape), dtype=np.uint8)
    mask = np.zeros((len(times), len(camera_names)), dtype=bool)
    for c, name in enumerate(camera_names):
        stream = cameras.get(name)
        indices, available = align_frames(stream, times, max_age_s)
        for h in np.flatnonzero(available):
            images[h, c] = stream.frames[indices[h]]
            mask[h, c] = True
    return images, mask


def history_times(observation_time, *, hz=POLICY_HZ, obs_steps=OBS_STEPS):
    return float(observation_time) - np.arange(obs_steps - 1, -1, -1) / hz


def live_sample(tcp, cameras, observation_time, request, *, hz=POLICY_HZ, obs_steps=OBS_STEPS, camera_names=CAMERAS,
                image_shape, image_max_age_s=IMAGE_MAX_AGE_S, state_max_age_s=STATE_MAX_AGE_S):
    """Observation for one inference call from buffered streams and a validated SkillRequest.

    Same causal state sampling, receive-time frame alignment and encoding as the training windows.
    Buffers need only samples at or before the observation time. History steps before the buffers'
    shared start repeat the state at that start, exactly as training windows pad at the recording start.
    """
    start = max(float(tcp[arm].times[0]) for arm in ARMS)
    times = np.maximum(history_times(observation_time, hz=hz, obs_steps=obs_steps), start)
    history = {arm: causal_state(tcp[arm], times, max_age_s=state_max_age_s) for arm in ARMS}
    sample = encode_observation({arm: history[arm][0] for arm in ARMS}, {arm: history[arm][1] for arm in ARMS},
                                skill_id=request.skill_id, active_arms=request.active_arms,
                                target_pose=request.target_pose, fragment_pose=request.fragment_pose)
    sample['images'], sample['image_mask'] = _image_history(cameras, camera_names, times, image_shape, image_max_age_s)
    sample['observation_time'] = float(observation_time)
    return sample


def policy_grid(episode, *, hz=POLICY_HZ):
    """10 Hz grid over the shared TCP span. The grid never depends on the supervision source; steps a
    commanded stream does not cover are invalid supervision (see PolicyDataset)."""
    start, end = episode.span()
    count = int(np.floor((end - start) * hz + 1e-6)) + 1
    return start + np.arange(max(count, 0)) / hz


def segment_ranges(grid, segments):
    """Contiguous grid index range [first, last] inside each segment (half-open, last segment closed)."""
    ranges = []
    for index, segment in enumerate(segments):
        final = index == len(segments) - 1
        inside = (grid >= segment.start - _TIME_EPS) & (
            (grid < segment.end - _TIME_EPS) | (final & (grid <= segment.end + _TIME_EPS)))
        steps = np.flatnonzero(inside)
        ranges.append((int(steps[0]), int(steps[-1])) if len(steps) else None)
    return ranges


# ---------------------------------------------------------------- splits

def split_episodes(episodes, *, fractions=(0.8, 0.1, 0.1), seed=0):
    """Deterministic split by initial condition (see episode.initial_condition_key), not by episode.

    Retries of one layout therefore never land in both train and test. Returns episode ids per split.
    """
    episodes = list(episodes)
    ids = [e.episode_id for e in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate episode ids')
    if len(fractions) != 3 or min(fractions) < 0 or not np.isclose(sum(fractions), 1):
        raise ValueError('fractions must be three non-negative numbers summing to one')
    groups = {}
    for episode in episodes:
        groups.setdefault(initial_condition_key(episode), []).append(episode.episode_id)
    order = sorted(groups, key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest())
    n = len(order)
    n_val = max(1, round(n * fractions[1])) if fractions[1] > 0 and n >= 3 else round(n * fractions[1])
    n_test = max(1, round(n * fractions[2])) if fractions[2] > 0 and n >= 3 else round(n * fractions[2])
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError('split leaves no training initial conditions')
    parts = {'train': order[:n_train], 'val': order[n_train:n_train + n_val], 'test': order[n_train + n_val:]}
    return {name: tuple(sorted(i for key in keys for i in groups[key])) for name, keys in parts.items()}


# ---------------------------------------------------------------- windows

@dataclass(frozen=True)
class WindowRef:
    episode: int
    segment: int
    step: int


class PolicyDataset:
    """Windows over a fixed set of episodes; samples are plain numpy dictionaries.

    `supervision` selects what the future steps are: the achieved (measured) TCP trajectory or the
    commanded targets (stamped at their intended reach time). One dataset never mixes the two.
    Only episodes with success is True are used unless `include_failures`; oracle conditioning
    (ground-truth goals or simulator fragment poses) raises unless `allow_oracle`. Windows whose
    supervision interpolates across a pose-stream gap longer than `max_gap_s`, or whose observation
    state is older than `state_max_age_s`, are dropped and counted in `stats`.
    """

    def __init__(self, episodes, *, supervision='achieved', hz=POLICY_HZ, obs_steps=OBS_STEPS, pred_steps=PRED_STEPS,
                 pad_after=None, image_max_age_s=IMAGE_MAX_AGE_S, state_max_age_s=STATE_MAX_AGE_S, max_gap_s=MAX_GAP_S,
                 cameras=CAMERAS, image_shape=None, include_failures=False, allow_oracle=False):
        if supervision not in SUPERVISION:
            raise ValueError(f'supervision must be one of {SUPERVISION}')
        pad_after = pred_steps - 1 if pad_after is None else int(pad_after)
        if not 0 <= pad_after <= pred_steps:
            raise ValueError('pad_after must be in 0..pred_steps')
        given = tuple(episodes)
        kept = tuple(e for e in given if include_failures or e.success is True)
        self.stats = Counter(episodes_given=len(given), episodes_used=len(kept),
                             excluded_failed=sum(e.success is False for e in given) if not include_failures else 0,
                             excluded_unknown_outcome=sum(e.success is None for e in given) if not include_failures else 0)
        oracle = [e.episode_id for e in kept if any(s.oracle for s in e.segments)]
        if oracle and not allow_oracle:
            raise ValueError(f'oracle conditioning in {oracle[:5]}; set allow_oracle for an explicit oracle ablation')
        if supervision == 'commanded':
            missing = [e.episode_id for e in kept if e.commanded is None]
            if missing:
                raise ValueError(f'commanded supervision but episodes lack commanded streams: {missing[:5]}')
        shapes = {c.frames.shape[1:] for e in kept for c in e.cameras.values()}
        if image_shape is not None:
            shapes.add(tuple(image_shape))
        if len(shapes) != 1:
            raise ValueError(f'need exactly one image shape across episodes, got {sorted(shapes)}')
        self.image_shape = shapes.pop()
        self.episodes, self.supervision, self.hz = kept, supervision, hz
        self.obs_steps, self.pred_steps, self.pad_after = obs_steps, pred_steps, pad_after
        self.cameras, self.image_max_age_s = tuple(cameras), image_max_age_s
        self.state_max_age_s, self.max_gap_s = state_max_age_s, max_gap_s
        self._cache = [self._prepare(e) for e in kept]
        self.windows = tuple(self._windows())

    def _prepare(self, episode):
        grid = policy_grid(episode, hz=self.hz)
        source = episode.tcp if self.supervision == 'achieved' else episode.commanded
        state_index = {arm: latest_index(episode.tcp[arm], grid, max_age_s=self.state_max_age_s) for arm in ARMS}
        # Outside a source's span the gap is infinite, so those steps are never used as supervision.
        supervised = {arm: resample(source[arm], np.clip(grid, source[arm].times[0], source[arm].times[-1]))
                      for arm in ARMS}
        gap_ok = np.all([interpolation_gaps(source[arm], grid) <= self.max_gap_s + _TIME_EPS for arm in ARMS], axis=0)
        state_ok = np.all([state_index[arm] >= 0 for arm in ARMS], axis=0)
        return {'grid': grid, 'state_index': state_index, 'supervised': supervised, 'gap_ok': gap_ok,
                'state_ok': state_ok, 'ranges': segment_ranges(grid, episode.segments)}

    def _steps(self, step, first, last):
        history = np.maximum(0, step - np.arange(self.obs_steps - 1, -1, -1))  # pads only at the recording start
        future = np.minimum(last, step + np.arange(1, self.pred_steps + 1))     # holds the segment's last step
        return history, future

    def _windows(self):
        for index, cache in enumerate(self._cache):
            for segment, bounds in enumerate(cache['ranges']):
                if bounds is None or not SKILLS[self.episodes[index].segments[segment].skill_id].moves:
                    continue
                first, last = bounds
                for step in range(first, max(first, last - self.pred_steps + self.pad_after) + 1):
                    history, future = self._steps(step, first, last)
                    if not cache['state_ok'][history].all():
                        self.stats['windows_dropped_stale_state'] += 1
                    elif not cache['gap_ok'][np.unique(future)].all():
                        self.stats['windows_dropped_gap'] += 1
                    else:
                        self.stats['windows'] += 1
                        yield WindowRef(index, segment, step)

    def __len__(self):
        return len(self.windows)

    def lowdim_action_target(self, ref):
        """Everything but images for one window (cheap; used for normalizer fitting)."""
        cache, episode = self._cache[ref.episode], self.episodes[ref.episode]
        segment = episode.segments[ref.segment]
        first, last = cache['ranges'][ref.segment]
        history, future = self._steps(ref.step, first, last)
        state = {arm: (episode.tcp[arm].poses[cache['state_index'][arm][history]],
                       episode.tcp[arm].widths[cache['state_index'][arm][history]]) for arm in ARMS}
        sample = encode_observation({arm: state[arm][0] for arm in ARMS}, {arm: state[arm][1] for arm in ARMS},
                                    skill_id=segment.skill_id, active_arms=segment.active_arms,
                                    target_pose=segment.target_pose, fragment_pose=segment.fragment_pose)
        anchors = {arm: sample['anchors'][i] for i, arm in enumerate(ARMS)}
        sample['action'] = np.concatenate([hand_vector(invert(anchors[arm]) @ cache['supervised'][arm][0][future],
                                                       cache['supervised'][arm][1][future]) for arm in ARMS], axis=-1)
        if sample['action'].shape != (self.pred_steps, ACTION_DIM):
            raise AssertionError(f'action window shape {sample["action"].shape}')
        sample.update(observation_time=float(cache['grid'][ref.step]),
                      future_times=cache['grid'][ref.step] + np.arange(1, self.pred_steps + 1) / self.hz,
                      padded_steps=int(np.sum(ref.step + np.arange(1, self.pred_steps + 1) > last)),
                      episode_id=episode.episode_id, segment_index=ref.segment, step=ref.step,
                      supervision=self.supervision)
        return sample

    def __getitem__(self, index):
        ref = self.windows[index]
        sample = self.lowdim_action_target(ref)
        times = self._cache[ref.episode]['grid'][self._steps(ref.step, *self._cache[ref.episode]['ranges'][ref.segment])[0]]
        sample['images'], sample['image_mask'] = _image_history(self.episodes[ref.episode].cameras, self.cameras, times,
                                                                self.image_shape, self.image_max_age_s)
        return sample


def trained_combinations(dataset, *, min_windows=MIN_WINDOWS_PER_COMBINATION):
    """(skill_id, arms) combinations with at least `min_windows` training windows, and all counts."""
    counts = Counter()
    for ref in dataset.windows:
        segment = dataset.episodes[ref.episode].segments[ref.segment]
        counts[(segment.skill_id, segment.active_arms)] += 1
    kept = sorted(key for key, count in counts.items() if count >= min_windows)
    return [[skill, list(arms)] for skill, arms in kept], {f'{s}:{"+".join(a)}': c for (s, a), c in sorted(counts.items())}


# ---------------------------------------------------------------- normalization

ROTATION_DIMS = {
    'action': tuple(range(3, 9)) + tuple(range(13, 19)),
    'lowdim': tuple(range(3, 9)) + tuple(range(12, 18)) + tuple(range(23, 29)),
    'target': tuple(range(3, 9)),
    'fragment': tuple(range(3, 9)),
}
FIELD_DIMS = {'action': ACTION_DIM, 'lowdim': LOWDIM_DIM, 'target': TARGET_DIM, 'fragment': FRAGMENT_DIM}


@dataclass(frozen=True)
class Normalizer:
    """Affine map to [-1, 1]: min-max for positions and widths, identity for rotation6 entries."""

    scale: Mapping[str, np.ndarray]
    offset: Mapping[str, np.ndarray]
    supervision: str
    fitted_on: tuple[str, ...]

    def __post_init__(self):
        if set(self.scale) != set(FIELD_DIMS) or set(self.offset) != set(FIELD_DIMS):
            raise ValueError(f'normalizer needs exactly the fields {sorted(FIELD_DIMS)}')
        object.__setattr__(self, 'scale', MappingProxyType({k: _frozen(v) for k, v in self.scale.items()}))
        object.__setattr__(self, 'offset', MappingProxyType({k: _frozen(v) for k, v in self.offset.items()}))
        object.__setattr__(self, 'fitted_on', tuple(self.fitted_on))

    def __deepcopy__(self, memo):
        return self  # immutable (read-only arrays in read-only mappings); also, mappingproxy cannot be copied

    @classmethod
    def fit(cls, dataset, *, min_span=0.01):
        """Fit on one dataset, which must be the training split only.

        Every non-rotation entry is in metres. A span below `min_span` (e.g. a still arm whose
        range is measurement noise) is widened to `min_span` around its centre.
        """
        if len(dataset) == 0:
            raise ValueError('cannot fit a normalizer on an empty dataset')
        values = {name: [] for name in FIELD_DIMS}
        for ref in dataset.windows:
            item = dataset.lowdim_action_target(ref)
            values['action'].append(item['action'])
            values['lowdim'].append(item['lowdim'])
            for name in ('target', 'fragment'):
                if item[f'{name}_mask'][0] > 0:
                    values[name].append(item[name][None])
        scale, offset = {}, {}
        for name, dim in FIELD_DIMS.items():
            s, o = np.ones(dim), np.zeros(dim)
            if values[name]:
                flat = np.concatenate([v.reshape(-1, dim) for v in values[name]])
                low, high = flat.min(axis=0), flat.max(axis=0)
                center, span = (high + low) / 2.0, np.maximum(high - low, min_span)
                s = 2.0 / span
                o = -center * s
            rotation = list(ROTATION_DIMS[name])
            s[rotation], o[rotation] = 1.0, 0.0
            scale[name], offset[name] = s, o
        episodes = sorted({e.episode_id for e in dataset.episodes})
        return cls(scale, offset, dataset.supervision, tuple(episodes))

    def normalize(self, name, value):
        return np.asarray(value, dtype=float) * self.scale[name] + self.offset[name]

    def unnormalize(self, name, value):
        return (np.asarray(value, dtype=float) - self.offset[name]) / self.scale[name]

    def to_dict(self):
        return {'schema_version': SCHEMA_VERSION, 'supervision': self.supervision, 'fitted_on': list(self.fitted_on),
                'scale': {k: v.tolist() for k, v in self.scale.items()},
                'offset': {k: v.tolist() for k, v in self.offset.items()}}

    @classmethod
    def from_dict(cls, data):
        if data.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('unsupported normalizer schema')
        return cls({k: np.asarray(v) for k, v in data['scale'].items()},
                   {k: np.asarray(v) for k, v in data['offset'].items()}, data['supervision'], tuple(data['fitted_on']))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + '\n')

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
