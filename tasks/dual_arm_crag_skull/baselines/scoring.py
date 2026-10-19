"""One score for every method: where each fragment ended up relative to its goal, plus the run's cost.

Fragments are judged relative to the anchor's final pose, so a complete assembly that was nudged as a
whole still counts as assembled; the anchor itself must stay within the same tolerance of its goal. A
fragment whose final pose is missing or invalid (lost from the scene, never reported, not a finite
rigid transform) fails. The poses come from the simulator's privileged state after the method has
finished and both grippers have released for the task's settle time (`settled`); no method sees this
function's input while it runs. Scoring needs the true assembly, so a task whose goals came from a
GARF candidate is refused until it carries the oracle goals separately.

Held-out seeds are used once per method, variant and task: `claim_heldout` claims one atomically in
the canonical claims directory before the trial touches anything, and the claim stays when the trial
crashes; `refuse_repeats` checks a records file as well.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from ..perception import rigid_transform
from ..workspace import repo_path
from .task import SPLITS, canonical_protocol, canonical_split, split_of

METHODS = ('s2_s1_s0', 'dexagent', 'aspire', 'scripted')
STATUSES = ('completed', 'blocked', 'budget_exhausted', 'timeout', 'rejected', 'error')
CLAIMS_ENV = 'ASSEMBLY_BASELINE_CLAIMS_DIR'


def pose_errors(goal, actual):
    """(translation error in metres, rotation error in degrees) of `actual` against `goal`."""
    goal, actual = rigid_transform(goal), rigid_transform(actual)
    position = float(np.linalg.norm(actual[:3, 3] - goal[:3, 3]))
    rotation = float(np.rad2deg(Rotation.from_matrix(actual[:3, :3] @ goal[:3, :3].T).magnitude()))
    return position, rotation


@dataclass(frozen=True)
class FragmentScore:
    """Errors relative to the anchor (judged) and in the world (reported); inf when the pose is missing."""

    fragment_id: str
    position_error_m: float
    rotation_error_deg: float
    world_position_error_m: float
    world_rotation_error_deg: float
    within_tolerance: bool

    def to_record(self):
        return {key: (None if isinstance(value, float) and math.isinf(value) else value)
                for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class AssemblyScore:
    """The score and what it was judged against (layout, start poses, goal source, tolerances, settling)."""

    task_id: str
    seed: int | None
    split: str | None
    layout_sha256: str
    initial_sha256: str
    goal_source: str
    position_tolerance_m: float
    rotation_tolerance_deg: float
    settled: bool
    anchor_id: str
    anchor_position_error_m: float
    anchor_rotation_error_deg: float
    anchor_within_tolerance: bool
    fragments: tuple
    invalid_ids: tuple
    placed: int
    success: bool

    @property
    def placed_fraction(self):
        return self.placed / len(self.fragments)

    def to_record(self):
        record = {key: value for key, value in self.__dict__.items() if key not in ('fragments', 'invalid_ids')}
        for key in ('anchor_position_error_m', 'anchor_rotation_error_deg'):
            if math.isinf(record[key]):
                record[key] = None
        record['fragments'] = [fragment.to_record() for fragment in self.fragments]
        record['invalid_ids'] = list(self.invalid_ids)
        record['placed_fraction'] = self.placed_fraction
        return record


def _valid_pose(value):
    try:
        return rigid_transform(value)
    except (ValueError, TypeError):
        return None


def score_assembly(task, final_world_T_fragment: Mapping[str, np.ndarray], *, settled):
    """Score final fragment poses (world, simulator truth) against `task`'s goals. `settled` says the poses were read
    after both grippers released and the task's settle time passed (or after the method's own equivalent check); a
    score that was not settled is not a success."""
    if task.goal_source != 'oracle_layout':
        raise ValueError('scoring needs the true assembly: a task with GARF-candidate goals must carry the oracle '
                         'goals separately, which is not implemented')
    if not isinstance(settled, bool):
        raise ValueError('settled must be True or False')
    unknown = set(final_world_T_fragment) - set(task.fragment_ids)
    if unknown:
        raise ValueError(f'final poses for unknown fragments: {sorted(unknown)}')
    poses, invalid = {}, []
    for key in sorted(final_world_T_fragment):
        pose = _valid_pose(final_world_T_fragment[key])
        if pose is None:
            invalid.append(key)
        else:
            poses[key] = pose
    tolerance_m, tolerance_deg = task.position_tolerance_m, task.rotation_tolerance_deg
    anchor_goal = task.fragment(task.anchor_id).goal_world_T_fragment
    anchor_final = poses.get(task.anchor_id)
    if anchor_final is None:
        anchor_errors = (math.inf, math.inf)
    else:
        anchor_errors = pose_errors(anchor_goal, anchor_final)
    anchor_ok = anchor_errors[0] <= tolerance_m and anchor_errors[1] <= tolerance_deg
    # Without an anchor pose, fragments are compared with the anchor's goal (the world errors).
    reference = anchor_goal if anchor_final is None else anchor_final
    scores = []
    for key in task.moving_ids:
        goal = task.fragment(key).goal_world_T_fragment
        actual = poses.get(key)
        if actual is None:
            scores.append(FragmentScore(key, math.inf, math.inf, math.inf, math.inf, False))
            continue
        relative = pose_errors(np.linalg.inv(anchor_goal) @ goal, np.linalg.inv(reference) @ actual)
        world = pose_errors(goal, actual)
        ok = relative[0] <= tolerance_m and relative[1] <= tolerance_deg
        scores.append(FragmentScore(key, *relative, *world, ok))
    placed = sum(score.within_tolerance for score in scores)
    return AssemblyScore(task.task_id, task.seed, task.split, task.layout_sha256, task.initial_sha256(),
                         task.goal_source, tolerance_m, tolerance_deg, settled, task.anchor_id, *anchor_errors,
                         anchor_ok, tuple(scores), tuple(invalid), placed,
                         bool(settled and anchor_ok and placed == len(scores)))


def _optional_number(value, name, integer=False):
    if value is None:
        return
    kinds = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kinds) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be None or a non-negative finite {"integer" if integer else "number"}')


@dataclass(frozen=True)
class RunRecord:
    """One method on one task. `score` is None when no final state exists; that run counts as a failure.

    `cost_usd` and `llm_calls` are what the method spent on model calls (None if it made none);
    `sim_trials` counts simulator episodes the method ran for itself, such as ASPIRE practice trials.
    `backend` names where the trial ran; `test_only` marks plumbing runs (physics-free double, mock model) that are
    never baseline results. The split is the canonical band of the seed.
    """

    method: str
    variant: str
    task_id: str
    seed: int | None
    split: str | None
    status: str
    score: AssemblyScore | None
    cost_usd: float | None = None
    llm_calls: int | None = None
    wall_s: float | None = None
    sim_trials: int | None = None
    notes: str = ''
    artifacts: Mapping[str, str] = field(default_factory=dict)
    _: KW_ONLY
    backend: str
    test_only: bool

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f'method must be one of {METHODS}')
        if self.status not in STATUSES:
            raise ValueError(f'status must be one of {STATUSES}')
        if (self.seed is None) != (self.split is None):
            raise ValueError('seed and split are set together')
        if self.seed is not None and (self.split not in SPLITS or self.split != canonical_split(self.seed)):
            raise ValueError(f'split of seed {self.seed} must be its canonical band, not {self.split!r}')
        if self.score is not None and ((self.score.task_id, self.score.seed, self.score.split)
                                       != (self.task_id, self.seed, self.split)):
            raise ValueError('score belongs to a different task, seed or split')
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError('backend must name where the trial ran')
        if not isinstance(self.test_only, bool):
            raise ValueError('test_only must be True or False')
        for name in ('cost_usd', 'wall_s'):
            _optional_number(getattr(self, name), name)
        for name in ('llm_calls', 'sim_trials'):
            _optional_number(getattr(self, name), name, integer=True)
        object.__setattr__(self, 'artifacts', MappingProxyType(dict(self.artifacts)))

    @property
    def success(self):
        return self.score is not None and self.score.success

    def to_record(self):
        record = {key: value for key, value in self.__dict__.items() if key not in ('score', 'artifacts')}
        record['score'] = None if self.score is None else self.score.to_record()
        record['artifacts'] = dict(self.artifacts)
        record['success'] = self.success
        return record


def append_records(path, records):
    with Path(path).open('a') as handle:
        for record in records:
            handle.write(json.dumps(record.to_record(), allow_nan=False) + '\n')


def recorded_seeds(path, method, variant, task_id):
    """Seeds with a record of `method` `variant` on `task_id` in the JSONL file at `path` (none if it is absent)."""
    path = Path(path)
    if not path.exists():
        return set()
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    wanted = (method, variant, task_id)
    return {row['seed'] for row in rows if (row['method'], row['variant'], row['task_id']) == wanted}


def refuse_repeats(path, method, variant, task_id, seeds):
    """Raise if `path` already records `method` `variant` on `task_id` for any of `seeds`."""
    repeated = sorted(set(seeds) & recorded_seeds(path, method, variant, task_id))
    if repeated:
        raise ValueError(f'{path} already has records of {method} {variant} on {task_id} for seeds {repeated}')


def claims_dir():
    """Where held-out trials are claimed: configs/jing.json baselines.heldout_claims_dir (relative to the repository
    root), unless the environment variable ASSEMBLY_BASELINE_CLAIMS_DIR points elsewhere (the tests use a temporary
    directory)."""
    override = os.environ.get(CLAIMS_ENV)
    return Path(override) if override else repo_path(canonical_protocol()['heldout_claims_dir'])


def _claim_path(method, variant, task_id, seed):
    name = f'{method}__{variant}__{task_id}__{seed}'
    readable = re.sub(r'[^A-Za-z0-9._-]', '_', name)
    return claims_dir() / f'{readable}.{hashlib.sha256(name.encode()).hexdigest()[:12]}.json'


def claim_heldout(method, variant, task_id, seed, *, records_path):
    """Claim held-out `seed` for one trial of `method` `variant` on `task_id`, before the trial touches anything.

    The claim is an exclusively created file in claims_dir(), so a second claim fails whatever records file or
    process asks; it is never removed, so a trial that crashes or is interrupted has used its seed (the runner
    records it as an error). Also refuses a seed already recorded in `records_path`. Returns the claim's path."""
    if canonical_split(seed) != 'heldout':
        raise ValueError(f'seed {seed} is not a held-out seed')
    refuse_repeats(records_path, method, variant, task_id, [seed])
    path = _claim_path(method, variant, task_id, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise ValueError(f'held-out seed {seed} of {method} {variant} on {task_id} was already claimed: '
                         f'{path}') from None
    with os.fdopen(descriptor, 'w') as handle:
        json.dump({'method': method, 'variant': variant, 'task_id': task_id, 'seed': seed,
                   'records': str(Path(records_path).resolve()), 'pid': os.getpid(),
                   'time_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}, handle)
    return path


def unrecorded_claims(records):
    """Held-out claims in claims_dir() without a matching record among `records` (dicts): trials that died before
    writing one, or records left out of a summary."""
    done = {(r['method'], r['variant'], r['task_id'], r['seed']) for r in records}
    directory = claims_dir()
    claims = [json.loads(path.read_text()) for path in sorted(directory.glob('*.json'))] if directory.is_dir() else []
    return [claim for claim in claims
            if (claim['method'], claim['variant'], claim['task_id'], claim['seed']) not in done]


def wilson_interval(successes, n, z=1.959963984540054):
    """95 % Wilson score interval for a binomial success rate."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _checked(records, include_test_only):
    """Records as counted, after refusing what cannot be counted as it stands."""
    bands, heldout = canonical_protocol()['seed_splits'], set()
    for record in records:
        if record['test_only'] and not include_test_only:
            raise ValueError(f'{record["method"]} {record["variant"]} on {record["task_id"]} is a test-only run '
                             f'({record["backend"]}); it is not a baseline result')
        if record['seed'] is not None and record['split'] != split_of(record['seed'], bands):
            raise ValueError(f'seed {record["seed"]} is recorded as {record["split"]!r}, not its canonical band')
        if bool(record['success']) != bool(record['score'] is not None and record['score']['success']):
            raise ValueError(f'{record["method"]} seed {record["seed"]}: the stored success contradicts its score')
        if record['split'] == 'heldout':
            key = (record['method'], record['variant'], record['task_id'], record['seed'])
            if key in heldout:
                raise ValueError(f'held-out seed recorded twice: {key}')
            heldout.add(key)
    return records


def summarize(records, *, include_test_only=False):
    """Per (method, variant, task, backend, test_only, split): runs, successes, rate with Wilson 95 % CI, placed
    fraction, cost.

    Refuses test-only runs (unless `include_test_only`), a held-out seed recorded twice, a stored success that its
    score contradicts, a split that is not its seed's canonical band, and a group whose scores were judged against
    different layouts, goal sources or tolerances."""
    groups = {}
    for record in _checked(list(records), include_test_only):
        key = (record['method'], record['variant'], record['task_id'], record['backend'], record['test_only'],
               record['split'])
        groups.setdefault(key, []).append(record)
    summary = []
    for (method, variant, task_id, backend, test_only, split), group in sorted(
            groups.items(), key=lambda item: tuple(map(str, item[0]))):
        judged = {(r['score']['layout_sha256'], r['score']['goal_source'], r['score']['position_tolerance_m'],
                   r['score']['rotation_tolerance_deg']) for r in group if r['score']}
        if len(judged) > 1:
            raise ValueError(f'{method} {variant} on {task_id}: scores judged against different layouts, goal '
                             f'sources or tolerances: {sorted(judged)}')
        successes = sum(bool(r['success']) for r in group)
        placed = [r['score']['placed_fraction'] if r['score'] else 0.0 for r in group]
        costs = [r['cost_usd'] for r in group if r['cost_usd'] is not None]
        summary.append({'method': method, 'variant': variant, 'task_id': task_id, 'backend': backend,
                        'test_only': test_only, 'split': split, 'runs': len(group),
                        'successes': successes, 'success_rate': successes / len(group),
                        'wilson95': list(wilson_interval(successes, len(group))),
                        'mean_placed_fraction': float(np.mean(placed)),
                        'cost_usd_total': float(sum(costs)) if costs else None,
                        'statuses': dict(Counter(r['status'] for r in group))})
    return summary
