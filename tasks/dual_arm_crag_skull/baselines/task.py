"""Method-agnostic skull assembly episodes, shared by the baselines and by S2/S1/S0.

An AssemblyTask fixes everything a method is given and judged against: fragment meshes, where each
fragment starts, where it must end and where that goal came from, the table, the success criterion
(pose tolerances and the settle time after release) and the seed. Every method is scored on the same
task by baselines.scoring.

Goals built from a layout are the ground-truth assembly (`goal_source = 'oracle_layout'`), placed
with an explicit convention: the layout's assembly frame sits at the centre of the staging pad on
the table top, unrotated. The anchor starts at its goal (pre-placed) and a fixture holds it there for
the whole trial (the protocol's `anchor`; alone it is not statically stable); every other fragment
starts at its staging pose. Layout poses are `{'position': [x, y, z], 'orientation_wxyz': [w, x, y, z]}`.

`initial_condition(task, seed, ...)` shifts and yaws the staged fragments on the table, deterministic
in the seed, and keeps their footprints apart, on the table, off the pad (touching it is allowed) and
out of the exclusion zones. Seeds are split into practice / selection / held-out bands so no method is
developed on the seeds it is evaluated on.

The comparison protocol (seed bands, placement parameters, success criterion, anchor fixture, held-out
claims directory) is the one in configs/jing.json: `require_canonical(config)` refuses another configuration,
and `verify_task(task)` rebuilds a task from its layout under that protocol and refuses one that
differs (another tolerance, placement or split label, edited layout or meshes). The generator is
public code, so held-out initial conditions are protected by the runners' seed guards and claims
(baselines.scoring), not by secrecy.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..__main__ import DEFAULT_CONFIG
from ..perception import rigid_transform

SPLITS = ('practice', 'selection', 'heldout')
GOAL_SOURCES = ('oracle_layout', 'garf_candidate')
PLACEMENT_KEYS = ('max_shift_m', 'max_yaw_deg', 'gap_m', 'attempts')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frozen_pose(value, name):
    try:
        pose = np.array(rigid_transform(value), dtype=float, copy=True)
    except ValueError as exc:
        raise ValueError(f'{name}: {exc}') from exc
    pose.flags.writeable = False
    return pose


def _finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite number')
    return float(value)


def pose_from_layout(record):
    """4x4 from a layout pose; the quaternion must already be unit length."""
    position = np.asarray(record['position'], dtype=float)
    quaternion = np.asarray(record['orientation_wxyz'], dtype=float)
    if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite([*position, *quaternion]).all():
        raise ValueError('layout pose needs a finite position (3) and orientation_wxyz (4)')
    if abs(np.linalg.norm(quaternion) - 1.0) > 1e-6:
        raise ValueError('layout orientation_wxyz is not a unit quaternion')
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    pose[:3, 3] = position
    return pose


def pose_to_wxyz(pose):
    """[x, y, z, qw, qx, qy, qz] with qw >= 0 (the INTERFACES.md raw pose order)."""
    pose = rigid_transform(pose)
    x, y, z, w = Rotation.from_matrix(np.array(pose[:3, :3])).as_quat()  # copy: scipy refuses read-only input
    quaternion = np.array([w, x, y, z]) * (1.0 if w >= 0 else -1.0)
    return [*map(float, pose[:3, 3]), *map(float, quaternion)]


@dataclass(frozen=True)
class FragmentSpec:
    """One rigid fragment: its mesh (layout npz, 'v' vertices and 'f' faces in metres in the fragment
    frame), the fragment-frame bounding box, and its start and goal poses in the world."""

    fragment_id: str
    mesh: str
    mesh_sha256: str
    local_bounds: tuple
    initial_world_T_fragment: np.ndarray
    goal_world_T_fragment: np.ndarray

    def __post_init__(self):
        if not isinstance(self.fragment_id, str) or not self.fragment_id:
            raise ValueError('fragment_id must be a non-empty string')
        if not isinstance(self.mesh_sha256, str) or len(self.mesh_sha256) != 64:
            raise ValueError(f'{self.fragment_id}: mesh_sha256 must be a sha256 hex digest')
        low, high = (np.asarray(corner, dtype=float) for corner in self.local_bounds)
        if low.shape != (3,) or high.shape != (3,) or not np.isfinite([*low, *high]).all() or (low >= high).any():
            raise ValueError(f'{self.fragment_id}: local_bounds must be finite (min, max) with min < max')
        object.__setattr__(self, 'mesh', str(self.mesh))
        object.__setattr__(self, 'local_bounds', (tuple(map(float, low)), tuple(map(float, high))))
        for name in ('initial_world_T_fragment', 'goal_world_T_fragment'):
            object.__setattr__(self, name, _frozen_pose(getattr(self, name), f'{self.fragment_id}.{name}'))

    def footprint(self, world_T_fragment=None):
        """World xy corners (4, 2) of the fragment-frame box's xy face, at `world_T_fragment`. This is the
        table footprint only while the fragment frame is upright (yaw-only rotation), as layout poses are."""
        pose = self.initial_world_T_fragment if world_T_fragment is None else rigid_transform(world_T_fragment)
        (x0, y0, z0), (x1, y1, _) = self.local_bounds
        corners = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]])
        return (corners @ pose[:3, :3].T + pose[:3, 3])[:, :2]

    def to_record(self):
        return {'fragment_id': self.fragment_id, 'mesh': self.mesh, 'mesh_sha256': self.mesh_sha256,
                'local_bounds': [list(corner) for corner in self.local_bounds],
                'initial_world_T_fragment': self.initial_world_T_fragment.tolist(),
                'goal_world_T_fragment': self.goal_world_T_fragment.tolist()}

    @classmethod
    def from_record(cls, record):
        return cls(record['fragment_id'], record['mesh'], record['mesh_sha256'], tuple(record['local_bounds']),
                   np.asarray(record['initial_world_T_fragment']), np.asarray(record['goal_world_T_fragment']))


@dataclass(frozen=True)
class AssemblyTask:
    """One episode definition. `fragments` are sorted by id; `seed` and `split` are set together. Success: every
    fragment within the tolerances relative to the anchor, the anchor within them of its goal, read `settle_s`
    seconds after both grippers have released (the `released_stable` completion predicate)."""

    task_id: str
    layout: str
    layout_sha256: str
    anchor_id: str
    goal_source: str
    world_T_assembly: np.ndarray
    table_top_z_m: float
    table_bounds_m: tuple
    pad_bounds_m: tuple
    exclusions_m: tuple
    position_tolerance_m: float
    rotation_tolerance_deg: float
    settle_s: float
    fragments: tuple
    seed: int | None = None
    split: str | None = None

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError('task_id must be a non-empty string')
        fragments = tuple(self.fragments)
        ids = [fragment.fragment_id for fragment in fragments]
        if not fragments or ids != sorted(set(ids)):
            raise ValueError('fragments must be non-empty, unique and sorted by fragment_id')
        if self.anchor_id not in ids:
            raise ValueError(f'anchor {self.anchor_id!r} is not one of the fragments')
        if len(ids) < 2:
            raise ValueError('an assembly needs the anchor and at least one fragment to place')
        if self.goal_source not in GOAL_SOURCES:
            raise ValueError(f'goal_source must be one of {GOAL_SOURCES}')
        if (self.seed is None) != (self.split is None):
            raise ValueError('seed and split are set together')
        if self.split is not None and self.split not in SPLITS:
            raise ValueError(f'split must be one of {SPLITS}')
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0):
            raise ValueError('seed must be a non-negative integer')
        if not np.isfinite(self.table_top_z_m):
            raise ValueError('table_top_z_m must be finite')
        object.__setattr__(self, 'fragments', fragments)
        object.__setattr__(self, 'world_T_assembly', _frozen_pose(self.world_T_assembly, 'world_T_assembly'))
        object.__setattr__(self, 'table_top_z_m', float(self.table_top_z_m))
        object.__setattr__(self, 'table_bounds_m', _rectangle(self.table_bounds_m, 'table_bounds_m'))
        object.__setattr__(self, 'pad_bounds_m', _rectangle(self.pad_bounds_m, 'pad_bounds_m'))
        object.__setattr__(self, 'exclusions_m', tuple(_rectangle(r, 'exclusions_m') for r in self.exclusions_m))
        for name in ('position_tolerance_m', 'rotation_tolerance_deg', 'settle_s'):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name))

    @property
    def fragment_ids(self):
        return tuple(fragment.fragment_id for fragment in self.fragments)

    @property
    def moving_ids(self):
        """Fragments a method must place (all but the anchor)."""
        return tuple(key for key in self.fragment_ids if key != self.anchor_id)

    def fragment(self, fragment_id):
        for fragment in self.fragments:
            if fragment.fragment_id == fragment_id:
                return fragment
        raise KeyError(fragment_id)

    def initial_poses(self):
        return {f.fragment_id: f.initial_world_T_fragment.copy() for f in self.fragments}

    def goal_poses(self):
        return {f.fragment_id: f.goal_world_T_fragment.copy() for f in self.fragments}

    def initial_sha256(self):
        """Digest of the fragment ids and start poses, identifying the initial condition in records."""
        payload = json.dumps([[f.fragment_id, f.initial_world_T_fragment.tolist()] for f in self.fragments])
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_record(self):
        return {'task_id': self.task_id, 'layout': self.layout, 'layout_sha256': self.layout_sha256,
                'anchor_id': self.anchor_id, 'goal_source': self.goal_source,
                'world_T_assembly': self.world_T_assembly.tolist(), 'table_top_z_m': self.table_top_z_m,
                'table_bounds_m': list(self.table_bounds_m), 'pad_bounds_m': list(self.pad_bounds_m),
                'exclusions_m': [list(r) for r in self.exclusions_m],
                'position_tolerance_m': self.position_tolerance_m,
                'rotation_tolerance_deg': self.rotation_tolerance_deg, 'settle_s': self.settle_s,
                'fragments': [fragment.to_record() for fragment in self.fragments],
                'seed': self.seed, 'split': self.split}

    @classmethod
    def from_record(cls, record):
        return cls(record['task_id'], record['layout'], record['layout_sha256'], record['anchor_id'],
                   record['goal_source'], np.asarray(record['world_T_assembly']), record['table_top_z_m'],
                   tuple(record['table_bounds_m']), tuple(record['pad_bounds_m']),
                   tuple(tuple(r) for r in record['exclusions_m']), record['position_tolerance_m'],
                   record['rotation_tolerance_deg'], record['settle_s'],
                   tuple(FragmentSpec.from_record(f) for f in record['fragments']), record['seed'], record['split'])


def _rectangle(value, name):
    values = tuple(float(v) for v in value)
    if len(values) != 4 or not np.isfinite(values).all() or values[0] >= values[1] or values[2] >= values[3]:
        raise ValueError(f'{name} must be [xmin, xmax, ymin, ymax] with min < max')
    return values


def _contains(outer, inner):
    return outer[0] <= inner[0] and inner[1] <= outer[1] and outer[2] <= inner[2] and inner[3] <= outer[3]


def _goal_vertices(fragment):
    with np.load(fragment.mesh) as data:
        vertices = np.asarray(data['v'], dtype=float)
    pose = fragment.goal_world_T_fragment
    return vertices @ pose[:3, :3].T + pose[:3, 3]


def check_layout_geometry(task):
    """Raise unless the pad and exclusions lie on the table, every fragment's mesh at its goal lies over the pad and not
    below the table top, and the staged footprints lie on the table, off the pad and apart from each other and the
    exclusions."""
    if not _contains(task.table_bounds_m, task.pad_bounds_m):
        raise ValueError('the pad is not on the table')
    for zone in task.exclusions_m:
        if not _contains(task.table_bounds_m, zone):
            raise ValueError(f'exclusion zone {zone} is not on the table')
    pad = _rectangle_polygon(task.pad_bounds_m)
    for fragment in task.fragments:
        points = _goal_vertices(fragment)
        if not _inside(points[:, :2], task.pad_bounds_m, 0.0):
            raise ValueError(f'{fragment.fragment_id}: the goal is not over the pad')
        if points[:, 2].min() < task.table_top_z_m - 1e-6:
            raise ValueError(f'{fragment.fragment_id}: the goal reaches below the table top')
    staged = {key: task.fragment(key).footprint() for key in task.moving_ids}
    blocked = [_rectangle_polygon(zone) for zone in task.exclusions_m]
    for index, (key, polygon) in enumerate(staged.items()):
        if not _inside(polygon, task.table_bounds_m, 0.0) or not _separated(polygon, pad, 0.0):
            raise ValueError(f'{key}: the staging footprint is off the table or on the pad')
        others = [*list(staged.values())[index + 1:], *blocked]
        if not all(_separated(polygon, other, 0.0) for other in others):
            raise ValueError(f'{key}: the staging footprint overlaps another fragment or an exclusion zone')


def task_from_layout(layout_path, *, anchor_id, criterion=None, task_id=None):
    """Oracle task from a schema-2 layout: goals are the layout's assembled poses on the pad. `criterion` is
    {'position_tolerance_m', 'rotation_tolerance_deg', 'settle_s'}, by default the canonical one
    (configs/jing.json). The default task id names the layout's object, its sha256 prefix and the anchor."""
    criterion = canonical_protocol()['success'] if criterion is None else criterion
    path = Path(layout_path).resolve()
    data = json.loads(path.read_text())
    if data.get('schema_version') != 2:
        raise ValueError('expected a schema_version 2 layout')
    staging = data['staging']
    xmin, xmax, ymin, ymax = _rectangle(staging['pad_bounds'], 'staging.pad_bounds')
    world_T_assembly = np.eye(4)
    world_T_assembly[:3, 3] = [(xmin + xmax) / 2, (ymin + ymax) / 2, float(staging['table_top_z'])]
    fragments = []
    for piece in sorted(data['pieces'], key=lambda p: p['name']):
        mesh = (path.parent / piece['mesh']).resolve()
        if path.parent not in mesh.parents:
            raise ValueError(f"{piece['name']}: mesh {piece['mesh']!r} is outside the layout directory")
        goal = world_T_assembly @ pose_from_layout(piece['goal_pose'])
        initial = goal if piece['name'] == anchor_id else pose_from_layout(piece['staging_pose'])
        fragments.append(FragmentSpec(piece['name'], str(mesh), file_sha256(mesh), tuple(piece['local_bounds']),
                                      initial, goal))
    layout_sha256 = file_sha256(path)
    object_id = data.get('object_id', path.parent.name)
    task = AssemblyTask(task_id or f'{object_id}-{layout_sha256[:12]}-anchor-{anchor_id}', str(path), layout_sha256,
                        anchor_id, 'oracle_layout', world_T_assembly, staging['table_top_z'],
                        tuple(staging['table_bounds']), (xmin, xmax, ymin, ymax),
                        tuple(tuple(r) for r in staging.get('exclusions', ())), criterion['position_tolerance_m'],
                        criterion['rotation_tolerance_deg'], criterion['settle_s'], tuple(fragments))
    check_layout_geometry(task)
    return task


def success_criterion(config):
    """{'position_tolerance_m', 'rotation_tolerance_deg', 'settle_s'} of the released_stable completion predicate
    named by baselines.success_tolerances_from (an s2.completion_defaults skill)."""
    name = config['baselines']['success_tolerances_from']
    if not isinstance(name, str) or '.' not in name:
        raise ValueError('baselines.success_tolerances_from must name an s2.completion_defaults skill')
    section, skill = name.rsplit('.', 1)
    if section != 's2.completion_defaults':
        raise ValueError('baselines.success_tolerances_from must name an s2.completion_defaults skill')
    predicate = config['s2']['completion_defaults'][skill]
    if predicate.get('type') != 'released_stable':
        raise ValueError(f's2.completion_defaults.{skill} is not a released_stable predicate')
    return {key: float(predicate[key]) for key in ('position_tolerance_m', 'rotation_tolerance_deg', 'settle_s')}


def protocol(config):
    """What every compared trial is held to: seed bands, placement parameters, success criterion, how the anchor is
    held ('fixture': at its goal for the whole trial, so nothing can move it), claims directory."""
    baselines = config['baselines']
    return {'seed_splits': check_splits(baselines['seed_splits']),
            'initial_condition': dict(baselines['initial_condition']),
            'success': success_criterion(config), 'anchor': baselines['anchor'],
            'heldout_claims_dir': baselines['heldout_claims_dir']}


def canonical_protocol():
    """The protocol of configs/jing.json, read afresh."""
    return protocol(json.loads(DEFAULT_CONFIG.read_text()))


def require_canonical(config):
    """Raise unless `config` carries the canonical protocol, so a trial cannot be relabelled or judged differently by
    passing another configuration."""
    if protocol(config) != canonical_protocol():
        raise ValueError('baselines.seed_splits, baselines.initial_condition, the success criterion, baselines.anchor '
                         f'and baselines.heldout_claims_dir must be the ones in {DEFAULT_CONFIG.name}')


def canonical_split(seed):
    return split_of(seed, canonical_protocol()['seed_splits'])


def configured_initial_condition(task, seed, config):
    """initial_condition with the placement parameters of `config`, which must carry the canonical protocol."""
    require_canonical(config)
    return initial_condition(task, seed, **config['baselines']['initial_condition'])


def verify_task(task):
    """Return `task` if it is what its layout gives under the canonical protocol (the layout and meshes by sha256,
    the success criterion, the default task id and, when seeded, the canonical initial condition and split of its
    seed); raise otherwise."""
    base = task_from_layout(task.layout, anchor_id=task.anchor_id)
    if task.seed is not None:
        base = configured_initial_condition(base, task.seed, json.loads(DEFAULT_CONFIG.read_text()))
    expected, actual = base.to_record(), task.to_record()
    if expected != actual:
        differing = sorted(key for key in expected if expected[key] != actual.get(key))
        raise ValueError(f'task {task.task_id} is not its layout under the canonical protocol (differs in '
                         f'{differing})')
    return task


def check_splits(splits):
    """{'practice': [lo, hi], 'selection': [lo, hi], 'heldout': [lo, hi]}: inclusive, disjoint integer bands."""
    if set(splits) != set(SPLITS):
        raise ValueError(f'seed_splits must name exactly {SPLITS}')
    bands = []
    for name in SPLITS:
        band = splits[name]
        if (len(band) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in band)
                or not 0 <= band[0] <= band[1]):
            raise ValueError(f'seed_splits.{name} must be [lo, hi] non-negative integers with lo <= hi')
        bands.append((band[0], band[1], name))
    bands.sort()
    for (_, high, first), (low, _, second) in zip(bands, bands[1:]):
        if low <= high:
            raise ValueError(f'seed_splits.{first} and seed_splits.{second} overlap')
    return {name: (int(splits[name][0]), int(splits[name][1])) for name in SPLITS}


def split_of(seed, splits):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    for name, (low, high) in check_splits(splits).items():
        if low <= seed <= high:
            return name
    raise ValueError(f'seed {seed} is in no seed split')


def _separated(first, second, gap):
    """True if two convex xy polygons are at least `gap` apart along some separating axis."""
    for polygon in (first, second):
        edges = np.roll(polygon, -1, axis=0) - polygon
        for edge in edges:
            axis = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
            a, b = first @ axis, second @ axis
            if a.max() + gap <= b.min() or b.max() + gap <= a.min():
                return True
    return False


def _inside(polygon, rectangle, margin):
    xmin, xmax, ymin, ymax = rectangle
    return (polygon[:, 0].min() >= xmin + margin and polygon[:, 0].max() <= xmax - margin
            and polygon[:, 1].min() >= ymin + margin and polygon[:, 1].max() <= ymax - margin)


def _rectangle_polygon(rectangle):
    xmin, xmax, ymin, ymax = rectangle
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


def initial_condition(task, seed, *, max_shift_m, max_yaw_deg, gap_m, attempts=200):
    """Task with every moving fragment shifted (uniform in a disc) and yawed about the vertical through
    its footprint centre, drawn from `seed`, starting from the unseeded base task's staging poses. The
    split is the seed's canonical band. Raises if a fragment finds no valid placement."""
    if task.seed is not None:
        raise ValueError('initial_condition starts from the base task (seed None), not from a seeded task')
    split = canonical_split(seed)
    for name, value in (('max_shift_m', max_shift_m), ('max_yaw_deg', max_yaw_deg), ('gap_m', gap_m)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be a non-negative finite number')
    rng = np.random.default_rng(seed)
    placed = {task.anchor_id: task.fragment(task.anchor_id).footprint()}
    blocked = [_rectangle_polygon(r) for r in task.exclusions_m]
    fragments = {task.anchor_id: task.fragment(task.anchor_id)}
    for key in task.moving_ids:
        fragment = task.fragment(key)
        start = fragment.initial_world_T_fragment
        for _ in range(attempts):
            radius, angle = max_shift_m * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
            yaw = np.deg2rad(rng.uniform(-max_yaw_deg, max_yaw_deg))
            centre = fragment.footprint().mean(axis=0)
            move = np.eye(4)
            move[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
            move[:2, 3] = centre + radius * np.array([np.cos(angle), np.sin(angle)]) - move[:2, :2] @ centre
            pose = move @ start
            polygon = fragment.footprint(pose)
            if (_inside(polygon, task.table_bounds_m, gap_m)
                    and _separated(polygon, _rectangle_polygon(task.pad_bounds_m), 0.0)
                    and all(_separated(polygon, other, gap_m) for other in [*placed.values(), *blocked])):
                placed[key] = polygon
                fragments[key] = replace(fragment, initial_world_T_fragment=pose)
                break
        else:
            raise ValueError(f'seed {seed}: no valid placement for {key} in {attempts} attempts')
    return replace(task, fragments=tuple(fragments[key] for key in task.fragment_ids), seed=seed, split=split)
