"""AssemblyTask -> a DexAgent Stage 3 input folder: Stage 1 semantics, Stage 2 scene, stand-in video, manifest.

DexAgent's Stage 3 starts from a human video, Stage 1's scene_semantics.json and Stage 2's scene_recon.json with
scene.xml, camera.json, frame 0 and the object masks beside it. We have no human demonstration and do not run
Stages 1-2, so every one of those inputs is constructed here from the AssemblyTask, deterministically:

- Stage 2 is the DexAgent bench as published (the clone's setup/scene.xml: YAM arms, Sharpa hands) with one
  body per fragment whose collision is the exact box decomposition of its voxel solid (geometry.py), staged by
  DexAgent's own stage_on_bench and then settled in MuJoCo with the robot held, because DexAgent's gates demand
  a keyframe at rest. The anchor is fixed at its assembled pose by default (`anchor_mode='fixed'`): alone on the
  table the assembled anchor of the skull layout is not statically stable;
- the video is a stand-in, not a demonstration: that Stage 2 scene rendered through `video_cam` (straight above
  the table) with the robot hidden and no hands, each moving fragment gliding from its settled start to its goal
  during its subgoal's time window;
- frame 0 and the masks come from DexAgent's CPU segment_source_frame on that video, run through DexAgent's own
  reference-evidence registry, which writes the replay recipes Stage 3 re-runs before the agent starts;
- Stage 1 has one `place` subgoal per moving fragment, bottom layer first, each with a relative_pose predicate to
  the anchor at the task's tolerances, plus task_success_predicates for every moving fragment and the anchor.

Frames: world is the task frame; rig is DexAgent's (origin between the arm mounts on the tabletop, x toward the
right station, y away from the operator, z up). `rig_T_world` yaws about z and puts the assembly pad centre at
`pad_xy` on the tabletop (rig z = 0 is the task's table top). Fragment body frames are the fragment frames.
Steps that need DexAgent's code run in the DexAgent venv through worker.py (dexenv.py).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..task import file_sha256, pose_to_wxyz
from . import dexenv
from .geometry import asset_mjcf, box_decomposition, goal_neighbours, load_mesh, voxel_surface, write_obj

DEXAGENT_COMMIT = 'e29bdad8ed3e95b2c9ee2eea9044a6b4115ec633'
EXPORT_SCHEMA = 'assembly_sim.dexagent_export.v1'
VARIANT = 'native_yam_sharpa_stage3'
# The bench DexAgent plans on: its own (YAM arms, Sharpa hands, placed on its table by pad_xy and yaw_deg) or the
# hardware every method shares (bench/: CRX arms, YAM grippers, the workcell's table; the rig is the world shifted down
# to the tabletop), which needs the port's checkout (bench/checkout.py).
VARIANTS = {'native': VARIANT, 'crx_yam': 'crx_yam_stage3'}
SAFE_ID = re.compile(r'^[a-z][a-z0-9_]*$')
ANCHOR_MODES = ('fixed', 'free')
# The bench lights sum to about 2.2 on an upward face, so material colours are 0.45 of the rendered colour.
PALETTE = tuple((name, tuple(round(0.45 * v, 3) for v in rgb)) for name, rgb in (
    ('red', (1.0, 0.08, 0.08)), ('blue', (0.12, 0.30, 1.0)), ('green', (0.08, 0.80, 0.20)), ('yellow', (1.0, 0.92, 0.08)),
    ('white', (1.0, 1.0, 1.0)), ('cyan', (0.08, 0.90, 0.95)), ('magenta', (0.95, 0.12, 0.75)),
    ('orange', (1.0, 0.50, 0.05)), ('purple', (0.60, 0.20, 0.95)), ('pink', (1.0, 0.62, 0.78)),
    ('lime', (0.60, 1.0, 0.15)), ('silver', (0.75, 0.75, 0.82))))
SUPPORT_SURFACE = 'dark brown table'
TASK_PROMPT = ('Reassemble the fractured skull: place every skull fragment at its assembled pose against the anchor '
               'fragment.')
STAND_IN = ('The video is a stand-in rendering, not a human demonstration: there are no hands in it; each skull '
            'fragment glides from where it starts to its assembled pose during its subgoal window.')
REPO_ROOT = Path(__file__).resolve().parents[4]
REQUIRED_CHECKS = ('source_video_identity_check', 'source_reference_provenance_check', 'scene_semantic_schema_check',
                   'scene_semantic_consistency_check', 'goal_language_coverage_check', 'initial_keyframe_state_check',
                   'rigid_runtime_compile', 'hand_scene_from_recon')


def _workcell_physics():
    physics = json.loads((REPO_ROOT / 'core' / 'config' / 'workcell.json').read_text())['physics']
    return float(physics['fragment_density']), float(physics['static_friction'])


@dataclass(frozen=True)
class ExportOptions:
    """Bench placement, stand-in video timing and fragment physics (density and friction from core/config)."""

    bench: str = 'native'
    pad_xy: tuple = (0.0, 0.40)
    yaw_deg: float = 0.0
    anchor_mode: str = 'fixed'
    settle_s: float = 2.0
    width: int = 640
    height: int = 480
    fps: float = 30.0
    fovy_deg: float = 60.0
    lead_s: float = 1.0
    seconds_per_fragment: float = 8.0
    tail_s: float = 1.0
    clearance_m: float = 0.05
    view_margin_m: float = 0.06
    mask_margin_px: int = 6
    fallback_pitch_m: float = 0.004
    density_kg_m3: float = field(default_factory=lambda: _workcell_physics()[0])
    friction: tuple = field(default_factory=lambda: (_workcell_physics()[1], 0.02, 0.001))

    def __post_init__(self):
        for name in ('fps', 'fovy_deg', 'seconds_per_fragment', 'density_kg_m3', 'fallback_pitch_m', 'settle_s'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a positive finite number')
        if self.anchor_mode not in ANCHOR_MODES:
            raise ValueError(f'anchor_mode must be one of {ANCHOR_MODES}')
        if self.bench not in VARIANTS:
            raise ValueError(f'bench must be one of {tuple(VARIANTS)}')
        if not 0 < self.fovy_deg < 150:
            raise ValueError('fovy_deg must be in (0, 150)')
        if min(self.width, self.height) < 64:
            raise ValueError('the stand-in video needs at least 64 x 64 pixels')
        object.__setattr__(self, 'pad_xy', tuple(map(float, self.pad_xy)))
        object.__setattr__(self, 'friction', tuple(map(float, self.friction)))

    def to_record(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def rig_transform(task, pad_xy, yaw_deg):
    """rig_T_world: yaw about +z, the pad centre to (pad_xy, 0) on the DexAgent tabletop (rig z = 0)."""
    rotation = Rotation.from_euler('z', float(yaw_deg), degrees=True).as_matrix()
    centre = np.array([*np.array(task.world_T_assembly[:2, 3], dtype=float), task.table_top_z_m])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.array([pad_xy[0], pad_xy[1], 0.0]) - rotation @ centre
    return transform


def bench_rig_transform(task):
    """rig_T_world of the shared-hardware bench: the world shifted down to the tabletop (bench/scene.py)."""
    from core.workcell.config import load_workcell_preset
    from .bench.scene import PRESET, rig_T_world, tabletop_z
    workcell = load_workcell_preset(PRESET)
    if abs(task.table_top_z_m - tabletop_z(workcell)) > 1e-9:
        raise ValueError(f'the task\'s table top is at {task.table_top_z_m} m, the workcell\'s at {tabletop_z(workcell)} m')
    return rig_T_world(workcell)


def object_ids(task):
    """{fragment_id: (object id, colour name, rgb)} in fragment-id order; ids are DexAgent-safe and unique."""
    if len(task.fragment_ids) > len(PALETTE):
        raise ValueError(f'at most {len(PALETTE)} fragments have distinct colour names, got {len(task.fragment_ids)}')
    return {key: (f'{name}_fragment', name, rgb) for key, (name, rgb) in zip(task.fragment_ids, PALETTE)}


def _frozen(array):
    array = np.array(array, dtype=float, copy=True)
    array.flags.writeable = False
    return array


def _rotation(pose):
    return Rotation.from_matrix(np.array(pose[:3, :3], dtype=float))


def _wxyz(rotation):
    x, y, z, w = rotation.as_quat()
    return [float(w), float(x), float(y), float(z)] if w >= 0 else [float(-w), float(-x), float(-y), float(-z)]


def pose_matrix(wxyz_pose):
    """4x4 from [x, y, z, qw, qx, qy, qz]."""
    values = np.asarray(wxyz_pose, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all() or abs(np.linalg.norm(values[3:]) - 1) > 1e-6:
        raise ValueError('a raw pose is [x, y, z, qw, qx, qy, qz] with a unit quaternion')
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(values[[4, 5, 6, 3]]).as_matrix()
    pose[:3, 3] = values[:3]
    return pose


def _yaw_deg(pose):
    return float(np.degrees(np.arctan2(pose[1, 0], pose[0, 0])))


def _check_planar(pose, what):
    if abs(float(pose[2, 2]) - 1.0) > 1e-6:
        raise ValueError(f'{what}: DexAgent Stage 2 poses are planar (x, y, z, yaw); this rotation tilts the fragment')


@dataclass(frozen=True)
class FragmentPlan:
    """One fragment in DexAgent terms: rig poses (body frame = fragment frame), its asset and its subgoal.
    `start_rig` is the settled pose once the bench has been staged, else the task's initial pose."""

    fragment_id: str
    object_id: str
    colour: str
    rgb: tuple
    decomposition: object
    vertices: np.ndarray
    faces: np.ndarray
    initial_rig: np.ndarray
    goal_rig: np.ndarray
    free: bool
    acting_hand: str | None
    window_s: tuple | None
    touching: tuple
    settled_rig: np.ndarray | None = None
    hand_reason: str = 'side of the robot it starts on'

    def __post_init__(self):
        if not SAFE_ID.match(self.object_id):
            raise ValueError(f'{self.object_id!r} is not a DexAgent object id')
        for name in ('vertices', 'initial_rig', 'goal_rig', 'settled_rig'):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _frozen(getattr(self, name)))
        faces = np.array(self.faces, dtype=np.int64, copy=True)
        faces.flags.writeable = False
        object.__setattr__(self, 'faces', faces)

    @property
    def name(self):
        return f'{self.colour} skull fragment'

    @property
    def root_body(self):
        return f'{self.object_id}_{self.object_id}'

    @property
    def free_joint(self):
        return f'{self.root_body}_free' if self.free else None

    @property
    def start_rig(self):
        return self.initial_rig if self.settled_rig is None else self.settled_rig

    def points(self, pose):
        return self.vertices @ pose[:3, :3].T + pose[:3, 3]


@dataclass(frozen=True)
class ExportPlan:
    task: object
    options: ExportOptions
    rig_T_world: np.ndarray
    fragments: tuple
    order: tuple
    duration_s: float
    camera: dict

    def fragment(self, object_id):
        for fragment in self.fragments:
            if fragment.object_id == object_id:
                return fragment
        raise KeyError(object_id)

    @property
    def anchor(self):
        return next(f for f in self.fragments if f.fragment_id == self.task.anchor_id)

    @property
    def id_map(self):
        return {f.object_id: f.fragment_id for f in self.fragments}

    def with_settled(self, settled, reach=None):
        """The plan with each fragment's settled rig start pose {object id: [x, y, z, qw, qx, qy, qz]} and, given
        object_reach_check rows {object id: {reachable_sides, best_side, ...}}, acting hands chosen by choose_hand."""
        if set(settled) != {f.object_id for f in self.fragments}:
            raise ValueError('settled poses must cover every fragment')
        fragments = []
        for fragment in self.fragments:
            moved = replace(fragment, settled_rig=pose_matrix(settled[fragment.object_id]))
            if fragment.acting_hand is not None and reach is not None:
                hand, reason = choose_hand(moved, reach.get(fragment.object_id))
                moved = replace(moved, acting_hand=hand, hand_reason=reason)
            fragments.append(moved)
        return replace(self, fragments=tuple(fragments))


def choose_hand(fragment, reach):
    """(hand, reason) for a moving fragment: the arm on its side when DexAgent's object_reach_check finds a pose
    witness for it, else the other arm with a witness, else the arm the check came closest with."""
    side = 'left' if fragment.points(fragment.start_rig)[:, 0].mean() < 0 else 'right'
    sides = (reach or {}).get('reachable_sides') or []
    if side in sides:
        return side, 'object_reach_check pose witness, own side'
    other = 'right' if side == 'left' else 'left'
    if other in sides:
        return other, 'object_reach_check pose witness, other side only'
    best = (reach or {}).get('best_side')
    if best in ('left', 'right'):
        return best, 'no object_reach_check pose witness; the arm it came closest with'
    return side, 'no object_reach_check report; side of the robot it starts on'


def _bottom_up(task, goal_points):
    """Moving fragments, lowest goal point first (bottom layer first), then nearest the anchor, then by id."""
    anchor_centre = goal_points[task.anchor_id].mean(axis=0)

    def key(fid):
        points = goal_points[fid]
        return (round(float(points[:, 2].min()), 4), round(float(np.linalg.norm(points.mean(axis=0) - anchor_centre)), 4),
                fid)
    return tuple(sorted(task.moving_ids, key=key))


def _camera(points, options):
    """Top-down camera (MuJoCo convention: looks along -z, image up = +y) framing every point with a margin."""
    low = points[:, :2].min(axis=0) - options.view_margin_m
    high = points[:, :2].max(axis=0) + options.view_margin_m
    centre, half = (low + high) / 2, (high - low) / 2
    tangent = np.tan(np.radians(options.fovy_deg) / 2)
    height = max(half[1] / tangent, half[0] / (tangent * options.width / options.height)) + float(points[:, 2].max())
    return {'pos': [float(centre[0]), float(centre[1]), float(height)], 'quat_wxyz': [1.0, 0.0, 0.0, 0.0],
            'fovy_deg': float(options.fovy_deg), 'width': int(options.width), 'height': int(options.height)}


def plan_export(task, options=None):
    options = options or ExportOptions()
    ids = object_ids(task)
    rig_T_world = (rig_transform(task, options.pad_xy, options.yaw_deg) if options.bench == 'native'
                   else bench_rig_transform(task))
    meshes, decompositions, initial, goal = {}, {}, {}, {}
    for fragment in task.fragments:
        vertices, faces = load_mesh(fragment.mesh)
        meshes[fragment.fragment_id] = (vertices, faces)
        decompositions[fragment.fragment_id] = box_decomposition(vertices, faces, options.fallback_pitch_m)
        initial[fragment.fragment_id] = rig_T_world @ np.array(fragment.initial_world_T_fragment, dtype=float)
        goal[fragment.fragment_id] = rig_T_world @ np.array(fragment.goal_world_T_fragment, dtype=float)
        _check_planar(initial[fragment.fragment_id], f'{fragment.fragment_id} initial pose')
    goal_points = {fid: meshes[fid][0] @ pose[:3, :3].T + pose[:3, 3] for fid, pose in goal.items()}
    start_points = {fid: meshes[fid][0] @ pose[:3, :3].T + pose[:3, 3] for fid, pose in initial.items()}
    touching = goal_neighbours(decompositions, goal)
    order = _bottom_up(task, goal_points)
    windows = {fid: (options.lead_s + k * options.seconds_per_fragment, options.lead_s + (k + 1) *
                     options.seconds_per_fragment) for k, fid in enumerate(order)}
    fragments = []
    for fid in task.fragment_ids:
        object_id, colour, rgb = ids[fid]
        is_anchor = fid == task.anchor_id
        hand = None if is_anchor else ('left' if start_points[fid][:, 0].mean() < 0 else 'right')
        fragments.append(FragmentPlan(fid, object_id, colour, tuple(rgb), decompositions[fid], meshes[fid][0],
                                      meshes[fid][1], initial[fid], goal[fid],
                                      not (is_anchor and options.anchor_mode == 'fixed'), hand, windows.get(fid),
                                      tuple(ids[k][0] for k in sorted(touching[fid]))))
    points = np.concatenate([*goal_points.values(), *start_points.values()])
    duration = options.lead_s + len(order) * options.seconds_per_fragment + options.tail_s
    return ExportPlan(task, options, rig_T_world, tuple(fragments), tuple(ids[fid][0] for fid in order), duration,
                      _camera(points, options))


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def video_poses(plan):
    """(frames, objects, 7) rig poses [x, y, z, qw, qx, qy, qz] per video frame, in plan.fragments order: a moving
    fragment rises above everything, glides over its goal while turning to its goal orientation, and descends."""
    options = plan.options
    times = np.arange(int(round(plan.duration_s * options.fps))) / options.fps
    top = max(float(f.points(pose)[:, 2].max()) for f in plan.fragments for pose in (f.start_rig, f.goal_rig))
    out = np.zeros((len(times), len(plan.fragments), 7))
    for index, fragment in enumerate(plan.fragments):
        start, goal = fragment.start_rig, fragment.goal_rig
        r0, r1 = _rotation(start), _rotation(goal)
        lift = max(top + options.clearance_m - float((fragment.vertices @ pose[:3, :3].T)[:, 2].min())
                   for pose in (start, goal))
        p0, p1 = np.array(start[:3, 3]), np.array(goal[:3, 3])
        up0, up1 = np.array([p0[0], p0[1], lift]), np.array([p1[0], p1[1], lift])
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))
        for t, time in enumerate(times):
            if fragment.window_s is None or time <= fragment.window_s[0]:
                position, rotation = p0, r0
            elif time >= fragment.window_s[1]:
                position, rotation = p1, r1
            else:
                u = (time - fragment.window_s[0]) / (fragment.window_s[1] - fragment.window_s[0])
                if u < 0.25:
                    position, rotation = p0 + _smooth(u / 0.25) * (up0 - p0), r0
                elif u < 0.7:
                    s = float(_smooth((u - 0.25) / 0.45))
                    position, rotation = up0 + s * (up1 - up0), slerp([s])[0]
                else:
                    position, rotation = up1 + _smooth((u - 0.7) / 0.25) * (p1 - up1), r1
            out[t, index] = [*position, *_wxyz(rotation)]
    return out


def project_box(camera, points, margin_px):
    """Pixel box [x0, y0, x1, y1] of rig points through the MuJoCo camera (looking along its -z, image up +y)."""
    rotation = Rotation.from_quat(np.array(camera['quat_wxyz'], dtype=float)[[1, 2, 3, 0]]).as_matrix()
    local = (np.asarray(points, dtype=float) - np.asarray(camera['pos'])) @ rotation
    depth = -local[:, 2]
    if (depth <= 0).any():
        raise ValueError('a fragment is behind the stand-in camera')
    focal = camera['height'] / 2 / np.tan(np.radians(camera['fovy_deg']) / 2)
    u = camera['width'] / 2 + focal * local[:, 0] / depth
    v = camera['height'] / 2 - focal * local[:, 1] / depth
    box = [int(np.floor(u.min())) - margin_px, int(np.floor(v.min())) - margin_px,
           int(np.ceil(u.max())) + margin_px, int(np.ceil(v.max())) + margin_px]
    box = [max(0, box[0]), max(0, box[1]), min(camera['width'], box[2]), min(camera['height'], box[3])]
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError('a fragment projects outside the stand-in camera image')
    return box


def relative_pose_params(anchor_goal, goal, position_tolerance_m, rotation_tolerance_deg):
    """DexAgent relative_pose params: the fragment's body pose in the anchor's body frame at the goal."""
    relative = np.linalg.inv(anchor_goal) @ goal
    return {'rel_pos': [float(v) for v in relative[:3, 3]], 'rel_quat': _wxyz(_rotation(relative)),
            'pos_tol_m': float(position_tolerance_m), 'rot_tol_deg': float(rotation_tolerance_deg)}


def _extent_text(fragment):
    size = (fragment.vertices.max(axis=0) - fragment.vertices.min(axis=0)) * 100
    return 'about {:.0f} by {:.0f} by {:.0f} cm'.format(*size)


def object_properties(plan, fragment):
    """Stage 1 `properties` of one fragment (what the reconstruction must reproduce), shared with Stage 2."""
    properties = [f'A rigid printed skull fragment, {_extent_text(fragment)}, with fracture surfaces that mate with '
                  f'neighbouring fragments of the assembled skull.']
    if fragment.fragment_id == plan.task.anchor_id:
        properties.append('It is the anchor: it starts at its assembled pose and stays there'
                          + (', held by a fixture.' if not fragment.free else '.'))
    else:
        properties.append('It is picked up from the table and set down at its assembled pose, where it rests '
                          'without hand support.')
    if fragment.touching:
        names = ' and '.join(f'the {plan.fragment(i).name}' for i in fragment.touching)
        properties.append(f'In the assembled skull it touches {names}.')
    return properties


def scene_semantics(plan, identity):
    """Stage 1 deliverable (stage1_semantic_understanding/schema.json), `input_identity` from DexAgent's
    video_identity, as Stage 1's own finalize step writes it."""
    task, anchor = plan.task, plan.anchor
    tol_m, tol_deg = task.position_tolerance_m, task.rotation_tolerance_deg
    duration = float(identity['duration_s'])
    objects, subgoals = [], []
    for fragment in plan.fragments:
        moves = (f'; it glides to its assembled pose from {fragment.window_s[0]:.1f} to {fragment.window_s[1]:.1f} s'
                 if fragment.window_s else '; it does not move')
        objects.append({'id': fragment.object_id, 'name': fragment.name, 'type': 'rigid',
                        'role': 'target' if fragment is anchor else 'manipulated',
                        'appearance': {'body': f'matte {fragment.colour}'}, 'approximate_size': _extent_text(fragment),
                        'properties': object_properties(plan, fragment),
                        'initial_state': ('at its assembled pose' if fragment is anchor
                                          else 'on the table beside the skull'),
                        'final_state': 'at its assembled pose',
                        'evidence': f'stand-in video: visible from 0.0 to {duration:.1f} s{moves}'})
    for index, object_id in enumerate(plan.order, 1):
        fragment = plan.fragment(object_id)
        other = 'right' if fragment.acting_hand == 'left' else 'left'
        subgoals.append({
            'index': index, 'acting_hand': fragment.acting_hand,
            'action': f'Place the {fragment.name} at its assembled pose against the skull.',
            'object_refs': [object_id, anchor.object_id], 'motion': 'place',
            'hand_roles': {fragment.acting_hand: (f'Picks up the {fragment.name}, carries it to the skull, sets it '
                                                  f'down at its assembled pose and lets go.'),
                           other: 'Holds still, clear of the fragments.'},
            'goal': f'The {fragment.name} is released.',
            'goal_predicates': [{'class': 'relative_pose', 'bodies': {'A': object_id, 'B': anchor.object_id},
                                 'params': relative_pose_params(anchor.goal_rig, fragment.goal_rig, tol_m, tol_deg)}],
            'time_window_s': [round(fragment.window_s[0], 6), round(fragment.window_s[1], 6)]})
    # Nothing lets go of the anchor: DexAgent compiles "released" to release_settle, which a fixture-held body touching
    # nothing fails, and "has not moved" to its invariant (relative to the table, throughout).
    clauses = [f'the {f.name} has not moved' if f is anchor else f'the {f.name} is released' for f in plan.fragments]
    success = ', '.join(clauses[:-1]) + (', and ' if len(clauses) > 2 else ' and ') + clauses[-1]
    predicates = [{'class': 'relative_pose', 'bodies': {'A': object_id, 'B': anchor.object_id},
                   'params': relative_pose_params(anchor.goal_rig, plan.fragment(object_id).goal_rig, tol_m, tol_deg)}
                  for object_id in plan.order]
    predicates.append({'class': 'relative_pose', 'bodies': {'A': anchor.object_id, 'B': 'table'},
                       'params': {'reference_frame': 0, 'pos_tol_m': float(tol_m), 'rot_tol_deg': float(tol_deg)}})
    fixture = '' if anchor.free else ' A fixture holds the anchor at its assembled pose.'
    return {
        'task_prompt': TASK_PROMPT,
        'task_summary': (f'The robot reassembles a fractured skull of {len(plan.fragments)} fragments on the table: '
                         f'the {anchor.name} is the anchor and stays at its assembled pose, and the other skull '
                         f'fragments are placed one at a time at their assembled poses against it and against each '
                         f'other, each set down and released.{fixture}'),
        'video': {'path': identity['path'], 'duration_s': duration, 'fps': identity['fps'],
                  'frames': identity['frames'], 'width': identity['width'], 'height': identity['height']},
        'environment': {'support_surface': SUPPORT_SURFACE,
                        'camera': ('fixed camera straight above the table; a stand-in rendering of the DexAgent bench '
                                   'with the robot hidden, not a recording'),
                        'notes': STAND_IN},
        'hands': {'used': sorted({g['acting_hand'] for g in subgoals}),
                  'notes': ('No human demonstrator: each fragment goes to the arm on its side of the robot when '
                            "DexAgent's object_reach_check finds a pose for it there, else to the other arm.")},
        'objects': objects, 'subgoals': subgoals,
        'task_success': success[0].upper() + success[1:] + '.',
        'task_success_predicates': predicates,
        'notes': [STAND_IN,
                  (f'Goals are the assembled poses relative to the anchor within {1000 * tol_m:.1f} mm and '
                   f'{tol_deg:.1f} deg (goal_predicates, task_success_predicates); the anchor must stay within the '
                   f'same tolerance of its start, which is its assembled pose.{fixture}'),
                  'Constructed by Assembly_Sim from its AssemblyTask; DexAgent Stage 1 did not run.'],
        'input_identity': dict(identity),
    }


def scene_recon(plan, staged, media):
    """Stage 2 deliverable from the worker's stage and media reports."""
    objects = []
    for fragment in plan.fragments:
        oid, start = fragment.object_id, fragment.start_rig
        mine = media['objects'][oid]
        objects.append({
            'id': oid, 'type': 'rigid', 'skill': 'assembly_sim_voxel_boxes', 'skill_origin': 'external',
            'params': {'mesh_sha256': file_sha256(plan.task.fragment(fragment.fragment_id).mesh),
                       'collision': 'exact_voxel_boxes' if fragment.decomposition.exact else 'approximate_voxel_boxes',
                       'pitch_m': fragment.decomposition.pitch_m, 'boxes': len(fragment.decomposition.boxes),
                       'cells': int(len(fragment.decomposition.cells)), 'density_kg_m3': plan.options.density_kg_m3,
                       'friction': list(plan.options.friction), 'rgba': [*fragment.rgb, 1.0], 'free': fragment.free},
            'asset_xml': f'assets/{oid}/{oid}.xml', 'root_body': fragment.root_body,
            'bodies': staged['bodies'][oid],
            'pose': {'x': float(start[0, 3]), 'y': float(start[1, 3]), 'z': float(start[2, 3]),
                     'yaw_deg': _yaw_deg(start), 'quat_wxyz': pose_to_wxyz(start)[3:]},
            'mask': f'masks/{oid}.png', 'iou': mine['iou'],
            'verification_plan': {'silhouette_iou_check': 'placement and size at frame 0 through video_cam',
                                  'settle_stability_check': 'rests on the table at frame 0 without drifting'},
            'verifiers': mine['verifiers'],
            'properties_covered': {text: ('exact box decomposition of the task mesh (geometry.py), placed at the '
                                          'settled AssemblyTask start; silhouette_iou_check, settle_stability_check')
                                   for text in object_properties(plan, fragment)}})
    return {'scene_semantic': '../stage1/scene_semantics.json', 'scene_xml': 'scene.xml', 'camera_json': 'camera.json',
            'frame0': 'frame0.png', 'frame0_compare_png': 'frame0_compare.png', 'simulator': 'mujoco',
            'objects': objects, 'scene_checks': {'settle_before_keyframe': staged['settle'], **media['scene_checks']},
            'notes': ['Constructed by Assembly_Sim from its AssemblyTask (export.py); DexAgent Stage 2 did not run.',
                      'Start poses are the task poses settled in MuJoCo with the robot held (written to keyframe 0); '
                      'pose.yaw_deg is approximate for a fragment that tipped, pose.quat_wxyz is exact.',
                      'Masks: DexAgent segment_source_frame (CPU GrabCut) on frame 0 of the stand-in video, recorded '
                      'by DexAgent reference_evidence (reference_evidence_recipes.json).',
                      'Collision: an exact union of boxes per voxel fragment; the visual mesh never collides.'],
            'source': 'assembly_sim export (stand-in inputs)'}


def write_assets(plan, out_dir):
    """Meshes and MJCF assets under out_dir/stage2/assets/<id>/; returns the stage_on_bench asset list."""
    assets = []
    for fragment in plan.fragments:
        folder = Path(out_dir) / 'stage2' / 'assets' / fragment.object_id
        folder.mkdir(parents=True, exist_ok=True)
        write_obj(folder / f'{fragment.object_id}.obj', *voxel_surface(fragment.decomposition))
        (folder / f'{fragment.object_id}.xml').write_text(asset_mjcf(
            fragment.object_id, f'{fragment.object_id}.obj', fragment.decomposition, (*fragment.rgb, 1.0),
            plan.options.density_kg_m3, plan.options.friction, free=fragment.free))
        x, y, z, *quat = pose_to_wxyz(fragment.initial_rig)
        assets.append({'name': fragment.object_id, 'xml': str((folder / f'{fragment.object_id}.xml').resolve()),
                       'x': x, 'y': y, 'z': z, 'quat': quat})
    return assets


def stage_spec(plan, out_dir, clone):
    """Arguments of `worker.py stage`: stage the assets on the bench, add video_cam, settle, write keyframe 0."""
    out = Path(out_dir).resolve()
    return {'clone': str(Path(clone).resolve()), 'out_dir': str(out), 'stage2_dir': str(out / 'stage2'),
            'camera': plan.camera, 'assets': write_assets(plan, out), 'settle_s': plan.options.settle_s,
            'objects': [{'id': f.object_id, 'root_body': f.root_body, 'free_joint': f.free_joint}
                        for f in plan.fragments]}


def media_spec(plan, out_dir, clone):
    """Arguments of `worker.py media` (plan with settled starts): video, identity, masks, Stage 2 checks."""
    out = Path(out_dir).resolve()
    np.savez_compressed(out / 'video_plan.npz', poses=video_poses(plan),
                        objects=np.array([f.object_id for f in plan.fragments]), fps=np.array(plan.options.fps))
    return {'clone': str(Path(clone).resolve()), 'out_dir': str(out), 'stage2_dir': str(out / 'stage2'),
            'video': str(out / 'video.mp4'), 'video_plan': str(out / 'video_plan.npz'), 'camera': plan.camera,
            'objects': [{'id': f.object_id, 'root_body': f.root_body, 'free_joint': f.free_joint,
                         'box': project_box(plan.camera, f.points(f.start_rig), plan.options.mask_margin_px)}
                        for f in plan.fragments]}


def check_spec(plan, out_dir, clone):
    out = Path(out_dir).resolve()
    first = plan.fragment(plan.order[0])
    return {'clone': str(Path(clone).resolve()), 'out_dir': str(out), 'video': str(out / 'video.mp4'),
            'semantic': str(out / 'stage1' / 'scene_semantics.json'), 'recon': str(out / 'stage2' / 'scene_recon.json'),
            'prompt': TASK_PROMPT, 'hand_scene_probe': {'side': first.acting_hand, 'target': first.object_id},
            'goal_bodies': [{'id': f.object_id, 'goal': pose_to_wxyz(f.goal_rig),
                             'mesh': str(out / 'stage2' / 'assets' / f.object_id / f'{f.object_id}.obj'),
                             'asset': str(out / 'stage2' / 'assets' / f.object_id / f'{f.object_id}.xml')}
                            for f in plan.fragments]}


def export_task(task, out_dir, *, clone, python, env, options=None, timeout_s=3600, s0_limits=None):
    """Write the DexAgent input folder for `task` into out_dir (new or empty); returns the export manifest. The
    manifest's `checks_ok` says whether every gate DexAgent applies before or at Stage 3 to these inputs passed. The
    shared-hardware bench needs the port's checkout, verified for `s0_limits`."""
    options = options or ExportOptions()
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out} is not empty; exports are never overwritten')
    if options.bench == 'native':
        commit = dexenv.verify_clone(clone, DEXAGENT_COMMIT)
    else:
        if s0_limits is None:
            raise ValueError('the shared-hardware bench needs the S0 limits its checkout was prepared for')
        from .bench.checkout import verify_checkout
        commit = verify_checkout(clone, s0_limits).commit
    for folder in ('stage1', 'stage2'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    plan = plan_export(task, options)
    staged = dexenv.run_worker(python, 'stage', stage_spec(plan, out, clone), out / 'worker_stage.log', env=env,
                               timeout_s=timeout_s)
    plan = plan.with_settled(staged['settled'], staged.get('reach'))
    media = dexenv.run_worker(python, 'media', media_spec(plan, out, clone), out / 'worker_media.log', env=env,
                              timeout_s=timeout_s)
    (out / 'stage1' / 'scene_semantics.json').write_text(json.dumps(scene_semantics(plan, media['identity']),
                                                                    indent=1) + '\n')
    (out / 'stage2' / 'scene_recon.json').write_text(json.dumps(scene_recon(plan, staged, media), indent=1) + '\n')
    checks = dexenv.run_worker(python, 'check', check_spec(plan, out, clone), out / 'worker_check.log', env=env,
                               timeout_s=timeout_s)
    reach = {'informational': True, **staged.get('reach_report', {}), 'objects': staged.get('reach')}
    (out / 'export_checks.json').write_text(json.dumps({**checks, 'object_reach_check': reach}, indent=1, default=str)
                                            + '\n')
    failed = [name for name in REQUIRED_CHECKS if (checks.get(name) or {}).get('ok') is not True]
    manifest = export_manifest(plan, out, commit, media, failed)
    (out / 'export_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


def export_manifest(plan, out_dir, commit, media, failed):
    out = Path(out_dir)
    files = {str(p.relative_to(out)): file_sha256(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p.name != 'export_manifest.json'}
    return {'schema': EXPORT_SCHEMA, 'variant': VARIANTS[plan.options.bench], 'dexagent_commit': commit,
            'rig_T_world': plan.rig_T_world.tolist(), 'options': plan.options.to_record(),
            'objects': [{'object_id': f.object_id, 'fragment_id': f.fragment_id, 'name': f.name, 'root_body': f.root_body,
                         'free_joint': f.free_joint, 'body_T_fragment': np.eye(4).tolist(),
                         'start_rig_wxyz': pose_to_wxyz(f.start_rig), 'task_start_rig_wxyz': pose_to_wxyz(f.initial_rig),
                         'goal_rig_wxyz': pose_to_wxyz(f.goal_rig), 'acting_hand': f.acting_hand,
                         'hand_reason': f.hand_reason if f.acting_hand else None,
                         'window_s': None if f.window_s is None else list(f.window_s), 'touching': list(f.touching),
                         'exact_collision': f.decomposition.exact, 'boxes': len(f.decomposition.boxes)}
                        for f in plan.fragments],
            'order': list(plan.order), 'anchor_object_id': plan.anchor.object_id, 'task': plan.task.to_record(),
            'stage1': 'stage1/scene_semantics.json', 'stage2': 'stage2/scene_recon.json', 'video': 'video.mp4',
            'video_is_stand_in': True, 'checks_failed': failed, 'checks_ok': not failed,
            'versions': media.get('versions'), 'files_sha256': files}


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('schema') != EXPORT_SCHEMA:
        raise ValueError(f'{path} is not a DexAgent export manifest ({EXPORT_SCHEMA})')
    return manifest
