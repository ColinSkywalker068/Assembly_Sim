"""Deterministic synthetic bimanual episodes that exercise the S1 pipeline. NOT the skull task.

Each episode uses one arm (left or right, 1/2 each) while the other holds still. Two identical-looking
fragments lie on the active arm's side; the request's observed fragment pose is the only thing that
says which one to pick. Then the hand carries it to a goal past a disc obstacle on the straight line,
passing on the left or the right with probability 1/2 (nothing observable before the carry starts
reveals which), releases it and retreats. The demo mode is ambiguous only at the start of the carry;
the evaluation reads privileged ground truth (mode, chosen fragment, obstacle) offline only.
Three tiny top-down rasters exercise the image path, receive latency, frame drops and missing views.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..contracts.actions import ARMS, hand_slice
from .episode import CameraStream, Episode, PoseStream, SkillSegment

GENERATOR = 'synthetic_pick_detour_v2'
Z0, HOVER, CARRY_LIFT, RETREAT_LIFT = 0.95, 0.08, 0.03, 0.12
OBSTACLE_RADIUS, DETOUR = 0.04, 0.10
TCP_HZ, CAMERA_HZ = 30.0, 15.0
CAMERA_LATENCY_S = (0.02, 0.05)
COMMAND_LEAD_S = 0.05
# Phase boundaries in seconds after the episode start.
PICK_MOVE_END, PICK_END = 1.1, 1.4
CARRY_START, CARRY_END, RELEASE_END, PLACE_END = 1.5, 2.9, 3.2, 3.4
EPISODE_END = 4.6
WIDTH_OPEN, WIDTH_HELD, WIDTH_IDLE = 0.08, 0.03, 0.05
COLORS = {'background': (90, 90, 90), 'target': (40, 200, 60), 'obstacle': (220, 40, 40), 'fragment': (240, 140, 20),
          'left': (40, 80, 230), 'right': (230, 210, 40)}
VIEWS = {'agent': (np.array([0.0, 0.0]), 0.45), 'left_wrist': (None, 0.2), 'right_wrist': (None, 0.2)}


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _tool_down(yaw):
    yaw = np.atleast_1d(yaw)
    return Rotation.from_euler('z', yaw).as_matrix() @ Rotation.from_euler('x', np.pi).as_matrix()


def _pose(xy, z, yaw):
    pose = np.eye(4)
    pose[:3, :3] = _tool_down(yaw)[0]
    pose[:3, 3] = [xy[0], xy[1], z]
    return pose


class Scene:
    """Latent layout of one synthetic episode; `arm`, `chosen` and `mode` are privileged."""

    def __init__(self, rng):
        self.arm = ARMS[int(rng.integers(2))]
        side = -1.0 if self.arm == 'left' else 1.0
        self.fragments = np.array([[side * 0.30, 0.07], [side * 0.30, -0.07]]) + rng.uniform(-0.01, 0.01, (2, 2))
        self.fragment_yaw = rng.uniform(-np.pi / 12, np.pi / 12, 2)
        self.chosen = int(rng.integers(2))
        self.home = np.array([side * 0.40, 0.0]) + rng.uniform(-0.02, 0.02, 2)
        self.rest = np.array([-side * 0.40, 0.25]) + rng.uniform(-0.02, 0.02, 2)
        self.target = np.array([side * 0.02, 0.0]) + rng.uniform(-0.02, 0.02, 2)
        self.target_yaw = rng.uniform(-np.pi / 7, np.pi / 7)
        self.start = self.fragments[self.chosen]
        direction = (self.target - self.start) / np.linalg.norm(self.target - self.start)
        self.normal = np.array([-direction[1], direction[0]])
        self.obstacle = (self.start + self.target) / 2 + self.normal * rng.uniform(-0.005, 0.005)
        self.mode = int(rng.choice([-1, 1]))
        self.via = self.obstacle + self.mode * DETOUR * self.normal
        self.control = 2 * self.via - (self.start + self.target) / 2

    def active_pose(self, t):
        """Planned pose and width of the active arm at times t (seconds after the episode start)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        reach = _smoothstep(t / PICK_MOVE_END)[:, None]
        carry = _smoothstep((t - CARRY_START) / (CARRY_END - CARRY_START))[:, None]
        pick_xy = (1 - reach) * self.home + reach * self.start
        carry_xy = (1 - carry) ** 2 * self.start + 2 * carry * (1 - carry) * self.control + carry ** 2 * self.target
        xy = np.where(t[:, None] < CARRY_START, pick_xy, carry_xy)
        z = Z0 + HOVER * (1 - reach[:, 0])
        z = z + CARRY_LIFT * np.sin(np.pi * carry[:, 0]) + RETREAT_LIFT * _smoothstep((t - PLACE_END) / (EPISODE_END - PLACE_END - 0.2))
        yaw = np.where(t < CARRY_START, reach[:, 0] * self.fragment_yaw[self.chosen],
                       self.fragment_yaw[self.chosen] + (self.target_yaw - self.fragment_yaw[self.chosen]) * carry[:, 0])
        closing = _smoothstep((t - PICK_MOVE_END) / (PICK_END - PICK_MOVE_END))
        opening = _smoothstep((t - CARRY_END) / (RELEASE_END - CARRY_END))
        width = WIDTH_OPEN + (WIDTH_HELD - WIDTH_OPEN) * closing + (WIDTH_OPEN - WIDTH_HELD) * opening
        poses = np.repeat(np.eye(4)[None], len(t), axis=0)
        poses[:, :3, :3] = _tool_down(yaw)
        poses[:, :3, 3] = np.column_stack([xy, z])
        return poses, width

    def idle_pose(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        poses = np.repeat(_pose(self.rest, Z0 + HOVER, 0.0)[None], len(t), axis=0)
        return poses, np.full(len(t), WIDTH_IDLE)

    def plan(self, arm):
        return self.active_pose if arm == self.arm else self.idle_pose

    def fragment_pose(self, index):
        return _pose(self.fragments[index], Z0, self.fragment_yaw[index])

    def target_pose(self):
        return _pose(self.target, Z0, self.target_yaw)

    def fragment_xy(self, t):
        """(2, len(t), 2) fragment positions: the chosen one rides in the hand from grasp to release."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        positions = np.repeat(self.fragments[:, None], len(t), axis=1)
        held = (t >= PICK_END) & (t < RELEASE_END)
        positions[self.chosen, held] = self.active_pose(t[held])[0][:, :2, 3]
        positions[self.chosen, t >= RELEASE_END] = self.target
        return positions


def _sample_times(rng, start, end, hz, jitter):
    nominal = start + np.arange(int(np.ceil((end - start) * hz)) + 1) / hz
    times = nominal + rng.uniform(-jitter, jitter, len(nominal))
    times[0], times[-1] = start, max(end, times[-2] + 1e-3)
    return np.maximum.accumulate(times + np.arange(len(times)) * 1e-9)


def _noisy(rng, poses, widths, position_sigma=2e-4, angle_sigma=np.deg2rad(0.05), width_sigma=1e-4):
    poses = poses.copy()
    poses[:, :3, 3] += rng.normal(0, position_sigma, (len(poses), 3))
    poses[:, :3, :3] = Rotation.from_rotvec(rng.normal(0, angle_sigma, (len(poses), 3))).as_matrix() @ poses[:, :3, :3]
    return poses, np.clip(widths + rng.normal(0, width_sigma, len(widths)), 0.0, None)


def render(scene, view, hands, fragments, size):
    """Top-down orthographic raster of the scene for one camera; hands {arm: xy}, fragments (2, 2)."""
    center, half = VIEWS[view]
    if view != 'agent':
        center = hands['left' if view == 'left_wrist' else 'right']
    axis = (np.arange(size) + 0.5) / size * 2 * half - half
    gx, gy = np.meshgrid(center[0] + axis, center[1] - axis)
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:] = COLORS['background']
    disks = [('target', scene.target, 0.025), ('obstacle', scene.obstacle, OBSTACLE_RADIUS),
             ('fragment', fragments[0], 0.02), ('fragment', fragments[1], 0.02),
             ('right', hands['right'], 0.02), ('left', hands['left'], 0.02)]
    for name, xy, radius in disks:
        image[(gx - xy[0]) ** 2 + (gy - xy[1]) ** 2 <= radius ** 2] = COLORS[name]
    return image


def _cameras(rng, scene, t0, image_size, drop_rate, missing_agent):
    cameras = {}
    for name in ('left_wrist', 'right_wrist', 'agent'):
        if name == 'agent' and missing_agent:
            continue
        times = _sample_times(rng, t0, t0 + EPISODE_END, CAMERA_HZ, 3e-3)
        keep = rng.random(len(times)) >= drop_rate
        keep[0] = True
        times = times[keep]
        received = np.maximum.accumulate(times + rng.uniform(*CAMERA_LATENCY_S, len(times)))
        local = times - t0
        hands = {arm: scene.plan(arm)(local)[0][:, :2, 3] for arm in ARMS}
        fragments = scene.fragment_xy(local)
        frames = np.stack([render(scene, name, {arm: hands[arm][i] for arm in ARMS}, fragments[:, i], image_size)
                           for i in range(len(times))])
        cameras[name] = CameraStream(times, frames, received)
    return cameras


def _segments(scene, episode_id, t0):
    arms, fragment = (scene.arm,), f'part_{scene.chosen}'
    observed = {'source': 'observed_geometry', 'observation_id': f'{episode_id}-obs', 'pose_source': 'operator'}
    grasped = scene.fragment_pose(scene.chosen)
    return (
        SkillSegment(start=t0, end=t0 + PICK_END, skill_id='pick', active_arms=arms, request_id=f'{episode_id}-pick',
                     fragment_id=fragment, fragment_pose=grasped, fragment_pose_provenance=observed),
        SkillSegment(start=t0 + PICK_END, end=t0 + PLACE_END, skill_id='place', active_arms=arms,
                     request_id=f'{episode_id}-place', fragment_id=fragment, target_pose=scene.target_pose(),
                     target_provenance={'source': 'operator', 'generator': GENERATOR},
                     fragment_pose=grasped, fragment_pose_provenance=observed),
        SkillSegment(start=t0 + PLACE_END, end=t0 + EPISODE_END, skill_id='retreat', active_arms=arms,
                     request_id=f'{episode_id}-retreat'),
    )


def make_episode(index, *, seed=0, image_size=32, drop_rate=0.05, missing_agent_rate=0.2, t0=0.013):
    """Returns (Episode, privileged) for one deterministic synthetic demonstration."""
    rng = np.random.default_rng([seed, index])
    scene = Scene(rng)
    episode_id = f'{GENERATOR}-s{seed}-{index:05d}'
    tcp, commanded = {}, {}
    for arm in ARMS:
        times = _sample_times(rng, t0, t0 + EPISODE_END, TCP_HZ, 2e-3)
        tcp[arm] = PoseStream(times, *_noisy(rng, *scene.plan(arm)(times - t0)))
        # Commanded samples are stamped at the time they are meant to be reached.
        commanded[arm] = PoseStream(times + COMMAND_LEAD_S, *scene.plan(arm)(times - t0 + COMMAND_LEAD_S))
    missing_agent = rng.random() < missing_agent_rate
    cameras = _cameras(rng, scene, t0, image_size, drop_rate, missing_agent)
    episode = Episode(episode_id=episode_id, source='umi' if missing_agent else 'sim_expert',
                      fragment_ids=('part_0', 'part_1'), geometry_hash=GENERATOR, calibration_version=GENERATOR,
                      tcp=tcp, segments=_segments(scene, episode_id, t0), cameras=cameras, commanded=commanded,
                      seed=seed, success=True, initial_condition_id=episode_id,
                      task_instruction=f'{scene.arm} arm: pick part_{scene.chosen}, place it at its goal, retreat')
    privileged = {'arm': np.int8(ARMS.index(scene.arm)), 'chosen': np.int8(scene.chosen), 'mode': np.int8(scene.mode),
                  'obstacle_xy': scene.obstacle, 'obstacle_radius': np.float64(OBSTACLE_RADIUS),
                  'fragments_xy': scene.fragments, 'start_xy': scene.start, 'target_xy': scene.target,
                  'via_xy': scene.via, 't0': np.float64(t0)}
    return episode, privileged


def make_episodes(count, *, seed=0, image_size=32, **kwargs):
    pairs = [make_episode(i, seed=seed, image_size=image_size, **kwargs) for i in range(count)]
    return [p[0] for p in pairs], {p[0].episode_id: p[1] for p in pairs}


# ---------------------------------------------------------------- evaluation (uses privileged GT)

def _point_segment_distance(point, a, b):
    ab = b - a
    t = np.clip(np.einsum('...i,...i', point - a, ab) / np.maximum(np.einsum('...i,...i', ab, ab), 1e-12), 0, 1)
    return np.linalg.norm(a + t[..., None] * ab - point, axis=-1)


def trajectory_metrics(path_xy, privileged):
    """path_xy: (K, 2) path starting at the current hand position."""
    obstacle, radius = privileged['obstacle_xy'], float(privileged['obstacle_radius'])
    distances = _point_segment_distance(obstacle, path_xy[:-1], path_xy[1:])
    closest = int(np.argmin(np.linalg.norm(path_xy - obstacle, axis=1)))
    direction = privileged['target_xy'] - privileged['start_xy']
    offset = path_xy[closest] - privileged['start_xy']
    side = float(np.sign(direction[0] * offset[1] - direction[1] * offset[0]))
    return {'hit': bool(distances.min() < radius), 'side': side, 'clearance_m': float(distances.min() - radius),
            'final_error_m': float(np.linalg.norm(path_xy[-1] - privileged['target_xy']))}


def window_at(dataset, skill_id, offset_s):
    """Per episode, the index of the window of `skill_id` whose observation time is closest to start + offset."""
    best = {}
    for index, ref in enumerate(dataset.windows):
        segment = dataset.episodes[ref.episode].segments[ref.segment]
        if segment.skill_id != skill_id:
            continue
        error = abs(dataset._cache[ref.episode]['grid'][ref.step] - (segment.start + offset_s))
        if ref.episode not in best or error < best[ref.episode][0]:
            best[ref.episode] = (error, index)
    return [index for _, index in sorted(best.values(), key=lambda item: item[1])]


def _paths(sample, prediction, arm):
    """World xy paths (n_samples, 17, 2) of `arm` from its anchor through the predicted waypoints."""
    anchor = sample['anchors'][ARMS.index(arm)]
    relative = prediction[..., hand_slice(arm)]
    positions = np.einsum('ij,snj->sni', anchor[:3, :3], relative[..., :3]) + anchor[:3, 3]
    start = np.repeat(anchor[:2, 3][None, None], len(prediction), axis=0)
    return np.concatenate([start, positions[..., :2]], axis=1), relative[..., 9]


def _place_report(samples, predictions, privileged):
    records = []
    for sample, prediction in zip(samples, predictions):
        info = privileged[sample['episode_id']]
        arm = ARMS[int(info['arm'])]
        paths, widths = _paths(sample, prediction, arm)
        for path, width in zip(paths, widths):
            records.append({'episode_id': sample['episode_id'], 'mode': int(info['mode']), 'final_width_m': float(width[-1]),
                            **trajectory_metrics(path, info)})
    hits = np.array([r['hit'] for r in records])
    sides = np.array([r['side'] for r in records])
    clear = sides[~hits]
    left_fraction = float((clear > 0).mean()) if len(clear) else None
    per_episode = {}
    for record in records:
        if not record['hit']:
            per_episode.setdefault(record['episode_id'], set()).add(record['side'])
    agreement = np.mean([r['side'] == r['mode'] for r in records if not r['hit']]) if len(clear) else None
    return {'episodes': len(samples), 'trajectories': len(records), 'obstacle_hit_rate': float(hits.mean()),
            'left_fraction_of_clear': left_fraction,
            'episodes_with_both_modes': float(np.mean([len(per_episode.get(s['episode_id'], ())) == 2 for s in samples])),
            'mode_agreement_of_clear': None if agreement is None else float(agreement),
            'final_error_m_median': float(np.median([r['final_error_m'] for r in records])),
            'final_width_m_median': float(np.median([r['final_width_m'] for r in records])),
            'min_clearance_m_median': float(np.median([r['clearance_m'] for r in records]))}


def _pick_report(samples, predictions, privileged):
    correct, errors = [], []
    for sample, prediction in zip(samples, predictions):
        info = privileged[sample['episode_id']]
        paths, _ = _paths(sample, prediction, ARMS[int(info['arm'])])
        ends = paths[:, -1]
        distance = np.linalg.norm(ends[:, None] - info['fragments_xy'][None], axis=-1)
        correct.extend(np.argmin(distance, axis=1) == int(info['chosen']))
        errors.extend(distance[:, int(info['chosen'])])
    return {'episodes': len(samples), 'trajectories': len(correct), 'chosen_fragment_rate': float(np.mean(correct)),
            'final_error_m_median': float(np.median(errors))}


def _arm_report(samples, predictions, privileged):
    """Raw predicted motion (before S1 forces a hold) of the active and the inactive arm."""
    active, inactive = [], []
    for sample, prediction in zip(samples, predictions):
        arm = ARMS[int(privileged[sample['episode_id']]['arm'])]
        for name, store in ((arm, active), (ARMS[1 - ARMS.index(arm)], inactive)):
            paths, _ = _paths(sample, prediction, name)
            store.extend(np.linalg.norm(paths - paths[:, :1], axis=-1).max(axis=1))
    return {'active_motion_m_median': float(np.median(active)), 'inactive_motion_m_median': float(np.median(inactive)),
            'inactive_motion_m_p90': float(np.percentile(inactive, 90))}


PHASES = {'pick_first': ('pick', 0.0), 'place_first': ('place', 0.0), 'place_mid': ('place', 0.7),
          'place_late': ('place', 1.6)}


def evaluate(predict, dataset, privileged, *, n_samples=8):
    """Metrics per phase on held-out episodes. predict(samples, n) -> (len, n, 16, 20) unnormalized actions.

    pick_first: which identical-looking fragment the hand goes to (fragment-pose conditioning).
    place_first: carry about to start, mode unobservable: hits, both modes, final error.
    place_mid: carry under way, mode visible in the history: agreement with the demo's side.
    place_late: release under way, future past the segment end (hold padding): final error and width.
    arms: raw motion of the requested vs the other arm over all evaluated windows.
    """
    report, everything = {}, ([], [])
    for phase, (skill, offset) in PHASES.items():
        samples = [dataset[i] for i in window_at(dataset, skill, offset)]
        if not samples:
            continue
        predictions = np.asarray(predict(samples, n_samples))
        everything[0].extend(samples)
        everything[1].extend(predictions)
        report[phase] = (_pick_report if skill == 'pick' else _place_report)(samples, predictions, privileged)
        report[phase]['observation_offset_s'] = offset
    if everything[0]:
        report['arms'] = _arm_report(everything[0], everything[1], privileged)
    return report


def demo_report(dataset, privileged):
    """The same metrics on the recorded demonstrations (upper reference for the policies)."""
    def predict(samples, n_samples):
        return np.stack([np.repeat(s['action'][None], n_samples, axis=0) for s in samples])
    return evaluate(predict, dataset, privileged, n_samples=1)
