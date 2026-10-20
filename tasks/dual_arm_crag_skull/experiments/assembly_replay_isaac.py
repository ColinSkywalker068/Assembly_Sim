"""A recorded Isaac Sim replay of a task's assembly plan.

From the task's start, settled on the backend (the anchor in its fixture, the robot at home), each fragment of the plan
in turn is carried (`carry`): up from where it rests until it clears everything, over while turning to its goal
orientation, down where it is just clear (along its insertion line) of the anchor and the fragments already placed,
and in along its insertion direction. It is carried kinematically: set to its pose and zero velocity every physics
step, by no robot, and it keeps colliding, so whatever it touches on the way reacts. Then it is let go and the stack
settles for --settle-s before the next one, and the drift of every placed fragment from its goal is recorded. Two
cameras of the replay's own (an overview of the table and a close-up of the skull) render at --frame-hz; each becomes
a video, and the close-up frame after each settle is that step's still. The fragments are painted in the colours they
are named by everywhere else (DexAgent's export: the red anchor, orange, blue, ...), not the layout's own colours,
unless --layout-colours.

  CUDA_VISIBLE_DEVICES=N OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. $ISAAC_PYTHON -B -m \\
      tasks.dual_arm_crag_skull.experiments.assembly_replay_isaac --task runs/baseline_tasks/skull_base.json --out DIR

Writes DIR/<camera>/*.jpg, DIR/<camera>.mp4, DIR/step_<k>.jpg and DIR/replay.json (written before Kit shuts down).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

CARRY_M_S = 0.3  # rising, gliding and coming down
INSERT_M_S = 0.05  # the last straight move
TURN_DEG_S = 90.0
MIN_MOVE_S = 0.3
CLEARANCE_M = 0.02
FILM_WIDTH_MM = 20.955
CAMERAS = {'overview': {'position': (0.0, -1.0, 1.32), 'target': (0.0, 0.05, 0.76), 'focal_mm': 20.0},
           'close': {'position': (0.30, -0.40, 1.02), 'target': (0.0, 0.01, 0.80), 'focal_mm': 30.0}}


def display_colours(task):
    """{fragment id: (colour name, rgb)}: the colours DexAgent's export names the fragments by, as rendered (its MJCF
    stores 0.45 of them for its lights), dimmed a little so white is not blown out."""
    from ..baselines.dexagent.export import object_ids
    return {key: (name, tuple(min(1.0, 0.92 * value / 0.45) for value in rgb))
            for key, (_, name, rgb) in object_ids(task).items()}


def paint(stage, colours):
    """Set each fragment's visual material (/World/Looks/<id>, built by the backend) to its display colour."""
    from pxr import Gf, UsdShade
    for key, (_, rgb) in colours.items():
        shader = UsdShade.Shader(stage.GetPrimAtPath(f'/World/Looks/{key}/Shader'))
        if not shader:
            raise ValueError(f'no visual material for {key}')
        shader.GetInput('diffuseColor').Set(Gf.Vec3f(*rgb))


def look_at(position, target, up=(0.0, 0.0, 1.0)):
    """World pose of a USD camera at `position` looking at `target` (it looks along its -z, image up is its +y)."""
    z = np.asarray(position, dtype=float) - np.asarray(target, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, dtype=float), z)
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    pose[:3, 3] = position
    return pose


def carry(vertices, start, goal, insertion, placed_points, *, top_z, clearance=CLEARANCE_M):
    """(points, durations): the world positions [start, above it, above where it starts to go in, that point, goal] of a
    carried fragment (`vertices` in its own frame, `start` and `goal` world poses) and the times of the four moves
    between them. It rises until its lowest point clears `top_z` by `clearance` at both orientations, glides, comes
    down where backing out of its goal against `insertion` leaves it `clearance` clear of every placed point along
    that line, and goes in straight, slowly."""
    out = -np.asarray(insertion, dtype=float)
    at_goal = vertices @ goal[:3, :3].T + goal[:3, 3]
    ahead = float((np.asarray(placed_points) @ out).max()) - float((at_goal @ out).min())
    begin = goal[:3, 3] + (max(ahead, 0.0) + clearance) * out
    lift = max(top_z + clearance - float((vertices @ pose[:3, :3].T)[:, 2].min()) for pose in (start, goal))
    p0 = np.array(start[:3, 3], dtype=float)
    points = [p0, np.array([p0[0], p0[1], lift]), np.array([begin[0], begin[1], max(lift, begin[2])]), begin,
              np.array(goal[:3, 3], dtype=float)]
    lengths = [float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:])]
    turn = np.degrees((Rotation.from_matrix(goal[:3, :3]) * Rotation.from_matrix(start[:3, :3]).inv()).magnitude())
    durations = [max(lengths[0] / CARRY_M_S, MIN_MOVE_S), max(lengths[1] / CARRY_M_S, turn / TURN_DEG_S, MIN_MOVE_S),
                 max(lengths[2] / CARRY_M_S, MIN_MOVE_S), lengths[3] / INSERT_M_S]
    return points, durations


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def pose_at(time_s, points, durations, start, goal):
    """(position, rotation matrix) of the carried fragment `time_s` into its carry: it eases along each move, keeps its
    start orientation while rising, turns while gliding and holds its goal orientation from then on."""
    edges = np.concatenate([[0.0], np.cumsum(durations)])
    r0, r1 = Rotation.from_matrix(start[:3, :3]), Rotation.from_matrix(goal[:3, :3])
    if time_s >= edges[-1]:
        return np.array(points[-1], dtype=float), r1.as_matrix()
    k = min(max(int(np.searchsorted(edges, time_s, side='right')) - 1, 0), len(durations) - 1)
    s = _smooth((time_s - edges[k]) / durations[k]) if durations[k] > 0 else 1.0
    position = points[k] + s * (points[k + 1] - points[k])
    if k == 0:
        rotation = r0
    elif k == 1:
        rotation = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))([s])[0]
    else:
        rotation = r1
    return position, rotation.as_matrix()


def _wxyz(matrix):
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return np.array([w, x, y, z]) * (1.0 if w >= 0 else -1.0)


class Recorder:
    """The replay's own cameras: an rgb render product each; `frame()` renders them all and saves a JPEG per camera."""

    def __init__(self, backend, out, size):
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        from core.workcell.environment import set_transform
        self.backend, self.out, self.count, self.products, self.annotators = backend, out, 0, [], {}
        for name, spec in CAMERAS.items():
            camera = UsdGeom.Camera.Define(backend.stage, f'/World/ReplayCameras/{name}')
            camera.CreateFocalLengthAttr(spec['focal_mm'])
            camera.CreateHorizontalApertureAttr(FILM_WIDTH_MM)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
            pose = look_at(spec['position'], spec['target'])
            set_transform(UsdGeom.Xformable(camera.GetPrim()), pose[:3, 3], _wxyz(pose[:3, :3]))
            product = rep.create.render_product(str(camera.GetPath()), tuple(size))
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach(product)
            self.products.append(product)
            self.annotators[name] = annotator
            (out / name).mkdir(parents=True, exist_ok=False)

    def frame(self):
        import omni.replicator.core as rep
        import omni.timeline
        from PIL import Image

        from ..sim.backend import RENDER_PASSES
        omni.timeline.get_timeline_interface().pause()
        try:
            for _ in range(RENDER_PASSES):
                rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=True)
        finally:
            self.backend.world.play()  # a bare timeline.play() is deferred to the next app update
        if not self.backend.world.is_playing():
            raise RuntimeError('the simulation did not resume after rendering')
        saved = {}
        for name, annotator in self.annotators.items():
            image = np.ascontiguousarray(np.asarray(annotator.get_data())[:, :, :3], dtype=np.uint8)
            saved[name] = self.out / name / f'{self.count:05d}.jpg'
            Image.fromarray(image).save(saved[name], quality=90)
        self.count += 1
        return saved

    def encode(self, fps):
        import imageio.v2 as imageio
        for name in self.annotators:
            with imageio.get_writer(self.out / f'{name}.mp4', fps=fps, codec='libx264', macro_block_size=16,
                                    ffmpeg_params=['-crf', '24', '-pix_fmt', 'yuv420p']) as writer:
                for path in sorted((self.out / name).glob('*.jpg')):
                    writer.append_data(imageio.imread(path))

    def close(self):
        for annotator in self.annotators.values():
            annotator.detach()
        for product in self.products:
            product.destroy()


def _drift(task, poses, keys):
    from ..sim.backend import _turn_deg
    drift = {}
    for key in keys:
        goal, now = task.fragment(key).goal_world_T_fragment, np.asarray(poses[key])
        drift[key] = [1000 * float(np.linalg.norm(now[:3, 3] - goal[:3, 3])), float(_turn_deg(goal, now))]
    return drift


def _points(task, key, pose):
    with np.load(task.fragment(key).mesh) as data:
        vertices = np.asarray(data['v'], dtype=float)
    return vertices, vertices @ np.asarray(pose)[:3, :3].T + np.asarray(pose)[:3, 3]


def replay(backend, recorder, task, *, frame_hz, settle_s):
    """Carry every fragment of the plan in and let it settle, rendering at frame_hz; returns the per-step record."""
    view, ids = backend._fragment_view, backend._fragment_ids  # the backend has no public way to carry a fragment
    per_frame = max(1, round(backend.physics_hz / frame_hz))
    steps, placed, physics = [], [task.anchor_id], 0

    def advance():
        nonlocal physics
        backend.world.step(render=False)
        physics += 1
        if physics % per_frame == 0:
            recorder.frame()

    recorder.frame()
    for number, step in enumerate(task.assembly_plan, 1):
        key, index = step.fragment_id, np.array([ids.index(step.fragment_id)])
        poses = backend.privileged_fragment_poses()
        vertices, _ = _points(task, key, poses[key])
        placed_points = np.concatenate([_points(task, other, poses[other])[1] for other in placed])
        top_z = max(float(_points(task, other, poses[other])[1][:, 2].max()) for other in task.fragment_ids)
        start, goal = np.asarray(poses[key]), task.fragment(key).goal_world_T_fragment
        points, durations = carry(vertices, start, goal, step.insertion_direction, placed_points, top_z=top_z)
        for tick in range(1, int(np.ceil(sum(durations) * backend.physics_hz)) + 1):
            position, rotation = pose_at(tick / backend.physics_hz, points, durations, start, goal)
            view.set_world_poses(positions=position[None], orientations=_wxyz(rotation)[None], indices=index)
            view.set_velocities(np.zeros((1, 6)), indices=index)
            advance()
        placed.append(key)
        for _ in range(int(round(settle_s * backend.physics_hz))):
            advance()
        still = recorder.frame()['close']
        shutil.copyfile(still, recorder.out / f'step_{number}.jpg')
        drift = _drift(task, backend.privileged_fragment_poses(), [k for k in placed if k != task.anchor_id])
        steps.append({'step': number, 'fragment': key, 'insertion_direction': list(step.insertion_direction),
                      'insertion_m': float(np.linalg.norm(points[4] - points[3])), 'carry_s': sum(durations),
                      'drift_mm_deg': drift, 'worst_mm': max(w[0] for w in drift.values()),
                      'worst_deg': max(w[1] for w in drift.values()), 'still': f'step_{number}.jpg'})
        print(json.dumps({k: steps[-1][k] for k in ('step', 'fragment', 'insertion_m', 'worst_mm', 'worst_deg')}),
              flush=True)
    for _ in range(int(round(frame_hz))):  # a second's hold at the end
        recorder.frame()
    return steps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frame-hz', type=float, default=20.0)
    parser.add_argument('--settle-s', type=float, default=1.0)
    parser.add_argument('--size', type=int, nargs=2, default=(1280, 960), metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--layout-colours', action='store_true', help="keep the layout's own fragment colours")
    args = parser.parse_args(argv)
    from ..__main__ import DEFAULT_CONFIG
    from ..baselines.task import AssemblyTask, verify_task
    from ..sim.backend import IsaacAssemblyBackend
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; a replay is never overwritten')
    config = json.loads(DEFAULT_CONFIG.read_text())
    task = verify_task(AssemblyTask.from_record(json.loads(args.task.read_text())))
    backend, recorder, code = None, None, 1
    try:
        backend = IsaacAssemblyBackend(config, cameras=False)
        backend.reset(task)
        colours = None if args.layout_colours else display_colours(task)
        if colours is not None:
            paint(backend.stage, colours)
        args.out.mkdir(parents=True)
        recorder = Recorder(backend, args.out, args.size)
        began = time.time()
        steps = replay(backend, recorder, task, frame_hz=args.frame_hz, settle_s=args.settle_s)
        recorder.encode(args.frame_hz)
        record = {'task_id': task.task_id, 'layout_sha256': task.layout_sha256, 'frame_hz': args.frame_hz,
                  'settle_s': args.settle_s, 'size': list(args.size), 'frames': recorder.count, 'cameras': CAMERAS,
                  'carry': {'m_s': CARRY_M_S, 'insert_m_s': INSERT_M_S, 'clearance_m': CLEARANCE_M},
                  'colours': colours or 'layout', 'start': backend.start.to_record(), 'steps': steps,
                  'wall_s': time.time() - began,
                  'videos': [f'{name}.mp4' for name in CAMERAS]}
        (args.out / 'replay.json').write_text(json.dumps(record, indent=1) + '\n')  # before Kit shuts the process
        print(f"REPLAY OK: {len(steps)} steps, {recorder.count} frames, worst drift "
              f"{max(s['worst_mm'] for s in steps):.4f} mm {max(s['worst_deg'] for s in steps):.4f} deg", flush=True)
        code = 0
    except BaseException:
        traceback.print_exc()
        print('REPLAY FAIL', flush=True)
        raise
    finally:
        if backend is None:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        if recorder is not None:
            recorder.close()
        backend.close()
        backend.shutdown(code)


if __name__ == '__main__':
    main()
