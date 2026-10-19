"""Runs inside the DexAgent venv (Python 3.11) on a pinned clone: the export steps that need DexAgent's own code.

    <dexagent python> -I -B worker.py <command> @spec.json

stage     stage the fragment assets on the clone's bench (stage_on_bench) with video_cam from camera.json, settle
          the fragments with the robot clamped, write their rest poses to keyframe 0 and probe which arm reaches
          each fragment (object_reach_check).
media     render the stand-in video from that scene (robot hidden) and encode it with DexAgent's write_video,
          record its identity (video_identity), segment frame 0 with segment_source_frame through DexAgent's
          reference-evidence registry (which writes reference_evidence_recipes.json), and run the Stage 2 checks
          whose reports scene_recon.json carries.
check     in a fresh process: the two Stage 3 pre-agent gates exactly as run_stage3.py calls them
          (check_video_identity, revalidate_source_recipes), the Stage 1 verifiers, the Stage 2 manifest check
          (informational), initial_keyframe_state_check, the rigid runtime compile, a hand scene, and the
          goal-pose interpenetration of exact boxes versus convex hulls.
delivery  agent.artifacts.accepted_delivery of a Stage 3 run folder, re-checked from a fresh clone against the
          exported Stage 2 (execution_scene_integrity_check, physics_recording_check).

The last stdout line is one JSON object {ok, result | error}; everything DexAgent prints goes to stderr. This
file must never import the Assembly_Sim package.
"""
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tool(clone, name, args, cwd=None):
    from agent.library import call_tool, load_index
    lib = clone / 'tool_library'
    raw = call_tool(lib, load_index(lib), name, args, cfg=None, cwd=cwd)
    if raw.get('ok') is not True:
        return {'ok': False, 'issues': [f'{name} could not run: {str(raw.get("error"))[:1500]}']}
    return raw['result']


def _render_video(spec, scene, path):
    """The stand-in video: the Stage 2 scene through video_cam with the robot hidden, fragments posed per frame."""
    import mujoco
    import numpy as np
    from tool_library.common.video import write_video
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for geom in range(model.ngeom):
        if model.body(int(model.geom_bodyid[geom])).name.startswith('robot_'):
            model.geom_group[geom] = 5
    option = mujoco.MjvOption()
    option.geomgroup[5] = 0
    with np.load(spec['video_plan'], allow_pickle=False) as plan:
        poses, order, fps = plan['poses'], [str(v) for v in plan['objects']], float(plan['fps'])
    joints = {obj['id']: obj['free_joint'] for obj in spec['objects']}
    moving = [(index, int(model.jnt_qposadr[model.joint(joints[name]).id])) for index, name in enumerate(order)
              if joints[name]]
    renderer = mujoco.Renderer(model, spec['camera']['height'], spec['camera']['width'])

    def frames():
        previous, image = None, None
        for row in poses:
            if previous is None or not np.array_equal(row, previous):
                for index, address in moving:
                    data.qpos[address:address + 7] = row[index]
                mujoco.mj_kinematics(model, data)
                mujoco.mj_camlight(model, data)
                renderer.update_scene(data, camera='video_cam', scene_option=option)
                image = renderer.render().copy()
                previous = row
            yield image
    started = time.time()
    write_video(frames(), path, fps=fps)
    renderer.close()
    return {'frames': int(len(poses)), 'fps': fps, 'render_s': round(time.time() - started, 2),
            'hidden': 'robot geoms (bodies named robot_*)'}


def _joint_sizes(model, joint):
    import mujoco
    kind = int(model.jnt_type[joint])
    return {int(mujoco.mjtJoint.mjJNT_FREE): (7, 6), int(mujoco.mjtJoint.mjJNT_BALL): (4, 3)}.get(kind, (1, 1))


def _settle(scene, objects, seconds, window_s=0.5, limit_s=10.0):
    """Simulate keyframe 0 with every robot joint clamped at its keyframe value until the free fragments rest (no
    motion above 0.2 mm or 0.2 deg over `window_s`, after at least `seconds`); write the rest poses to keyframe 0."""
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    spec = mujoco.MjSpec.from_file(str(scene))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    key = data.qpos.copy()
    robot_q, robot_v = [], []
    for joint in range(model.njnt):
        if model.body(int(model.jnt_bodyid[joint])).name.startswith('robot_'):
            nq, nv = _joint_sizes(model, joint)
            robot_q += range(int(model.jnt_qposadr[joint]), int(model.jnt_qposadr[joint]) + nq)
            robot_v += range(int(model.jnt_dofadr[joint]), int(model.jnt_dofadr[joint]) + nv)
    free = {o['id']: int(model.jnt_qposadr[model.joint(o['free_joint']).id]) for o in objects if o['free_joint']}
    start = {oid: data.qpos[a:a + 7].copy() for oid, a in free.items()}

    def moved(a, b):
        turn = (Rotation.from_quat(a[[4, 5, 6, 3]]).inv() * Rotation.from_quat(b[[4, 5, 6, 3]])).magnitude()
        return float(np.linalg.norm(a[:3] - b[:3])), float(np.degrees(turn))
    window = max(1, int(round(window_s / model.opt.timestep)))
    elapsed, before = 0.0, {oid: data.qpos[a:a + 7].copy() for oid, a in free.items()}
    while True:
        for _ in range(window):
            mujoco.mj_step(model, data)
            data.qpos[robot_q] = key[robot_q]
            data.qvel[robot_v] = 0.0
        elapsed += window * model.opt.timestep
        now = {oid: data.qpos[a:a + 7].copy() for oid, a in free.items()}
        motion = {oid: moved(before[oid], now[oid]) for oid in free}
        resting = all(d < 2e-4 and r < 0.2 for d, r in motion.values())
        if (elapsed >= seconds and resting) or elapsed >= limit_s:
            break
        before = now
    if not resting:
        raise ValueError(f'fragments still move after {limit_s} s of settling: {motion}')
    settled = {}
    for oid, address in free.items():
        pose = data.qpos[address:address + 7].copy()
        pose[3:] /= np.linalg.norm(pose[3:])
        settled[oid] = pose
    qpos = np.array(spec.keys[0].qpos, dtype=float)
    for oid, address in free.items():
        qpos[address:address + 7] = settled[oid]
    spec.keys[0].qpos = qpos.tolist()
    Path(scene).write_text(spec.to_xml())
    change = {oid: dict(zip(('position_m', 'rotation_deg'), moved(start[oid], settled[oid]))) for oid in free}
    return {oid: pose.tolist() for oid, pose in settled.items()}, {'simulated_s': round(elapsed, 3), 'change': change}


def stage(clone, spec):
    import mujoco
    import numpy as np
    from tool_library.common import rigcam
    from tool_library.skills.stage_on_bench import stage_on_bench
    stage2 = Path(spec['stage2_dir'])
    camera = spec['camera']
    cam = rigcam.camera_from_mujoco(camera['width'], camera['height'], camera['fovy_deg'], camera['pos'],
                                    camera['quat_wxyz'], extra={'source': 'given', 'note': 'Assembly_Sim stand-in '
                                                                'camera straight above the table (rig frame)'})
    rigcam.save(cam, stage2 / 'camera.json')
    scene = stage2 / 'scene.xml'
    staged = stage_on_bench(str(clone / 'setup' / 'scene.xml'), spec['assets'], str(scene),
                            rig={'origin': [0.0, 0.0, 0.0], 'R': np.eye(3).tolist()},
                            cameras={'video_cam': str(stage2 / 'camera.json')})
    settled, report = _settle(scene, spec['objects'], spec['settle_s'])
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    poses = {}
    for obj in spec['objects']:
        body = model.body(obj['root_body']).id
        poses[obj['id']] = [*map(float, data.xpos[body]), *map(float, data.xquat[body])]
    reach = _tool(clone, 'object_reach_check', {'scene': str(scene)})
    found = {obj['id']: (reach.get('objects') or {}).get(obj['root_body']) for obj in spec['objects'] if obj['free_joint']}
    witnesses = {oid: None if row is None else {k: row.get(k) for k in ('reachable_sides', 'best_side', 'residual_mm',
                                                                         'status')} for oid, row in found.items()}
    return {'settled': poses, 'settle': report, 'bodies': {a['name']: a['bodies'] for a in staged['assets']},
            'nq': staged['nq_after'], 'reach': witnesses,
            'reach_report': {k: reach.get(k) for k in ('ok', 'verdict', 'issues')}}


def media(clone, spec):
    from agent.input_identity import video_identity
    from agent.library import call_tool, load_index
    from agent.reference_evidence import begin
    stage2, video = Path(spec['stage2_dir']), Path(spec['video'])
    (stage2 / 'masks').mkdir(parents=True, exist_ok=True)
    rendering = _render_video(spec, stage2 / 'scene.xml', video)
    identity = video_identity(video)
    lib = clone / 'tool_library'
    index = load_index(lib)
    authority = begin(stage2, video)
    objects, sources = {}, set()
    for obj in spec['objects']:
        args = {'video': str(video), 'frame_index': 0, 'box': obj['box'], 'out_dir': str(stage2 / 'segment' / obj['id'])}
        token = authority.prepare('segment_source_frame', args)
        raw = call_tool(lib, index, 'segment_source_frame', args, cfg=None, cwd=stage2)
        if raw.get('ok') is not True:
            raise ValueError(f'segment_source_frame failed for {obj["id"]}: {raw.get("error")}')
        authority.complete(token, raw)
        result = raw['result']
        shutil.copyfile(result['mask'], stage2 / 'masks' / f'{obj["id"]}.png')
        sources.add(_digest(result['source_frame']))
        if not (stage2 / 'frame0.png').exists():
            shutil.copyfile(result['source_frame'], stage2 / 'frame0.png')
        objects[obj['id']] = {'box': obj['box'], 'foreground_pixels': result['foreground_pixels']}
    if len(sources) != 1 or _digest(stage2 / 'frame0.png') not in sources:
        raise ValueError('segment_source_frame decoded different frame-0 bytes for different objects')
    scene = str(stage2 / 'scene.xml')
    silhouette = _tool(clone, 'silhouette_iou_check', {
        'scene': scene, 'masks': {o['root_body']: str(stage2 / 'masks' / f'{o["id"]}.png') for o in spec['objects']},
        'camera': 'video_cam', 'out_dir': str(stage2 / 'checks' / 'silhouette')})
    settle = _tool(clone, 'settle_stability_check', {'scene': scene, 'seconds': 2.0})
    compare = _tool(clone, 'render_frame_compare', {'scene_xml': scene, 'camera': str(stage2 / 'camera.json'),
                                                    'frame': str(stage2 / 'frame0.png'),
                                                    'out_png': str(stage2 / 'frame0_compare.png')})
    for obj in spec['objects']:
        root = obj['root_body']
        iou = (silhouette.get('iou') or {}).get(root)
        drift = (settle.get('drift') or {}).get(root)
        settle_report = ({'ok': True, 'verdict': 'fixed body (no free joint)'} if not obj['free_joint'] else
                         {'ok': bool(drift) and root not in settle.get('failures', []),
                          'pos_drift_m': (drift or {}).get('pos_m'), 'rot_deg': (drift or {}).get('rot_deg'),
                          'verdict': (drift or {}).get('verdict')})
        objects[obj['id']].update({
            'iou': None if iou is None else round(float(iou), 4),
            'verifiers': {'silhouette_iou_check': {'ok': iou is not None and root not in silhouette.get('failures', []),
                                                   'iou': iou, 'source': 'silhouette_iou_check (DexAgent)'},
                          'settle_stability_check': {**settle_report, 'source': 'settle_stability_check (DexAgent)'}}})
    return {'identity': identity, 'objects': objects, 'rendering': rendering,
            'scene_checks': {'silhouette_iou_check': {k: silhouette.get(k) for k in ('ok', 'iou', 'failures',
                                                                                   'warnings', 'camera_suspect')},
                             'settle_stability_check': {k: settle.get(k) for k in ('ok', 'failures', 'diverged')},
                             'render_frame_compare': compare},
            'versions': versions()}


def versions():
    import cv2
    import imageio_ffmpeg
    import mujoco
    import numpy
    import scipy
    return {'python': sys.version.split()[0], 'numpy': numpy.__version__, 'scipy': scipy.__version__,
            'mujoco': mujoco.__version__, 'opencv': cv2.__version__, 'ffmpeg': imageio_ffmpeg.get_ffmpeg_version()}


def _goal_penetration(bodies, hull):
    """Deepest fragment-fragment penetration (m) with every fragment at its goal: exact boxes or convex hulls."""
    import mujoco
    spec = mujoco.MjSpec()
    for item in bodies:
        x, y, z, *quat = item['goal']
        if hull:
            spec.add_mesh(name=f'm_{item["id"]}', file=item['mesh'])
            body = spec.worldbody.add_body(name=item['id'], pos=[x, y, z], quat=quat)
            body.add_freejoint()  # MuJoCo never collides two bodies welded to the world
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'm_{item["id"]}')
        else:
            frame = spec.worldbody.add_frame(pos=[x, y, z], quat=quat)
            spec.attach(mujoco.MjSpec.from_file(item['asset']), prefix=f'{item["id"]}_', frame=frame)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    deepest, pairs = 0.0, {}
    for contact in data.contact[:data.ncon]:
        a, b = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        if a == b:
            continue
        depth = max(0.0, -float(contact.dist))
        key = tuple(sorted((model.body(a).name, model.body(b).name)))
        pairs[key] = max(pairs.get(key, 0.0), depth)
        deepest = max(deepest, depth)
    return {'max_penetration_m': round(deepest, 6), 'pairs_over_1mm': sum(d > 0.001 for d in pairs.values()),
            'pairs_in_contact': len(pairs), 'worst': sorted(([*k, round(d, 5)] for k, d in pairs.items()),
                                                            key=lambda r: -r[2])[:5]}


def _runtime(scene, recon):
    from tool_library.common import episode as E
    from tool_library.common import rigid_robot as RR
    started = time.time()
    model, pairs = E.build_execution_model(scene, scene_recon_json=recon)
    runtime = RR.compile_rigid(model)
    return {'ok': True, 'proposal_nq': int(model.nq), 'runtime_nq': int(runtime.model.nq),
            'runtime_ngeom': int(runtime.model.ngeom), 'hand_object_pairs': int(pairs),
            'runtime_npair': int(runtime.model.npair), 'compile_s': round(time.time() - started, 2)}


def _hand_scene(clone, recon, probe):
    scene = _tool(clone, 'hand_scene_from_recon', {'scene_recon_json': str(recon), 'side': probe['side'],
                                                   'target': probe['target']})
    objects = scene.get('objects') or []
    return {'ok': scene.get('ok') is not False and bool(objects), 'objects': len(objects),
            'issues': list(scene.get('issues') or [])}


def check(clone, spec):
    from agent.input_identity import check_video_identity
    from agent.reference_evidence import revalidate_source_recipes
    video, semantic, recon = Path(spec['video']), Path(spec['semantic']), Path(spec['recon'])
    stage2 = recon.parent
    semantics = json.loads(semantic.read_text())
    reports = {}

    def guarded(name, function):
        try:
            reports[name] = function()
        except Exception as exc:  # noqa: BLE001 - a crashing check is a failed check, with its reason
            reports[name] = {'ok': False, 'issues': [f'crashed: {type(exc).__name__}: {exc}'],
                             'traceback': traceback.format_exc()[-3000:]}
    guarded('source_video_identity_check', lambda: check_video_identity(video, semantic))
    guarded('source_reference_provenance_check',
            lambda: revalidate_source_recipes(stage2, video, library=clone / 'tool_library', cfg=None))
    guarded('scene_semantic_schema_check', lambda: _tool(clone, 'scene_semantic_schema_check', {'path': str(semantic)}))
    guarded('scene_semantic_consistency_check', lambda: _tool(clone, 'scene_semantic_consistency_check', {
        'path': str(semantic), 'prompt': spec['prompt'], 'video_duration_s': semantics['input_identity']['duration_s']}))
    guarded('goal_language_coverage_check', lambda: _tool(clone, 'goal_language_coverage_check', {'path': str(semantic)}))
    guarded('scene_recon_manifest_check', lambda: _tool(clone, 'scene_recon_manifest_check', {
        'path': str(recon), 'scene_semantic_json': str(semantic), 'library_index': str(clone / 'tool_library' /
                                                                                       'index.json')}))
    guarded('initial_keyframe_state_check', lambda: _tool(clone, 'initial_keyframe_state_check', {
        'scene': str(stage2 / 'scene.xml'), 'setup': str(clone / 'setup')}))
    guarded('rigid_runtime_compile', lambda: _runtime(str(stage2 / 'scene.xml'), str(recon)))
    probe = spec.get('hand_scene_probe')
    if probe:
        guarded('hand_scene_from_recon', lambda: _hand_scene(clone, recon, probe))
    guarded('goal_penetration', lambda: {'ok': True, 'exact_boxes': _goal_penetration(spec['goal_bodies'], False),
                                         'convex_hulls': _goal_penetration(spec['goal_bodies'], True)})
    return reports


def delivery(clone, spec):
    """DexAgent's acceptance of a Stage 3 run, re-checked from a fresh clone against our own Stage 2: the delivered
    scene must keep our objects (execution_scene_integrity_check against the exported scene_recon.json) and the
    recording must replay in physics (physics_recording_check)."""
    from agent.artifacts import accepted_delivery, verify_delivery
    directory = accepted_delivery(Path(spec['public']))
    verify = verify_delivery(directory)
    scene, trajectory = str(directory / 'scene.xml'), str(directory / 'trajectory.npz')
    checks = {'execution_scene_integrity_check': _tool(clone, 'execution_scene_integrity_check', {
                  'scene': scene, 'scene_recon_json': spec['recon'], 'trajectory': trajectory}),
              'physics_recording_check': _tool(clone, 'physics_recording_check', {
                  'scene': scene, 'trajectory': trajectory, 'scene_recon_json': spec['recon']})}
    summary = {name: {'ok': report.get('ok') is True, 'issues': list(report.get('issues') or [])[:5]}
               for name, report in checks.items()}
    return {'directory': str(directory), 'verify': verify, 'checks': summary,
            'ok': verify.get('ok') is True and all(row['ok'] for row in summary.values())}


COMMANDS = {'stage': stage, 'media': media, 'check': check, 'delivery': delivery}


def main(argv):
    if len(argv) != 3 or argv[1] not in COMMANDS or not argv[2].startswith('@'):
        print(json.dumps({'ok': False, 'error': f'usage: worker.py {"|".join(COMMANDS)} @spec.json'}))
        return 2
    sys.stdout.flush()
    real = os.dup(1)
    os.dup2(2, 1)
    try:
        spec = json.loads(Path(argv[2][1:]).read_text())
        clone = Path(spec['clone']).resolve()
        sys.path.insert(0, str(clone))
        os.environ.setdefault('MUJOCO_GL', 'osmesa')
        out = {'ok': True, 'result': COMMANDS[argv[1]](clone, spec)}
    except Exception as exc:  # noqa: BLE001 - reported to the caller as one JSON line
        out = {'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()[-4000:]}
    sys.stdout.flush()
    with os.fdopen(real, 'w') as stream:
        stream.write(json.dumps(out, default=str) + '\n')
    return 0 if out['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
