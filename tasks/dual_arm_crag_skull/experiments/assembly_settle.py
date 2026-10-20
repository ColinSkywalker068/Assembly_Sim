"""Does every partial assembly stay put under gravity? MuJoCo, on a DexAgent export's scene (box collision).

For an insertion order, each prefix (the fragments placed so far, with the anchor the scene holds fixed) is put at its
goals, every other fragment far off along x, the robot held at keyframe 0 as DexAgent's own settle holds it (the
joints of bodies named robot_*), and stepped for `seconds`; a placed fragment's drift is its worst distance (mm) and rotation (deg) from its goal over the run. Every
prefix starts again from the goals, so a set's drift depends on the set alone. The orders are the export's own (its
task's assembly plan) or, with --all-orders, every order the layout's plan directions allow
(interlock_repair.insertion_orders); each order's worst step is read against the task's tolerances.

  envs/dexagent/bin/python -B -m tasks.dual_arm_crag_skull.experiments.assembly_settle --export EXPORT_DIR \
      --out OUT.json [--all-orders] [--seconds 2] [--workers 12]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

FAR_M = 5.0


def _robot_dofs(model):
    """qpos and qvel indices of the joints of bodies named robot_*."""
    import mujoco
    sizes = {int(mujoco.mjtJoint.mjJNT_FREE): (7, 6), int(mujoco.mjtJoint.mjJNT_BALL): (4, 3)}
    q, v = [], []
    for joint in range(model.njnt):
        if model.body(int(model.jnt_bodyid[joint])).name.startswith('robot_'):
            nq, nv = sizes.get(int(model.jnt_type[joint]), (1, 1))
            q += range(int(model.jnt_qposadr[joint]), int(model.jnt_qposadr[joint]) + nq)
            v += range(int(model.jnt_dofadr[joint]), int(model.jnt_dofadr[joint]) + nv)
    return q, v


def drift(model, joints, goals, placed, seconds):
    """{name: [worst mm, worst deg]} of the `placed` free bodies over `seconds` from their goals; `joints` {name: free
    joint} and `goals` {name: [x, y, z, qw, qx, qy, qz]} cover every free body, and those not placed start FAR_M m and
    more off along x. The robot starts at keyframe 0 (the model's qpos0 without one) and is clamped there every step."""
    import mujoco
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        data.qpos[:] = model.qpos0
    hold_q, hold_v = _robot_dofs(model)
    held = data.qpos[hold_q].copy()
    address = {name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)])
               for name, joint in joints.items()}
    for k, name in enumerate(sorted(joints)):
        pose = np.asarray(goals[name], dtype=float).copy()
        if name not in placed:
            pose[0] += FAR_M + k
        data.qpos[address[name]:address[name] + 7] = pose
    mujoco.mj_forward(model, data)
    target = {name: (np.asarray(goals[name][:3], dtype=float), Rotation.from_quat(np.asarray(goals[name])[[4, 5, 6, 3]]))
              for name in placed}
    worst = {name: [0.0, 0.0] for name in placed}
    for _ in range(int(round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
        data.qpos[hold_q] = held
        data.qvel[hold_v] = 0.0
        for name, (position, rotation) in target.items():
            q = data.qpos[address[name]:address[name] + 7]
            turn = (Rotation.from_quat(q[[4, 5, 6, 3]]) * rotation.inv()).magnitude()
            worst[name] = [max(worst[name][0], 1000 * float(np.linalg.norm(q[:3] - position))),
                           max(worst[name][1], float(np.degrees(turn)))]
    return worst


def _set_drift(job):
    xml, joints, goals, placed, seconds = job
    import mujoco
    return placed, drift(mujoco.MjModel.from_xml_path(xml), joints, goals, set(placed), seconds)


def settle_orders(export, orders, seconds, workers):
    """{'sets': {'a,b': drift}, 'orders': rows best first} for insertion `orders` of object ids on `export`."""
    export = Path(export)
    manifest = json.loads((export / 'inputs' / 'export_manifest.json').read_text())
    free = {o['object_id']: o for o in manifest['objects'] if o['free_joint']}
    joints = {key: o['free_joint'] for key, o in free.items()}
    goals = {key: o['goal_rig_wxyz'] for key, o in free.items()}
    sets = sorted({tuple(sorted(order[:k])) for order in orders for k in range(1, len(order) + 1)},
                  key=lambda s: (len(s), s))
    xml = str(export / 'inputs' / 'stage2' / 'scene.xml')
    with get_context('spawn').Pool(workers) as pool:
        table = dict(pool.map(_set_drift, [(xml, joints, goals, placed, seconds) for placed in sets]))
    tolerance_mm = 1000 * manifest['task']['position_tolerance_m']
    tolerance_deg = manifest['task']['rotation_tolerance_deg']
    rows = []
    for order in orders:
        steps = [table[tuple(sorted(order[:k]))] for k in range(1, len(order) + 1)]
        mm = max(w[0] for step in steps for w in step.values())
        deg = max(w[1] for step in steps for w in step.values())
        rows.append({'order': list(order), 'worst_mm': mm, 'worst_deg': deg,
                     'worst_over_tolerance': max(mm / tolerance_mm, deg / tolerance_deg),
                     'steps': [{'placed': list(order[:k]), 'drift_mm_deg': step} for k, step in enumerate(steps, 1)]})
    rows.sort(key=lambda row: (row['worst_over_tolerance'], row['order']))
    return {'sets': {','.join(placed): drift for placed, drift in table.items()}, 'orders': rows}


def orders_of(manifest, all_orders):
    """Insertion orders (object ids): the export's own, or every one its layout's plan directions allow."""
    if not all_orders:
        return [list(manifest['order'])]
    from ..interlock_repair import insertion_orders
    path = Path(manifest['task']['layout'])
    if _sha256(path) != manifest['task']['layout_sha256']:
        raise ValueError(f'{path} is not the layout the export was made from (its sha256 differs)')
    layout = json.loads(path.read_text())
    repair = (layout.get('source') or {}).get('ownership_repair') or {}
    if not repair.get('plan'):
        raise ValueError(f"{manifest['task']['layout']} has no assembly plan to take the directions from")
    owner_of = {tuple(cell): piece['name'] for piece in layout['pieces'] for cell in piece['cells']}
    directions = {step['fragment']: step['exit_direction'] for step in repair['plan']}
    ids = {o['fragment_id']: o['object_id'] for o in manifest['objects']}
    return [[ids[name] for name in order] for order in insertion_orders(owner_of, directions, repair['anchor'])]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--all-orders', action='store_true')
    parser.add_argument('--seconds', type=float, default=2.0)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args(argv)
    manifest_path = args.export / 'inputs' / 'export_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    orders = orders_of(manifest, args.all_orders)
    report = settle_orders(args.export, orders, args.seconds, args.workers)
    model = mujoco.MjModel.from_xml_path(str(args.export / 'inputs' / 'stage2' / 'scene.xml'))
    report = {'export': str(args.export.resolve()), 'export_manifest_sha256': _sha256(manifest_path),
              'scene_sha256': _sha256(args.export / 'inputs' / 'stage2' / 'scene.xml'),
              'layout': manifest['task']['layout'], 'layout_sha256': manifest['task']['layout_sha256'],
              'orders_from': 'layout plan directions, every order' if args.all_orders else 'the export',
              'seconds': args.seconds, 'timestep_s': float(model.opt.timestep), 'mujoco': mujoco.__version__,
              **report}
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    within = sum(row['worst_over_tolerance'] < 1 for row in report['orders'])
    print(f"{len(orders)} orders, {within} within the tolerances at every step; best first:")
    for row in report['orders'][:8]:
        print(f"  {row['worst_mm']:.4f} mm {row['worst_deg']:.4f} deg  {' '.join(row['order'])}")


if __name__ == '__main__':
    main()
