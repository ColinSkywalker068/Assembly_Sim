"""How deep a fragment must pass through the others to reach its goal in a straight line (MuJoCo collision boxes).

The voxel check (assembly_paths.py) says whether a straight way in exists; this measures how far one is from existing.
For every fragment i, every other fragment j (the anchor included; the table plane too, for directions going down) and
each direction d, the deepest penetration of i's collision geoms into j's while i moves out of its goal along d by
0.5..150 mm (rotation fixed), with every other fragment at its goal: pen[i, j, d] in mm, saved with the fragment ids
and directions, read from the contacts MuJoCo's own collision pipeline makes there (not mj_geomDistance, which reports
millimetres of penetration for boxes that only share an edge). `lattice` takes the 26 directions of the voxel grid,
`sphere` 800 directions spread over the sphere.
Needs a DexAgent export (inputs/stage2/scene.xml, inputs/export_manifest.json) and its venv's MuJoCo; the fixed anchor
has no free joint, so its own row is NaN (its exits are its neighbours' reversed).

  envs/dexagent/bin/python -B -m tasks.dual_arm_crag_skull.experiments.assembly_depths EXPORT_DIR OUT.npz WORKERS \
      lattice|sphere
"""
import json, sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np

export, out, workers, kind = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
if kind == 'lattice':
    import itertools
    DIRS = np.array([np.array(v, float) / np.linalg.norm(v) for v in itertools.product((-1, 0, 1), repeat=3) if any(v)])
elif kind == 'sphere':
    golden, n = (1 + 5 ** 0.5) / 2, 800
    DIRS = np.array([[np.sqrt(1 - z * z) * np.cos(2 * np.pi * i / golden), np.sqrt(1 - z * z) * np.sin(2 * np.pi * i / golden), z]
                     for i in range(n) for z in [1 - (i + 0.5) / n * 2]])
else:
    raise SystemExit('directions: lattice or sphere')
TRAVEL = np.array([0.5, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 65, 80, 100, 125, 150]) / 1000


def setup():
    import mujoco
    manifest = json.loads((export / 'inputs' / 'export_manifest.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(export / 'inputs' / 'stage2' / 'scene.xml'))
    data = mujoco.MjData(model)
    objs = manifest['objects']
    def geoms(body_name):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        sub = {c for c in range(model.nbody) if c == b or model.body_rootid[c] == b}
        return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in sub and (model.geom_contype[g] or model.geom_conaffinity[g])]
    data.qpos[:] = model.qpos0
    adr = {}
    for o in objs:
        if o['free_joint'] is None:
            continue
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, o['free_joint'])
        adr[o['object_id']] = int(model.jnt_qposadr[j])
        data.qpos[adr[o['object_id']]:adr[o['object_id']] + 7] = o['goal_rig_wxyz']
    return mujoco, manifest, model, data, objs, geoms, adr


def work(index):
    mujoco, manifest, model, data, objs, geoms, adr = setup()
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'g_table')
    moving = objs[index]
    if moving['free_joint'] is None:
        return index, None
    owner = {}  # collision geom -> column: the fragment's index, or len(objs) for the table
    for j, o in enumerate(objs):
        owner.update({g: j for g in geoms(o['root_body'])})
    owner[table] = len(objs)
    goal = np.asarray(moving['goal_rig_wxyz'], float)
    a = adr[moving['object_id']]
    base = data.qpos.copy()
    pen = np.zeros((len(objs) + 1, len(DIRS)))  # last row: the table
    for k, d_ in enumerate(DIRS):
        for s in TRAVEL:
            data.qpos[:] = base
            data.qpos[a:a + 3] = goal[:3] + s * d_
            mujoco.mj_fwdPosition(model, data)  # the contacts the simulation itself would make
            if data.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number:
                raise RuntimeError('contact buffer full: contacts were dropped')
            for c in data.contact[:data.ncon]:
                pair = owner.get(int(c.geom1)), owner.get(int(c.geom2))
                if index not in pair or None in pair:
                    continue
                j = pair[1] if pair[0] == index else pair[0]
                if j != index and (j < len(objs) or d_[2] < 0):
                    pen[j, k] = max(pen[j, k], -float(c.dist))
    return index, 1000 * pen


if __name__ == '__main__':
    manifest = json.loads((export / 'inputs' / 'export_manifest.json').read_text())
    objs = manifest['objects']
    with get_context('spawn').Pool(workers) as pool:
        results = dict(pool.map(work, range(len(objs))))
    ids = [o['object_id'] for o in objs]
    pen = np.full((len(objs), len(objs) + 1, len(DIRS)), np.nan)
    for i, value in results.items():
        if value is not None:
            pen[i] = value
    np.savez(out, pen=pen, ids=np.array(ids), dirs=DIRS, anchor=manifest['anchor_object_id'])
    print('saved', out, pen.shape, flush=True)
