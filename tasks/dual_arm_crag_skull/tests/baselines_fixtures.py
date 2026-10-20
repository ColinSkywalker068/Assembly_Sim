"""A small synthetic schema-2 layout for the baseline tests: three 4 cm cubes, frag_a is the anchor. Its assembly plan
(as prepare --straight-assembly-anchor records it) puts frag_b in from +x, then frag_c from -x."""
import json

import numpy as np

HALF = 0.02
BOX_V = np.array([[x, y, z] for x in (-HALF, HALF) for y in (-HALF, HALF) for z in (0.0, 2 * HALF)], np.float32)
BOX_F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int32)
GOALS = {'frag_a': [0.0, 0.0, 0.0], 'frag_b': [0.04, 0.0, 0.0], 'frag_c': [-0.04, 0.0, 0.0]}
STAGING = {'frag_a': [0.42, 0.25, 0.75], 'frag_b': [-0.42, 0.10, 0.75], 'frag_c': [0.42, -0.30, 0.75]}
PAD = [-0.2, 0.2, -0.2, 0.2]
TABLE = [-0.97, 0.97, -0.57, 0.57]
EXCLUSIONS = [[-0.95, -0.61, -0.17, 0.17], [0.61, 0.95, -0.17, 0.17]]
EXITS = {'frag_b': [1, 0, 0], 'frag_c': [-1, 0, 0]}
CELLS = {'frag_a': [[1, 0, 0]], 'frag_b': [[2, 0, 0]], 'frag_c': [[0, 0, 0]]}  # one 4 cm cell each, c a b along x


def write_layout(directory, names=('frag_a', 'frag_b', 'frag_c'), plan=True):
    directory.mkdir(parents=True, exist_ok=True)
    pieces = []
    for name in names:
        np.savez(directory / f'{name}.npz', v=BOX_V, f=BOX_F)
        pieces.append({'name': name, 'mesh': f'{name}.npz', 'cells': CELLS[name],
                       'local_bounds': [[-HALF, -HALF, 0.0], [HALF, HALF, 2 * HALF]],
                       'goal_pose': {'position': GOALS[name], 'orientation_wxyz': [1.0, 0.0, 0.0, 0.0]},
                       'staging_pose': {'position': STAGING[name], 'orientation_wxyz': [1.0, 0.0, 0.0, 0.0]}})
    layout = {'schema_version': 2, 'object_id': 'cubes', 'pieces': pieces,
              'staging': {'pad_bounds': PAD, 'table_bounds': TABLE, 'table_top_z': 0.75, 'exclusions': EXCLUSIONS},
              'processing': {'pitch': 2 * HALF},  # the cells: cube boxes in the assembly frame
              'normalization': {'grid_origin': [-3 * HALF, -HALF, 0.0], 'grid_shape': [3, 1, 1]}}
    if plan:
        layout['source'] = {'ownership_repair': {
            'anchor': 'frag_a', 'plan_source': 'given',
            'plan': [{'fragment': name, 'exit_direction': EXITS[name]} for name in names if name != 'frag_a']}}
    path = directory / 'layout.json'
    path.write_text(json.dumps(layout))
    return path
