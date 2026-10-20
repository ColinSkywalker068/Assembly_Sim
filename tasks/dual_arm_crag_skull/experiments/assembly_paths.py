"""Can the task's fragments be put together by straight moves? Checked on the voxel grid they are cut from.

A layout's fragments are unit cubes of `processing.pitch` on one grid (layout.json `pieces[].cells`, in the assembled
skull). A fragment moved straight along one of the lattice directions d (components in -1, 0, 1; not downwards: the
assembly stands on the table) runs through another fragment's cube g exactly when, for some distance t > 0, its moved
cube c + t d overlaps g on every axis (|c_a + t d_a - g_a| < 1). For each fragment that says whether its way out of the
assembly, and so its way back in, is free of the fixed anchor alone and of all the others; a greedy disassembly
(take out any fragment that can leave, repeat) then says, for each choice of anchor, whether some order assembles all
of them (taking a fragment out never blocks another). Exact for cubes moved straight; rotations and curved paths are
not searched. `--figure` draws the slice through the anchor and one fragment where the anchor hangs over it.

  python -m tasks.dual_arm_crag_skull.experiments.assembly_paths --layout runs/skull_diagnostic/layout.json \\
      --anchor fragment_45266904a743c03b --out runs/assembly_paths/report.json [--figure runs/assembly_paths/hook.png]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

DIRECTIONS = tuple(d for d in itertools.product((-1, 0, 1), repeat=3) if any(d) and d[2] >= 0)


def blocked(moving, fixed, direction):
    """Whether cells `moving` (N, 3) moved straight along `direction` run into any of the cells `fixed` (M, 3)."""
    if not len(fixed):
        return False
    d = np.asarray(direction)
    delta = fixed[None, :, :] - moving[:, None, :]  # g - c
    still = d == 0
    same_row = np.all(delta[..., still] == 0, axis=-1) if still.any() else np.ones(delta.shape[:2], dtype=bool)
    ahead = delta[..., ~still] * d[~still]  # along each moving axis, how many cells ahead g is
    low = np.maximum(np.max(ahead - 1, axis=-1), 0)
    high = np.min(ahead + 1, axis=-1)
    return bool(np.any(same_row & (low < high)))


def free_directions(cells, name, present):
    """The lattice directions along which `name` leaves the fragments `present` (it among them) without a hit."""
    others = [cells[o] for o in present if o != name]
    fixed = np.vstack(others) if others else np.zeros((0, 3), dtype=int)
    return [d for d in DIRECTIONS if not blocked(cells[name], fixed, d)]


def disassemble(cells, anchor):
    """Greedy disassembly with `anchor` fixed: (order of (fragment, direction)), fragments that cannot leave)."""
    present, order = set(cells), []
    while len(present) > 1:
        movable = [(name, free_directions(cells, name, present)) for name in sorted(present - {anchor})]
        movable = [(name, free) for name, free in movable if free]
        if not movable:
            break
        name, free = movable[0]
        order.append((name, list(free[0])))
        present.remove(name)
    return order, sorted(present - {anchor})


def hook_slice(cells, anchor, fragment):
    """(y, x, z) cells of a y slice where an anchor cube stands above one of `fragment` (it cannot go up), or None."""
    top = {tuple(c) for c in cells[anchor]}
    for c in cells[fragment]:
        for k in range(c[2] + 1, c[2] + 6):
            if (c[0], c[1], k) in top:
                return int(c[1]), int(c[0]), int(c[2])
    return None


def figure(cells, anchor, fragment, path, names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    hit = hook_slice(cells, anchor, fragment)
    if hit is None:
        return None
    y, x0, z0 = hit
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, colour in ((anchor, '#cccccc'), (fragment, '#f0a030')):
        for c in cells[name]:
            if c[1] == y:
                ax.add_patch(Rectangle((c[0], c[2]), 1, 1, facecolor=colour, edgecolor='k', linewidth=0.4))
    for name, c in cells.items():
        if name not in (anchor, fragment):
            for cell in c:
                if cell[1] == y:
                    ax.add_patch(Rectangle((cell[0], cell[2]), 1, 1, facecolor='none', edgecolor='#4060c0',
                                           linewidth=0.6, hatch='//'))
    ax.add_patch(Rectangle((x0, z0), 1, 1, facecolor='none', edgecolor='red', linewidth=2.0))
    ax.set_xlim(min(c[0] for c in itertools.chain(*cells.values())) - 1, max(c[0] for c in itertools.chain(*cells.values())) + 2)
    ax.set_ylim(-0.5, max(c[2] for c in itertools.chain(*cells.values())) + 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x (cells)')
    ax.set_ylabel('z (cells), table at 0')
    ax.set_title(f'slice y = {y}: the fixed anchor {names.get(anchor, anchor)} (grey) over {names.get(fragment, fragment)} '
                 f'(orange)\nhatched: the other fragments; red outline: a cube with anchor above it, so it cannot go up')
    ax.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {'y': y, 'cell': [x0, y, z0]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--names', type=Path, help='an export manifest, to name the fragments as the baselines do')
    parser.add_argument('--figure', type=Path, help='draw where the anchor hangs over a fragment')
    args = parser.parse_args(argv)
    layout = json.loads(args.layout.read_text())
    cells = {p['name']: np.asarray(p['cells'], dtype=int) for p in layout['pieces']}
    names = {}
    if args.names:
        names = {o['fragment_id']: o['object_id'] for o in json.loads(args.names.read_text())['objects']}
    if args.anchor not in cells:
        parser.error(f'{args.anchor} is not a fragment of {args.layout}')
    report = {'layout': str(args.layout), 'pitch_m': layout['processing']['pitch'], 'anchor': args.anchor,
              'directions': [list(d) for d in DIRECTIONS], 'fragments': {}, 'anchors': {}}
    for name in sorted(cells):
        if name == args.anchor:
            continue
        report['fragments'][name] = {
            'name': names.get(name, name), 'cells': int(len(cells[name])),
            'free_with_the_anchor_alone': [list(d) for d in free_directions(cells, name, {name, args.anchor})],
            'free_with_all_placed': [list(d) for d in free_directions(cells, name, set(cells))]}
    for anchor in sorted(cells):
        order, stuck = disassemble(cells, anchor)
        report['anchors'][anchor] = {'name': names.get(anchor, anchor), 'disassembled': len(order),
                                     'order': [[names.get(n, n), d] for n, d in order],
                                     'stuck': [names.get(n, n) for n in stuck]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')  # kept even if the drawing fails
    if args.figure:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        locked = [n for n, row in report['fragments'].items() if not row['free_with_the_anchor_alone']]
        for name in locked:
            drawn = figure(cells, args.anchor, name, args.figure, names)
            if drawn is not None:
                report['figure'] = {'path': str(args.figure), 'fragment': names.get(name, name), **drawn}
                args.out.write_text(json.dumps(report, indent=1) + '\n')
                break
    for name, row in report['fragments'].items():
        print(f"{row['name']:22s} {row['cells']:4d} cells  free with the anchor alone: "
              f"{len(row['free_with_the_anchor_alone']):2d}/{len(DIRECTIONS)} directions; with all placed: "
              f"{len(row['free_with_all_placed']):2d}")
    for anchor, row in report['anchors'].items():
        print(f"anchor {row['name']:22s} disassembled {row['disassembled']}/{len(cells) - 1}"
              + (f"; stuck {row['stuck']}" if row['stuck'] else '; complete'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
