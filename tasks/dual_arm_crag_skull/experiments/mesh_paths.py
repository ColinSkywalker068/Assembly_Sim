"""Can the original CRAG fragment meshes, rather than the task's 8 mm voxels, be put together by straight moves?

Each fragment of the source GLB, in the task's canonical frame (geometry_manifest.json `source_to_canonical`), is
filled as solid voxels on one fine grid (ray parity along z, so the meshes must be watertight) and shrunk by
`--shrink` cells, which leaves neighbours that touched about twice that much clearance. For each direction d of a
sample of the upper hemisphere (plus a ring of horizontal ones and the 17 lattice directions), space is cut into
square tubes along d: B blocks A along d where, in some tube, part of B lies ahead of part of A, so that A moved far
enough along d runs into B. Reported: pairs that no sampled direction separates, either way (checked again by moving
one fragment's cells step by step), and, with the anchor fixed, the orders that take all the others out, each along
an upper direction free of every fragment still present. Straight moves only; rotations are not searched.

  python -m tasks.dual_arm_crag_skull.experiments.mesh_paths --manifest runs/skull_straight_red/geometry_manifest.json \\
      --source view_gt.glb --anchor fragment_038755611176a8c9 --out runs/assembly_paths/mesh [--names EXPORT_MANIFEST]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

from .assembly_paths import DIRECTIONS


def canonical_meshes(source, manifest):
    """(names, meshes) of the fragments of the GLB `source`, posed in the frame of prepare's geometry `manifest`."""
    from ..preparation import load_skull
    meta = json.loads(Path(manifest).read_text())
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != meta['source_sha256']:
        raise ValueError(f'{source} is not the source {manifest} was prepared from')
    transform = np.asarray(meta['source_to_canonical'], float)
    fragments = load_skull(source).fragments
    return ([f.name for f in fragments],
            [f.mesh.copy().apply_transform(transform @ f.ground_truth_transform) for f in fragments])


def solid(mesh, origin, pitch, shape):
    """(cells whose centres are inside `mesh`, columns crossed an odd number of times) on the grid of cells
    `pitch` wide from `origin`, `shape` cells. Counts surface crossings below each centre along z."""
    tri = mesh.vertices[mesh.faces]
    cx = origin[0] + (0.5 + 1e-4 * np.sqrt(2)) * pitch  # column centres nudged so that none lies on an edge
    cy = origin[1] + (0.5 + 1e-4 * np.sqrt(3)) * pitch
    x, y, z = tri[..., 0], tri[..., 1], tri[..., 2]
    i0, i1 = np.ceil((x.min(1) - cx) / pitch).astype(np.int64), np.floor((x.max(1) - cx) / pitch).astype(np.int64)
    j0, j1 = np.ceil((y.min(1) - cy) / pitch).astype(np.int64), np.floor((y.max(1) - cy) / pitch).astype(np.int64)
    ni, nj = np.clip(i1 - i0 + 1, 0, None), np.clip(j1 - j0 + 1, 0, None)
    count = ni * nj  # the columns under each triangle's bounding box
    t = np.repeat(np.arange(len(tri)), count)
    local = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    ii, jj = i0[t] + local // nj[t], j0[t] + local % nj[t]
    px, py = cx + ii * pitch, cy + jj * pitch
    (x0, x1, x2), (y0, y1, y2), (z0, z1, z2) = x[t].T, y[t].T, z[t].T
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    with np.errstate(divide='ignore', invalid='ignore'):  # triangles seen edge-on (den 0) cross no column
        l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
        l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
        l2 = 1 - l0 - l1
        hit = (den != 0) & (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
    height = (l0 * z0 + l1 * z1 + l2 * z2)[hit]
    nx, ny, nz = (int(s) for s in shape)
    k = np.clip(np.ceil((height - origin[2]) / pitch - 0.5).astype(np.int64), 0, nz)  # first centre above it
    crossings = np.bincount((ii[hit] * ny + jj[hit]) * (nz + 1) + k, minlength=nx * ny * (nz + 1))
    crossings = crossings.reshape(nx, ny, nz + 1)
    odd = int((crossings.sum(2) % 2).sum())
    return (np.cumsum(crossings, axis=2)[..., :nz] % 2).astype(bool), odd


def shrink(mask, r):
    """The cells of `mask` more than `r` cells from any cell outside it (beyond the array counts as outside)."""
    if r <= 0:
        return mask
    return (ndimage.distance_transform_edt(np.pad(mask, 1)) > r)[1:-1, 1:-1, 1:-1]


def blocking(points, labels, n, direction, tube):
    """(n, n) counts: [a, b] is the number of tubes `tube` wide along `direction` in which some point labelled b
    lies ahead of some point labelled a."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    u = np.cross(d, (1.0, 0, 0) if abs(d[0]) < 0.9 else (0, 1.0, 0))
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    bu, bv = np.floor(points @ u / tube).astype(np.int64), np.floor(points @ v / tube).astype(np.int64)
    key = ((bu - bu.min()) * (bv.max() - bv.min() + 1) + (bv - bv.min())) * n + labels
    order = np.argsort(key, kind='stable')
    key, ahead = key[order], (points @ d)[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    _, tube_of = np.unique(key[starts] // n, return_inverse=True)
    first = np.full((tube_of.max() + 1, n), np.inf)
    last = np.full((tube_of.max() + 1, n), -np.inf)
    first[tube_of, key[starts] % n] = np.minimum.reduceat(ahead, starts)
    last[tube_of, key[starts] % n] = np.maximum.reduceat(ahead, starts)
    return (last[:, None, :] > first[:, :, None]).sum(0) * (1 - np.eye(n, dtype=int))


def upper_directions(n):
    """`n` directions spread over the upper hemisphere, 72 horizontal ones and the 17 lattice ones, as unit rows."""
    k = np.arange(n) + 0.5
    up = k / n
    turn = k * np.pi * (3 - np.sqrt(5))
    spread = np.stack([np.sqrt(1 - up ** 2) * np.cos(turn), np.sqrt(1 - up ** 2) * np.sin(turn), up], 1)
    ring = np.radians(np.arange(0, 360, 5))
    lattice = np.asarray(DIRECTIONS, float)
    return np.concatenate([lattice / np.linalg.norm(lattice, axis=1, keepdims=True),
                           np.stack([np.cos(ring), np.sin(ring), np.zeros_like(ring)], 1), spread])


def orders(counts, dirs, anchor, threshold=0):
    """Every order taking all fragments but `anchor` out one at a time, each along a direction of `dirs` in which no
    fragment still present blocks it on more than `threshold` tubes. `counts[k]` is `blocking` along dirs[k]."""
    blocked = counts > threshold

    def free(a, present):
        others = sorted(present - {a})
        return bool((~blocked[:, a][:, others].any(1)).any())

    @lru_cache(None)
    def rest(present):
        if present == frozenset({anchor}):
            return ((),)
        return tuple((a,) + tail for a in sorted(present - {anchor}) if free(a, present)
                     for tail in rest(present - {a}))
    return [list(o) for o in rest(frozenset(range(counts.shape[1])))]


def locked_pairs(counts, threshold=0):
    """Pairs (a, b) that no direction separates, either way, with the fewest tubes blocked over all of them."""
    n = counts.shape[1]
    return [(a, b, int(min(counts[:, a, b].min(), counts[:, b, a].min()))) for a in range(n) for b in range(a + 1, n)
            if (counts[:, a, b] > threshold).all() and (counts[:, b, a] > threshold).all()]


def swept_overlap(moving, fixed, direction, steps):
    """Most cells of the mask `moving` that land in the mask `fixed` when moved along `direction` (in cells), over
    `steps` (cells moved), each cell rounded to the nearest."""
    cells = np.argwhere(moving).astype(float)
    worst = 0
    for t in steps:
        moved = np.rint(cells + t * np.asarray(direction, float)).astype(np.int64)
        moved = moved[np.all((moved >= 0) & (moved < fixed.shape), 1)]
        worst = max(worst, int(fixed[tuple(moved.T)].sum()))
    return worst


_POINTS = _LABELS = None


def _share(points, labels):
    global _POINTS, _LABELS
    _POINTS, _LABELS = points, labels


def _blocking_along(job):
    n, direction, tube = job
    return blocking(_POINTS, _LABELS, n, direction, tube)


def fragment_cells(bits, index, r):
    """Cells of fragment `index` in the bit grid, shrunk by `r` cells (computed on its bounding box only)."""
    mask = ((bits >> index) & 1).astype(bool)
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0) + 1
    return np.argwhere(shrink(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], r)) + lo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True, help="prepare's geometry_manifest.json")
    parser.add_argument('--source', type=Path, required=True, help='the GLB it was prepared from')
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pitch-mm', type=float, default=0.5)
    parser.add_argument('--shrink', type=float, nargs='+', default=[1.0, 2.0], help='cells, each run separately')
    parser.add_argument('--threshold', type=int, default=4, help='blocked tubes ignored as voxel noise')
    parser.add_argument('--directions', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--names', type=Path, help='an export manifest, to name the fragments as the baselines do')
    args = parser.parse_args(argv)
    names, meshes = canonical_meshes(args.source, args.manifest)
    if args.anchor not in names:
        raise SystemExit(f'the anchor {args.anchor} is not one of {names}')
    label = names
    if args.names:
        ids = {o['fragment_id']: o['object_id'] for o in json.loads(args.names.read_text())['objects']}
        label = [ids.get(name, name) for name in names]
    pitch = args.pitch_mm / 1000
    low = np.min([m.bounds[0] for m in meshes], 0) - 2 * pitch
    shape = (np.ceil((np.max([m.bounds[1] for m in meshes], 0) - low) / pitch) + 2).astype(int)
    bits = np.zeros(shape, np.uint8)
    for index, mesh in enumerate(meshes):
        inside, odd = solid(mesh, low, pitch, shape)
        if odd or not mesh.is_watertight:
            raise SystemExit(f'{names[index]}: not a closed solid ({odd} columns crossed an odd number of times)')
        bits |= inside.astype(np.uint8) << index
        print(f'{label[index]}: {inside.sum() * pitch ** 3 * 1e6:.2f} cm3 of voxels, mesh {mesh.volume * 1e6:.2f}')
    overlap = int((np.unpackbits(bits[..., None], axis=-1).sum(-1) >= 2).sum())
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / 'solid.npz', bits=bits, names=np.array(names), origin=low, pitch=pitch)
    dirs, n, anchor = upper_directions(args.directions), len(names), names.index(args.anchor)
    report = {'source': str(args.source), 'manifest': str(args.manifest), 'pitch_m': pitch, 'cells_in_two': overlap,
              'directions': len(dirs), 'threshold_tubes': args.threshold, 'anchor': args.anchor, 'shrink': []}
    for r in args.shrink:
        cells = [fragment_cells(bits, index, r) for index in range(n)]
        points = np.concatenate([(low + (c + 0.5) * pitch) * 1000 for c in cells])  # millimetres
        labels = np.concatenate([np.full(len(c), index) for index, c in enumerate(cells)])
        with Pool(args.workers, initializer=_share, initargs=(points, labels)) as pool:
            counts = np.stack(pool.map(_blocking_along, [(n, d, args.pitch_mm) for d in dirs], chunksize=8))
        np.savez_compressed(args.out / f'blocking_shrink{r:g}.npz', counts=counts, dirs=dirs, names=np.array(names))
        both = np.concatenate([dirs, -dirs])
        locked = []
        for a, b, fewest in locked_pairs(counts, args.threshold):
            masks = [np.zeros(shape, bool) for _ in (a, b)]
            for mask, c in zip(masks, (cells[a], cells[b])):
                mask[tuple(c.T)] = True
            tried = np.argsort(np.concatenate([counts[:, a, b], counts[:, b, a]]))[:8]
            swept = min(swept_overlap(masks[0], masks[1], both[k], np.arange(0.5, 160, 0.5)) for k in tried)
            locked.append({'pair': [label[a], label[b]], 'fewest_tubes': fewest, 'swept_mm3': swept * args.pitch_mm ** 3})
        found = orders(counts, dirs, anchor, args.threshold)
        entry = {'shrink_cells': r, 'shrink_mm': r * args.pitch_mm, 'locked_pairs': locked, 'orders': len(found),
                 'first_out': [label[a] for a in range(n) if a != anchor and any(o[0] == a for o in found)]}
        report['shrink'].append(entry)
        print(f'shrink {r:g} cells ({r * args.pitch_mm:g} mm): {len(found)} removal orders with {label[anchor]} fixed;'
              f' locked: ' + ', '.join(f"{p['pair'][0]}-{p['pair'][1]} ({p['swept_mm3']:.1f} mm3)" for p in locked))
    (args.out / 'report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
