"""LEGO-ify a CRAG assembly: each fragment becomes one stud-topped brick on a common voxel grid.
usage: python asm_bricks.py <gt_assembly.glb> <out_dir> [--pred predicted.glb] [--length 0.40] [--pitch 0.016] [--stable]
Occupancy = surface samples (keeps thin skull walls) + interior containment; pieces keep their fracture boundaries.
Writes piece_k.npz (v, f), ghost.npz, plate.npz and layout.json (assembled positions, grasps, resting offsets, anchor,
and, if --pred is given, CRAG's predicted placement of every piece relative to the ground truth)."""
import os, sys, json, math, argparse
import numpy as np, trimesh
from scipy import ndimage
ap = argparse.ArgumentParser(); ap.add_argument('glb'); ap.add_argument('out'); ap.add_argument('--pred', default=None)
ap.add_argument('--length', type=float, default=0.40); ap.add_argument('--pitch', type=float, default=0.016); ap.add_argument('--max_grip', type=float, default=0.076)
ap.add_argument('--stable', action='store_true', help='(kept for compatibility; orientation is always chosen among stable poses)')
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); np.random.seed(0)
def load_parts(f):
    sc = trimesh.load(f, force='scene'); out = []
    for node in sc.graph.nodes_geometry:
        T, g = sc.graph[node]; m = sc.geometry[g].copy(); m.apply_transform(T); out.append((g, m))
    return sorted(out, key=lambda t: t[0])
gt = load_parts(args.glb); names = [g for g, _ in gt]; parts = [m for _, m in gt]
pred = dict(load_parts(args.pred)) if args.pred else None
whole = trimesh.util.concatenate(parts)
# ---- orientation candidates: stable resting poses (convex hull) and their upside-down flips, longest horizontal axis along x.
def orient_candidates():
    cands = []
    tfs, probs = trimesh.poses.compute_stable_poses(whole.convex_hull, n_samples=1, threshold=0.0)
    for Tw0, pr in list(zip(tfs, probs))[:6]:
        for flip in (False, True):
            Tw = (trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]) @ Tw0) if flip else Tw0
            w = whole.copy(); w.apply_transform(Tw)
            if w.extents[1] > w.extents[0]: Tw = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 0, 1]) @ Tw
            cands.append((Tw, float(pr), flip))
    return cands
def quick_grid(Tw, pitch):
    """coarse shell voxelisation of every piece for a candidate orientation -> per-piece cell sets"""
    L = [m.copy().apply_transform(Tw) for m in parts]; W = trimesh.util.concatenate(L); sc = args.length / (W.bounds[1, 0] - W.bounds[0, 0])
    L = [m.apply_scale(sc) for m in L]; W = trimesh.util.concatenate(L); lo0 = W.bounds[0] - pitch; cells = []
    for m in L:
        pts = m.sample(int(max(8000, 6 * m.area / (pitch * pitch)))); c = np.floor((pts - lo0) / pitch).astype(int); cells.append(set(map(tuple, np.unique(c, axis=0))))
    return cells
def insertability(cells):
    """build bottom-up (by lowest cell): a piece is insertable from above if no earlier (lower) piece has a cell above one of its columns"""
    order = sorted(range(len(cells)), key=lambda k_: min(c[2] for c in cells[k_]))
    placed_cols = {}; score = 0.0; n_ok = 0
    for k_ in order:
        c = cells[k_]; capped = sum(1 for (i, j, k) in c if placed_cols.get((i, j), -1) > k); frac = capped / max(1, len(c)); score += 1 - frac; n_ok += frac == 0
        for (i, j, k) in c: placed_cols[(i, j)] = max(placed_cols.get((i, j), -1), k)
    return n_ok, score
best = None
for Tw, pr, flip in orient_candidates():
    n_ok, sc_ = insertability(quick_grid(Tw, args.pitch * 1.5))
    key = (n_ok, round(sc_, 3), pr); print('orientation candidate: flip', flip, 'stable p', round(pr, 3), 'insertable pieces', n_ok, 'score', round(sc_, 3))
    if best is None or key > best[0]: best = (key, Tw, flip)
Tw = best[1]; print('chosen orientation: flipped' if best[2] else 'chosen orientation: as resting', 'key', best[0])
loc = [m.copy().apply_transform(Tw) for m in parts]; wl = trimesh.util.concatenate(loc)
scale = args.length / (wl.bounds[1, 0] - wl.bounds[0, 0]); loc = [m.apply_scale(scale) for m in loc]; wl = trimesh.util.concatenate(loc)
print('oriented extents (m):', np.round(wl.extents, 3), 'scale', round(scale, 4))
# ---- common grid
p = args.pitch; lo = wl.bounds[0] - p; hi = wl.bounds[1] + p; n = np.ceil((hi - lo) / p).astype(int)
grid = np.full(tuple(n), -1, dtype=int); dist = np.full(tuple(n), np.inf)
ijk = np.stack(np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]), indexing='ij'), -1).reshape(-1, 3); centers = lo + (ijk + 0.5) * p
# shell voxelisation from dense surface samples; a cell belongs to the piece with the most samples in it (fast, keeps thin skull walls)
votes = {}
for k, m in enumerate(loc):
    npts = int(max(20000, 12 * m.area / (p * p))); pts = m.sample(npts); cells = np.floor((pts - lo) / p).astype(int)
    cells = cells[(cells >= 0).all(1) & (cells < n).all(1)]
    uc, cnt = np.unique(cells, axis=0, return_counts=True)
    for c, ct in zip(map(tuple, uc), cnt):
        if ct > votes.get(c, (0, -1))[0]: votes[c] = (int(ct), k)
for c, (ct, k) in votes.items(): grid[c] = k
S26 = np.ones((3, 3, 3), dtype=int)
for k in range(len(loc)):
    mask = grid == k; lab, num = ndimage.label(mask, structure=S26)
    if num > 1:
        sizes = ndimage.sum(mask, lab, range(1, num + 1)); keep = 1 + int(np.argmax(sizes)); grid[(lab != keep) & mask] = -1
zs = np.where((grid >= 0).any(axis=(0, 1)))[0]; grid = grid[:, :, zs[0]:]; n = np.array(grid.shape); lo_z_shift = zs[0]
counts = [(grid == k).sum() for k in range(len(loc))]; print('voxels per piece', dict(zip(names, counts)), 'grid', n.tolist())
occ = grid >= 0
# ---- geometry helpers
CUBE_F = np.array([[0, 1, 2], [0, 2, 3]]); stud_r, stud_h = 0.30 * p, 0.17 * p
def box_faces(i, j, k, mask_same):
    x0, y0, z0 = i * p, j * p, k * p; x1, y1, z1 = x0 + p, y0 + p, z0 + p; out = []
    nb = lambda di, dj, dk: (0 <= i+di < n[0] and 0 <= j+dj < n[1] and 0 <= k+dk < n[2] and mask_same[i+di, j+dj, k+dk])
    if not nb(1, 0, 0): out.append([(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)])
    if not nb(-1, 0, 0): out.append([(x0,y1,z0),(x0,y0,z0),(x0,y0,z1),(x0,y1,z1)])
    if not nb(0, 1, 0): out.append([(x1,y1,z0),(x0,y1,z0),(x0,y1,z1),(x1,y1,z1)])
    if not nb(0, -1, 0): out.append([(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)])
    if not nb(0, 0, 1): out.append([(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)])
    if not nb(0, 0, -1): out.append([(x0,y1,z0),(x1,y1,z0),(x1,y0,z0),(x0,y0,z0)])
    return out
def stud(cx, cy, z, segs=12):
    ang = np.linspace(0, 2 * math.pi, segs, endpoint=False); ring0 = [(cx + stud_r * math.cos(a), cy + stud_r * math.sin(a), z) for a in ang]; ring1 = [(x, y, z + stud_h) for x, y, _ in ring0]
    V = ring0 + ring1 + [(cx, cy, z + stud_h)]; F = []
    for s in range(segs):
        t = (s + 1) % segs; F += [[s, t, segs + t], [s, segs + t, segs + s], [segs + s, segs + t, 2 * segs]]
    return np.array(V), np.array(F)
def build(mask, studs_any):
    V, F = [], []; base = 0
    for i, j, k in zip(*np.where(mask)):
        for quad in box_faces(i, j, k, mask): V += quad; F += (CUBE_F + base).tolist(); base += 4
        if studs_any is not None and not (k + 1 < n[2] and studs_any[i, j, k + 1]):
            sv, sf = stud((i + 0.5) * p, (j + 0.5) * p, (k + 1) * p); V += sv.tolist(); F += (sf + base).tolist(); base += len(sv)
    return np.array(V, dtype=np.float32), np.array(F, dtype=np.int32)
palette = [(0.80, 0.17, 0.15), (0.96, 0.76, 0.12), (0.12, 0.38, 0.78), (0.20, 0.64, 0.32), (0.93, 0.48, 0.12), (0.55, 0.30, 0.70), (0.25, 0.70, 0.75), (0.92, 0.86, 0.70), (0.80, 0.45, 0.60), (0.40, 0.55, 0.30)]
def grasp_for(k, allow_bottom=False):
    """top-down grasp near the piece's highest point: fingers close along a direction where the run of this piece is <= max_grip
    and both ends are air in the full assembly."""
    mask = grid == k; cells = np.argwhere(mask); cen = cells[:, :2].mean(0); kmax = cells[:, 2].max(); best = None
    dirs = {0: (1, 0), 90: (0, 1), 45: (1, 1), 135: (-1, 1)}
    for (i, j, kk) in cells:
        if kk + 1 < n[2] and mask[i, j, kk + 1]: continue  # top cell of THIS piece (later pieces may cover it in the finished assembly)
        if kk == 0 and not allow_bottom: continue
        for ang, (di, dj) in dirs.items():
            step = p * math.hypot(di, dj)
            def run(sign):
                r = 0; a, b = i, j
                while True:
                    a += sign * di; b += sign * dj
                    if not (0 <= a < n[0] and 0 <= b < n[1]) or not mask[a, b, kk]: return r, (a, b)
                    r += 1
            r1, e1 = run(1); r2, e2 = run(-1); width = (r1 + r2 + 1) * step
            clear = all((not (0 <= a < n[0] and 0 <= b < n[1])) or not occ[a, b, kk] for a, b in (e1, e2))
            if width <= args.max_grip:
                score = 0.6 * np.linalg.norm(cen - np.array([i, j])) + 2.0 * (kmax - kk) + (1.5 if width <= 1.01 * p else 0) + (0 if clear else 6.0)
                if best is None or score < best[0]: best = (score, i, j, kk, ang, width, clear)
    if best is None and not allow_bottom: return grasp_for(k, allow_bottom=True)
    return best
anchor = -1; layout = {"pitch": p, "grid": n.tolist(), "anchor": None, "pieces": [], "source": os.path.abspath(args.glb), "pred_source": os.path.abspath(args.pred) if args.pred else None}
top_cols = {}
for (i, j, k) in zip(*np.where(occ)): top_cols[(i, j)] = max(top_cols.get((i, j), -1), k)
Tfull = np.diag([scale, scale, scale, 1.0]) @ Tw
for k, name in enumerate(names):
    mask = grid == k
    if mask.sum() == 0: print('WARNING empty piece', name); continue
    V, F = build(mask, occ); cells = np.argwhere(mask); fp = cells[:, :2].mean(0) * p + p / 2
    np.savez(os.path.join(args.out, f'piece_{k}.npz'), v=V - np.array([fp[0], fp[1], 0], dtype=np.float32), f=F)
    g = grasp_for(k); entry = {"name": f"piece_{k}", "source_name": name, "color": palette[k % len(palette)], "npz": f"piece_{k}.npz", "assembled_xy": fp.round(4).tolist(),
        "min_z": float(cells[:, 2].min() * p), "top_z": float((cells[:, 2].max() + 1) * p), "extent_xy": ((cells[:, :2].max(0) - cells[:, :2].min(0) + 1) * p).round(4).tolist(), "voxels": int(mask.sum()), "anchor": False, "top_accessible": bool(all(grid[int(i), int(j), kk2] == k for (i, j, kk) in cells for kk2 in range(int(kk) + 1, n[2]) if grid[int(i), int(j), kk2] >= 0)),
        "capped_frac": float(np.mean([any(grid[int(i), int(j), kk2] not in (-1, k) for kk2 in range(int(kk) + 1, n[2])) for (i, j, kk) in cells])),

        "grasp": None if g is None else {"xy": [round((g[1] + 0.5) * p - fp[0], 4), round((g[2] + 0.5) * p - fp[1], 4)], "z_top": round((g[3] + 1) * p, 4), "angle_deg": g[4], "width": round(g[5], 4), "clear": bool(g[6])},
        "cells": cells.tolist(), "fp_cell": [float(fp[0] / p), float(fp[1] / p)]}
    if pred is not None and name in pred:
        mg = loc[k]; mp = pred[name].copy().apply_transform(Tfull)
        Dp = trimesh.transformations.superimposition_matrix(mg.vertices[::max(1, len(mg.vertices)//2000)].T, mp.vertices[::max(1, len(mp.vertices)//2000)].T) if len(mg.vertices) == len(mp.vertices) else None
        yaw = math.degrees(math.atan2(Dp[1, 0], Dp[0, 0])) if Dp is not None else 0.0
        entry["pred"] = {"centroid_offset": (mp.centroid - mg.centroid).round(4).tolist(), "yaw_deg": round(yaw, 1), "rot_deg": round(math.degrees(math.acos(max(-1, min(1, (np.trace(Dp[:3, :3]) - 1) / 2)))), 1) if Dp is not None else None}
    layout["pieces"].append(entry); print(f"{entry['name']} ({name}): {mask.sum()} vox, min_z {entry['min_z']:.3f} top {entry['top_z']:.3f}, anchor={entry['anchor']}, grasp {entry['grasp']}, pred {entry.get('pred')}")
V, F = build(occ, None); np.savez(os.path.join(args.out, 'ghost.npz'), v=V, f=F)
m_ = 2; nx, ny = n[0] + 2 * m_, n[1] + 2 * m_; th = 0.6 * p; V, F = [], []; base = 0
box = [[(0,0,-th),(nx*p,0,-th),(nx*p,ny*p,-th),(0,ny*p,-th)][::-1], [(0,0,0),(nx*p,0,0),(nx*p,ny*p,0),(0,ny*p,0)],
       [(0,0,-th),(nx*p,0,-th),(nx*p,0,0),(0,0,0)], [(nx*p,0,-th),(nx*p,ny*p,-th),(nx*p,ny*p,0),(nx*p,0,0)], [(nx*p,ny*p,-th),(0,ny*p,-th),(0,ny*p,0),(nx*p,ny*p,0)], [(0,ny*p,-th),(0,0,-th),(0,0,0),(0,ny*p,0)]]
for q in box: V += q; F += (CUBE_F + base).tolist(); base += 4
for i in range(nx):
    for j in range(ny):
        sv, sf = stud((i + 0.5) * p, (j + 0.5) * p, 0.0, 10); V += sv.tolist(); F += (sf + base).tolist(); base += len(sv)
V = np.array(V, dtype=np.float32) - np.array([m_ * p, m_ * p, 0], dtype=np.float32); np.savez(os.path.join(args.out, 'plate.npz'), v=V, f=np.array(F, dtype=np.int32))
layout["plate"] = {"npz": "plate.npz", "size": [nx * p, ny * p, th], "offset": [-m_ * p, -m_ * p], "color": [0.78, 0.79, 0.82]}; layout["ghost"] = {"npz": "ghost.npz"}; layout["assembly_extent"] = (n * p).tolist()
json.dump(layout, open(os.path.join(args.out, 'layout.json'), 'w'), indent=1); print('layout written; assembly extent', np.round(n * p, 3), 'top-accessible', [e['name'] for e in layout['pieces'] if e['top_accessible']])
