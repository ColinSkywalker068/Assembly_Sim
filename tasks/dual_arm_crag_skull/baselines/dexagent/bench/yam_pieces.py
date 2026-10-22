"""Convex collision pieces of the YAM linear_4310's fingers for MuJoCo (yam.py builds the hand from them).

DexAgent pairs every hand collision geom with every object geom near it, and the fragments are hundreds of boxes, so the
tip's 2 mm slabs (as sim/yam_usd.py cut the whole tip until it made the solid pad one hull, 2026-10-22) are merged along
the approach axis (tip_pieces): a slab's surface splits into its connected pieces (the pad, the two rails beside the
window), consecutive pieces of one chain are merged while the merged hull stands out of the mesh by at most
MERGE_TOLERANCE_M in every axis direction (Silhouettes: how far a shape reaches along +-approach, +-closing and
+-lateral over a grid of the two other coordinates; a convex hull's exactly, where each grid line through a cell's
centre meets it), and never where the mesh has no material (the window). Above ROOT_FROM_TIP_M the finger's root (an
L-shaped plate with bolt holes, then the carriage) is one hull per ROOT_BAND_M band: it bridges the L's corner and the
holes, as Isaac's slabs bridge the corner; nothing held here is that tall. Each piece records how far it stands out
(TipPiece.bulge_m). The body collides through its mesh's convex hull, as in Isaac.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import ConvexHull, QhullError

from ....sim.pick import surface_points
from ....sim.yam_usd import ROOT_FROM_TIP_M, SLAB_M, _clipped_points, _slab_groups, tip_triangles_in_link

MERGE_TOLERANCE_M = 2.5e-4
ROOT_BAND_M = 0.008  # above ROOT_FROM_TIP_M the finger's root is merged into one hull per band this tall
CELL_M = 2.5e-4  # silhouette grid
SAMPLE_M = 1.25e-4  # surface sampling of the mesh, half a cell


def tip_axes(gripper, tip):
    """(approach, closing, lateral) unit vectors in `tip`'s link frame: the TCP's approach axis, the mount's x (the
    closing axis, s0.kinematics.YAM_MOUNT_ROTATION) and their cross product."""
    rotation = gripper.link_frames[tip][:3, :3]
    approach = rotation.T @ gripper.mount_T_tcp[:3, 2]
    closing = rotation.T @ np.array([1.0, 0.0, 0.0])
    return approach, closing, np.cross(approach, closing)


class Silhouettes:
    """How far a shape reaches in each of six directions (+-approach, +-closing, +-lateral) over a grid of the two
    other coordinates, in a tip's link frame: tables keyed (axis, sign), -inf where it has nothing."""

    def __init__(self, axes, low, high, cell_m=CELL_M):
        self.axes, self.low, self.cell = np.asarray(axes, dtype=float), np.asarray(low, dtype=float), cell_m
        self.shape = np.ceil((np.asarray(high, dtype=float) - self.low) / cell_m).astype(int) + 1

    def tables(self, points):
        """Tables of a point set: in each cell, the farthest of its points that fall in it."""
        coords = np.asarray(points, dtype=float) @ self.axes.T
        cells = np.clip(np.floor((coords - self.low) / self.cell).astype(int), 0, self.shape - 1)
        out = {}
        for axis in range(3):
            i, j = (k for k in range(3) if k != axis)
            for sign in (1.0, -1.0):
                table = np.full((self.shape[i], self.shape[j]), -np.inf)
                np.maximum.at(table, (cells[:, i], cells[:, j]), sign * coords[:, axis])
                out[axis, sign] = table
        return out

    def convex_blocks(self, vertices):
        """Tables of the convex hull of `vertices`, exactly, where each grid line through a cell's centre meets it:
        {(axis, sign): (rows, columns, block)}, the block covering the hull's extent."""
        hull = ConvexHull(np.asarray(vertices, dtype=float))
        normals, offsets = hull.equations[:, :3] @ self.axes.T, hull.equations[:, 3]  # inside: n . x + b <= 0
        coords = hull.points[hull.vertices] @ self.axes.T
        first = np.clip(np.floor((coords.min(axis=0) - self.low) / self.cell).astype(int) - 1, 0, self.shape - 1)
        last = np.clip(np.floor((coords.max(axis=0) - self.low) / self.cell).astype(int) + 1, 0, self.shape - 1)
        out = {}
        for axis in range(3):
            i, j = (k for k in range(3) if k != axis)
            rows, columns = slice(first[i], last[i] + 1), slice(first[j], last[j] + 1)
            ci = self.low[i] + (np.arange(first[i], last[i] + 1) + 0.5) * self.cell
            cj = self.low[j] + (np.arange(first[j], last[j] + 1) + 0.5) * self.cell
            rhs = -(offsets + normals[:, i] * ci[:, None, None] + normals[:, j] * cj[None, :, None])
            n = normals[:, axis]
            with np.errstate(divide='ignore', invalid='ignore'):
                bound = rhs / n
            upper = np.where(n > 1e-12, bound, np.inf).min(axis=2)
            lower = np.where(n < -1e-12, bound, -np.inf).max(axis=2)
            meets = (upper >= lower) & ~((np.abs(n) <= 1e-12) & (rhs < 0)).any(axis=2)
            out[axis, 1.0] = (rows, columns, np.where(meets, upper, -np.inf))
            out[axis, -1.0] = (rows, columns, np.where(meets, -lower, -np.inf))
        return out


@dataclass(frozen=True)
class TipPiece:
    vertices: np.ndarray  # hull vertices in the tip's link frame
    bulge_m: float  # how far it stands out of the mesh (silhouettes), inf where the mesh has no material
    heights_m: tuple[float, float]  # along the approach axis, from the fingertip end


def _bulge(silhouettes, mesh_tables, vertices):
    """How far the hull of `vertices` reaches beyond the mesh's silhouettes (each widened by a cell to absorb the
    grids); inf when it reaches a cell the mesh does not."""
    worst = 0.0
    for key, (rows, columns, block) in silhouettes.convex_blocks(vertices).items():
        mesh = mesh_tables[key][rows, columns]
        present = np.isfinite(block)
        if (present & ~np.isfinite(mesh)).any():
            return np.inf
        if present.any():
            worst = max(worst, float((block[present] - mesh[present]).max()))
    return worst


def _hull_vertices(points):
    return points[ConvexHull(points).vertices]


def _chains(slabs):
    """Runs of slab pieces [(slab, index)]: `slabs` lists each slab's pieces as (points, (closing min, max, lateral
    min, max)); a run continues into the only piece of the next slab whose extent overlaps, if nothing else
    overlaps that one."""
    successors, predecessors = {}, {}
    for k in range(len(slabs) - 1):
        for i, (_, first) in enumerate(slabs[k]):
            for j, (_, second) in enumerate(slabs[k + 1]):
                a, b = first, second
                if a[0] <= b[1] + 1e-4 and b[0] <= a[1] + 1e-4 and a[2] <= b[3] + 1e-4 and b[2] <= a[3] + 1e-4:
                    successors.setdefault((k, i), []).append((k + 1, j))
                    predecessors.setdefault((k + 1, j), []).append((k, i))
    runs, seen = [], set()
    for k, pieces in enumerate(slabs):
        for i in range(len(pieces)):
            if (k, i) in seen:
                continue
            before = predecessors.get((k, i), [])
            if len(before) == 1 and len(successors.get(before[0], [])) == 1 and before[0] not in seen:
                continue  # the middle of a run: started from its head
            run, node = [], (k, i)
            while True:
                run.append(node)
                seen.add(node)
                after = successors.get(node, [])
                if len(after) != 1 or len(predecessors.get(after[0], [])) != 1:
                    break
                node = after[0]
            runs.append(run)
    leftover = [(k, i) for k, pieces in enumerate(slabs) for i in range(len(pieces)) if (k, i) not in seen]
    return runs + [[node] for node in leftover]


_PIECES = {}


def _tip_pieces(gripper, tip, tolerance_m):
    triangles = tip_triangles_in_link(gripper, tip)
    approach, closing, lateral = tip_axes(gripper, tip)
    along = triangles @ approach
    low, high = float(along.min()), float(along.max())
    mesh = surface_points(triangles.reshape(-1, 3), np.arange(len(triangles) * 3).reshape(-1, 3), SAMPLE_M)
    coords = mesh @ np.array([approach, closing, lateral]).T
    silhouettes = Silhouettes([approach, closing, lateral], coords.min(axis=0) - 0.002, coords.max(axis=0) + 0.002)
    mesh_tables = {key: maximum_filter(table, size=3, mode='nearest')
                   for key, table in silhouettes.tables(mesh).items()}
    slabs = []
    for k in range(max(1, int(np.ceil((high - low) / SLAB_M - 1e-9)))):
        start, stop = low + k * SLAB_M, min(high, low + (k + 1) * SLAB_M)
        pieces = []
        for group in _slab_groups(triangles, along, start, stop):
            points = np.unique(_clipped_points(triangles[group], along[group], start, stop), axis=0)
            try:
                ConvexHull(points)
            except (QhullError, ValueError):
                continue
            c, l = points @ closing, points @ lateral
            pieces.append((points, (c.min(), c.max(), l.min(), l.max())))
        slabs.append(pieces)
    out = []
    for run in _chains(slabs):
        current, measured = None, 0.0
        for k, i in run:
            points = slabs[k][i][0]
            if current is None:
                current = _hull_vertices(points)
                measured = _bulge(silhouettes, mesh_tables, current)
                continue
            merged = _hull_vertices(np.vstack([current, points]))
            bulge = _bulge(silhouettes, mesh_tables, merged)
            if bulge <= tolerance_m:
                current, measured = merged, bulge
            else:
                out.append(_piece(current, measured, approach, high))
                current = _hull_vertices(points)
                measured = _bulge(silhouettes, mesh_tables, current)
        out.append(_piece(current, measured, approach, high))
    finger = [piece for piece in out if piece.heights_m[0] < ROOT_FROM_TIP_M]
    root = [piece for piece in out if piece.heights_m[0] >= ROOT_FROM_TIP_M]
    bands = {}
    for piece in root:
        bands.setdefault(int((piece.heights_m[0] - ROOT_FROM_TIP_M) // ROOT_BAND_M), []).append(piece.vertices)
    for band in bands.values():
        vertices = _hull_vertices(np.vstack(band))
        finger.append(_piece(vertices, _bulge(silhouettes, mesh_tables, vertices), approach, high))
    return tuple(sorted(finger, key=lambda piece: piece.heights_m))


def _piece(vertices, bulge, approach, end):
    heights = end - vertices @ approach
    return TipPiece(vertices, float(bulge), (float(heights.min()), float(heights.max())))


def tip_pieces(gripper, tip, tolerance_m=MERGE_TOLERANCE_M):
    """The tip's convex collision pieces in its link frame (module docstring), from the fingertip end up."""
    key = (str(gripper.mesh_files[tip]), np.asarray(gripper.mesh_poses[tip]).tobytes(),
           np.asarray(gripper.link_frames[tip]).tobytes(), np.asarray(gripper.mount_T_tcp).tobytes(), float(tolerance_m))
    if key not in _PIECES:
        _PIECES[key] = _tip_pieces(gripper, tip, float(tolerance_m))
    return list(_PIECES[key])


def body_vertices(gripper):
    """The gripper body's convex hull vertices in its link frame (Isaac collides with the same hull)."""
    from core.robots.linear_4310 import read_binary_stl
    pose = gripper.mesh_poses['gripper']
    points = np.unique(read_binary_stl(gripper.mesh_files['gripper']) @ pose[:3, :3].T + pose[:3, 3], axis=0)
    return points[ConvexHull(points).vertices]
