"""Fragment geometry for the DexAgent export: exact box decompositions, OBJ files, MJCF assets, goal neighbours.

The skull fragments are voxel solids (layout npz meshes are the exposed faces of 8 mm cells), so a union of
axis-aligned boxes reproduces their collision geometry exactly, concavities included. The lattice is read off
the mesh itself (every vertex coordinate on `origin + k * pitch`); occupancy comes from vertical ray parity
through each column, and occupied cells are merged greedily (lowest z, then y, then x first; grow along x, y,
z), the rule tasks/dual_arm_breakingbad/scene/geometry.merge_voxel_cells uses for the Isaac colliders. A mesh
that is not on a lattice is voxelized at `fallback_pitch_m` and marked inexact.

MuJoCo collides a mesh geom by its convex hull, which is why the visual mesh never collides here. The visual mesh is
the solid's exterior (voxel_surface): the layout's voxel surface is not watertight where two cells meet along an edge
only (four faces share it), and DexAgent's blocking object_visual_penetration_audit needs a closed mesh, without faces
inside the solid either (it takes a point's depth as its distance to the nearest surface sample).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LATTICE_TOL = 1e-6
SOLREF = (0.006, 1.0)  # DexAgent's contact_defaults for unspecified rigid assets, stated explicitly
SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)


def load_mesh(path):
    """(vertices (N, 3) float, faces (M, 3) int) from a layout npz with 'v' and 'f'."""
    with np.load(path, allow_pickle=False) as data:
        if 'v' not in data.files or 'f' not in data.files:
            raise ValueError(f'{path}: mesh npz needs arrays v and f')
        vertices, faces = np.array(data['v'], dtype=float), np.array(data['f'], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError(f'{path}: v must be finite (N >= 4, 3) vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f'{path}: f must be (M, 3) vertex indices')
    return vertices, faces


def detect_lattice(values, tol=LATTICE_TOL):
    """(origin, pitch) when every value is origin + k * pitch for integers k, else None."""
    unique = np.unique(np.round(np.asarray(values, dtype=float), 9))
    gaps = np.diff(unique)
    gaps = gaps[gaps > tol]
    if not len(gaps):
        return None
    pitch = float(gaps.min())
    steps = (unique - unique[0]) / pitch
    if np.abs(steps - np.round(steps)).max() * pitch > tol:
        return None
    return float(unique[0]), pitch


def mesh_occupancy(vertices, faces, origin, pitch, shape):
    """Cells (i, j, k) whose centres lie inside the closed mesh, by parity of a vertical ray per column."""
    triangles = vertices[faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    keep = np.abs(area) > 1e-16  # vertical faces are parallel to the ray
    a, b, c, area = a[keep], b[keep], c[keep], area[keep]
    nx, ny, nz = shape
    # A cell-centre ray would run along the diagonal of a triangulated square face; a small offset avoids it.
    xs = origin[0] + (np.arange(nx) + 0.5 + 0.0131) * pitch
    ys = origin[1] + (np.arange(ny) + 0.5 + 0.0213) * pitch
    zs = origin[2] + (np.arange(nz) + 0.5) * pitch
    occupied = np.zeros(shape, dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            w1 = ((b[:, 0] - x) * (c[:, 1] - y) - (c[:, 0] - x) * (b[:, 1] - y)) / area
            w2 = ((c[:, 0] - x) * (a[:, 1] - y) - (a[:, 0] - x) * (c[:, 1] - y)) / area
            w3 = 1.0 - w1 - w2
            hit = (w1 > 0) & (w2 > 0) & (w3 > 0)
            if not hit.any():
                continue
            crossings = np.sort(w1[hit] * a[hit, 2] + w2[hit] * b[hit, 2] + w3[hit] * c[hit, 2])
            if len(crossings) % 2:
                raise ValueError('mesh is not closed: a vertical ray crosses it an odd number of times')
            for low, high in crossings.reshape(-1, 2):
                occupied[i, j] |= (zs > low) & (zs < high)
    return occupied


def merge_boxes(occupied):
    """Disjoint boxes (i0, i1, j0, j1, k0, k1), inclusive, covering exactly the occupied cells."""
    left = np.array(occupied, dtype=bool, copy=True)
    nx, ny, nz = left.shape
    boxes = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if not left[i, j, k]:
                    continue
                i1 = i
                while i1 + 1 < nx and left[i1 + 1, j, k]:
                    i1 += 1
                j1 = j
                while j1 + 1 < ny and left[i:i1 + 1, j1 + 1, k].all():
                    j1 += 1
                k1 = k
                while k1 + 1 < nz and left[i:i1 + 1, j:j1 + 1, k1 + 1].all():
                    k1 += 1
                left[i:i1 + 1, j:j1 + 1, k:k1 + 1] = False
                boxes.append((i, i1, j, j1, k, k1))
    return boxes


@dataclass(frozen=True)
class BoxDecomposition:
    """Occupied cells and the merged boxes, in the fragment frame (metres). `exact` when the mesh is on a lattice."""

    pitch_m: float
    origin_m: tuple
    cells: np.ndarray
    boxes: tuple
    exact: bool

    def __post_init__(self):
        cells = np.array(self.cells, dtype=np.int64, copy=True)
        if cells.ndim != 2 or cells.shape[1] != 3 or not len(cells):
            raise ValueError('a decomposition needs at least one occupied cell')
        cells.flags.writeable = False
        object.__setattr__(self, 'cells', cells)
        object.__setattr__(self, 'origin_m', tuple(map(float, self.origin_m)))
        object.__setattr__(self, 'boxes', tuple(tuple(map(int, box)) for box in self.boxes))

    @property
    def volume_m3(self):
        return len(self.cells) * self.pitch_m ** 3

    def cell_centres(self):
        return np.asarray(self.origin_m) + (self.cells + 0.5) * self.pitch_m

    def box_geometry(self, inset_m=0.0):
        """[(centre (3,), half_size (3,))] in the fragment frame, each box `inset_m` smaller on each face: a length for
        every face, or one per axis of the fragment frame."""
        origin, pitch, out = np.asarray(self.origin_m), self.pitch_m, []
        inset = np.asarray(inset_m, dtype=float)
        for i0, i1, j0, j1, k0, k1 in self.boxes:
            low = origin + np.array([i0, j0, k0]) * pitch
            high = origin + np.array([i1 + 1, j1 + 1, k1 + 1]) * pitch
            half = (high - low) / 2 - inset
            if (half <= 0).any():
                raise ValueError(f'an inset of {np.round(inset, 6).tolist()} m leaves nothing of a '
                                 f'{np.round(high - low, 6).tolist()} m box')
            out.append(((low + high) / 2, half))
        return out


def box_decomposition(vertices, faces, fallback_pitch_m=0.004):
    """Exact boxes for a voxel-surface mesh, else a `fallback_pitch_m` voxelization (marked inexact)."""
    lattices = [detect_lattice(vertices[:, axis]) for axis in range(3)]
    exact = all(lattice is not None for lattice in lattices)
    if exact:
        pitches = [lattice[1] for lattice in lattices]
        exact = max(pitches) - min(pitches) <= LATTICE_TOL
    if exact:
        origin, pitch = np.array([lattice[0] for lattice in lattices]), float(np.mean(pitches))
    else:
        if not np.isfinite(fallback_pitch_m) or fallback_pitch_m <= 0:
            raise ValueError('fallback_pitch_m must be positive')
        origin, pitch = vertices.min(axis=0), float(fallback_pitch_m)
    shape = tuple(int(np.ceil((vertices[:, axis].max() - origin[axis]) / pitch - 1e-6)) for axis in range(3))
    occupied = mesh_occupancy(vertices, faces, origin, pitch, shape)
    if not occupied.any():
        raise ValueError('mesh encloses no cell centre')
    return BoxDecomposition(pitch, tuple(origin), np.argwhere(occupied), tuple(merge_boxes(occupied)), exact)


VISUAL_SEPARATION_M = 1e-6
_UNIT_CORNERS = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]  # a cell's corners, index 4x + 2y + z
# a cell's faces, each wound outward, and the neighbouring cell across it: -x, +x, -y, +y, -z, +z
_OUTWARD_QUADS = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
_ACROSS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))


def voxel_surface(decomposition, separation_m=VISUAL_SEPARATION_M):
    """(vertices, faces) of the decomposition's solid: the exposed faces of its cells, wound outward, every edge shared
    by exactly two faces. Where two cells meet along an edge only, four faces share it; each cell's two faces there pass
    instead through a point of their own at the edge's middle, moved `separation_m` toward their cell (such a face is
    a fan around its centre)."""
    cells = {tuple(int(v) for v in cell) for cell in decomposition.cells}
    faces = []  # (cell, its four lattice points, outward)
    for cell in sorted(cells):
        for quad, across in zip(_OUTWARD_QUADS, _ACROSS):
            if tuple(c + a for c, a in zip(cell, across)) not in cells:
                faces.append((cell, tuple(tuple(c + u for c, u in zip(cell, _UNIT_CORNERS[i])) for i in quad)))
    uses = Counter(frozenset((quad[i], quad[(i + 1) % 4])) for _, quad in faces for i in range(4))
    pitch, origin = decomposition.pitch_m, np.asarray(decomposition.origin_m)
    ids, vertices = {}, []

    def vertex(point):  # lattice coordinates
        key = tuple(float(v) for v in point)
        if key not in ids:
            ids[key] = len(vertices)
            vertices.append(origin + np.asarray(key) * pitch)
        return ids[key]

    triangles = []
    for cell, quad in faces:
        ring = []
        for i in range(4):
            first, second = quad[i], quad[(i + 1) % 4]
            ring.append(vertex(first))
            if uses[frozenset((first, second))] > 2:
                middle = (np.asarray(first, dtype=float) + second) / 2
                towards = np.asarray(cell, dtype=float) + 0.5 - middle
                ring.append(vertex(middle + towards / np.linalg.norm(towards) * separation_m / pitch))
        if len(ring) == 4:
            triangles += [(ring[0], ring[1], ring[2]), (ring[0], ring[2], ring[3])]
        else:
            centre = vertex(np.mean(np.asarray(quad, dtype=float), axis=0))
            triangles += [(centre, ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
    return np.array(vertices), np.array(triangles, dtype=np.int64)


def write_obj(path, vertices, faces):
    lines = [f'v {x:.9g} {y:.9g} {z:.9g}' for x, y, z in vertices]
    lines += [f'f {a + 1} {b + 1} {c + 1}' for a, b, c in faces]
    Path(path).write_text('\n'.join(lines) + '\n')


def _vec(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


def asset_mjcf(name, mesh_file, decomposition, rgba, density_kg_m3, friction, free=True, inset_m=0.0):
    """An MJCF asset with one root body named `name` (DexAgent's asset contract), with a free joint `name_free`
    unless `free` is False (a fixed body). The visual mesh is massless and never collides; the disjoint boxes, each
    `inset_m` smaller on each face (a length, or one per axis: the protocol's collider_inset_m through
    box_axis_insets), carry the mass, the friction and explicit contact parameters."""
    if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
        raise ValueError(f'asset name {name!r} must be a DexAgent-safe identifier')
    geoms = [f'      <geom name="{name}_visual" type="mesh" mesh="{name}_visual" rgba="{_vec(rgba)}" contype="0" '
             f'conaffinity="0" group="2" density="0"/>']
    for index, (centre, half) in enumerate(decomposition.box_geometry(inset_m)):
        geoms.append(f'      <geom name="{name}_c{index:03d}" type="box" size="{_vec(half)}" pos="{_vec(centre)}" '
                     f'rgba="0 0 0 0" density="{float(density_kg_m3):.9g}" friction="{_vec(friction)}" '
                     f'solref="{_vec(SOLREF)}" solimp="{_vec(SOLIMP)}"/>')
    return (f'<mujoco model="{name}">\n  <compiler angle="radian"/>\n  <asset>\n'
            f'    <mesh name="{name}_visual" file="{mesh_file}"/>\n  </asset>\n  <worldbody>\n'
            f'    <body name="{name}">\n' + (f'      <freejoint name="{name}_free"/>\n' if free else '') +
            '\n'.join(geoms) +
            '\n    </body>\n  </worldbody>\n</mujoco>\n')


def goal_neighbours(decompositions, goal_poses):
    """{id: set of ids} of fragments that touch at the goal poses: cell centres at most 1.05 pitch apart (face
    neighbours of one lattice; approximate for fragments whose lattices do not align at the goal)."""
    from scipy.spatial import cKDTree
    centres = {}
    for key, decomposition in decompositions.items():
        pose = np.asarray(goal_poses[key], dtype=float)
        centres[key] = decomposition.cell_centres() @ pose[:3, :3].T + pose[:3, 3]
    trees = {key: cKDTree(points) for key, points in centres.items()}
    touching = {key: set() for key in centres}
    keys = sorted(centres)
    for index, first in enumerate(keys):
        for second in keys[index + 1:]:
            pitch = max(decompositions[first].pitch_m, decompositions[second].pitch_m)
            if trees[first].count_neighbors(trees[second], r=1.05 * pitch):
                touching[first].add(second)
                touching[second].add(first)
    return touching
