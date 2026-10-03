"""Build the studded voxel assets used only by the legacy storyboard."""

import argparse
import json
import math
import os

import numpy as np
import trimesh
from scipy import ndimage


parser = argparse.ArgumentParser()
parser.add_argument("glb")
parser.add_argument("out")
parser.add_argument("--pred")
parser.add_argument("--length", type=float, default=0.40)
parser.add_argument("--pitch", type=float, default=0.016)
parser.add_argument("--max-grip", type=float, default=0.076)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
np.random.seed(0)


def load_parts(path):
    scene = trimesh.load(path, force="scene")
    result = []
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        mesh.apply_transform(transform)
        result.append((geometry, mesh))
    return sorted(result, key=lambda item: item[0])


ground_truth = load_parts(args.glb)
source_names = [name for name, _ in ground_truth]
parts = [mesh for _, mesh in ground_truth]
predicted = dict(load_parts(args.pred)) if args.pred else {}
whole = trimesh.util.concatenate(parts)

# Preserve the original storyboard's stable-pose policy, then put the longest
# horizontal extent on X.
stable, probabilities = trimesh.poses.compute_stable_poses(
    whole.convex_hull, n_samples=1, threshold=0.0
)
orientation = stable[int(np.argmax(probabilities))]
oriented = whole.copy()
oriented.apply_transform(orientation)
if oriented.extents[1] > oriented.extents[0]:
    orientation = (
        trimesh.transformations.rotation_matrix(math.pi / 2, [0, 0, 1])
        @ orientation
    )
local = [mesh.copy().apply_transform(orientation) for mesh in parts]
combined = trimesh.util.concatenate(local)
scale = args.length / float(combined.extents[0])
local = [mesh.apply_scale(scale) for mesh in local]
combined = trimesh.util.concatenate(local)

pitch = float(args.pitch)
origin = combined.bounds[0] - pitch
shape = np.ceil((combined.bounds[1] + pitch - origin) / pitch).astype(int)
grid = np.full(tuple(shape), -1, dtype=int)
votes = {}
for index, mesh in enumerate(local):
    count = int(max(20000, 12 * mesh.area / (pitch * pitch)))
    cells = np.floor((mesh.sample(count) - origin) / pitch).astype(int)
    cells = cells[(cells >= 0).all(1) & (cells < shape).all(1)]
    unique, cell_votes = np.unique(cells, axis=0, return_counts=True)
    for cell, value in zip(map(tuple, unique), cell_votes):
        if int(value) > votes.get(cell, (0, -1))[0]:
            votes[cell] = (int(value), index)
for cell, (_, index) in votes.items():
    grid[cell] = index

for index in range(len(local)):
    mask = grid == index
    labels, components = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=int))
    if components > 1:
        sizes = ndimage.sum(mask, labels, range(1, components + 1))
        grid[(labels != 1 + int(np.argmax(sizes))) & mask] = -1
occupied_z = np.where((grid >= 0).any(axis=(0, 1)))[0]
grid = grid[:, :, occupied_z[0] :]
shape = np.asarray(grid.shape)
occupied = grid >= 0

triangles = np.asarray([[0, 1, 2], [0, 2, 3]])
stud_radius = 0.30 * pitch
stud_height = 0.17 * pitch


def cube_quads(i, j, k, mask):
    x0, y0, z0 = i * pitch, j * pitch, k * pitch
    x1, y1, z1 = x0 + pitch, y0 + pitch, z0 + pitch

    def filled(di, dj, dk):
        x, y, z = i + di, j + dj, k + dk
        return (
            0 <= x < shape[0]
            and 0 <= y < shape[1]
            and 0 <= z < shape[2]
            and mask[x, y, z]
        )

    candidates = (
        ((1, 0, 0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
        ((-1, 0, 0), [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)]),
        ((0, 1, 0), [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)]),
        ((0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
        ((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
        ((0, 0, -1), [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)]),
    )
    return [quad for direction, quad in candidates if not filled(*direction)]


def stud(cx, cy, z, segments=12):
    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    lower = [(cx + stud_radius * math.cos(a), cy + stud_radius * math.sin(a), z) for a in angles]
    upper = [(x, y, z + stud_height) for x, y, _ in lower]
    vertices = lower + upper + [(cx, cy, z + stud_height)]
    faces = []
    for index in range(segments):
        following = (index + 1) % segments
        faces.extend(
            ([index, following, segments + following],
             [index, segments + following, segments + index],
             [segments + index, segments + following, 2 * segments])
        )
    return np.asarray(vertices), np.asarray(faces)


def build_mesh(mask, add_studs):
    vertices, faces, offset = [], [], 0
    for i, j, k in zip(*np.where(mask)):
        for quad in cube_quads(i, j, k, mask):
            vertices.extend(quad)
            faces.extend((triangles + offset).tolist())
            offset += 4
        if add_studs and not (k + 1 < shape[2] and occupied[i, j, k + 1]):
            sv, sf = stud((i + 0.5) * pitch, (j + 0.5) * pitch, (k + 1) * pitch)
            vertices.extend(sv.tolist())
            faces.extend((sf + offset).tolist())
            offset += len(sv)
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


palette = (
    (0.80, 0.17, 0.15), (0.96, 0.76, 0.12), (0.12, 0.38, 0.78),
    (0.20, 0.64, 0.32), (0.93, 0.48, 0.12), (0.55, 0.30, 0.70),
    (0.25, 0.70, 0.75), (0.92, 0.86, 0.70),
)
layout = {
    "pitch": pitch,
    "grid": shape.tolist(),
    "anchor": None,
    "pieces": [],
    "source": os.path.abspath(args.glb),
    "pred_source": os.path.abspath(args.pred) if args.pred else None,
}
full_transform = np.diag([scale, scale, scale, 1.0]) @ orientation
for index, source_name in enumerate(source_names):
    mask = grid == index
    if not mask.any():
        continue
    cells = np.argwhere(mask)
    pivot = cells[:, :2].mean(0) * pitch + pitch / 2
    vertices, faces = build_mesh(mask, True)
    filename = f"piece_{index}.npz"
    np.savez(
        os.path.join(args.out, filename),
        v=vertices - np.asarray([pivot[0], pivot[1], 0], dtype=np.float32),
        f=faces,
    )
    extents = (cells[:, :2].max(0) - cells[:, :2].min(0) + 1) * pitch
    width = float(min(extents))
    entry = {
        "name": f"piece_{index}",
        "source_name": source_name,
        "color": palette[index % len(palette)],
        "npz": filename,
        "assembled_xy": pivot.round(4).tolist(),
        "min_z": float(cells[:, 2].min() * pitch),
        "top_z": float((cells[:, 2].max() + 1) * pitch),
        "extent_xy": extents.round(4).tolist(),
        "voxels": int(mask.sum()),
        "anchor": False,
        "top_accessible": True,
        "capped_frac": 0.0,
        "grasp": {
            "xy": [0.0, 0.0],
            "z_top": float((cells[:, 2].max() + 1) * pitch),
            "angle_deg": 0 if extents[1] <= extents[0] else 90,
            "width": min(width, args.max_grip),
            "clear": True,
        },
        "cells": cells.tolist(),
        "fp_cell": [float(pivot[0] / pitch), float(pivot[1] / pitch)],
    }
    if source_name in predicted:
        predicted_mesh = predicted[source_name].copy()
        predicted_mesh.apply_transform(full_transform)
        entry["pred"] = {
            "centroid_offset": (predicted_mesh.centroid - local[index].centroid).round(4).tolist(),
            "yaw_deg": 0.0,
            "rot_deg": None,
        }
    layout["pieces"].append(entry)

ghost_vertices, ghost_faces = build_mesh(occupied, False)
np.savez(os.path.join(args.out, "ghost.npz"), v=ghost_vertices, f=ghost_faces)

margin = 2
nx, ny = int(shape[0] + 2 * margin), int(shape[1] + 2 * margin)
thickness = 0.6 * pitch
plate = trimesh.creation.box((nx * pitch, ny * pitch, thickness))
plate.apply_translation((nx * pitch / 2 - margin * pitch, ny * pitch / 2 - margin * pitch, -thickness / 2))
plate_vertices = plate.vertices.tolist()
plate_faces = plate.faces.tolist()
offset = len(plate_vertices)
for i in range(nx):
    for j in range(ny):
        sv, sf = stud(
            (i + 0.5 - margin) * pitch,
            (j + 0.5 - margin) * pitch,
            0.0,
            10,
        )
        plate_vertices.extend(sv.tolist())
        plate_faces.extend((sf + offset).tolist())
        offset += len(sv)
np.savez(
    os.path.join(args.out, "plate.npz"),
    v=np.asarray(plate_vertices, dtype=np.float32),
    f=np.asarray(plate_faces, dtype=np.int32),
)
layout["plate"] = {
    "npz": "plate.npz",
    "size": [nx * pitch, ny * pitch, thickness],
    "offset": [-margin * pitch, -margin * pitch],
    "color": [0.78, 0.79, 0.82],
}
layout["ghost"] = {"npz": "ghost.npz"}
layout["assembly_extent"] = (shape * pitch).tolist()
with open(os.path.join(args.out, "layout.json"), "w", encoding="utf-8") as stream:
    json.dump(layout, stream, indent=1)
