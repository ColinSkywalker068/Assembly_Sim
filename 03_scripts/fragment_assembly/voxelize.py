"""Deterministic canonicalization and common-grid fragment voxelization."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import trimesh

from .mesh_export import build_voxel_surface
from .model import AssemblyInput


Cell = tuple[int, int, int]

PALETTE = (
    (0.80, 0.17, 0.15),
    (0.96, 0.76, 0.12),
    (0.12, 0.38, 0.78),
    (0.20, 0.64, 0.32),
    (0.93, 0.48, 0.12),
    (0.55, 0.30, 0.70),
    (0.25, 0.70, 0.75),
    (0.92, 0.86, 0.70),
    (0.80, 0.45, 0.60),
    (0.40, 0.55, 0.30),
)


@dataclass(frozen=True)
class VoxelizationConfig:
    pitch: float = 0.016
    target_length: float = 0.40
    seed: int = 0
    min_surface_samples: int = 20000
    samples_per_pitch_area: float = 12.0
    max_grip: float = 0.076

    def __post_init__(self) -> None:
        if self.pitch <= 0 or self.target_length <= 0:
            raise ValueError("pitch and target_length must be positive")
        if self.min_surface_samples <= 0 or self.samples_per_pitch_area <= 0:
            raise ValueError("surface sampling parameters must be positive")


@dataclass(frozen=True)
class ProcessedFragment:
    name: str
    source_name: str
    cells: tuple[Cell, ...]
    local_pivot_cells: tuple[float, float, float]
    local_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    vertices: np.ndarray
    faces: np.ndarray
    goal_pose: Mapping[str, tuple[float, ...]]
    color: tuple[float, float, float]
    grasp: Mapping[str, object] | None


@dataclass(frozen=True)
class ProcessedAssembly:
    object_id: str
    source: Mapping[str, object]
    config: VoxelizationConfig
    source_to_canonical: np.ndarray
    grid_origin: tuple[float, float, float]
    grid_shape: tuple[int, int, int]
    assembly_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    fragments: tuple[ProcessedFragment, ...]


def choose_vote_owner(votes: Mapping[int, int]) -> int:
    """Choose the greatest vote count, breaking ties by fragment order."""

    if not votes:
        raise ValueError("cannot choose an owner without votes")
    return min(votes, key=lambda index: (-int(votes[index]), int(index)))


def keep_largest_component(cells: Sequence[Sequence[int]]) -> tuple[Cell, ...]:
    """Return the largest deterministic 26-connected cell component."""

    remaining = {tuple(int(value) for value in cell) for cell in cells}
    components: list[set[Cell]] = []
    offsets = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    )
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            x, y, z = queue.popleft()
            for dx, dy, dz in offsets:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    if not components:
        return ()
    largest = min(components, key=lambda component: (-len(component), min(component)))
    return tuple(sorted(largest))


def _world_meshes(assembly: AssemblyInput) -> list[trimesh.Trimesh]:
    result = []
    for fragment in assembly.fragments:
        mesh = fragment.mesh.copy()
        mesh.apply_transform(fragment.ground_truth_transform)
        result.append(mesh)
    return result


def _stable_orientation(whole: trimesh.Trimesh) -> np.ndarray:
    """Choose a deterministic stable pose with a low profile and long X axis."""

    transforms, probabilities = trimesh.poses.compute_stable_poses(
        whole.convex_hull, n_samples=1, threshold=0.0
    )
    candidates = []
    for transform, probability in list(zip(transforms, probabilities))[:12]:
        for flipped in (False, True):
            oriented = (
                trimesh.transformations.rotation_matrix(math.pi, (1.0, 0.0, 0.0))
                @ transform
                if flipped
                else np.asarray(transform, dtype=float)
            )
            mesh = whole.copy()
            mesh.apply_transform(oriented)
            if mesh.extents[1] > mesh.extents[0]:
                oriented = (
                    trimesh.transformations.rotation_matrix(
                        math.pi / 2, (0.0, 0.0, 1.0)
                    )
                    @ oriented
                )
                mesh = whole.copy()
                mesh.apply_transform(oriented)
            key = (
                round(float(mesh.extents[2]), 12),
                -round(float(mesh.extents[0]), 12),
                -round(float(probability), 12),
                int(flipped),
                tuple(np.round(oriented.reshape(-1), 12)),
            )
            candidates.append((key, oriented))
    if not candidates:
        return np.eye(4)
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _sample_cells(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    shape: np.ndarray,
    config: VoxelizationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(
        max(
            config.min_surface_samples,
            config.samples_per_pitch_area * mesh.area / (config.pitch * config.pitch),
        )
    )
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=config.seed)
    cells = np.floor((points - origin) / config.pitch).astype(int)
    inside = (cells >= 0).all(axis=1) & (cells < shape).all(axis=1)
    return np.unique(cells[inside], axis=0, return_counts=True)


def _grasp_for(cells: tuple[Cell, ...], all_cells: set[Cell], pitch: float, max_grip: float):
    cell_set = set(cells)
    xy = np.asarray([(cell[0], cell[1]) for cell in cells], dtype=float)
    center = xy.mean(axis=0)
    highest = max(cell[2] for cell in cells)
    best = None
    for i, j, k in cells:
        if (i, j, k + 1) in cell_set:
            continue
        for angle, (di, dj) in ((0, (1, 0)), (90, (0, 1)), (45, (1, 1)), (135, (-1, 1))):
            runs = []
            ends = []
            for sign in (1, -1):
                length = 0
                a, b = i, j
                while (a + sign * di, b + sign * dj, k) in cell_set:
                    a += sign * di
                    b += sign * dj
                    length += 1
                runs.append(length)
                ends.append((a + sign * di, b + sign * dj, k))
            width = (runs[0] + runs[1] + 1) * pitch * math.hypot(di, dj)
            if width > max_grip:
                continue
            clear = all(end not in all_cells for end in ends)
            score = (
                0.6 * np.linalg.norm(center - np.asarray((i, j)))
                + 2.0 * (highest - k)
                + (0.0 if clear else 6.0)
            )
            candidate = (score, i, j, k, angle, width, clear)
            if best is None or candidate < best:
                best = candidate
    return best


def process_assembly(
    assembly: AssemblyInput, config: VoxelizationConfig | None = None
) -> ProcessedAssembly:
    """Normalize and surface-voxelize every fragment on one common grid."""

    config = config or VoxelizationConfig()
    world_meshes = _world_meshes(assembly)
    whole = trimesh.util.concatenate(world_meshes)
    orientation = _stable_orientation(whole)
    oriented = []
    for mesh in world_meshes:
        candidate = mesh.copy()
        candidate.apply_transform(orientation)
        oriented.append(candidate)
    oriented_whole = trimesh.util.concatenate(oriented)
    x_extent = float(oriented_whole.extents[0])
    if not np.isfinite(x_extent) or x_extent <= 0:
        raise ValueError("assembled object has no positive canonical X extent")
    scale = config.target_length / x_extent
    source_to_canonical = np.diag((scale, scale, scale, 1.0)) @ orientation
    canonical = []
    for mesh in world_meshes:
        candidate = mesh.copy()
        candidate.apply_transform(source_to_canonical)
        canonical.append(candidate)
    canonical_whole = trimesh.util.concatenate(canonical)
    assembly_bounds = tuple(tuple(float(value) for value in row) for row in canonical_whole.bounds)

    origin = canonical_whole.bounds[0] - config.pitch
    shape = np.ceil((canonical_whole.bounds[1] + config.pitch - origin) / config.pitch).astype(int)
    votes: dict[Cell, dict[int, int]] = {}
    for fragment_index, mesh in enumerate(canonical):
        unique, counts = _sample_cells(mesh, origin, shape, config)
        for cell_array, count in zip(unique, counts):
            cell = tuple(int(value) for value in cell_array)
            votes.setdefault(cell, {})[fragment_index] = int(count)
    owned: list[list[Cell]] = [[] for _ in canonical]
    for cell, cell_votes in sorted(votes.items()):
        owned[choose_vote_owner(cell_votes)].append(cell)
    cleaned = [keep_largest_component(cells) for cells in owned]
    for fragment, cells in zip(assembly.fragments, cleaned):
        if not cells:
            raise ValueError(f"fragment {fragment.name} has no occupied voxels")

    min_z = min(cell[2] for cells in cleaned for cell in cells)
    cleaned = [tuple((i, j, k - min_z) for i, j, k in cells) for cells in cleaned]
    shape = np.asarray(
        (
            int(shape[0]),
            int(shape[1]),
            max(cell[2] for cells in cleaned for cell in cells) + 1,
        ),
        dtype=int,
    )
    grid_origin = origin + np.asarray((0.0, 0.0, min_z * config.pitch))
    all_cells = {cell for cells in cleaned for cell in cells}
    processed_fragments = []
    for index, (fragment, cells) in enumerate(zip(assembly.fragments, cleaned)):
        array = np.asarray(cells, dtype=float)
        pivot = (float(np.mean(array[:, 0] + 0.5)), float(np.mean(array[:, 1] + 0.5)), 0.0)
        local_min = (
            (float(array[:, 0].min()) - pivot[0]) * config.pitch,
            (float(array[:, 1].min()) - pivot[1]) * config.pitch,
            float(array[:, 2].min()) * config.pitch,
        )
        local_max = (
            (float(array[:, 0].max()) + 1.0 - pivot[0]) * config.pitch,
            (float(array[:, 1].max()) + 1.0 - pivot[1]) * config.pitch,
            (float(array[:, 2].max()) + 1.0) * config.pitch,
        )
        grasp_result = _grasp_for(cells, all_cells, config.pitch, config.max_grip)
        grasp = None
        if grasp_result is not None:
            _, i, j, k, angle, width, clear = grasp_result
            grasp = {
                "xy": (
                    (i + 0.5 - pivot[0]) * config.pitch,
                    (j + 0.5 - pivot[1]) * config.pitch,
                ),
                "z_top": (k + 1) * config.pitch,
                "angle_deg": angle,
                "width": width,
                "clear": clear,
            }
        vertices, faces = build_voxel_surface(cells, config.pitch, pivot)
        processed_fragments.append(
            ProcessedFragment(
                name=fragment.name,
                source_name=fragment.source_name,
                cells=cells,
                local_pivot_cells=pivot,
                local_bounds=(local_min, local_max),
                vertices=vertices,
                faces=faces,
                goal_pose={
                    "position": tuple(
                        float(grid_origin[axis] + pivot[axis] * config.pitch)
                        for axis in range(3)
                    ),
                    "orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
                },
                color=PALETTE[index % len(PALETTE)],
                grasp=grasp,
            )
        )
    return ProcessedAssembly(
        object_id=assembly.object_id,
        source={
            "loader": assembly.loader,
            "path": str(assembly.source_path),
            "fragment_names": tuple(fragment.source_name for fragment in assembly.fragments),
        },
        config=config,
        source_to_canonical=source_to_canonical,
        grid_origin=tuple(float(value) for value in grid_origin),
        grid_shape=tuple(int(value) for value in shape),
        assembly_bounds=assembly_bounds,
        fragments=tuple(processed_fragments),
    )
