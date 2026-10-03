"""Dataset adapters which stop at the dataset-neutral assembly model."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import trimesh

from ..domain.model import AssemblyInput, FragmentInput


_BREAKING_BAD_PIECE = re.compile(r"^piece_(\d+)\.obj$", re.IGNORECASE)


def _load_triangle_mesh(path: Path) -> trimesh.Trimesh:
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:
        raise ValueError(f"failed to load {path.name} as a triangle mesh: {exc}") from exc
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"{path.name} is an empty triangle mesh")
    if np.asarray(mesh.faces).ndim != 2 or np.asarray(mesh.faces).shape[1] != 3:
        raise ValueError(f"{path.name} is not a triangle mesh")
    return mesh


def load_breaking_bad(directory: Path, object_id: str | None = None) -> AssemblyInput:
    """Load `piece_<integer>.obj` meshes already expressed in one GT frame."""

    source = Path(directory).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Breaking Bad sample directory not found: {source}")
    discovered: list[tuple[int, Path]] = []
    seen_ids: dict[int, Path] = {}
    for path in source.iterdir():
        match = _BREAKING_BAD_PIECE.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        piece_id = int(match.group(1))
        if piece_id in seen_ids:
            raise ValueError(
                f"duplicate piece id {piece_id}: {seen_ids[piece_id].name}, {path.name}"
            )
        seen_ids[piece_id] = path
        discovered.append((piece_id, path))
    if not discovered:
        raise ValueError(f"no piece_<integer>.OBJ meshes found in {source}")

    fragments = tuple(
        FragmentInput(
            name=f"piece_{piece_id}",
            source_name=path.name,
            source_path=path,
            mesh=_load_triangle_mesh(path),
            ground_truth_transform=np.eye(4),
        )
        for piece_id, path in sorted(discovered, key=lambda item: item[0])
    )
    return AssemblyInput(
        object_id=object_id or source.name,
        loader="breaking_bad",
        source_path=source,
        fragments=fragments,
    )


def load_crag_glb(path: Path, object_id: str | None = None) -> AssemblyInput:
    """Load CRAG-style GLB scene nodes while retaining their GT transforms."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"CRAG GLB not found: {source}")
    loaded = trimesh.load(source, force="scene", process=False)
    if not isinstance(loaded, trimesh.Scene):
        raise ValueError(f"CRAG source is not a mesh scene: {source}")
    nodes = sorted(loaded.graph.nodes_geometry)
    fragments = []
    for index, node_name in enumerate(nodes):
        transform, geometry_name = loaded.graph[node_name]
        mesh = loaded.geometry[geometry_name].copy()
        fragments.append(
            FragmentInput(
                name=f"piece_{index}",
                source_name=str(geometry_name),
                source_path=source,
                mesh=mesh,
                ground_truth_transform=np.asarray(transform, dtype=float),
            )
        )
    if not fragments:
        raise ValueError(f"CRAG scene contains no triangle meshes: {source}")
    return AssemblyInput(
        object_id=object_id or source.stem,
        loader="crag_glb",
        source_path=source,
        fragments=tuple(fragments),
    )
