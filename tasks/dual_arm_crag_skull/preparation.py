"""Strict CRAG input validation and reproducible simulation asset preparation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import trimesh

from core.workcell.config import load_workcell_preset
from tasks.dual_arm_breakingbad.domain.model import AssemblyInput, FragmentInput
from tasks.dual_arm_breakingbad.preprocessing.voxelize import VoxelizationConfig, process_assembly
from tasks.dual_arm_breakingbad.preprocessing.preprocess import write_processed_assembly


def load_skull(path: Path) -> AssemblyInput:
    """Caller must supply an assembled GT scene, never a scattered/predicted GLB."""
    path = Path(path).resolve()
    scene = trimesh.load(path, force="scene", process=False)
    fragments = []
    for node in sorted(scene.graph.nodes_geometry):
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        if not np.isfinite(mesh.vertices).all():
            raise ValueError(f"nonfinite vertices: {node}")
        if abs(np.linalg.det(transform[:3, :3])) < 1e-12:
            raise ValueError(f"singular transform: {node}")
        # Node identity survives ordering changes and shared geometry instances.
        name = "fragment_" + hashlib.sha256(str(node).encode()).hexdigest()[:16]
        fragments.append(FragmentInput(name, str(node), path, mesh, transform))
    if len(fragments) < 6:
        raise ValueError("CRAG skull requires at least six mesh nodes")
    return AssemblyInput(path.stem, "crag_gt_glb", path, tuple(fragments))


def require_connected(cells):
    remaining = set(map(tuple, cells))
    if not remaining:
        raise ValueError("empty fragment")
    todo = [remaining.pop()]
    while todo:
        cell = todo.pop()
        for axis in range(3):
            for sign in (-1, 1):
                neighbor = list(cell)
                neighbor[axis] += sign
                neighbor = tuple(neighbor)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    todo.append(neighbor)
    if remaining:
        raise ValueError("fragment is not six-connected; change pitch or repair source")


def prepare(source, output, *, pitch, target_length, seed=0):
    if not all(np.isfinite(v) and v > 0 for v in (pitch, target_length)):
        raise ValueError("pitch and target length must be finite and positive")
    assembly = load_skull(source)
    # Preserve all owned cells. Never silently discard geometry for printability.
    config = VoxelizationConfig(pitch=pitch, target_length=target_length, seed=seed,
                                preserve_all_components=True)
    processed = process_assembly(assembly, config)
    meshes = {}
    for fragment in processed.fragments:
        require_connected(fragment.cells)
        mesh = trimesh.Trimesh(fragment.vertices, fragment.faces, process=True)
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
            raise ValueError(f"non-manifold or invalid volume: {fragment.name}")
        meshes[fragment.name] = mesh
    # Robotiq-derived grasp heuristics are not valid YAM grasp labels.
    processed = replace(processed, fragments=tuple(replace(f, grasp=None) for f in processed.fragments))
    layout_path = write_processed_assembly(processed, Path(output), load_workcell_preset("dual_arm"))
    manifest = {
        "schema_version": 1, "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "source_pose_semantics": "assembled_ground_truth", "minimum_connectivity": 6,
        "voxelization": "sampled_surface_not_solid_fill", "clearance_m": 0,
        "manufacturing_validated": False, "gripper": "robotiq_proxy_not_yam",
        "source_to_canonical": processed.source_to_canonical.tolist(), "pieces": [],
    }
    for original, fragment in zip(assembly.fragments, processed.fragments):
        mesh = meshes[fragment.name].copy()
        # STL has no units: explicitly export millimetres for slicers.
        mesh.apply_scale(1000)
        stl = layout_path.parent / (fragment.name + '.stl')
        mesh.export(stl)
        manifest['pieces'].append({
            "id": fragment.name, "source_node": original.source_name,
            "source_node_transform": original.ground_truth_transform.tolist(),
            "stl": stl.name, "stl_units": "mm", "simulation_units": "m",
            "stl_sha256": hashlib.sha256(stl.read_bytes()).hexdigest(),
            "volume_m3": meshes[fragment.name].volume,
        })
    (layout_path.parent / 'geometry_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return layout_path
