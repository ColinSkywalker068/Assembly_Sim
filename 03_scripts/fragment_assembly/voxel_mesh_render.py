"""CPU-only multi-view rendering of generated voxel visual assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .layout import load_layout
from .model import AssemblyInput, FragmentInput
from .original_mesh_render import RenderResult, render_assembly


def _pose_matrix(pose) -> np.ndarray:
    """Convert a generated layout pose from scalar-first quaternion form."""

    quaternion = np.asarray(pose["orientation_wxyz"], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=float,
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = pose["position"]
    return transform


def load_voxel_assembly(layout_path: Path) -> AssemblyInput:
    """Load generated local voxel surfaces in their stored goal configuration."""

    layout = load_layout(layout_path)
    fragments = []
    for name in layout.fragment_names:
        piece = layout.fragment(name)
        with np.load(layout.mesh_path(name)) as archive:
            mesh = trimesh.Trimesh(
                vertices=np.asarray(archive["v"], dtype=float),
                faces=np.asarray(archive["f"], dtype=np.int64),
                process=False,
            )
        transform = _pose_matrix(layout.goal_pose(name))
        mesh.apply_transform(transform)
        fragments.append(
            FragmentInput(
                name=name,
                source_name=str(piece["mesh"]),
                source_path=layout.mesh_path(name),
                mesh=mesh,
                ground_truth_transform=transform,
            )
        )
    return AssemblyInput(
        object_id=layout.object_id,
        loader="generated_voxel_layout",
        source_path=layout.path,
        fragments=tuple(fragments),
    )


def render_voxel_assembly(
    layout_path: Path, output_directory: Path, image_size: int = 900
) -> RenderResult:
    """Render exactly the voxel visual meshes consumed by the Isaac scene."""

    return render_assembly(load_voxel_assembly(layout_path), output_directory, image_size)
