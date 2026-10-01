"""Dataset-neutral input model for one complete fragmented assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class FragmentInput:
    """One source fragment and its ground-truth pose in the assembly frame."""

    name: str
    source_name: str
    source_path: Path
    mesh: trimesh.Trimesh
    ground_truth_transform: np.ndarray

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("fragment name must not be empty")
        if not isinstance(self.mesh, trimesh.Trimesh):
            raise TypeError(f"fragment {self.name} is not a triangle mesh")
        if len(self.mesh.vertices) == 0 or len(self.mesh.faces) == 0:
            raise ValueError(f"fragment {self.source_name} is an empty triangle mesh")
        faces = np.asarray(self.mesh.faces)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"fragment {self.source_name} is not a triangle mesh")
        transform = np.asarray(self.ground_truth_transform, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"fragment {self.name} has an invalid ground-truth transform")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0)):
            raise ValueError(f"fragment {self.name} has a non-affine ground-truth transform")
        object.__setattr__(self, "source_path", Path(self.source_path).expanduser().resolve())
        object.__setattr__(self, "ground_truth_transform", transform.copy())


@dataclass(frozen=True)
class AssemblyInput:
    """One complete assembly as an ordered collection of source fragments."""

    object_id: str
    loader: str
    source_path: Path
    fragments: tuple[FragmentInput, ...]

    def __post_init__(self) -> None:
        fragments = tuple(self.fragments)
        if not self.object_id:
            raise ValueError("object_id must not be empty")
        if not fragments:
            raise ValueError("assembly must contain at least one fragment")
        names = [fragment.name for fragment in fragments]
        if len(names) != len(set(names)):
            raise ValueError("assembly fragment names must be unique")
        object.__setattr__(self, "source_path", Path(self.source_path).expanduser().resolve())
        object.__setattr__(self, "fragments", fragments)
