"""CPU-only multi-view rendering for an assembled set of original meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from ..domain.model import AssemblyInput


@dataclass(frozen=True)
class ViewSpec:
    name: str
    elevation: float
    azimuth: float
    orthographic: bool = True


VIEW_SPECS = (
    ViewSpec("front", 12.0, -90.0),
    ViewSpec("back", 12.0, 90.0),
    ViewSpec("left", 12.0, 0.0),
    ViewSpec("right", 12.0, 180.0),
    ViewSpec("top", 90.0, -90.0),
    ViewSpec("perspective", 25.0, -55.0, orthographic=False),
)

_PALETTE = (
    (0.78, 0.20, 0.16),
    (0.98, 0.69, 0.16),
    (0.18, 0.42, 0.78),
    (0.28, 0.68, 0.36),
    (0.63, 0.32, 0.71),
    (0.18, 0.67, 0.70),
    (0.91, 0.45, 0.18),
    (0.45, 0.45, 0.45),
)


@dataclass(frozen=True)
class RenderResult:
    view_paths: tuple[Path, ...]
    contact_sheet: Path


def _scene_bounds(assembly: AssemblyInput) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.concatenate([fragment.mesh.vertices for fragment in assembly.fragments])
    return np.min(vertices, axis=0), np.max(vertices, axis=0)


def _configure_axes(ax, lower: np.ndarray, upper: np.ndarray) -> None:
    centre = (lower + upper) / 2.0
    span = float(np.max(upper - lower))
    half = max(span * 0.56, 1e-6)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_axis_off()


def _add_fragments(ax, assembly: AssemblyInput) -> None:
    for index, fragment in enumerate(assembly.fragments):
        mesh = fragment.mesh
        triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
        collection = Poly3DCollection(
            triangles,
            facecolors=_PALETTE[index % len(_PALETTE)],
            edgecolors="none",
            linewidths=0.0,
            alpha=1.0,
        )
        ax.add_collection3d(collection)


def _render_view(
    assembly: AssemblyInput,
    spec: ViewSpec,
    destination: Path,
    image_size: int,
) -> None:
    lower, upper = _scene_bounds(assembly)
    figure = plt.figure(figsize=(image_size / 100.0, image_size / 100.0), dpi=100)
    try:
        figure.patch.set_facecolor("white")
        axes = figure.add_subplot(111, projection="3d")
        _add_fragments(axes, assembly)
        _configure_axes(axes, lower, upper)
        axes.view_init(elev=spec.elevation, azim=spec.azimuth)
        axes.set_proj_type("ortho" if spec.orthographic else "persp")
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        figure.savefig(destination, dpi=100, facecolor="white")
    finally:
        plt.close(figure)


def _write_contact_sheet(view_paths: tuple[Path, ...], destination: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in view_paths]
    try:
        width, height = images[0].size
        sheet = Image.new("RGB", (width * 3, height * 2), "white")
        for index, image in enumerate(images):
            sheet.paste(image, ((index % 3) * width, (index // 3) * height))
        sheet.save(destination)
    finally:
        for image in images:
            image.close()


def render_assembly(
    assembly: AssemblyInput, output_directory: Path, image_size: int = 900
) -> RenderResult:
    """Render source meshes in their supplied ground-truth assembly frame."""

    if image_size < 64:
        raise ValueError("image_size must be at least 64 pixels")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    view_paths = tuple(output / f"{spec.name}.png" for spec in VIEW_SPECS)
    for spec, path in zip(VIEW_SPECS, view_paths):
        _render_view(assembly, spec, path, image_size)
    contact_sheet = output / "contact_sheet.png"
    _write_contact_sheet(view_paths, contact_sheet)
    return RenderResult(view_paths=view_paths, contact_sheet=contact_sheet)
