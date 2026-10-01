from pathlib import Path

import trimesh

from fragment_assembly.loaders import load_breaking_bad
from fragment_assembly.original_mesh_render import VIEW_SPECS, render_assembly


def _write_piece(path: Path, translation) -> None:
    mesh = trimesh.creation.box(extents=(0.4, 0.2, 0.3))
    mesh.apply_translation(translation)
    mesh.export(path)


def test_view_specs_are_six_named_orthographic_and_perspective_views():
    assert tuple(spec.name for spec in VIEW_SPECS) == (
        "front",
        "back",
        "left",
        "right",
        "top",
        "perspective",
    )


def test_renderer_writes_each_view_and_contact_sheet_from_original_meshes(tmp_path):
    source = tmp_path / "fractured_0"
    source.mkdir()
    _write_piece(source / "piece_0.obj", (0.0, 0.0, 0.0))
    _write_piece(source / "piece_1.obj", (0.4, 0.0, 0.0))
    output = tmp_path / "renders"

    result = render_assembly(load_breaking_bad(source), output, image_size=240)

    assert tuple(path.name for path in result.view_paths) == (
        "front.png",
        "back.png",
        "left.png",
        "right.png",
        "top.png",
        "perspective.png",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in result.view_paths)
    assert result.contact_sheet.is_file()
    assert result.contact_sheet.stat().st_size > 0
