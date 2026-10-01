#!/usr/bin/env python3
"""Render generated voxel fragment assets in their assembled goal configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from fragment_assembly.layout import load_layout
from fragment_assembly.voxel_mesh_render import render_voxel_assembly


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True, help="generated layout.json")
    parser.add_argument("--output", type=Path, required=True, help="directory for PNG renders")
    parser.add_argument("--size", type=int, default=900, help="square view size in pixels")
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    layout = load_layout(arguments.assembly)
    result = render_voxel_assembly(layout.path, arguments.output, image_size=arguments.size)
    readme = Path(arguments.output).expanduser().resolve() / "README.md"
    readme.write_text(
        "# Voxel assembly render\n\n"
        f"- Generated layout: `{layout.path}`\n"
        f"- Object: `{layout.object_id}`\n"
        f"- Voxel pitch: `{layout.pitch:g} m`\n"
        f"- Fragment count: `{len(layout.fragment_names)}`\n\n"
        "The fragments are the exported voxel visual meshes placed with their stored "
        "goal poses, i.e. the same blocky assets used by the Isaac scene.\n",
        encoding="utf-8",
    )
    print(f"Rendered {len(result.view_paths)} views: {result.contact_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
