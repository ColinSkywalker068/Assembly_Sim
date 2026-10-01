#!/usr/bin/env python3
"""Render a Breaking Bad source fragment set in its assembled mesh coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

from fragment_assembly.loaders import load_breaking_bad
from fragment_assembly.original_mesh_render import render_assembly


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="directory containing piece_<n>.obj")
    parser.add_argument("--output", type=Path, required=True, help="directory for PNG renders")
    parser.add_argument("--size", type=int, default=900, help="square view size in pixels")
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    assembly = load_breaking_bad(arguments.input)
    result = render_assembly(assembly, arguments.output, image_size=arguments.size)
    print(f"Rendered {len(result.view_paths)} views: {result.contact_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
