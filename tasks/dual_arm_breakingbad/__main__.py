"""Prepare, inspect, validate, build, or launch one Breaking Bad assembly."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from core.workcell.config import load_workcell_preset

from .domain.layout import load_layout
from .paths import require_external_output
from .preprocessing import (
    VoxelizationConfig,
    load_breaking_bad,
    process_assembly,
    write_processed_assembly,
)


def default_object_id(source: Path) -> str:
    source = Path(source)
    candidate = f"{source.parent.name}_{source.name}"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", candidate).strip("_")
    if not cleaned:
        raise ValueError(f"cannot derive an object identifier from {source}")
    return cleaned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="voxelize and stage one fragment set")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--object-id")
    prepare.add_argument("--pitch", type=float, default=0.016)
    prepare.add_argument("--target-length", type=float, default=0.40)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--overwrite", action="store_true")
    for name in ("validate", "inspect", "build", "launch"):
        command = commands.add_parser(name)
        command.add_argument("--assembly", type=Path, required=True)
        if name == "inspect":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--image-size", type=int, default=1200)
        elif name == "build":
            command.add_argument("--output-usd", dest="output", type=Path, required=True)
            command.add_argument("--capture-directory", type=Path)
        elif name == "launch":
            command.add_argument("--stream", action="store_true")
            command.add_argument("--output-directory", type=Path)
            command.add_argument("--smoke-frames", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def _prepare(arguments) -> Path:
    source = arguments.input.expanduser().resolve()
    object_id = arguments.object_id or default_object_id(source)
    output = arguments.output
    assembly = load_breaking_bad(source, object_id=object_id)
    processed = process_assembly(
        assembly,
        VoxelizationConfig(arguments.pitch, arguments.target_length, arguments.seed),
    )
    return write_processed_assembly(
        processed,
        require_external_output(output),
        load_workcell_preset("dual_arm"),
        overwrite=arguments.overwrite,
    )


def _validate(path: Path) -> dict:
    layout = load_layout(path)
    result = {
        "ok": True,
        "schema_version": layout.schema_version,
        "object_id": layout.object_id,
        "fragment_count": len(layout.fragment_names),
        "fragments": list(layout.fragment_names),
        "layout": str(layout.path),
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "prepare":
        print(_prepare(arguments))
        return 0
    if arguments.command == "validate":
        _validate(arguments.assembly)
        return 0
    if arguments.command == "inspect":
        from .rendering.voxel_mesh import render_voxel_assembly

        render_voxel_assembly(
            arguments.assembly,
            require_external_output(arguments.output),
            arguments.image_size,
        )
        return 0
    if arguments.command == "build":
        from core.workcell.cameras import capture_rgb

        from .scene.builder import build_assembly_scene
        from .scene.runtime import capture_requests
        from .scene.validation import validate_assembly_scene

        output = require_external_output(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        handles = build_assembly_scene(arguments.assembly)
        try:
            report = validate_assembly_scene(handles)
            if not report.ok:
                raise RuntimeError(f"scene validation failed: {report}")
            handles.stage.GetRootLayer().Export(str(output))
            if arguments.capture_directory is not None:
                capture_directory = require_external_output(arguments.capture_directory)
                capture_directory.mkdir(parents=True, exist_ok=True)
                for name, camera_path, _ in capture_requests(handles.cameras):
                    capture_rgb(
                        camera_path,
                        capture_directory / f"{name}.png",
                        resolution=handles.workcell.config.camera_resolution,
                    )
        finally:
            handles.app.close()
        return 0
    from .scene.runtime import run_interactive

    output_directory = (
        require_external_output(arguments.output_directory)
        if arguments.output_directory is not None
        else None
    )
    return run_interactive(
        arguments.assembly,
        stream=arguments.stream,
        smoke_frames=arguments.smoke_frames,
        output_directory=output_directory,
    )


if __name__ == "__main__":
    raise SystemExit(main())
