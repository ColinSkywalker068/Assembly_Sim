#!/usr/bin/env python3
"""Prepare, validate, or launch a voxelized fragmented-object assembly."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SHARED_SCRIPTS = REPO_ROOT / "03_scripts"
if str(SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS))

DEFAULT_SCENE_CONFIG = REPO_ROOT / "06_interactive_scene/config/scene.json"


def default_object_id(dataset: str, source: Path) -> str:
    source = Path(source)
    if dataset == "breaking-bad" and source.parent.name:
        candidate = f"{source.parent.name}_{source.name}"
    else:
        candidate = source.stem if source.suffix else source.name
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", candidate).strip("_")
    if not cleaned:
        raise ValueError(f"cannot derive an object identifier from {source}")
    return cleaned


def default_output_path(object_id: str) -> Path:
    return REPO_ROOT / "04_intermediate" / "assemblies" / object_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="load, normalize, voxelize, stage, and export one object")
    prepare.add_argument("--dataset", choices=("breaking-bad", "crag-glb"), required=True)
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--object-id")
    prepare.add_argument("--pitch", type=float, default=0.016)
    prepare.add_argument("--target-length", type=float, default=0.40)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    prepare.add_argument("--overwrite", action="store_true")
    validate = commands.add_parser("validate", help="validate generated metadata and assets")
    validate.add_argument("--assembly", type=Path, required=True)
    launch = commands.add_parser("launch", help="launch the selected object under Isaac Sim Python")
    launch.add_argument("--assembly", type=Path, required=True)
    launch.add_argument("--config", type=Path, default=DEFAULT_SCENE_CONFIG)
    launch.add_argument(
        "--stream",
        action="store_true",
        help="run headlessly with the Isaac Sim WebRTC livestream UI",
    )
    launch.add_argument("--smoke-frames", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def _load_workcell(path: Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "config_path": str(config_path),
        "environment": data["environment"],
        "robots": data["robots"],
        "placement": data.get("placement", {}),
    }


def prepare_command(arguments) -> Path:
    from fragment_assembly.loaders import load_breaking_bad, load_crag_glb
    from fragment_assembly.preprocess import write_processed_assembly
    from fragment_assembly.voxelize import VoxelizationConfig, process_assembly

    source = arguments.input.expanduser().resolve()
    object_id = arguments.object_id or default_object_id(arguments.dataset, source)
    assembly = (
        load_breaking_bad(source, object_id=object_id)
        if arguments.dataset == "breaking-bad"
        else load_crag_glb(source, object_id=object_id)
    )
    processed = process_assembly(
        assembly,
        VoxelizationConfig(
            pitch=arguments.pitch,
            target_length=arguments.target_length,
            seed=arguments.seed,
        ),
    )
    output = (
        arguments.output.expanduser().resolve()
        if arguments.output is not None
        else default_output_path(object_id)
    )
    return write_processed_assembly(
        processed,
        output,
        _load_workcell(arguments.scene_config),
        overwrite=arguments.overwrite,
    )


def validate_command(path: Path) -> dict:
    from fragment_assembly.layout import load_layout

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


def build_launch_command(
    config: Path,
    assembly: Path,
    smoke_frames: int = 0,
    python_executable: str | None = None,
    stream: bool = False,
) -> list[str]:
    command = [
        python_executable or sys.executable,
        str(REPO_ROOT / "06_interactive_scene/scripts/run_scene.py"),
        "--config",
        str(Path(config).expanduser().resolve()),
        "--assembly",
        str(Path(assembly).expanduser().resolve()),
    ]
    if smoke_frames:
        command.extend(("--smoke-frames", str(smoke_frames)))
    if stream:
        command.append("--stream")
    return command


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "prepare":
            layout_path = prepare_command(arguments)
            print(f"Generated assembly: {layout_path}")
            print(
                "Launch with Isaac Sim Python: "
                f"<isaac-python> {Path(__file__).name} launch --assembly {layout_path}"
            )
            return 0
        if arguments.command == "validate":
            validate_command(arguments.assembly)
            return 0
        command = build_launch_command(
            arguments.config,
            arguments.assembly,
            smoke_frames=arguments.smoke_frames,
            stream=arguments.stream,
        )
        return subprocess.call(command, cwd=REPO_ROOT)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
