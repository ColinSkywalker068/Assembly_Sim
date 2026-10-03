"""Build or view the static stack-sawtooth-shulker scene."""

from __future__ import annotations

import argparse
from pathlib import Path

from .builder import build_stack_sawtooth_scene
from .runtime import require_external_output, run_viewer
from .validation import validate_stack_sawtooth_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="validate and export the scene")
    build.add_argument("--output-usd", type=Path, required=True)
    launch = commands.add_parser("launch", help="open a view-only simulator window")
    launch.add_argument(
        "--smoke-frames", type=int, default=0, help=argparse.SUPPRESS
    )
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "launch":
        return run_viewer(smoke_frames=arguments.smoke_frames)

    output = require_external_output(arguments.output_usd)
    handles = build_stack_sawtooth_scene()
    try:
        report = validate_stack_sawtooth_scene(handles)
        if not report.ok:
            raise RuntimeError(f"scene validation failed: {report}")
        output.parent.mkdir(parents=True, exist_ok=True)
        handles.stage.GetRootLayer().Export(str(output))
    finally:
        handles.app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
