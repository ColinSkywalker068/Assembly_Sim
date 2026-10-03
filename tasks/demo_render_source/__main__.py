"""Command-line workflow for the original dual-arm CRAG storyboard."""

from __future__ import annotations

import argparse
from pathlib import Path

from tasks.demo_render_source.choreography import build_choreography
from tasks.demo_render_source.paths import require_external_output
from tasks.demo_render_source.preprocessing import prepare_storyboard_assets
from tasks.demo_render_source.probe import probe_robot_models
from tasks.demo_render_source.storyboard import render_storyboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="create studded CRAG storyboard assets")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--predicted", type=Path)
    prepare.add_argument("--work-dir", type=Path, required=True)
    prepare.add_argument("--target-length", type=float, default=0.40)
    prepare.add_argument("--pitch", type=float, default=0.016)

    choreograph = commands.add_parser("choreograph", help="build the scripted timeline")
    choreograph.add_argument("--layout", type=Path, required=True)
    choreograph.add_argument("--probe", type=Path, required=True)
    choreograph.add_argument("--output", type=Path, required=True)
    choreograph.add_argument("--fps", type=int, default=30)

    render = commands.add_parser("render", help="render storyboard frames")
    render.add_argument("--choreography", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--quick", action="store_true")
    render.add_argument("--subframes", type=int, default=6)

    run = commands.add_parser("run", help="prepare, choreograph, and render")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--predicted", type=Path)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--fps", type=int, default=30)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare_storyboard_assets(
                args.input,
                require_external_output(args.work_dir) / "bricks",
                args.predicted,
                args.target_length,
                args.pitch,
            )
        elif args.command == "choreograph":
            build_choreography(args.layout, args.probe, args.output, args.fps)
        elif args.command == "render":
            render_storyboard(
                args.choreography,
                args.output,
                quick=args.quick,
                subframes=args.subframes,
            )
        else:
            work = require_external_output(args.work_dir)
            output = require_external_output(args.output)
            layout = prepare_storyboard_assets(
                args.input, work / "bricks", args.predicted
            )
            probe = probe_robot_models(work / "probe")
            choreography = build_choreography(
                layout, probe, work / "choreography.json", args.fps
            )
            render_storyboard(choreography, output)
        return 0
    except Exception as exc:
        parser = build_parser()
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
