"""Build, preview, execute, or collect the small single-arm shulker task."""

import argparse
import os
import sys
from pathlib import Path

from .builder import build_stack_sawtooth_scene
from .runtime import require_external_output, run_viewer
from .validation import validate_stack_sawtooth_scene


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collect = commands.add_parser('collect-manifest', help='attempt all 50 conditions with synchronized recording')
    collect.add_argument('--manifest', type=Path, required=True)
    collect.add_argument('--output-root', type=Path, required=True)
    attempt = commands.add_parser('collect-attempt', help='execute and record one manifest condition')
    attempt.add_argument('--manifest', type=Path, required=True)
    attempt.add_argument('--demo-id', required=True)
    attempt.add_argument('--output', type=Path, required=True)
    generate = commands.add_parser('generate-manifest', help='write 50 mating-ready scene conditions')
    generate.add_argument('--dataset-seed', type=int, required=True)
    generate.add_argument('--output-json', type=Path, required=True)
    preview = commands.add_parser('render-manifest', help='render annotated initial scenes and contact sheets')
    preview.add_argument('--manifest', type=Path, required=True)
    preview.add_argument('--output-root', type=Path, required=True)
    build = commands.add_parser('build', help='validate and export the initial scene')
    build.add_argument('--output-usd', type=Path, required=True)
    launch = commands.add_parser('launch', help='open the initial scene')
    launch.add_argument('--smoke-frames', type=int, default=0, help=argparse.SUPPRESS)
    scripted = commands.add_parser('scripted', help='physically pick blue, then stack green using GT poses')
    scripted.add_argument('--output-root', type=Path, default=Path(
        '/local_data/yz11445/experiments/assembly_sim/stack_sawtooth_shulker_small'))
    scripted.add_argument('--gui', action='store_true')
    teleop = commands.add_parser('teleop', help='control the right EEF with the keyboard')
    teleop.add_argument('--speed', type=float, default=0.05)
    teleop.add_argument('--roll-speed', type=float, default=90.)
    teleop.add_argument('--smoke-frames', type=int, default=0, help=argparse.SUPPRESS)
    teleop.add_argument('--headless', action='store_true', help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    selected_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if selected_gpu.isdecimal():
        sys.argv.append(f'--/renderer/activeGpu={selected_gpu}')
    if args.command == 'collect-manifest':
        from .collection import collect_manifest
        print(collect_manifest(args.manifest, args.output_root))
        return 0
    if args.command == 'collect-attempt':
        from .collection import run_attempt
        run_attempt(args.manifest, args.demo_id, args.output)
        return 0
    if args.command == 'generate-manifest':
        from .dataset import generate_manifest, write_manifest
        write_manifest(generate_manifest(args.dataset_seed), require_external_output(args.output_json))
        return 0
    if args.command == 'render-manifest':
        from .dataset import read_manifest
        from .preview import render_dataset_preview
        result = render_dataset_preview(read_manifest(args.manifest), require_external_output(args.output_root))
        print(result.run_directory)
        return 0
    if args.command == 'scripted':
        from .scripted import run_scripted
        return run_scripted(args.output_root, headless=not args.gui)
    if args.command == 'launch':
        return run_viewer(smoke_frames=args.smoke_frames)
    if args.command == 'teleop':
        from .teleop import run_teleop
        return run_teleop(speed=args.speed, roll_speed=args.roll_speed,
                          smoke_frames=args.smoke_frames, headless=args.headless)
    output = require_external_output(args.output_usd)
    handles = build_stack_sawtooth_scene()
    try:
        report = validate_stack_sawtooth_scene(handles)
        if not report.ok:
            raise RuntimeError(f'scene validation failed: {report}')
        output.parent.mkdir(parents=True, exist_ok=True)
        handles.stage.GetRootLayer().Export(str(output))
    finally:
        handles.app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
