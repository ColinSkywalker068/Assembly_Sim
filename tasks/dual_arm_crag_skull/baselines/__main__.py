"""Write an AssemblyTask record, or summarize RunRecord JSONL files. Nothing here runs a method.

  python -m tasks.dual_arm_crag_skull.baselines task --layout LAYOUT.json --anchor ID [--seed N [--allow-heldout]]
      --output TASK.json
  python -m tasks.dual_arm_crag_skull.baselines summarize RUNS.jsonl [RUNS.jsonl ...] [--include-test-only]

Tasks follow the canonical protocol of configs/jing.json (seed bands, placement, success criterion). Writing a held-out
seed's initial condition to disk needs --allow-heldout. `summarize` refuses records it cannot count as they stand (see
baselines.scoring.summarize) and lists held-out claims that have no record among the given files.
"""
import argparse
import json
from pathlib import Path

from ..__main__ import DEFAULT_CONFIG, validate
from .scoring import summarize, unrecorded_claims
from .task import canonical_split, configured_initial_condition, task_from_layout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest='command', required=True)
    task_parser = commands.add_parser('task', help='write an AssemblyTask record (base, or seeded with --seed)')
    task_parser.add_argument('--layout', type=Path, required=True)
    task_parser.add_argument('--anchor', required=True)
    task_parser.add_argument('--seed', type=int)
    task_parser.add_argument('--allow-heldout', action='store_true', help='allow writing a held-out seed')
    task_parser.add_argument('--output', type=Path, required=True)
    summary_parser = commands.add_parser('summarize', help='success rates with Wilson 95%% intervals')
    summary_parser.add_argument('runs', type=Path, nargs='+')
    summary_parser.add_argument('--include-test-only', action='store_true',
                                help='also count plumbing runs (physics-free double, mock model)')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    validate(config)
    if args.command == 'task':
        if args.seed is not None and canonical_split(args.seed) == 'heldout' and not args.allow_heldout:
            parser.error(f'seed {args.seed} is held out; writing its initial condition needs --allow-heldout')
        task = task_from_layout(args.layout, anchor_id=args.anchor)
        if args.seed is not None:
            task = configured_initial_condition(task, args.seed, config)
        if args.output.exists():
            parser.error(f'{args.output} exists; refusing to overwrite')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(task.to_record(), indent=1) + '\n')
        print(json.dumps({'task_id': task.task_id, 'seed': task.seed, 'split': task.split, 'output': str(args.output)}))
        return 0
    paths = [path.resolve() for path in args.runs]
    if len(set(paths)) != len(paths):
        parser.error('a records file is given more than once')
    records = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    print(json.dumps({'groups': summarize(records, include_test_only=args.include_test_only),
                      'unrecorded_heldout_claims': unrecorded_claims(records)}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
