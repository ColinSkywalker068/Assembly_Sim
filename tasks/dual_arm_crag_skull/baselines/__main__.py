"""Write an AssemblyTask record, or summarize RunRecord JSONL files. Nothing here runs a method.

  python -m tasks.dual_arm_crag_skull.baselines task --layout LAYOUT.json --anchor ID [--seed N [--allow-heldout]]
      --output TASK.json
  python -m tasks.dual_arm_crag_skull.baselines summarize RUNS.jsonl [RUNS.jsonl ...] [--include-test-only]
  python -m tasks.dual_arm_crag_skull.baselines budget create --note TEXT [--pool DIR]
  python -m tasks.dual_arm_crag_skull.baselines budget status [--pool DIR]
  python -m tasks.dual_arm_crag_skull.baselines sandbox-check [--python VENV_PYTHON] [--out DIR]

Tasks follow the canonical protocol of configs/jing.json (seed bands, placement, success criterion). Writing a held-out
seed's initial condition to disk needs --allow-heldout. `summarize` refuses records it cannot count as they stand (see
baselines.scoring.summarize) and lists held-out claims that have no record among the given files. `budget create`
makes the one model budget every live baseline run spends from (baselines/budget.py), with baselines.llm_budget's
total, once; `budget status` shows what is spent, held and left. `sandbox-check` runs baselines/sandbox.py's probe
in the sandbox DexAgent's processes get (DexAgent's venv by default, its scrubbed environment) and exits 1 unless the
home directory's .config and .ssh, the other runs, environments and upstream clones are out of sight, no network
answers and only the output directory takes writes.
"""
import argparse
import json
from pathlib import Path

from ..__main__ import DEFAULT_CONFIG, validate
from .budget import BudgetPool, configured_pool_dir, configured_total_usd
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
    budget_parser = commands.add_parser('budget', help='the model budget pool shared by the live baseline runs')
    budget_parser.add_argument('action', choices=('create', 'status'))
    budget_parser.add_argument('--pool', type=Path, help='the pool directory (default: baselines.llm_budget.pool_dir)')
    budget_parser.add_argument('--note', help='why the pool exists (create)')
    check_parser = commands.add_parser('sandbox-check', help='probe the bwrap sandbox DexAgent runs in')
    check_parser.add_argument('--python', type=Path, help="the venv interpreter to probe with (default: DexAgent's)")
    check_parser.add_argument('--out', type=Path, help='the writable directory (default: a new one under runs/)')
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
    if args.command == 'sandbox-check':
        return sandbox_check(args)
    if args.command == 'budget':
        directory = args.pool or configured_pool_dir(config)
        if args.action == 'create':
            if not args.note:
                parser.error('budget create needs --note')
            pool = BudgetPool.create(directory, configured_total_usd(config), note=args.note)
        else:
            pool = BudgetPool(directory)
        print(json.dumps({'pool': str(pool.dir), **pool.status()}, indent=1))
        return 0
    paths = [path.resolve() for path in args.runs]
    if len(set(paths)) != len(paths):
        parser.error('a records file is given more than once')
    records = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    print(json.dumps({'groups': summarize(records, include_test_only=args.include_test_only),
                      'unrecorded_heldout_claims': unrecorded_claims(records)}, indent=1))
    return 0


def sandbox_check(args):
    import time
    from ..workspace import REPO_ROOT, RUNS
    from . import sandbox
    from .dexagent import dexenv
    python = args.python or Path(dexenv.dexagent_python())
    out = args.out or RUNS / 'sandbox_check' / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    home = Path.home()
    readonly = [*sandbox.python_paths(python), REPO_ROOT / 'core', REPO_ROOT / 'tasks']
    bound = [inside for _, inside in readonly[:-2]] + [REPO_ROOT / 'core', REPO_ROOT / 'tasks', out]
    # what must not exist inside; a bound path, or a directory above one (the home directory above a venv's
    # interpreter, runs/ above the writable trial directory), exists by construction: the home directory is checked
    # by its listing instead, and the runs beside this one by the budget pool's directory
    hidden = [path for path in (home / '.config', home / '.ssh', home, REPO_ROOT / 'runs', RUNS / 'llm_budget',
                                REPO_ROOT / 'third_party', REPO_ROOT / 'envs' / 'asm', REPO_ROOT / 'envs' / 'aspire')
              if not any(path == b or path in b.parents for b in bound)]
    prefix = sandbox.bwrap_prefix(writable=[out], readonly=readonly, cwd=out)
    found = sandbox.probe(python, prefix, hidden=hidden, home=home, writable=[out], readonly=[REPO_ROOT / 'tasks'],
                          env=dexenv.scrubbed_env(out, python))
    print(json.dumps({'python': str(python), 'out': str(out), 'hidden': [str(p) for p in hidden], **found}, indent=1))
    return 0 if found['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
