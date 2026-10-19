"""DexAgent baseline: export inputs, run one Stage 3 trial, score a finished trial, or execute a delivery.

  python -m tasks.dual_arm_crag_skull.baselines.dexagent export --task TASK.json --out DIR [--bench crx_yam]
  python -m tasks.dual_arm_crag_skull.baselines.dexagent run --task TASK.json --trial DIR --llm mock:TURNS.json
  python -m tasks.dual_arm_crag_skull.baselines.dexagent run --task TASK.json --trial DIR --live [--budget-pool DIR]
  python -m tasks.dual_arm_crag_skull.baselines.dexagent score --trial DIR [--records RUNS.jsonl]
  "$ISAAC_PYTHON" -m tasks.dual_arm_crag_skull.baselines.dexagent execute --trial DIR --backend isaac [--records R]

A task is an AssemblyTask record (`python -m tasks.dual_arm_crag_skull.baselines task ...`) or --layout/--anchor
[--seed] under the canonical protocol of configs/jing.json; `run` refuses a task that is not its layout under it. DEXAGENT_PYTHON and
DEXAGENT_SOURCE select the DexAgent venv and the clone each trial is cloned from. `run --live` calls the Claude
API through the transport (key from this process's environment), paying from the budget pool every live baseline run
shares (baselines.llm_budget, made once with `python -m tasks.dual_arm_crag_skull.baselines budget create`); nothing
else here calls the model.

--bench crx_yam puts DexAgent on the hardware every method shares (README "Port to the shared hardware"): the port's
checkout with its CRX + YAM bench, under S0's TEST limits (ASPIRE's; configs/jing.json has no real ones yet, so every
record there is test_only). Such a trial's delivery is scored once `execute` has run it through S0 on a backend:
isaac (Isaac Sim's interpreter, sim/backend.py) or kinematic (the physics-free test double).
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

from ...__main__ import DEFAULT_CONFIG, validate
from ...contracts.actions import ARMS
from ..budget import configured_pool_dir
from ..task import AssemblyTask, configured_initial_condition, task_from_layout
from . import dexenv
from .bench.checkout import prepare_checkout
from .export import ANCHOR_MODES, DEXAGENT_COMMIT, VARIANTS, ExportOptions, export_task
from .runner import TrialConfig, rescore, run_trial
from .s0_check import limits_record

TEST_ONLY_WARNING = ('the shared hardware runs under S0 TEST limits (configs/jing.json has none), so its records are '
                     'test_only')


def load_task(args, config):
    if args.task is not None:
        if args.layout is not None or args.seed is not None:
            raise SystemExit('give --task or --layout/--anchor/--seed, not both')
        return AssemblyTask.from_record(json.loads(args.task.read_text()))
    if args.layout is None or args.anchor is None:
        raise SystemExit('give --task TASK.json or --layout LAYOUT.json --anchor ID')
    task = task_from_layout(args.layout, anchor_id=args.anchor)
    return task if args.seed is None else configured_initial_condition(task, args.seed, config)


def _task_arguments(parser):
    parser.add_argument('--task', type=Path, help='AssemblyTask record (JSON)')
    parser.add_argument('--layout', type=Path)
    parser.add_argument('--anchor')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--pad-xy', type=float, nargs=2, default=None, metavar=('X', 'Y'),
                        help='rig position of the assembly pad centre (default 0.0 0.40)')
    parser.add_argument('--anchor-mode', choices=ANCHOR_MODES, default='fixed')
    parser.add_argument('--bench', choices=tuple(VARIANTS), default='native',
                        help="native: DexAgent's own YAM + Sharpa bench; crx_yam: the hardware every method shares")


def _options(args):
    return ExportOptions(anchor_mode=args.anchor_mode, bench=args.bench,
                         **({'pad_xy': tuple(args.pad_xy)} if args.pad_xy else {}))


def shared_test_limits(task):
    """S0Limits of the shared hardware for `task`: ASPIRE's TEST values, the YAM's width range on both arms."""
    from ...s0.kinematics import yam_width_range_m
    from ..aspire.__main__ import test_limits
    return test_limits(task, {arm: yam_width_range_m() for arm in ARMS})


def execute(args, config):
    """`execute`: a shared-hardware trial's delivery through S0 on the kinematic double or on Isaac Sim."""
    import numpy as np
    from core.workcell.config import load_workcell_preset

    from ...s0.kinematics import yam_model, yam_width_range_m
    from ..backend import KinematicBackend
    from .execute_trial import execute_trial, trial_limits

    print(f'warning: {TEST_ONLY_WARNING}', file=sys.stderr)
    isaac = None
    if args.backend == 'isaac':
        from ..aspire.__main__ import isaac_rig  # the same construction (and failure exit) as ASPIRE's trials
        task = AssemblyTask.from_record(json.loads((args.trial / 'inputs' / 'export_manifest.json').read_text())['task'])
        model, _, _, isaac = isaac_rig(task, config)
        backend = isaac
    else:
        limits, workcell = trial_limits(args.trial), load_workcell_preset('dual_arm')
        model = yam_model()
        backend = KinematicBackend(model, home_joints={arm: np.array(workcell.robot(arm).home_joint_positions)
                                                       for arm in ARMS},
                                   width_range_m={arm: yam_width_range_m() for arm in ARMS}, control_hz=limits.servo_hz)
    code = 1
    try:
        record = execute_trial(args.trial, backend, model, test_only=True, records=args.records)
        print(json.dumps(record.to_record(), indent=1), flush=True)
        code = 0
        return code
    except BaseException:
        if isaac is not None:  # the shutdown below ends the process before Python could report the exception
            traceback.print_exc()
        raise
    finally:
        if isaac is not None:
            try:
                isaac.close()
            except BaseException:  # reported and counted as a failure; the execution's own exception propagates
                traceback.print_exc()
                code = 1
            isaac.shutdown(code)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest='command', required=True)
    export_parser = commands.add_parser('export', help='write the DexAgent inputs for one task (fresh clone in OUT)')
    _task_arguments(export_parser)
    export_parser.add_argument('--out', type=Path, required=True)
    run_parser = commands.add_parser('run', help='one Stage 3 trial: clone, export, run, score')
    _task_arguments(run_parser)
    run_parser.add_argument('--trial', type=Path, required=True)
    run_parser.add_argument('--llm', default='claude', help="'claude' (needs --live) or mock:TURNS.json")
    run_parser.add_argument('--live', action='store_true', help='allow live Claude API calls')
    run_parser.add_argument('--budget-pool', type=Path,
                            help='the shared budget pool a live run pays from (default: baselines.llm_budget.pool_dir)')
    run_parser.add_argument('--max-rounds', type=int, default=100)
    run_parser.add_argument('--max-wall-minutes', type=float, default=240.0)
    run_parser.add_argument('--methods', choices=('none', 'published'), default='none',
                            help="Stage 3 method modules after an exhausted search: none (default) or DexAgent's three")
    run_parser.add_argument('--records', type=Path, help='append the RunRecord to this JSONL file')
    run_parser.add_argument('--allow-heldout', action='store_true',
                            help='run a held-out seed (final evaluation only; needs --records, once per variant)')
    score_parser = commands.add_parser('score', help='score a finished trial again from its files')
    score_parser.add_argument('--trial', type=Path, required=True)
    score_parser.add_argument('--records', type=Path)
    execute_parser = commands.add_parser('execute', help="run a shared-hardware trial's delivery through S0 and score "
                                                         'it')
    execute_parser.add_argument('--trial', type=Path, required=True)
    execute_parser.add_argument('--backend', choices=('kinematic', 'isaac'), required=True,
                                help="isaac: Isaac Sim (run under its interpreter); kinematic: the physics-free test "
                                     "double, whose results are not baseline results")
    execute_parser.add_argument('--records', type=Path, help='append the RunRecord to this JSONL file')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    validate(config)
    if args.command == 'score':
        print(json.dumps(rescore(args.trial, records=args.records).to_record(), indent=1))
        return 0
    if args.command == 'execute':
        return execute(args, config)
    task = load_task(args, config)
    limits = shared_test_limits(task) if args.bench == 'crx_yam' else None
    if limits is not None:
        print(f'warning: {TEST_ONLY_WARNING}', file=sys.stderr)
    if args.command == 'export':
        if args.out.exists() and any(args.out.iterdir()):
            parser.error(f'{args.out} is not empty')
        clone = args.out / 'DexAgent'
        if limits is None:
            dexenv.fresh_clone(dexenv.dexagent_source(), clone, DEXAGENT_COMMIT)
        else:
            prepare_checkout(dexenv.dexagent_source(), clone, limits_record(limits))
        manifest = export_task(task, args.out / 'inputs', clone=clone, python=dexenv.dexagent_python(),
                               env=dexenv.scrubbed_env(args.out), options=_options(args),
                               s0_limits=None if limits is None else limits_record(limits))
        print(json.dumps({k: manifest[k] for k in ('checks_ok', 'checks_failed', 'order', 'anchor_object_id')}))
        return 0 if manifest['checks_ok'] else 1
    if args.llm == 'claude' and not args.live:
        parser.error('a live run calls the Claude API: add --live, or use --llm mock:TURNS.json')
    methods = ('physics_informed_retargeting', 'sampling_mpc_refinement', 'reinforcement_learning_tracking') \
        if args.methods == 'published' else ()
    pool = (args.budget_pool or configured_pool_dir(config)) if args.llm == 'claude' else None
    trial = TrialConfig(args.trial, llm=args.llm, budget_pool=pool, max_rounds=args.max_rounds,
                        max_wall_minutes=args.max_wall_minutes, methods=methods, export=_options(args),
                        s0_limits=limits)
    record = run_trial(task, trial, records=args.records, allow_heldout=args.allow_heldout)
    print(json.dumps(record.to_record(), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
