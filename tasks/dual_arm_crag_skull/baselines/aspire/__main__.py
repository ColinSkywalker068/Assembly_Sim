"""ASPIRE baseline CLI: prepare the patched checkout, run one trial, or evaluate a frozen program on held-out seeds.

Every rig is the one hardware set every method uses: two CRX arms with the YAM linear_4310 gripper (s0.kinematics
yam_model, provisional adapter plate). Backends:
- `kinematic`: the physics-free KinematicBackend at S0's servo rate; a test double, never a baseline result.
- `isaac`: sim/backend.py on Isaac Sim, one instance for every seed of the process. Run this CLI with the Isaac Sim
  interpreter and the patched checkout's `src` on PYTHONPATH (cap_harness needs only numpy, scipy, PyYAML and
  requests; the Isaac Sim environment on maxwell already has them). The process ends through the backend's
  shutdown, Kit's way out.
The S0 limits are TEST values on both (configs/jing.json still has none), so every record is test_only.
Generated programs run in this process without a sandbox: use an isolated host without credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

from ...__main__ import DEFAULT_CONFIG, validate
from .checkout import DEFAULT_DESTINATION, PATCH, PINNED_COMMIT, prepare_checkout

BACKENDS = ('kinematic', 'isaac')
# Step budget of a trial at S0's 60 Hz servo rate: 5 minutes of simulated time. Provisional; every compared method
# must get the same budget, so fix it with the other protocol values before any held-out seed runs.
MAX_STEPS = 18000
TEST_ONLY_WARNING = ('S0 limits are TEST values (configs/jing.json has none), so every record is test_only; backend '
                     'kinematic is moreover a physics-free test double')
# TEST S0 values, the same as tests/test_s0_adapter.py limits() plus the contact phase of the S0 tests and S0's boxes at
# the grippers' openings (2026-10-21). Their 4 mm margin: with the 1 mm seams, in the scripted runs on Isaac, a gripper
# held at one command opened at most 1.15 mm wider while it moved (a fragment knocked in the grip) and a release
# overshot by 0.05 mm at most; jams before the seams pried fingers open by up to 12 mm, which no margin covers (S0 reads
# the opening at each chunk's start); the real gripper's width is not calibrated. The real CRX acceleration limit,
# clearances and contact values are still null in configs/jing.json.
TEST_ONLY_LIMITS = dict(max_joint_acceleration_rad_s2=20.0, arm_arm_clearance_m=0.02, table_clearance_m=0.01,
                        tcp_min_height_m=0.02, servo_hz=60, max_observation_age_s=0.5,
                        anchor_position_tolerance_m=0.002, anchor_rotation_tolerance_rad=0.005,
                        contact_table_clearance_m=0.002, contact_region_radius_m=0.12, contact_max_speed_m_s=0.05,
                        finger_opening_margin_m=0.004)


def test_limits(task, widths):
    """S0Limits with the TEST values for `task`'s table and the gripper width range `widths` (arm -> (low, high))."""
    import numpy as np

    from ...s0.adapter import S0Limits
    from ...s0.kinematics import load_kinematics
    values = dict(TEST_ONLY_LIMITS)
    return S0Limits(max_joint_velocity_rad_s=load_kinematics()['max_joint_velocity_rad_s'],
                    max_joint_acceleration_rad_s2=np.full(6, values.pop('max_joint_acceleration_rad_s2')),
                    gripper_width_range_m=dict(widths), table_top_z_m=task.table_top_z_m, **values)


def kinematic_rig(task):
    """(model, backend factory, S0 limits) of the test-only kinematic backend for `task`, at S0's servo rate."""
    import numpy as np
    from core.workcell.config import load_workcell_preset

    from ...contracts.actions import ARMS
    from ...s0.kinematics import yam_model, yam_width_range_m
    from ..backend import KinematicBackend

    model, workcell = yam_model(), load_workcell_preset('dual_arm')
    home = {arm: np.array(workcell.robot(arm).home_joint_positions) for arm in ARMS}
    widths = {arm: yam_width_range_m() for arm in ARMS}
    limits = test_limits(task, widths)
    return model, lambda: KinematicBackend(model, home_joints=home, width_range_m=widths,
                                           control_hz=limits.servo_hz), limits


def isaac_rig(task, config):
    """(model, backend factory, S0 limits, backend) on Isaac Sim; the factory returns the one backend every time."""
    try:
        from ...sim.backend import IsaacAssemblyBackend
        import isaacsim  # noqa: F401  (only to fail early and clearly outside the Isaac Sim interpreter)
    except ImportError as exc:
        raise ValueError('backend isaac needs the Isaac Sim interpreter (with the patched checkout on PYTHONPATH): '
                         f'{exc}') from exc
    try:
        backend = IsaacAssemblyBackend(config)
        limits = test_limits(task, backend.width_range)
    except BaseException:  # Kit may be up already and would end the process with status 0: report and exit 1
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    return backend.model, lambda: backend, limits, backend


def parse_seeds(text):
    """'1-100' or '1,2,5' or a mix: distinct integers in the given order."""
    seeds = []
    for part in text.split(','):
        low, _, high = part.strip().partition('-')
        try:
            first, last = int(low), int(high or low)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f'invalid seed list {text!r}') from exc
        if last < first:
            raise argparse.ArgumentTypeError(f'seed range {part.strip()!r} runs backwards')
        seeds.extend(range(first, last + 1))
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(f'seed list {text!r} is empty or repeats a seed')
    return seeds


def build_parser():
    parser = argparse.ArgumentParser(prog='python -m tasks.dual_arm_crag_skull.baselines.aspire', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    checkout = commands.add_parser('checkout', help='clone the local ASPIRE clone, pin the commit, apply the patch')
    checkout.add_argument('--source', type=Path, required=True, help='local ASPIRE clone (only read)')
    checkout.add_argument('--destination', type=Path, default=None, help='empty directory for the patched checkout')
    for name, text in (('run', 'run one trial'), ('evaluate', 'run a frozen program on held-out seeds')):
        command = commands.add_parser(name, help=text)
        command.add_argument('--program', type=Path, required=True)
        command.add_argument('--task', type=Path, required=True, help='JSON record of the base AssemblyTask')
        if name == 'run':
            command.add_argument('--seed', type=int, required=True)
            command.add_argument('--mode', choices=('practice', 'evaluation'), required=True)
            command.add_argument('--records', type=Path, default=None,
                                 help='JSONL file to append the RunRecord to (required in evaluation mode)')
        else:
            command.add_argument('--seeds', type=parse_seeds, required=True, help="held-out seeds, e.g. '1-100'")
            command.add_argument('--records', type=Path, required=True, help='JSONL file to append RunRecords to')
        command.add_argument('--output-root', type=Path, required=True, help='ASPIRE run directories go here')
        command.add_argument('--backend', required=True,
                             help='kinematic: physics-free test double, test-only (results are not baseline results)')
        command.add_argument('--max-steps', type=int, default=MAX_STEPS,
                             help=f'control steps per trial (default {MAX_STEPS}: 5 minutes at 60 Hz)')
        command.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
        command.add_argument('--no-videos', action='store_true', help='skip per-camera MP4 capture')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return _command(args)
    except (ValueError, FileNotFoundError) as exc:  # invalid input (seed, task record, paths); harness errors propagate
        print(f'error: {exc}', file=sys.stderr)
        return 2


def _command(args):
    if args.command == 'checkout':
        prepared = prepare_checkout(args.source, destination=args.destination or DEFAULT_DESTINATION)
        print(json.dumps({'checkout': str(prepared.path), 'commit': prepared.commit,
                          'patch_sha256': prepared.patch_sha256}))
        return 0
    if args.backend not in BACKENDS:
        raise ValueError(f'backend {args.backend!r} is not available; choose one of {BACKENDS}')
    from ..task import AssemblyTask
    from .adapter import guard_seed
    from .runner import evaluate_frozen_program, run_program

    config = json.loads(args.config.read_text())
    validate(config)
    if config['baselines']['aspire']['upstream_commit'] != PINNED_COMMIT:
        raise ValueError(f'baselines.aspire.upstream_commit is not {PINNED_COMMIT}, the commit the patch is pinned to')
    if config['baselines']['aspire']['patch_sha256'] != hashlib.sha256(PATCH.read_bytes()).hexdigest():
        raise ValueError(f'baselines.aspire.patch_sha256 does not match {PATCH.name}')
    for seed in [args.seed] if args.command == 'run' else args.seeds:  # before any backend exists
        guard_seed(seed, args.mode if args.command == 'run' else 'evaluation', config)
    task = AssemblyTask.from_record(json.loads(args.task.read_text()))
    if args.backend == 'isaac':
        model, backend_factory, limits, isaac = isaac_rig(task, config)
    else:
        (model, backend_factory, limits), isaac = kinematic_rig(task), None
    print(f'warning: {TEST_ONLY_WARNING}; the program runs unsandboxed in this process', file=sys.stderr)
    common = dict(config=config, output_root=args.output_root, max_steps=args.max_steps, model=model, limits=limits,
                  capture_videos=not args.no_videos, test_only=True)
    code = 1
    try:
        if args.command == 'run':
            records = [run_program(args.program, task, args.seed, backend=backend_factory(), mode=args.mode,
                                   records_path=args.records, **common)]
        else:
            records = evaluate_frozen_program(args.program, args.seeds, task=task, backend_factory=backend_factory,
                                              records_path=args.records, **common)
        for record in records:
            print(json.dumps({'backend': args.backend, 'test_only': True, 'record': record.to_record()}), flush=True)
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
            except BaseException:  # reported and counted as a failure; the trial's own exception still propagates
                traceback.print_exc()
                code = 1
            isaac.shutdown(code)


if __name__ == '__main__':
    sys.exit(main())
