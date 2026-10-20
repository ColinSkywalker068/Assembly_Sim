"""Does each partial assembly stay put under gravity in Isaac Sim, the engine every method is scored on?

For each set of fragments, a task whose fragments in that set start at their goals (the others at their staging poses
on the table, off the pad) is reset on the Isaac backend: the anchor held by the fixture, the robot at home, the free
fragments settled until at rest (at least --settle-s). A placed fragment's drift is its settled pose against its goal.
The sets are the prefixes of the task's assembly plan or, with --all-orders, of every order the layout's plan
directions allow (assembly_settle.plan_orders); each order is judged by its worst prefix against the task's tolerances.
assembly_settle.py does the same in MuJoCo on a DexAgent export.

  CUDA_VISIBLE_DEVICES=N OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. $ISAAC_PYTHON -B -m \\
      tasks.dual_arm_crag_skull.experiments.assembly_settle_isaac --task runs/baseline_tasks/skull_base.json \\
      --out OUT.json [--all-orders]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np


def prefix_sets(orders):
    """Every set of fragments that some prefix of `orders` has placed, once each, smallest first."""
    return sorted({tuple(sorted(order[:k])) for order in orders for k in range(1, len(order) + 1)},
                  key=lambda placed: (len(placed), placed))


def placed_task(task, placed):
    """`task` with the fragments `placed` starting at their goals and the others where the task starts them."""
    unknown = sorted(set(placed) - set(task.fragment_ids))
    if unknown:
        raise ValueError(f'not fragments of the task: {unknown}')
    return replace(task, fragments=tuple(replace(f, initial_world_T_fragment=f.goal_world_T_fragment)
                                         if f.fragment_id in placed else f for f in task.fragments))


def _drift(task, poses, placed, turn_deg):
    drift = {}
    for key in placed:
        goal, now = task.fragment(key).goal_world_T_fragment, np.asarray(poses[key])
        drift[key] = [1000 * float(np.linalg.norm(now[:3, 3] - goal[:3, 3])), float(turn_deg(goal, now))]
    return drift


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--all-orders', action='store_true')
    parser.add_argument('--settle-s', type=float, default=2.0)
    parser.add_argument('--max-settle-s', type=float, default=6.0)
    args = parser.parse_args(argv)
    from ..__main__ import DEFAULT_CONFIG
    from ..baselines.task import AssemblyTask, verify_task
    from ..sim.backend import IsaacAssemblyBackend, _turn_deg
    from .assembly_settle import plan_orders
    config = json.loads(DEFAULT_CONFIG.read_text())
    task = verify_task(AssemblyTask.from_record(json.loads(args.task.read_text())))
    orders = (plan_orders(task.layout, task.layout_sha256) if args.all_orders
              else [tuple(step.fragment_id for step in task.assembly_plan)])
    backend, code, rows = None, 1, {}
    try:
        backend = IsaacAssemblyBackend(config, cameras=False, start_settle_s=args.settle_s,
                                       max_settle_s=args.max_settle_s)
        for placed in prefix_sets(orders):
            error = None
            try:
                backend.reset(placed_task(task, placed))
            except RuntimeError as exc:  # not at rest within --max-settle-s: still read where everything is
                error = str(exc)
            rows[placed] = {'placed': list(placed), 'at_rest': error is None, 'error': error,
                            'drift_mm_deg': _drift(task, backend.privileged_fragment_poses(), placed, _turn_deg)}
            print(json.dumps(rows[placed]), flush=True)
        tolerance = (1000 * task.position_tolerance_m, task.rotation_tolerance_deg)
        judged = []
        for order in orders:
            steps = [rows[tuple(sorted(order[:k]))] for k in range(1, len(order) + 1)]
            mm = max(w[0] for step in steps for w in step['drift_mm_deg'].values())
            deg = max(w[1] for step in steps for w in step['drift_mm_deg'].values())
            judged.append({'order': list(order), 'worst_mm': mm, 'worst_deg': deg,
                           'all_at_rest': all(step['at_rest'] for step in steps),
                           'worst_over_tolerance': max(mm / tolerance[0], deg / tolerance[1])})
        report = {'task_id': task.task_id, 'layout': task.layout, 'layout_sha256': task.layout_sha256,
                  'orders_from': 'layout plan directions, every order' if args.all_orders else 'the task plan',
                  'settle_s': args.settle_s, 'max_settle_s': args.max_settle_s, 'physics_hz': backend.physics_hz,
                  'orders': sorted(judged, key=lambda row: (row['worst_over_tolerance'], row['order'])),
                  'sets': list(rows.values())}
        args.out.write_text(json.dumps(report, indent=1) + '\n')  # before shutdown: Kit ends the process there
        within = sum(row['worst_over_tolerance'] < 1 and row['all_at_rest'] for row in judged)
        print(f'{len(judged)} orders, {within} within the tolerances and at rest at every step; worst '
              f"{max(row['worst_mm'] for row in judged):.4f} mm {max(row['worst_deg'] for row in judged):.4f} deg",
              flush=True)
        code = 0
    except BaseException:
        traceback.print_exc()
        print('SETTLE FAIL', flush=True)
        raise
    finally:
        if backend is None:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        backend.close()
        backend.shutdown(code)


if __name__ == '__main__':
    main()
