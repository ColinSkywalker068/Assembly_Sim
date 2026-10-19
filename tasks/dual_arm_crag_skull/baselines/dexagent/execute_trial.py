"""A shared-hardware DexAgent trial's delivery, executed and scored (README "Port to the shared hardware", step 5).

The trial's result.json (runner.score_trial) names the delivery DexAgent accepted, re-checked from a fresh port
checkout. Its commands run on an AssemblyBackend through S0 (executor.py) under the S0 limits the trial was prepared
for (inputs/s0_spec.json), on a backend of that rig (check_rig), and the RunRecord, with the Stage 3 trial's costs,
names that backend. Run it under Isaac Sim's Python for a result (`"$ISAAC_PYTHON" -m
tasks.dual_arm_crag_skull.baselines.dexagent execute --trial T --backend isaac`); the kinematic double is for tests
only. A delivery is executed once: `execution.started` is created before the backend is reset, and an execution that
fails is recorded as an error. A held-out seed needs a records file and is recorded once per variant.

DexAgent plans open loop from the start its own MuJoCo settled (the export manifest's start poses), and a backend
that settles the fragments itself (Isaac) starts from its own: the two are compared fragment by fragment, and a
delivery is not executed from a start more than START_GAP_M or START_GAP_DEG away from the one it was planned from
(seed 101: Isaac and MuJoCo settle every fragment to within 0.09 mm and 0.17 deg; one tips over by 18 deg in both).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from core.robots.linear_4310 import load_linear_4310

from ...contracts.actions import ARMS
from ..backend import KinematicBackend
from ..scoring import RunRecord, append_records, refuse_repeats
from ..task import AssemblyTask, verify_task
from .executor import Execution, execute, read_delivery, servo_commands
from .export import VARIANTS, load_manifest, pose_matrix
from .s0_check import contact_objects, limits_from_record

VARIANT = VARIANTS['crx_yam']
STATUS = {'completed': 'completed', 'rejected': 'rejected', 'refused': 'error', 'protective_stop': 'protective_stop'}
START_GAP_M, START_GAP_DEG = 1e-3, 1.0


def backend_label(backend):
    """A RunRecord's backend, named as ASPIRE's records name it (aspire/runner.backend_label)."""
    if isinstance(backend, KinematicBackend):
        return 'KinematicBackend (physics-free test double; its results are not baseline results)'
    return type(backend).__name__


def check_rig(limits, width_range, control_hz):
    """Refuse a backend other than the rig the trial's S0 limits were prepared for: its gripper width range and servo
    rate."""
    for arm in ARMS:
        if tuple(map(float, width_range[arm])) != tuple(map(float, limits.gripper_width_range_m[arm])):
            raise ValueError(f'the backend\'s {arm} gripper width range {tuple(width_range[arm])} is not the trial\'s '
                             f'{tuple(limits.gripper_width_range_m[arm])}')
    if float(control_hz) != float(limits.servo_hz):
        raise ValueError(f'the backend steps at {control_hz} Hz, not at the trial\'s servo rate {limits.servo_hz} Hz')


def start_gap(manifest, poses):
    """{fragment id: (metres, degrees)} from the start DexAgent planned from (the export manifest's settled rig poses,
    in world) to `poses` (world_T_fragment)."""
    world_T_rig = np.linalg.inv(np.asarray(manifest['rig_T_world'], dtype=float))
    gap = {}
    for obj in manifest['objects']:
        planned = world_T_rig @ pose_matrix(obj['start_rig_wxyz']) @ np.asarray(obj['body_T_fragment'], dtype=float)
        actual = np.asarray(poses[obj['fragment_id']], dtype=float)
        turn = np.arccos(np.clip((np.trace(actual[:3, :3] @ planned[:3, :3].T) - 1.0) / 2.0, -1.0, 1.0))
        gap[obj['fragment_id']] = (float(np.linalg.norm(actual[:3, 3] - planned[:3, 3])), float(np.degrees(turn)))
    return gap


def trial_limits(trial_dir):
    """The S0 limits a trial was prepared for (inputs/s0_spec.json)."""
    return limits_from_record(json.loads((Path(trial_dir) / 'inputs' / 's0_spec.json').read_text())['limits'])


def execute_trial(trial_dir, backend, model, *, test_only, records=None):
    """Execute a trial's delivery on `backend` (built for its task's layout) and return its RunRecord, also written
    to trial/execution.json and appended to `records`."""
    trial = Path(trial_dir).resolve()
    result = json.loads((trial / 'result.json').read_text())
    if not result.get('awaiting_execution'):
        raise ValueError(f'{trial} has no delivery awaiting execution on the shared hardware')
    if (trial / 'execution.json').exists():
        raise ValueError(f'{trial} was already executed (execution.json); a delivery is executed and recorded once')
    manifest = load_manifest(trial / 'inputs' / 'export_manifest.json')
    if manifest.get('variant') != VARIANT:
        raise ValueError(f'{trial} was not exported for the shared hardware ({VARIANT})')
    task = verify_task(AssemblyTask.from_record(manifest['task']))
    if task.split == 'heldout':
        if records is None:
            raise ValueError(f'seed {task.seed} is held out: its execution must be recorded (pass records)')
        refuse_repeats(records, 'dexagent', VARIANT, task.task_id, [task.seed])
    limits = trial_limits(trial)
    if float(task.table_top_z_m) != float(limits.table_top_z_m):
        raise ValueError(f'the trial\'s S0 limits are for a table at {limits.table_top_z_m} m, the task\'s is at '
                         f'{task.table_top_z_m} m')
    check_rig(limits, backend.width_range, backend.control_hz)
    delivery = read_delivery(Path(result['delivery']) / 'trajectory.npz')
    commands = servo_commands(delivery, load_linear_4310(), limits.servo_hz, contact_objects(task, manifest['objects']))
    label = backend_label(backend)
    try:  # from here on the delivery may move a robot: once only
        handle = os.open(trial / 'execution.started', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise ValueError(f'{trial} was already started (execution.started): a delivery is executed once') from None
    with os.fdopen(handle, 'w') as marker:
        marker.write(json.dumps({'backend': label, 'started_unix_s': time.time()}) + '\n')
    try:
        return _execute(trial, result, manifest, task, limits, commands, backend, model, label, test_only, records)
    except BaseException as exc:
        if records is not None:
            stage3 = result['record']
            append_records(records, [RunRecord(
                'dexagent', VARIANT, task.task_id, task.seed, task.split, 'error', None,
                cost_usd=stage3.get('cost_usd'), llm_calls=stage3.get('llm_calls'), wall_s=stage3.get('wall_s'),
                sim_trials=stage3.get('sim_trials'), notes=f'execution harness: {type(exc).__name__}: {str(exc)[:300]}',
                artifacts={**stage3.get('artifacts', {}), 'execution_started': str(trial / 'execution.started')},
                backend=label, test_only=bool(test_only or stage3.get('test_only')))])
        raise


def _execute(trial, result, manifest, task, limits, commands, backend, model, label, test_only, records):
    backend.reset(task)
    artifacts = {**result['record'].get('artifacts', {}), 'execution': str(trial / 'execution.json')}
    start, gap, far = getattr(backend, 'start', None), None, []
    if start is not None:  # a backend that settles the fragments first (Isaac); saved as ASPIRE's runs save it
        (trial / 'settled_start.json').write_text(json.dumps(start.to_record(), indent=1) + '\n')
        artifacts['settled_start'] = str(trial / 'settled_start.json')
        gap = start_gap(manifest, start.poses)
        far = [f'{key} {metres * 1000:.2f} mm / {degrees:.2f} deg' for key, (metres, degrees) in gap.items()
               if metres > START_GAP_M or degrees > START_GAP_DEG]
    if far:
        execution = Execution('refused', 0, f'the backend settled {", ".join(far)} away from the start DexAgent '
                                            f'planned from (at most {START_GAP_M * 1000:g} mm / {START_GAP_DEG:g} deg)')
    else:
        execution = execute(backend, task, model, limits, commands)
    final = {key: np.asarray(pose, dtype=float).tolist() for key, pose in backend.privileged_fragment_poses().items()}
    stage3 = result['record']
    notes = [stage3.get('notes') or '', f'executed on {label} through S0: {execution.status}'
             + (f' ({execution.reason})' if execution.reason else '')]
    if execution.settle_error:
        notes.append(execution.settle_error)
    record = RunRecord('dexagent', VARIANT, task.task_id, task.seed, task.split, STATUS[execution.status],
                       execution.score, cost_usd=stage3.get('cost_usd'), llm_calls=stage3.get('llm_calls'),
                       wall_s=stage3.get('wall_s'), sim_trials=stage3.get('sim_trials'), notes='; '.join(notes),
                       artifacts=artifacts,
                       backend=label, test_only=bool(test_only or stage3.get('test_only')))
    (trial / 'execution.json').write_text(json.dumps({
        'record': record.to_record(), 'status': execution.status, 'reason': execution.reason, 'steps': execution.steps,
        'settled': execution.settled, 'settle_error': execution.settle_error, 'start_gap': gap, 'final_poses': final,
        'protective_stop_log': execution.stop_log,
        'windows': [{'time_s': t, 'arm': arm, 'kind': kind, 'fragment': fragment, 'center_xy': list(centre)}
                    for t, arm, kind, fragment, centre in execution.windows or []]}, indent=1, default=str) + '\n')
    if records is not None:
        append_records(records, [record])
    return record
