"""Isaac Sim check of the scripted robot assembly (sim/assemble.py): the task's whole assembly plan carried out with the
dual CRX + YAM through S0, recorded, with the telemetry a final S0 parameter table needs. GPU. Our reference that the
plan can be done with this hardware, not a baseline. The S0 limits are TEST values (configs/jing.json has none), and the
YAM adapter plate and finger drive are provisional.

Criteria, fixed before the first run (CRITERIA, judge):
- every step is planned, and S0 accepts every segment (no rejection, no protective stop);
- after each step, every fragment placed so far is within the task's tolerances of its goal (position and rotation);
- after the ending every method gets (both grippers open, the task's settle time), the task is scored a success.

Telemetry, every control period (Telemetry): the segment, the contact window it declared, and both arms' commanded
targets, measured joints, joint speeds and widths. From it, with S0's own geometry, s0_margins works out what the run
used of each S0 value, on the commanded joints (what S0 checked) and the measured ones (what the arms did): the
arm-arm clearance, the heights over the table of the links, tool box and TCP, how often and how a contact window was
needed (fingers, TCP and body heights, reach from its centre, finger speed), the commanded accelerations, the joint
speeds and the tracking error. They are the margins the S0 values must allow for normal assembly.

--order ID,ID,... runs the plan's steps in another order the layout's straight insertions allow (reordered_plan), for
an order the robot may manage where the plan's fails; the report records the order and whether it is the plan's.
--no-video skips the cameras (about three fifths of a run's wall time) for quicker rounds; the telemetry, with every
fragment's pose each control period, and the report are the same. --collider-inset-mm X makes every fragment's
colliders X mm smaller on each face instead of the protocol's baselines.collider_inset_m (an experiment; recorded).

Writes DIR/assembly.json (last, before Kit shuts down), DIR/telemetry.npz, DIR/<camera>.mp4 and DIR/step_<k>.jpg (the
close-up after each step's settle; fragments painted in the colours they are named by); exits 1 on any failure.

  CUDA_VISIBLE_DEVICES=N OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. $ISAAC_PYTHON -B -m \\
      tasks.dual_arm_crag_skull.sim.check_assembly --task runs/baseline_tasks/skull_101.json --output DIR
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from ..contracts.actions import ARMS

REPORT, FAILED = 'assembly.json', 'assembly_failed.json'
RECORD_EVERY = 6  # control periods between frames: 10 Hz at S0's 60 Hz
CRITERIA = {'every_step': True, 'within_tolerance_after_each_step': True, 'final_success': True}


class Telemetry:
    """Every control period as it went: segment, time, the contact window declared for it (arm and centre, or '' and
    NaN), both arms' commanded targets, measured joints, joint speeds and widths, and, if given, every fragment's pose
    (world, the simulator's; `fragment_ids` orders them). The geometry S0 judges is worked out from these afterwards
    (s0_margins)."""

    def __init__(self):
        self.rows, self.fragment_ids = [], None

    def __call__(self, name, state, targets, contact=None, fragments=None):
        row = {'segment': name, 'time_s': float(state.time_s), 'contact_arm': '' if contact is None else contact.arm,
               'contact_xy': np.full(2, np.nan) if contact is None else np.asarray(contact.center_xy, dtype=float)}
        for arm in ARMS:
            row[f'{arm}_target'] = np.asarray(targets[arm], dtype=float)
            row[f'{arm}_joints'] = np.asarray(state.joints[arm], dtype=float)
            row[f'{arm}_speed'] = np.asarray(state.joint_velocities[arm], dtype=float)
            row[f'{arm}_width'] = float(state.widths[arm])
        if fragments is not None:
            self.fragment_ids = self.fragment_ids or tuple(sorted(fragments))
            row['fragments'] = np.array([fragments[key] for key in self.fragment_ids], dtype=float)
        self.rows.append(row)

    def arrays(self):
        out = {key: np.array([row[key] for row in self.rows]) for key in (self.rows[0] if self.rows else ())}
        if self.fragment_ids is not None:
            out['fragment_ids'] = np.array(self.fragment_ids)
        return out


def _number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _jsonable(value):
    """`value` as plain JSON has it: numpy scalars and arrays as numbers and lists, bools as bools, a number that is
    not finite as None (json.dumps would write NaN or Infinity, and default=float a bool as 1.0)."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _number(value)
    return value


def _geometry(model, limits, arrays, key):
    """Per sample, with S0's own rules, on the joints arrays[f'{arm}_{key}']: the arm-arm clearance and, for each
    arm, the heights over the table of its lowest checked link, tool box, TCP, finger box and body box, its finger-box
    corners, whether S0's normal tool-box or TCP-height rule fails there (so the sample needs its contact window) and
    then how far the TCP and the finger box below the floor reach from the declared window's centre."""
    from ..s0.adapter import TABLE_CHECKED_LINKS, _below, _table_problem
    from ..s0.collision import capsule_clearance, lowest_point_z
    table, count = limits.table_top_z_m, len(arrays['segment'])
    floor = table + limits.table_clearance_m
    clearance = np.empty(count)
    per = {arm: {name: np.full(count, np.nan) for name in ('link', 'tool', 'tcp', 'finger', 'body', 'reach')}
           for arm in ARMS}
    for arm in ARMS:
        per[arm]['corners'], per[arm]['contact'] = np.empty((count, 8, 3)), np.zeros(count, dtype=bool)
    for k in range(count):
        joints = {arm: np.asarray(arrays[f'{arm}_{key}'][k], dtype=float) for arm in ARMS}
        capsules = {arm: model.arms[arm].capsules(joints[arm]) for arm in ARMS}
        clearance[k] = capsule_clearance(capsules['left'].arrays(), capsules['right'].arrays())[0]
        for arm in ARMS:
            arm_model, q, row = model.arms[arm], joints[arm], per[arm]
            lowest = lowest_point_z(capsules[arm].arrays())
            row['link'][k] = min(z for name, z in zip(capsules[arm].names, lowest)
                                 if name.split(':')[1] in TABLE_CHECKED_LINKS) - table
            row['tool'][k] = arm_model.tool_box_corners(q)[:, 2].min() - table
            row['corners'][k] = arm_model.finger_box_corners(q)
            row['finger'][k] = row['corners'][k][:, 2].min() - table
            row['body'][k] = arm_model.body_box_corners(q)[:, 2].min() - table
            problem, tcp = _table_problem(arm_model, limits, q, float(arrays['time_s'][k]))
            row['tcp'][k] = tcp[2, 3] - table
            if problem is not None:
                row['contact'][k] = True
                if arrays['contact_arm'][k] == arm:
                    centre = arrays['contact_xy'][k]
                    points = np.vstack([tcp[:3, 3], _below(row['corners'][k], floor)])
                    row['reach'][k] = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1]).max()
    return clearance, per


def _least(values, segments, mask=None):
    """(The smallest of `values` where `mask` holds, the segment it is in); (None, None) when there is none."""
    mask = np.ones(len(values), dtype=bool) if mask is None else mask
    if not mask.any():
        return None, None
    index = np.flatnonzero(mask)[int(np.argmin(values[mask]))]
    return _number(values[index]), str(segments[index])


def s0_margins(model, limits, arrays):
    """What the run used of each S0 value, from the telemetry with S0's own geometry: on the commanded joints (the
    targets S0 checked, though each segment's first interval it took from the measured state the segment started
    from, so the commanded numbers there can differ from S0's a little) and on the measured joints (what the arms
    did, with their speeds and tracking). Heights are over the table. A sample uses its contact window where S0's
    normal rule fails; the finger speed there is over the servo intervals next to such samples, as S0 takes it."""
    segments, times = arrays['segment'], np.asarray(arrays['time_s'], dtype=float)
    out = {'limits': {'arm_arm_clearance_m': limits.arm_arm_clearance_m, 'table_clearance_m': limits.table_clearance_m,
                      'tcp_min_height_m': limits.tcp_min_height_m,
                      'contact_table_clearance_m': limits.contact_table_clearance_m,
                      'contact_region_radius_m': limits.contact_region_radius_m,
                      'contact_max_speed_m_s': limits.contact_max_speed_m_s,
                      'max_joint_acceleration_rad_s2': np.asarray(limits.max_joint_acceleration_rad_s2).tolist(),
                      'max_joint_velocity_rad_s': np.asarray(limits.max_joint_velocity_rad_s).tolist(),
                      'servo_hz': limits.servo_hz}}
    for mode, key in (('commanded', 'target'), ('measured', 'joints')):
        clearance, per = _geometry(model, limits, arrays, key)
        least, where = _least(clearance, segments)
        summary = {'clearance_min_m': least, 'clearance_min_segment': where}
        for arm in ARMS:
            row, used = per[arm], per[arm]['contact']
            speeds = np.linalg.norm(np.diff(row['corners'], axis=0), axis=2).max(axis=1) / np.diff(times)
            near = used[1:] | used[:-1]  # interval k runs from sample k to k + 1
            entry = {'contact_samples': int(used.sum())}
            for name, mask in (('link', None), ('tool', ~used), ('tcp', ~used)):
                label = name if mask is None else f'{name}'
                value, segment = _least(row[name], segments, mask)
                suffix = '' if mask is None else '_normal'
                entry[f'{label}_min_m{suffix}'], entry[f'{label}_min_segment{suffix}'] = value, segment
            for name in ('finger', 'tcp', 'body'):
                entry[f'contact_{name}_min_m'] = _least(row[name], segments, used)[0]
            reach = row['reach'][used & np.isfinite(row['reach'])]
            entry['contact_reach_max_m'] = _number(reach.max()) if len(reach) else None
            entry['contact_finger_speed_max_m_s'] = _number(speeds[near].max()) if near.any() else None
            command = np.asarray(arrays[f'{arm}_target'], dtype=float)
            if mode == 'commanded':
                acceleration = np.diff(command, n=2, axis=0) * float(limits.servo_hz) ** 2
                entry['acceleration_max_rad_s2'] = np.abs(acceleration).max(axis=0, initial=0.0).round(4).tolist()
            else:
                tracking = np.abs(np.asarray(arrays[f'{arm}_joints'], dtype=float) - command).max(axis=1)
                entry['tracking_max_rad'] = _number(tracking.max())
                entry['tracking_p99_rad'] = _number(np.percentile(tracking, 99))
                entry['tracking_max_segment'] = str(segments[int(np.argmax(tracking))])
                entry['speed_max_rad_s'] = np.abs(np.asarray(arrays[f'{arm}_speed'])).max(axis=0).round(4).tolist()
            summary[arm] = entry
        out[mode] = summary
    return out


def judge(report):
    failures = []
    for step in report['steps']:
        if step['error']:
            failures.append(f"{step['fragment_id']}: {step['error']}")
        for key, (mm, deg) in step['drift_mm_deg'].items():
            if mm > report['tolerance']['position_mm'] or deg > report['tolerance']['rotation_deg']:
                failures.append(f"{key} {mm:.2f} mm / {deg:.2f} deg off its goal after placing {step['fragment_id']}")
    if len(report['steps']) != report['plan_steps'] or any(step['error'] for step in report['steps']):
        failures.append(f"{sum(not step['error'] for step in report['steps'])} of {report['plan_steps']} steps done")
    if report['ending_error']:
        failures.append(report['ending_error'])
    if not report['score']['success']:
        failures.append('the assembly is not scored a success')
    return failures


def run(task_path, output, order=None, video=True, collider_inset_m=None):
    from ..__main__ import DEFAULT_CONFIG
    from ..baselines.task import AssemblyTask, verify_task
    from ..experiments.assembly_replay_isaac import Recorder, display_colours, paint
    from .assemble import load_layout, reordered_plan, run_assembly
    from .backend import IsaacAssemblyBackend
    from .check_grasp import s0_test_limits

    if output.exists():
        raise SystemExit(f'{output} exists; a check is never overwritten')
    config = json.loads(DEFAULT_CONFIG.read_text())
    task = verify_task(AssemblyTask.from_record(json.loads(Path(task_path).read_text())))
    layout = load_layout(task)
    plan = task.assembly_plan if order is None else reordered_plan(task, layout, order)  # before Kit starts
    backend, recorder, code, telemetry = None, None, 1, Telemetry()
    try:
        backend = IsaacAssemblyBackend(config, cameras=False, collider_inset_m=collider_inset_m)
        backend.reset(task)
        paint(backend.stage, display_colours(task))
        output.mkdir(parents=True)
        recorder = Recorder(backend, output, (1280, 960)) if video else None
        limits = s0_test_limits(task, backend)
        state = {'count': 0, 'last': None, 'steps': 0}

        def still():  # the close-up after a step's settle
            state['steps'] += 1
            latest = None if recorder is None else output / 'close' / f'{recorder.count - 1:05d}.jpg'
            if latest is not None and latest.exists():
                shutil.copyfile(latest, output / f"step_{state['steps']}.jpg")

        def on_step(name, snapshot, targets, contact=None):
            telemetry(name, snapshot, targets, contact, backend.privileged_fragment_poses())
            if state['last'] == 'settle' and name != 'settle':
                still()
            state['last'] = name
            state['count'] += 1
            if recorder is not None and state['count'] % RECORD_EVERY == 0:
                recorder.frame()

        if recorder is not None:
            recorder.frame()
        began = time.time()
        result = run_assembly(backend, backend.model, limits, task, layout, on_step=on_step, plan=plan)
        if recorder is not None:
            recorder.frame()
        if state['last'] == 'settle':  # the last step ended without a later segment
            still()
        if recorder is not None:
            recorder.encode(1000.0 / (RECORD_EVERY * 1000.0 / limits.servo_hz))
        arrays = telemetry.arrays()
        np.savez(output / 'telemetry.npz', **arrays)
        try:  # the report is written whatever happens here
            margins, margins_error = s0_margins(backend.model, limits, arrays), None
        except Exception:
            margins, margins_error = None, traceback.format_exc()
        report = {'task_id': task.task_id, 'seed': task.seed, 'criteria': CRITERIA,
                  'plan_steps': len(task.assembly_plan),
                  'order': [step.fragment_id for step in plan], 'order_is_plan': tuple(plan) == task.assembly_plan,
                  'tolerance': {'position_mm': 1000 * task.position_tolerance_m,
                                'rotation_deg': task.rotation_tolerance_deg},
                  's0_limits': 'TEST values (sim/check_grasp.s0_test_limits); YAM plate and finger drive provisional',
                  'start': backend.start.to_record(), 'steps': [step.to_record() for step in result['steps']],
                  'error': result['error'], 'ending_error': result['ending_error'],
                  'score': result['score'].to_record(), 'sim_s': float(backend.time_s), 'wall_s': time.time() - began,
                  'video': recorder is not None, 'frames': 0 if recorder is None else recorder.count,
                  'collider_inset_m': backend.collider_inset_m, 'collider_inset_is_protocol': collider_inset_m is None,
                  's0_margins': margins, 's0_margins_error': margins_error}
        report['failures'] = judge(report)
        report['pass'] = not report['failures']
        (output / REPORT).write_text(json.dumps(_jsonable(report), indent=1, allow_nan=False) + '\n')
        print('CHECK PASS' if report['pass'] else 'CHECK FAIL', flush=True)
        for failure in report['failures']:
            print('  ' + failure, flush=True)
        code = 0 if report['pass'] else 1
        return report
    except BaseException:
        if output.exists():
            (output / FAILED).write_text(json.dumps({'error': traceback.format_exc()}, indent=1) + '\n')
            if telemetry.rows and not (output / 'telemetry.npz').exists():
                np.savez(output / 'telemetry.npz', **telemetry.arrays())
        traceback.print_exc()
        print('CHECK FAIL', flush=True)
        raise
    finally:
        if backend is None:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        if recorder is not None:
            recorder.close()
        backend.close()
        backend.shutdown(code)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--order', type=lambda text: text.split(','), help="fragment ids, comma separated")
    parser.add_argument('--no-video', action='store_true', help='no cameras: quicker, the same telemetry and report')
    parser.add_argument('--collider-inset-mm', type=float, default=None,
                        help="an experiment: fragment colliders this much smaller on each face, not the protocol's")
    args = parser.parse_args(argv)
    inset = None if args.collider_inset_mm is None else args.collider_inset_mm / 1000
    run(args.task, args.output, args.order, video=not args.no_video, collider_inset_m=inset)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
