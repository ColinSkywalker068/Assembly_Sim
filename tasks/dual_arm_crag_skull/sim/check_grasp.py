"""Isaac Sim acceptance of one physical pick and place (TODO P2: a fragment is grasped, held and released by physics
alone, with no attach or teleport, and the cameras work). GPU.

Choice of fragment and arm: after the Isaac reset, fragments the settling left tipped (more than SETTLED_TURN_DEG)
are skipped; every remaining (fragment, arm) pair is screened on KinematicBackend (sim/pick.py through S0, CPU, no
physics), and the first pair S0 accepts end to end runs on IsaacAssemblyBackend. The S0 limits are TEST values
(configs/jing.json has none) and the YAM adapter plate and finger force are provisional.

Criteria (CRITERIA, judge), fixed before the first run on 2026-10-19 and tightened after it, where review found that
they could not fail: "closed on nothing" was width >= half the chord, which closing on air (the command, chord - 6 mm)
always meets, and slip was measured over the hold only. The first run's log passes the tightened ones (see
TODO.md):
- every segment is accepted by S0 and executed;
- grasped: the squeezed pads stop on the fragment, at least 3 mm short of the commanded width (closing on nothing
  reaches the command) and at most 2 mm wider than the chord;
- lifted at least 80% of LIFT_M, and relative to the TCP the fragment moves at most 2 mm / 2 deg over the 2 s hold
  and at every step from the end of the squeeze to the end of the carry;
- carried: after the carry it is within 1 cm of where the carry should take it;
- released: after the release, retreat and settle hold, it is back on the table (height within 5 mm of its start),
  within 1.5 cm of its place target, tilted at most 10 deg, and at rest (RestTolerance over the last window);
- no teleport or attach: between control steps it never moves more than 1 cm (catches drops and gross jumps), no
  physics joint holds it and it stays a dynamic rigid body;
- the other fragments and the anchor move at most 2 mm / 2 deg;
- the three cameras, sampled once during the hold, are valid and carry the current time.

Writes check_grasp.json (last), check_grasp_log.npz (per step) and camera PNGs; exits 1 on any failure.

  PYTHONPATH=. python -m tasks.dual_arm_crag_skull.sim.check_grasp --task runs/baseline_tasks/skull_101.json --output DIR
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

CRITERIA = {'blocked_min_m': 0.003, 'squeeze_over_m': 0.002, 'lift_fraction': 0.8, 'slip_m': 0.002,
            'slip_deg': 2.0, 'carry_m': 0.01, 'place_height_m': 0.005, 'place_xy_m': 0.015, 'tilt_deg': 10.0,
            'step_m': 0.01, 'others_m': 0.002, 'others_deg': 2.0}
SETTLED_TURN_DEG = 5.0
CARRY_M = 0.05
CARRY_CLEARANCE_M = 0.02
CAMERA_AFTER_S = 0.5
REPORT = 'check_grasp.json'
FAILED = 'check_grasp_failed.json'


def judge(report):
    """Failures of a report against CRITERIA (empty when it passes)."""
    c, failures = CRITERIA, []
    if not report['completed']:
        return [f"the pick and place did not complete: {report['refusal']}"]
    grasp = report['grasp']
    if not grasp['squeezed_width_m'] >= grasp['close_width_m'] + c['blocked_min_m']:
        failures.append(f"closed on nothing: the pads stopped at {grasp['squeezed_width_m'] * 1000:.1f} mm, less than "
                        f"{c['blocked_min_m'] * 1000:.0f} mm short of the {grasp['close_width_m'] * 1000:.1f} mm command")
    if not grasp['squeezed_width_m'] <= grasp['chord_m'] + c['squeeze_over_m']:
        failures.append(f"squeezed width {grasp['squeezed_width_m'] * 1000:.1f} mm is wider than the chord")
    if not report['lift_m'] >= c['lift_fraction'] * report['lift_target_m']:
        failures.append(f"lifted only {report['lift_m'] * 1000:.1f} mm of {report['lift_target_m'] * 1000:.0f} mm")
    if not (report['hold_slip_m'] <= c['slip_m'] and report['hold_slip_deg'] <= c['slip_deg']):
        failures.append(f"slipped {report['hold_slip_m'] * 1000:.2f} mm / {report['hold_slip_deg']:.2f} deg in the hold")
    if not (report['transport_slip_m'] <= c['slip_m'] and report['transport_slip_deg'] <= c['slip_deg']):
        failures.append(f"slipped {report['transport_slip_m'] * 1000:.2f} mm / {report['transport_slip_deg']:.2f} deg "
                        'relative to the TCP between the squeeze and the end of the carry')
    if not report['carry_error_m'] <= c['carry_m']:
        failures.append(f"carried {report['carry_error_m'] * 1000:.1f} mm away from the carry target")
    place = report['place']
    if not place['height_error_m'] <= c['place_height_m']:
        failures.append(f"not back on the table: {place['height_error_m'] * 1000:.1f} mm from its start height")
    if not place['xy_error_m'] <= c['place_xy_m']:
        failures.append(f"{place['xy_error_m'] * 1000:.1f} mm from its place target")
    if not place['tilt_deg'] <= c['tilt_deg']:
        failures.append(f"tilted {place['tilt_deg']:.1f} deg after the place")
    if not place['at_rest']:
        failures.append('not at rest after the settle hold')
    if not report['max_step_m'] <= c['step_m']:
        failures.append(f"jumped {report['max_step_m'] * 1000:.1f} mm in one control step")
    if report['attachments'] or not report['carried_dynamic']:
        failures.append(f"attached: joints {report['attachments']}, dynamic body {report['carried_dynamic']}")
    if not (report['others_moved_m'] <= c['others_m'] and report['others_turned_deg'] <= c['others_deg']):
        failures.append(f"other fragments moved {report['others_moved_m'] * 1000:.2f} mm / "
                        f"{report['others_turned_deg']:.2f} deg")
    if not (report['cameras']['valid'] and report['cameras']['time_matches']):
        failures.append(f"cameras: {report['cameras']}")
    return failures


def _footprint_radius(fragment):
    (x0, y0, _), (x1, y1, _) = fragment.local_bounds
    return 0.5 * math.hypot(x1 - x0, y1 - y0)


def free_carry(task, fragment_id, poses, *, distance_m=CARRY_M, clearance_m=CARRY_CLEARANCE_M):
    """An (x, y) carry of `distance_m` that keeps the fragment's footprint circle on the table, out of the
    exclusion zones and `clearance_m` from every other fragment's; ValueError if no axis direction does."""
    fragment = task.fragment(fragment_id)
    radius = _footprint_radius(fragment)
    centre = poses[fragment_id][:3, :3] @ ((np.asarray(fragment.local_bounds[0]) + fragment.local_bounds[1]) / 2)
    for direction in ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)):
        carry = np.asarray(direction) * distance_m
        moved = poses[fragment_id][:2, 3] + centre[:2] + carry
        x0, x1, y0, y1 = task.table_bounds_m
        if not (x0 + radius <= moved[0] <= x1 - radius and y0 + radius <= moved[1] <= y1 - radius):
            continue
        if any(z0 - radius <= moved[0] <= z1 + radius and w0 - radius <= moved[1] <= w1 + radius
               for z0, z1, w0, w1 in task.exclusions_m):
            continue
        clear = True
        for key in task.fragment_ids:
            if key == fragment_id:
                continue
            other = task.fragment(key)
            other_centre = poses[key][:3, :3] @ ((np.asarray(other.local_bounds[0]) + other.local_bounds[1]) / 2)
            gap = np.linalg.norm(moved - (poses[key][:2, 3] + other_centre[:2])) - radius - _footprint_radius(other)
            clear &= gap >= clearance_m
        if clear:
            return tuple(float(v) for v in carry)
    raise ValueError(f'{fragment_id}: no free carry of {distance_m} m along an axis')


def attachments(stage, fragment_path):
    """Paths of the physics joints with a body at or under `fragment_path`: a grasp held by friction has none."""
    from pxr import Sdf, UsdPhysics
    root = Sdf.Path(fragment_path)
    found = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            if any(target.HasPrefix(root)
                   for target in (*joint.GetBody0Rel().GetTargets(), *joint.GetBody1Rel().GetTargets())):
                found.append(str(prim.GetPath()))
    return found


def dynamic_body(stage, path):
    """Whether the prim at `path` is a rigid body physics moves: RigidBodyAPI, enabled, not kinematic."""
    from pxr import UsdPhysics
    prim = stage.GetPrimAtPath(path)
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return False
    body = UsdPhysics.RigidBodyAPI(prim)
    return bool(body.GetRigidBodyEnabledAttr().Get()) and not bool(body.GetKinematicEnabledAttr().Get())


def fragment_points(fragment, pose):
    from .pick import surface_points
    with np.load(fragment.mesh) as data:
        vertices, faces = np.asarray(data['v'], dtype=float), np.asarray(data['f'], dtype=int)
    return surface_points(vertices @ pose[:3, :3].T + pose[:3, 3], faces)


def screen(task, model, limits, home, *, tip_reach_m, fragment_ids=None, exclude=()):
    """Every (fragment, arm) pair run through the scripted pick and place on KinematicBackend: which S0 accepts end to
    end, and why the others were refused. Poses are the task's initial ones."""
    from ..baselines.backend import KinematicBackend
    from ..contracts.actions import ARMS
    from ..s0.adapter import Rejected
    from .pick import run_pick_and_place
    results = []
    for fragment_id in fragment_ids or task.moving_ids:
        if fragment_id in exclude:
            continue
        for arm in ARMS:
            backend = KinematicBackend(model, home_joints=home, width_range_m=limits.gripper_width_range_m,
                                       control_hz=limits.servo_hz)
            backend.reset(task)
            poses = backend.privileged_fragment_poses()
            result = {'fragment_id': fragment_id, 'arm': arm, 'accepted': False, 'refusal': None, 'carry_xy': None}
            try:
                result['carry_xy'] = free_carry(task, fragment_id, poses)
                record = run_pick_and_place(backend, model, limits, arm=arm, fragment_id=fragment_id,
                                            points=fragment_points(task.fragment(fragment_id), poses[fragment_id]),
                                            carry_xy=result['carry_xy'], tip_reach_m=tip_reach_m)
                result.update(accepted=True, chord_m=record.grasp.chord_m, yaw_deg=math.degrees(record.grasp.yaw_rad))
            except (Rejected, ValueError) as exc:
                result['refusal'] = f'{type(exc).__name__}: {exc}'
            results.append(result)
    return results


def s0_test_limits(task, backend):
    """S0 limits with TEST values (configs/jing.json has none), the YAM width range and the asset's velocities."""
    from ..s0.adapter import S0Limits
    from ..s0.kinematics import load_kinematics
    return S0Limits(max_joint_velocity_rad_s=np.asarray(load_kinematics()['max_joint_velocity_rad_s'], dtype=float),
                    max_joint_acceleration_rad_s2=np.full(6, 20.0), gripper_width_range_m=backend.width_range,
                    arm_arm_clearance_m=0.02, table_top_z_m=task.table_top_z_m, table_clearance_m=0.01,
                    tcp_min_height_m=0.02, servo_hz=_whole_hz(backend.control_hz), max_observation_age_s=0.5,
                    anchor_position_tolerance_m=0.002, anchor_rotation_tolerance_rad=0.005,
                    contact_table_clearance_m=0.002, contact_region_radius_m=0.12, contact_max_speed_m_s=0.05)


def _whole_hz(rate):
    if float(rate) != int(rate):
        raise ValueError(f'S0 needs a whole servo rate, the backend steps at {rate} Hz')
    return int(rate)


def _turn(first, second):
    from .backend import _turn_deg
    return _turn_deg(first, second)


def _tilt_deg(first, second):
    return float(np.degrees(np.arccos(np.clip(first[:3, 2] @ second[:3, 2], -1.0, 1.0))))


def measure(log, record, start, end, fragment_id, carry_xy, rest):
    """The report's measurements from the per-step log [(segment, time, world_T_fragment, world_T_tcp, width)]."""
    from .backend import at_rest
    from .pick import LIFT_M

    def last(name):
        return [entry for entry in log if entry[0] == name][-1]

    def tcp_relative(entry):
        return np.linalg.inv(entry[3]) @ entry[2]

    hold = [entry for entry in log if entry[0] == 'hold']
    relative = [tcp_relative(entry) for entry in (hold[0], hold[-1])]
    squeezed, carried_index = len(log) - 1 - [entry[0] for entry in reversed(log)].index('squeeze'), log.index(last('carry'))
    held = [tcp_relative(entry) for entry in log[squeezed:carried_index + 1]]
    initial = start[fragment_id]
    carried = last('carry')[2]
    target = initial[:2, 3] + np.asarray(carry_xy)
    settle = [entry for entry in log if entry[0] == 'settle']
    window = [entry for entry in settle if entry[1] >= settle[-1][1] - rest.window_s]
    still = (settle[-1][1] - settle[0][1] >= rest.window_s - 1e-9
             and all(at_rest({'f': window[0][2]}, {'f': entry[2]}, rest) for entry in window[1:]))
    positions = np.array([entry[2][:3, 3] for entry in log])
    others = [key for key in start if key != fragment_id]
    return {'grasp': {'chord_m': record.grasp.chord_m, 'yaw_deg': math.degrees(record.grasp.yaw_rad),
                      'squeezed_width_m': float(last('squeeze')[4]), 'open_width_m': record.grasp.open_width_m,
                      'close_width_m': record.grasp.close_width_m},
            'lift_target_m': LIFT_M, 'lift_m': float(last('hold')[2][2, 3] - initial[2, 3]),
            'hold_slip_m': float(np.linalg.norm(relative[1][:3, 3] - relative[0][:3, 3])),
            'hold_slip_deg': _turn(relative[0], relative[1]),
            'transport_slip_m': max(float(np.linalg.norm(pose[:3, 3] - held[0][:3, 3])) for pose in held),
            'transport_slip_deg': max(_turn(held[0], pose) for pose in held),
            'carry_error_m': float(np.linalg.norm(carried[:2, 3] - target)),
            'place': {'height_error_m': float(abs(end[fragment_id][2, 3] - initial[2, 3])),
                      'xy_error_m': float(np.linalg.norm(end[fragment_id][:2, 3] - target)),
                      'tilt_deg': _tilt_deg(initial, end[fragment_id]),
                      'at_rest': still},
            'max_step_m': float(np.linalg.norm(np.diff(positions, axis=0), axis=1).max()),
            'others_moved_m': max(float(np.linalg.norm(end[key][:3, 3] - start[key][:3, 3])) for key in others),
            'others_turned_deg': max(_turn(start[key], end[key]) for key in others)}


def run(task_path, output):
    from ..__main__ import DEFAULT_CONFIG
    from ..baselines.task import AssemblyTask, verify_task
    from ..contracts.actions import ARMS
    from ..s0.adapter import Rejected
    from ..s0.kinematics import yam_tool_geometry
    from .backend import IsaacAssemblyBackend
    from .pick import run_pick_and_place

    output.mkdir(parents=True, exist_ok=True)
    for stale in (REPORT, FAILED):
        (output / stale).unlink(missing_ok=True)
    config = json.loads(DEFAULT_CONFIG.read_text())
    task = verify_task(AssemblyTask.from_record(json.loads(Path(task_path).read_text())))
    backend, code = None, 1
    try:
        backend = IsaacAssemblyBackend(config)
        backend.reset(task)
        limits = s0_test_limits(task, backend)
        home = {arm: backend.robots[arm].home for arm in ARMS}
        fingers = yam_tool_geometry()[2]['parts']['fingers']['max'][0]
        reaches = {arm: float(fingers - backend.model.arms[arm].flange_T_tcp[0, 3]) for arm in ARMS}
        if abs(reaches['left'] - reaches['right']) > 1e-9:
            raise ValueError(f'the arms carry different YAM tool geometry: fingertip reaches {reaches}')
        reach = reaches['left']
        tipped = [key for key, turn in backend.start.turned_deg.items() if turn > SETTLED_TURN_DEG]
        screening = screen(task, backend.model, limits, home, tip_reach_m=reach, exclude=(task.anchor_id, *tipped))
        report = {'task_id': task.task_id, 'seed': task.seed, 's0_limits': 'TEST values (configs/jing.json has none)',
                  'gripper': 'I2RT linear_4310, PROVISIONAL adapter plate and finger drive', 'criteria': CRITERIA,
                  'start': backend.start.to_record(), 'skipped_tipped': tipped, 'screening': screening,
                  'completed': False, 'refusal': None}
        choice = next((result for result in screening if result['accepted']), None)
        if choice is None:
            report['refusal'] = 'S0 accepted no (fragment, arm) pair on the kinematic screen'
        else:
            fragment_id, arm = choice['fragment_id'], choice['arm']
            report['choice'] = {'fragment_id': fragment_id, 'arm': arm, 'carry_xy': choice['carry_xy']}
            start = backend.privileged_fragment_poses()
            observed = backend.observed_fragment_poses()[fragment_id].world_T_fragment
            log, cameras = [], {'valid': False, 'time_matches': False}

            def on_step(name, state):
                pose = backend.privileged_fragment_poses()[fragment_id]
                log.append((name, state.time_s, pose, state.tcp[arm], state.widths[arm]))
                held = [entry for entry in log if entry[0] == 'hold']
                if name == 'hold' and 'frames' not in cameras and state.time_s - held[0][1] >= CAMERA_AFTER_S:
                    frames = backend.camera_frames()
                    from PIL import Image
                    for camera, frame in frames.items():
                        Image.fromarray(frame.rgb).save(output / f'hold_{camera}.png')
                    cameras['frames'] = sorted(frames)
                    cameras['valid'] = len(frames) == 3 and all(
                        frame.rgb.max() > 4 and (np.isfinite(frame.depth_m) & (frame.depth_m > 0)).mean() >= 0.2
                        for frame in frames.values())
                    cameras['time_matches'] = all(frame.time_s == state.time_s for frame in frames.values())

            record = None
            try:
                record = run_pick_and_place(backend, backend.model, limits, arm=arm, fragment_id=fragment_id,
                                            points=fragment_points(task.fragment(fragment_id), observed),
                                            carry_xy=choice['carry_xy'], tip_reach_m=reach, on_step=on_step)
            except (Rejected, ValueError) as exc:  # S0 or the planner refused: the run did not complete
                report['refusal'] = f'{type(exc).__name__}: {exc}'
            if record is not None:  # outside the try: a bug in the measurements must not read as a refusal
                end = backend.privileged_fragment_poses()
                path = backend._fragments[fragment_id]
                report.update(completed=True, segments=[entry.to_record() for entry in record.segments],
                              attachments=attachments(backend.stage, path),
                              carried_dynamic=dynamic_body(backend.stage, path),
                              **measure(log, record, start, end, fragment_id, choice['carry_xy'], backend.rest))
            report['cameras'] = cameras
            np.savez(output / 'check_grasp_log.npz', segment=np.array([entry[0] for entry in log]),
                     time_s=np.array([entry[1] for entry in log]),
                     fragment=np.array([entry[2] for entry in log]), tcp=np.array([entry[3] for entry in log]),
                     width_m=np.array([entry[4] for entry in log]))
        report['failures'] = judge(report)
        report['pass'] = not report['failures']
        (output / REPORT).write_text(json.dumps(report, indent=1, default=float) + '\n')
        print('CHECK PASS' if report['pass'] else 'CHECK FAIL', flush=True)
        for failure in report['failures']:
            print('  ' + failure, flush=True)
        code = 0 if report['pass'] else 1
        return report
    except BaseException:
        (output / FAILED).write_text(json.dumps({'error': traceback.format_exc()}, indent=1) + '\n')
        traceback.print_exc()
        print('CHECK FAIL', flush=True)
        raise
    finally:
        if backend is None:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        backend.close()
        backend.shutdown(code)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.task, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
