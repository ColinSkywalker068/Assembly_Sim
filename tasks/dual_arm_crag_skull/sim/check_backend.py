"""Isaac Sim check of IsaacAssemblyBackend (sim/backend.py) on the skull task (GPU). Not a grasp test.

Criteria, fixed before any run (CRITERIA, judge):
- reset: both arms at home and the grippers fully open; the anchor, held by the fixture, exactly at its initial
  pose; the free fragments settled to rest; the clock at 0.
- stepping: each step advances the clock by exactly one control period, and holding the home targets keeps the arms
  at home.
- gripper: a commanded width is reached on both arms.
- motion: a smooth joint ramp (left joint 1 by MOTION_RAD over MOTION_STEPS) is tracked.
- fast motion: a faster ramp (left joint 5 by FAST_RAD over FAST_STEPS, about 1.2 rad/s at its peak) is tracked
  within fast_track_rad at every step, and the arm stops when the commands stop (joint speeds at most stopped_rad_s
  STOP_STEPS later): a drive that lags (zero velocity targets against the asset's damping) fails both.
- cameras: agent, left_wrist and right_wrist frames are not blank, have valid depth, carry the current time, and
  were rendered from where the camera is now (the wrist ones from the measured flange pose, so a lagging render
  fails); rendering moves neither the clock nor any fragment; a second call within one camera period returns the
  cached frames.
- after rendering, the simulation is playing and a second ramp (right joint 1 by -MOTION_RAD) is tracked: physics
  still steps (a render that left the timeline paused would freeze the robot while the clock ran on).
- a second reset, on another seed, starts the fragments from that seed's poses with the anchor still fixed.

Writes check_backend.json (last) and camera PNGs; exits 1 on any failure.

  PYTHONPATH=. python -m tasks.dual_arm_crag_skull.sim.check_backend --task runs/baseline_tasks/skull_101.json \
    --second-seed 102 --output DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from ..contracts.actions import ARMS

CRITERIA = {'home_rad': 1e-3, 'width_m': 5e-4, 'anchor_m': 1e-6, 'anchor_deg': 1e-4, 'clock_s': 1e-9,
            'track_rad': 2e-3, 'fast_track_rad': 0.01, 'stopped_rad_s': 0.05, 'valid_depth_fraction': 0.2,
            'camera_pose_m': 1e-3, 'render_moves_m': 1e-9}
HOLD_STEPS = 60
GRIP_WIDTH_M = 0.03
GRIP_STEPS = 60
MOTION_RAD = 0.1
MOTION_STEPS = 60
FAST_RAD = 0.6
FAST_STEPS = 48
STOP_STEPS = 3
CAMERAS = ('agent', 'left_wrist', 'right_wrist')
REPORT = 'check_backend.json'
FAILED = 'check_backend_failed.json'


def judge(report):
    """Failures of a report against CRITERIA (empty when it passes)."""
    c, failures = CRITERIA, []
    for record in report['resets']:
        label = f"reset seed {record['seed']}"
        if not record['home_error_rad'] <= c['home_rad']:
            failures.append(f"{label}: arms {record['home_error_rad']:.2e} rad from home")
        if not record['open_width_error_m'] <= c['width_m']:
            failures.append(f"{label}: grippers {record['open_width_error_m']:.2e} m from fully open")
        if not (record['anchor_moved_m'] <= c['anchor_m'] and record['anchor_turned_deg'] <= c['anchor_deg']):
            failures.append(f"{label}: anchor moved {record['anchor_moved_m']:.2e} m / {record['anchor_turned_deg']:.2e} deg")
        if not record['at_rest']:
            failures.append(f"{label}: fragments not at rest after {record['settle_s']:.2f} s")
        if record['time_s'] != 0.0:
            failures.append(f"{label}: clock at {record['time_s']} s after reset")
        if not record['start_matches_seed']:
            failures.append(f'{label}: fragments did not start from the seed poses')
    if len(report['resets']) != 2:
        failures.append('expected two resets')
    stepping = report['stepping']
    if not stepping['max_clock_error_s'] <= c['clock_s']:
        failures.append(f"stepping: clock off by {stepping['max_clock_error_s']:.2e} s")
    if not stepping['max_hold_error_rad'] <= c['home_rad']:
        failures.append(f"stepping: arms drifted {stepping['max_hold_error_rad']:.2e} rad while held")
    for arm, width in report['gripper']['measured_m'].items():
        if not abs(width - report['gripper']['target_m']) <= c['width_m']:
            failures.append(f'gripper {arm}: width {width:.5f} m for {report["gripper"]["target_m"]} m')
    if not report['motion']['final_error_rad'] <= c['track_rad']:
        failures.append(f"motion: {report['motion']['final_error_rad']:.2e} rad behind the ramp")
    fast = report['fast_motion']
    if not fast['max_tracking_error_rad'] <= c['fast_track_rad']:
        failures.append(f"fast move: the arm was {fast['max_tracking_error_rad']:.3f} rad off its commands")
    if not fast['speed_after_stop_rad_s'] <= c['stopped_rad_s']:
        failures.append(f"fast move: still moving at {fast['speed_after_stop_rad_s']:.3f} rad/s after the commands "
                        'stopped')
    cameras = report['cameras']
    if set(cameras['frames']) != set(CAMERAS):
        failures.append(f"cameras: got {sorted(cameras['frames'])}")
    for name, frame in cameras['frames'].items():
        if frame['blank'] or not frame['valid_depth_fraction'] >= c['valid_depth_fraction']:
            failures.append(f"camera {name}: blank or {frame['valid_depth_fraction']:.2f} valid depth")
        if frame['time_s'] != cameras['time_s']:
            failures.append(f"camera {name}: sampled at {frame['time_s']} s, not {cameras['time_s']} s")
        if not frame['pose_error_m'] <= c['camera_pose_m']:
            failures.append(f"camera {name}: rendered {frame['pose_error_m'] * 1000:.2f} mm from where it is")
    if not (cameras['render_clock_change_s'] == 0.0 and cameras['render_fragment_change_m'] <= c['render_moves_m']):
        failures.append('cameras: rendering advanced the clock or moved a fragment')
    if not cameras['cached_within_period']:
        failures.append('cameras: a second call within one camera period rendered again')
    after = report['after_render']
    if not after['playing']:
        failures.append('the simulation is not playing after rendering')
    if not (after['moved_rad'] >= 0.9 * after['target_rad'] and after['final_error_rad'] <= c['track_rad']):
        failures.append(f"after rendering the arm moved {after['moved_rad']:.4f} of {after['target_rad']} rad "
                        f"(error {after['final_error_rad']:.2e} rad)")
    return failures


def _ramp(backend, start, arm, joint, amount, widths):
    """Raised-cosine ramp of one joint over MOTION_STEPS, then 10 held steps; returns the last state and target."""
    for k in range(1, MOTION_STEPS + 1):
        target = {key: value.copy() for key, value in start.items()}
        target[arm][joint] += amount * (1 - np.cos(np.pi * k / MOTION_STEPS)) / 2
        state = backend.step(target, widths)
    for _ in range(10):
        state = backend.step(target, widths)
    return state, target


def _fast_move(backend, start, widths):
    """Left joint 5 by FAST_RAD over FAST_STEPS (raised cosine) and back to rest: the worst tracking error at any step
    and the joint speeds STOP_STEPS after the commands stop."""
    worst = 0.0
    for k in range(1, FAST_STEPS + 1):
        target = {key: value.copy() for key, value in start.items()}
        target['left'][4] += FAST_RAD * (1 - np.cos(np.pi * k / FAST_STEPS)) / 2
        state = backend.step(target, widths)
        worst = max(worst, float(np.abs(state.joints['left'] - target['left']).max()))
    for _ in range(STOP_STEPS):
        state = backend.step(target, widths)
    return {'target_rad': FAST_RAD, 'steps': FAST_STEPS, 'max_tracking_error_rad': worst,
            'speed_after_stop_rad_s': float(max(np.abs(state.joint_velocities[arm]).max() for arm in ARMS))}, target


def _hold(backend, home, widths, steps):
    errors, clock = [], []
    start = backend.time_s
    for k in range(1, steps + 1):
        state = backend.step(home, widths)
        clock.append(abs(state.time_s - (start + k / backend.control_hz)))
        errors.append(max(float(np.abs(state.joints[arm] - home[arm]).max()) for arm in home))
    return max(clock), max(errors)


def _reset_record(backend, task, home):
    from .backend import _turn_deg
    state = backend.reset(task)
    start = backend.start
    anchor = task.anchor_id
    initial = task.fragment(anchor).initial_world_T_fragment
    free = [key for key in task.fragment_ids if key != anchor]
    return {'seed': task.seed, 'time_s': state.time_s, 'settle_s': start.settle_s, 'at_rest': start.at_rest,
            'home_error_rad': max(float(np.abs(state.joints[arm] - home[arm]).max()) for arm in home),
            'open_width_error_m': max(abs(state.widths[arm] - backend.gripper.max_width_m) for arm in home),
            'anchor_moved_m': float(np.linalg.norm(start.poses[anchor][:3, 3] - initial[:3, 3])),
            'anchor_turned_deg': _turn_deg(initial, start.poses[anchor]),
            # A free fragment may settle, but it starts where the seed put it (well under the seeds' 3 cm shifts).
            'start_matches_seed': all(start.moved_m[key] < 0.02 for key in free),
            'moved_m': dict(start.moved_m), 'turned_deg': dict(start.turned_deg), 'speed_m_s': dict(start.speed_m_s),
            'spin_rad_s': dict(start.spin_rad_s)}


def _camera_records(backend, output):
    from PIL import Image
    from pxr import Usd, UsdGeom
    before_time, before_poses = backend.time_s, backend.privileged_fragment_poses()
    frames = backend.camera_frames()
    after_poses = backend.privileged_fragment_poses()
    cached = backend.camera_frames() is frames
    state = backend.robot_state()
    records = {}
    for name, frame in frames.items():
        prim = backend.stage.GetPrimAtPath(backend.cameras.path(name))
        if name.endswith('_wrist'):
            arm = name.split('_')[0]
            local = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default()), dtype=float).T
            world_T_usd = backend.model.arms[arm].frames(state.joints[arm])[0]['flange'] @ local
        else:
            world_T_usd = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                                     dtype=float).T
        expected = world_T_usd @ np.diag([1.0, -1.0, -1.0, 1.0])
        depth = frame.depth_m
        Image.fromarray(frame.rgb).save(output / f'{name}.png')
        records[name] = {'blank': bool(frame.rgb.max() <= 4),
                         'valid_depth_fraction': float((np.isfinite(depth) & (depth > 0)).mean()),
                         'fx': float(frame.intrinsics[0, 0]), 'time_s': frame.time_s,
                         'pose_error_m': float(np.linalg.norm(frame.world_T_camera[:3, 3] - expected[:3, 3]))}
    change = max(float(np.abs(after_poses[key] - before_poses[key]).max()) for key in before_poses)
    return {'time_s': before_time, 'frames': records, 'render_clock_change_s': backend.time_s - before_time,
            'render_fragment_change_m': change, 'cached_within_period': cached}


def run(task_path, second_seed, output, max_settle_s=3.0):
    from ..__main__ import DEFAULT_CONFIG
    from ..baselines.task import AssemblyTask, verify_task
    from ..contracts.actions import ARMS
    from .backend import IsaacAssemblyBackend

    output.mkdir(parents=True, exist_ok=True)
    for stale in (REPORT, FAILED):
        (output / stale).unlink(missing_ok=True)
    config = json.loads(DEFAULT_CONFIG.read_text())
    task = verify_task(AssemblyTask.from_record(json.loads(Path(task_path).read_text())))
    later = second_task(task, second_seed, config)
    backend, code = None, 1
    try:
        backend = IsaacAssemblyBackend(config, max_settle_s=max_settle_s)
        home = {arm: backend.robots[arm].home for arm in ARMS}
        report = {'task_id': task.task_id, 'gripper': 'I2RT linear_4310 on the PROVISIONAL adapter plate',
                  'rates_hz': {'physics': backend.physics_hz, 'control': backend.control_hz, 'camera': backend.camera_hz},
                  'criteria': CRITERIA, 'resets': [_reset_record(backend, task, home)]}
        open_widths = {arm: backend.gripper.max_width_m for arm in ARMS}
        clock, hold = _hold(backend, home, open_widths, HOLD_STEPS)
        report['stepping'] = {'steps': HOLD_STEPS, 'max_clock_error_s': clock, 'max_hold_error_rad': hold}
        grip = {arm: GRIP_WIDTH_M for arm in ARMS}
        for _ in range(GRIP_STEPS):
            state = backend.step(home, grip)
        report['gripper'] = {'target_m': GRIP_WIDTH_M, 'measured_m': dict(state.widths)}
        state, ramp = _ramp(backend, home, 'left', 0, MOTION_RAD, open_widths)
        report['motion'] = {'target_rad': MOTION_RAD,
                            'final_error_rad': float(np.abs(state.joints['left'] - ramp['left']).max())}
        report['fast_motion'], ramp = _fast_move(backend, ramp, open_widths)
        report['cameras'] = _camera_records(backend, output)
        before = backend.robot_state().joints['right'][0]
        state, again = _ramp(backend, ramp, 'right', 0, -MOTION_RAD, open_widths)
        report['after_render'] = {'playing': bool(backend.world.is_playing()), 'target_rad': MOTION_RAD,
                                  'moved_rad': float(abs(state.joints['right'][0] - before)),
                                  'final_error_rad': float(np.abs(state.joints['right'] - again['right']).max())}
        report['resets'].append(_reset_record(backend, later, home))
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


def second_task(task, seed, config):
    """The same layout under another seed's initial condition (verified like the first)."""
    from ..baselines.task import configured_initial_condition, task_from_layout, verify_task
    return verify_task(configured_initial_condition(task_from_layout(task.layout, anchor_id=task.anchor_id), seed,
                                                    config))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--second-seed', type=int, default=102)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-settle-s', type=float, default=3.0)
    args = parser.parse_args(argv)
    run(args.task, args.second_seed, args.output, args.max_settle_s)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
