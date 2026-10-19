"""The judging half of sim/check_backend.py (the Isaac half runs on a GPU)."""
import copy

import numpy as np
import pytest

from tasks.dual_arm_crag_skull.sim.check_backend import CAMERAS, CRITERIA, judge


def passing():
    reset = {'seed': 101, 'time_s': 0.0, 'settle_s': 1.0, 'at_rest': True, 'home_error_rad': 1e-5,
             'open_width_error_m': 1e-6, 'anchor_moved_m': 0.0, 'anchor_turned_deg': 0.0, 'start_matches_seed': True}
    frame = {'blank': False, 'valid_depth_fraction': 0.9, 'fx': 1649.0, 'time_s': 2.0, 'pose_error_m': 1e-5}
    return {'resets': [reset, {**reset, 'seed': 102}],
            'stepping': {'max_clock_error_s': 0.0, 'max_hold_error_rad': 1e-5},
            'gripper': {'target_m': 0.03, 'measured_m': {'left': 0.03, 'right': 0.0301}},
            'motion': {'final_error_rad': 1e-4},
            'fast_motion': {'max_tracking_error_rad': 0.002, 'speed_after_stop_rad_s': 0.001},
            'after_render': {'playing': True, 'moved_rad': 0.1, 'target_rad': 0.1, 'final_error_rad': 1e-4},
            'cameras': {'time_s': 2.0, 'frames': {name: dict(frame) for name in CAMERAS},
                        'render_clock_change_s': 0.0, 'render_fragment_change_m': 0.0, 'cached_within_period': True}}


def test_a_report_within_every_criterion_passes():
    assert judge(passing()) == []


@pytest.mark.parametrize('change, message', [
    (lambda r: r['resets'][0].update(home_error_rad=2 * CRITERIA['home_rad']), 'from home'),
    (lambda r: r['resets'][1].update(open_width_error_m=float('nan')), 'fully open'),
    (lambda r: r['resets'][0].update(anchor_moved_m=1e-5), 'anchor moved'),
    (lambda r: r['resets'][1].update(anchor_turned_deg=1e-3), 'anchor moved'),
    (lambda r: r['resets'][0].update(at_rest=False), 'not at rest'),
    (lambda r: r['resets'][0].update(time_s=0.004), 'clock at'),
    (lambda r: r['resets'][1].update(start_matches_seed=False), 'seed poses'),
    (lambda r: r['resets'].pop(), 'two resets'),
    (lambda r: r['stepping'].update(max_clock_error_s=1e-6), 'clock off'),
    (lambda r: r['stepping'].update(max_hold_error_rad=0.01), 'drifted'),
    (lambda r: r['gripper']['measured_m'].update(left=0.031), 'gripper left'),
    (lambda r: r['motion'].update(final_error_rad=0.01), 'behind the ramp'),
    (lambda r: r['cameras']['frames'].pop('agent'), 'cameras: got'),
    (lambda r: r['cameras']['frames']['agent'].update(blank=True), 'camera agent'),
    (lambda r: r['cameras']['frames']['left_wrist'].update(valid_depth_fraction=0.1), 'camera left_wrist'),
    (lambda r: r['cameras']['frames']['right_wrist'].update(time_s=1.9), 'sampled at'),
    (lambda r: r['cameras']['frames']['left_wrist'].update(pose_error_m=0.005), 'rendered 5.00 mm'),
    (lambda r: r['cameras'].update(render_clock_change_s=1 / 240), 'advanced the clock'),
    (lambda r: r['cameras'].update(render_fragment_change_m=1e-6), 'moved a fragment'),
    (lambda r: r['cameras'].update(cached_within_period=False), 'rendered again'),
    (lambda r: r['fast_motion'].update(max_tracking_error_rad=0.15), 'fast move'),
    (lambda r: r['fast_motion'].update(speed_after_stop_rad_s=0.5), 'still moving'),
    (lambda r: r['after_render'].update(playing=False), 'not playing after rendering'),
    (lambda r: r['after_render'].update(moved_rad=0.0, final_error_rad=0.1), 'after rendering the arm'),
    (lambda r: r['after_render'].update(final_error_rad=0.01), 'after rendering the arm'),
])
def test_any_criterion_missed_fails(change, message):
    report = copy.deepcopy(passing())
    change(report)
    failures = judge(report)
    assert failures and any(message in failure for failure in failures), failures


def test_fast_motion_reports_the_worst_tracking_error_over_the_whole_ramp_and_the_speed_after_it():
    """On a first-order lag the error peaks mid-ramp, where the target moves fastest, and is small at its end."""
    from types import SimpleNamespace

    from tasks.dual_arm_crag_skull.contracts.actions import ARMS
    from tasks.dual_arm_crag_skull.sim import check_backend

    class Lag:  # each step closes half of the remaining error, at 60 Hz
        def __init__(self, start):
            self.q = {arm: np.array(value, dtype=float) for arm, value in start.items()}
            self.errors = []

        def step(self, targets, widths):
            velocities = {}
            for arm in ARMS:
                change = 0.5 * (np.asarray(targets[arm], dtype=float) - self.q[arm])
                self.q[arm] = self.q[arm] + change
                velocities[arm] = change * 60
            self.errors.append(float(np.abs(self.q['left'] - targets['left']).max()))
            return SimpleNamespace(joints={arm: q.copy() for arm, q in self.q.items()}, joint_velocities=velocities)

    start = {arm: np.zeros(6) for arm in ARMS}
    lag = Lag(start)
    report, final = check_backend._fast_move(lag, start, {arm: 0.05 for arm in ARMS})
    ramp = lag.errors[:check_backend.FAST_STEPS]
    assert report['max_tracking_error_rad'] == pytest.approx(max(ramp)) and max(ramp) > 3 * ramp[-1]
    assert final['left'][4] == pytest.approx(check_backend.FAST_RAD)
    held = check_backend.STOP_STEPS
    # after the ramp the remaining error halves each held step; the last step's change is the speed reported
    assert report['speed_after_stop_rad_s'] == pytest.approx(ramp[-1] * 0.5 ** held * 60, rel=1e-6)
