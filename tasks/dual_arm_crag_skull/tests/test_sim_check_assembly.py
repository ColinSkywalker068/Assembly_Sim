"""The Isaac assembly check's CPU parts (sim/check_assembly.py): the telemetry of a run, the margins S0's values must
allow worked out from it, and the judgement."""
import json

import numpy as np
import pytest

from tasks.dual_arm_crag_skull.baselines.backend import KinematicBackend
from tasks.dual_arm_crag_skull.baselines.task import task_from_layout
from tasks.dual_arm_crag_skull.contracts.actions import ARMS
from tasks.dual_arm_crag_skull.s0.adapter import Rejected, check_geometry
from tasks.dual_arm_crag_skull.sim import assemble, check_assembly, pick
from tasks.dual_arm_crag_skull.tests.baselines_fixtures import write_layout
from tasks.dual_arm_crag_skull.tests.test_sim_assemble import HOME, LIMITS, MODEL, WIDTHS


@pytest.fixture(scope='module')
def cube_run(tmp_path_factory):
    task = task_from_layout(write_layout(tmp_path_factory.mktemp('layout')), anchor_id='frag_a')
    backend = KinematicBackend(MODEL, home_joints=HOME, width_range_m=WIDTHS, control_hz=60)
    backend.reset(task)
    telemetry, steps = check_assembly.Telemetry(), []
    step = backend.step
    backend.step = lambda targets, widths: steps.append(1) or step(targets, widths)
    result = assemble.run_assembly(backend, MODEL, LIMITS, task, assemble.load_layout(task), on_step=telemetry)
    return task, result, telemetry.arrays(), len(steps)


def test_the_telemetry_has_every_control_period_and_the_window_it_declared(cube_run):
    task, result, arrays, steps = cube_run
    assert result['error'] is None
    assert len(arrays['segment']) == steps and (np.diff(arrays['time_s']) > 0).all()
    for arm in ARMS:
        assert arrays[f'{arm}_target'].shape == arrays[f'{arm}_joints'].shape == (steps, 6)
    declared = arrays['contact_arm'] != ''
    assert set(arrays['segment'][declared]) <= {'descend', 'close', 'squeeze', 'lift', 'lower', 'steady', 'align',
                                                'insert', 'release', 'back', 'leave'}
    first = result['steps'][0].placement  # its pick is low enough that descending declares the grasp window
    descend = (arrays['segment'] == 'descend') & (np.arange(steps) < list(arrays['segment']).index('settle'))
    assert set(arrays['contact_arm'][descend]) == {first.arm}
    assert np.isnan(arrays['contact_xy'][~declared]).all() and np.isfinite(arrays['contact_xy'][declared]).all()


def test_the_telemetry_keeps_where_every_fragment_was(cube_run):
    task, *_ = cube_run
    backend = KinematicBackend(MODEL, home_joints=HOME, width_range_m=WIDTHS, control_hz=60)
    state = backend.reset(task)
    telemetry, poses = check_assembly.Telemetry(), backend.privileged_fragment_poses()
    for _ in range(2):
        telemetry('hold', state, state.joints, None, poses)
    arrays = telemetry.arrays()
    assert list(arrays['fragment_ids']) == sorted(poses) and arrays['fragments'].shape == (2, len(poses), 4, 4)
    for k, key in enumerate(sorted(poses)):
        np.testing.assert_allclose(arrays['fragments'][1, k], poses[key])


def test_the_margins_of_an_accepted_run_are_inside_the_limits_it_ran_under(cube_run):
    task, result, arrays, steps = cube_run
    margins = check_assembly.s0_margins(MODEL, LIMITS, arrays)
    commanded = margins['commanded']
    assert commanded['clearance_min_m'] >= LIMITS.arm_arm_clearance_m
    for arm in ARMS:
        used = commanded[arm]
        assert used['link_min_m'] >= LIMITS.table_clearance_m
        assert used['tool_min_m_normal'] >= LIMITS.table_clearance_m
        assert used['tcp_min_m_normal'] >= LIMITS.tcp_min_height_m
        assert np.max(used['acceleration_max_rad_s2']) <= 20.0 + 1e-6
    assert margins['measured']['left']['tracking_max_rad'] == pytest.approx(0.0, abs=1e-9)  # the kinematic backend


@pytest.mark.parametrize('low_first', [False, True])
def test_a_sample_below_the_normal_floor_counts_as_using_its_contact_window(low_first):
    """Two commanded samples of the right arm, the TCP 0.3 m and 5 mm over the table, 40 mm from the window centre,
    in either order: only the low one needs the window (S0 refuses it without one), and the margins say how it was
    used; the interval between them is next to it either way, so its finger speed counts."""
    arm_model = MODEL.arms['right']
    centre = np.array([0.35, 0.0])
    low = pick.down_pose(centre + [0.04, 0.0], LIMITS.table_top_z_m + 0.005, 0.0)
    high = assemble._translated(low, (0.0, 0.0, 0.295))
    q_high, _ = pick.solve_reachable(arm_model, high, HOME['right'])
    q_low, _ = pick.solve_reachable(arm_model, low, q_high, yaw_flip=False)
    times = np.array([0.0, 1.0 / LIMITS.servo_hz])
    order = (q_low, q_high) if low_first else (q_high, q_low)
    joints = {'left': np.repeat(HOME['left'][None], 2, axis=0), 'right': np.array(order)}
    with pytest.raises(Rejected, match='no contact declared'):
        check_geometry(MODEL, LIMITS, times, joints)
    arrays = {'segment': np.array(['descend', 'descend']), 'time_s': times,
              'contact_arm': np.array(['right', 'right']), 'contact_xy': np.array([centre, centre])}
    for side in ARMS:
        for key in ('target', 'joints'):
            arrays[f'{side}_{key}'] = joints[side]
        arrays[f'{side}_speed'] = np.zeros((2, 6))
        arrays[f'{side}_width'] = np.full(2, 0.05)
    used = check_assembly.s0_margins(MODEL, LIMITS, arrays)['commanded']['right']
    assert used['contact_samples'] == 1
    assert used['contact_tcp_min_m'] == pytest.approx(0.005, abs=1e-4)
    corners = [arm_model.finger_box_corners(q) for q in (q_high, q_low)]
    spread = np.hypot(*(corners[1][:, :2] - arm_model.tcp(q_low)[:2, 3]).T).max()  # corners[1] is the low sample's
    assert 0.04 - 1e-4 <= used['contact_reach_max_m'] <= 0.04 + spread + 1e-4  # the TCP, the finger box below the floor
    assert used['contact_finger_speed_max_m_s'] == pytest.approx(
        np.linalg.norm(corners[1] - corners[0], axis=1).max() * LIMITS.servo_hz)
    assert used['tcp_min_m_normal'] == pytest.approx(0.3, abs=1e-4)  # the sample that did not need it


def test_the_report_is_plain_json():
    """numpy scalars and arrays become numbers and lists, and a value that is not finite becomes null, so the report
    parses anywhere (a bare json.dumps would write NaN or Infinity, and default=float would turn a bool into 1.0)."""
    record = {'ok': np.bool_(True), 'n': np.int64(3), 'x': np.float64(np.nan), 'big': np.inf, 'v': np.array([1.5, 2.0]),
              'nested': [{'y': np.float32(0.25)}], 'text': 'a'}
    assert json.loads(json.dumps(check_assembly._jsonable(record), allow_nan=False)) == \
        {'ok': True, 'n': 3, 'x': None, 'big': None, 'v': [1.5, 2.0], 'nested': [{'y': 0.25}], 'text': 'a'}


def test_judge_names_every_failure():
    report = {'plan_steps': 2, 'tolerance': {'position_mm': 3.0, 'rotation_deg': 3.0}, 'ending_error': None,
              'score': {'success': False},
              'steps': [{'fragment_id': 'frag_b', 'error': None, 'drift_mm_deg': {'frag_b': [3.5, 0.1]}},
                        {'fragment_id': 'frag_c', 'error': 'Rejected: arm-arm clearance', 'drift_mm_deg': {}}]}
    failures = check_assembly.judge(report)
    assert any('frag_b 3.50 mm' in failure for failure in failures)
    assert any(failure.startswith('frag_c: Rejected') for failure in failures)
    assert '1 of 2 steps done' in failures and 'the assembly is not scored a success' in failures
