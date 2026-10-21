"""DexAgent's delivery on the shared hardware (baselines/dexagent/executor.py): labels to contact windows, the command
ramp at S0's servo rate, jaw targets to widths, and execution through S0 on the kinematic double."""
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.robots.linear_4310 import load_linear_4310
from tasks.dual_arm_crag_skull.baselines.backend import KinematicBackend, ProtectiveStop
from tasks.dual_arm_crag_skull.baselines.dexagent import executor
from tasks.dual_arm_crag_skull.baselines.task import task_from_layout
from tasks.dual_arm_crag_skull.contracts.actions import ARMS
from tasks.dual_arm_crag_skull.s0.ik import IKOptions, solve_ik
from tasks.dual_arm_crag_skull.s0.kinematics import yam_model, yam_width_range_m
from tasks.dual_arm_crag_skull.sim.pick import down_pose
from tasks.dual_arm_crag_skull.tests.test_s0_adapter import HOME, limits

GRIPPER = load_linear_4310()
MODEL = yam_model()
WIDTHS = {arm: yam_width_range_m(GRIPPER) for arm in ARMS}
CONTACT = dict(contact_table_clearance_m=0.002, contact_region_radius_m=0.12, contact_max_speed_m_s=0.05)
LIMITS = limits(gripper_width_range_m=WIDTHS, **CONTACT, finger_opening_margin_m=0.004)  # ASPIRE's TEST values
OBJECTS = {'frag_a_frag_a': 'frag_a', 'frag_b_frag_b': 'frag_b'}
HEIGHT = 0.12
GOALS = {'frag_a': [0.03, 0.0, 0.0], 'frag_b': [-0.03, 0.0, 0.0]}
STAGING = {'frag_a': [0.42, 0.25, 0.75], 'frag_b': [-0.42, 0.10, 0.75]}
OPEN = 0.0  # DexAgent's jaw target fully open


@pytest.fixture
def task(tmp_path):
    folder = tmp_path / 'layout'
    folder.mkdir()
    vertices = np.array([[x, y, z] for x in (-.02, .02) for y in (-.02, .02) for z in (0.0, HEIGHT)], np.float32)
    pieces = []
    for name in GOALS:
        np.savez(folder / f'{name}.npz', v=vertices, f=np.array([[0, 1, 3], [0, 3, 2]], np.int32))
        pieces.append({'name': name, 'mesh': f'{name}.npz', 'cells': [[int(GOALS[name][0] > 0), 0, 0]],
                       'local_bounds': [[-.02, -.02, 0.0], [.02, .02, HEIGHT]],
                       'goal_pose': {'position': GOALS[name], 'orientation_wxyz': [1.0, 0.0, 0.0, 0.0]},
                       'staging_pose': {'position': STAGING[name], 'orientation_wxyz': [1.0, 0.0, 0.0, 0.0]}})
    (folder / 'layout.json').write_text(json.dumps({'schema_version': 2, 'object_id': 'tall_boxes', 'pieces': pieces,
                                                    'staging': {'pad_bounds': [-.15, .15, -.15, .15],
                                                                'table_bounds': [-.97, .97, -.57, .57],
                                                                'table_top_z': 0.75,
                                                                'exclusions': [[-.95, -.61, -.17, .17],
                                                                               [.61, .95, -.17, .17]]},
                                                    'source': {'ownership_repair': {  # frag_b goes in along +x
                                                        'anchor': 'frag_a', 'plan_source': 'given',
                                                        'plan': [{'fragment': 'frag_b', 'exit_direction': [-1, 0, 0]}]}}}))
    return task_from_layout(folder / 'layout.json', anchor_id='frag_a')


def backend():
    return KinematicBackend(MODEL, home_joints={arm: HOME for arm in ARMS}, width_range_m=WIDTHS, control_hz=60)


def delivery(left, labels, fps=30.0, jaws=None):
    """A delivery whose left arm follows `left` (T, 6) and whose right arm stays home."""
    left = np.asarray(left, dtype=float)
    t = len(left)
    jaws = np.full((t, 2), OPEN) if jaws is None else np.asarray(jaws, dtype=float)
    return executor.Delivery(np.arange(t) / fps, {'left': left, 'right': np.repeat(HOME[None], t, axis=0)},
                             {'left': jaws, 'right': np.full((t, 2), OPEN)}, tuple(labels))


def run(task, deliv):
    sim = backend()
    sim.reset(task)
    commands = executor.servo_commands(deliv, GRIPPER, LIMITS.servo_hz, OBJECTS)
    return executor.execute(sim, task, MODEL, LIMITS, commands), sim


def ramp(a, b, seconds, fps=30.0):
    """A smooth (cosine) joint ramp from a to b over `seconds` at `fps`, both ends included."""
    u = (1 - np.cos(np.linspace(0, np.pi, int(round(seconds * fps)) + 1))) / 2
    return np.asarray(a)[None] + u[:, None] * (np.asarray(b) - np.asarray(a))[None]


def above(z):
    pose = down_pose(STAGING['frag_b'][:2], LIMITS.table_top_z_m + z, 0.0)
    solved = solve_ik(MODEL.arms['left'], pose, HOME, IKOptions(max_seed_jump_rad=3.5))
    assert solved.ok, solved.reason
    return solved.q


# -- labels and windows -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('label, expected', [
    ('left: carry frag_b_frag_b #3', {'left': 'carry frag_b_frag_b'}),
    ('left: grip frag_b_frag_b | right: lift frag_a_frag_a #4.2', {'left': 'grip frag_b_frag_b',
                                                                  'right': 'lift frag_a_frag_a'}),
    ('both: wait', {}), ('', {})])
def test_labels_split_into_each_arms_phase(label, expected):
    assert executor.split_label(label) == expected


@pytest.mark.parametrize('phase, expected', [
    ('descend on frag_b_frag_b', ('grasp', 'frag_b')), ('close on frag_b_frag_b', ('grasp', 'frag_b')),
    ('grip frag_b_frag_b', ('grasp', 'frag_b')), ('lift frag_b_frag_b', ('grasp', 'frag_b')),
    ('carry frag_b_frag_b (down)', ('place', 'frag_b')), ('set down frag_b_frag_b', ('place', 'frag_b')),
    ('let go of frag_b_frag_b', ('place', 'frag_b')), ('settle after release of frag_b_frag_b', ('place', 'frag_b')),
    ('retreat', None), ('return home', None), ('wait', None), ('grip frag_c_frag_c', None)])
def test_phases_map_to_grasp_and_place_contacts(phase, expected):
    assert executor.phase_contact(phase, OBJECTS) == expected


def test_a_retreat_keeps_the_arms_last_window():
    labels = ['left: reach for frag_b_frag_b'] * 3 + ['left: retreat'] * 3 + ['left: carry frag_b_frag_b'] * 3
    deliv = delivery(np.repeat(HOME[None], 9, axis=0), labels)
    commands = executor.servo_commands(deliv, GRIPPER, 60, OBJECTS)
    kinds = [c and c[0] for c in commands.contacts['left']]
    assert kinds[:9] == ['grasp'] * 9 and kinds[-1] == 'place' and set(commands.contacts['right']) == {None}


# -- commands -----------------------------------------------------------------------------------------------------------

def test_the_command_ramp_is_sampled_at_the_servo_rate():
    left = ramp(HOME, HOME + 0.1, 1.0)
    deliv = delivery(left, ['left: wait'] * len(left))
    commands = executor.servo_commands(deliv, GRIPPER, 60, OBJECTS)
    assert np.allclose(np.diff(commands.time_s), 1 / 60) and commands.time_s[0] == pytest.approx(1 / 60)
    assert commands.time_s[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(commands.joints['left'][0], (left[0] + left[1]) / 2)  # t = 1/60: between frames 0 and 1
    np.testing.assert_allclose(commands.joints['left'][1], left[1])  # t = 2/60: frame 1
    np.testing.assert_allclose(commands.joints['left'][-1], left[-1])
    np.testing.assert_array_equal(commands.start['left'], left[0])


@pytest.mark.parametrize('closing', [0.0, 0.01, 0.02])
def test_jaw_targets_become_the_pad_gap(closing):
    width = executor.jaw_width(GRIPPER, [closing, closing])
    assert width == pytest.approx(GRIPPER.width(GRIPPER.travel_m - closing))
    assert executor.jaw_width(GRIPPER, [0.0, 0.0]) == pytest.approx(GRIPPER.max_width_m)
    top = GRIPPER.travel_m - GRIPPER.contact_joint_m
    assert executor.jaw_width(GRIPPER, [top, top]) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        executor.jaw_width(GRIPPER, [top + 0.001, 0.0])


# -- execution ----------------------------------------------------------------------------------------------------------

def test_a_delivery_within_s0_runs_releases_settles_and_is_scored(task):
    left = np.vstack([ramp(HOME, HOME + 0.1, 1.0), ramp(HOME + 0.1, HOME, 1.0)[1:]])
    result, sim = run(task, delivery(left, ['left: wait'] * len(left)))
    assert result.status == 'completed' and result.steps == 120 and result.settled
    np.testing.assert_allclose(sim.robot_state().joints['left'], HOME)
    assert result.score is not None and result.score.success is False  # nothing was assembled


def test_a_start_within_a_held_arms_error_is_the_delivery_start(task):
    sim = backend()
    sim.reset(task)
    sim.joints['left'] = HOME + 2e-4  # Isaac holds the arms 1.2e-4 rad off home after its settle (measured)
    left = np.vstack([ramp(HOME, HOME + 0.1, 1.0), ramp(HOME + 0.1, HOME, 1.0)[1:]])
    commands = executor.servo_commands(delivery(left, ['left: wait'] * len(left)), GRIPPER, 60, OBJECTS)
    assert executor.execute(sim, task, MODEL, LIMITS, commands).status == 'completed'


def test_a_delivery_that_does_not_start_at_the_robot_is_refused(task):
    left = ramp(HOME + 0.01, HOME + 0.1, 1.0)
    result, sim = run(task, delivery(left, ['left: wait'] * len(left)))
    assert result.status == 'refused' and result.steps == 0 and sim.robot_state().time_s == 0.0


def test_s0_rejects_a_too_fast_delivery_before_moving_it(task):
    left = ramp(HOME, HOME + 1.5, 0.5)  # 1.5 rad in half a second: 4.7 rad/s at the peak
    result, sim = run(task, delivery(left, ['left: wait'] * len(left)))
    assert result.status == 'rejected' and result.steps == 0 and 'S0 rejected' in result.reason
    np.testing.assert_array_equal(sim.robot_state().joints['left'], HOME)  # held where it stood, then released
    assert result.settled and result.score is not None  # every trial ends released, settled and scored


def test_a_slow_grasp_descent_declares_its_window_on_the_fragment_on_entering_the_band(task):
    high, low = above(0.10), above(0.015)  # TCP 1.5 cm over the table: inside S0's band
    left = np.vstack([ramp(HOME, high, 3.0), ramp(high, low, 4.0)[1:], ramp(low, high, 4.0)[1:]])
    labels = (['left: reach for frag_b_frag_b'] * 91 + ['left: descend on frag_b_frag_b'] * 120
              + ['left: lift frag_b_frag_b'] * 120)
    result, sim = run(task, delivery(left, labels))
    assert result.status == 'completed', result.reason
    (time, arm, kind, fragment, centre), = result.windows
    assert (arm, kind, fragment) == ('left', 'grasp', 'frag_b') and time >= 3.0  # not before the descent
    np.testing.assert_allclose(centre, STAGING['frag_b'][:2])


def test_an_arm_that_stays_out_of_the_band_declares_nothing(task):
    high = above(0.10)
    left = np.vstack([ramp(HOME, high, 3.0), ramp(high, HOME, 3.0)[1:]])
    result, _ = run(task, delivery(left, ['left: descend on frag_b_frag_b'] * len(left)))
    assert result.status == 'completed' and result.windows == []


class Observed:
    def __init__(self, poses):
        self.poses = poses

    def observed_fragment_poses(self):
        from tasks.dual_arm_crag_skull.baselines.backend import ObservedPose
        return {key: ObservedPose(pose, 'oracle_sim', 0.0) for key, pose in self.poses.items()}


def test_a_place_window_is_on_the_goal_only_when_setting_the_fragment_down_there(task):
    goal = task.fragment('frag_b').goal_world_T_fragment
    near = goal.copy()
    near[:2, 3] += [0.05, 0.03]  # within S0's 12 cm of its goal
    window = executor.contact_window(Observed({'frag_b': near}), task, LIMITS, 'left', 'place', 'frag_b')
    assert window.kind == 'place' and window.center_xy == pytest.approx(tuple(goal[:2, 3]))
    far = task.fragment('frag_b').initial_world_T_fragment
    window = executor.contact_window(Observed({'frag_b': far}), task, LIMITS, 'left', 'place', 'frag_b')
    assert window.kind == 'grasp' and window.center_xy == pytest.approx(tuple(far[:2, 3]))  # set down where it is
    window = executor.contact_window(Observed({'frag_b': near}), task, LIMITS, 'left', 'grasp', 'frag_b')
    assert window.kind == 'grasp' and window.center_xy == pytest.approx(tuple(near[:2, 3]))


def test_a_descent_labelled_with_no_fragment_is_s0s_rejection(task):
    high, low = above(0.10), above(0.015)
    left = np.vstack([ramp(HOME, high, 3.0), ramp(high, low, 4.0)[1:]])
    result, _ = run(task, delivery(left, ['left: wait'] * len(left)))
    assert result.status == 'rejected' and result.windows == []


def test_a_fast_descent_into_the_band_is_rejected(task):
    high, low = above(0.10), above(0.015)
    left = np.vstack([ramp(HOME, high, 3.0), ramp(high, low, 0.6)[1:]])  # ~0.25 m/s at the bottom
    labels = ['left: descend on frag_b_frag_b'] * len(left)
    result, _ = run(task, delivery(left, labels))
    assert result.status == 'rejected' and 'm/s near the table' in result.reason


def test_a_width_outside_s0s_range_is_rejected(task):
    narrow = replace(LIMITS, gripper_width_range_m={arm: (0.0, 0.05) for arm in ARMS})  # the open jaws are wider
    left = ramp(HOME, HOME + 0.05, 1.0)
    sim = backend()
    sim.reset(task)
    commands = executor.servo_commands(delivery(left, ['left: wait'] * len(left)), GRIPPER, narrow.servo_hz, OBJECTS)
    result = executor.execute(sim, task, MODEL, narrow, commands)
    assert result.status == 'rejected' and result.steps == 0 and 'width outside' in result.reason


def test_every_execution_ends_with_both_grippers_fully_open(task):
    left = ramp(HOME, HOME + 0.05, 1.0)
    half = np.full((len(left), 2), (GRIPPER.travel_m - GRIPPER.contact_joint_m) / 2)
    result, sim = run(task, delivery(left, ['left: wait'] * len(left), jaws=half))
    assert result.status == 'completed' and result.settled
    assert sim.robot_state().widths == pytest.approx({arm: LIMITS.gripper_width_range_m[arm][1] for arm in ARMS})


def test_a_window_is_cleared_out_of_the_band_and_declared_anew_on_entering_it_again(task):
    high, low = above(0.10), above(0.015)
    down, up = ramp(high, low, 4.0)[1:], ramp(low, high, 4.0)[1:]
    left = np.vstack([ramp(HOME, high, 3.0), down, up, down, up])
    result, _ = run(task, delivery(left, ['left: descend on frag_b_frag_b'] * len(left)))
    assert result.status == 'completed', result.reason
    first, second = result.windows  # one per entry, each on the fragment where it is then observed
    assert first[1:4] == second[1:4] == ('left', 'grasp', 'frag_b') and second[0] - first[0] > 4.0


def test_a_tool_box_low_over_the_table_is_in_the_band_though_its_tcp_is_high():
    pose = down_pose(STAGING['frag_b'][:2], LIMITS.table_top_z_m + 0.04, 0.0)
    pose[:3, :3] = pose[:3, :3] @ Rotation.from_euler('y', 90, degrees=True).as_matrix()  # the jaws side by side
    solved = solve_ik(MODEL.arms['left'], pose, HOME, IKOptions(max_seed_jump_rad=3.5))
    assert solved.ok, solved.reason
    assert MODEL.arms['left'].tcp(solved.q)[2, 3] >= LIMITS.table_top_z_m + LIMITS.tcp_min_height_m
    assert executor.in_band(MODEL, LIMITS, 'left', solved.q)


def recorded(left_reference, left_state0):
    """read_delivery's input as DexAgent writes it: each frame's commanded reference and recorded state (here the same
    after frame 0)."""
    t = len(left_reference)
    names, reference = [], []
    for arm in ARMS:
        joints = left_reference if arm == 'left' else np.repeat(HOME[None], t, axis=0)
        for i in range(6):
            names.append(f'robot_{arm}_arm_dof_joint{i + 1}')
            reference.append(joints[:, i])
        for jaw in ('thumb', 'index'):
            names.append(f'robot_{arm}_{arm}_{jaw}_FE')
            reference.append(np.zeros(t))
    reference = np.column_stack(reference)
    qpos = reference.copy()
    qpos[0, :6] = left_state0
    return {'joint_names': np.array(names), 'reference': reference, 'qpos': qpos, 'time_s': np.arange(t) / 30.0,
            'labels': np.array(['left: wait'] * t)}


def test_a_delivery_starts_from_its_recorded_state_not_its_first_target(task):
    left = ramp(HOME, HOME + 0.1, 1.0)
    left[0] += 3e-5  # the runtime's first target is already on its way (DexAgent's reference at frame 0)
    sim = backend()
    sim.reset(task)
    commands = executor.servo_commands(executor.read_delivery(recorded(left, HOME)), GRIPPER, 60, OBJECTS)
    np.testing.assert_array_equal(commands.start['left'], HOME)
    assert executor.execute(sim, task, MODEL, LIMITS, commands).status == 'completed'
    elsewhere = executor.servo_commands(executor.read_delivery(recorded(left, HOME + 0.01)), GRIPPER, 60, OBJECTS)
    sim.reset(task)
    assert executor.execute(sim, task, MODEL, LIMITS, elsewhere).status == 'refused'  # planned from another start


# -- review 2026-10-19 ---------------------------------------------------------------------------------------------------

def cosine(fps):
    """Joint 1 of the left arm through A (1 - cos 2 pi t) for 2 s at `fps`: 7 rad/s^2 at its peaks (t = 0, 1, 2 s), so
    S0's knots see at most twice that on DexAgent's 30 fps ramp, and four times at a chunk that starts on a frame."""
    t = np.arange(0, round(2.0 * fps) + 1) / fps
    left = np.repeat(HOME[None], len(t), axis=0)
    left[:, 0] += 7.0 / (2 * np.pi) ** 2 * (1 - np.cos(2 * np.pi * t))
    return t, left


def burst(fps=60.0, a=15.0):
    """Rest, 15 rad/s^2 for 0.1 s from 0.95 s, -15 for 0.1 s, rest: across the 1 s chunk limit at 60 fps."""
    t = np.arange(0, round(2.0 * fps) + 1) / fps
    u = np.clip(t - 0.95, 0, None)
    up, down = np.minimum(u, 0.1), np.clip(u - 0.1, 0, 0.1)
    left = np.repeat(HOME[None], len(t), axis=0)
    left[:, 0] += 0.5 * a * up ** 2 + a * 0.1 * down - 0.5 * a * down ** 2
    return t, left


@pytest.mark.parametrize('make', [lambda: cosine(30.0), burst], ids=['dexagent_30fps', 'burst_60fps'])
def test_s0s_verdict_does_not_depend_on_where_the_executor_cuts_its_chunks(task, make):
    t, left = make()
    deliv = executor.Delivery(t, {'left': left, 'right': np.repeat(HOME[None], len(t), axis=0)},
                              {arm: np.zeros((len(t), 2)) for arm in ARMS}, tuple(['left: wait'] * len(t)))
    verdicts = []
    for chunk_s in (1.0, 1e9):  # S0 reads a chunk's first segment from the measured state: 2 (v1 - qd_now) / dt
        sim = backend()
        sim.reset(task)
        commands = executor.servo_commands(deliv, GRIPPER, 60, OBJECTS)
        verdicts.append(executor.execute(sim, task, MODEL, LIMITS, commands, chunk_s=chunk_s).status)
    assert verdicts == ['completed', 'completed']


@pytest.mark.parametrize('accel, status', [(1.4, 'completed'), (1.6, 'rejected')])
def test_a_steady_acceleration_across_chunk_boundaries_reads_as_it_does_in_one_piece(task, accel, status):
    """Joint 6 of the left arm accelerates at `accel` for 1.5 s and decelerates for 1.5 s at 60 fps, under a
    1.5 rad/s^2 limit: the first 1 s chunk can only end inside the constant acceleration, so the next starts from a
    velocity the backend reports as the last period's. S0 reads that first step as it reads every other one: 1.4
    passes (twice 1.4, the reading for an instantaneous velocity, would not) and 1.6 fails on its acceleration."""
    fps, a = 60.0, accel
    t = np.arange(0, round(3.5 * fps) + 1) / fps
    up, down = np.minimum(t, 1.5), np.clip(t - 1.5, 0, 1.5)
    left = np.repeat(HOME[None], len(t), axis=0)
    left[:, 5] += 0.5 * a * up ** 2 + a * 1.5 * down - 0.5 * a * down ** 2  # 3.15 rad at most, 2.1 rad/s at most
    deliv = executor.Delivery(t, {'left': left, 'right': np.repeat(HOME[None], len(t), axis=0)},
                              {arm: np.zeros((len(t), 2)) for arm in ARMS}, tuple(['left: wait'] * len(t)))
    slow = limits(gripper_width_range_m=WIDTHS, max_joint_acceleration_rad_s2=np.full(6, 1.5), **CONTACT)
    sim = backend()
    sim.reset(task)
    commands = executor.servo_commands(deliv, GRIPPER, 60, OBJECTS)
    result = executor.execute(sim, task, MODEL, slow, commands, chunk_s=1.0)
    assert result.status == status, result.reason
    if status == 'rejected':
        assert 'acceleration exceeded on joint_6' in result.reason


class Stopping(KinematicBackend):
    """The robot stops (ProtectiveStop) at its `at`-th step, as the Isaac backend's StopMonitor would stop it."""

    def __init__(self, at):
        super().__init__(MODEL, home_joints={arm: HOME for arm in ARMS}, width_range_m=WIDTHS, control_hz=60)
        self.at, self.count = at, 0

    def step(self, joints, widths):
        self.count += 1
        if self.count == self.at:
            raise ProtectiveStop('left joint_2 lagged its target by 20.0 mrad for 3 control periods',
                                 [{'time_s': self.time_s}])
        return super().step(joints, widths)


@pytest.mark.parametrize('extra', [-20, 10])  # inside the delivery; in the release-and-settle hold after it
def test_a_protective_stop_ends_the_execution_where_the_robot_stopped(task, extra):
    """No release, no settle, no score: the trial ends there as a failure, with the stop's log."""
    left = ramp(HOME, HOME + np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0]), 1.0)
    commands = executor.servo_commands(delivery(left, ['left: wait'] * len(left)), GRIPPER, 60, OBJECTS)
    at = len(commands.time_s) + extra
    sim = Stopping(at)
    sim.reset(task)
    result = executor.execute(sim, task, MODEL, LIMITS, commands)
    assert result.status == 'protective_stop' and result.score is None and not result.settled
    assert 'lagged its target' in result.reason and result.stop_log == [{'time_s': pytest.approx((at - 1) / 60)}]
    assert sim.count == at and result.steps == min(at - 1, len(commands.time_s))


class Carrying(KinematicBackend):
    """frag_b observed at `start_xy` until the carry starts at 3 s, then moving with it to its goal over `carry_s`."""

    def __init__(self, start_xy, carry_s):
        super().__init__(MODEL, home_joints={arm: HOME for arm in ARMS}, width_range_m=WIDTHS, control_hz=60)
        self.start_xy, self.carry_s = np.asarray(start_xy), carry_s

    def observed_fragment_poses(self):
        from tasks.dual_arm_crag_skull.baselines.backend import ObservedPose
        out = dict(super().observed_fragment_poses())
        goal = self.task.fragment('frag_b').goal_world_T_fragment.copy()
        u = float(np.clip((self.time_s - 3.0) / self.carry_s, 0, 1))
        goal[:2, 3] = (1 - u) * self.start_xy + u * goal[:2, 3]
        out['frag_b'] = ObservedPose(goal, 'oracle_sim', self.time_s)
        return out

    def reset(self, task):
        self.task = task
        return super().reset(task)


def test_a_set_down_after_a_carry_is_a_place_on_the_goal_declared_on_entering_the_band(task):
    goal = task.fragment('frag_b').goal_world_T_fragment[:2, 3]
    start = goal + np.array([-0.40, 0.0])

    def q(xy, z, seed):
        solved = solve_ik(MODEL.arms['left'], down_pose(xy, LIMITS.table_top_z_m + z, 0.0), seed,
                          IKOptions(max_seed_jump_rad=3.5))
        assert solved.ok, solved.reason
        return solved.q

    a = q(start, 0.05, HOME)
    b = q(goal, 0.021, a)  # carried 40 cm (3.0-4.75 s) to just above the band, set down: in the band at about 4.88 s
    c = q(goal, 0.015, b)
    reach, carry, down = ramp(HOME, a, 3.0), ramp(a, b, 1.75)[1:], ramp(b, c, 0.5)[1:]
    left = np.vstack([reach, carry, down, np.repeat(c[None], 30, axis=0)])
    labels = (['left: reach for frag_b_frag_b'] * len(reach) + ['left: carry frag_b_frag_b'] * len(carry)
              + ['left: set down frag_b_frag_b'] * (len(down) + 30))
    sim = Carrying(start, 1.75)
    sim.reset(task)
    commands = executor.servo_commands(delivery(left, labels), GRIPPER, 60, OBJECTS)
    entry = next(k for k in range(len(commands.time_s)) if executor.in_band(MODEL, LIMITS, 'left',
                                                                            commands.joints['left'][k]))
    result = executor.execute(sim, task, MODEL, LIMITS, commands)
    assert result.status == 'completed', result.reason
    (time, arm, kind, fragment, centre), = result.windows
    assert (arm, kind, fragment) == ('left', 'place', 'frag_b') and np.allclose(centre, goal)
    # just before its first target in the band (its chunk starts there or one command before, executor._chunks)
    assert commands.time_s[entry] - 2.5 / 60 <= time < commands.time_s[entry]


def test_a_rejected_chunk_declares_nothing_and_the_release_keeps_the_windows_it_had(task, monkeypatch):
    high, low = above(0.10), above(0.015)
    left = np.vstack([ramp(HOME, high, 3.0), ramp(high, low, 0.6)[1:]])  # enters the band too fast
    seen = {}
    real = executor.release_and_settle

    def spy(backend_, task_, model, limits, targets, contacts=None):
        seen['contacts'] = dict(contacts or {})
        return real(backend_, task_, model, limits, targets, contacts)

    monkeypatch.setattr(executor, 'release_and_settle', spy)
    result, _ = run(task, delivery(left, ['left: descend on frag_b_frag_b'] * len(left)))
    assert result.status == 'rejected' and result.windows == [] and seen['contacts'] == {}


def slowing_into_the_band(fps=30.0):
    """A descent at 0.3 rad/s that slows at the frame just before S0's band (the review's entry_accel.py): the first
    command in the band carries that frame's velocity change (17 rad/s^2 read inside a chunk at 30 fps), the one before
    it none."""
    high, low = above(0.10), above(0.015)

    def q(s):
        return high + s * (low - high)

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if executor.in_band(MODEL, LIMITS, 'left', q(mid)) else (mid, hi)
    joint = hi - (1 - hi) / 60 / 4
    fast = np.linspace(0, joint, int(np.ceil(joint / (0.3 / fps / np.abs(low - high).max()))) + 1)
    slow = np.linspace(joint, 1.0, 61)[1:]
    return np.vstack([ramp(HOME, high, 3.0, fps), [q(s) for s in np.concatenate([fast[1:], slow, np.ones(30)])]])


def test_a_chunk_entering_the_band_starts_where_the_velocity_changes_less(task):
    left = slowing_into_the_band()
    commands = executor.servo_commands(delivery(left, ['left: descend on frag_b_frag_b'] * len(left)), GRIPPER, 60,
                                       OBJECTS)
    entry, = executor.band_entries(MODEL, LIMITS, commands)
    change = executor.velocity_change(commands)
    assert change[entry] * 3600 > 10 and change[entry - 1] * 3600 < 1e-9  # S0 would read 2 x 17 from the entry
    result, _ = run(task, delivery(left, ['left: descend on frag_b_frag_b'] * len(left)))
    assert result.status == 'completed', result.reason
    (time, arm, kind, fragment, centre), = result.windows
    assert (arm, kind, fragment) == ('left', 'grasp', 'frag_b') and time < commands.time_s[entry]


def test_a_window_takes_the_label_of_the_arms_first_command_in_the_band(task):
    left = slowing_into_the_band(fps=60.0)  # at 60 fps each command is a frame with its own label
    probe = executor.servo_commands(executor.Delivery(np.arange(len(left)) / 60.0,
                                                      {'left': left, 'right': np.repeat(HOME[None], len(left), axis=0)},
                                                      {arm: np.zeros((len(left), 2)) for arm in ARMS},
                                                      tuple(['left: wait'] * len(left))), GRIPPER, 60, OBJECTS)
    entry, = executor.band_entries(MODEL, LIMITS, probe)
    change = executor.velocity_change(probe)
    assert change[entry - 1] < change[entry]  # its chunk starts one command early, still labelled 'wait' there
    labels = ['left: wait'] * (entry + 1) + ['left: descend on frag_b_frag_b'] * (len(left) - entry - 1)
    result, _ = run(task, delivery(left, labels, fps=60.0))
    assert result.status == 'completed', result.reason
    (time, arm, kind, fragment, centre), = result.windows
    assert (arm, kind, fragment) == ('left', 'grasp', 'frag_b')


def test_s0_boxes_the_fingers_at_the_jaws_opening(task):
    """With S0's boxes at the openings (the TEST limits since 2026-10-21), a delivery that closes the jaws high up may
    then tilt the gripper low over the table where one box over the whole stroke would not pass; with the jaws open
    the same descent is S0's rejection (the pose of tests/test_s0_opening.py, the TCP 25 mm over the table)."""
    from tasks.dual_arm_crag_skull.tests.test_s0_opening import Q_TILT, TILTED, tcp_joints
    q_up = tcp_joints([-0.42, 0.0, 0.75 + 0.085], TILTED, seed=Q_TILT)
    path = np.vstack([np.repeat(q_up[None], 31, axis=0), ramp(q_up, Q_TILT, 2.0)[1:]])
    narrow = GRIPPER.travel_m - (0.028 - GRIPPER.closed_gap_m) / 2  # the jaw target for a 28 mm gap
    closing = np.minimum(np.linspace(OPEN, 2 * narrow, 31), narrow)
    results = {}
    for name, jaw in (('closed', np.concatenate([closing, np.full(len(path) - 31, narrow)])),
                      ('open', np.full(len(path), OPEN))):
        sim = KinematicBackend(MODEL, home_joints={'left': q_up, 'right': HOME}, width_range_m=WIDTHS, control_hz=60)
        sim.reset(task)
        deliv = delivery(path, [''] * len(path), jaws=np.column_stack([jaw, jaw]))
        commands = executor.servo_commands(deliv, GRIPPER, LIMITS.servo_hz, OBJECTS)
        results[name] = executor.execute(sim, task, MODEL, LIMITS, commands)
    assert results['closed'].status == 'completed', results['closed'].reason
    assert results['open'].status == 'rejected' and 'tool box' in results['open'].reason


def test_the_executor_gives_s0_how_far_the_fingers_reach(task, monkeypatch):
    """S0 boxes the fingers by how far they reach (RobotSnapshot.spans): the widest opening the executor gives S0
    for a chunk starts from the measured span, not the gap."""
    sim = backend()
    sim.reset(task)
    sim.step({arm: HOME for arm in ARMS}, {arm: 0.02 for arm in ARMS})
    measured = sim.robot_state
    sim.robot_state = lambda: (lambda state: replace(state, spans={'left': max(state.widths['left'], 0.05),
                                                                   'right': state.widths['right']}))(measured())
    seen = []
    monkeypatch.setattr(executor, 'check_joint_path', lambda *args, **kwargs: seen.append(kwargs['openings']))
    narrow = GRIPPER.travel_m - (0.02 - GRIPPER.closed_gap_m) / 2  # the jaw target for a 20 mm gap
    deliv = delivery(np.repeat(HOME[None], 75, axis=0), [''] * 75, jaws=np.full((75, 2), narrow))  # 2.5 s
    executor.execute(sim, task, MODEL, LIMITS, executor.servo_commands(deliv, GRIPPER, LIMITS.servo_hz, OBJECTS))
    chunks = seen[:-1]  # the last is the ending, fully open
    assert len(chunks) >= 2  # after the first, the last command is the 20 mm one: only the span says 50 mm
    assert all(opening['left'] == pytest.approx(0.05) for opening in chunks)  # the right jaws are fully open


def test_the_executor_bounds_the_fingers_reach_by_what_s0_checked(task):
    """Once S0 accepts a chunk and before it is stepped, the executor tells the backend how far the fingers may reach
    (the protective stop's finger rule): the chunk's widest opening plus S0's margin; the ending opens them fully."""
    sim = backend()
    sim.reset(task)
    bounds, real = [], sim.bound_reach
    sim.bound_reach = lambda value: bounds.append(value) or real(value)
    narrow = GRIPPER.travel_m - (0.02 - GRIPPER.closed_gap_m) / 2  # the jaw target for a 20 mm gap
    deliv = delivery(np.repeat(HOME[None], 75, axis=0), [''] * 75, jaws=np.full((75, 2), narrow))  # 2.5 s
    commands = executor.servo_commands(deliv, GRIPPER, LIMITS.servo_hz, OBJECTS)
    result = executor.execute(sim, task, MODEL, LIMITS, commands)
    assert result.status == 'completed' and result.settled and len(bounds) >= 3
    margin = LIMITS.finger_opening_margin_m
    assert bounds[0]['left'] == pytest.approx(WIDTHS['left'][1] + margin)  # the jaws start open
    assert bounds[1]['left'] == pytest.approx(0.02 + margin)
    assert bounds[-1]['left'] == pytest.approx(WIDTHS['left'][1] + margin)
