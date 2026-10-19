"""CPU half of the Isaac AssemblyBackend (sim/backend.py): its timing, target and settling rules."""
import numpy as np
import pytest

from core.robots.kinematics import T
from core.robots.linear_4310 import load_linear_4310
from tasks.dual_arm_crag_skull.sim.backend import (RestTolerance, at_rest, camera_due, control_substeps,
                                                    finger_joint_target, interpolated_targets, measured_width)

GRIPPER = load_linear_4310()


def test_control_period_is_a_whole_number_of_physics_steps():
    assert control_substeps(240, 60) == 4
    assert control_substeps(120.0, 60.0) == 2
    for physics, control in ((240, 70), (240, 0), (0, 60), (float('nan'), 60), (60, 120)):
        with pytest.raises(ValueError, match='physics'):
            control_substeps(physics, control)


def test_targets_are_interpolated_linearly_over_the_period():
    previous, target = np.array([0.0, 1.0]), np.array([1.0, -1.0])
    steps = interpolated_targets(previous, target, 4)
    np.testing.assert_allclose(steps, [[0.25, 0.5], [0.5, 0.0], [0.75, -0.5], [1.0, -1.0]])
    np.testing.assert_array_equal(interpolated_targets(previous, target, 1), [target])


def test_finger_targets_follow_the_pad_gap():
    assert finger_joint_target(GRIPPER, 0.03) == pytest.approx(GRIPPER.joint_for_width(0.03))
    assert finger_joint_target(GRIPPER, 0.0) == pytest.approx(GRIPPER.contact_joint_m)
    with pytest.raises(ValueError, match='width'):
        finger_joint_target(GRIPPER, GRIPPER.max_width_m + 1e-4)


def test_measured_width_clips_solver_overshoot():
    """PhysX may hold a finger a hair past its limit; the width reads as the limit."""
    assert measured_width(GRIPPER, GRIPPER.travel_m + 1e-6, GRIPPER.travel_m) == pytest.approx(GRIPPER.max_width_m)
    assert measured_width(GRIPPER, -1e-6, 0.0) == 0.0
    assert measured_width(GRIPPER, 0.01, 0.02) == pytest.approx(GRIPPER.width_between(0.01, 0.02))
    with pytest.raises(ValueError, match='finite'):
        measured_width(GRIPPER, float('nan'), 0.01)


def test_rest_is_no_drift_over_the_window_in_both_senses():
    """Resting PhysX contacts jitter at about 1 mm/s with micrometre amplitude, so rest is judged by how far each
    body drifted over the last window, not by its instantaneous velocity."""
    tolerance = RestTolerance(drift_m=1e-4, drift_deg=0.05, window_s=0.5)
    before = {'a': T([0.1, 0.2, 0.75], [1, 0, 0, 0]), 'b': T([0.3, 0.0, 0.75], [1, 0, 0, 0])}
    jitter = {'a': T([0.1, 0.2, 0.75 + 5e-6], [1, 0, 0, 0]), 'b': before['b']}
    assert at_rest(before, jitter, tolerance)
    slid = {**before, 'b': T([0.3, 2e-4, 0.75], [1, 0, 0, 0])}
    assert not at_rest(before, slid, tolerance)
    half = np.radians(0.1) / 2
    turned = {**before, 'a': T([0.1, 0.2, 0.75], [np.cos(half), np.sin(half), 0, 0])}
    assert not at_rest(before, turned, tolerance)
    broken = {**before, 'a': np.full((4, 4), np.nan)}
    assert not at_rest(before, broken, tolerance)
    with pytest.raises(ValueError, match='same bodies'):
        at_rest(before, {'a': before['a']}, tolerance)
    with pytest.raises(ValueError, match='tolerance'):
        RestTolerance(drift_m=0.0)


def test_cameras_render_at_their_own_rate():
    assert camera_due(None, 0.0, 10)
    assert not camera_due(0.0, 0.05, 10)
    assert camera_due(0.0, 0.1, 10) and camera_due(0.0, 0.1 - 1e-12, 10)


# -- IsaacAssemblyBackend with fakes in place of Isaac --------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from tasks.dual_arm_crag_skull.baselines.backend import ProtectiveStop, StopMonitor, StopRule, checked_targets  # noqa: E402
from tasks.dual_arm_crag_skull.baselines.task import task_from_layout  # noqa: E402
from tasks.dual_arm_crag_skull.contracts.actions import ARMS  # noqa: E402
from tasks.dual_arm_crag_skull.s0.kinematics import yam_model, yam_width_range_m  # noqa: E402
from tasks.dual_arm_crag_skull.sim.backend import (IsaacAssemblyBackend, check_fixture, checked_start,  # noqa: E402
                                                    settled_start)
from tasks.dual_arm_crag_skull.tests.baselines_fixtures import write_layout  # noqa: E402

HOME = np.array([0.42959, -0.22024, -0.68657, -0.00021, -1.10447, -0.42928])


class FakeRobot:
    """Joints jump to their targets, like the asset's very stiff drives; records every action."""

    def __init__(self):
        self.q, self.actions, self.velocity_targets = np.concatenate([HOME, [GRIPPER.travel_m] * 2]), [], []

    def get_articulation_controller(self):
        return SimpleNamespace(apply_action=self._apply)

    def _apply(self, action):
        self.actions.append((np.array(action.joint_positions), np.array(action.joint_indices)))
        self.velocity_targets.append(np.array(action.joint_velocities))
        self.q[np.asarray(action.joint_indices)] = action.joint_positions

    def get_joint_positions(self, joint_indices):
        return self.q[np.asarray(joint_indices)].copy()

    def get_joint_velocities(self, joint_indices):
        return np.zeros(len(joint_indices))


class FakeWorld:
    """Steps physics only while playing, like SimulationContext.step(render=False); play() commits."""

    def __init__(self):
        self.playing, self.physics_steps = True, 0

    def is_playing(self):
        return self.playing

    def step(self, render=False):
        self.physics_steps += self.playing

    def play(self):
        self.playing = True


class JammedRobot(FakeRobot):
    """Its arm's joint 2 stays where it is, whatever it is told: a held part pressed against something fixed."""

    def _apply(self, action):
        stuck = self.q[1]
        super()._apply(action)
        self.q[1] = stuck


def bare_backend(robot=FakeRobot):
    backend = IsaacAssemblyBackend.__new__(IsaacAssemblyBackend)
    backend.gripper, backend.model = GRIPPER, yam_model()
    backend.width_range = {arm: yam_width_range_m(GRIPPER) for arm in ARMS}
    backend.physics_hz, backend.control_hz, backend.substeps = 240.0, 60.0, 4
    backend.world = FakeWorld()
    backend._action = lambda joint_positions, joint_velocities, joint_indices: SimpleNamespace(
        joint_positions=joint_positions, joint_velocities=joint_velocities, joint_indices=joint_indices)
    backend.robots = {arm: SimpleNamespace(robot=robot(), arm_indices=np.arange(6), finger_indices=np.array([6, 7]),
                                           home=HOME.copy()) for arm in ARMS}
    backend.stop_rule = StopRule(tracking_error_rad=0.01, periods=3, speed_factor=1.1)
    backend._stop = StopMonitor(backend.stop_rule, {arm: backend.model.arms[arm].max_velocity for arm in ARMS})
    backend.task, backend._steps = object(), 0
    backend._targets = {arm: HOME.copy() for arm in ARMS}
    return backend


def test_step_interpolates_the_arm_and_sets_the_fingers_once():
    backend = bare_backend()
    delta = np.array([0.04, -0.02, 0.0, 0.0, 0.0, 0.08])
    state = backend.step({'left': HOME + delta, 'right': HOME}, {'left': 0.03, 'right': GRIPPER.max_width_m})
    actions = backend.robots['left'].robot.actions
    assert len(actions) == 4 and backend.world.physics_steps == 4 and state.time_s == pytest.approx(4 / 240)
    for k, (values, indices) in enumerate(actions):
        np.testing.assert_allclose(values[:6], HOME + delta * (k + 1) / 4)
        assert len(indices) == (8 if k == 0 else 6)
    np.testing.assert_allclose(actions[0][0][6:], [GRIPPER.joint_for_width(0.03)] * 2)
    assert state.widths['left'] == pytest.approx(0.03) and state.widths['right'] == pytest.approx(GRIPPER.max_width_m)
    np.testing.assert_allclose(state.tcp['left'], backend.model.arms['left'].tcp(HOME + delta))


def test_a_joint_held_off_its_target_stops_the_robot_after_the_rules_periods():
    """configs/jing.json s0.protective_stop on the backend every method runs on: 20 mrad off target for three control
    periods ends the trial; the stop carries the periods that led to it."""
    backend = bare_backend(JammedRobot)
    target = {'left': HOME + np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0]), 'right': HOME}
    widths = {arm: GRIPPER.max_width_m for arm in ARMS}
    for _ in range(2):
        backend.step(target, widths)
    with pytest.raises(ProtectiveStop, match='left joint_2 lagged its target by 20.0 mrad for 3 control periods') as stop:
        backend.step(target, widths)
    assert len(stop.value.log) == 3
    free = bare_backend()
    for _ in range(10):
        free.step(target, widths)


def test_its_snapshots_say_the_velocities_are_each_period_s():
    """PhysX's velocity at a period's end is that period's commanded velocity (measured on Isaac 2026-10-19), so the
    snapshots declare one control period as its window, which S0 reads a path's first step with."""
    backend = bare_backend()
    state = backend.step({'left': HOME, 'right': HOME}, {arm: GRIPPER.max_width_m for arm in ARMS})
    assert state.velocity_window_s == pytest.approx(1 / 60) == backend.velocity_window_s


def test_step_feeds_the_period_velocity_forward_to_the_arm_drives():
    """The asset's arm drives have damping/stiffness = 0.1 s: with zero velocity targets the damping holds a moving
    arm back (measured: 50 mm behind a 1.5 rad/s move), and S0 then sees it still moving after the plan stopped."""
    backend = bare_backend()
    delta = np.array([0.04, -0.02, 0.0, 0.0, 0.0, 0.08])
    backend.step({'left': HOME + delta, 'right': HOME}, {'left': 0.03, 'right': GRIPPER.max_width_m})
    velocities = backend.robots['left'].robot.velocity_targets
    assert len(velocities) == 4
    for k, values in enumerate(velocities):
        np.testing.assert_allclose(values[:6], delta * 60)
        assert len(values) == (8 if k == 0 else 6) and not values[6:].any()
    backend.step({'left': HOME + delta, 'right': HOME}, {'left': 0.03, 'right': GRIPPER.max_width_m})
    np.testing.assert_allclose(backend.robots['left'].robot.velocity_targets[-1][:6], 0.0)


def test_step_refuses_while_the_simulation_is_not_playing():
    """After a render paused the timeline, physics would silently not step while the clock ran on."""
    backend = bare_backend()
    backend.world.playing = False
    with pytest.raises(RuntimeError, match='not playing'):
        backend.step({arm: HOME for arm in ARMS}, {arm: 0.05 for arm in ARMS})
    assert backend._steps == 0 and not backend.robots['left'].robot.actions


def test_resume_after_rendering_plays_the_world():
    backend = bare_backend()
    backend.world.playing = False
    backend._resume()
    assert backend.world.is_playing()


def test_targets_are_copied_not_shared_with_the_caller():
    backend = bare_backend()
    command = HOME + 0.01
    backend.step({'left': command, 'right': HOME}, {arm: 0.05 for arm in ARMS})
    command[0] += 0.5
    np.testing.assert_allclose(backend._targets['left'], HOME + 0.01)
    targets = checked_targets(backend.model, backend.width_range, {arm: HOME for arm in ARMS},
                              {arm: 0.05 for arm in ARMS})
    assert not targets['left'].flags.writeable and targets['left'] is not HOME


def start(at_rest=True):
    ids = ('a', 'b')
    poses = {key: T([0.1 * i, 0.0, 0.75], [1, 0, 0, 0]) for i, key in enumerate(ids)}
    moved = {**poses, 'b': T([0.1, 0.002, 0.75], [1, 0, 0, 0])}
    return settled_start(ids, moved, poses, 1.0, at_rest, np.zeros((2, 6)))


def test_a_settled_start_is_read_only_and_recordable():
    record = start()
    assert record.moved_m == {'a': 0.0, 'b': pytest.approx(0.002)} and record.turned_deg['b'] == 0.0
    assert not record.poses['a'].flags.writeable
    import json
    data = json.loads(json.dumps(record.to_record()))
    assert data['at_rest'] is True and data['settle_s'] == 1.0 and len(data['poses']['b']) == 4


def test_a_start_that_never_came_to_rest_is_refused():
    assert checked_start(start()).at_rest
    with pytest.raises(RuntimeError, match='did not come to rest'):
        checked_start(start(at_rest=False))


def test_the_fixture_holds_the_anchor_at_its_goal(tmp_path):
    task = task_from_layout(write_layout(tmp_path / 'layout'), anchor_id='frag_a')
    check_fixture(task)
    anchor = task.fragment('frag_a')
    moved = anchor.goal_world_T_fragment.copy()
    moved[0, 3] += 1e-3
    from dataclasses import replace
    shifted = replace(task, fragments=tuple(replace(f, initial_world_T_fragment=moved) if f.fragment_id == 'frag_a'
                                            else f for f in task.fragments))
    with pytest.raises(ValueError, match='fixture'):
        check_fixture(shifted)
