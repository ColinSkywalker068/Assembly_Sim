"""ASPIRE baseline integration: the patch, adapter contracts, S0 rejections, IK, the seed guard and whole trials.

Every trial here runs the hand-written reference program on the physics-free KinematicBackend over a synthetic
two-box layout; nothing in this file is a baseline result, and the held-out seeds used below draw initial conditions
of that synthetic layout only, not of the skull task.
"""
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('cap_harness')

from cap_harness.api import CapApi  # noqa: E402
from cap_harness.artifacts import RunRecorder  # noqa: E402
from cap_harness.contracts import ArmCommand, MotionStrategy, Pose, RobotAction, Trajectory  # noqa: E402
from cap_harness.errors import ApiError, ErrorCode  # noqa: E402
from cap_harness.geometry import matrix_to_pose  # noqa: E402
from cap_harness.protocols import EnvironmentAdapter, IKProvider  # noqa: E402

from tasks.dual_arm_crag_skull.__main__ import DEFAULT_CONFIG  # noqa: E402
from tasks.dual_arm_crag_skull.baselines.aspire import checkout  # noqa: E402
from tasks.dual_arm_crag_skull.baselines.aspire.__main__ import main, parse_seeds  # noqa: E402
from tasks.dual_arm_crag_skull.baselines.aspire.adapter import (BASE_FRAME, EMBODIMENT,  # noqa: E402
                                                                SkullAssemblyAdapter)
from tasks.dual_arm_crag_skull.baselines.aspire.ik import IK_SOLVER, CrxIKProvider  # noqa: E402
from tasks.dual_arm_crag_skull.baselines.aspire.runner import (evaluate_frozen_program, installed_harness,  # noqa: E402
                                                               run_program)
from tasks.dual_arm_crag_skull.baselines.backend import KinematicBackend, ObservedPose, ProtectiveStop  # noqa: E402
from tasks.dual_arm_crag_skull.baselines.task import task_from_layout  # noqa: E402
from tasks.dual_arm_crag_skull.contracts.actions import ARMS  # noqa: E402
from tasks.dual_arm_crag_skull.s0.ik import IK_FAILURES  # noqa: E402
from tasks.dual_arm_crag_skull.s0.adapter import Rejected  # noqa: E402
from tasks.dual_arm_crag_skull.s0.kinematics import ROBOTIQ_PROXY_MAX_WIDTH_M  # noqa: E402
from tasks.dual_arm_crag_skull.tests.test_s0_adapter import HOME, MODEL, limits  # noqa: E402
from tasks.dual_arm_crag_skull.workspace import THIRD_PARTY  # noqa: E402

ASPIRE_SOURCE = Path(os.environ.get('ASPIRE_SOURCE', THIRD_PARTY / 'ASPIRE'))
needs_source = pytest.mark.skipif(not (ASPIRE_SOURCE / '.git').is_dir(), reason='reference ASPIRE clone is absent')
REFERENCE = Path(checkout.__file__).with_name('reference_program.py')
CONFIG = json.loads(DEFAULT_CONFIG.read_text())
# The S0 checks themselves, without the hold after a refusal (s0.rejected_command_hold): a refused command moves nothing
# and no time passes.
NO_HOLD = {**CONFIG, 's0': {**CONFIG['s0'], 'rejected_command_hold': {'periods': 0}}}
WIDTHS = {arm: (0.0, ROBOTIQ_PROXY_MAX_WIDTH_M) for arm in ARMS}
# A top-down grasp at the box centre must keep the tool box 1 cm above the table (S0 has no contact phase), which a
# 4 cm cube's 2 cm centre height cannot; 12 cm boxes put the centre at 6 cm.
HEIGHT = 0.12
GOALS = {'frag_a': [0.03, 0.0, 0.0], 'frag_b': [-0.03, 0.0, 0.0]}
STAGING = {'frag_a': [0.42, 0.25, 0.75], 'frag_b': [-0.42, 0.10, 0.75]}
NUDGE = np.array([0.0, 0.02, -0.02, 0.0, 0.02, 0.02])
SHOULDER = np.array([0.0, 0.0225, 0.0, 0.0, 0.0, 0.0])  # one period: 0.45 rad/s, from rest 0.45 / 0.05 = 9 rad/s^2


def write_layout(directory):
    directory.mkdir(parents=True)
    vertices = np.array([[x, y, z] for x in (-.02, .02) for y in (-.02, .02) for z in (0.0, HEIGHT)], np.float32)
    pieces = []
    for name in GOALS:
        np.savez(directory / f'{name}.npz', v=vertices, f=np.array([[0, 1, 3], [0, 3, 2]], np.int32))
        pieces.append({'name': name, 'mesh': f'{name}.npz', 'cells': [[int(GOALS[name][0] > 0), 0, 0]],
                       'local_bounds': [[-.02, -.02, 0.0], [.02, .02, HEIGHT]],
                       'goal_pose': {'position': GOALS[name], 'orientation_wxyz': [1.0, 0.0, 0.0, 0.0]},
                       'staging_pose': {'position': STAGING[name], 'orientation_wxyz': [1.0, 0.0, 0.0, 0.0]}})
    path = directory / 'layout.json'
    path.write_text(json.dumps({'schema_version': 2, 'object_id': 'tall_boxes', 'pieces': pieces, 'staging': {
        'pad_bounds': [-.15, .15, -.15, .15], 'table_bounds': [-.97, .97, -.57, .57], 'table_top_z': 0.75,
        'exclusions': [[-.95, -.61, -.17, .17], [.61, .95, -.17, .17]]},
        'source': {'ownership_repair': {'anchor': 'frag_a', 'plan_source': 'given',  # frag_b goes in along +x
                                        'plan': [{'fragment': 'frag_b', 'exit_direction': [-1, 0, 0]}]}}}))
    return path


@pytest.fixture
def task(tmp_path):
    return task_from_layout(write_layout(tmp_path / 'layout'), anchor_id='frag_a')


def backend(cls=KinematicBackend, widths=WIDTHS):
    return cls(MODEL, home_joints={arm: HOME for arm in ARMS}, width_range_m=widths, control_hz=20)


def adapter(sim=None, mode='practice', widths=WIDTHS, config=CONFIG, **kwargs):
    return SkullAssemblyAdapter(sim or backend(widths=widths), MODEL, limits(gripper_width_range_m=widths),
                                config=config, mode=mode, **kwargs)


def api_for(env, recorder=None):
    return CapApi(env, ik_providers={IK_SOLVER: CrxIKProvider(MODEL)}, default_camera='agent',
                  default_motion_strategy=MotionStrategy(ik_solver=IK_SOLVER), recorder=recorder)


def command(target, gripper=None, arm='left'):
    return RobotAction({arm: ArmCommand('joint_position', target, gripper, embodiment=EMBODIMENT)})


def trial(task, tmp_path, program, seed=101, mode='practice', max_steps=500, sim=None, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / 'program.py'
    path.write_text(program if isinstance(program, str) else program.read_text())
    return run_program(path, task, seed, config=CONFIG, backend=sim or backend(), output_root=tmp_path / 'runs',
                       mode=mode, max_steps=max_steps, model=MODEL, limits=limits(), **kwargs)


def program_result(record):
    return json.loads((Path(record.artifacts['run_dir']) / 'source/program-result.json').read_text())['result']


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True, check=True).stdout


def source_state():
    return (git('-C', str(ASPIRE_SOURCE), '--no-optional-locks', 'status', '--porcelain', '--ignored'),
            git('-C', str(ASPIRE_SOURCE), 'rev-parse', 'HEAD'))


@needs_source
def test_patch_applies_at_the_pinned_commit_and_the_checkout_differs_by_exactly_the_patch(tmp_path):
    before = source_state()
    plain = tmp_path / 'plain'
    git('clone', '--quiet', '--shared', '--no-checkout', str(ASPIRE_SOURCE), str(plain))
    git('-C', str(plain), '-c', 'advice.detachedHead=false', 'checkout', '--quiet', '--detach', checkout.PINNED_COMMIT)
    git('-C', str(plain), 'apply', '--check', str(checkout.PATCH))
    prepared = checkout.prepare_checkout(ASPIRE_SOURCE, destination=tmp_path / 'patched')
    assert prepared.commit == checkout.PINNED_COMMIT
    assert prepared.patch_sha256 == hashlib.sha256(checkout.PATCH.read_bytes()).hexdigest()
    diff = git('-C', str(prepared.path), 'diff', *checkout.DIFF_OPTIONS, 'HEAD', '--')
    assert diff == checkout.patch_body(checkout.PATCH) and '"fanuc_crx": 6' in diff and '"crx_dls"' in diff
    assert source_state() == before
    contracts = prepared.path / 'src/cap_harness/contracts.py'
    contracts.write_text(contracts.read_text() + '\n')
    with pytest.raises(ValueError, match='exactly'):
        checkout.verify_checkout(prepared.path)
    git('-C', str(prepared.path), 'checkout', '--', '.')
    with pytest.raises(ValueError, match='exactly'):
        checkout.verify_checkout(prepared.path)
    git('-C', str(prepared.path), 'apply', str(checkout.PATCH))
    (prepared.path / 'stray.py').write_text('')
    with pytest.raises(ValueError, match='untracked'):
        checkout.verify_checkout(prepared.path)
    (prepared.path / 'stray.py').unlink()
    bytecode = prepared.path / 'src/cap_harness/__pycache__'
    bytecode.mkdir()
    (bytecode / 'contracts.cpython-312.pyc').write_bytes(b'')
    with pytest.raises(ValueError, match='ignored files'):
        checkout.verify_checkout(prepared.path)


@needs_source
def test_checkout_refuses_unsafe_destinations_before_cloning(tmp_path):
    occupied = tmp_path / 'occupied'
    occupied.mkdir()
    (occupied / 'keep.txt').write_text('mine')
    with pytest.raises(ValueError, match='not an empty directory'):
        checkout.prepare_checkout(ASPIRE_SOURCE, destination=occupied)
    assert [p.name for p in occupied.iterdir()] == ['keep.txt']
    nested = ASPIRE_SOURCE / 'nested-checkout-must-not-exist'
    with pytest.raises(ValueError, match='inside the source working tree'):
        checkout.prepare_checkout(ASPIRE_SOURCE, destination=nested)
    assert not nested.exists()
    with pytest.raises(ValueError, match='has no commit'):
        checkout.prepare_checkout(ASPIRE_SOURCE, commit='0' * 40, destination=tmp_path / 'fresh')
    assert not (tmp_path / 'fresh').exists()


def test_installed_cap_harness_is_the_verified_patched_checkout():
    harness = installed_harness()
    assert harness.commit == checkout.PINNED_COMMIT and harness.patch == checkout.PATCH


def test_adapter_and_ik_provider_satisfy_aspire_protocols_without_robosuite_fallback(task):
    env = adapter()
    assert isinstance(env, EnvironmentAdapter) and isinstance(CrxIKProvider(MODEL), IKProvider)
    env.reset(task, 101)
    api = api_for(env)
    assert api.get_robot_state().embodiment == EMBODIMENT  # CapApi raises here if it fell back to robosuite
    assert MotionStrategy(ik_solver=IK_SOLVER).ik_solver == 'crx_dls'
    assert api.step(command(HOME + 0.1 * NUDGE)).ok


def test_robot_state_observation_and_task_context_contracts(task):
    sim = backend()
    env = adapter(sim)
    observation = env.reset(task, 101)
    state = observation.robot_state
    assert (state.embodiment, state.base_frame, state.arms) == (EMBODIMENT, BASE_FRAME, ('left', 'right'))
    names = tuple(f'joint_{index}' for index in range(1, 7))
    for arm in ARMS:
        np.testing.assert_array_equal(state.joint_positions[arm], HOME)
        assert state.joint_positions[arm].shape == (6,) and state.joint_names[arm] == names
        np.testing.assert_allclose(state.end_effector_poses[arm].as_matrix(), MODEL.arms[arm].tcp(HOME), atol=1e-12)
        assert state.end_effector_poses[arm].frame == BASE_FRAME and state.gripper_positions[arm] == 1.0
    frame, camera = sim.camera_frames()['agent'], observation.cameras['agent']
    assert camera.frame == 'camera/agent' and camera.camera_pose.frame == BASE_FRAME
    np.testing.assert_allclose(camera.camera_pose.as_matrix(), frame.world_T_camera, atol=1e-12)
    np.testing.assert_array_equal(camera.intrinsics, frame.intrinsics)
    np.testing.assert_array_equal(camera.depth_m, frame.depth_m)
    context = observation.task_context
    assert (context.suite, context.task_name) == ('fragment_assembly', task.task_id) and 'frag_a' in context.language
    assert 'held at its goal by a fixture' in context.language
    metadata = context.metadata
    assert metadata['anchor_fixture'] == 'held at its goal pose by a fixture for the whole trial; it cannot move'
    assert (metadata['anchor_id'], metadata['moving_ids'], metadata['goal_source']) == ('frag_a', ('frag_b',),
                                                                                       'oracle_layout')
    assert metadata['assembly_plan'] == ({'fragment_id': 'frag_b', 'insertion_direction': (1.0, 0.0, 0.0)},)
    assert 'metadata assembly_plan' in context.language and 'insertion_direction' in context.language
    seeded = env.task
    assert seeded.seed == 101 and seeded.split == 'practice'
    for fragment in seeded.fragments:
        key = fragment.fragment_id
        np.testing.assert_allclose(Pose(**metadata['goal_poses'][key]).as_matrix(), fragment.goal_world_T_fragment,
                                   atol=1e-12)
        np.testing.assert_allclose(Pose(**metadata['observed_fragment_poses'][key]).as_matrix(),
                                   fragment.initial_world_T_fragment, atol=1e-12)
        assert metadata['observed_pose_sources'][key] == 'oracle_sim' and metadata['observed_pose_times_s'][key] == 0.0
        assert metadata['fragment_local_bounds'][key] == {'min': fragment.local_bounds[0],
                                                         'max': fragment.local_bounds[1]}
    assert metadata['placement_tolerance']['position_m'] == 0.003
    assert metadata['controller']['control_period_s'] == pytest.approx(0.05)
    staged = task.fragment('frag_b').initial_world_T_fragment
    assert not np.allclose(seeded.fragment('frag_b').initial_world_T_fragment, staged)  # the seed moved it


class StaleCamera(KinematicBackend):
    """Delivers frames sampled one control period before the robot state."""

    def camera_frames(self):
        return {name: replace(frame, time_s=max(0.0, frame.time_s - 0.05))
                for name, frame in super().camera_frames().items()}


class CountingCamera(KinematicBackend):
    """Counts the frames the adapter asks for."""

    renders = 0

    def camera_frames(self):
        self.renders += 1
        return super().camera_frames()


@pytest.mark.parametrize('step_frames', [True, False])
def test_a_motion_asks_for_fresh_frames_each_step_only_when_the_recording_needs_them(task, step_frames):
    """With step_frames the backend is asked for frames at every step of a motion (per-step video needs it), without
    only for the motion's final observation, which is what the program gets back: fresh either way."""
    sim = backend(CountingCamera)
    env = adapter(sim, step_frames=step_frames)
    env.reset(task, 101)
    state = api_for(env).get_robot_state()
    before = sim.renders
    path = Trajectory(joint_positions=[HOME + 0.1 * NUDGE, HOME + 0.2 * NUDGE, HOME + 0.3 * NUDGE], dt_s=0.05,
                      joint_names=state.joint_names['left'], planner='test', collision_aware=False,
                      expected_start=state, arm='left', embodiment=EMBODIMENT)
    executed = api_for(env).execute_trajectory(path)
    assert executed.ok and executed.steps_executed >= 3
    assert sim.renders - before == (executed.steps_executed if step_frames else 1)
    final = executed.final_observation.cameras['agent']
    assert final.timestamp_s == pytest.approx(sim.robot_state().time_s)


def test_camera_observations_keep_the_frame_sample_time(task):
    env = adapter(backend(StaleCamera))
    env.reset(task, 101)
    observation = env.step(command(HOME + 0.1 * NUDGE)).observation
    assert observation.robot_state.timestamp_s == observation.timestamp_s == pytest.approx(0.05)
    assert observation.cameras['agent'].timestamp_s == 0.0


class SkyCamera(KinematicBackend):
    """Some rays hit nothing: Isaac reports them as infinite depth (and a renderer may give NaN or a negative)."""

    def camera_frames(self):
        frames = {}
        for name, frame in super().camera_frames().items():
            depth = np.array(frame.depth_m, dtype=np.float32)
            depth[0, :3] = [np.inf, np.nan, -1.0]
            frames[name] = replace(frame, depth_m=depth)
        return frames


def test_a_ray_that_hits_nothing_is_invalid_depth_not_a_failed_observation(task):
    env = adapter(backend(SkyCamera))
    observation = env.reset(task, 101)
    depth = observation.cameras['agent'].depth_m
    assert depth[0, :3].tolist() == [0.0, 0.0, 0.0] and np.isfinite(depth).all() and depth[1, 1] == 1.0
    moved = env.step(command(HOME + 0.1 * NUDGE)).observation
    assert moved.cameras['agent'].depth_m[0, 0] == 0.0


SHIFT = np.eye(4)
SHIFT[0, 3] = 0.01


class OffsetObservation(KinematicBackend):
    """Reports every fragment 1 cm off along its own x axis, as an ICP estimate with an error would."""

    def observed_fragment_poses(self):
        return {key: ObservedPose(pose @ SHIFT, 'icp', self.time_s)
                for key, pose in self.privileged_fragment_poses().items()}


def test_programs_see_observed_poses_and_are_scored_on_privileged_ones(task, tmp_path):
    sim = backend(OffsetObservation)
    metadata = adapter(sim).reset(task, 101).task_context.metadata
    for key, pose in sim.privileged_fragment_poses().items():
        np.testing.assert_allclose(Pose(**metadata['observed_fragment_poses'][key]).as_matrix(), pose @ SHIFT,
                                   atol=1e-12)
        assert metadata['observed_pose_sources'][key] == 'icp'
    # The program trusts the observation: it puts the observed pose on the goal, so the true fragment ends 1 cm off.
    record = trial(task, tmp_path, REFERENCE, sim=backend(OffsetObservation), capture_videos=False)
    assert record.status == 'completed' and not record.success
    assert record.score.fragments[0].world_position_error_m == pytest.approx(0.01, abs=1e-3)
    assert program_result(record)['final_position_error_m'] < 1e-3


def test_gripper_positions_map_linearly_onto_the_configured_width_range(task):
    widths = {arm: (0.01, ROBOTIQ_PROXY_MAX_WIDTH_M) for arm in ARMS}
    sim = backend(widths=widths)
    env = adapter(sim, widths=widths)
    env.reset(task, 101)
    api = api_for(env)
    result = api.set_gripper(0.25, arm='primary')  # primary is the left arm
    assert result.ok and result.diagnostics['completion'] == 'reached'
    assert sim.robot_state().widths['left'] == pytest.approx(0.01 + 0.25 * (ROBOTIQ_PROXY_MAX_WIDTH_M - 0.01))
    assert api.get_robot_state().gripper_positions == {'left': pytest.approx(0.25), 'right': 1.0}
    both = api.set_grippers({'left': 0.0, 'right': 0.5})
    assert both.ok and both.steps_executed == 1
    assert sim.robot_state().widths == {'left': pytest.approx(0.01),
                                        'right': pytest.approx(0.01 + 0.5 * (ROBOTIQ_PROXY_MAX_WIDTH_M - 0.01))}
    assert api.close_gripper(arm='secondary').ok and sim.robot_state().widths['right'] == pytest.approx(0.01)
    sim.widths['left'] = 0.0  # a measured opening far below the configured range is a configuration error
    with pytest.raises(ValueError, match='outside the configured range'):
        env.get_robot_state()


def test_fully_open_gripper_stays_inside_a_range_that_does_not_start_at_zero(task):
    widths = {arm: (0.004, 0.041) for arm in ARMS}  # 0.004 + 1.0 * (0.041 - 0.004) rounds above 0.041
    sim = backend(widths=widths)
    env = adapter(sim, widths=widths)
    env.reset(task, 101)
    api = api_for(env)
    assert api.open_gripper(arm='left').ok and sim.robot_state().widths['left'] == 0.041
    tcp = api.get_robot_state().end_effector_poses['left']
    lower = [float(tcp.position[0]), float(tcp.position[1]), float(tcp.position[2]) - 0.05]
    moved = api.move_to_pose({'position': lower, 'quaternion_wxyz': [float(c) for c in tcp.quaternion_wxyz],
                              'frame': BASE_FRAME}, arm='left')
    assert moved.ok, moved.error


class Observer:
    def __init__(self):
        self.calls = []

    def on_reset(self, observation, metadata):
        self.calls.append(('reset', metadata['seed']))

    def before_step(self, action):
        self.calls.append(('before', tuple(action.arms)))

    def after_step(self, action, result):
        self.calls.append(('after', result.ok))


def test_rejected_commands_move_nothing_and_return_typed_failures(task):
    sim, observer = backend(), Observer()
    env = adapter(sim, run_observer=observer, config=NO_HOLD)
    env.reset(task, 101)
    api = api_for(env)
    before = sim.robot_state()
    far = HOME + np.array([0.5, 0, 0, 0, 0, 0])
    stepped = api.step(command(far, 0.0))
    assert not stepped.ok and stepped.error.code is ErrorCode.SAFETY_INTERLOCK
    assert 'velocity exceeded' in stepped.error.message
    state = api.get_robot_state()
    path = Trajectory(joint_positions=[HOME + 0.5 * NUDGE, HOME + NUDGE, far], dt_s=0.05,
                      joint_names=state.joint_names['left'], planner='test', collision_aware=False,
                      expected_start=state, arm='left', embodiment=EMBODIMENT)
    executed = api.execute_trajectory(path)
    assert not executed.ok and executed.steps_executed == 0 and executed.error.code is ErrorCode.SAFETY_INTERLOCK
    after = sim.robot_state()
    assert after.time_s == before.time_s and after.widths == before.widths
    for arm in ARMS:
        np.testing.assert_array_equal(after.joints[arm], before.joints[arm])
    assert observer.calls == [('reset', 101)] and env.rejected_commands == 2
    assert api.step(command(HOME + SHOULDER)).ok
    np.testing.assert_allclose(sim.robot_state().joints['left'], HOME + SHOULDER)
    np.testing.assert_array_equal(sim.robot_state().joints['right'], HOME)
    assert observer.calls[1:] == [('before', ARMS), ('after', True)]
    reverse = api.step(command(HOME - SHOULDER))  # 0.45 rad/s now; -0.9 in one period: -1.35 / 0.05 = -27 rad/s^2
    assert not reverse.ok and 'acceleration exceeded' in reverse.error.message
    low = SkullAssemblyAdapter(backend(), MODEL, limits(tcp_min_height_m=0.5), config=NO_HOLD, mode='practice')
    low.reset(task, 101)  # the TCPs at home are 0.40 m above the table: S0's geometry checks refuse even a small step
    refused = low.step(command(HOME + 0.5 * NUDGE))
    assert not refused.ok and 'TCP' in refused.error.message and low.backend.robot_state().time_s == 0.0


def test_a_refused_command_holds_the_last_targets_for_one_period_so_time_passes(task):
    sim, observer = backend(), Observer()
    env = adapter(sim, run_observer=observer)
    env.reset(task, 101)
    api = api_for(env)
    refused = api.step(command(HOME + np.array([0.5, 0, 0, 0, 0, 0]), 0.0))
    assert not refused.ok and refused.error.code is ErrorCode.SAFETY_INTERLOCK
    assert 'velocity exceeded' in refused.error.message and 'held their last targets for 1 control period' in \
        refused.error.message
    state = sim.robot_state()
    assert state.time_s == pytest.approx(env.control_period_s) and env.rejected_commands == 1
    for arm in ARMS:
        np.testing.assert_array_equal(state.joints[arm], HOME)
    assert observer.calls == [('reset', 101), ('before', ARMS), ('after', True)] and env.protective_stop is None
    assert api.step(command(HOME + SHOULDER)).ok


def test_an_arm_stuck_with_speed_recovers_once_the_hold_lets_it_slow(task):
    """Moving at 0.45 rad/s, a reversal needs 27 rad/s^2 and is refused; holding decelerates at 9 rad/s^2, which S0
    takes, so time passes and the arm comes to rest; then the command S0 refused before goes through in steps."""
    sim = backend()
    env = adapter(sim)
    env.reset(task, 101)
    api = api_for(env)
    assert api.step(command(HOME + SHOULDER)).ok
    reverse = api.step(command(HOME - SHOULDER))
    assert not reverse.ok and 'acceleration exceeded' in reverse.error.message and env.protective_stop is None
    assert sim.robot_state().time_s == pytest.approx(2 * env.control_period_s)
    assert api.step(command(HOME)).ok and env.protective_stop is None


def test_a_refused_command_whose_hold_is_refused_too_is_a_protective_stop(task):
    sim = backend()
    env = adapter(sim)
    env.reset(task, 101)
    api = api_for(env)
    for k in (1, 3, 6):  # 0.45, 0.9, 1.35 rad/s: holding still would decelerate at 1.35 / 0.05 = 27 rad/s^2
        assert api.step(command(HOME + k * SHOULDER)).ok
    refused = api.step(command(HOME - SHOULDER))
    assert not refused.ok and 'holding the last targets was refused too' in refused.error.message
    assert env.protective_stop is not None and 'holding the last targets' in env.protective_stop.reason
    later = api.step(command(HOME + 6 * SHOULDER))
    assert not later.ok and 'the trial is over' in later.error.message


def test_a_refused_gripper_command_is_not_latched_although_its_hold_period_stepped(task, monkeypatch):
    """CapApi keeps a gripper target as the opening of later motions unless the call raises; the hold after a refusal
    steps a period, and the refused opening must still not come back."""
    sim = backend()
    env = adapter(sim)
    env.reset(task, 101)
    api = api_for(env)
    check = env._check

    def refuse_openings(snapshot, times, path, samples):  # S0 refusing an opening, say for the fingers' boxes
        if any(widths != env._widths for _, widths in samples):
            raise Rejected('the opening puts a finger box below the table')
        return check(snapshot, times, path, samples)

    monkeypatch.setattr(env, '_check', refuse_openings)
    for refused in (api.set_gripper(0.0, arm='left'), api.set_grippers({'left': 0.0, 'right': 0.0})):
        assert not refused.ok and 'held their last targets' in refused.error.message
    assert sim.robot_state().time_s == pytest.approx(2 * env.control_period_s) and env.rejected_commands == 2
    assert all(width == ROBOTIQ_PROXY_MAX_WIDTH_M for width in sim.robot_state().widths.values())
    monkeypatch.setattr(env, '_check', check)
    plan = api.plan_motion(HOME + SHOULDER, arm='left')
    assert plan.ok and np.all(plan.trajectory.gripper_positions == 1.0)  # neither refused close was latched


def test_geometry_is_also_checked_between_control_periods(task):
    """A wrist sweep whose samples clear the TCP height limit while the path between them dips below it."""
    wrist = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    start, end = HOME - 0.1 * wrist, HOME + 0.1 * wrist
    tcp_z = {s: MODEL.arms['left'].tcp(HOME + s * wrist)[2, 3] for s in (-0.1, -0.1 / 3, 0.1 / 3, 0.1)}
    ends, between = min(tcp_z[-0.1], tcp_z[0.1]), max(tcp_z[-0.1 / 3], tcp_z[0.1 / 3])
    assert between < ends and MODEL.arms['right'].tcp(start)[2, 3] >= ends  # the TCP is lowest with the wrist at HOME
    floor = (ends + between) / 2 - 0.75
    loose = limits(max_joint_velocity_rad_s=np.full(6, 10.0), max_joint_acceleration_rad_s2=np.full(6, 500.0),
                   tcp_min_height_m=floor)
    sim = KinematicBackend(MODEL, home_joints={arm: start for arm in ARMS}, width_range_m=WIDTHS, control_hz=20)
    env = SkullAssemblyAdapter(sim, MODEL, loose, config=NO_HOLD, mode='practice')
    env.reset(task, 101)
    refused = env.step(command(end))
    assert not refused.ok and 'TCP' in refused.error.message and sim.robot_state().time_s == 0.0


def test_refused_gripper_commands_are_not_replayed_by_later_motions(task):
    sim = backend()
    env = adapter(sim, config=NO_HOLD)
    env.reset(task, 101)
    api = api_for(env)
    for k in (1, 3, 6):  # 0.45, 0.9, 1.35 rad/s: 9 rad/s^2 a period
        assert api.step(command(HOME + k * SHOULDER)).ok
    with pytest.raises(ApiError, match='safety_interlock'):  # holding still would decelerate at 1.35 / 0.05 = 27
        env.set_gripper(0.0, arm='left')
    refused = api.set_gripper(0.0, arm='left')
    assert not refused.ok and refused.steps_executed == 0 and 'safety_interlock' in refused.error.message
    assert not api.set_grippers({'left': 0.0, 'right': 0.0}).ok
    homing = api.go_home(arm='left')
    assert not homing.ok and homing.error.code is ErrorCode.SAFETY_INTERLOCK and homing.steps_executed == 0
    assert sim.robot_state().widths['left'] == ROBOTIQ_PROXY_MAX_WIDTH_M
    for k in (8, 9, 9):  # slow down: 0.9, 0.45, 0 rad/s
        assert api.step(command(HOME + k * SHOULDER)).ok
    plan = api.plan_motion(HOME, arm='left')
    assert plan.ok and np.all(plan.trajectory.gripper_positions == 1.0)  # the refused close was not latched
    assert api.execute_trajectory(plan.trajectory).ok
    assert sim.robot_state().widths['left'] == ROBOTIQ_PROXY_MAX_WIDTH_M


def test_the_recorder_step_budget_stops_commands_before_the_backend(task, tmp_path):
    sim = backend()
    recorder = RunRecorder(output_root=tmp_path, benchmark='test', suite='test', task_id=0, task_name='budget',
                           seed=101, max_steps=2, capture_videos=False)
    env = adapter(sim, run_observer=recorder)
    env.reset(task, 101)
    api = api_for(env, recorder)
    assert api.step(command(HOME + 0.5 * NUDGE)).ok and api.step(command(HOME + NUDGE)).ok
    third = api.step(command(HOME + 1.5 * NUDGE))
    assert not third.ok and 'max_steps=2' in third.error.message and recorder.terminal_reason == 'step_limit'
    np.testing.assert_allclose(sim.robot_state().joints['left'], HOME + NUDGE)
    recorder.finalize(program_ok=False, task_success=None, termination_reason='step_limit', program_result=None)


class Lagging(KinematicBackend):
    """Covers only half the remaining distance to each joint target per control period."""

    def step(self, joints, widths):
        return super().step({arm: (self.joints[arm] + np.asarray(joints[arm])) / 2 for arm in ARMS}, widths)


def test_trajectories_are_checked_whole_end_at_rest_and_report_convergence(task):
    sim = backend()
    env = adapter(sim)
    env.reset(task, 101)
    api = api_for(env)
    plan = api.plan_motion(HOME + 10 * NUDGE, arm='left')
    assert plan.ok and plan.trajectory.planner == 'joint_interpolation'
    executed = api.execute_trajectory(plan.trajectory)
    assert executed.ok and executed.steps_executed == len(plan.trajectory.joint_positions) + 1
    np.testing.assert_allclose(sim.robot_state().joints['left'], HOME + 10 * NUDGE)
    assert not np.any(sim.robot_state().joint_velocities['left'])  # the hold period brought the arm to rest
    jump = api.move_to_joints(HOME, arm='left')  # one-period jump back: what CapApi's own go_home would command
    assert not jump.ok and jump.error.code is ErrorCode.SAFETY_INTERLOCK
    homed = api.go_home(arm='primary')
    assert homed.ok and homed.steps_executed > 2
    np.testing.assert_array_equal(sim.robot_state().joints['left'], HOME)
    assert not np.any(sim.robot_state().joint_velocities['left'])
    lagging = backend(Lagging)
    env = adapter(lagging)
    env.reset(task, 101)
    api = api_for(env)
    state = api.get_robot_state()
    steps = [HOME + 1.2 * k * NUDGE for k in (1, 2, 3, 4)]  # 0.48 rad/s: 19.2 rad/s^2 from rest; lags 0.011 rad
    path = Trajectory(joint_positions=steps, dt_s=0.05, joint_names=state.joint_names['left'], planner='test',
                      collision_aware=False, expected_start=state, arm='left', embodiment=EMBODIMENT)
    executed = api.execute_trajectory(path)
    assert not executed.ok and executed.error.code is ErrorCode.TIMEOUT and executed.steps_executed == 5


def test_ik_provider_round_trips_and_fails_typed(task):
    env = adapter()
    env.reset(task, 101)
    state, provider = env.get_robot_state(), CrxIKProvider(MODEL)
    q = HOME + np.array([0.1, -0.05, 0.08, 0.02, -0.1, 0.15])
    target = matrix_to_pose(MODEL.arms['left'].tcp(q), frame=BASE_FRAME)
    solved = provider.solve_ik(target, state, arm='left')
    assert solved.ok and solved.arm == 'left' and solved.embodiment == EMBODIMENT
    reached = MODEL.arms['left'].tcp(solved.joint_positions)
    assert np.linalg.norm(reached[:3, 3] - target.position) < 1e-4
    assert np.allclose(reached[:3, :3], target.as_matrix()[:3, :3], atol=2e-3)
    unreachable = Pose(position=[3.0, 0.0, 1.0], quaternion_wxyz=target.quaternion_wxyz, frame=BASE_FRAME)
    failed = provider.solve_ik(unreachable, state, arm='left')
    assert not failed.ok and failed.joint_positions is None and failed.error.code is ErrorCode.IK_FAILED
    assert failed.error.details['reason'] in IK_FAILURES
    elsewhere = Pose(position=target.position, quaternion_wxyz=target.quaternion_wxyz, frame='world')
    assert provider.solve_ik(elsewhere, state, arm='left').error.code is ErrorCode.INVALID_REQUEST
    through_api = api_for(env).solve_ik(target, arm='primary')  # default backend crx_dls, primary is left
    assert through_api.ok and through_api.arm == 'left'
    np.testing.assert_allclose(through_api.joint_positions, solved.joint_positions)


class Untouchable:
    control_hz = 20.0

    def __getattr__(self, name):
        raise AssertionError(f'the backend was touched: {name}')


@pytest.mark.parametrize('mode, refused, allowed', [('practice', (1, 100, 131, 500, -1), (101, 125, 126, 130)),
                                                    ('evaluation', (101, 125, 126, 130, 0), (1, 100))])
def test_seed_guard_refuses_other_bands_before_the_backend_is_touched(task, tmp_path, mode, refused, allowed):
    guarded = adapter(Untouchable(), mode=mode)
    for seed in refused:
        with pytest.raises(ValueError):
            guarded.reset(task, seed)
    relabelled = json.loads(json.dumps(CONFIG))
    relabelled['baselines']['seed_splits'] = {'practice': [1, 125], 'selection': [126, 130], 'heldout': [131, 230]}
    with pytest.raises(ValueError, match='seed_splits'):
        SkullAssemblyAdapter(Untouchable(), MODEL, limits(), config=relabelled, mode=mode).reset(task, allowed[0])
    record = tmp_path / 'task.json'
    record.write_text(json.dumps(task.to_record()))
    for seed in allowed:
        env = adapter(mode=mode)
        assert env.reset(record, seed).task_context.task_name == task.task_id and env.task.seed == seed
    with pytest.raises(ValueError, match='base task'):
        adapter(mode=mode).reset(env.task, allowed[0])
    with pytest.raises(ValueError, match='mode'):
        adapter(mode='heldout')


def test_reference_program_assembles_the_fixture_and_writes_aspire_artifacts(task, tmp_path):
    idle = trial(task, tmp_path / 'idle', 'result = {}\n', capture_videos=False)
    assert idle.status == 'completed' and not idle.success and idle.score.placed == 0
    record = trial(task, tmp_path / 'reference', REFERENCE)
    assert (record.method, record.variant) == ('aspire', hashlib.sha256(REFERENCE.read_bytes()).hexdigest()[:12])
    assert (record.task_id, record.seed, record.split) == (task.task_id, 101, 'practice')
    assert (record.status, record.success, record.score.placed) == ('completed', True, 1)
    assert (record.sim_trials, record.cost_usd, record.llm_calls) == (1, None, None)
    assert 's0_rejections=0' in record.notes and 'settled=True' in record.notes and record.score.settled
    assert record.test_only and 'not baseline results' in record.backend
    run = Path(record.artifacts['run_dir'])
    outcome = json.loads((run / 'outcome.json').read_text())
    assert outcome['task_success'] is True and outcome['program_ok'] is True
    assert outcome['termination_reason'] == 'task_succeeded'
    assert outcome['steps_executed'] == len((run / 'episode/steps.jsonl').read_text().splitlines()) > 0
    assert json.loads((run / 'run.json').read_text())['status'] == 'finalized'
    assert (run / 'source/program.py').read_bytes() == REFERENCE.read_bytes()
    assert (run / 'media/videos/agent.mp4').stat().st_size > 0 and any((run / 'trace/calls').iterdir())
    assert json.loads((run / 'media/video-index.json').read_text())['fps'] == 20.0
    listed = {entry['path'] for entry in json.loads((run / 'manifest.json').read_text())['files']}
    assert {'outcome.json', 'episode/reset.json', 'source/program-result.json'} <= listed
    result = program_result(record)
    assert result['fragment'] == 'frag_b' and result['final_position_error_m'] < 1e-3


@pytest.mark.parametrize('program, status, termination', [
    ("state = get_robot_state()\nfor index in range(10):\n    step(RobotAction({'left': ArmCommand('joint_position', "
     "state.joint_positions['left'], None, embodiment='fanuc_crx')}))\nresult = {}\n", 'budget_exhausted',
     'step_limit'),
    ("raise ValueError('no plan')\n", 'error', 'program_error'),
    ('import os\nresult = {}\n', 'error', 'program_error'),
])
def test_termination_reasons_map_to_run_record_statuses(task, tmp_path, program, status, termination):
    record = trial(task, tmp_path, program, max_steps=3, capture_videos=False)
    assert record.status == status and f'termination={termination}' in record.notes and not record.success


def test_held_out_trials_are_recorded_once_per_program_variant(task, tmp_path):
    records_path, runs = tmp_path / 'records.jsonl', tmp_path / 'runs'
    common = dict(task=task, config=CONFIG, backend_factory=backend, output_root=runs, records_path=records_path,
                  max_steps=500, model=MODEL, limits=limits(), capture_videos=False)
    records = evaluate_frozen_program(REFERENCE, [1, 2], **common)
    assert [(r.seed, r.split, r.success) for r in records] == [(1, 'heldout', True), (2, 'heldout', True)]
    lines = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert [(line['method'], line['seed'], line['success']) for line in lines] == [('aspire', 1, True),
                                                                                   ('aspire', 2, True)]
    with pytest.raises(ValueError, match='already has records'):
        evaluate_frozen_program(REFERENCE, [3, 2], **common)  # refused before seed 3 runs
    with pytest.raises(ValueError, match='practice seed'):
        evaluate_frozen_program(REFERENCE, [3, 101], **common)
    single = dict(config=CONFIG, output_root=runs, mode='evaluation', max_steps=500, model=MODEL, limits=limits())
    with pytest.raises(ValueError, match='already has records'):
        run_program(REFERENCE, task, 2, backend=backend(), records_path=records_path, **single)
    with pytest.raises(ValueError, match='must be recorded'):
        run_program(REFERENCE, task, 3, backend=backend(), **single)
    assert len(records_path.read_text().splitlines()) == 2
    assert sorted(path.name for path in runs.rglob('outcome.json')) == ['outcome.json'] * 2


class Closing(KinematicBackend):
    closed = False

    def close(self):
        self.closed = True
        super().close()


def test_evaluation_stops_when_the_program_changes(task, tmp_path):
    program, built = tmp_path / 'program.py', []
    program.write_text(REFERENCE.read_text())

    def factory():
        built.append(backend(Closing))
        if len(built) == 2:
            program.write_text(REFERENCE.read_text() + '\n# edited after the first seed\n')
        return built[-1]

    with pytest.raises(ValueError, match='changed'):
        evaluate_frozen_program(program, [1, 2], task=task, config=CONFIG, backend_factory=factory,
                                output_root=tmp_path / 'runs', records_path=tmp_path / 'records.jsonl',
                                max_steps=500, model=MODEL, limits=limits(), capture_videos=False)
    assert len((tmp_path / 'records.jsonl').read_text().splitlines()) == 1
    assert built[0].closed and built[1].closed  # the refused trial still closed the backend it was handed


def test_harness_failures_close_the_backend_and_the_run_directory(task, tmp_path):
    sim = backend(Closing)
    with pytest.raises(ValueError, match='table'):  # S0 limits whose table height is not the task's
        run_program(REFERENCE, task, 101, config=CONFIG, backend=sim, output_root=tmp_path / 'runs', mode='practice',
                    max_steps=500, model=MODEL, limits=limits(table_top_z_m=0.7), capture_videos=False)
    assert sim.closed
    run = next((tmp_path / 'runs').rglob('outcome.json')).parent
    assert json.loads((run / 'run.json').read_text())['status'] == 'finalized'
    assert json.loads((run / 'outcome.json').read_text())['termination_reason'] == 'harness_error'
    broken = KinematicBackend(MODEL, home_joints={arm: HOME for arm in ARMS}, width_range_m=WIDTHS, control_hz=20)
    broken.control_hz = float('nan')  # the double refuses this itself; the runner must too, for any backend
    with pytest.raises(ValueError, match='control_hz'):
        run_program(REFERENCE, task, 101, config=CONFIG, backend=broken, output_root=tmp_path / 'nothing',
                    mode='practice', max_steps=500, model=MODEL, limits=limits(), capture_videos=False)
    assert not (tmp_path / 'nothing').exists()


def test_cli_runs_only_the_test_only_kinematic_backend(task, tmp_path, capsys):
    record = tmp_path / 'task.json'
    record.write_text(json.dumps(task.to_record()))
    args = ['run', '--program', str(REFERENCE), '--task', str(record), '--seed', '101', '--mode', 'practice',
            '--output-root', str(tmp_path / 'runs'), '--no-videos']
    assert main([*args, '--backend', 'isaac']) == 2  # this interpreter has no Isaac Sim
    assert 'Isaac Sim interpreter' in capsys.readouterr().err and not (tmp_path / 'runs').exists()
    held_out = [*args[:6], '1', *args[7:]]
    assert main([*held_out, '--backend', 'kinematic']) == 2
    assert 'heldout seed' in capsys.readouterr().err and not (tmp_path / 'runs').exists()
    unrecorded = [*args[:6], '1', '--mode', 'evaluation', *args[9:]]
    assert main([*unrecorded, '--backend', 'kinematic']) == 2
    assert 'must be recorded' in capsys.readouterr().err and not (tmp_path / 'runs').exists()
    wrong = json.loads(json.dumps(CONFIG))
    wrong['baselines']['aspire']['patch_sha256'] = '0' * 64
    (tmp_path / 'wrong.json').write_text(json.dumps(wrong))
    assert main([*args, '--backend', 'kinematic', '--config', str(tmp_path / 'wrong.json')]) == 2
    assert 'patch_sha256' in capsys.readouterr().err and not (tmp_path / 'runs').exists()
    assert main([*args, '--backend', 'kinematic', '--records', str(tmp_path / 'records.jsonl')]) == 0
    printed = capsys.readouterr()
    assert 'test double' in printed.err
    line = json.loads(printed.out)
    assert line['test_only'] and line['record']['success'] and line['record']['split'] == 'practice'
    assert len((tmp_path / 'records.jsonl').read_text().splitlines()) == 1
    assert parse_seeds('1-3,7') == [1, 2, 3, 7]
    with pytest.raises(Exception, match='backwards'):
        parse_seeds('1,5-3')


def test_scoring_happens_after_the_grippers_release_and_the_task_settle_time(task, tmp_path):
    held = REFERENCE.read_text().replace("gripper(1.0, 'release')\nmove(raised(place), 'retreat')\n", '')
    assert held != REFERENCE.read_text()
    sim = backend()
    record = trial(task, tmp_path, held, sim=sim, capture_videos=False)
    assert record.success and record.score.settled and 'settled=True' in record.notes
    assert program_result(record)['moves'][-1][0] == 'place'  # the program ended holding the fragment
    assert sim.holding() == {'left': None, 'right': None} and sim.widths == {arm: WIDTHS[arm][1] for arm in ARMS}
    assert sim.time_s >= task.settle_s


def test_a_settle_hold_that_s0_refuses_leaves_the_trial_unsettled(task, tmp_path):
    racing = ("state = get_robot_state()\nq = [float(v) for v in state.joint_positions['left']]\n"
              "for rise in (0.0225, 0.0675, 0.135):\n    target = list(q)\n    target[1] += rise\n"
              "    outcome = step(RobotAction({'left': ArmCommand('joint_position', target, None, "
              "embodiment='fanuc_crx')}))\n    if not outcome.ok:\n        raise ValueError(str(outcome.error))\n"
              "result = {}\n")
    record = trial(task, tmp_path, racing, capture_videos=False)
    assert record.status == 'completed' and not record.score.settled and not record.success
    assert 'settle_error=S0 rejected the release-and-settle hold' in record.notes


class Stopping(KinematicBackend):
    """The robot stops (ProtectiveStop) at its `at`-th step, as the Isaac backend's StopMonitor would stop it."""

    def __init__(self, at):
        super().__init__(MODEL, home_joints={arm: HOME for arm in ARMS}, width_range_m=WIDTHS, control_hz=20)
        self.at, self.count = at, 0

    def step(self, joints, widths):
        self.count += 1
        if self.count == self.at:
            raise ProtectiveStop('left joint_2 lagged its target by 20.0 mrad for 3 control periods',
                                 [{'time_s': self.time_s}])
        return super().step(joints, widths)


def test_a_protective_stop_ends_the_trial_where_the_robot_stopped(task):
    """The command fails, the episode is terminated, nothing moves after it, and the trial is not scored."""
    sim = Stopping(at=2)
    env = adapter(sim)
    env.reset(task, 101)
    api = api_for(env)
    assert api.step(command(HOME + SHOULDER)).ok
    stopped = api.step(command(HOME + 2 * SHOULDER))
    assert not stopped.ok and stopped.terminated and 'protective stop' in stopped.error.message
    assert env.protective_stop is not None and sim.count == 2
    later = api.step(command(HOME + 2 * SHOULDER))
    assert not later.ok and 'trial is over' in later.error.message and sim.count == 2
    assert env.check_success() is False and env.last_score is None and sim.count == 2  # no release, no settle


def test_a_protective_stop_while_holding_after_a_refusal_ends_the_trial(task):
    sim = Stopping(at=2)
    env = adapter(sim)
    env.reset(task, 101)
    api = api_for(env)
    assert api.step(command(HOME + SHOULDER)).ok
    stopped = api.step(command(HOME - SHOULDER))  # refused (27 rad/s^2); the robot stops during the hold
    assert not stopped.ok and stopped.terminated and sim.count == 2
    assert 'acceleration exceeded' in stopped.error.message and 'protective stop' in stopped.error.message
    assert env.protective_stop is not None and env.rejected_commands == 1


def test_a_refused_hold_is_recorded_as_a_protective_stop(task, tmp_path):
    reversing = ("state = get_robot_state()\nq = [float(v) for v in state.joint_positions['left']]\n"
                 "for k in (1, 3, 6, -1):\n    target = list(q)\n    target[1] += 0.0225 * k\n"
                 "    step(RobotAction({'left': ArmCommand('joint_position', target, None, embodiment='fanuc_crx')}))\n"
                 "result = {}\n")
    record = trial(task, tmp_path, reversing, capture_videos=False)
    assert record.status == 'protective_stop' and record.score is None and not record.success
    assert 'protective_stop=S0 refused a command and then holding the last targets' in record.notes
    assert 'steps=3/' in record.notes and 's0_rejections=1' in record.notes
    written = json.loads((Path(record.artifacts['run_dir']) / 'protective_stop.json').read_text())
    assert written['log'] == []  # no control period led to it: S0 refused the hold before it was stepped


def test_a_protective_stop_is_recorded_as_one_with_its_log(task, tmp_path):
    moving = ("state = get_robot_state()\nq = [float(v) for v in state.joint_positions['left']]\n"
              "for rise in (0.0225, 0.045, 0.0675):\n    target = list(q)\n    target[1] += rise\n"
              "    step(RobotAction({'left': ArmCommand('joint_position', target, None, embodiment='fanuc_crx')}))\n"
              "result = {}\n")
    record = trial(task, tmp_path, moving, sim=Stopping(at=2), capture_videos=False)
    assert record.status == 'protective_stop' and record.score is None and not record.success
    assert 'protective_stop=left joint_2 lagged its target' in record.notes
    assert 'termination=environment_terminated' in record.notes  # ASPIRE's own record ends the episode there
    written = json.loads((Path(record.artifacts['run_dir']) / 'protective_stop.json').read_text())
    assert written['reason'].startswith('left joint_2 lagged') and written['log'] == [{'time_s': pytest.approx(0.05)}]


class Interrupting(KinematicBackend):
    """Raises KeyboardInterrupt on the fifth control period, as Ctrl-C during a trial would."""

    def step(self, joints, widths):
        self.steps = getattr(self, 'steps', 0) + 1
        if self.steps == 5:
            raise KeyboardInterrupt
        return super().step(joints, widths)


def test_an_interrupted_heldout_trial_is_recorded_and_its_seed_is_used(task, tmp_path):
    records, other = tmp_path / 'records.jsonl', tmp_path / 'other.jsonl'
    single = dict(config=CONFIG, output_root=tmp_path / 'runs', mode='evaluation', max_steps=500, model=MODEL,
                  limits=limits(), capture_videos=False)
    with pytest.raises(KeyboardInterrupt):
        run_program(REFERENCE, task, 7, backend=backend(Interrupting), records_path=records, **single)
    lines = [json.loads(line) for line in records.read_text().splitlines()]
    assert [(line['seed'], line['status'], line['success']) for line in lines] == [(7, 'error', False)]
    assert 'KeyboardInterrupt' in lines[0]['notes'] and lines[0]['artifacts']['heldout_claim']
    with pytest.raises(ValueError, match='already has records'):
        run_program(REFERENCE, task, 7, backend=backend(), records_path=records, **single)
    with pytest.raises(ValueError, match='already claimed'):
        run_program(REFERENCE, task, 7, backend=backend(), records_path=other, **single)
    assert not other.exists()


class Exploding(KinematicBackend):
    def privileged_fragment_poses(self):
        return {**super().privileged_fragment_poses(), 'frag_b': np.full((4, 4), np.nan)}


def test_an_exploded_fragment_scores_as_a_failure_not_a_crash(task, tmp_path):
    record = trial(task, tmp_path, REFERENCE, sim=backend(Exploding), capture_videos=False)
    assert record.status == 'completed' and not record.success and record.score.invalid_ids == ('frag_b',)


def test_the_adapter_resets_only_to_the_layout_under_the_canonical_protocol(task):
    for forged in (replace(task, position_tolerance_m=0.05), replace(task, settle_s=0.1),
                   replace(task, task_id='renamed')):
        with pytest.raises(ValueError, match='not its layout under the canonical protocol'):
            adapter(Untouchable()).reset(forged, 101)


# -- Contact declarations (the S0 contact phase through ASPIRE's API), on the YAM gripper ----------------------------
from cap_harness.registry import FANUC_CRX_PUBLIC_TOOL_NAMES, ToolRegistry  # noqa: E402

from tasks.dual_arm_crag_skull.s0.kinematics import yam_model, yam_width_range_m  # noqa: E402

YAM = yam_model()
YAM_WIDTHS = {arm: yam_width_range_m() for arm in ARMS}
CONTACT = dict(contact_table_clearance_m=0.002, contact_region_radius_m=0.12, contact_max_speed_m_s=0.05)


def yam_rig(control_hz=60, home=None, contact=True):
    home = {arm: HOME for arm in ARMS} if home is None else home
    sim = KinematicBackend(YAM, home_joints=home, width_range_m=YAM_WIDTHS, control_hz=control_hz)
    env = SkullAssemblyAdapter(sim, YAM, limits(gripper_width_range_m=YAM_WIDTHS, **(CONTACT if contact else {})),
                               config=CONFIG, mode='practice')
    api = CapApi(env, ik_providers={IK_SOLVER: CrxIKProvider(YAM)}, default_camera='agent',
                 default_motion_strategy=MotionStrategy(ik_solver=IK_SOLVER))
    return sim, env, api


def down(xy, z):
    """TCP pointing straight down (TCP +x along world -z), fingers closing along world x, as a pose mapping."""
    from core.robots.kinematics import R_to_quat
    from tasks.dual_arm_crag_skull.sim.pick import down_pose
    quaternion = R_to_quat(down_pose(xy, z, 0.0)[:3, :3])
    return {'position': [float(xy[0]), float(xy[1]), z], 'quaternion_wxyz': [float(v) for v in quaternion],
            'frame': BASE_FRAME}


def test_declared_windows_are_centred_by_the_adapter_on_the_fragment(task):
    sim, env, _ = yam_rig()
    env.reset(task, 101)
    observed = sim.observed_fragment_poses()['frag_b'].world_T_fragment
    assert env.declare_contact('grasp', 'frag_b', arm='left').ok
    assert env.contacts['left'].center_xy == pytest.approx(tuple(observed[:2, 3]))
    assert env.declare_contact('place', 'frag_b', arm='left').ok
    goal = env.task.fragment('frag_b').goal_world_T_fragment
    assert env.contacts['left'].center_xy == pytest.approx(tuple(goal[:2, 3]))
    assert env.clear_contact(arm='left').ok and 'left' not in env.contacts


@pytest.mark.parametrize('arm, kind, fragment, code', [
    ('left', 'retreat', 'frag_b', ErrorCode.INVALID_REQUEST),
    ('left', 'grasp', 'frag_z', ErrorCode.NOT_FOUND),
    ('left', 'grasp', 'frag_a', ErrorCode.INVALID_REQUEST),  # the anchor is held by its fixture
    ('middle', 'grasp', 'frag_b', ErrorCode.NOT_FOUND),
])
def test_invalid_declarations_are_refused(task, arm, kind, fragment, code):
    _, env, _ = yam_rig()
    env.reset(task, 101)
    result = env.declare_contact(kind, fragment, arm=arm)
    assert not result.ok and result.error.code == code and not env.contacts


def test_reaching_the_table_needs_a_declared_window_and_keeps_it_until_the_arm_is_out(task):
    sim, env, api = yam_rig()
    env.reset(task, 101)
    xy = sim.observed_fragment_poses()['frag_b'].world_T_fragment[:2, 3]
    assert api.move_to_pose(down(xy, 0.85), arm='left', max_steps=3000).ok
    refused = api.move_to_pose(down(xy, 0.753), arm='left', max_steps=600)
    assert not refused.ok and refused.error.code == ErrorCode.SAFETY_INTERLOCK and 'contact' in refused.error.message
    assert env.declare_contact('grasp', 'frag_b', arm='left').ok
    assert api.move_to_pose(down(xy, 0.753), arm='left', max_steps=2000).ok
    for attempt in (env.declare_contact('place', 'frag_b', arm='left'), env.clear_contact(arm='left')):
        assert not attempt.ok and attempt.error.code == ErrorCode.SAFETY_INTERLOCK and 'lift' in attempt.error.message
    assert api.move_to_pose(down(xy, 0.85), arm='left', max_steps=2000).ok
    assert env.clear_contact(arm='left').ok


def test_motion_slows_down_while_a_contact_is_declared(task):
    _, env, _ = yam_rig()
    env.reset(task, 101)
    normal = env.action_bounds[1].copy()
    assert env.declare_contact('grasp', 'frag_b', arm='left').ok
    slow = env.action_bounds[1]
    assert (slow < normal).all()
    # Every finger corner then moves at most CONTACT_SPEED_FRACTION of S0's contact speed: speeds summed over the
    # joints, each times its lever to the farthest corner.
    assert slow.max() * env.contact_lever_m('left') <= 0.8 * CONTACT['contact_max_speed_m_s'] + 1e-12
    assert env.clear_contact(arm='left').ok
    np.testing.assert_allclose(env.action_bounds[1], normal)


def test_action_bounds_follow_the_meaning_of_the_backends_velocity(task):
    """ASPIRE's interpolation reaches its speed in one control period. S0 reads that step from rest over half a period
    plus the period the measured velocity averages over (RobotSnapshot.velocity_window_s): for the backends, which
    report each period's velocity, a_max dt; for an instantaneous velocity, a_max dt / 2."""
    sim, env, _ = yam_rig()
    env.reset(task, 101)
    dt, limits = env.control_period_s, env.limits
    assert sim.velocity_window_s == pytest.approx(dt)
    expected = 0.9 * np.minimum(limits.max_joint_velocity_rad_s, limits.max_joint_acceleration_rad_s2 * dt)
    np.testing.assert_allclose(env.action_bounds[1], expected)
    sim.velocity_window_s = 0.0
    np.testing.assert_allclose(env.action_bounds[1], 0.9 * np.minimum(limits.max_joint_velocity_rad_s,
                                                                      limits.max_joint_acceleration_rad_s2 * dt / 2))


def test_contact_motions_need_a_backend_at_s0_servo_rate(task):
    """S0 checks nothing between targets next to a contact, so it refuses targets further apart than one servo
    period there: a 20 Hz backend cannot work inside a contact window."""
    sim, env, api = yam_rig(control_hz=20)
    env.reset(task, 101)
    xy = sim.observed_fragment_poses()['frag_b'].world_T_fragment[:2, 3]
    assert api.move_to_pose(down(xy, 0.85), arm='left', max_steps=3000).ok
    assert env.declare_contact('grasp', 'frag_b', arm='left').ok
    refused = api.move_to_pose(down(xy, 0.753), arm='left', max_steps=2000)
    assert not refused.ok and 'at most one servo period' in refused.error.message


def test_the_contact_lever_is_the_summed_corner_speed_of_each_joint(task):
    """Independent of contact_lever_m's formula: per joint, the farthest finger corner's speed at unit joint speed
    by finite differences, summed over the joints."""
    sim, env, _ = yam_rig()
    env.reset(task, 101)
    arm, q = YAM.arms['left'], np.asarray(sim.robot_state().joints['left'], dtype=float)
    corners, total = arm.finger_box_corners(q), 0.0
    for joint in range(len(q)):
        step = np.zeros(len(q))
        step[joint] = 1e-7
        total += float(np.linalg.norm(arm.finger_box_corners(q + step) - corners, axis=1).max()) / 1e-7
    assert env.contact_lever_m('left') == pytest.approx(total, rel=1e-5)


def test_a_reset_forgets_the_declared_contacts(task):
    _, env, _ = yam_rig()
    env.reset(task, 101)
    normal = env.action_bounds[1].copy()
    assert env.declare_contact('grasp', 'frag_b', arm='left').ok
    env.reset(task, 102)
    assert not env.contacts
    np.testing.assert_allclose(env.action_bounds[1], normal)


def test_declarations_need_s0_contact_phase(task):
    _, env, _ = yam_rig(contact=False)
    observation = env.reset(task, 101)
    result = env.declare_contact('grasp', 'frag_b', arm='left')
    assert not result.ok and result.error.code == ErrorCode.UNSUPPORTED and not env.contacts
    assert observation.task_context.metadata['contact_rule']['configured'] is False


def test_a_tool_box_inside_the_band_locks_the_declarations_even_with_the_tcp_above_it(task):
    """Tilted 30 degrees about the axis across the fingers, 3 cm up: the TCP is above S0's TCP band but the finger
    box reaches into the tool band, so a new window is refused until the arm lifts out."""
    from scipy.spatial.transform import Rotation

    from tasks.dual_arm_crag_skull.s0.ik import IKOptions, solve_ik
    from tasks.dual_arm_crag_skull.sim.pick import down_pose
    table = limits().table_top_z_m
    pose = down_pose((-0.42, 0.10), table + 0.03, 0.0)
    pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec([0.0, 0.0, np.radians(30.0)]).as_matrix()
    solved = solve_ik(YAM.arms['left'], pose, HOME, IKOptions(max_seed_jump_rad=3.5))
    assert solved.ok, solved.reason
    arm, bands = YAM.arms['left'], limits(gripper_width_range_m=YAM_WIDTHS, **CONTACT)
    assert arm.tcp(solved.q)[2, 3] >= table + bands.tcp_min_height_m
    assert arm.tool_box_corners(solved.q)[:, 2].min() < table + bands.table_clearance_m
    _, env, _ = yam_rig(home={'left': solved.q, 'right': HOME})
    env.reset(task, 101)
    refused = env.declare_contact('grasp', 'frag_b', arm='left')
    assert not refused.ok and refused.error.code == ErrorCode.SAFETY_INTERLOCK and 'lift' in refused.error.message


def test_programs_reach_the_declarations_as_extension_tools_and_read_the_rule(task):
    _, env, api = yam_rig()
    observation = env.reset(task, 101)
    registry = ToolRegistry(public_extension_allowlist=FANUC_CRX_PUBLIC_TOOL_NAMES)  # as s2.aspire builds it
    names = {spec.name for spec in api.register_tools(registry).all_specs(public_only=True)}
    assert {'fanuc_crx.declare_contact', 'fanuc_crx.clear_contact'} <= names
    assert 'fanuc_crx.declare_contact' in observation.task_context.language
    rule = observation.task_context.metadata['contact_rule']
    assert rule['declare'] == 'fanuc_crx.declare_contact(kind, fragment_id, arm=...)'
    assert (rule['region_radius_m'], rule['max_speed_m_s'], rule['fingers_clearance_m']) == (0.12, 0.05, 0.002)



def test_cli_rigs_use_the_one_hardware_set_with_test_s0_values(task):
    """Every rig carries the YAM gripper (one hardware set for every method) and S0 TEST values, labelled as such."""
    from tasks.dual_arm_crag_skull.baselines.aspire.__main__ import kinematic_rig
    model, factory, limits = kinematic_rig(task)
    sim = factory()
    assert limits.gripper_width_range_m == YAM_WIDTHS and limits.contact_enabled
    assert sim.control_hz == limits.servo_hz == 60
    np.testing.assert_allclose(model.arms['left'].flange_T_tcp, YAM.arms['left'].flange_T_tcp)


def test_records_of_test_s0_values_are_test_only_on_any_backend(task, tmp_path):
    """A physics backend does not make a run with TEST S0 limits reportable: test_only marks it."""
    class Physics:  # stands in for the Isaac backend: a backend that is not the kinematic test double
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def physics():
        return Physics(KinematicBackend(MODEL, home_joints={arm: HOME for arm in ARMS}, width_range_m=WIDTHS,
                                        control_hz=20))

    assert not trial(task, tmp_path / 'default', REFERENCE, sim=physics()).test_only
    assert trial(task, tmp_path / 'flagged', REFERENCE, sim=physics(), test_only=True).test_only


def test_a_backend_settled_start_is_kept_with_the_trial(task, tmp_path):
    from types import SimpleNamespace

    class Started:  # a backend that settles its start, as the Isaac backend does
        def __init__(self, inner):
            self._inner = inner
            self.start = SimpleNamespace(to_record=lambda: {'settle_s': 1.25})

        def __getattr__(self, name):
            return getattr(self._inner, name)

    record = trial(task, tmp_path, REFERENCE, sim=Started(backend()))
    path = Path(record.artifacts['settled_start'])
    assert json.loads(path.read_text()) == {'settle_s': 1.25} and path.parent == Path(record.artifacts['run_dir'])


def test_the_physics_reference_program_declares_its_contacts_and_runs_clean(task, tmp_path, capsys):
    """Through the CLI's kinematic rig (YAM, 60 Hz, contact phase): every motion passes S0, declarations bracket the
    near-table moves. The kinematic double does not lift the box here (its capture needs the box centre at the TCP)."""
    record = tmp_path / 'task.json'
    record.write_text(json.dumps(task.to_record()))
    program = Path(checkout.__file__).with_name('reference_program_physics.py')
    assert main(['run', '--program', str(program), '--task', str(record), '--seed', '101', '--mode', 'practice',
                 '--output-root', str(tmp_path / 'runs'), '--no-videos', '--backend', 'kinematic']) == 0
    line = json.loads(capsys.readouterr().out)
    assert 'program_ok=True' in line['record']['notes'] and 's0_rejections=0' in line['record']['notes']
    run_dir = Path(line['record']['artifacts']['run_dir'])
    result = json.loads((run_dir / 'source/program-result.json').read_text())['result']
    assert [step[0] for step in result['steps']] == ['open', 'approach', 'declare grasp', 'descend', 'close', 'lift',
                                                     'clear grasp', 'transfer', 'declare place', 'lower', 'release',
                                                     'retreat', 'clear place']
    assert all(step[1] for step in result['steps']) and result['fragment'] == 'frag_b'


@pytest.mark.parametrize('trial_fails, close_fails, code', [(False, False, 0), (True, False, 1), (True, True, 1),
                                                          (False, True, 1)])
def test_the_isaac_cli_reports_failures_before_kit_ends_the_process(task, tmp_path, monkeypatch, capsys, trial_fails,
                                                                   close_fails, code):
    """The Isaac backend's shutdown ends the process, so every exception must be on stderr before it, any failure
    (of the trial or of close()) must give a non-zero exit code, and a close() that fails must not skip the
    shutdown."""
    from tasks.dual_arm_crag_skull.baselines.aspire import __main__ as cli
    from tasks.dual_arm_crag_skull.baselines.aspire import runner
    events = []

    class Kit:  # stands in for IsaacAssemblyBackend: its shutdown would end the process
        def close(self):
            events.append('close')
            if close_fails:
                raise RuntimeError('close broke')

        def shutdown(self, code):
            events.append(('shutdown', code, capsys.readouterr().err))

    class Record:
        def to_record(self):
            return {}

    calls = []

    def run_program(*args, **kwargs):
        calls.append(kwargs)
        if trial_fails:
            raise RuntimeError('trial broke')
        return Record()

    model, factory, limits = cli.kinematic_rig(task)
    monkeypatch.setattr(cli, 'isaac_rig', lambda task, config: (model, factory, limits, Kit()))
    monkeypatch.setattr(runner, 'run_program', run_program)
    record = tmp_path / 'task.json'
    record.write_text(json.dumps(task.to_record()))
    argv = ['run', '--program', str(REFERENCE), '--task', str(record), '--seed', '101', '--mode', 'practice',
            '--output-root', str(tmp_path / 'runs'), '--backend', 'isaac']
    if trial_fails:
        with pytest.raises(RuntimeError, match='trial broke'):
            main(argv)
    else:
        main(argv)
    assert events[0] == 'close' and events[1][:2] == ('shutdown', code)
    assert calls[0]['test_only'] is True  # TEST S0 values: never reportable, whatever the backend
    for failed, message in ((trial_fails, 'RuntimeError: trial broke'), (close_fails, 'RuntimeError: close broke')):
        assert (message in events[1][2]) == failed


def test_an_isaac_backend_that_fails_to_start_ends_the_process_non_zero(task, tmp_path, monkeypatch, capsys):
    """Kit may already be up when the backend's construction fails, and would then end the process with status 0:
    the CLI reports the exception and exits 1 itself, as sim/check_backend.py does."""
    import sys
    import types

    from tasks.dual_arm_crag_skull.sim import backend as sim_backend

    class Exited(Exception):
        pass

    def broken(config):
        raise RuntimeError('stage broke')

    def exit_now(code):
        raise Exited(code, capsys.readouterr().err)

    monkeypatch.setitem(sys.modules, 'isaacsim', types.ModuleType('isaacsim'))
    monkeypatch.setattr(sim_backend, 'IsaacAssemblyBackend', broken)
    monkeypatch.setattr(os, '_exit', exit_now)
    record = tmp_path / 'task.json'
    record.write_text(json.dumps(task.to_record()))
    with pytest.raises(Exited) as exited:
        main(['run', '--program', str(REFERENCE), '--task', str(record), '--seed', '101', '--mode', 'practice',
              '--output-root', str(tmp_path / 'runs'), '--backend', 'isaac'])
    code, err = exited.value.args
    assert code == 1 and 'RuntimeError: stage broke' in err


def test_s0_boxes_the_yam_fingers_at_the_opening_the_program_commands(task):
    """With S0's boxes at the openings (the TEST limits since 2026-10-21), a program that has closed the YAM may tilt it
    low over the table where one box over the whole stroke would not pass; fully open, the same descent is refused
    (the pose of tests/test_s0_opening.py, the TCP 25 mm over the table)."""
    from tasks.dual_arm_crag_skull.s0.kinematics import yam_model, yam_width_range_m
    from tasks.dual_arm_crag_skull.tests.test_s0_opening import Q_TILT, TILTED, tcp_joints
    yam, widths = yam_model(), {arm: yam_width_range_m() for arm in ARMS}
    q_up = tcp_joints([-0.42, 0.0, 0.75 + 0.085], TILTED, seed=Q_TILT)
    descent = [q_up + (Q_TILT - q_up) * f for f in np.linspace(0.0, 1.0, 41)[1:]]  # 2 s at 20 Hz
    outcomes = {}
    for opening in (0.3, 1.0):  # 28 mm; fully open
        sim = KinematicBackend(yam, home_joints={'left': q_up, 'right': HOME}, width_range_m=widths, control_hz=20)
        env = SkullAssemblyAdapter(sim, yam, limits(gripper_width_range_m=widths, finger_opening_margin_m=0.004),
                                   config=CONFIG, mode='practice')
        env.reset(task, 101)
        env.set_gripper(opening, arm='left')
        state = env.get_robot_state()
        outcomes[opening] = env.execute_trajectory(Trajectory(
            joint_positions=descent, dt_s=0.05, joint_names=state.joint_names['left'], planner='test',
            collision_aware=False, expected_start=state, arm='left', embodiment=EMBODIMENT))
    assert outcomes[0.3].ok, outcomes[0.3].error
    assert not outcomes[1.0].ok and 'tool box' in outcomes[1.0].error.message


def test_the_adapter_gives_s0_how_far_the_fingers_reach_and_bands_by_it(task, monkeypatch):
    """S0 boxes the fingers by how far they reach (RobotSnapshot.spans): the widest opening the adapter gives S0
    starts from the measured span, and whether the arm is in the contact band (for declaring and clearing a window)
    is S0's own rule at that opening."""
    from tasks.dual_arm_crag_skull.baselines.aspire import adapter as adapter_module
    from tasks.dual_arm_crag_skull.s0.kinematics import yam_model, yam_width_range_m
    from tasks.dual_arm_crag_skull.tests.test_s0_opening import Q_TILT
    yam, widths = yam_model(), {arm: yam_width_range_m() for arm in ARMS}
    sim = KinematicBackend(yam, home_joints={'left': Q_TILT, 'right': HOME}, width_range_m=widths, control_hz=20)
    env = SkullAssemblyAdapter(sim, yam, limits(gripper_width_range_m=widths, finger_opening_margin_m=0.004),
                               config=CONFIG, mode='practice')
    env.reset(task, 101)
    sim.widths = {arm: 0.028 for arm in ARMS}  # closed to 28 mm without moving (the test pose of test_s0_opening)
    env._widths = dict(sim.widths)
    assert not env._in_contact_band('left')  # S0 needs no window there at 28 mm
    measured = sim.robot_state
    sim.robot_state = lambda: (lambda state: replace(state, spans={'left': 0.09, 'right': state.widths['right']}))(
        measured())
    assert env._in_contact_band('left')  # one finger pushed far out: the boxes dip into the table
    seen = []
    monkeypatch.setattr(adapter_module, 'check_joint_path', lambda *args, **kwargs: seen.append(kwargs['openings']))
    monkeypatch.setattr(adapter_module, 'check_geometry', lambda *args, **kwargs: None)
    env.set_gripper(0.3, arm='right')
    assert seen and seen[0]['left'] == pytest.approx(0.09)


def test_the_adapter_bounds_the_fingers_reach_by_what_s0_checked(task):
    """Once S0 accepts a command and before it is stepped, the adapter tells the backend how far the fingers may reach
    (the protective stop's finger rule): the command's widest opening plus S0's margin."""
    from tasks.dual_arm_crag_skull.s0.kinematics import yam_model, yam_width_range_m
    yam, widths = yam_model(), {arm: yam_width_range_m() for arm in ARMS}
    sim = KinematicBackend(yam, home_joints={arm: HOME for arm in ARMS}, width_range_m=widths, control_hz=20)
    env = SkullAssemblyAdapter(sim, yam, limits(gripper_width_range_m=widths, finger_opening_margin_m=0.004),
                               config=CONFIG, mode='practice')
    env.reset(task, 101)
    bounds, real = [], sim.bound_reach
    sim.bound_reach = lambda value: bounds.append(value) or real(value)
    env.set_gripper(0.3, arm='left')
    assert bounds[0]['left'] == pytest.approx(widths['left'][1] + 0.004)  # it was fully open
    env.set_gripper(0.3, arm='left')
    assert bounds[-1]['left'] == pytest.approx(0.3 * widths['left'][1] + 0.004)
