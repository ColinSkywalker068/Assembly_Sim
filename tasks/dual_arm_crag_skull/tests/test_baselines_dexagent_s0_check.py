"""S0's verdict on a DexAgent delivery before it runs (baselines/dexagent/s0_check.py): the spec, the replay that
tracks the commands exactly and observes the fragments where the recording has them."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tasks.dual_arm_crag_skull.baselines.dexagent import s0_check
from tasks.dual_arm_crag_skull.baselines.dexagent.trajectory import free_joint_columns
from tasks.dual_arm_crag_skull.contracts.actions import ARMS
from tasks.dual_arm_crag_skull.s0.ik import IKOptions, solve_ik
from tasks.dual_arm_crag_skull.sim.pick import down_pose
from tasks.dual_arm_crag_skull.tests.test_baselines_dexagent_executor import (HOME, LIMITS, MODEL, STAGING, above,
                                                                               ramp, task)  # noqa: F401  (fixture)

RIG_T_WORLD = np.eye(4)  # a yawed and shifted rig, so that every use of it must be the right way round
RIG_T_WORLD[:3, :3] = Rotation.from_euler('z', 30, degrees=True).as_matrix()
RIG_T_WORLD[:3, 3] = [0.1, -0.2, -0.75]


def manifest(task):
    objects = []
    for key in task.fragment_ids:
        start = RIG_T_WORLD @ task.fragment(key).initial_world_T_fragment
        x, y, z, w = Rotation.from_matrix(start[:3, :3]).as_quat()
        objects.append({'fragment_id': key, 'root_body': f'{key}_{key}',
                        'free_joint': None if key == task.anchor_id else f'{key}_{key}_free',
                        'body_T_fragment': np.eye(4).tolist(), 'start_rig_wxyz': [*start[:3, 3], w, x, y, z]})
    return {'rig_T_world': RIG_T_WORLD.tolist(), 'objects': objects}


def recording(path, left, labels, objects, fps=30.0):
    """A trajectory.npz as DexAgent writes one: named columns, the reference, frame times and labels, and the
    moving fragments' free joints where the manifest starts them."""
    t = len(left)
    names, reference = [], []
    for arm in ARMS:
        joints = left if arm == 'left' else np.repeat(HOME[None], t, axis=0)
        for i in range(6):
            names.append(f'robot_{arm}_arm_dof_joint{i + 1}')
            reference.append(joints[:, i])
        for jaw in ('thumb', 'index'):
            names.append(f'robot_{arm}_{arm}_{jaw}_FE')
            reference.append(np.zeros(t))
    for obj in objects:
        if obj['free_joint'] is None:
            continue
        for k, value in enumerate(obj['start_rig_wxyz']):
            names.append(f"{obj['free_joint']}[{k}]")
            reference.append(np.full(t, value))
    reference = np.column_stack(reference)
    np.savez(path, joint_names=np.array(names), reference=reference, qpos=reference, time_s=np.arange(t) / fps,
             labels=np.array(labels))
    return path


def test_limits_survive_the_spec(task, tmp_path):
    record = s0_check.limits_record(LIMITS)
    again = s0_check.limits_from_record(record)
    np.testing.assert_array_equal(again.max_joint_velocity_rad_s, LIMITS.max_joint_velocity_rad_s)
    assert again.gripper_width_range_m == LIMITS.gripper_width_range_m
    assert again.contact_max_speed_m_s == LIMITS.contact_max_speed_m_s


def test_a_slow_grasp_descent_passes_with_its_window_on_the_recorded_fragment(task, tmp_path):
    spec = s0_check.write_spec(tmp_path / 's0_spec.json', task=task, limits=LIMITS, manifest=manifest(task),
                               repo=tmp_path)
    high, low = above(0.10), above(0.015)
    left = np.vstack([ramp(HOME, high, 3.0), ramp(high, low, 4.0)[1:], ramp(low, high, 4.0)[1:]])
    labels = ['left: descend on frag_b_frag_b'] * len(left)
    verdict = s0_check.check_trajectory(recording(tmp_path / 't.npz', left, labels, manifest(task)['objects']), spec)
    assert verdict['ok'], verdict
    (window,) = verdict['windows']
    np.testing.assert_allclose(window['center_xy'], STAGING['frag_b'][:2], atol=1e-12)  # rig back to world


def test_a_fast_descent_is_s0s_rejection(task, tmp_path):
    spec = s0_check.write_spec(tmp_path / 's0_spec.json', task=task, limits=LIMITS, manifest=manifest(task),
                               repo=tmp_path)
    high, low = above(0.10), above(0.015)
    left = np.vstack([ramp(HOME, high, 3.0), ramp(high, low, 0.6)[1:]])
    verdict = s0_check.check_trajectory(recording(tmp_path / 't.npz', left, ['left: descend on frag_b_frag_b'] * len(left),
                                                  manifest(task)['objects']), spec)
    assert not verdict['ok'] and verdict['status'] == 'rejected' and 'near the table' in verdict['reason']


def test_a_spec_of_another_schema_is_refused(task, tmp_path):
    with pytest.raises(ValueError, match='S0 spec'):
        s0_check.check_trajectory(tmp_path / 'x.npz', {'schema': 'other'})


def test_the_replay_observes_each_fragment_where_the_recording_has_it_then(task, tmp_path):
    objects = manifest(task)['objects']
    path = recording(tmp_path / 't.npz', np.repeat(HOME[None], 31, axis=0), ['left: wait'] * 31, objects)
    with np.load(path) as data:
        names, qpos, time_s = [str(n) for n in data['joint_names']], data['qpos'].copy(), data['time_s']
    moving = next(o for o in objects if o['free_joint'] is not None)
    qpos[15:, free_joint_columns(names, moving['free_joint'])[0]] += 0.05  # 5 cm along the rig's x at 0.5 s
    hold, widths = {arm: HOME for arm in ARMS}, {arm: 0.05 for arm in ARMS}
    replay = s0_check.ReplayBackend(hold, 60, widths, (time_s, qpos, names), np.linalg.inv(RIG_T_WORLD), objects)
    initial = task.fragment(moving['fragment_id']).initial_world_T_fragment
    np.testing.assert_allclose(replay.privileged_fragment_poses()[moving['fragment_id']], initial, atol=1e-12)
    for _ in range(29):
        replay.step(hold, widths)
    np.testing.assert_allclose(replay.privileged_fragment_poses()[moving['fragment_id']], initial, atol=1e-12)
    replay.step(hold, widths)  # 0.5 s: frame 15
    moved = replay.privileged_fragment_poses()[moving['fragment_id']][:3, 3] - initial[:3, 3]
    np.testing.assert_allclose(moved, np.linalg.inv(RIG_T_WORLD)[:3, :3] @ [0.05, 0.0, 0.0], atol=1e-12)


def test_the_replay_reports_each_servo_period_s_velocity_and_says_so(task, tmp_path):
    objects = manifest(task)['objects']
    path = recording(tmp_path / 't.npz', np.repeat(HOME[None], 31, axis=0), ['left: wait'] * 31, objects)
    with np.load(path) as data:
        names, qpos, time_s = [str(n) for n in data['joint_names']], data['qpos'].copy(), data['time_s']
    hold, widths = {arm: HOME for arm in ARMS}, {arm: 0.05 for arm in ARMS}
    replay = s0_check.ReplayBackend(hold, 60, widths, (time_s, qpos, names), np.linalg.inv(RIG_T_WORLD), objects)
    assert replay.robot_state().velocity_window_s == pytest.approx(1 / 60) == replay.velocity_window_s
    nudged = {arm: HOME + 0.001 for arm in ARMS}
    replay.step(nudged, widths)
    np.testing.assert_allclose(replay.robot_state().joint_velocities['left'], np.full(6, 0.06))


def test_no_window_is_declared_on_the_anchor(task, tmp_path):
    spec = s0_check.write_spec(tmp_path / 's0_spec.json', task=task, limits=LIMITS, manifest=manifest(task),
                               repo=tmp_path)
    xy = task.fragment(task.anchor_id).initial_world_T_fragment[:2, 3] + np.array([0.0, 0.06])

    def q(z, seed):
        solved = solve_ik(MODEL.arms['left'], down_pose(xy, LIMITS.table_top_z_m + z, 0.0), seed,
                          IKOptions(max_seed_jump_rad=3.5))
        assert solved.ok, solved.reason
        return solved.q

    high = q(0.10, HOME)
    low = q(0.015, high)
    left = np.vstack([ramp(HOME, high, 3.0), ramp(high, low, 4.0)[1:]])
    labels = ['left: descend on frag_a_frag_a'] * len(left)  # the fixture holds the anchor: no grasp contact on it
    verdict = s0_check.check_trajectory(recording(tmp_path / 't.npz', left, labels, manifest(task)['objects']), spec)
    assert not verdict['ok'] and verdict['status'] == 'rejected' and verdict['windows'] == []


def test_the_check_starts_the_robot_at_home_as_the_backends_do(task, tmp_path):
    spec = s0_check.write_spec(tmp_path / 's0_spec.json', task=task, limits=LIMITS, manifest=manifest(task),
                               repo=tmp_path)
    left = ramp(HOME + 0.01, HOME + 0.05, 1.0)  # planned from elsewhere
    verdict = s0_check.check_trajectory(recording(tmp_path / 't.npz', left, ['left: wait'] * len(left),
                                                  manifest(task)['objects']), spec)
    assert not verdict['ok'] and verdict['status'] == 'refused'
