import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tasks.dual_arm_crag_skull.contracts.actions import ARMS, decode_chunk, executable_reason, hand_pose
from tasks.dual_arm_crag_skull.contracts.skills import SkillRequest
from tasks.dual_arm_crag_skull.s1 import synthetic
from tasks.dual_arm_crag_skull.s1.chunking import chunk_from_prediction
from tasks.dual_arm_crag_skull.s1.dataset import (
    LOWDIM_DIM, POLICY_INPUTS, Normalizer, PolicyDataset, encode_observation, live_sample, policy_grid, split_episodes,
    trained_combinations)
from tasks.dual_arm_crag_skull.s1.episode import (
    CAMERAS, CameraStream, Episode, PoseStream, SkillSegment, align_frames, buffers_until, causal_state,
    initial_condition_key, load_episode, load_privileged, resample, save_episode)
from tasks.dual_arm_crag_skull.s1.synthetic import demo_report, make_episode, make_episodes

OBSERVED = {'source': 'observed_geometry', 'observation_id': 'o', 'pose_source': 'icp'}


def pose(translation, rotvec=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    t[:3, 3] = translation
    return t


def replace_episode(episode, **changes):
    return Episode(**{**episode.__dict__, **changes})


def test_resample_slerps_rotation_and_interpolates_position_and_width():
    stream = PoseStream([0.0, 1.0], [pose([0, 0, 0]), pose([1, 2, 3], [0, 0, np.pi / 2])], [0.02, 0.06])
    poses, widths = resample(stream, [0.25, 0.5])
    np.testing.assert_allclose(poses[1, :3, 3], [0.5, 1.0, 1.5])
    np.testing.assert_allclose(poses[1, :3, :3], Rotation.from_rotvec([0, 0, np.pi / 4]).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(poses[0, :3, :3], Rotation.from_rotvec([0, 0, np.pi / 8]).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(widths, [0.03, 0.04])


@pytest.mark.parametrize('time', [-0.001, 1.001])
def test_resample_refuses_to_extrapolate(time):
    stream = PoseStream([0.0, 1.0], [pose([0, 0, 0]), pose([1, 0, 0])], [0.02, 0.02])
    with pytest.raises(ValueError, match='extrapolate'):
        resample(stream, [0.5, time])
    with pytest.raises(ValueError, match='two pose samples'):
        resample(PoseStream([0.0], [pose([0, 0, 0])], [0.02]), [0.0])


def test_causal_state_uses_the_latest_sample_not_a_future_one():
    stream = PoseStream([0.0, 0.1], [pose([0, 0, 0]), pose([1, 0, 0])], [0.02, 0.06])
    poses, widths = causal_state(stream, [0.05, 0.1, 0.14])
    np.testing.assert_allclose(poses[:, 0, 3], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(widths, [0.02, 0.06, 0.06])
    with pytest.raises(ValueError, match='within'):
        causal_state(stream, [0.25])  # newest sample 0.15 s old
    with pytest.raises(ValueError, match='no pose sample'):
        causal_state(stream, [-0.01])


def test_frames_are_usable_only_after_they_are_received():
    frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    camera = CameraStream([0.0, 0.1, 0.3], frames, [0.03, 0.14, 0.34])
    index, available = align_frames(camera, [0.02, 0.05, 0.12, 0.2, 0.35, 0.5], max_age_s=0.15)
    np.testing.assert_array_equal(index, [-1, 0, 0, 1, 2, -1])
    np.testing.assert_array_equal(available, [False, True, True, True, True, False])
    index, available = align_frames(None, [0.1, 0.2])
    assert not available.any() and (index == -1).all()
    with pytest.raises(ValueError, match='receive'):
        CameraStream([0.0, 0.1], frames[:2], [0.05, 0.04])
    with pytest.raises(ValueError, match='clock domain'):
        CameraStream([0.0], frames[:1], [0.0], clock_domain='wall_clock')


def test_stream_and_segment_validation():
    with pytest.raises(ValueError):
        PoseStream([0.0, 0.0], [pose([0, 0, 0])] * 2, [0.0, 0.0])
    with pytest.raises(ValueError):
        PoseStream([0.0, 1.0], [pose([0, 0, 0]), np.diag([2.0, 1, 1, 1])], [0.0, 0.0])
    with pytest.raises(ValueError):
        CameraStream([0.0], np.zeros((1, 2, 2, 3), dtype=np.float32), [0.0])
    with pytest.raises(ValueError, match='needs target pose'):
        SkillSegment(0.0, 1.0, 'place', ('left',), 'r', fragment_id='p')
    with pytest.raises(ValueError, match='provenance source'):
        SkillSegment(0.0, 1.0, 'place', ('left',), 'r', fragment_id='p', target_pose=np.eye(4))
    with pytest.raises(ValueError, match='provenance source'):
        SkillSegment(0.0, 1.0, 'place', ('left',), 'r', fragment_id='p', target_pose=np.eye(4),
                     target_provenance={'source': 'gt_object_pose'})
    with pytest.raises(ValueError, match='needs the observed fragment pose'):
        SkillSegment(0.0, 1.0, 'pick', ('left',), 'r', fragment_id='p')
    with pytest.raises(ValueError, match='observed_geometry'):
        SkillSegment(0.0, 1.0, 'pick', ('left',), 'r', fragment_id='p', fragment_pose=np.eye(4))
    with pytest.raises(ValueError, match='takes no fragment_id'):
        SkillSegment(0.0, 1.0, 'retreat', ('left',), 'r', fragment_id='p')
    both = SkillSegment(0.0, 1.0, 'retreat', ('right', 'left'), 'r')
    assert both.active_arms == ('left', 'right')
    oracle = SkillSegment(0.0, 1.0, 'place', ('left',), 'r', fragment_id='p', target_pose=np.eye(4),
                          target_provenance={'source': 'oracle_ground_truth'})
    assert oracle.oracle and not both.oracle


def test_episode_rejects_segments_outside_span_and_unknown_source():
    episode, _ = make_episode(0, seed=1)
    fields = dict(episode_id='e', source='sim_expert', fragment_ids=('part_0', 'part_1'), geometry_hash='g',
                  calibration_version='c', tcp=dict(episode.tcp))
    late = SkillSegment(episode.segments[0].start, episode.span()[1] + 1.0, 'retreat', ('left',), 'r')
    with pytest.raises(ValueError, match='inside the recorded TCP span'):
        Episode(segments=(late,), **fields)
    with pytest.raises(ValueError, match='source'):
        Episode(segments=episode.segments, **{**fields, 'source': 'youtube'})


def test_supervision_stays_in_the_segment_and_holds_its_last_step():
    episodes, _ = make_episodes(3, seed=2)
    dataset = PolicyDataset(episodes)
    for ref in dataset.windows:
        episode = dataset.episodes[ref.episode]
        segment = episode.segments[ref.segment]
        first, last = dataset._cache[ref.episode]['ranges'][ref.segment]
        history, future = dataset._steps(ref.step, first, last)
        assert history.min() >= 0 and history.max() == ref.step and first <= ref.step <= last
        assert future.min() > ref.step or ref.step == last
        assert future.max() <= last
        grid = policy_grid(episode)
        final = ref.segment == len(episode.segments) - 1
        assert (grid[future] >= segment.start - 1e-9).all()
        assert (grid[future] <= segment.end + 1e-9).all() if final else (grid[future] < segment.end - 1e-9).all()
    per_segment = {}
    for ref in dataset.windows:
        per_segment[(ref.episode, ref.segment)] = per_segment.get((ref.episode, ref.segment), 0) + 1
    # pick covers grid steps 0..13, place 14..33, retreat 34..46: every step but the last is an observation.
    assert sorted(set(per_segment.values())) == [12, 13, 19]
    late = dataset.lowdim_action_target([r for r in dataset.windows if r.segment == 1][-1])
    np.testing.assert_allclose(late['action'][-1], late['action'][-2], atol=1e-12)  # held after the segment end
    assert late['padded_steps'] == 15


def test_the_end_of_every_segment_is_observed():
    episodes, _ = make_episodes(2, seed=3)
    dataset = PolicyDataset(episodes)
    for (episode, segment), group in _group(dataset).items():
        span = dataset.episodes[episode].segments[segment]
        observed = [dataset._cache[episode]['grid'][r.step] for r in group]
        assert min(observed) - span.start < 0.05 and span.end - max(observed) < 0.25


def _group(dataset):
    groups = {}
    for ref in dataset.windows:
        groups.setdefault((ref.episode, ref.segment), []).append(ref)
    return groups


def test_short_segment_still_yields_windows():
    episode, _ = make_episode(0, seed=4)
    pick, place, retreat = episode.segments
    t = place.start
    short = SkillSegment(t, t + 0.15, 'place', place.active_arms, 'r', fragment_id=place.fragment_id,
                         target_pose=place.target_pose, target_provenance=place.target_provenance)
    rest = SkillSegment(t + 0.15, retreat.end, 'retreat', retreat.active_arms, 'r2')
    dataset = PolicyDataset([replace_episode(episode, segments=(pick, short, rest))])
    assert sum(r.segment == 1 for r in dataset.windows) >= 1


def test_failed_and_unknown_outcome_episodes_are_excluded_by_default():
    ok, _ = make_episode(0, seed=5)
    failed = replace_episode(ok, episode_id='failed', success=False, failure_reason='dropped part')
    unknown = replace_episode(ok, episode_id='unknown', success=None)
    dataset = PolicyDataset([ok, failed, unknown])
    assert {e.episode_id for e in dataset.episodes} == {ok.episode_id}
    assert dataset.stats['excluded_failed'] == 1 and dataset.stats['excluded_unknown_outcome'] == 1
    assert Normalizer.fit(dataset).fitted_on == (ok.episode_id,)
    assert len(PolicyDataset([ok, failed], include_failures=True).episodes) == 2


def test_oracle_conditioning_is_refused_unless_enabled():
    episode, _ = make_episode(0, seed=6)
    pick, place, retreat = episode.segments
    oracle_place = SkillSegment(place.start, place.end, 'place', place.active_arms, place.request_id,
                                fragment_id=place.fragment_id, target_pose=place.target_pose,
                                target_provenance={'source': 'oracle_ground_truth'})
    oracle = replace_episode(episode, segments=(pick, oracle_place, retreat))
    with pytest.raises(ValueError, match='oracle'):
        PolicyDataset([oracle])
    assert len(PolicyDataset([oracle], allow_oracle=True)) > 0


def test_windows_across_a_pose_stream_gap_are_dropped():
    episode, _ = make_episode(0, seed=7)
    left = episode.tcp['left']
    keep = (left.times - left.times[0] < 0.6) | (left.times - left.times[0] > 1.6)
    gapped = replace_episode(episode, tcp={'left': PoseStream(left.times[keep], left.poses[keep], left.widths[keep]),
                                           'right': episode.tcp['right']})
    full, dataset = PolicyDataset([episode]), PolicyDataset([gapped])
    assert dataset.stats['windows_dropped_gap'] > 0 and len(dataset) < len(full)
    t0 = left.times[0]
    for ref in dataset.windows:
        first, last = dataset._cache[0]['ranges'][ref.segment]
        times = dataset._cache[0]['grid'][dataset._steps(ref.step, first, last)[1]] - t0
        assert not ((times > 0.6 + 0.04) & (times < 1.6 - 0.04)).any()


def test_sample_inputs_exclude_absolute_pose_and_privileged_truth():
    episodes, _ = make_episodes(2, seed=8)
    sample = PolicyDataset(episodes)[0]
    assert set(POLICY_INPUTS) <= set(sample)
    assert sample['lowdim'].shape == (2, LOWDIM_DIM)
    assert not {'mode', 'obstacle_xy', 'via_xy', 'chosen', 'arm'} & set(sample)
    # The current step of each hand's own history is the identity: no absolute world pose leaks in.
    np.testing.assert_allclose(sample['lowdim'][-1, :9], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-12)


def test_missing_camera_is_masked_with_zero_images():
    episode, _ = make_episode(0, seed=9, missing_agent_rate=1.0)
    assert episode.source == 'umi' and 'agent' not in episode.cameras
    sample = PolicyDataset([episode])[4]
    agent = CAMERAS.index('agent')
    assert not sample['image_mask'][:, agent].any()
    assert not sample['images'][:, agent].any()
    assert sample['image_mask'][:, CAMERAS.index('left_wrist')].all()


def test_split_groups_retries_of_one_initial_condition():
    episodes, _ = make_episodes(30, seed=10)
    retries = [replace_episode(e, episode_id=e.episode_id + '-retry', success=False) for e in episodes[:10]]
    split = split_episodes(episodes + retries, seed=3)
    where = {i: name for name, ids in split.items() for i in ids}
    for retry in retries:
        assert where[retry.episode_id] == where[retry.episode_id[:-len('-retry')]]
    assert split == split_episodes(list(reversed(episodes + retries)), seed=3)
    assert split_episodes(episodes, seed=4) != split_episodes(episodes, seed=3)
    bare = replace_episode(episodes[0], initial_condition_id=None, seed=5)
    assert initial_condition_key(bare) == f'geometry_seed:{bare.geometry_hash}:5'
    with pytest.raises(ValueError):
        split_episodes([episodes[0], episodes[0]])


def test_normalizer_uses_training_split_only_and_round_trips(tmp_path):
    episodes, _ = make_episodes(6, seed=11)
    split = split_episodes(episodes, fractions=(0.5, 0.5, 0.0))
    train = PolicyDataset([e for e in episodes if e.episode_id in split['train']])
    val = PolicyDataset([e for e in episodes if e.episode_id in split['val']])
    normalizer = Normalizer.fit(train)
    assert set(normalizer.fitted_on) == set(split['train'])
    actions = np.stack([train.lowdim_action_target(r)['action'] for r in train.windows]).reshape(-1, 20)
    normalized = normalizer.normalize('action', actions)
    assert normalized.min() >= -1 - 1e-9 and normalized.max() <= 1 + 1e-9
    assert Normalizer.fit(val).offset['action'][0] != normalizer.offset['action'][0]
    np.testing.assert_allclose(normalizer.unnormalize('action', normalized), actions, atol=1e-12)
    for name in ('action', 'target', 'fragment'):
        np.testing.assert_array_equal(normalizer.scale[name][3:9], 1.0)
    path = tmp_path / 'normalizer.json'
    normalizer.save(path)
    loaded = Normalizer.load(path)
    for name in normalizer.scale:
        np.testing.assert_array_equal(loaded.scale[name], normalizer.scale[name])
        np.testing.assert_array_equal(loaded.offset[name], normalizer.offset[name])
    assert json.loads(path.read_text())['supervision'] == 'achieved'


def test_action_window_decodes_to_future_world_poses_through_contract():
    episodes, _ = make_episodes(2, seed=13)
    dataset = PolicyDataset(episodes)
    index = next(i for i, r in enumerate(dataset.windows) if dataset.lowdim_action_target(r)['padded_steps'] == 0)
    sample = dataset[index]
    chunk, info = chunk_from_prediction(sample['action'], anchors=sample['anchors'], held_widths={},
                                        observation_time=sample['observation_time'], request_id='r', chunk_id=0,
                                        active_arms=('left', 'right'))
    assert info == {'width_clipped': 0, 'held_arms': ()}
    episode = dataset.episodes[dataset.windows[index].episode]
    decoded = decode_chunk(chunk)
    for arm in ARMS:
        poses, widths = resample(episode.tcp[arm], sample['future_times'])
        np.testing.assert_allclose(decoded[arm][0], poses, atol=1e-9)
        np.testing.assert_allclose(decoded[arm][1], widths, atol=1e-12)
    np.testing.assert_allclose(chunk.waypoint_times, sample['observation_time'] + np.arange(1, 17) / 10)


def test_held_arm_keeps_the_commanded_width_not_the_measured_one():
    episodes, _ = make_episodes(1, seed=14)
    sample = PolicyDataset(episodes)[2]
    actions = sample['action'].copy()
    active = ('left',) if sample['arm_mask'][0] else ('right',)
    held = ARMS[1 - ARMS.index(active[0])]
    actions[3, 9 if active == ('left',) else 19] = -0.002
    chunk, info = chunk_from_prediction(actions, anchors=sample['anchors'], held_widths={held: 0.020},
                                        observation_time=sample['observation_time'], request_id='r', chunk_id=1,
                                        active_arms=active)
    assert info == {'width_clipped': 1, 'held_arms': (held,)}
    assert executable_reason(chunk, now=sample['observation_time'] + 0.05) is None
    world, width = decode_chunk(chunk)[held]
    np.testing.assert_allclose(world, np.repeat(sample['anchors'][ARMS.index(held)][None], 16, axis=0), atol=1e-12)
    np.testing.assert_allclose(width, 0.020)
    assert abs(sample['widths_now'][ARMS.index(held)] - 0.020) > 0.01  # measured width differs from the command
    with pytest.raises(ValueError, match='commanded width'):
        chunk_from_prediction(actions, anchors=sample['anchors'], held_widths={}, observation_time=0.0,
                              request_id='r', chunk_id=1, active_arms=active)


def test_commanded_supervision_is_explicit_stamped_at_reach_time_and_never_mixed():
    episodes, _ = make_episodes(2, seed=15)
    episode = episodes[0]
    np.testing.assert_allclose(episode.commanded['left'].times, episode.tcp['left'].times + synthetic.COMMAND_LEAD_S)
    achieved, commanded = PolicyDataset(episodes), PolicyDataset(episodes, supervision='commanded')
    assert commanded[3]['supervision'] == 'commanded'
    np.testing.assert_allclose(commanded[3]['lowdim'], achieved[3]['lowdim'])
    # Stamped at the reach time, a command equals the plan there: it tracks the achieved motion closely.
    assert np.abs(commanded[3]['action'][:, :3] - achieved[3]['action'][:, :3]).max() < 2e-3
    bare = replace_episode(episodes[0], commanded=None)
    with pytest.raises(ValueError, match='lack commanded'):
        PolicyDataset([episodes[1], bare], supervision='commanded')
    with pytest.raises(ValueError):
        PolicyDataset(episodes, supervision='mixed')


def test_save_load_round_trip_never_opens_privileged(tmp_path):
    episode, privileged = make_episode(3, seed=16)
    folder = save_episode(episode, tmp_path / 'ep', privileged=privileged)
    with pytest.raises(FileExistsError):
        save_episode(episode, folder)
    loaded = load_episode(folder)
    assert loaded.initial_condition_id == episode.initial_condition_id
    np.testing.assert_array_equal(loaded.cameras['left_wrist'].receive_times, episode.cameras['left_wrist'].receive_times)
    a, b = PolicyDataset([episode]), PolicyDataset([loaded])
    assert len(a) == len(b)
    for index in (0, 7, len(a) - 1):
        for key in ('action', 'lowdim', 'images', 'image_mask', 'target', 'fragment', 'arm_mask'):
            np.testing.assert_array_equal(a[index][key], b[index][key])
    (folder / 'privileged.npz').write_bytes(b'not an npz')
    load_episode(folder)  # still loads: the privileged file is never read
    with pytest.raises(Exception):
        load_privileged(folder)


def test_demonstrations_pick_the_requested_fragment_and_keep_both_modes():
    episodes, privileged = make_episodes(30, seed=17)
    report = demo_report(PolicyDataset(episodes), privileged)
    assert report['pick_first']['chosen_fragment_rate'] == 1.0
    assert report['place_first']['obstacle_hit_rate'] == 0.0
    assert 0.2 < report['place_first']['left_fraction_of_clear'] < 0.8
    assert report['place_mid']['mode_agreement_of_clear'] == 1.0
    assert report['place_late']['final_error_m_median'] < 0.003 and report['place_late']['final_width_m_median'] > 0.075
    assert report['arms']['inactive_motion_m_median'] < 0.002 and report['arms']['active_motion_m_median'] > 0.05
    assert {int(p['arm']) for p in privileged.values()} == {0, 1}


def _request(segment):
    completion = {'pick': {'type': 'grasp_held'}, 'retreat': {'type': 'retreated', 'clearance_m': 0.1}}.get(
        segment.skill_id, {'type': 'pose_within', 'position_tolerance_m': .003, 'rotation_tolerance_deg': 3.0})
    return SkillRequest(request_id='r', observation_id='obs', skill_id=segment.skill_id, active_arms=segment.active_arms,
                        fragment_id=segment.fragment_id, target_pose=segment.target_pose,
                        target_provenance=segment.target_provenance, fragment_pose=segment.fragment_pose,
                        fragment_pose_provenance=segment.fragment_pose_provenance, completion_predicate=completion,
                        timeout_s=10.0)


def test_live_sample_from_past_only_buffers_matches_the_training_window():
    episodes, _ = make_episodes(1, seed=19)
    dataset = PolicyDataset(episodes)
    episode = dataset.episodes[0]
    for index in (0, 6, 13, 20, 32, len(dataset) - 1):
        window = dataset[index]
        request = _request(episode.segments[dataset.windows[index].segment])
        tcp, cameras = buffers_until(episode, window['observation_time'])
        live = live_sample(tcp, cameras, window['observation_time'], request, image_shape=dataset.image_shape)
        for key in ('lowdim', 'target', 'target_mask', 'fragment', 'fragment_mask', 'arm_mask', 'anchors',
                    'widths_now', 'images', 'image_mask'):
            np.testing.assert_allclose(live[key], window[key], atol=1e-12, err_msg=f'{key} @ {index}')
        assert live['skill_index'] == window['skill_index']
    with pytest.raises(ValueError, match='within'):  # stale buffers: newest sample too old
        tcp, cameras = buffers_until(episode, episode.span()[0] + 0.5)
        live_sample(tcp, cameras, episode.span()[0] + 1.0, request, image_shape=dataset.image_shape)


def test_conditioning_tells_arms_and_fragments_apart_and_ignores_arm_order():
    episode, _ = make_episode(0, seed=20)
    history = {arm: episode.tcp[arm].poses[:2] for arm in ARMS}
    widths = {arm: episode.tcp[arm].widths[:2] for arm in ARMS}
    fragments = [pose([-0.3, 0.07, 0.95]), pose([-0.3, -0.07, 0.95])]

    def encode(skill, arms, target=None, fragment=None):
        return encode_observation(history, widths, skill_id=skill, active_arms=arms, target_pose=target,
                                  fragment_pose=fragment)

    def inputs(sample):
        return np.concatenate([np.atleast_1d(np.asarray(sample[k], dtype=float)).ravel() for k in POLICY_INPUTS
                               if k not in ('images', 'image_mask')])
    left, right = encode('retreat', ('left',)), encode('retreat', ('right',))
    np.testing.assert_array_equal(left['arm_mask'], [1, 0])
    assert not np.allclose(inputs(left), inputs(right))
    part0, part1 = encode('pick', ('left',), fragment=fragments[0]), encode('pick', ('left',), fragment=fragments[1])
    assert part0['fragment_mask'][0] == 1 and not np.allclose(inputs(part0), inputs(part1))
    goal = pose([0.0, 0.0, 0.95])
    ordered = encode('align', ('left', 'right'), target=goal)
    np.testing.assert_array_equal(inputs(ordered), inputs(encode('align', ('right', 'left'), target=goal)))


def test_relative_target_and_fragment_use_the_active_hand_anchor():
    episodes, _ = make_episodes(1, seed=18)
    dataset = PolicyDataset(episodes)
    segment_of = lambda i: dataset.episodes[0].segments[dataset.windows[i].segment]  # noqa: E731
    place = next(i for i in range(len(dataset)) if segment_of(i).skill_id == 'place')
    sample, segment = dataset[place], segment_of(place)
    arm = segment.active_arms[0]
    for name, world in (('target', segment.target_pose), ('fragment', segment.fragment_pose)):
        relative, _ = hand_pose(np.r_[sample[name], 0.0])
        np.testing.assert_allclose(sample['anchors'][ARMS.index(arm)] @ relative, world, atol=1e-12)
        assert sample[f'{name}_mask'][0] == 1.0
    retreat = next(dataset[i] for i in range(len(dataset)) if segment_of(i).skill_id == 'retreat')
    assert retreat['target_mask'][0] == retreat['fragment_mask'][0] == 0.0 and not retreat['target'].any()


def test_trained_combinations_count_windows_per_skill_and_arms():
    episodes, _ = make_episodes(6, seed=21)
    dataset = PolicyDataset(episodes)
    combinations, counts = trained_combinations(dataset, min_windows=1)
    arms = {e.segments[0].active_arms for e in episodes}
    assert {(s, tuple(a)) for s, a in combinations} == {(s, a) for s in ('pick', 'place', 'retreat') for a in arms}
    assert sum(counts.values()) == len(dataset)
    few, _ = trained_combinations(dataset, min_windows=10 ** 6)
    assert few == []
