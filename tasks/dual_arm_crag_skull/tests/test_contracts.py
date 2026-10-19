import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tasks.dual_arm_crag_skull.contracts.actions import (
    ACTION_DIM, ARMS, ActionChunk, ActionRejected, S0Status, decode_chunk, encode_chunk,
    executable_reason, hand_slice, matrix_from_rotation6, rotation6_from_matrix)
from tasks.dual_arm_crag_skull.contracts.skills import (
    SKILL_IDS, SkillRejected, SkillRequest, trained_combination_set, validate_skill_request)


def pose(translation, rotvec=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    t[:3, 3] = translation
    return t


ANCHORS = {'left': pose([.1, .2, .9], [0, .3, 0]), 'right': pose([-.1, -.2, .9], [.2, 0, 0])}


def chunk(**overrides):
    rng = np.random.default_rng(0)
    targets = {arm: np.stack([ANCHORS[arm] @ pose(rng.normal(scale=.02, size=3), rng.normal(scale=.2, size=3))
                              for _ in range(4)]) for arm in ARMS}
    targets['right'] = np.stack([ANCHORS['right']] * 4)
    widths = {'left': np.array([.04, .03, .02, .01]), 'right': np.full(4, .05)}
    args = dict(request_id='req-1', chunk_id=0, observation_time=1.0, valid_until=1.3, anchors=ANCHORS,
                waypoint_times=1.1 + .1 * np.arange(4), targets=targets, widths=widths, active_arms=('left',))
    args.update(overrides)
    return encode_chunk(**args), targets, widths


def test_rotation6_round_trip_uses_first_two_columns():
    rotation = Rotation.random(5, random_state=1).as_matrix()
    values = rotation6_from_matrix(rotation)
    np.testing.assert_allclose(values[:, :3], rotation[:, :, 0])
    np.testing.assert_allclose(values[:, 3:], rotation[:, :, 1])
    np.testing.assert_allclose(matrix_from_rotation6(values), rotation, atol=1e-12)


def test_rotation6_orthonormalizes_unnormalized_network_output():
    raw = np.array([2.0, 0.1, 0.0, 0.3, 1.5, 0.2])
    rotation = matrix_from_rotation6(raw)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1)
    np.testing.assert_allclose(rotation[:, 0], raw[:3] / np.linalg.norm(raw[:3]))


@pytest.mark.parametrize('bad', [[0, 0, 0, 0, 1, 0], [1, 0, 0, 2, 1e-5, 0], [np.nan, 0, 0, 0, 1, 0]])
def test_rotation6_rejects_degenerate_input(bad):
    with pytest.raises(ActionRejected):
        matrix_from_rotation6(bad)


def test_encode_decode_recovers_world_targets_and_widths():
    action, targets, widths = chunk()
    assert action.waypoints.shape == (4, ACTION_DIM)
    decoded = decode_chunk(action)
    for arm in ARMS:
        np.testing.assert_allclose(decoded[arm][0], targets[arm], atol=1e-12)
        np.testing.assert_allclose(decoded[arm][1], widths[arm])
    np.testing.assert_allclose(action.waypoints[0, hand_slice('right')][:3], 0, atol=1e-12)
    assert executable_reason(action, now=1.05) is None


def test_chunk_arrays_are_read_only_copies():
    action, _, _ = chunk()
    with pytest.raises(ValueError):
        action.waypoints[0, 0] = 1.0
    with pytest.raises(ValueError):
        action.anchors['left'][0, 3] = 1.0
    with pytest.raises(TypeError):
        action.anchors['left'] = np.eye(4)


def test_encode_rejects_non_rigid_targets():
    _, targets, widths = chunk()
    scaled = {**targets, 'left': targets['left'].copy()}
    scaled['left'][1, :3, :3] *= 1.1
    with pytest.raises(ActionRejected, match='left target'):
        chunk(targets=scaled)


@pytest.mark.parametrize('overrides', [
    {'waypoint_times': np.array([1.1, 1.1, 1.2, 1.3])},
    {'waypoint_times': np.array([0.9, 1.1, 1.2, 1.3])},
    {'valid_until': 0.5},
    {'active_arms': ('left', 'left')},
    {'request_id': ''},
])
def test_invalid_chunks_are_rejected(overrides):
    with pytest.raises(ActionRejected):
        chunk(**overrides)


def test_negative_width_and_missing_anchor_rejected():
    action, _, _ = chunk()
    waypoints = action.waypoints.copy()
    waypoints[0, 9] = -.001
    with pytest.raises(ActionRejected):
        ActionChunk('r', 0, 1.0, 1.3, ANCHORS, action.waypoint_times, waypoints, ('left',))
    with pytest.raises(ActionRejected):
        ActionChunk('r', 0, 1.0, 1.3, {'left': ANCHORS['left']}, action.waypoint_times, action.waypoints, ('left',))


def test_executable_reason_catches_staleness_and_moving_inactive_arm():
    action, targets, widths = chunk()
    assert 'stale' in executable_reason(action, now=1.31)
    assert 'earlier than observation_time' in executable_reason(action, now=0.5)
    assert 'old' in executable_reason(action, now=1.2, max_observation_age_s=0.15)
    assert executable_reason(action, now=1.1, max_observation_age_s=0.15) is None
    moved = dict(targets)
    moved['right'] = targets['left']
    action, _, _ = chunk(targets=moved)
    assert 'inactive arm moves: right' == executable_reason(action, now=1.05)


def test_s0_status_requires_reason_for_rejection():
    with pytest.raises(ValueError):
        S0Status('r', 0, 'rejected', 1.0)
    assert S0Status('r', 0, 'accepted', 1.0).reason is None


def request(**overrides):
    args = dict(request_id='s-1', observation_id='obs-3', skill_id='place', active_arms=('left',),
                fragment_id='part_1', target_pose=pose([.3, 0, .8]),
                target_provenance={'source': 'garf_candidate', 'observation_id': 'obs-3', 'candidate_id': 'c-42',
                                   'checkpoint_sha256': 'cc65', 'geometry_status': 'needs_motion_validation',
                                   'fragment_id': 'part_1'},
                completion_predicate={'type': 'released_stable', 'position_tolerance_m': .003,
                                      'rotation_tolerance_deg': 3.0, 'settle_s': 1.0},
                timeout_s=20.0)
    args.update(overrides)
    return SkillRequest(**args)


def validate(item, trained=('pick', 'place')):
    return validate_skill_request(item, trained_skills=trained, known_fragments=('part_0', 'part_1'),
                                  current_observation_id='obs-3')


def test_valid_place_request_passes_and_serializes():
    item = validate(request())
    record = item.to_record()
    assert record['target_pose'][0][3] == pytest.approx(.3)
    assert SKILL_IDS.index('place') == 1


@pytest.mark.parametrize('overrides, message', [
    ({'observation_id': 'obs-2'}, 'stale observation'),
    ({'skill_id': 'insert'}, 'no trained S1 policy'),
    ({'skill_id': 'teleport'}, 'unknown skill'),
    ({'active_arms': ('left', 'right')}, 'distinct arms'),
    ({'fragment_id': 'part_9'}, 'unknown fragment'),
    ({'support_fragment_id': 'part_1'}, 'different known fragment'),
    ({'skill_id': 'pick', 'target_pose': None, 'target_provenance': None, 'completion_predicate': {'type': 'grasp_held'},
      'support_fragment_id': 'part_0'}, 'takes no support fragment'),
    ({'target_pose': None}, 'needs a target pose'),
    ({'target_pose': np.diag([2., 1, 1, 1])}, 'target pose'),
    ({'timeout_s': 0}, 'timeout_s'),
    ({'completion_predicate': {'type': 'grasp_held'}}, 'completion must be one of'),
    ({'completion_predicate': {'type': 'pose_within', 'position_tolerance_m': .003}}, 'needs exactly'),
])
def test_invalid_requests_are_rejected(overrides, message):
    with pytest.raises(SkillRejected, match=message):
        validate(request(**overrides))


def test_garf_target_requires_screen_and_same_observation():
    provenance = dict(request().target_provenance)
    for key, value, message in [('geometry_status', 'all_candidates_rejected', 'geometry screen'),
                                ('observation_id', 'obs-1', 'different observation'),
                                ('candidate_id', '', 'candidate_id'),
                                ('fragment_id', 'part_0', 'different fragment')]:
        with pytest.raises(SkillRejected, match=message):
            validate(request(target_provenance={**provenance, key: value}))


def test_reobserve_needs_no_trained_policy_and_no_arms():
    item = request(skill_id='reobserve', active_arms=(), fragment_id=None, target_pose=None,
                   target_provenance=None, completion_predicate={'type': 'observed'})
    assert validate(item, trained=()).skill_id == 'reobserve'


def test_request_fields_are_immutable():
    item = request()
    with pytest.raises(TypeError):
        item.completion_predicate['type'] = 'observed'
    with pytest.raises(ValueError):
        item.target_pose[0, 3] = 9.0


def observed(observation_id='obs-3', **overrides):
    return {'source': 'observed_geometry', 'observation_id': observation_id, 'pose_source': 'icp', **overrides}


def test_pick_needs_an_observed_fragment_pose():
    pick = dict(skill_id='pick', target_pose=None, target_provenance=None, completion_predicate={'type': 'grasp_held'})
    with pytest.raises(SkillRejected, match='needs the observed pose'):
        validate(request(**pick))
    assert validate(request(**pick, fragment_pose=pose([.3, -.2, .78]), fragment_pose_provenance=observed())).fragment_id
    for provenance, message in [(observed('obs-2'), 'different observation'),
                                (observed(source='garf_candidate'), 'observed geometry'),
                                (observed(pose_source='guess'), 'known pose source'),
                                (None, 'observed geometry')]:
        with pytest.raises(SkillRejected, match=message):
            validate(request(**pick, fragment_pose=pose([.3, -.2, .78]), fragment_pose_provenance=provenance))
    with pytest.raises(SkillRejected, match='fragment pose'):
        validate(request(**pick, fragment_pose=np.diag([2., 1, 1, 1]), fragment_pose_provenance=observed()))


def test_skills_without_fragment_take_no_fragment_pose():
    retreat = dict(skill_id='retreat', fragment_id=None, target_pose=None, target_provenance=None,
                   completion_predicate={'type': 'retreated', 'clearance_m': 0.1})
    with pytest.raises(SkillRejected, match='takes no fragment pose'):
        validate(request(**retreat, fragment_pose=pose([0, 0, 1]), fragment_pose_provenance=observed()),
                 trained=('retreat',))


def test_trained_combinations_restrict_arms_in_canonical_order():
    combinations = trained_combination_set([('place', ('left',)), ('align', ('left', 'right'))],
                                           trained_skills=('place', 'align'))

    def check(item):
        return validate_skill_request(item, trained_skills=('place', 'align'), known_fragments=('part_0', 'part_1'),
                                      current_observation_id='obs-3', trained_combinations=combinations)
    assert check(request()).active_arms == ('left',)
    with pytest.raises(SkillRejected, match=r"place with arms \['right'\] has no trained S1 policy"):
        check(request(active_arms=('right',)))
    align = request(skill_id='align', active_arms=('right', 'left'),
                    completion_predicate={'type': 'pose_within', 'position_tolerance_m': .003,
                                          'rotation_tolerance_deg': 3.0})
    assert check(align).skill_id == 'align'
    reobserve = request(skill_id='reobserve', active_arms=(), fragment_id=None, target_pose=None,
                        target_provenance=None, completion_predicate={'type': 'observed'})
    assert check(reobserve).skill_id == 'reobserve'
    as_stored = [['place', ['left']]]  # checkpoint JSON: lists, not tuples
    assert validate_skill_request(request(), trained_skills=('place',), known_fragments=('part_0', 'part_1'),
                                  current_observation_id='obs-3', trained_combinations=as_stored).skill_id == 'place'


@pytest.mark.parametrize('combinations', [
    [('align', ('right', 'left'))], [('align', ('left', 'left'))], [('pick', ('left', 'right'))],
    [('reobserve', ())], [('teleport', ('left',))],
])
def test_trained_combination_set_rejects_invalid_entries(combinations):
    with pytest.raises(ValueError, match='invalid trained combination'):
        trained_combination_set(combinations, trained_skills={skill for skill, _ in combinations})


def test_trained_skills_must_be_the_skills_of_the_combinations():
    with pytest.raises(ValueError, match='trained_skills'):
        trained_combination_set([('pick', ('left',))], trained_skills=('pick', 'place'))
