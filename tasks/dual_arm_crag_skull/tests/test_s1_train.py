import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
torch.set_num_threads(min(4, torch.get_num_threads()))

from tasks.dual_arm_crag_skull.contracts.actions import ARMS  # noqa: E402
from tasks.dual_arm_crag_skull.s1 import smoke, train  # noqa: E402
from tasks.dual_arm_crag_skull.s1.dataset import Normalizer, PolicyDataset  # noqa: E402
from tasks.dual_arm_crag_skull.s1.episode import buffers_until, save_episode  # noqa: E402
from tasks.dual_arm_crag_skull.s1.policy import load_checkpoint  # noqa: E402
from tasks.dual_arm_crag_skull.s1.synthetic import make_episode, make_episodes  # noqa: E402

TINY = {'steps': 4, 'batch_size': 4, 'log_every': 2, 'eval_every': 2, 'image_size': 16, 'split': [0.5, 0.25, 0.25],
        'eval_samples': 2, 'latency_steps': [1, 2], 'latency_ddpm': False, 'latency_repeats': 1,
        'min_windows_per_combination': 8,
        'policy': {'down_dims': [16, 32], 'image_feature_dim': 8, 'step_embed_dim': 16, 'shared_encoder': True,
                   'regression_hidden': [32]}}


def write_config(path, **overrides):
    config = {**TINY, **overrides, 'policy': {**TINY['policy'], **overrides.get('policy', {})}}
    path.write_text(json.dumps(config))
    return path


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp('s1runs')
    config = write_config(root / 'tiny.json')
    result = {}
    for kind in ('diffusion', 'regression'):
        output = root / kind
        train.main(['--synthetic', '8', '--output', str(output), '--config', str(config), '--kind', kind, '--cpu'])
        result[kind] = output
    return result


def test_training_writes_reloadable_checkpoint_and_metrics(runs):
    output = runs['diffusion']
    metrics = json.loads((output / 'metrics.json').read_text())
    assert metrics['checkpoint']['reload_max_abs_diff'] <= 1e-6
    assert metrics['data']['episodes'] == {'train': 4, 'val': 2, 'test': 2}
    assert metrics['data']['excluded'] == {'episodes_given': 8, 'excluded_failed': 0, 'excluded_unknown_outcome': 0}
    assert [s for s, _ in metrics['training']['train_loss_log']] == [2, 4]
    assert set(metrics['latency_batch1']) == {'ddim_1', 'ddim_2'}
    assert set(metrics['multimodal']['test']) == {'pick_first', 'place_first', 'place_mid', 'place_late', 'arms'}
    assert metrics['code_sha256'] == train.code_sha256()
    check = metrics['inference_check']
    assert check['attempted'] == 2 * 3 * 2  # held-out episodes x moving segments x (first, last window)
    assert check['chunks'] + len(check['rejected']) == check['attempted']
    model, raw = load_checkpoint(output / 'checkpoint.pt')
    assert raw['ema_step'] == raw['step'] == TINY['steps']
    assert raw['extra']['trained_combinations'] == metrics['data']['trained_combinations']
    assert raw['extra']['trained_skills'] == sorted({s for s, _ in raw['extra']['trained_combinations']})
    trained, _ = load_checkpoint(output / 'checkpoint.pt', use_ema=False)
    differs = [not torch.equal(a, b) for a, b in zip(model.state_dict().values(), trained.state_dict().values())]
    assert any(differs)  # EMA weights are an average, not a copy of the last iterate
    saved = Normalizer.load(output / 'normalizer.json')
    for name in saved.scale:
        np.testing.assert_array_equal(model.normalizer.scale[name], saved.scale[name])
    assert set(saved.fitted_on) == set(raw['extra']['split']['train'])
    assert not set(saved.fitted_on) & set(raw['extra']['split']['test'])


def test_inference_check_records_rejections_and_stale_chunks():
    from tasks.dual_arm_crag_skull.contracts.actions import ActionRejected
    from tasks.dual_arm_crag_skull.s1.chunking import chunk_from_prediction
    episodes, _ = make_episodes(2, seed=31, image_size=16)
    dataset = PolicyDataset(episodes)

    class Stub:
        def __init__(self, stale_after_s, reject=False):
            self.stale_after_s, self.reject = stale_after_s, reject

        def predict_chunk(self, sample, *, request_id, chunk_id, active_arms, held_widths, generator=None):
            if self.reject:
                raise ActionRejected('near-collinear rotation6 columns')
            truth = dataset[0]['action']  # any valid relative trajectory
            return chunk_from_prediction(truth, anchors=sample['anchors'], held_widths=held_widths,
                                         observation_time=sample['observation_time'], request_id=request_id,
                                         chunk_id=chunk_id, active_arms=active_arms, stale_after_s=self.stale_after_s)
    fresh = train.inference_check(Stub(5.0), dataset, torch.device('cpu'))
    assert fresh['attempted'] == fresh['chunks'] == 12 and not fresh['not_executable'] and not fresh['rejected']
    stale = train.inference_check(Stub(1e-12), dataset, torch.device('cpu'))
    assert len(stale['not_executable']) == 12 and 'stale' in stale['not_executable'][0]
    rejected = train.inference_check(Stub(5.0, reject=True), dataset, torch.device('cpu'))
    assert rejected['chunks'] == 0 and len(rejected['rejected']) == 12 and rejected['latency_ms_median'] is None


def _saved_episodes(folder, count, seed, failed=()):
    for index in range(count):
        episode, privileged = make_episode(index, seed=seed, image_size=16)
        if index in failed:
            from tasks.dual_arm_crag_skull.s1.episode import Episode
            episode = Episode(**{**episode.__dict__, 'success': False, 'failure_reason': 'dropped the part'})
        saved = save_episode(episode, folder / episode.episode_id, privileged=privileged)
        (saved / 'privileged.npz').write_bytes(b'corrupt on purpose: must never be opened')
    return folder


def test_training_from_saved_episodes_never_reads_privileged_truth(tmp_path):
    folder = _saved_episodes(tmp_path / 'episodes', 8, seed=41)
    config = write_config(tmp_path / 'tiny.json', min_windows_per_combination=1)
    train.main(['--episodes', str(folder), '--output', str(tmp_path / 'run'), '--config', str(config), '--cpu'])
    metrics = json.loads((tmp_path / 'run' / 'metrics.json').read_text())
    assert metrics['data']['source'] == 'episodes' and 'multimodal' not in metrics
    assert sum(metrics['data']['episodes'].values()) == 8


def test_failed_demonstrations_are_excluded_unless_requested(tmp_path):
    folder = _saved_episodes(tmp_path / 'episodes', 10, seed=42, failed=(1, 4))
    config = write_config(tmp_path / 'tiny.json', min_windows_per_combination=1)
    train.main(['--episodes', str(folder), '--output', str(tmp_path / 'clean'), '--config', str(config), '--cpu'])
    clean = json.loads((tmp_path / 'clean' / 'metrics.json').read_text())
    assert clean['data']['excluded']['excluded_failed'] == 2 and sum(clean['data']['episodes'].values()) == 8
    raw = torch.load(tmp_path / 'clean' / 'checkpoint.pt', weights_only=True)
    assert not any('00001' in i or '00004' in i for ids in raw['extra']['split'].values() for i in ids)
    opt_in = write_config(tmp_path / 'all.json', min_windows_per_combination=1, include_failures=True)
    train.main(['--episodes', str(folder), '--output', str(tmp_path / 'all'), '--config', str(opt_in), '--cpu'])
    assert sum(json.loads((tmp_path / 'all' / 'metrics.json').read_text())['data']['episodes'].values()) == 10


def test_dataset_follows_the_policy_horizon(tmp_path):
    config = write_config(tmp_path / 'short.json', policy={'pred_steps': 8})
    train.main(['--synthetic', '8', '--output', str(tmp_path / 'run'), '--config', str(config), '--cpu',
                '--kind', 'regression'])
    model, raw = load_checkpoint(tmp_path / 'run' / 'checkpoint.pt')
    assert raw['layout']['pred_steps'] == 8 and model.config.pred_steps == 8
    episodes, _ = make_episodes(2, seed=0, image_size=16)
    sample = PolicyDataset(episodes, pred_steps=8)[0]
    assert sample['action'].shape == (8, 20)
    from tasks.dual_arm_crag_skull.s1.policy import collate
    assert model.predict_action(collate([sample], 'cpu')).shape == (1, 1, 8, 20)


def test_runtime_uses_past_only_buffers_commanded_widths_and_refuses_untrained_combinations(runs):
    from tasks.dual_arm_crag_skull.contracts.actions import ActionRejected, decode_chunk, executable_reason
    from tasks.dual_arm_crag_skull.contracts.skills import SkillRequest
    from tasks.dual_arm_crag_skull.s1.runtime import S1Runtime
    runtime = S1Runtime(runs['diffusion'] / 'checkpoint.pt')
    assert runtime.image_shape == (16, 16, 3) and runtime.trained_skills
    skill, arms = sorted(runtime.trained_combinations)[0]
    episode = next(e for e in make_episodes(12, seed=7, image_size=16)[0] if e.segments[0].active_arms == arms)
    segment = next(s for s in episode.segments if s.skill_id == skill)
    completion = {'pick': {'type': 'grasp_held'}, 'retreat': {'type': 'retreated', 'clearance_m': 0.1}}.get(
        skill, {'type': 'pose_within', 'position_tolerance_m': .003, 'rotation_tolerance_deg': 3.0})
    request = SkillRequest(request_id='r1', observation_id='o', skill_id=skill, active_arms=arms,
                           fragment_id=segment.fragment_id, target_pose=segment.target_pose,
                           target_provenance=segment.target_provenance, fragment_pose=segment.fragment_pose,
                           fragment_pose_provenance=segment.fragment_pose_provenance, timeout_s=10.0,
                           completion_predicate=completion)
    now = segment.start + 0.3
    tcp, cameras = buffers_until(episode, now)
    held = ARMS[1 - ARMS.index(arms[0])]
    dataset = PolicyDataset([episode])
    window = next(dataset[i] for i, r in enumerate(dataset.windows) if abs(dataset._cache[0]['grid'][r.step] - now) < 0.051)
    truth = torch.tensor(runtime.model.normalizer.normalize('action', window['action']), dtype=torch.float32)
    runtime.model.sample = lambda batch, n_samples=1, **kw: truth.view(1, 1, 16, 20).expand(1, n_samples, 16, 20)
    chunk, info = runtime.chunk(request, tcp, cameras, commanded_widths={held: 0.021})
    newest = min(float(tcp[arm].times[-1]) for arm in ARMS)
    assert chunk.observation_time == newest <= now and info['held_arms'] == (held,)
    np.testing.assert_allclose(decode_chunk(chunk)[held][1], 0.021)
    assert executable_reason(chunk, newest + 0.01) is None
    # The chunk declares where the skill works near the table, taken from the request (S0 contact phase).
    (contact,) = chunk.contacts
    centre = segment.fragment_pose if skill in ('pick', 'support') else segment.target_pose
    assert skill != 'retreat' and contact.arm == arms[0]
    np.testing.assert_allclose(contact.center_xy, np.asarray(centre)[:2, 3])
    # A retreat keeps the arm's window only once the executor has reported a chunk S0 accepted with it.
    retreat = SkillRequest(request_id='r2', observation_id='o', skill_id='retreat', active_arms=arms, timeout_s=10.0,
                           completion_predicate={'type': 'retreated', 'clearance_m': 0.1})
    runtime.trained_combinations = runtime.trained_combinations | {('retreat', arms)}
    assert runtime.chunk(retreat, tcp, cameras, commanded_widths={held: 0.021})[0].contacts == ()
    runtime.contact_memory.accepted(chunk)
    first, _ = runtime.chunk(retreat, tcp, cameras, commanded_widths={held: 0.021}, observation_time=newest - 0.1)
    later, _ = runtime.chunk(retreat, tcp, cameras, commanded_widths={held: 0.021}, chunk_id=1)
    assert first.contacts == later.contacts == chunk.contacts
    with pytest.raises(ValueError, match='after the newest TCP sample'):
        runtime.chunk(request, tcp, cameras, commanded_widths={held: 0.021}, observation_time=now + 0.2)
    runtime.model.sample = lambda batch, n_samples=1, **kw: torch.zeros(1, n_samples, 16, 20)
    with pytest.raises(ActionRejected):  # all-zero rotation6 is degenerate: rejected, never executed
        runtime.chunk(request, tcp, cameras, commanded_widths={held: 0.021})
    untrained = SkillRequest(**{**request.__dict__, 'active_arms': ('left', 'right')})
    with pytest.raises(ValueError, match='not trained'):
        runtime.chunk(untrained, tcp, cameras, commanded_widths={})


def test_training_refuses_to_overwrite(runs):
    with pytest.raises(FileExistsError):
        train.main(['--synthetic', '4', '--output', str(runs['diffusion']), '--cpu'])


def test_unknown_config_keys_rejected(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps({'stepz': 3}))
    with pytest.raises(ValueError, match='unknown config keys'):
        train.load_config(path)
    path.write_text(json.dumps({'policy': {'down_dimz': [8, 16]}}))
    with pytest.raises(ValueError, match='unknown policy config keys'):
        train.load_config(path)


def test_same_seed_reproduces_the_loss_log(runs, tmp_path):
    config = write_config(tmp_path / 'tiny.json')
    train.main(['--synthetic', '8', '--output', str(tmp_path / 'again'), '--config', str(config), '--kind', 'diffusion',
                '--cpu'])
    first = json.loads((runs['diffusion'] / 'metrics.json').read_text())['training']['train_loss_log']
    again = json.loads((tmp_path / 'again' / 'metrics.json').read_text())['training']['train_loss_log']
    np.testing.assert_allclose(first, again, rtol=1e-6)


def test_smoke_verdict_applies_all_criteria(runs):
    diffusion = json.loads((runs['diffusion'] / 'metrics.json').read_text())
    regression = json.loads((runs['regression'] / 'metrics.json').read_text())
    place = {'obstacle_hit_rate': 0.05, 'left_fraction_of_clear': 0.5, 'episodes_with_both_modes': 0.9,
             'mode_agreement_of_clear': 0.5, 'final_error_m_median': 0.01, 'final_width_m_median': 0.05}
    diffusion['multimodal']['test'] = {
        'place_first': place, 'place_mid': {**place, 'mode_agreement_of_clear': 0.95},
        'place_late': {**place, 'final_width_m_median': 0.078}, 'pick_first': {'chosen_fragment_rate': 0.95},
        'arms': {'active_motion_m_median': 0.12, 'inactive_motion_m_median': 0.004}}
    regression['multimodal']['test'] = {'place_first': {**place, 'obstacle_hit_rate': 0.4}}
    diffusion['latency_batch1']['ddim_16'] = {'median_ms': 20.0}
    diffusion['inference_check'] = {'attempted': 4, 'chunks': 4, 'rejected': [], 'not_executable': []}
    junit = {'tests': 5, 'failures': 0, 'errors': 0, 'skipped': 0}
    assert smoke.verdict(diffusion, regression, junit, 0, 5)['pass']

    def broken(path, value, which=diffusion):
        copy = json.loads(json.dumps(which))
        node = copy
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return copy
    test = ('multimodal', 'test')
    cases = {
        'C3_diffusion_modes': broken(test + ('place_first', 'left_fraction_of_clear'), 0.97),
        'C6_inference_chunks': broken(('inference_check', 'not_executable'), ['r: stale chunk']),
        'C7_pick_conditioned_on_fragment': broken(test + ('pick_first', 'chosen_fragment_rate'), 0.6),
        'C8_mid_window_follows_mode': broken(test + ('place_mid', 'mode_agreement_of_clear'), 0.7),
        'C9_late_window_releases_at_goal': broken(test + ('place_late', 'final_width_m_median'), 0.05),
        'C10_only_requested_arm_moves': broken(test + ('arms', 'inactive_motion_m_median'), 0.03),
    }
    for check, metrics in cases.items():
        result = smoke.verdict(metrics, regression, junit, 0, 5)
        assert not result['checks'][check] and not result['pass'], check
    assert not smoke.verdict(diffusion, broken(test + ('place_first', 'obstacle_hit_rate'), 0.1, regression),
                             junit, 0, 5)['checks']['C4_regression_averages_modes']
    assert not smoke.verdict(diffusion, regression, {**junit, 'skipped': 1}, 0, 5)['pass']
    assert not smoke.verdict(diffusion, regression, junit, 0, 6)['pass']
