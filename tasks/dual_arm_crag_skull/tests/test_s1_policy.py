import math
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
torch.set_num_threads(min(4, torch.get_num_threads()))

from torch import nn  # noqa: E402

from tasks.dual_arm_crag_skull.contracts.actions import ARMS, decode_chunk  # noqa: E402
from tasks.dual_arm_crag_skull.s1.dataset import POLICY_INPUTS, Normalizer, PolicyDataset  # noqa: E402
from tasks.dual_arm_crag_skull.s1.policy import (  # noqa: E402
    EMA, ConditionalUnet1D, NoiseScheduler, ObservationEncoder, PolicyConfig, build_policy, collate,
    load_checkpoint, save_checkpoint, squaredcos_cap_v2_betas)
from tasks.dual_arm_crag_skull.s1.synthetic import make_episodes  # noqa: E402

TINY = PolicyConfig(down_dims=(16, 32), image_feature_dim=8, step_embed_dim=16, regression_hidden=(64,),
                    shared_encoder=True, random_shift=0)


@pytest.fixture(scope='module')
def data():
    episodes, privileged = make_episodes(3, seed=21, image_size=16)
    dataset = PolicyDataset(episodes)
    return dataset, Normalizer.fit(dataset)


def test_cosine_schedule_matches_closed_form_and_cap():
    betas = squaredcos_cap_v2_betas(100)
    f = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
    assert betas[0].item() == pytest.approx(1 - f(0.01) / f(0.0))
    assert betas.max().item() == pytest.approx(0.999)
    cumulative = torch.cumprod(1 - betas, 0)
    assert (cumulative[1:] < cumulative[:-1]).all() and cumulative[-1] < 1e-4
    np.testing.assert_allclose(cumulative[:50].numpy(), [f((t + 1) / 100) / f(0) for t in range(50)], rtol=1e-9)


@pytest.mark.parametrize('t', [0, 50, 99])
def test_forward_process_statistics(t):
    scheduler = NoiseScheduler()
    generator = torch.Generator().manual_seed(t)
    x0 = torch.full((200_000, 1, 1), 0.7, dtype=torch.float64)
    noise = torch.randn(x0.shape, generator=generator, dtype=torch.float64)
    xt = scheduler.add_noise(x0, noise, torch.full((len(x0),), t))
    ab = scheduler.alphas_cumprod[t].item()
    assert xt.mean().item() == pytest.approx(math.sqrt(ab) * 0.7, abs=4 * math.sqrt((1 - ab) / len(x0)) + 1e-12)
    assert xt.var().item() == pytest.approx(1 - ab, rel=0.02, abs=1e-9)


def test_trailing_inference_timesteps():
    scheduler = NoiseScheduler()
    steps = scheduler.inference_timesteps(16)
    old_formula = (np.round(np.arange(100, 0, -100 / 16)).astype(int) - 1).tolist()
    assert steps == old_formula and len(steps) == 16  # the default n=16 schedule is unchanged by the fix
    assert scheduler.inference_timesteps(100) == list(range(99, -1, -1))
    assert scheduler.inference_timesteps(1) == [99]
    with pytest.raises(ValueError):
        scheduler.inference_timesteps(0)


@pytest.mark.parametrize('train_steps', [50, 100, 1000])
def test_every_inference_step_count_gives_exactly_n_valid_timesteps(train_steps):
    scheduler = NoiseScheduler(train_steps)
    for n in range(1, train_steps + 1):
        steps = scheduler.inference_timesteps(n)
        assert len(steps) == n and min(steps) >= 0 and steps[0] == train_steps - 1, n
        assert all(a > b for a, b in zip(steps, steps[1:])), n


def _oracle(scheduler, x0):
    def eps(x, t):
        ab = scheduler.alphas_cumprod[t].to(x.dtype)
        return (x - ab.sqrt() * x0) / (1 - ab).sqrt()
    return eps


@pytest.mark.parametrize('n', [1, 4, 16, 29, 31, 100])
def test_ddim_eta0_with_oracle_noise_recovers_x0_deterministically(n):
    scheduler = NoiseScheduler()
    x0 = torch.linspace(-0.9, 0.9, 40, dtype=torch.float64).view(2, 4, 5)
    eps = _oracle(scheduler, x0)
    results = []
    for _ in range(2):
        x = torch.randn(x0.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
        steps = scheduler.inference_timesteps(n)
        for t, t_prev in zip(steps, steps[1:] + [-1]):
            x = scheduler.ddim_step(eps(x, t), t, t_prev, x, eta=0.0)
        results.append(x)
    torch.testing.assert_close(results[0], x0, atol=1e-9, rtol=0)
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)


def test_ddim_step_matches_closed_form_for_an_arbitrary_noise_prediction():
    scheduler = NoiseScheduler(clip_sample=False)
    generator = torch.Generator().manual_seed(6)
    x = torch.randn(2, 16, 20, generator=generator, dtype=torch.float64)
    eps = torch.randn(2, 16, 20, generator=generator, dtype=torch.float64)
    t, t_prev = 60, 40
    ab, ab_prev = scheduler.alphas_cumprod[t], scheduler.alphas_cumprod[t_prev]
    x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
    expected = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    torch.testing.assert_close(scheduler.ddim_step(eps, t, t_prev, x, eta=0.0), expected, atol=1e-12, rtol=0)


def test_ddim_eta1_mean_equals_ddpm_posterior_mean():
    scheduler = NoiseScheduler(clip_sample=False)
    generator = torch.Generator().manual_seed(8)
    x = torch.randn(2, 16, 20, generator=generator, dtype=torch.float64)
    eps = torch.randn(2, 16, 20, generator=generator, dtype=torch.float64)
    t = 30
    ab, ab_prev, beta = scheduler.alphas_cumprod[t], scheduler.alphas_cumprod[t - 1], scheduler.betas[t]
    sigma2 = (1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)
    assert sigma2.item() == pytest.approx((beta * (1 - ab_prev) / (1 - ab)).item(), rel=1e-12)
    x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
    ddim_mean = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma2).sqrt() * eps
    posterior = (ab_prev.sqrt() * beta / (1 - ab)) * x0 + ((1 - beta).sqrt() * (1 - ab_prev) / (1 - ab)) * x
    torch.testing.assert_close(ddim_mean, posterior, atol=1e-10, rtol=0)
    zero = torch.Generator().manual_seed(1)
    sampled = torch.stack([scheduler.ddpm_step(eps, t, x, generator=zero) for _ in range(400)])
    torch.testing.assert_close(sampled.mean(0), posterior, atol=4 * sigma2.sqrt().item() / 20 + 1e-9, rtol=0)


def test_ddpm_ancestral_with_oracle_noise_recovers_x0():
    scheduler = NoiseScheduler()
    x0 = torch.linspace(-0.9, 0.9, 40, dtype=torch.float64).view(2, 4, 5)
    eps = _oracle(scheduler, x0)
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(x0.shape, generator=generator, dtype=torch.float64)
    for t in reversed(range(100)):
        x = scheduler.ddpm_step(eps(x, t), t, x, generator=generator)
    torch.testing.assert_close(x, x0, atol=1e-9, rtol=0)


@pytest.mark.parametrize('down_dims', [(16, 32), (16, 32, 64)])
def test_unet_shapes(down_dims):
    net = ConditionalUnet1D(20, 50, 16, down_dims, kernel_size=5, n_groups=8)
    out = net(torch.randn(3, 16, 20), torch.tensor([0, 50, 99]), torch.randn(3, 50))
    assert out.shape == (3, 16, 20)


def test_encoder_uses_groupnorm_and_masks_missing_views():
    config = PolicyConfig(image_feature_dim=8, shared_encoder=False, random_shift=0)
    encoder = ObservationEncoder(config).eval()
    assert not any(isinstance(m, nn.BatchNorm2d) for m in encoder.modules())
    assert any(isinstance(m, nn.GroupNorm) for m in encoder.modules())
    images = torch.randint(0, 255, (2, 2, 3, 16, 16, 3), dtype=torch.uint8)
    mask = torch.ones(2, 2, 3, dtype=torch.bool)
    mask[0, :, 2] = False
    out = encoder(images, mask).view(2, 2, 3, 9)
    assert (out[0, :, 2] == 0).all()
    assert (out[..., -1] == mask.float()).all()
    assert out[1, :, 2, :-1].abs().sum() > 0
    shared = ObservationEncoder(PolicyConfig(image_feature_dim=8, shared_encoder=True))
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(encoder) == 3 * count(shared)


def test_collate_only_carries_policy_inputs(data):
    dataset, _ = data
    batch = collate([dataset[0], dataset[1]], 'cpu')
    assert set(batch) == set(POLICY_INPUTS) | {'action'}
    assert batch['images'].dtype == torch.uint8 and batch['skill_index'].dtype == torch.long
    with pytest.raises(ValueError, match='empty'):
        collate([], 'cpu')


def test_policy_config_rejects_unknown_keys():
    assert PolicyConfig.from_dict({'down_dims': [16, 32]}).down_dims == (16, 32)
    with pytest.raises(ValueError, match='down_dimz'):
        PolicyConfig.from_dict({'down_dimz': [16, 32]})


@pytest.mark.parametrize('eta', [-0.1, 1.5, float('nan'), True, '1'])
def test_policy_config_rejects_sampler_eta_outside_0_1(eta):
    with pytest.raises(ValueError, match='sampler_eta'):
        PolicyConfig(sampler_eta=eta)


def test_configured_sampler_eta_is_the_default_and_survives_a_checkpoint(tmp_path, data):
    dataset, normalizer = data
    assert PolicyConfig.from_dict({'down_dims': [16, 32]}).sampler_eta == 0.0  # older configs stay deterministic
    model = build_policy('diffusion', replace(TINY, sampler_eta=1.0), normalizer).eval()
    batch = collate([dataset[0]], 'cpu')

    def draw(**kwargs):
        return model.sample(batch, n_samples=2, num_inference_steps=4, generator=torch.Generator().manual_seed(5),
                            **kwargs)
    torch.testing.assert_close(draw(), draw(eta=1.0), rtol=0, atol=0)
    assert not torch.allclose(draw(), draw(eta=0.0))
    with pytest.raises(ValueError, match='eta'):
        draw(eta=1.5)
    ema = EMA(model)
    ema.update(model)
    save_checkpoint(tmp_path / 'ckpt.pt', model=model, ema=ema, config=model.config, step=1)
    assert load_checkpoint(tmp_path / 'ckpt.pt')[0].config.sampler_eta == 1.0


def test_unet_learns_conditional_denoising_of_identifiable_targets():
    torch.manual_seed(0)
    scheduler = NoiseScheduler()
    x0 = torch.rand(8, 16, 20) * 2 - 1
    cond = torch.eye(8)
    net = ConditionalUnet1D(20, 8, 32, (64, 128))
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

    def paired_loss():
        generator = torch.Generator().manual_seed(9)
        losses = []
        with torch.no_grad():
            for t in (10, 30, 60, 95):
                noise = torch.randn(x0.shape, generator=generator)
                steps = torch.full((8,), t)
                losses.append(nn.functional.mse_loss(net(scheduler.add_noise(x0, noise, steps), steps, cond), noise).item())
        return np.mean(losses)
    before = paired_loss()
    for _ in range(600):
        noise = torch.randn_like(x0)
        steps = torch.randint(0, 100, (8,))
        loss = nn.functional.mse_loss(net(scheduler.add_noise(x0, noise, steps), steps, cond), noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert paired_loss() < 0.5 * before


@pytest.mark.parametrize('kind, ratio', [('diffusion', 0.85), ('regression', 0.5)])
def test_full_policy_overfit_paired_loss_falls(kind, ratio, data):
    dataset, normalizer = data
    torch.manual_seed(0)
    config = PolicyConfig(down_dims=(32, 64), image_feature_dim=8, step_embed_dim=16, regression_hidden=(64,),
                          shared_encoder=True, random_shift=0)
    model = build_policy(kind, config, normalizer)
    batch = collate([dataset[i] for i in range(0, 16, 2)], 'cpu')

    def paired_loss():
        generator = torch.Generator().manual_seed(123)
        with torch.no_grad():
            return np.mean([model.compute_loss(batch, generator=generator).item() for _ in range(8)])
    before = paired_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(0)
    for _ in range(150):
        loss = model.compute_loss(batch, generator=generator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert paired_loss() < ratio * before


def test_gradients_reach_every_observation_path(data):
    dataset, normalizer = data
    model = build_policy('diffusion', TINY, normalizer)
    conditioned = [i for i in range(len(dataset)) if dataset[i]['target_mask'][0] and dataset[i]['fragment_mask'][0]]
    batch = collate([dataset[i] for i in conditioned[:4]], 'cpu')
    model.compute_loss(batch, generator=torch.Generator().manual_seed(1)).backward()
    for name in ('obs_encoder', 'unet'):
        grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        assert grads and sum(g.abs().sum().item() for g in grads) > 0, name
    used = int(batch['skill_index'][0])
    assert model.skill_embedding.weight.grad[used].abs().sum() > 0
    with torch.no_grad():
        a = model.global_condition(batch)
        changed = dict(batch, images=255 - batch['images'])
        b = model.global_condition(changed)
        c = model.global_condition(dict(batch, target_mask=torch.zeros_like(batch['target_mask'])))
        flipped = model.global_condition(dict(batch, arm_mask=1 - batch['arm_mask']))
        moved = model.global_condition(dict(batch, fragment=batch['fragment'] + 0.05))
        for name in ('target', 'fragment'):
            absent = dict(batch, **{f'{name}_mask': torch.zeros_like(batch[f'{name}_mask'])})
            d = model.global_condition(dict(absent, **{name: torch.randn_like(batch[name]) * 5}))
            e = model.global_condition(dict(absent, **{name: torch.zeros_like(batch[name])}))
            torch.testing.assert_close(d, e)  # a masked-out pose must not influence the condition
    assert not torch.allclose(a, b) and not torch.allclose(a, c) and not torch.allclose(a, flipped)
    assert batch['fragment_mask'].any() and not torch.allclose(a, moved)


def test_ema_warmup_and_update():
    model = nn.Linear(2, 1)
    ema = EMA(model)
    assert ema.decay(0) == 0.0 and ema.decay(1) == 0.0
    assert ema.decay(2) == pytest.approx(1 - 2 ** -0.75)
    assert ema.decay(10 ** 9) == pytest.approx(0.9999)
    ema.update(model)
    ema.update(model)
    torch.testing.assert_close(ema.averaged.weight, model.weight)
    old = ema.averaged.weight.clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    decay = ema.decay(ema.step)
    ema.update(model)
    torch.testing.assert_close(ema.averaged.weight, decay * old + (1 - decay) * model.weight)


def test_checkpoint_round_trip_reproduces_samples(tmp_path, data):
    dataset, normalizer = data
    model = build_policy('diffusion', TINY, normalizer).eval()
    ema = EMA(model)
    ema.update(model)
    path = tmp_path / 'ckpt.pt'
    save_checkpoint(path, model=model, ema=ema, config=TINY, step=1, extra={'note': 'test'})
    loaded, raw = load_checkpoint(path)
    batch = collate([dataset[0]], 'cpu')
    a = model.predict_action(batch, n_samples=2, generator=torch.Generator().manual_seed(5), num_inference_steps=4)
    b = loaded.predict_action(batch, n_samples=2, generator=torch.Generator().manual_seed(5), num_inference_steps=4)
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert raw['extra'] == {'note': 'test'} and raw['kind'] == 'diffusion'
    assert raw['layout']['skill_ids'][:2] == ['pick', 'place'] and raw['layout']['conditioning_version'] == 2
    for name in normalizer.scale:
        np.testing.assert_array_equal(loaded.normalizer.scale[name], normalizer.scale[name])


def test_checkpoint_with_another_layout_is_refused(tmp_path, data):
    _, normalizer = data
    model = build_policy('diffusion', TINY, normalizer).eval()
    path = tmp_path / 'ckpt.pt'
    save_checkpoint(path, model=model, ema=None, config=TINY, step=1)
    raw = torch.load(path, weights_only=True)
    skills = raw['layout']['skill_ids']
    raw['layout']['skill_ids'] = [skills[1], skills[0], *skills[2:]]  # a reordered SKILL_IDS
    torch.save(raw, tmp_path / 'reordered.pt')
    with pytest.raises(ValueError, match='layout'):
        load_checkpoint(tmp_path / 'reordered.pt')
    del raw['layout']
    torch.save(raw, tmp_path / 'legacy.pt')
    with pytest.raises(ValueError, match='layout'):
        load_checkpoint(tmp_path / 'legacy.pt')


def test_loading_a_checkpoint_never_reads_encoder_weight_files(tmp_path, data):
    import dataclasses
    _, normalizer = data
    weights = tmp_path / 'resnet18.pt'
    from torchvision.models import resnet18
    torch.save(resnet18(weights=None).state_dict(), weights)
    config = dataclasses.replace(TINY, encoder_weights=str(weights))
    model = build_policy('regression', config, normalizer).eval()
    save_checkpoint(tmp_path / 'ckpt.pt', model=model, ema=None, config=config, step=1)
    weights.unlink()  # the deployment machine does not have it
    loaded, _ = load_checkpoint(tmp_path / 'ckpt.pt')
    for (name, a), b in zip(model.state_dict().items(), loaded.state_dict().values()):
        torch.testing.assert_close(a, b, msg=name)


def test_predict_chunk_unnormalizes_into_a_valid_contract_chunk(data):
    dataset, normalizer = data
    model = build_policy('diffusion', TINY, normalizer).eval()
    sample = dataset[3]
    truth = torch.tensor(normalizer.normalize('action', sample['action']), dtype=torch.float32)
    model.sample = lambda batch, n_samples=1, **kw: truth.view(1, 1, 16, 20).expand(1, n_samples, 16, 20)
    chunk, info = model.predict_chunk(sample, request_id='r', chunk_id=2, active_arms=('left',),
                                      held_widths={'right': 0.031})
    assert info['held_arms'] == ('right',)
    left = decode_chunk(chunk)['left'][0]
    relative = np.linalg.inv(sample['anchors'][ARMS.index('left')]) @ left
    np.testing.assert_allclose(relative[:, :3, 3], sample['action'][:, :3], atol=1e-5)
    np.testing.assert_allclose(decode_chunk(chunk)['right'][1], 0.031)
