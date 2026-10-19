"""Train S1 (diffusion policy or the MSE regression baseline). Writes checkpoint.pt, normalizer.json and
metrics.json into a new output directory.

    python -m tasks.dual_arm_crag_skull.s1.train --output RUN --synthetic 300 [--config cfg.json]
    python -m tasks.dual_arm_crag_skull.s1.train --output RUN --episodes DIR_WITH_EPISODE_DIRS

Splits are by initial condition; the normalizer is fit on the training split only; only successful
demonstrations are used unless `include_failures`. Synthetic runs also report the multimodal
metrics on held-out episodes, which read privileged ground truth for evaluation only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import synthetic
from .dataset import Normalizer, PolicyDataset, live_sample, split_episodes, trained_combinations
from .episode import buffers_until, causal_state, load_episode
from .policy import EMA, PolicyConfig, build_policy, collate, load_checkpoint, save_checkpoint

DEFAULTS = {
    'kind': 'diffusion', 'seed': 0, 'steps': 8000, 'batch_size': 256, 'lr': 1e-4, 'weight_decay': 1e-6,
    'adam_betas': [0.95, 0.999], 'warmup_steps': 500, 'log_every': 100, 'eval_every': 1000,
    'supervision': 'achieved', 'split': [0.8, 0.1, 0.1], 'split_seed': 0,
    'include_failures': False, 'allow_oracle': False, 'pad_after': None, 'min_windows_per_combination': 16,
    'synthetic_seed': 0, 'image_size': 32, 'eval_samples': 8,
    'latency_steps': [1, 2, 4, 8, 16], 'latency_ddpm': True, 'latency_repeats': 20,
    'policy': {'down_dims': [128, 256, 512], 'image_feature_dim': 64},
}
SOURCE_FILES = ('s1/__init__.py', 's1/episode.py', 's1/dataset.py', 's1/chunking.py', 's1/policy.py', 's1/synthetic.py',
                's1/train.py', 's1/runtime.py',
                'contracts/__init__.py', 'contracts/actions.py', 'contracts/skills.py', 'perception.py')


def code_sha256():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def versions(device):
    import scipy
    import torchvision
    info = {'python': platform.python_version(), 'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'numpy': np.__version__, 'scipy': scipy.__version__, 'device': str(device), 'host': platform.node()}
    if device.type == 'cuda':
        info.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version())
    return info


def load_config(path):
    config = json.loads(json.dumps(DEFAULTS))
    if path:
        override = json.loads(Path(path).read_text())
        policy = {**config['policy'], **override.pop('policy', {})}
        unknown = set(override) - set(config)
        if unknown:
            raise ValueError(f'unknown config keys: {sorted(unknown)}')
        config.update(override)
        config['policy'] = policy
    PolicyConfig.from_dict(config['policy'])  # rejects unknown policy keys before any work
    return config


def episodes_from(args, config):
    if args.synthetic:
        episodes, privileged = synthetic.make_episodes(args.synthetic, seed=config['synthetic_seed'],
                                                       image_size=config['image_size'])
        return episodes, privileged, 'synthetic'
    folders = sorted(p for p in Path(args.episodes).iterdir() if (p / 'meta.json').exists())
    if not folders:
        raise ValueError(f'no episode directories under {args.episodes}')
    return [load_episode(p) for p in folders], None, 'episodes'


def build_datasets(config, policy_config, episodes):
    """Split by initial condition, then one PolicyDataset per non-empty split, shaped by the policy config."""
    usable = [e for e in episodes if config['include_failures'] or e.success is True]
    split = split_episodes(usable, fractions=config['split'], seed=config['split_seed'])
    by_id = {e.episode_id: e for e in usable}
    datasets = {}
    for name, ids in split.items():
        if not ids:
            continue
        dataset = PolicyDataset([by_id[i] for i in ids], supervision=config['supervision'],
                                obs_steps=policy_config.obs_steps, pred_steps=policy_config.pred_steps,
                                pad_after=config['pad_after'], cameras=policy_config.cameras,
                                include_failures=config['include_failures'], allow_oracle=config['allow_oracle'])
        if len(dataset):
            datasets[name] = dataset
    if 'train' not in datasets:
        raise ValueError('the training split has no windows')
    excluded = {'episodes_given': len(episodes), 'excluded_failed': sum(e.success is False for e in episodes),
                'excluded_unknown_outcome': sum(e.success is None for e in episodes)} \
        if not config['include_failures'] else {'episodes_given': len(episodes)}
    return split, datasets, excluded


def stack(dataset, device):
    return collate([dataset[i] for i in range(len(dataset))], device)


def lr_lambda(config):
    warmup, total = config['warmup_steps'], config['steps']

    def schedule(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return schedule


def batch_at(tensors, indices):
    return {key: value[indices] for key, value in tensors.items()}


@torch.no_grad()
def dataset_loss(model, tensors, device, batch_size=256, seed=1234):
    """Mean loss over a whole split with a fixed generator, so evaluations are paired across time."""
    generator = torch.Generator(device=device).manual_seed(seed)
    count = len(tensors['action'])
    total = 0.0
    for start in range(0, count, batch_size):
        indices = torch.arange(start, min(count, start + batch_size), device=device)
        total += model.compute_loss(batch_at(tensors, indices), generator=generator).item() * len(indices)
    return total / max(1, count)


def fit(model, ema, tensors, config, device):
    """Train in place; returns (train loss log, val loss log, seconds)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=tuple(config['adam_betas']),
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(config))
    index_generator = torch.Generator().manual_seed(config['seed'])
    noise_generator = torch.Generator(device=device).manual_seed(config['seed'] + 1)
    count = len(tensors['train']['action'])
    log, val_log, running = [], [], []
    started = time.perf_counter()
    model.train()
    for step in range(1, config['steps'] + 1):
        indices = torch.randint(0, count, (config['batch_size'],), generator=index_generator).to(device)
        loss = model.compute_loss(batch_at(tensors['train'], indices), generator=noise_generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)
        running.append(loss.item())
        if step % config['log_every'] == 0 or step == config['steps']:
            log.append([step, float(np.mean(running))])
            running = []
        if 'val' in tensors and (step % config['eval_every'] == 0 or step == config['steps']):
            val_log.append([step, dataset_loss(ema.averaged, tensors['val'], device)])
            print(json.dumps({'step': step, 'train_loss': log[-1][1], 'val_loss_ema': val_log[-1][1]}), flush=True)
    return log, val_log, time.perf_counter() - started


def make_predictor(model, device, num_inference_steps, seed=99):
    def predict(samples, n_samples):
        generator = torch.Generator(device=device).manual_seed(seed)
        outputs = []
        for start in range(0, len(samples), 64):
            batch = collate(samples[start:start + 64], device)
            kwargs = {'num_inference_steps': num_inference_steps} if model.kind == 'diffusion' else {}
            outputs.append(model.predict_action(batch, n_samples=n_samples, generator=generator, **kwargs))
        return np.concatenate(outputs)
    return predict


@torch.no_grad()
def latency(model, sample, device, config):
    batch = collate([sample], device)

    def timed(**kwargs):
        times = []
        for repeat in range(config['latency_repeats'] + 3):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model.sample(batch, n_samples=1, **kwargs)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            if repeat >= 3:
                times.append((time.perf_counter() - start) * 1000)
        return {'median_ms': float(np.median(times)), 'p90_ms': float(np.percentile(times, 90))}
    if model.kind == 'regression':
        return {'forward': timed()}
    result = {f'ddim_{n}': timed(num_inference_steps=n) for n in config['latency_steps']}
    if config['latency_ddpm']:
        result['ddpm_100'] = timed(method='ddpm')
    return result


def _request_for(segment):
    from ..contracts.skills import COMPLETIONS, SKILLS, SkillRequest
    kind = SKILLS[segment.skill_id].completions[0]
    completion = {'type': kind, **{name: high / 2 for name, (_, high) in COMPLETIONS[kind].items()}}
    return SkillRequest(request_id=segment.request_id, observation_id='offline-check', skill_id=segment.skill_id,
                        active_arms=segment.active_arms, fragment_id=segment.fragment_id,
                        target_pose=segment.target_pose, target_provenance=segment.target_provenance,
                        fragment_pose=segment.fragment_pose, fragment_pose_provenance=segment.fragment_pose_provenance,
                        completion_predicate=completion, timeout_s=10.0)


def _held_widths(episode, now):
    """Last commanded width per arm (what S0 would report); measured width if no commanded stream."""
    streams = episode.commanded if episode.commanded is not None else episode.tcp
    widths = {}
    for arm, stream in streams.items():
        try:
            widths[arm] = float(causal_state(stream, [now], max_age_s=1.0)[1][0])
        except ValueError:
            widths[arm] = float(causal_state(episode.tcp[arm], [now], max_age_s=1.0)[1][0])
    return widths


def inference_check(model, dataset, device, limit=64):
    """Deployment path on held-out episodes, first and last window of each moving segment: buffers holding only
    samples at or before the observation -> live_sample -> predict_chunk -> contracts ActionChunk ->
    executable at the measured latency."""
    from ..contracts.actions import ActionRejected, executable_reason
    by_segment = {}
    for ref in dataset.windows:
        by_segment.setdefault((ref.episode, ref.segment), []).append(ref)
    refs = [r for group in by_segment.values() for r in {group[0], group[-1]}][:limit]
    report = {'attempted': len(refs), 'chunks': 0, 'rejected': [], 'not_executable': [], 'latency_ms': []}
    generator = torch.Generator(device=device).manual_seed(11)
    for ref in sorted(refs, key=lambda r: (r.episode, r.segment, r.step)):
        episode = dataset.episodes[ref.episode]
        request = _request_for(episode.segments[ref.segment])
        now = dataset.lowdim_action_target(ref)['observation_time']
        tcp, cameras = buffers_until(episode, now)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        sample = live_sample(tcp, cameras, now, request, image_shape=dataset.image_shape)
        try:
            chunk, _ = model.predict_chunk(sample, request_id=request.request_id, chunk_id=0,
                                           active_arms=request.active_arms, held_widths=_held_widths(episode, now),
                                           generator=generator)
        except ActionRejected as exc:
            report['rejected'].append(f'{request.request_id}: {exc}')
            continue
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        report['latency_ms'].append(elapsed * 1000)
        report['chunks'] += 1
        reason = executable_reason(chunk, now + elapsed)
        if reason is not None:
            report['not_executable'].append(f'{request.request_id}: {reason}')
    latencies = report.pop('latency_ms')
    report['latency_ms_median'] = float(np.median(latencies)) if latencies else None
    report['latency_ms_max'] = float(np.max(latencies)) if latencies else None
    return report


def save_run(output, model, ema, policy_config, config, split, datasets):
    combinations, counts = trained_combinations(datasets['train'], min_windows=config['min_windows_per_combination'])
    if not combinations:
        raise ValueError('no (skill, arms) combination reached min_windows_per_combination')
    checkpoint = output / 'checkpoint.pt'
    save_checkpoint(checkpoint, model=model, ema=ema, config=policy_config, step=config['steps'],
                    extra={'code_sha256': code_sha256(), 'train_config': config, 'trained_combinations': combinations,
                           'trained_skills': sorted({skill for skill, _ in combinations}),
                           'combination_window_counts': counts, 'image_shape': list(datasets['train'].image_shape),
                           'split': {k: list(v) for k, v in split.items()}})
    return checkpoint, combinations, counts


def reload_difference(model, checkpoint, sample, policy_config, device):
    reloaded, _ = load_checkpoint(checkpoint, device=device)
    probe = collate([sample], device)
    kwargs = {'num_inference_steps': policy_config.num_inference_steps} if model.kind == 'diffusion' else {}
    a = model.predict_action(probe, n_samples=2, generator=torch.Generator(device=device).manual_seed(7), **kwargs)
    b = reloaded.predict_action(probe, n_samples=2, generator=torch.Generator(device=device).manual_seed(7), **kwargs)
    return float(np.abs(a - b).max())


def collect_metrics(evaluated, run, config, device):
    """Everything metrics.json records about a finished run."""
    datasets, tensors, checkpoint, policy_config = run['datasets'], run['tensors'], run['checkpoint'], run['policy_config']
    held_out = datasets.get('test', datasets['train'])
    metrics = {
        'kind': config['kind'], 'config': config, 'policy_config': policy_config.to_dict(),
        'data': {'source': run['source'], 'supervision': config['supervision'],
                 'image_shape': list(datasets['train'].image_shape), 'excluded': run['excluded'],
                 'episodes': {k: len(v) for k, v in run['split'].items()},
                 'windows': {k: len(v) for k, v in datasets.items()},
                 'window_stats': {k: dict(v.stats) for k, v in datasets.items()},
                 'trained_combinations': run['combinations'], 'combination_window_counts': run['counts']},
        'training': {'steps': config['steps'], 'seconds': run['seconds'], 'steps_per_second': config['steps'] / run['seconds'],
                     'train_loss_log': run['log'], 'val_loss_ema_log': run['val_log'],
                     'parameters_millions': sum(p.numel() for p in evaluated.parameters()) / 1e6},
        'final_loss_ema': {name: dataset_loss(evaluated, t, device) for name, t in tensors.items() if name != 'train'},
        'checkpoint': {'file': checkpoint.name, 'sha256': file_sha256(checkpoint),
                       'reload_max_abs_diff': reload_difference(evaluated, checkpoint, datasets['train'][0],
                                                                policy_config, device)},
        'latency_batch1': latency(evaluated, datasets['train'][0], device, config),
        'inference_check': inference_check(evaluated, held_out, device),
        'versions': versions(device), 'code_sha256': code_sha256(), 'argv': sys.argv[1:],
    }
    if run['privileged'] is not None:
        predict = make_predictor(evaluated, device, policy_config.num_inference_steps)
        metrics['multimodal'] = {name: synthetic.evaluate(predict, datasets[name], run['privileged'],
                                                          n_samples=config['eval_samples'])
                                 for name in ('val', 'test') if name in datasets}
        metrics['multimodal']['demonstrations_test'] = synthetic.demo_report(held_out, run['privileged'])
    return metrics


def train(args):
    config = load_config(args.config)
    for key in ('kind', 'steps', 'batch_size', 'seed'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite non-empty {output}')
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    seed_everything(config['seed'])
    episodes, privileged, source = episodes_from(args, config)
    policy_config = PolicyConfig.from_dict(config['policy'])
    split, datasets, excluded = build_datasets(config, policy_config, episodes)
    normalizer = Normalizer.fit(datasets['train'])
    normalizer.save(output / 'normalizer.json')
    model = build_policy(config['kind'], policy_config, normalizer).to(device)
    ema = EMA(model)
    tensors = {name: stack(ds, device) for name, ds in datasets.items()}
    log, val_log, seconds = fit(model, ema, tensors, config, device)
    checkpoint, combinations, counts = save_run(output, model, ema, policy_config, config, split, datasets)
    run = {'datasets': datasets, 'tensors': tensors, 'checkpoint': checkpoint, 'policy_config': policy_config,
           'split': split, 'excluded': excluded, 'source': source, 'privileged': privileged, 'log': log,
           'val_log': val_log, 'seconds': seconds, 'combinations': combinations, 'counts': counts}
    metrics = collect_metrics(ema.averaged.eval(), run, config, device)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic', type=int, help='generate this many synthetic episodes')
    source.add_argument('--episodes', type=Path, help='directory containing saved episode directories')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--kind', choices=('diffusion', 'regression'))
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch-size', dest='batch_size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args(argv)
    metrics = train(args)
    summary = {k: metrics[k] for k in ('kind', 'final_loss_ema', 'checkpoint')}
    if 'multimodal' in metrics:
        summary['multimodal_test'] = metrics['multimodal'].get('test')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
