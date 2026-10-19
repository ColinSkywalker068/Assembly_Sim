"""Exploratory sampler diagnosis of one S1 smoke run (post hoc: changes no verdict, criterion or code).

Re-evaluates the run's saved diffusion checkpoint (EMA weights) on the same held-out test windows with
the same evaluation (`s1.synthetic.evaluate`, `eval_samples` per window, generator seed 99 as in
`s1.train.make_predictor`) and varies only the sampler. `ddim_16` is the deployed sampler (16 steps at
the checkpoint's `sampler_eta`) and must reproduce the run's metrics.json; `reproduction` reports the
largest difference. The other DDIM variants pass eta explicitly, so they mean the same for any checkpoint.

usage, from the repo root (needs the GPU training environment, s1/requirements-train.txt):
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. python -B \
    tasks/dual_arm_crag_skull/reports/s1_smoke/diagnose_sampler.py RUN_DIR OUTPUT_JSON
"""
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from tasks.dual_arm_crag_skull.contracts.actions import ARMS
from tasks.dual_arm_crag_skull.s1 import synthetic
from tasks.dual_arm_crag_skull.s1.policy import NoiseScheduler, PolicyConfig, collate, load_checkpoint
from tasks.dual_arm_crag_skull.s1.train import build_datasets, seed_everything

# (name, model.sample keyword arguments, scheduler patch). Every DDIM-16 variant costs 16 network calls.
VARIANTS = (
    ('ddim_16', {'num_inference_steps': 16}, None),
    ('ddim_100', {'num_inference_steps': 100, 'eta': 0.0}, None),
    ('ddpm_100', {'method': 'ddpm'}, None),
    ('ddim_16_eta_0.5', {'num_inference_steps': 16, 'eta': 0.5}, None),
    ('ddim_16_eta_1', {'num_inference_steps': 16, 'eta': 1.0}, None),
    ('ddim_16_unclipped_eps_direction', {'num_inference_steps': 16, 'eta': 0.0}, 'diffusers_eps'),
    ('ddim_16_no_x0_clip', {'num_inference_steps': 16, 'eta': 0.0}, 'no_clip'),
)


def predictor(model, device, **kwargs):
    def predict(samples, n_samples):
        generator = torch.Generator(device=device).manual_seed(99)
        out = []
        for start in range(0, len(samples), 64):
            batch = collate(samples[start:start + 64], device)
            out.append(model.predict_action(batch, n_samples=n_samples, generator=generator, **kwargs))
        return np.concatenate(out)
    return predict


@contextlib.contextmanager
def patched(scheduler, patch):
    """'diffusers_eps': the DDIM direction uses the network's eps instead of the eps recomputed from the
    clipped x0 (diffusers' default, use_clipped_model_output=False); 'no_clip': no x0 clipping."""
    original_x0, original_clip = NoiseScheduler._x0, scheduler.clip_sample

    def x0_network_eps(self, eps, t, x):
        ab = self._ab(torch.tensor(t), x)
        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        return (x0.clamp(-1, 1) if self.clip_sample else x0), eps, ab
    try:
        if patch == 'diffusers_eps':
            NoiseScheduler._x0 = x0_network_eps
        elif patch == 'no_clip':
            scheduler.clip_sample = False
        yield
    finally:
        NoiseScheduler._x0, scheduler.clip_sample = original_x0, original_clip


def split_place_first(samples, predictions, privileged):
    """place_first metrics per active arm and per demonstrated detour side."""
    groups = {}
    for sample, prediction in zip(samples, predictions):
        info = privileged[sample['episode_id']]
        for key in (f"arm={ARMS[int(info['arm'])]}", f"mode={int(info['mode'])}"):
            groups.setdefault(key, ([], []))
            groups[key][0].append(sample)
            groups[key][1].append(prediction)
    keep = ('episodes', 'obstacle_hit_rate', 'min_clearance_m_median', 'final_error_m_median')
    return {key: {k: v for k, v in synthetic._place_report(s, np.asarray(p), privileged).items() if k in keep}
            for key, (s, p) in sorted(groups.items())}


def finite(value):
    """JSON-safe copy: non-finite floats (a diverging sampler) become None."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def largest_difference(a, b):
    if isinstance(a, dict):
        return max((largest_difference(a[k], b[k]) for k in a if k in b), default=0.0)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    return 0.0


def main():
    run, output = Path(sys.argv[1]), Path(sys.argv[2])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = json.loads((run / 'diffusion' / 'metrics.json').read_text())
    model, raw = load_checkpoint(run / 'diffusion' / 'checkpoint.pt', device=device)
    model.eval()
    config = raw['extra']['train_config']
    seed_everything(config['seed'])  # same deterministic kernels as the evaluation inside train.py
    count = int(metrics['argv'][metrics['argv'].index('--synthetic') + 1])
    episodes, privileged = synthetic.make_episodes(count, seed=config['synthetic_seed'], image_size=config['image_size'])
    _, datasets, _ = build_datasets(config, PolicyConfig.from_dict(config['policy']), episodes)
    test = datasets['test']
    place = [test[i] for i in synthetic.window_at(test, 'place', 0.0)]
    result = {'run': str(run), 'checkpoint_sha256': metrics['checkpoint']['sha256'], 'device': str(device),
              'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
              'samples_per_window': config['eval_samples'], 'checkpoint_sampler_eta': model.config.sampler_eta,
              'variants': {}}
    for name, kwargs, patch in VARIANTS:
        with torch.no_grad(), patched(model.scheduler, patch):
            predict = predictor(model, device, **kwargs)
            report = synthetic.evaluate(predict, test, privileged, n_samples=config['eval_samples'])
            report['place_first_split'] = split_place_first(place, predict(place, config['eval_samples']), privileged)
        result['variants'][name] = {'sample_kwargs': kwargs, 'patch': patch, **report}
        print(name, json.dumps({'place_first_hit': report['place_first']['obstacle_hit_rate'],
                                'inactive_median_m': report['arms']['inactive_motion_m_median']}), flush=True)
    recorded = metrics['multimodal']['test']
    result['reproduction'] = {'ddim_16_vs_metrics_json_max_abs_diff':
                              largest_difference(recorded, result['variants']['ddim_16'])}
    output.write_text(json.dumps(finite(result), indent=2, allow_nan=False) + '\n')
    print(output, result['reproduction'])


if __name__ == '__main__':
    main()
