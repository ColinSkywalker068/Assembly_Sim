"""Conditional 1D U-Net Diffusion Policy (Chi et al., RSS 2023) over dual-TCP action chunks. torch.

Native DDPM training (squaredcos_cap_v2 betas, epsilon prediction) and DDIM / DDPM sampling; no
diffusers dependency. One ResNet-18 per camera (or one shared) with every BatchNorm replaced by
GroupNorm; an unavailable view contributes zero features plus a 0 flag. The network reads exactly
`dataset.POLICY_INPUTS`; anchors and other absolute world poses are used only to decode. Checkpoints
record the conditioning layout (skill order, arms, cameras, input layout version) and refuse to load
under a different one.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..contracts.actions import ACTION_DIM
from ..contracts.skills import SKILL_IDS
from .chunking import chunk_from_prediction
from .dataset import ARM_MASK_DIM, FRAGMENT_DIM, LAYOUT, LOWDIM_DIM, POLICY_INPUTS, TARGET_DIM, Normalizer
from .episode import CAMERAS, OBS_STEPS, PRED_STEPS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PolicyConfig:
    obs_steps: int = OBS_STEPS
    pred_steps: int = PRED_STEPS
    action_dim: int = ACTION_DIM
    lowdim_dim: int = LOWDIM_DIM
    cameras: tuple = CAMERAS
    image_feature_dim: int = 64
    shared_encoder: bool = False
    encoder_weights: str | None = None
    random_shift: int = 2
    skill_embed_dim: int = 16
    down_dims: tuple = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    step_embed_dim: int = 128
    num_train_timesteps: int = 100
    clip_sample: bool = True
    num_inference_steps: int = 16
    regression_hidden: tuple = field(default=(1024, 1024))
    sampler_eta: float = 0.0  # DDIM noise at sampling: 0 deterministic, 1 matches the DDPM posterior variance

    def __post_init__(self):
        eta = self.sampler_eta
        if isinstance(eta, bool) or not isinstance(eta, (int, float)) or not 0.0 <= eta <= 1.0:
            raise ValueError(f'sampler_eta must be a number in [0, 1], got {eta!r}')

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'unknown policy config keys: {sorted(unknown)}')
        known = dict(data)
        for key in ('cameras', 'down_dims', 'regression_hidden'):
            if key in known:
                known[key] = tuple(known[key])
        return cls(**known)

    def to_dict(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


# ---------------------------------------------------------------- noise schedule

def squaredcos_cap_v2_betas(num_steps, max_beta=0.999):
    """Nichol and Dhariwal cosine schedule, as diffusers' `squaredcos_cap_v2`."""
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    return torch.tensor([min(1 - alpha_bar((i + 1) / num_steps) / alpha_bar(i / num_steps), max_beta)
                         for i in range(num_steps)], dtype=torch.float64)


class NoiseScheduler:
    """Forward process q(x_t | x_0) plus DDIM (eta >= 0) and DDPM ancestral reverse steps."""

    def __init__(self, num_train_timesteps=100, clip_sample=True):
        self.num_train_timesteps = int(num_train_timesteps)
        self.clip_sample = clip_sample
        self.betas = squaredcos_cap_v2_betas(self.num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def _ab(self, t, like):
        return self.alphas_cumprod.to(like.device)[t].to(like.dtype)

    def add_noise(self, x0, noise, t):
        ab = self._ab(t, x0).view(-1, *([1] * (x0.ndim - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def inference_timesteps(self, num_inference_steps):
        """'trailing' spacing: always starts at T-1, ends near 0; n = T gives every step."""
        n = int(num_inference_steps)
        if not 1 <= n <= self.num_train_timesteps:
            raise ValueError(f'num_inference_steps must be in 1..{self.num_train_timesteps}')
        # linspace, not a float-step arange: the latter returns n + 1 steps ending at -1 for some n.
        steps = np.round(np.linspace(self.num_train_timesteps, 0, n, endpoint=False)).astype(int) - 1
        if len(steps) != n or steps.min() < 0 or len(set(steps.tolist())) != n:
            raise AssertionError(f'bad DDIM schedule for n={n}: {steps.tolist()}')
        return steps.tolist()

    def _x0(self, eps, t, x):
        ab = self._ab(torch.tensor(t), x)
        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        if self.clip_sample:
            x0 = x0.clamp(-1, 1)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        return x0, eps, ab

    def ddim_step(self, eps, t, t_prev, x, eta=0.0, generator=None):
        x0, eps, ab = self._x0(eps, t, x)
        ab_prev = self._ab(torch.tensor(t_prev), x) if t_prev >= 0 else torch.ones((), dtype=x.dtype, device=x.device)
        variance = (1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)
        sigma = eta * variance.clamp(min=0).sqrt()
        x_prev = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        if eta > 0:
            x_prev = x_prev + sigma * torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        return x_prev

    def ddpm_step(self, eps, t, x, generator=None):
        x0, _, ab = self._x0(eps, t, x)
        ab_prev = self._ab(torch.tensor(t - 1), x) if t > 0 else torch.ones((), dtype=x.dtype, device=x.device)
        beta = self.betas.to(x.device)[t].to(x.dtype)
        alpha = 1 - beta
        mean = (ab_prev.sqrt() * beta / (1 - ab)) * x0 + (alpha.sqrt() * (1 - ab_prev) / (1 - ab)) * x
        if t == 0:
            return mean
        variance = (beta * (1 - ab_prev) / (1 - ab)).clamp(min=1e-20)
        return mean + variance.sqrt() * torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)


# ---------------------------------------------------------------- 1D U-Net (Chi et al.)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        frequency = torch.exp(torch.arange(half, device=x.device) * -(math.log(10000) / (half - 1)))
        angles = x.float()[:, None] * frequency[None]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_groups):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                                   nn.GroupNorm(n_groups, out_channels), nn.Mish())

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two conv blocks with FiLM (per-channel scale and bias) from the condition after the first."""

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size, n_groups):
        super().__init__()
        self.blocks = nn.ModuleList([Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
                                     Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)])
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels * 2))
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        scale, bias = self.cond_encoder(cond).unsqueeze(-1).chunk(2, dim=1)
        out = self.blocks[1](scale * out + bias)
        return out + self.residual(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, step_embed_dim=128, down_dims=(256, 512, 1024),
                 kernel_size=5, n_groups=8):
        super().__init__()
        dims = [input_dim, *down_dims]
        pairs = list(zip(dims[:-1], dims[1:]))
        self.step_encoder = nn.Sequential(SinusoidalPosEmb(step_embed_dim), nn.Linear(step_embed_dim, step_embed_dim * 4),
                                          nn.Mish(), nn.Linear(step_embed_dim * 4, step_embed_dim))
        cond = step_embed_dim + global_cond_dim

        def block(a, b):
            return ConditionalResidualBlock1D(a, b, cond, kernel_size, n_groups)
        self.down_modules = nn.ModuleList()
        for index, (a, b) in enumerate(pairs):
            last = index == len(pairs) - 1
            self.down_modules.append(nn.ModuleList([block(a, b), block(b, b),
                                                    nn.Conv1d(b, b, 3, 2, 1) if not last else nn.Identity()]))
        self.mid_modules = nn.ModuleList([block(dims[-1], dims[-1]), block(dims[-1], dims[-1])])
        self.up_modules = nn.ModuleList()
        for a, b in reversed(pairs[1:]):
            self.up_modules.append(nn.ModuleList([block(b * 2, a), block(a, a), nn.ConvTranspose1d(a, a, 4, 2, 1)]))
        self.final_conv = nn.Sequential(Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
                                        nn.Conv1d(down_dims[0], input_dim, 1))

    def forward(self, sample, timestep, global_cond):
        x = sample.transpose(1, 2)
        cond = torch.cat([self.step_encoder(timestep), global_cond], dim=-1)
        skips = []
        for first, second, down in self.down_modules:
            x = second(first(x, cond), cond)
            skips.append(x)
            x = down(x)
        for module in self.mid_modules:
            x = module(x, cond)
        for first, second, up in self.up_modules:
            x = torch.cat([x, skips.pop()], dim=1)
            x = up(second(first(x, cond), cond))
        return self.final_conv(x).transpose(1, 2)


# ---------------------------------------------------------------- observation encoder

def replace_batchnorm(module):
    """In place: every BatchNorm2d -> GroupNorm(channels // 16, channels)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(child.num_features // 16, child.num_features))
        else:
            replace_batchnorm(child)
    return module


def resnet18_groupnorm(feature_dim, weights_path=None):
    from torchvision.models import resnet18
    net = resnet18(weights=None)
    if weights_path:
        net.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    replace_batchnorm(net)
    net.fc = nn.Identity()
    return nn.Sequential(net, nn.Linear(512, feature_dim))


def random_shift(images, pad):
    """DrQ-style random translation by up to `pad` pixels; images (N, 3, H, H) float."""
    n, _, h, w = images.shape
    if h != w:
        raise ValueError('random_shift expects square images')
    padded = F.pad(images, (pad,) * 4, mode='replicate')
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=images.device, dtype=images.dtype)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    grid = torch.cat([arange, arange.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2), device=images.device).to(images.dtype)
    return F.grid_sample(padded, grid + shift * 2.0 / (h + 2 * pad), padding_mode='zeros', align_corners=False)


class ObservationEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        count = 1 if config.shared_encoder else len(config.cameras)
        self.encoders = nn.ModuleList([resnet18_groupnorm(config.image_feature_dim, config.encoder_weights)
                                       for _ in range(count)])
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def output_dim(self):
        return len(self.config.cameras) * (self.config.image_feature_dim + 1)

    def forward(self, images, mask):
        """images (B, S, C, H, W, 3) uint8, mask (B, S, C) bool -> (B, S, C * (F + 1))."""
        b, s, c, h, w, _ = images.shape
        x = images.permute(0, 1, 2, 5, 3, 4).float() / 255.0
        features = []
        for camera in range(c):
            view = x[:, :, camera].reshape(b * s, 3, h, w)
            if self.training and self.config.random_shift > 0:
                view = random_shift(view, self.config.random_shift)
            encoder = self.encoders[0 if self.config.shared_encoder else camera]
            features.append(encoder((view - self.mean) / self.std).view(b, s, -1))
        flags = mask.float().unsqueeze(-1)
        features = torch.stack(features, dim=2) * flags
        return torch.cat([features, flags], dim=-1).reshape(b, s, -1)


# ---------------------------------------------------------------- policies

def collate(samples, device):
    """List of episode samples -> dict of tensors holding only POLICY_INPUTS (+ action if present)."""
    if not samples:
        raise ValueError('cannot collate an empty list of samples')
    keys = list(POLICY_INPUTS) + (['action'] if 'action' in samples[0] else [])
    batch = {}
    for key in keys:
        values = np.stack([np.asarray(s[key]) for s in samples])
        if key == 'images':
            batch[key] = torch.as_tensor(values, dtype=torch.uint8, device=device)
        elif key == 'image_mask':
            batch[key] = torch.as_tensor(values, dtype=torch.bool, device=device)
        elif key == 'skill_index':
            batch[key] = torch.as_tensor(values, dtype=torch.long, device=device)
        else:
            batch[key] = torch.as_tensor(values, dtype=torch.float32, device=device)
    return batch


class _ConditionedPolicy(nn.Module):
    """Shared observation path: images + low-dim history + skill + arm mask + goal and fragment poses."""

    def __init__(self, config, normalizer):
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.obs_encoder = ObservationEncoder(config)
        self.skill_embedding = nn.Embedding(len(SKILL_IDS), config.skill_embed_dim)
        for name in ('action', 'lowdim', 'target', 'fragment'):
            self.register_buffer(f'{name}_scale', torch.tensor(normalizer.scale[name], dtype=torch.float32))
            self.register_buffer(f'{name}_offset', torch.tensor(normalizer.offset[name], dtype=torch.float32))

    @property
    def cond_dim(self):
        c = self.config
        return (c.obs_steps * (self.obs_encoder.output_dim + c.lowdim_dim) + c.skill_embed_dim + ARM_MASK_DIM
                + TARGET_DIM + 1 + FRAGMENT_DIM + 1)

    def normalize(self, name, x):
        return x * getattr(self, f'{name}_scale') + getattr(self, f'{name}_offset')

    def unnormalize(self, name, y):
        return (y - getattr(self, f'{name}_offset')) / getattr(self, f'{name}_scale')

    def global_condition(self, batch):
        visual = self.obs_encoder(batch['images'], batch['image_mask'])
        lowdim = self.normalize('lowdim', batch['lowdim'])
        target = self.normalize('target', batch['target']) * batch['target_mask']
        fragment = self.normalize('fragment', batch['fragment']) * batch['fragment_mask']
        return torch.cat([torch.cat([visual, lowdim], dim=-1).flatten(1), self.skill_embedding(batch['skill_index']),
                          batch['arm_mask'], target, batch['target_mask'], fragment, batch['fragment_mask']], dim=-1)

    def predict_action(self, batch, n_samples=1, **kwargs):
        """Unnormalized actions as numpy (B, n_samples, pred_steps, action_dim)."""
        normalized = self.sample(batch, n_samples=n_samples, **kwargs)
        return self.unnormalize('action', normalized).double().cpu().numpy()

    def predict_chunk(self, sample, *, request_id, chunk_id, active_arms, held_widths, generator=None,
                      **chunk_kwargs):
        """One live observation sample -> (contracts ActionChunk, info). `held_widths`: last commanded widths."""
        batch = collate([sample], self.action_scale.device)
        actions = self.predict_action(batch, n_samples=1, generator=generator)[0, 0]
        return chunk_from_prediction(actions, anchors=sample['anchors'], held_widths=held_widths,
                                     observation_time=sample['observation_time'], request_id=request_id,
                                     chunk_id=chunk_id, active_arms=active_arms, **chunk_kwargs)


class DiffusionPolicy(_ConditionedPolicy):
    kind = 'diffusion'

    def __init__(self, config, normalizer):
        super().__init__(config, normalizer)
        self.unet = ConditionalUnet1D(config.action_dim, self.cond_dim, config.step_embed_dim, config.down_dims,
                                      config.kernel_size, config.n_groups)
        self.scheduler = NoiseScheduler(config.num_train_timesteps, config.clip_sample)

    def compute_loss(self, batch, generator=None):
        x0 = self.normalize('action', batch['action'])
        noise = torch.randn(x0.shape, generator=generator, device=x0.device)
        t = torch.randint(0, self.scheduler.num_train_timesteps, (x0.shape[0],), generator=generator, device=x0.device)
        prediction = self.unet(self.scheduler.add_noise(x0, noise, t), t, self.global_condition(batch))
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample(self, batch, n_samples=1, num_inference_steps=None, method='ddim', eta=None, generator=None):
        eta = self.config.sampler_eta if eta is None else eta
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f'eta must be in [0, 1], got {eta!r}')
        cond = self.global_condition(batch).repeat_interleave(n_samples, dim=0)
        shape = (cond.shape[0], self.config.pred_steps, self.config.action_dim)
        x = torch.randn(shape, generator=generator, device=cond.device)
        if method == 'ddim':
            steps = self.scheduler.inference_timesteps(num_inference_steps or self.config.num_inference_steps)
            for t, t_prev in zip(steps, steps[1:] + [-1]):
                eps = self.unet(x, torch.full((x.shape[0],), t, device=x.device), cond)
                x = self.scheduler.ddim_step(eps, t, t_prev, x, eta=eta, generator=generator)
        elif method == 'ddpm':
            for t in reversed(range(self.scheduler.num_train_timesteps)):
                eps = self.unet(x, torch.full((x.shape[0],), t, device=x.device), cond)
                x = self.scheduler.ddpm_step(eps, t, x, generator=generator)
        else:
            raise ValueError(f'unknown sampling method {method!r}')
        return x.view(-1, n_samples, *shape[1:])


class RegressionPolicy(_ConditionedPolicy):
    """Baseline: same observation path, MLP head, MSE on normalized actions (one deterministic output)."""

    kind = 'regression'

    def __init__(self, config, normalizer):
        super().__init__(config, normalizer)
        layers, width = [], self.cond_dim
        for hidden in config.regression_hidden:
            layers += [nn.Linear(width, hidden), nn.Mish()]
            width = hidden
        layers.append(nn.Linear(width, config.pred_steps * config.action_dim))
        self.head = nn.Sequential(*layers)

    def compute_loss(self, batch, generator=None):
        prediction = self.head(self.global_condition(batch)).view(-1, self.config.pred_steps, self.config.action_dim)
        return F.mse_loss(prediction, self.normalize('action', batch['action']))

    @torch.no_grad()
    def sample(self, batch, n_samples=1, **kwargs):
        out = self.head(self.global_condition(batch)).view(-1, 1, self.config.pred_steps, self.config.action_dim)
        return out.expand(-1, n_samples, -1, -1)


POLICIES = {'diffusion': DiffusionPolicy, 'regression': RegressionPolicy}


def build_policy(kind, config, normalizer):
    if kind not in POLICIES:
        raise ValueError(f'policy kind must be one of {sorted(POLICIES)}')
    return POLICIES[kind](config, normalizer)


class EMA:
    """Parameter EMA with Diffusion Policy's warmup: decay = 1 - (1 + step / inv_gamma) ** -power."""

    def __init__(self, model, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999):
        self.averaged = copy.deepcopy(model).eval().requires_grad_(False)
        self.inv_gamma, self.power, self.min_value, self.max_value = inv_gamma, power, min_value, max_value
        self.step = 0

    def decay(self, step):
        step = max(0, step - 1)
        if step <= 0:
            return 0.0
        return max(self.min_value, min(1 - (1 + step / self.inv_gamma) ** -self.power, self.max_value))

    @torch.no_grad()
    def update(self, model):
        decay = self.decay(self.step)
        for averaged, current in zip(self.averaged.parameters(), model.parameters()):
            averaged.mul_(decay).add_(current.detach(), alpha=1 - decay)
        for averaged, current in zip(self.averaged.buffers(), model.buffers()):
            averaged.copy_(current)
        self.step += 1


def checkpoint_layout(config):
    return {**LAYOUT, 'cameras': list(config.cameras), 'obs_steps': config.obs_steps, 'pred_steps': config.pred_steps}


def save_checkpoint(path, *, model, ema, config, step, extra=None):
    torch.save({'kind': model.kind, 'config': config.to_dict(), 'normalizer': model.normalizer.to_dict(),
                'layout': checkpoint_layout(config),
                'model': model.state_dict(), 'ema': None if ema is None else ema.averaged.state_dict(),
                'ema_step': None if ema is None else ema.step, 'step': int(step), 'extra': extra or {}}, path)


def load_checkpoint(path, device='cpu', use_ema=True):
    """Restricted (weights_only) load -> (model in eval mode, raw checkpoint dict).

    Refuses a checkpoint whose conditioning layout differs from this code (e.g. a reordered
    SKILL_IDS would silently map embeddings to other skills). Encoder weight files are not read:
    every weight comes from the checkpoint.
    """
    data = torch.load(path, map_location=device, weights_only=True)
    config = PolicyConfig.from_dict(data['config'])
    expected = checkpoint_layout(config)
    if data.get('layout') != expected:
        raise ValueError(f'checkpoint layout {data.get("layout")} does not match this code {expected}')
    model = build_policy(data['kind'], replace(config, encoder_weights=None), Normalizer.from_dict(data['normalizer']))
    model = model.to(device)
    model.load_state_dict(data['ema'] if use_ema and data['ema'] is not None else data['model'])
    return model.eval(), data
