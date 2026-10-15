"""GARF flow-matching adapter, upstream commit pinned below; CUDA required."""
from __future__ import annotations
import copy
import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

UPSTREAM_COMMIT = '2d489c0b7b58de8af774aa1657c86c638e4744c2'


class GarfBackend:
    def __init__(self, repository, checkpoint, device='cuda:0', steps=20, points_per_part=512, iterations=6):
        repository = Path(repository).resolve()
        actual = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != UPSTREAM_COMMIT:
            raise ValueError(f'unsupported GARF revision {actual}; expected {UPSTREAM_COMMIT}')
        sys.path.insert(0, str(repository))
        import torch
        import hydra
        from omegaconf import OmegaConf
        if not torch.cuda.is_available():
            raise RuntimeError('official GARF encoder requires CUDA')
        if not OmegaConf.has_resolver('getIndex'):
            OmegaConf.register_new_resolver('getIndex', lambda values, index: values[index])
        with hydra.initialize_config_dir(version_base=None, config_dir=str(repository/'configs')):
            config = hydra.compose(config_name='app')
        self.model = hydra.utils.instantiate(config.model)
        # Never fall back to unrestricted checkpoint pickle execution.
        from collections import defaultdict
        from typing import Any
        from omegaconf import ListConfig
        from omegaconf.nodes import AnyNode
        from omegaconf.base import ContainerMetadata, Metadata
        # Only known declarative metadata classes from this checkpoint are allowed.
        with torch.serialization.safe_globals([ListConfig, AnyNode, ContainerMetadata,
                                               Metadata, defaultdict, int, list, dict, Any]):
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        weights = state['state_dict']
        weights = {k.replace('adjacency_model.', 'feature_extractor.', 1) if k.startswith('adjacency_model.') else k: v for k,v in weights.items()}
        self.model.load_state_dict(weights, strict=True)
        self.model.to(device).eval()
        self.device, self.steps, self.points_per_part = device, steps, points_per_part
        self.iterations = iterations
        if not 1 <= steps <= 100 or not 32 <= points_per_part <= 2048 or not 1 <= iterations <= 6:
            raise ValueError('inference limits exceeded')
        digest = hashlib.sha256()
        with open(checkpoint, 'rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        self.checkpoint_sha256 = digest.hexdigest()
        self.metadata = {"upstream_commit":UPSTREAM_COMMIT,"steps":steps,"iterations":iterations,
                         "geometry_seed":42,"nominal_points_per_part":points_per_part,"sampling":"input_density_weighted"}

    def predict(self, parts, anchor_index, seed):
        import torch
        rng = np.random.default_rng(seed)
        geometry_rng = np.random.default_rng(42)
        torch.manual_seed(seed)
        cloud, normals, centers, scales = [], [], [], []
        common_scale = max(float(np.ptp(p, axis=0).max()) for p, _ in parts)
        counts = [max(32, int(self.points_per_part*len(parts)*len(p)/sum(len(v[0]) for v in parts))) for p,_ in parts]
        for (points, normal), count_for_part in zip(parts, counts):
            selected = geometry_rng.choice(len(points), count_for_part, replace=len(points)<count_for_part)
            p = points[selected]
            center = p.mean(axis=0)
            centered = (p-center)/common_scale
            scale = np.abs(centered).max()
            cloud.append(centered/scale); normals.append(normal[selected])
            centers.append(center); scales.append(scale)
        count = len(parts)
        def tensor(value, dtype=torch.float32):
            return torch.as_tensor(np.asarray(value), dtype=dtype, device=self.device)
        batch = {
            'pointclouds': tensor(np.concatenate(cloud)[None]),
            'pointclouds_normals': tensor(np.concatenate(normals)[None]),
            'points_per_part': tensor([counts], torch.long),
            'graph': tensor(np.ones((1,count,count))),
        }
        valid = batch['points_per_part'] != 0
        ref = torch.zeros(count, dtype=torch.bool, device=self.device); ref[anchor_index] = True
        scale_tensor = tensor(scales)[:,None]
        x = torch.randn((count,7), device=self.device)
        x[:,3:] = tensor(Rotation.random(count, random_state=rng).as_quat()[:,[3,0,1,2]])
        anchor = tensor([0,0,0,1,0,0,0]); x[anchor_index] = anchor
        scheduler = copy.deepcopy(self.model.val_noise_scheduler)
        scheduler.set_timesteps(self.steps)
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
            latent = self.model.feature_extractor(batch)['point']
            latent['batch'] = latent['batch'].clone()
            for _ in range(self.iterations):
                scheduler.set_timesteps(self.steps)
                for t in scheduler.timesteps:
                    times = t.reshape(-1).repeat(count).to(self.device)
                    pred = self.model.denoiser(x=x, timesteps=times, latent=latent,
                                              part_valids=valid, scale=scale_tensor, ref_part=ref)['pred']
                    x = scheduler.step(pred, t, x).prev_sample
                    x[anchor_index] = anchor
        poses = []
        for values, center in zip(x.float().cpu().numpy(), centers):
            rotation = Rotation.from_quat(values[3:][[1,2,3,0]]).as_matrix()
            transform = np.eye(4)
            transform[:3,:3] = rotation
            transform[:3,3] = values[:3]*common_scale - rotation @ center
            poses.append(transform)
        return poses
