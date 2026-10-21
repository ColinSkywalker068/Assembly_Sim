"""Production point-cloud inference built by the pinned official mesh sampler.
Registered meshes are complete geometry in independent local metre frames.
No assembled transforms or GT are accepted by this backend.
"""
import copy
import hashlib
import tempfile
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from .garf_backend import GarfBackend


class OfficialGarfBackend(GarfBackend):
    def __init__(self, repository, checkpoint, meshes, model_unit_m, device='cuda:0', minimum_points_per_part=20):
        if not np.isfinite(model_unit_m) or model_unit_m <= 0:
            raise ValueError('explicit positive metres per model unit required')
        if type(minimum_points_per_part) is not int or not 20 <= minimum_points_per_part <= 400:
            raise ValueError("minimum_points_per_part must be an integer in [20, 400]")
        self.minimum_points_per_part = minimum_points_per_part
        super().__init__(repository, checkpoint, device)
        self.meshes = meshes
        self.model_unit_m = float(model_unit_m)
        self._prepared = None
        self.metadata.update(preprocessing='official_MeshInferenceDataset', total_points=5000,
                             minimum_points=self.minimum_points_per_part, model_unit_m=self.model_unit_m,
                             sampling_profile="official_default" if self.minimum_points_per_part == 20 else "explicit_minimum_ablation")
        self.metadata.pop('sampling', None)
        self.metadata.pop('nominal_points_per_part', None)

    def prepare(self, ids):
        if len(ids) * self.minimum_points_per_part > 5000:
            raise ValueError("minimum part allocations exceed total 5000-point budget")
        from assembly.data.inference import MeshInferenceDataset
        digest=hashlib.sha256()
        for key in ids:
            mesh=self.meshes[key]
            digest.update(key.encode()); digest.update(mesh.vertices.tobytes()); digest.update(mesh.faces.tobytes())
        key=digest.hexdigest()
        self.metadata['registered_mesh_sha256']=key
        if self._prepared is not None and self._prepared[0]==key:
            return self._prepared[1]
        with tempfile.TemporaryDirectory(prefix='garf-official-') as folder:
            paths=[]
            for index,part in enumerate(ids):
                m=self.meshes[part].copy(); m.apply_scale(1/self.model_unit_m)
                path=Path(folder)/f'{index}.ply'; m.export(path); paths.append(str(path))
            ds=MeshInferenceDataset(name='registered', meshes_paths=paths,
                pad_to_parts=len(ids), num_points_to_sample=5000, mesh_type='ply',
                sample_method='weighted', min_points_per_part=self.minimum_points_per_part, seed=42, sample_strategy='uniform')
            # Upstream currently leaves uniform sampling RNG unspecified. Fix only
            # that RNG, preserving the official sampler, counts and transforms.
            original=trimesh.sample.sample_surface
            rng=np.random.default_rng(42)
            def sample(*a, **kw):
                kw.setdefault('seed',rng)
                return original(*a,**kw)
            trimesh.sample.sample_surface=sample
            try: data=ds[0]
            finally: trimesh.sample.sample_surface=original
        self._prepared=(key,data); self._feature_cache=None
        return data

    def predict(self, parts, anchor_index, seed):
        raise ValueError('official backend requires registered geometry IDs')

    def predict_registered(self, ids, anchor_index, seed):
        import torch
        data=self.prepare(ids)
        batch={k:torch.as_tensor(np.ascontiguousarray(v), device=self.device).unsqueeze(0).contiguous()
               for k,v in data.items() if isinstance(v,np.ndarray)}
        valid=batch['points_per_part']!=0
        ref=torch.zeros(len(ids),dtype=torch.bool,device=self.device); ref[anchor_index]=True
        # Known input-local anchor, NOT the assembled reference pose.
        anchor=torch.cat([batch['translations'][valid][anchor_index],batch['quaternions'][valid][anchor_index]])
        generator=torch.Generator(self.device).manual_seed(seed)
        x=torch.randn((len(ids),7),generator=generator,device=self.device)
        x[:,3:]=torch.as_tensor(Rotation.random(len(ids),random_state=np.random.default_rng(seed)).as_quat()[:,[3,0,1,2]],device=self.device,dtype=x.dtype)
        x[anchor_index]=anchor
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
            hit=self._feature_cache is not None
            if not hit:
                device_index=torch.device(self.device).index
                if device_index is None: device_index=torch.cuda.current_device()
                with torch.random.fork_rng(devices=[device_index]):
                    torch.random.default_generator.manual_seed(42)
                    with torch.cuda.device(device_index): torch.cuda.manual_seed(42)
                    self._feature_cache=self.model.feature_extractor(batch)
                self._feature_cache['point']['batch']=self._feature_cache['point']['batch'].clone()
            latent=self._feature_cache['point']
            self.last_diagnostics={'data':data,'fracture_probability':self._feature_cache['coarse_seg_pred'].float().cpu().numpy().reshape(-1)}
            self.metadata['encoder_cache_hit']=hit
            self.metadata['encoder_feature_sha256']=hashlib.sha256(latent['feat'].cpu().contiguous().numpy().tobytes()).hexdigest()
            scheduler=copy.deepcopy(self.model.val_noise_scheduler)
            for _ in range(self.iterations):
                scheduler.set_timesteps(self.steps)
                for t in scheduler.timesteps:
                    pred=self.model.denoiser(x=x,timesteps=t.reshape(-1).repeat(len(ids)).to(self.device),
                        latent=latent,part_valids=valid,scale=batch['scale'][valid],ref_part=ref)['pred']
                    x=scheduler.step(pred,t,x).prev_sample; x[anchor_index]=anchor
        common=float(data['mesh_scale'])*self.model_unit_m
        return decode_local_poses(x.float().cpu().numpy(),data,common)


def decode_local_poses(values,data,common):
    poses=[]
    for v, q, center in zip(values,data['quaternions'],data['translations']):
        predicted=Rotation.from_quat(v[3:][[1,2,3,0]]).as_matrix()
        augmentation_inverse=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        rotation=predicted @ augmentation_inverse.T
        t=np.eye(4); t[:3,:3]=rotation
        t[:3,3]=common*(v[:3]-rotation@center)
        poses.append(t)
    return poses
