"""S1 at run time: checkpoint -> contracts ActionChunks from buffered TCP and camera streams. torch.

The checkpoint records which (skill_id, arms) combinations had enough training windows; S1 refuses
any other combination, so a policy trained only on left-arm placing is never asked to place with
the right arm. Pass `trained_skills` and `trained_combinations` to the S2 AssemblyOrchestrator so
S2 refuses an untrained arm set before it reaches S1 (`contracts.skills.validate_skill_request`).
"""
from __future__ import annotations

from dataclasses import replace

import torch

from ..contracts.actions import ARMS
from ..contracts.skills import ContactMemory
from .dataset import live_sample
from .episode import canonical_arms
from .policy import load_checkpoint


class S1Runtime:
    def __init__(self, checkpoint, *, device='cpu', seed=0, stale_after_s=0.25):
        self.device = torch.device(device)
        self.model, raw = load_checkpoint(checkpoint, device=self.device)
        extra = raw.get('extra', {})
        combinations = extra.get('trained_combinations')
        if combinations is None or extra.get('image_shape') is None:
            raise ValueError('checkpoint does not record trained_combinations and image_shape')
        self.trained_combinations = frozenset((skill, tuple(arms)) for skill, arms in combinations)
        self.trained_skills = tuple(sorted({skill for skill, _ in self.trained_combinations}))
        self.image_shape = tuple(extra['image_shape'])
        self.stale_after_s = stale_after_s
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        # The executor reports each chunk S0 accepts to it and resets it per episode (contracts.skills.ContactMemory).
        self.contact_memory = ContactMemory()

    def chunk(self, request, tcp, cameras, *, commanded_widths, observation_time=None, chunk_id=0):
        """Returns (ActionChunk, info).

        `commanded_widths`: arm -> last commanded gripper width (from S0), used for held arms.
        `observation_time` defaults to the latest time both TCP buffers have reached, so buffers with
        only past samples work; a later time is refused. The chunk declares the request's contact windows
        (contracts.skills.contact_windows), where S0 may let the active arms work close to the table: a grasp
        or place window from the request's own pose, and for a retreat each arm's last window that S0 accepted
        (`contact_memory`, which this method only reads). Raises contracts.actions.ActionRejected for a
        degenerate prediction and ValueError for an untrained combination or insufficient buffers.
        """
        arms = canonical_arms(request.active_arms)
        if (request.skill_id, arms) not in self.trained_combinations:
            raise ValueError(f'S1 checkpoint was not trained on {request.skill_id!r} with arms {arms}')
        reached = min(float(tcp[arm].times[-1]) for arm in ARMS)
        observation_time = reached if observation_time is None else float(observation_time)
        if observation_time > reached + 1e-9:
            raise ValueError(f'observation_time {observation_time:.6f} s is after the newest TCP sample {reached:.6f} s')
        sample = live_sample(tcp, cameras, observation_time, request, image_shape=self.image_shape,
                             camera_names=self.model.config.cameras, obs_steps=self.model.config.obs_steps)
        chunk, info = self.model.predict_chunk(sample, request_id=request.request_id, chunk_id=chunk_id,
                                               active_arms=arms, held_widths=commanded_widths,
                                               generator=self.generator, stale_after_s=self.stale_after_s)
        return replace(chunk, contacts=self.contact_memory.windows(request)), info
