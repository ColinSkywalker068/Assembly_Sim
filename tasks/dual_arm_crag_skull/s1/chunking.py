"""Turn one unnormalized S1 prediction (16, 20) into a validated contracts ActionChunk. numpy only.

Waypoint k (1-based) is stamped `observation_time + time_offset_s + k / hz`: the time its pose
was recorded after the observation in training. Inference and transport latency are handled by
S0 dropping waypoints that are already in the past (and refusing chunks past valid_until), not
by relabelling the predicted trajectory.

Held (inactive) arms get an exact hold of their anchor pose and of their last COMMANDED width,
which the caller takes from S0's command history. For a position-controlled gripper the commanded
width sets the squeeze: commanding the measured width of a gripper blocked by a fragment would
release it.
"""
from __future__ import annotations

import numpy as np

from ..contracts.actions import ACTION_DIM, ARMS, ActionChunk, hand_slice
from .episode import POLICY_HZ, canonical_arms

_HOLD = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float)  # identity relative pose as pose9


def chunk_from_prediction(actions, *, anchors, held_widths, observation_time, request_id, chunk_id, active_arms,
                          hz=POLICY_HZ, time_offset_s=0.0, stale_after_s=0.25, max_width=None):
    """Returns (ActionChunk, info). Predicted widths are clipped to [0, max_width] and every clip is
    counted; inactive arms hold their anchor and `held_widths` (arm -> last commanded width, m)."""
    actions = np.array(actions, dtype=float, copy=True)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f'expected (N, {ACTION_DIM}) predicted actions')
    anchors = np.asarray(anchors, dtype=float)
    if anchors.shape != (len(ARMS), 4, 4):
        raise ValueError('anchors must be (2, 4, 4) in ARMS order')
    active = canonical_arms(active_arms)
    held = tuple(arm for arm in ARMS if arm not in active)
    for arm in held:
        width = held_widths.get(arm) if hasattr(held_widths, 'get') else None
        if width is None or not np.isfinite(width) or width < 0:
            raise ValueError(f'held arm {arm} needs its last commanded width')
        columns = hand_slice(arm)
        actions[:, columns.start:columns.start + 9] = _HOLD
        actions[:, columns.stop - 1] = float(width)
    width_columns = [hand_slice(arm).stop - 1 for arm in active]
    upper = np.inf if max_width is None else float(max_width)
    clipped = int(((actions[:, width_columns] < 0) | (actions[:, width_columns] > upper)).sum())
    actions[:, width_columns] = np.clip(actions[:, width_columns], 0.0, upper)
    times = float(observation_time) + float(time_offset_s) + np.arange(1, len(actions) + 1) / hz
    chunk = ActionChunk(request_id=request_id, chunk_id=chunk_id, observation_time=float(observation_time),
                        valid_until=float(observation_time) + float(stale_after_s),
                        anchors={arm: anchors[i] for i, arm in enumerate(ARMS)}, waypoint_times=times,
                        waypoints=actions, active_arms=active)
    return chunk, {'width_clipped': clipped, 'held_arms': held}
