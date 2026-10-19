"""DexAgent's accepted Stage 3 result -> final fragment poses in the task's world frame -> score_assembly.

A Stage 3 run publishes its delivery in <run>/stage3/verified/<id>/, named by accepted.json and sealed by
delivery.json; only a completed run whose verdict passed has one. agent.artifacts.accepted_delivery() decides
that, in the DexAgent venv from a fresh clone (worker.py `delivery`), which also re-runs
execution_scene_integrity_check against our exported Stage 2 and physics_recording_check: the run directory is
writable by the agent's code, so its own certificate is not taken on trust. The delivery's trajectory.npz is the
rigid runtime recording: qpos [T, nq] with `joint_names` naming every column ('<joint>[k]', k = 0..6, for a free
joint: position, then the wxyz quaternion, in the rig frame). The last frame is the final state.

world_T_fragment = inverse(rig_T_world) @ rig_T_body @ body_T_fragment, with rig_T_world and body_T_fragment
from the export manifest; a fixed body (the default anchor) keeps the start pose the manifest records.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from ..scoring import score_assembly
from . import dexenv
from .export import pose_matrix


def free_joint_columns(joint_names, joint):
    """qpos columns of free joint `joint` ('<joint>[0]' .. '<joint>[6]'), each present exactly once."""
    names = [str(name) for name in joint_names]
    columns = []
    for k in range(7):
        hits = [i for i, name in enumerate(names) if name == f'{joint}[{k}]']
        if len(hits) != 1:
            raise ValueError(f'trajectory has {len(hits)} columns named {joint}[{k}]; expected exactly one')
        columns.append(hits[0])
    return columns


def _pose_from_qpos(values, what):
    values = np.array(values, dtype=float)
    norm = np.linalg.norm(values[3:])
    if not np.isfinite(values).all() or norm < 0.5:
        raise ValueError(f'{what}: final free-joint state is not a finite pose with a quaternion')
    return pose_matrix(np.concatenate([values[:3], values[3:] / norm]))


def final_rig_poses(trajectory, manifest):
    """{object id: rig_T_body at the last frame}; fixed bodies at their recorded start pose."""
    with np.load(trajectory, allow_pickle=False) as data:
        if 'qpos' not in data.files or 'joint_names' not in data.files:
            raise ValueError(f'{trajectory}: a runtime recording needs qpos and joint_names')
        qpos, names = np.asarray(data['qpos'], dtype=float), list(data['joint_names'])
    if qpos.ndim != 2 or len(qpos) < 1 or qpos.shape[1] != len(names):
        raise ValueError(f'{trajectory}: qpos {qpos.shape} does not match {len(names)} joint names')
    poses = {}
    for obj in manifest['objects']:
        if obj['free_joint'] is None:
            poses[obj['object_id']] = pose_matrix(obj['start_rig_wxyz'])
        else:
            poses[obj['object_id']] = _pose_from_qpos(qpos[-1, free_joint_columns(names, obj['free_joint'])],
                                                      obj['object_id'])
    return poses


def world_poses(rig_poses, manifest):
    """{fragment id: world_T_fragment} from {object id: rig_T_body}."""
    world_T_rig = np.linalg.inv(np.asarray(manifest['rig_T_world'], dtype=float))
    out = {}
    for obj in manifest['objects']:
        out[obj['fragment_id']] = world_T_rig @ rig_poses[obj['object_id']] @ np.asarray(obj['body_T_fragment'])
    return out


def check_task(task, manifest):
    record = manifest['task']
    if (record['task_id'], record['seed'], record['layout_sha256']) != (task.task_id, task.seed, task.layout_sha256):
        raise ValueError('the export manifest belongs to a different task, seed or layout')


def score_trajectory(task, manifest, trajectory, *, settled):
    """AssemblyScore of a delivered (or any runtime) trajectory's last frame against `task`; `settled` as in
    score_assembly (a delivery DexAgent accepted has passed its own release_settle gate)."""
    check_task(task, manifest)
    return score_assembly(task, world_poses(final_rig_poses(trajectory, manifest), manifest), settled=settled)


def read_run(public):
    """Stage 3's run.json as written by run_stage3.py, or None."""
    path = Path(public) / 'run.json'
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def accepted_trajectory(public, *, clone, python, env, log_path, recon, timeout_s=3600, sandbox=None):
    """(delivery directory, trajectory.npz) of a run whose delivery DexAgent accepts and that passes the re-check
    against our exported Stage 2 (`recon`), run from `clone` (a fresh clone), else (None, reason). `sandbox`: the
    re-check's bwrap paths (runner.sandbox_paths), as for the run itself."""
    spec = {'clone': str(clone), 'public': str(Path(public).resolve()), 'recon': str(Path(recon).resolve()),
            'out_dir': str(Path(log_path).parent)}
    try:
        result = dexenv.run_worker(python, 'delivery', spec, log_path, env=env, timeout_s=timeout_s, sandbox=sandbox)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        return None, str(exc).splitlines()[0][:500]
    if not result.get('ok'):
        failed = [name for name, row in (result.get('checks') or {}).items() if not row.get('ok')]
        issues = list(result['verify'].get('issues') or []) + [f'{name} failed' for name in failed]
        return None, '; '.join(issues) or 'delivery does not verify'
    directory = Path(result['directory'])
    return directory, directory / 'trajectory.npz'
