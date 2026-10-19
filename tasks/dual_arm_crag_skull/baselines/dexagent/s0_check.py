"""What S0 makes of a DexAgent delivery before it runs on the shared hardware: the executor's own commands and contact
windows (executor.py), executed on a replay that tracks the commands exactly and whose fragments are where DexAgent's
recording has them, through S0 and the release-and-settle hold.

The port's DexAgent runs this as a blocking verifier (its patch's verifiers/s0_delivery_check.py, which imports this
module from the path in its spec), so its agent sees S0's verdict on a delivery before handing it over, as an S1 chunk
or an ASPIRE program sees S0's on each command. The spec (write_spec) is written by our runner into the trial's
inputs: the task, S0's limits, the arms' home (where every backend starts them), the rig frame and DexAgent's body
names of the fragments. The fixture-held anchor is never a contact.
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from core.robots.linear_4310 import load_linear_4310

from core.workcell.config import load_workcell_preset

from ...contracts.actions import ARMS
from ...s0.adapter import S0Limits
from ...s0.kinematics import yam_model
from ..backend import ObservedPose, RobotSnapshot
from ..task import AssemblyTask
from . import executor
from .export import pose_matrix
from .trajectory import _pose_from_qpos, free_joint_columns

SPEC_SCHEMA = 'assembly_sim.dexagent_s0_spec.v2'


def limits_record(limits):
    """S0Limits as JSON-able values."""
    out = {}
    for field in fields(limits):
        value = getattr(limits, field.name)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif field.name == 'gripper_width_range_m':
            value = {arm: list(pair) for arm, pair in value.items()}
        out[field.name] = value
    return out


def limits_from_record(record):
    values = dict(record)
    for name in ('max_joint_velocity_rad_s', 'max_joint_acceleration_rad_s2'):
        values[name] = np.asarray(values[name], dtype=float)
    values['gripper_width_range_m'] = {arm: tuple(pair) for arm, pair in values['gripper_width_range_m'].items()}
    return S0Limits(**values)


def home_joints():
    """{arm: joints} every backend resets the arms to: the workcell's home."""
    workcell = load_workcell_preset('dual_arm')
    return {arm: [float(v) for v in workcell.robot(arm).home_joint_positions] for arm in ARMS}


def contact_objects(task, objects):
    """{DexAgent body name: fragment id} of the fragments a phase label may name as a contact: all but the anchor."""
    return {o['root_body']: o['fragment_id'] for o in objects if o['fragment_id'] != task.anchor_id}


def write_spec(path, *, task, limits, manifest, repo):
    """The verifier's spec: our repository (imported from), the task, S0's limits, the arms' home, the rig frame and
    the fragments' DexAgent bodies and free joints (export manifest)."""
    spec = {'schema': SPEC_SCHEMA, 'repo': str(Path(repo).resolve()), 'task': task.to_record(),
            'limits': limits_record(limits), 'home': home_joints(), 'rig_T_world': manifest['rig_T_world'],
            'objects': [{'fragment_id': o['fragment_id'], 'root_body': o['root_body'], 'free_joint': o['free_joint'],
                         'body_T_fragment': o['body_T_fragment'], 'start_rig_wxyz': o['start_rig_wxyz']}
                        for o in manifest['objects']]}
    Path(path).write_text(json.dumps(spec, indent=1) + '\n')
    return Path(path)


class ReplayBackend:
    """An AssemblyBackend that follows its commands exactly (velocity: the last step's difference) and observes
    the fragments where a DexAgent recording has them at the replay's time (frame by frame, the latest at or before
    it)."""

    def __init__(self, start, servo_hz, widths, recording, world_T_rig, objects):
        self.joints = {arm: np.asarray(start[arm], dtype=float).copy() for arm in ARMS}
        self.velocities = {arm: np.zeros(6) for arm in ARMS}
        self.widths = dict(widths)
        self.period, self.time_s = 1.0 / float(servo_hz), 0.0
        self.velocity_window_s = self.period  # each step's difference over the period
        self.recording, self.world_T_rig, self.objects = recording, world_T_rig, objects

    def robot_state(self):
        return RobotSnapshot(self.time_s, self.joints, self.velocities, self.widths, {arm: np.eye(4) for arm in ARMS},
                             self.velocity_window_s)

    def privileged_fragment_poses(self):
        time_s, qpos, names = self.recording
        frame = max(0, int(np.searchsorted(time_s, self.time_s + 1e-9, side='right')) - 1)
        out = {}
        for obj in self.objects:
            if obj['free_joint'] is None:
                rig_T_body = pose_matrix(obj['start_rig_wxyz'])
            else:
                rig_T_body = _pose_from_qpos(qpos[frame, free_joint_columns(names, obj['free_joint'])], obj['root_body'])
            out[obj['fragment_id']] = self.world_T_rig @ rig_T_body @ np.asarray(obj['body_T_fragment'], dtype=float)
        return out

    def observed_fragment_poses(self):
        return {key: ObservedPose(pose, 'oracle_sim', self.time_s) for key, pose in self.privileged_fragment_poses().items()}

    def step(self, joints, widths):
        for arm in ARMS:
            target = np.asarray(joints[arm], dtype=float)
            self.velocities[arm] = (target - self.joints[arm]) / self.period
            self.joints[arm] = target.copy()
        self.widths = dict(widths)
        self.time_s += self.period


def check_trajectory(trajectory, spec):
    """S0's verdict on a delivered trajectory.npz: {ok, status, reason, steps, windows}."""
    spec = json.loads(Path(spec).read_text()) if isinstance(spec, (str, Path)) else spec
    if spec.get('schema') != SPEC_SCHEMA:
        raise ValueError(f'not a DexAgent S0 spec ({SPEC_SCHEMA})')
    task = AssemblyTask.from_record(spec['task'])
    limits = limits_from_record(spec['limits'])
    gripper = load_linear_4310()
    with np.load(trajectory, allow_pickle=False) as data:
        loaded = {key: data[key] for key in ('joint_names', 'reference', 'time_s', 'labels', 'qpos')}
    delivery = executor.read_delivery(loaded)
    commands = executor.servo_commands(delivery, gripper, limits.servo_hz, contact_objects(task, spec['objects']))
    world_T_rig = np.linalg.inv(np.asarray(spec['rig_T_world'], dtype=float))
    names = [str(name) for name in loaded['joint_names']]
    widths = {arm: float(executor.jaw_width(gripper, delivery.jaws[arm][0])) for arm in ARMS}
    replay = ReplayBackend(spec['home'], limits.servo_hz, widths,
                           (delivery.time_s, np.asarray(loaded['qpos'], dtype=float), names), world_T_rig,
                           spec['objects'])
    result = executor.execute(replay, task, yam_model(), limits, commands)
    return {'ok': result.status == 'completed' and result.settled, 'status': result.status, 'reason':
            result.reason or result.settle_error, 'steps': result.steps,
            'windows': [{'time_s': t, 'arm': arm, 'kind': kind, 'fragment': fragment, 'center_xy': list(centre)}
                        for t, arm, kind, fragment, centre in result.windows or []]}
