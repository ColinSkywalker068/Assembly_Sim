"""Replay a kinematic storyboard (choreo.json `arm_q`) through the S0 limit and capsule checks.

Read-only diagnostic for the earlier FANUC CRAG skull demo, which drove both arms by IK with no
physics. Held bricks are not in the collision model; only arm links and the Robotiq proxy are.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.robots.kinematics import T
from ..contracts.actions import ARMS
from .adapter import TABLE_CHECKED_LINKS
from .collision import capsule_clearance, lowest_point_z
from .kinematics import robotiq_proxy_model

TABLE_TOP_Z_M = 0.75  # core/config/workcell.json: table centre z 0.725 + half thickness 0.025


def _segment(scenes, frame):
    return next((name for name, (start, stop) in scenes.items() if start <= frame <= stop), None)


def replay(path, model=None):
    data = json.loads(Path(path).read_text())
    model = robotiq_proxy_model() if model is None else model
    fps = float(data['fps'])
    trajectories = {arm: np.asarray(q, dtype=float) for arm, q in zip(ARMS, data['arm_q'])}
    for arm, placement in zip(ARMS, data['arms']):
        np.testing.assert_allclose(T(placement['base_pos'], placement['base_quat']), model.arms[arm].world_T_base,
                                   atol=1e-9, err_msg=f'{arm} base differs from the dual_arm preset')
    report = {'frames': int(len(trajectories['left'])), 'fps': fps, 'tcp_offset_m': data.get('tcp_offset'), 'arms': {}}
    for arm, q in trajectories.items():
        arm_model = model.arms[arm]
        velocity = np.diff(q, axis=0) * fps
        acceleration = np.diff(velocity, axis=0) * fps
        ratio = np.abs(velocity) / arm_model.max_velocity
        frame, joint = np.unravel_index(np.argmax(ratio), ratio.shape)
        report['arms'][arm] = {
            'outside_joint_limits_frames': int(((q < arm_model.lower) | (q > arm_model.upper)).any(axis=1).sum()),
            'max_velocity_fraction_of_asset_limit': float(ratio.max()),
            'max_velocity_at': {'frame': int(frame), 'joint': f'joint_{joint + 1}',
                                'rad_s': float(velocity[frame, joint]), 'scene': _segment(data['scenes'], int(frame))},
            'frames_over_velocity_limit': int((ratio > 1).any(axis=1).sum()),
            'max_abs_acceleration_rad_s2': np.abs(acceleration).max(axis=0).round(3).tolist(),
        }
    clearance, table, tcp = [], {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}
    tool = {arm: [] for arm in ARMS}
    worst_pair = None
    for index in range(report['frames']):
        capsules = {arm: model.arms[arm].capsules(trajectories[arm][index]) for arm in ARMS}
        value, i, j = capsule_clearance(capsules['left'].arrays(), capsules['right'].arrays())
        clearance.append(value)
        if worst_pair is None or value < worst_pair[0]:
            worst_pair = (value, capsules['left'].names[i], capsules['right'].names[j], index)
        for arm in ARMS:
            lowest = lowest_point_z(capsules[arm].arrays()) - TABLE_TOP_Z_M
            table[arm].append({name.split(':')[1]: float(lowest[k]) for k, name in enumerate(capsules[arm].names)})
            tcp[arm].append(model.arms[arm].tcp(trajectories[arm][index])[2, 3] - TABLE_TOP_Z_M)
            tool[arm].append(float(model.arms[arm].tool_box_corners(trajectories[arm][index])[:, 2].min()) - TABLE_TOP_Z_M)
    clearance = np.array(clearance)
    report['arm_arm'] = {
        'min_capsule_clearance_m': float(clearance.min()), 'at_frame': int(worst_pair[3]),
        'scene': _segment(data['scenes'], int(worst_pair[3])), 'pair': [worst_pair[1], worst_pair[2]],
        'frames_overlapping': int((clearance < 0).sum()),
        'frames_below_2cm': int((clearance < 0.02).sum()),
    }
    for arm in ARMS:
        heights = {name: min(frame[name] for frame in table[arm]) for name in table[arm][0]}
        report['arms'][arm]['min_capsule_height_above_table_m'] = {k: round(v, 4) for k, v in heights.items()}
        report['arms'][arm]['min_checked_link_height_above_table_m'] = min(heights[k] for k in TABLE_CHECKED_LINKS)
        report['arms'][arm]['min_tcp_height_above_table_m'] = float(np.min(tcp[arm]))
        report['arms'][arm]['min_tool_box_height_above_table_m'] = float(np.min(tool[arm]))
        report['arms'][arm]['frames_tool_box_below_table'] = int((np.array(tool[arm]) < 0).sum())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--choreo', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    report = replay(args.choreo)
    text = json.dumps(report, indent=1)
    if args.output:
        Path(args.output).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
