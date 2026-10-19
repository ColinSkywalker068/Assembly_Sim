"""Does S0's contact phase let the arms work close to the table where the skull fragments are? CPU only.

For every moving fragment of a baseline task record (runs/baseline_tasks/), a pose pointing straight down over the
fragment's box centre with the fingertips `--fingertip-height-m` above the table is checked by S0 for each arm
(several yaws, the other arm at its workcell home), first without a contact and then with the grasp contact that
contracts.skills.contact_windows would declare for a pick (centred on the fragment's observed pose). The S0 values
are TEST values, because configs/jing.json still has none; the result says nothing about grasp quality (the
fragments are curved shells) or about reach in the real cell.

  python -m tasks.dual_arm_crag_skull.experiments.s0_contact_check runs/baseline_tasks/skull_101.json --output R.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from core.workcell.config import load_workcell_preset

from ..contracts.actions import ARMS
from ..contracts.skills import SkillRequest, contact_windows
from ..s0.adapter import Rejected, S0Limits, check_joint_path
from ..s0.ik import IKOptions, solve_ik
from ..s0.kinematics import (ROBOTIQ_PROXY_MAX_WIDTH_M, YAM_PROVISIONAL_ADAPTER_THICKNESS_M, load_kinematics,
                             robotiq_proxy_model, yam_model, yam_tool_geometry, yam_width_range_m)

# gripper -> (model, width range, tool box record)
GRIPPERS = {
    'yam': (yam_model, yam_width_range_m, lambda: yam_tool_geometry()[2]),
    'robotiq': (robotiq_proxy_model, lambda: (0.0, ROBOTIQ_PROXY_MAX_WIDTH_M),
                lambda: load_kinematics()['robotiq_proxy_tool_box']),
}

TEST_LIMITS = dict(max_joint_acceleration_rad_s2=np.full(6, 20.0), arm_arm_clearance_m=0.02, table_clearance_m=0.01,
                   tcp_min_height_m=0.02, servo_hz=60, max_observation_age_s=0.5, anchor_position_tolerance_m=0.002,
                   anchor_rotation_tolerance_rad=0.005, contact_table_clearance_m=0.002, contact_max_speed_m_s=0.05)
DOWN = Rotation.from_euler('y', 90, degrees=True).as_matrix()  # tool +x, the approach axis, points at the table


def reason(call):
    try:
        call()
        return 'accepted'
    except Rejected as exc:
        return str(exc)


def check_fragment(model, limits, home, fragment, reach, height):
    pose = np.array(fragment['initial_world_T_fragment'])
    low, high = (np.array(bound) for bound in fragment['local_bounds'])
    centre = pose[:3, :3] @ ((low + high) / 2) + pose[:3, 3]
    request = SkillRequest(request_id='check', observation_id='check', skill_id='pick', active_arms=(), timeout_s=10.0,
                           completion_predicate={'type': 'grasp_held'}, fragment_id=fragment['fragment_id'],
                           fragment_pose=pose)
    result = {'fragment_id': fragment['fragment_id'], 'box_centre_m': centre.round(4).tolist(),
              'height_m': round(float(high[2] - low[2]), 4), 'arms': {}}
    for arm in ARMS:
        attempts = []
        for yaw in range(0, 180, 30):
            target = np.eye(4)
            target[:3, :3] = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ DOWN
            target[:3, 3] = [centre[0], centre[1], limits.table_top_z_m + reach + height]
            solution = solve_ik(model.arms[arm], target, home[arm], IKOptions(max_seed_jump_rad=3.0))
            if not solution.ok:
                continue
            q_now = {a: solution.q if a == arm else home[a] for a in ARMS}
            hold = {a: np.repeat(q_now[a][None], 4, axis=0) for a in ARMS}
            times = 1.0 + np.arange(1, 5) / limits.servo_hz
            (contact,) = contact_windows(SkillRequest(**{**request.__dict__, 'active_arms': (arm,)}))

            def check(contacts):
                return reason(lambda: check_joint_path(model, limits, now=1.0, q_now=q_now,
                                                       qd_now={a: np.zeros(6) for a in ARMS}, times=times,
                                                       joints=hold, contacts=contacts))

            attempts.append({'yaw_deg': yaw, 'without_contact': check(()), 'with_contact': check((contact,))})
            if attempts[-1]['with_contact'] == 'accepted':
                break
        result['arms'][arm] = attempts or 'no IK solution at any yaw'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('task', type=Path, help='baseline task record (runs/baseline_tasks/*.json)')
    parser.add_argument('--fingertip-height-m', type=float, default=0.003)
    parser.add_argument('--radius-m', type=float, default=0.12, help='contact_region_radius_m to test with')
    parser.add_argument('--gripper', choices=GRIPPERS, default='yam',
                        help='yam: I2RT linear_4310 on the provisional adapter plate; robotiq: the 2F-85 proxy')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    task = json.loads(args.task.read_text())
    make_model, width_range, tool_box = GRIPPERS[args.gripper]
    model, workcell = make_model(), load_workcell_preset('dual_arm')
    home = {arm: np.array(workcell.robot(arm).home_joint_positions, dtype=float) for arm in ARMS}
    limits = S0Limits(max_joint_velocity_rad_s=load_kinematics()['max_joint_velocity_rad_s'],
                      gripper_width_range_m={arm: width_range() for arm in ARMS},
                      table_top_z_m=task['table_top_z_m'], contact_region_radius_m=args.radius_m, **TEST_LIMITS)
    reach = float(tool_box()['parts']['fingers']['max'][0] - model.arms['left'].flange_T_tcp[0, 3])
    fragments = [check_fragment(model, limits, home, fragment, reach, args.fingertip_height_m)
                 for fragment in task['fragments'] if fragment['fragment_id'] != task['anchor_id']]
    graspable = [f['fragment_id'] for f in fragments
                 if any(isinstance(a, list) and a[-1]['with_contact'] == 'accepted' for a in f['arms'].values())]
    gripper = ('I2RT linear_4310 on a PROVISIONAL '
               f'{YAM_PROVISIONAL_ADAPTER_THICKNESS_M * 1000:g} mm adapter plate' if args.gripper == 'yam'
               else 'Robotiq 2F-85 proxy')
    report = {'task_id': task['task_id'], 'seed': task['seed'], 'gripper': gripper,
              'fingertip_height_m': args.fingertip_height_m,
              'fingertip_reach_past_tcp_m': round(reach, 5), 'limits': 'TEST values, configs/jing.json has none',
              'contact_region_radius_m': args.radius_m, 'fragments': fragments,
              'accepted_with_contact': graspable, 'moving_fragments': len(fragments)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1) + '\n')
    for fragment in fragments:
        cells = []
        for arm, attempts in fragment['arms'].items():
            last = attempts[-1] if isinstance(attempts, list) else None
            cells.append(f"{arm}: {'no IK' if last is None else last['with_contact'][:60]}"
                         f"{'' if last is None else ' (without: ' + last['without_contact'][:40] + ')'}")
        print(fragment['fragment_id'], ' | '.join(cells))
    print(f'accepted with a contact: {len(graspable)}/{len(fragments)} moving fragments')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
