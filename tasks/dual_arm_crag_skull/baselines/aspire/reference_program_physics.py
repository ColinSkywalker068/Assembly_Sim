"""Reference ASPIRE-style program for a physics backend: pick one fragment top down with the left arm and place it at
its goal, declaring the contacts S0 needs close to the table.

Hand-written to prove the integration path on Isaac Sim (sim/backend.py). It is NOT an ASPIRE baseline result:
ASPIRE's coding-agent loop neither wrote nor saw it. It reads only get_task_context().metadata and the robot state,
uses only the public API and the fanuc_crx contact extensions, and obeys ASPIRE's program rules (top-level script,
`import math` only, no classes or private names, result in `result`).

It takes the first fragment on the left of the table that stands upright (its box's z axis within UPRIGHT_DEG of
vertical) and whose box is narrow enough for the open fingers. The fingers come straight down and close across the
narrower horizontal side of the box, centred on it, the fingertips FINGERTIP_M above the table; the place pose keeps
the same grip relative to the fragment.
"""
import math

ARM = 'left'
CLEARANCE_M = 0.1
FINGERTIP_M = 0.003
OPEN_MARGIN_M = 0.01
SQUEEZE_M = 0.006
MAX_STEPS = 6000
UPRIGHT_DEG = 5.0


def multiply(a, b):
    return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]]


def conjugate(q):
    return [q[0], -q[1], -q[2], -q[3]]


def rotate(q, v):
    return multiply(multiply(q, [0.0, v[0], v[1], v[2]]), conjugate(q))[1:]


def compose(a, b):
    """a * b for poses given as (position, quaternion_wxyz)."""
    return [p + r for p, r in zip(a[0], rotate(a[1], b[0]))], multiply(a[1], b[1])


def inverse(pose):
    q = conjugate(pose[1])
    return [-c for c in rotate(q, pose[0])], q


def as_pose(mapping):
    return [float(c) for c in mapping['position']], [float(c) for c in mapping['quaternion_wxyz']]


def quaternion_from_columns(x, y, z):
    """Unit quaternion (w, x, y, z) of the rotation whose columns are the axes x, y, z."""
    m = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]


def down_pose(position, yaw):
    """TCP pointing straight down (TCP +x along world -z), the fingers closing along the yaw direction."""
    approach, closing = [0.0, 0.0, -1.0], [math.cos(yaw), math.sin(yaw), 0.0]
    third = [approach[1] * closing[2] - approach[2] * closing[1], approach[2] * closing[0] - approach[0] * closing[2],
             approach[0] * closing[1] - approach[1] * closing[0]]
    return list(position), quaternion_from_columns(approach, closing, third)


def raised(pose):
    return [pose[0][0], pose[0][1], pose[0][2] + CLEARANCE_M], pose[1]


def check(outcome, label):
    report['steps'].append([label, outcome.ok, outcome.steps_executed])
    if not outcome.ok:
        raise ValueError(f'{label}: {outcome.error}')


def move(pose, label):
    norm = math.sqrt(sum(c * c for c in pose[1]))
    target = {'position': pose[0], 'quaternion_wxyz': [c / norm for c in pose[1]], 'frame': 'robot_base'}
    check(move_to_pose(target, arm=ARM, max_steps=MAX_STEPS), label)


def graspable(key):
    pose = as_pose(metadata['observed_fragment_poses'][key])
    bounds = metadata['fragment_local_bounds'][key]
    narrow = min(bounds['max'][0] - bounds['min'][0], bounds['max'][1] - bounds['min'][1])
    upright = math.degrees(math.acos(max(-1.0, min(1.0, rotate(pose[1], [0.0, 0.0, 1.0])[2])))) <= UPRIGHT_DEG
    return pose[0][0] < 0 and upright and narrow + 2 * OPEN_MARGIN_M <= high


report = {'steps': []}
metadata = get_task_context().metadata
low, high = metadata['controller']['gripper_width_range_m'][ARM]
fragment = [key for key in metadata['moving_ids'] if graspable(key)][0]
bounds = metadata['fragment_local_bounds'][fragment]
size = [hi - lo for lo, hi in zip(bounds['min'], bounds['max'])]
centre_local = [(lo + hi) / 2.0 for lo, hi in zip(bounds['min'], bounds['max'])]
start = as_pose(metadata['observed_fragment_poses'][fragment])
goal = as_pose(metadata['goal_poses'][fragment])
narrow_axis = rotate(start[1], [1.0, 0.0, 0.0] if size[0] <= size[1] else [0.0, 1.0, 0.0])
chord = min(size[0], size[1])
centre = compose(start, (centre_local, [1.0, 0.0, 0.0, 0.0]))[0]
table = metadata['table_top_z_m']
grasp = down_pose([centre[0], centre[1], table + FINGERTIP_M], math.atan2(narrow_axis[1], narrow_axis[0]))
place = compose(goal, compose(inverse(start), grasp))
open_position = min(1.0, (chord + 2 * OPEN_MARGIN_M - low) / (high - low))
close_position = max(0.0, (chord - SQUEEZE_M - low) / (high - low))

check(set_gripper(open_position, arm=ARM), 'open')
move(raised(grasp), 'approach')
check(fanuc_crx.declare_contact('grasp', fragment, arm=ARM), 'declare grasp')
move(grasp, 'descend')
check(set_gripper(close_position, arm=ARM), 'close')
move(raised(grasp), 'lift')
check(fanuc_crx.clear_contact(arm=ARM), 'clear grasp')
move(raised(place), 'transfer')
check(fanuc_crx.declare_contact('place', fragment, arm=ARM), 'declare place')
move(place, 'lower')
check(set_gripper(open_position, arm=ARM), 'release')
move(raised(place), 'retreat')
check(fanuc_crx.clear_contact(arm=ARM), 'clear place')

final = as_pose(get_task_context().metadata['observed_fragment_poses'][fragment])
report['fragment'] = fragment
report['chord_m'] = chord
report['final_position_error_m'] = math.dist(final[0], goal[0])
result = report
