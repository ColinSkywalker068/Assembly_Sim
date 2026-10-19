"""Reference ASPIRE-style program: pick the first moving fragment with the left arm and place it at its goal.

Hand-written to prove the integration path end to end on the physics-free KinematicBackend in tests. It is NOT an
ASPIRE baseline result: ASPIRE's coding-agent loop neither wrote nor saw it, and it relies on the kinematic backend's
rigid capture of a fragment whose box centre is at the closing TCP. It uses only the public API, reads goal and
observed poses from get_task_context().metadata like any program would, and obeys ASPIRE's program rules (top-level
script, `import math` only, no classes or private names, result in `result`).
"""
import math

ARM = 'left'
CLEARANCE_M = 0.1


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


def raised(pose):
    return [pose[0][0], pose[0][1], pose[0][2] + CLEARANCE_M], pose[1]


def as_pose(mapping):
    return [float(c) for c in mapping['position']], [float(c) for c in mapping['quaternion_wxyz']]


def move(pose, label):
    norm = math.sqrt(sum(c * c for c in pose[1]))
    target = {'position': pose[0], 'quaternion_wxyz': [c / norm for c in pose[1]], 'frame': 'robot_base'}
    outcome = move_to_pose(target, arm=ARM, max_steps=3000)
    report['moves'].append([label, outcome.ok, outcome.steps_executed])
    if not outcome.ok:
        raise ValueError(f'{label}: {outcome.error}')


def gripper(position, label):
    outcome = set_gripper(position, arm=ARM)
    report['moves'].append([label, outcome.ok, outcome.steps_executed])
    if not outcome.ok:
        raise ValueError(f'{label}: {outcome.error}')


report = {'moves': []}
metadata = get_task_context().metadata
fragment = metadata['moving_ids'][0]
bounds = metadata['fragment_local_bounds'][fragment]
centre = [(low + high) / 2.0 for low, high in zip(bounds['min'], bounds['max'])]
start = as_pose(metadata['observed_fragment_poses'][fragment])
goal = as_pose(metadata['goal_poses'][fragment])
tcp = get_robot_state().end_effector_poses[ARM]
grasp = compose(start, (centre, [1.0, 0.0, 0.0, 0.0]))[0], [float(c) for c in tcp.quaternion_wxyz]
place = compose(goal, compose(inverse(start), grasp))

gripper(1.0, 'open')
move(raised(grasp), 'approach')
move(grasp, 'grasp')
gripper(0.0, 'close')
move(raised(grasp), 'lift')
move(raised(place), 'transfer')
move(place, 'place')
gripper(1.0, 'release')
move(raised(place), 'retreat')

final = as_pose(get_task_context().metadata['observed_fragment_poses'][fragment])
report['fragment'] = fragment
report['final_position_error_m'] = math.dist(final[0], goal[0])
result = report
