"""The CPU parts of experiments/assembly_replay_isaac.py (the replay itself needs Isaac Sim and a GPU)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tasks.dual_arm_crag_skull.experiments.assembly_replay_isaac import carry, look_at, pose_at

CUBE = np.array([[x, y, z] for x in (-0.02, 0.02) for y in (-0.02, 0.02) for z in (0.0, 0.04)])  # origin at its bottom


def pose(position, yaw_deg=0.0):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix()
    out[:3, 3] = position
    return out


def test_a_sideways_insertion_starts_just_clear_of_what_is_in_place():
    anchor = CUBE + [0.0, 0.0, 0.75]  # the cube's goal is beside it at +x; it goes in along -x
    start, goal = pose([-0.4, 0.1, 0.75], 30.0), pose([0.04, 0.0, 0.75])
    points, durations = carry(CUBE, start, goal, (-1.0, 0.0, 0.0), anchor, top_z=0.79, clearance=0.02)
    p0, above, over, begin, end = points
    np.testing.assert_allclose(p0, start[:3, 3])
    np.testing.assert_allclose(end, goal[:3, 3])
    assert above[2] == over[2] == pytest.approx(0.79 + 0.02)  # its bottom clears the top of everything
    np.testing.assert_allclose(begin, [0.04 + 0.02, 0.0, 0.75])  # its -x face 2 cm clear of the anchor's +x face
    assert len(durations) == 4 and min(durations) > 0
    assert durations[3] == pytest.approx(0.02 / 0.05)  # the last straight move is slow


def test_from_above_it_comes_down_over_its_goal():
    anchor = CUBE + [0.0, 0.0, 0.75]
    points, _ = carry(CUBE, pose([0.4, 0.0, 0.75]), pose([0.0, 0.0, 0.79]), (0.0, 0.0, -1.0), anchor, top_z=0.79,
                      clearance=0.02)
    begin, end = points[3], points[4]
    np.testing.assert_allclose(begin, [0.0, 0.0, 0.81])  # its bottom 2 cm over the anchor's top
    np.testing.assert_allclose(end, [0.0, 0.0, 0.79])


def test_poses_follow_the_points_turning_only_while_gliding():
    start, goal = pose([-0.4, 0.1, 0.75], 30.0), pose([0.04, 0.0, 0.75])
    points, durations = carry(CUBE, start, goal, (-1.0, 0.0, 0.0), CUBE + [0, 0, 0.75], top_z=0.79, clearance=0.02)
    edges = np.concatenate([[0.0], np.cumsum(durations)])
    for time, point in zip(edges, points):
        position, _ = pose_at(time, points, durations, start, goal)
        np.testing.assert_allclose(position, point, atol=1e-12)
    for time in np.linspace(edges[2], edges[4], 7):  # coming down and going in: already at the goal orientation
        _, rotation = pose_at(time, points, durations, start, goal)
        np.testing.assert_allclose(rotation, goal[:3, :3], atol=1e-9)
    _, rotation = pose_at(edges[1] / 2, points, durations, start, goal)
    np.testing.assert_allclose(rotation, start[:3, :3], atol=1e-9)
    position, _ = pose_at(edges[4] + 5.0, points, durations, start, goal)
    np.testing.assert_allclose(position, goal[:3, 3])


def test_a_camera_looks_along_its_minus_z():
    camera = look_at((0.0, -1.0, 1.3), (0.0, 0.0, 0.8))
    view = -camera[:3, 2]
    np.testing.assert_allclose(view, np.array([0.0, 1.0, -0.5]) / np.linalg.norm([0.0, 1.0, -0.5]), atol=1e-12)
    assert camera[2, 1] > 0  # image up points up


def test_fragments_take_the_colours_they_are_named_by(tmp_path):
    from tasks.dual_arm_crag_skull.baselines.task import task_from_layout
    from tasks.dual_arm_crag_skull.experiments.assembly_replay_isaac import display_colours
    from tasks.dual_arm_crag_skull.tests.baselines_fixtures import write_layout
    task = task_from_layout(write_layout(tmp_path / 'layout'), anchor_id='frag_a')
    colours = display_colours(task)
    assert {key: name for key, (name, _) in colours.items()} == {'frag_a': 'red', 'frag_b': 'blue', 'frag_c': 'green'}
    red = np.array(colours['frag_a'][1])
    assert red.max() <= 1.0 and red[0] > 0.8 and red[1] < 0.1  # the rendered red of DexAgent's export, not its 0.45
