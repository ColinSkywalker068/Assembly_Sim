"""The I2RT linear_4310 gripper description against the official model.

Reference boxes and the grasp site below were computed by MuJoCo 3.10 itself (mj_forward of the vendored
linear_4310.xml in envs/dexagent, mesh vertices taken from model.mesh_vert through geom_xpos/geom_xmat), so they
check this module's reading of the MJCF, including MuJoCo's mesh frame convention, not only its arithmetic.
"""
import hashlib
import json
import shutil

import numpy as np
import pytest

from core.robots.linear_4310 import ASSET_DIR, load_linear_4310

# Mount frame (the YAM joint6 output: +z approach, +x right, +y down), metres, at q = 0 and q = 0.0475.
MUJOCO_BOXES = {
    0.0: {'gripper': ([-0.051, -0.0325, -0.0105], [0.051, 0.0325, 0.0635]),
          'tip_left': ([-0.023223, -0.033906, 0.050876], [0.048348, 0.034094, 0.144719]),
          'tip_right': ([-0.048348, -0.034095, 0.050877], [0.023223, 0.033905, 0.14472])},
    0.0475: {'gripper': ([-0.051, -0.0325, -0.0105], [0.051, 0.0325, 0.0635]),
             'tip_left': ([-0.070723, -0.033906, 0.050876], [0.000848, 0.034094, 0.144719]),
             'tip_right': ([-0.000848, -0.034095, 0.050877], [0.070723, 0.033905, 0.144719])},
}
MUJOCO_GRASP_SITE = ([3.89e-05, 9.68e-05, 0.14465], [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


@pytest.fixture(scope='module')
def gripper():
    return load_linear_4310()


def boxes(gripper, q):
    parts = {'gripper': gripper.body_points, **gripper.tips_at(q)}
    return {name: (points.min(axis=0), points.max(axis=0)) for name, points in parts.items()}


def test_vendored_files_are_the_pinned_upstream_files():
    source = json.loads((ASSET_DIR / 'SOURCE.json').read_text())
    assert source['repository'] == 'https://github.com/i2rt-robotics/i2rt'
    assert source['commit'] == '120c3c81400171174604e503943f8d1ebc891058' and source['license'] == 'MIT'
    assert (ASSET_DIR / 'LICENSE').read_text().startswith('MIT License')
    for name, record in source['files'].items():
        data = (ASSET_DIR / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record['sha256'] and len(data) == record['bytes']
        assert hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest() == record['git_blob']


@pytest.mark.parametrize('q', sorted(MUJOCO_BOXES))
def test_meshes_sit_where_mujoco_puts_them(gripper, q):
    for name, (low, high) in boxes(gripper, q).items():
        np.testing.assert_allclose(low, MUJOCO_BOXES[q][name][0], atol=2e-6, err_msg=name)
        np.testing.assert_allclose(high, MUJOCO_BOXES[q][name][1], atol=2e-6, err_msg=name)


def test_tcp_is_the_grasp_site_at_the_fingertip_ends(gripper):
    position, rotation = MUJOCO_GRASP_SITE
    np.testing.assert_allclose(gripper.mount_T_tcp[:3, 3], position, rtol=0, atol=1e-7)
    np.testing.assert_allclose(gripper.mount_T_tcp[:3, :3], rotation, atol=1e-5)
    # I2RT's station README: 144.650 mm along the mount's +z and 0.104 mm off its axis.
    assert gripper.mount_T_tcp[2, 3] == pytest.approx(0.14465, abs=5e-7)
    assert np.hypot(*gripper.mount_T_tcp[:2, 3]) == pytest.approx(0.000104, abs=1e-6)
    tip_end = max(points[:, 2].max() for points in gripper.tips_at(0.0).values())
    assert 0 <= tip_end - gripper.mount_T_tcp[2, 3] < 1e-4


def test_fingers_slide_apart_along_the_mount_x_axis(gripper):
    np.testing.assert_allclose(gripper.slide_axes['tip_left'], [-1.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(gripper.slide_axes['tip_right'], [1.0, 0.0, 0.0], atol=1e-5)
    assert gripper.travel_m == 0.0475


def test_pad_gap_and_width_mapping(gripper):
    """At q = 0 the pad faces overlap by 1 mm (contacts between the tips are excluded in the MJCF), so the pads meet
    at q = 0.5 mm and the widest opening is 2 * 47.5 - 1 = 94 mm."""
    assert gripper.closed_gap_m == pytest.approx(-0.001, abs=5e-5)
    assert gripper.contact_joint_m == pytest.approx(0.0005, abs=3e-5)
    assert gripper.max_width_m == pytest.approx(0.094, abs=1e-4)
    for q in (gripper.contact_joint_m, 0.01, 0.03, gripper.travel_m):
        assert gripper.width(q) == pytest.approx(2 * q + gripper.closed_gap_m, abs=1e-12)
        assert gripper.joint_for_width(gripper.width(q)) == pytest.approx(q, abs=1e-12)
    assert gripper.width(0.0) == 0.0 and gripper.width(gripper.contact_joint_m / 2) == 0.0  # overlapping pads
    for width in (-1e-6, gripper.max_width_m + 1e-6, float('nan')):
        with pytest.raises(ValueError, match='width'):
            gripper.joint_for_width(width)
    for q in (-1e-6, gripper.travel_m + 1e-6):
        with pytest.raises(ValueError, match='joint'):
            gripper.width(q)


def test_pad_faces_measured_from_the_meshes(gripper):
    tips = gripper.tips_at(0.03)
    left, right = tips['tip_left'], tips['tip_right']
    pad = gripper.mount_T_tcp[2, 3] - 0.03
    inner_left = left[left[:, 2] > pad][:, 0].max()
    inner_right = right[right[:, 2] > pad][:, 0].min()
    assert inner_right - inner_left == pytest.approx(gripper.width(0.03), abs=1e-6)


def test_masses_follow_the_mjcf(gripper):
    assert gripper.masses_kg == {'gripper': 0.553219, 'tip_left': 0.0710042, 'tip_right': 0.0710042}


def test_inertials_follow_the_mjcf_in_each_link_frame(gripper):
    tip = gripper.inertials['tip_left']
    assert tip.mass_kg == 0.0710042
    np.testing.assert_allclose(tip.center_of_mass_m, [-0.0224716, 0.0143408, -0.0426253])
    np.testing.assert_allclose(tip.diagonal_inertia_kg_m2, [6.24193e-05, 6.01079e-05, 2.83591e-05])
    quat = np.array([0.498899, 0.455348, -0.0576406, 0.735143])
    np.testing.assert_allclose(tip.principal_axes_wxyz, quat / np.linalg.norm(quat))
    assert set(gripper.inertials) == {'gripper', 'tip_left', 'tip_right'}
    assert {name: value.mass_kg for name, value in gripper.inertials.items()} == dict(gripper.masses_kg)


def test_width_between_two_unequal_fingers(gripper):
    """Two independently driven fingers: each moves its own pad face."""
    assert gripper.width_between(0.01, 0.03) == pytest.approx(0.04 + gripper.closed_gap_m, abs=1e-12)
    assert gripper.width_between(0.02, 0.02) == pytest.approx(gripper.width(0.02), abs=1e-12)
    assert gripper.width_between(0.0, 0.0) == 0.0
    with pytest.raises(ValueError, match='joint'):
        gripper.width_between(0.01, gripper.travel_m + 1e-6)


@pytest.mark.parametrize('old, new, message', [
    ('polycoef="0 1 0 0 0"', 'polycoef="0 -1 0 0 0"', 'coupled 1:1'),
    ('<joint name="joint8" pos="0 0 0" type="slide" axis="0 0 -1" range="0 0.0475"/>',
     '<joint name="joint8" pos="0 0 0" type="slide" axis="0 0 -1" range="0 0.04"/>', 'same distance'),
    ('<joint name="joint7" pos="0 0 0" type="slide" axis="0 0 -1" range="0 0.0475"/>',
     '<joint name="joint7" pos="0 0 0" type="hinge" axis="0 0 -1" range="0 0.0475"/>', 'slide joint'),
    ('<body name="tip_right" pos="0.0238981 -0.0450619 -0.0545599" quat="0.707105 0.707108 0 0">',
     '<body name="tip_right" pos="0.0238981 -0.0450619 -0.0545599" euler="1.5708 0 0">', 'only quat'),
    ('<mesh name="tip_left" file="tip_left.stl"/>', '<mesh name="tip_left" file="tip_left.stl" scale="2 2 2"/>',
     'scale'),
])
def test_an_mjcf_this_reader_does_not_understand_is_refused(tmp_path, old, new, message):
    copy = tmp_path / 'i2rt_linear_4310'
    shutil.copytree(ASSET_DIR, copy)
    mjcf = copy / 'linear_4310.xml'
    text = mjcf.read_text()
    assert old in text
    mjcf.write_text(text.replace(old, new))
    with pytest.raises(ValueError, match=message):
        load_linear_4310(copy, verify=False)


def test_a_modified_asset_is_refused(tmp_path):
    copy = tmp_path / 'i2rt_linear_4310'
    shutil.copytree(ASSET_DIR, copy)
    mesh = copy / 'assets' / 'tip_left.stl'
    data = bytearray(mesh.read_bytes())
    data[200] ^= 1
    mesh.write_bytes(bytes(data))
    with pytest.raises(ValueError, match='tip_left.stl'):
        load_linear_4310(copy)
    load_linear_4310(copy, verify=False)
