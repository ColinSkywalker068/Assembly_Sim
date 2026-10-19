"""DexAgent's CRX + YAM bench (baselines/dexagent/bench): the CRX arm as MJCF. MuJoCo checks need the DexAgent venv
(DEXAGENT_PYTHON) and skip without it; the rest checks the generated XML against S0's kinematics directly."""
import json
import subprocess
import textwrap
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from tasks.dual_arm_crag_skull.baselines.dexagent import dexenv
from tasks.dual_arm_crag_skull.baselines.dexagent.bench import crx
from tasks.dual_arm_crag_skull.s0.kinematics import load_kinematics, yam_model

AVAILABLE, WHY = dexenv.available()
needs_dexagent = pytest.mark.skipif(not AVAILABLE, reason=WHY)


@pytest.fixture(scope='module')
def arm(tmp_path_factory):
    out = tmp_path_factory.mktemp('crx')
    return out, crx.write_arm(out)


def _matrix(element):
    pose = np.eye(4)
    pose[:3, 3] = [float(v) for v in element.get('pos', '0 0 0').split()]
    w, x, y, z = (float(v) for v in element.get('quat', '1 0 0 0').split())
    pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    return pose


def mjcf_fk(root, q):
    """World pose of every named body of an MJCF body tree with hinge joints, read the way MuJoCo reads it: the body
    frame at its pos/quat in the parent, then each joint's rotation about its axis through its pos."""
    poses, angles = {}, iter(q)

    def visit(body, parent):
        pose = parent @ _matrix(body)
        for joint in body.findall('joint'):
            axis = np.array([float(v) for v in joint.get('axis').split()])
            point = np.array([float(v) for v in joint.get('pos', '0 0 0').split()])
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * next(angles)).as_matrix()
            turn[:3, 3] = point - turn[:3, :3] @ point
            pose = pose @ turn
        poses[body.get('name')] = pose
        for child in body.findall('body'):
            visit(child, pose)

    visit(root.find('worldbody/body'), np.eye(4))
    return poses


# The probe's quaternions are float32 (|q|^2 - 1 = -3.4e-8) and core.robots.kinematics.quat_to_R assumes unit ones, so
# S0's link rotations are off by about that much (30 nm at a metre); MJCF normalises its quaternions and axes.
S0_PRECISION = 1e-7
S0_LINKS = {'base': 'base_link', 'link1': 'link_1', 'link2': 'link_2', 'link3': 'link_3', 'link4': 'link_4',
            'link5': 'link_5', 'gripper': 'flange'}


def test_the_arm_carries_dexagents_names_and_the_crx_limits(arm):
    _, path = arm
    root = ET.parse(path).getroot()
    names = [body.get('name') for body in root.iter('body')]
    assert names == ['root', 'base', 'link1', 'link2', 'link3', 'link4', 'link5', 'gripper']
    joints = list(root.iter('joint'))
    assert [joint.get('name') for joint in joints] == [f'dof_joint{i}' for i in range(1, 7)]
    lower, upper = load_kinematics()['arm']['limits']
    for joint, low, high in zip(joints, lower, upper):
        assert [float(v) for v in joint.get('range').split()] == pytest.approx([low, high], abs=1e-12)
    assert root.find('actuator') is None  # the bench's runtime adds the drives


@pytest.mark.parametrize('seed', range(5))
def test_the_mjcf_chain_is_s0s_chain(arm, seed):
    """Any joint angles: every link where S0 puts it (s0.kinematics, the frames S0 checks commands with)."""
    _, path = arm
    model = yam_model().arms['left']
    rng = np.random.default_rng(seed)
    q = rng.uniform(model.lower, model.upper)
    expected = model.frames(q)[0]
    got = mjcf_fk(ET.parse(path).getroot(), q)
    base = model.world_T_base
    for name, link in S0_LINKS.items():
        np.testing.assert_allclose(base @ got[name], expected[link], atol=S0_PRECISION)


def test_convex_mass_properties_of_a_box():
    corners = np.array([[x, y, z] for x in (0, 0.2) for y in (0, 0.1) for z in (0, 0.4)]) + [1.0, -2.0, 0.5]
    inside = corners.mean(axis=0) + np.random.default_rng(0).uniform(-0.04, 0.04, (50, 3)) * [0.4, 0.2, 0.8]
    mass, com, inertia = crx.convex_mass_properties(np.concatenate([corners, inside]), density_kg_m3=500.0)
    assert mass == pytest.approx(500.0 * 0.2 * 0.1 * 0.4)
    np.testing.assert_allclose(com, corners.mean(axis=0), atol=1e-12)
    expected = mass / 12 * np.diag([0.1 ** 2 + 0.4 ** 2, 0.2 ** 2 + 0.4 ** 2, 0.2 ** 2 + 0.1 ** 2])
    np.testing.assert_allclose(inertia, expected, atol=1e-12)


def test_each_link_gets_its_collision_hulls_mass_at_the_stated_density(arm):
    out, path = arm
    root = ET.parse(path).getroot()
    masses = {}
    for body in root.iter('body'):
        inertial = body.find('inertial')
        if body.get('name') == 'root':
            assert inertial is None
            continue
        values = [float(v) for v in inertial.get('fullinertia').split()]
        tensor = np.array([[values[0], values[3], values[4]], [values[3], values[1], values[5]],
                           [values[4], values[5], values[2]]])
        principal = np.linalg.eigvalsh(tensor)
        assert principal.min() > 0 and principal[2] <= principal[0] + principal[1] + 1e-12
        masses[body.get('name')] = float(inertial.get('mass'))
    assert set(masses) == set(S0_LINKS) and min(masses.values()) > 0
    manifest = json.loads((out / crx.MANIFEST).read_text())
    assert manifest['density_kg_m3'] == crx.DENSITY_KG_M3 and manifest['provisional'] is True
    assert manifest['masses_kg'] == pytest.approx(masses)
    for link in crx.crx_links():
        assert masses[link.name] == pytest.approx(crx.DENSITY_KG_M3 * ConvexHull(link.vertices).volume, rel=1e-9)


def test_every_link_draws_and_collides_with_its_usd_collision_mesh(arm):
    out, path = arm
    root = ET.parse(path).getroot()
    meshes = {mesh.get('name'): mesh.get('file') for mesh in root.iter('mesh')}
    assert root.find('compiler').get('meshdir') == '../assets/crx/'
    for body in root.iter('body'):
        if body.get('name') == 'root':
            continue
        (geom,) = body.findall('geom')
        assert geom.get('type') == 'mesh' and geom.get('density') == '0'
        assert (path.parent / root.find('compiler').get('meshdir') / meshes[geom.get('mesh')]).is_file()
    manifest = json.loads((out / crx.MANIFEST).read_text())
    assert manifest['arm_usd_sha256'] == load_kinematics()['provenance']['arm_usd_sha256']


@needs_dexagent
def test_mujoco_reads_the_arm_as_s0_and_the_isaac_probe_do(arm, tmp_path):
    """MuJoCo itself: S0's FK to S0_PRECISION, and the probe's fk_check (stored to 0.1 mm) to its precision."""
    _, path = arm
    check = load_kinematics()['fk_check']
    script = tmp_path / 'fk.py'
    script.write_text(textwrap.dedent('''
        import json, sys
        import mujoco
        model = mujoco.MjModel.from_xml_path(sys.argv[1])
        data = mujoco.MjData(model)
        data.qpos[:] = json.loads(sys.argv[2])
        mujoco.mj_kinematics(model, data)
        print(json.dumps({model.body(b).name: [data.xpos[b].tolist(), data.xquat[b].tolist()]
                          for b in range(1, model.nbody)}))
    '''))
    result = subprocess.run([dexenv.dexagent_python(), '-I', '-B', str(script), str(path), json.dumps(check['q'])],
                            capture_output=True, text=True, timeout=300, check=True)
    poses = json.loads(result.stdout)
    model = yam_model().arms['left']
    base_T_world = np.linalg.inv(model.world_T_base)  # the arm file's root is the arm's base, at the origin
    expected = {link: base_T_world @ pose for link, pose in model.frames(np.array(check['q']))[0].items()}
    for name, link in S0_LINKS.items():
        position, quat = (np.array(v) for v in poses[name])
        rotation = Rotation.from_quat(np.r_[quat[1:], quat[0]]).as_matrix()
        np.testing.assert_allclose(position, expected[link][:3, 3], atol=S0_PRECISION)
        np.testing.assert_allclose(rotation, expected[link][:3, :3], atol=S0_PRECISION)
        reference = check['links'][f'/World/arm/{link}']
        assert np.linalg.norm(position - reference['pos']) < 1.5e-4
        w, x, y, z = reference['quat_wxyz']
        turn = Rotation.from_matrix(Rotation.from_quat([x, y, z, w]).as_matrix().T @ rotation).magnitude()
        assert turn < 3e-4
