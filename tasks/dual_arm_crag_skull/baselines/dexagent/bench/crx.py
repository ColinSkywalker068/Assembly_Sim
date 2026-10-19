"""One FANUC CRX-10iA/L arm as MJCF for DexAgent's bench, from the arm USD and configs/crx10ial_kinematics.json.

DexAgent finds its arm by names (episode.robot_parts, rigid_robot), so the bodies and joints carry its YAM arm's names:
root (the station mount), base (the USD's base_link), link1..link5 (link_1..link_5) and gripper (link_6, whose frame
is the flange's: the kinematics file's joint_6_flange is the identity, checked), joints dof_joint1..6. The chain is
S0's (s0.kinematics reads the same file the same way), so the MJCF's FK is the FK S0 checks commands with: a link at
joint angle q sits at before @ R(q) @ after in its parent, which MJCF writes as the body at before @ after and a hinge
about inverse(after)'s axis through inverse(after)'s origin. Each link draws with the USD's collision mesh (visual
only, like DexAgent's arm files: its runtime adds the collision copies), which PhysX cooks to a convex hull.

Mass: the USD authors none (MassAPI mass and density 0 on every body), so PhysX gives each body its collider's volume
at the default density; here each link gets the mass properties of its collision mesh's convex hull at
DENSITY_KG_M3. PROVISIONAL: neither is the real arm's (about 40 kg in all), and PhysX's cooked hull has at most 64
vertices; read back from Isaac before calibrating the two simulators against each other.
"""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from core.robots.kinematics import Arm, R_to_quat

from ....s0.kinematics import KINEMATICS_PATH, load_kinematics
from ....workspace import REPO_ROOT

DENSITY_KG_M3 = 1000.0  # PhysX's default density for a body with no authored mass
ARM_XML = 'robot/crx10ial.xml'
MANIFEST = 'robot/crx10ial.json'
MESH_DIR = 'assets/crx'
USD_PRIM = '/fanuc_crx10ial'
# DexAgent's arm names (setup/robot/yam_arm.xml) <- the CRX USD's links
LINKS = (('base', 'base_link'), ('link1', 'link_1'), ('link2', 'link_2'), ('link3', 'link_3'), ('link4', 'link_4'),
         ('link5', 'link_5'), ('gripper', 'link_6'))
JOINTS = tuple(f'dof_joint{i}' for i in range(1, 7))
AXES = {'X': np.array([1.0, 0.0, 0.0]), 'Y': np.array([0.0, 1.0, 0.0]), 'Z': np.array([0.0, 0.0, 1.0])}
RGBA = '0.86 0.85 0.81 1'


@dataclass(frozen=True)
class Link:
    name: str
    usd_link: str
    parent_T_link: np.ndarray  # at joint angle 0
    joint: str | None
    joint_pos: np.ndarray | None  # the hinge's origin and axis in this link's frame
    joint_axis: np.ndarray | None
    joint_range: tuple[float, float] | None
    mass_kg: float
    com: np.ndarray
    inertia: np.ndarray  # about the centre of mass, in this link's frame
    vertices: np.ndarray  # the collision mesh in this link's frame
    triangles: np.ndarray


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convex_mass_properties(points, density_kg_m3=DENSITY_KG_M3):
    """(mass, centre of mass, inertia tensor about it) of the convex hull of `points` at a uniform density."""
    hull = ConvexHull(np.asarray(points, dtype=float))
    inside = hull.points[hull.vertices].mean(axis=0)
    volume, first, second = 0.0, np.zeros(3), np.zeros((3, 3))
    for simplex in hull.simplices:  # one tetrahedron per facet, from a point inside: all of them positive
        corners = np.vstack([inside, hull.points[simplex]])
        size = abs(np.linalg.det(corners[1:] - corners[0])) / 6
        total = corners.sum(axis=0)
        volume += size
        first += size * total / 4
        second += size / 20 * (corners.T @ corners + np.outer(total, total))
    com = first / volume
    covariance = second - volume * np.outer(com, com)
    inertia = density_kg_m3 * (np.trace(covariance) * np.eye(3) - covariance)
    return density_kg_m3 * volume, com, inertia


def _usd_mesh(stage, usd_link):
    """(vertices, triangles) of `usd_link`'s collision mesh in the link's frame."""
    from pxr import UsdGeom
    link = stage.GetPrimAtPath(f'{USD_PRIM}/{usd_link}')
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f'{USD_PRIM}/{usd_link}/collisions'))
    if not (link.IsValid() and mesh.GetPrim().IsValid()):
        raise ValueError(f'the arm USD has no collision mesh for {usd_link}')

    def world(prim):  # USD stores row-vector matrices
        return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0), dtype=float).T

    link_T_mesh = np.linalg.inv(world(link)) @ world(mesh.GetPrim())
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    triangles, start = [], 0
    for count in counts:  # fans, for any polygon
        face = indices[start:start + count]
        triangles += [(face[0], face[k], face[k + 1]) for k in range(1, count - 1)]
        start += count
    return points @ link_T_mesh[:3, :3].T + link_T_mesh[:3, 3], np.asarray(triangles, dtype=int)


def crx_links(usd_path=None, kinematics_path=KINEMATICS_PATH, density_kg_m3=DENSITY_KG_M3):
    """The arm's links, base first, from the kinematics file and the USD it was probed from."""
    from pxr import Usd
    data = load_kinematics(kinematics_path)
    usd_path = REPO_ROOT / data['provenance']['arm_usd'] if usd_path is None else Path(usd_path)
    if _sha256(usd_path) != data['provenance']['arm_usd_sha256']:
        raise ValueError(f'{usd_path} is not the arm USD the kinematics were probed from')
    chain = Arm({'arm': data['arm']})
    if not np.allclose(chain.T_flange, np.eye(4), atol=1e-12):
        raise ValueError('the CRX flange is not link_6\'s frame; DexAgent mounts its hand on the last link')
    lower, upper = data['arm']['limits']
    stage = Usd.Stage.Open(str(usd_path))
    links = []
    for index, (name, usd_link) in enumerate(LINKS):
        vertices, triangles = _usd_mesh(stage, usd_link)
        mass, com, inertia = convex_mass_properties(vertices, density_kg_m3)
        if index == 0:
            parent_T_link, joint = np.eye(4), None
            joint_pos = joint_axis = joint_range = None
        else:
            axis, before, after = chain.chain[index - 1]
            parent_T_link, joint = before @ after, JOINTS[index - 1]
            link_T_joint = np.linalg.inv(after)
            joint_pos, joint_axis = link_T_joint[:3, 3], link_T_joint[:3, :3] @ AXES[axis]
            joint_range = (float(lower[index - 1]), float(upper[index - 1]))
        links.append(Link(name, usd_link, parent_T_link, joint, joint_pos, joint_axis, joint_range, mass, com, inertia,
                          vertices, triangles))
    return tuple(links)


def write_stl(path, vertices, triangles):
    """Binary STL of `triangles` (indices into `vertices`)."""
    corners = np.asarray(vertices, dtype=float)[np.asarray(triangles, dtype=int)]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    record = np.zeros(len(corners), dtype=[('normal', '<f4', 3), ('corners', '<f4', (3, 3)), ('attribute', '<u2')])
    record['normal'], record['corners'] = normals, corners
    with open(path, 'wb') as stream:
        stream.write(b'CRX-10iA/L collision mesh'.ljust(80, b' '))
        stream.write(np.uint32(len(record)).tobytes())
        stream.write(record.tobytes())


def _values(values):
    return ' '.join(format(float(v), '.17g') for v in np.asarray(values, dtype=float).ravel())


def _pose_attributes(element, pose):
    element.set('pos', _values(pose[:3, 3]))
    element.set('quat', _values(R_to_quat(pose[:3, :3])))


def arm_element(links, prefix='', root_pose=None, mesh_prefix=None):
    """The arm's body tree under a body `<prefix>root` at `root_pose` (identity by default); meshes are named
    `<mesh_prefix><link>` (`mesh_prefix` defaults to `prefix`)."""
    mesh_prefix = prefix if mesh_prefix is None else mesh_prefix
    root = ET.Element('body', name=f'{prefix}root')
    if root_pose is not None:
        _pose_attributes(root, np.asarray(root_pose, dtype=float))
    parent = root
    for link in links:
        body = ET.SubElement(parent, 'body', name=f'{prefix}{link.name}')
        if not np.allclose(link.parent_T_link, np.eye(4), atol=0):
            _pose_attributes(body, link.parent_T_link)
        i = link.inertia
        ET.SubElement(body, 'inertial', pos=_values(link.com), mass=format(link.mass_kg, '.17g'),
                      fullinertia=_values([i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]]))
        if link.joint is not None:
            ET.SubElement(body, 'joint', name=f'{prefix}{link.joint}', pos=_values(link.joint_pos),
                          axis=_values(link.joint_axis), range=_values(link.joint_range))
        ET.SubElement(body, 'geom', type='mesh', mesh=f'{mesh_prefix}{link.name}', contype='0', conaffinity='0',
                      group='1', density='0', rgba=RGBA)
        parent = body
    return root


def mesh_files(links):
    """Link name -> mesh file name (under MESH_DIR)."""
    return {link.name: f'{link.usd_link}.stl' for link in links}


def write_arm(setup_dir, links=None):
    """Write the arm on its own (ARM_XML), its meshes (MESH_DIR) and MANIFEST under `setup_dir`; return the XML's
    path. DexAgent's runtime reads the arm's inertias from this file."""
    setup_dir = Path(setup_dir)
    links = crx_links() if links is None else links
    (setup_dir / MESH_DIR).mkdir(parents=True, exist_ok=True)
    (setup_dir / ARM_XML).parent.mkdir(parents=True, exist_ok=True)
    files = mesh_files(links)
    for link in links:
        write_stl(setup_dir / MESH_DIR / files[link.name], link.vertices, link.triangles)
    root = ET.Element('mujoco', model='fanuc_crx10ial')
    ET.SubElement(root, 'compiler', angle='radian', meshdir=f'../{MESH_DIR}/')
    asset = ET.SubElement(root, 'asset')
    for link in links:
        ET.SubElement(asset, 'mesh', name=link.name, file=files[link.name])
    ET.SubElement(root, 'worldbody').append(arm_element(links))
    ET.indent(root)
    path = setup_dir / ARM_XML
    path.write_text(ET.tostring(root, encoding='unicode') + '\n')
    data = load_kinematics()
    manifest = {'schema': 'assembly_sim.dexagent_crx_arm.v1', 'arm_usd': data['provenance']['arm_usd'],
                'arm_usd_sha256': data['provenance']['arm_usd_sha256'],
                'kinematics_sha256': _sha256(KINEMATICS_PATH), 'density_kg_m3': DENSITY_KG_M3, 'provisional': True,
                'mass_note': 'convex hull of each USD collision mesh at the PhysX default density; not the real arm',
                'masses_kg': {link.name: link.mass_kg for link in links},
                'meshes': {name: {'file': f'{MESH_DIR}/{file}', 'sha256': _sha256(setup_dir / MESH_DIR / file)}
                           for name, file in files.items()},
                'xml_sha256': _sha256(path)}
    (setup_dir / MANIFEST).write_text(json.dumps(manifest, indent=1) + '\n')
    return path
