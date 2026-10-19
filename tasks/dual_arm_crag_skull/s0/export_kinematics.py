"""Regenerate configs/crx10ial_kinematics.json from the Isaac probe dump and the repo USD assets.

Needs `usd-core` (pxr) only for regeneration; S0 itself reads the JSON. Joint frames and the
fk_check reference come from fanuc_probe.py (Isaac Sim 4.5 reading the same USD); the USD is
re-read here to cross-check those frames, take the asset-declared joint velocity limits, and fit
bounding capsules to each link's collision mesh in that link's frame.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .collision import fit_capsule

ARM_LINKS = ('base_link', 'link_1', 'link_2', 'link_3', 'link_4', 'link_5', 'link_6')
GRIPPER_ROOT = '/World/Robotiq_2F_85'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _matrix(gf_matrix):
    # USD stores row-vector matrices (p' = p M); transpose to the column convention used here.
    return np.asarray(gf_matrix, dtype=float).T


def _mesh_points(stage, root, frame_prim):
    """All Mesh vertices under `root`, expressed in the frame of `frame_prim`."""
    from pxr import Usd, UsdGeom
    frame_T_world = np.linalg.inv(_matrix(UsdGeom.Xformable(frame_prim).ComputeLocalToWorldTransform(0)))
    clouds = []
    # The 2F-85 collision Xforms are instanceable; their meshes exist only as instance proxies.
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            local = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            world_T_mesh = _matrix(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))
            transform = frame_T_world @ world_T_mesh
            clouds.append(local @ transform[:3, :3].T + transform[:3, 3])
    if not clouds:
        raise ValueError(f'no collision mesh under {root.GetPath()}')
    return np.concatenate(clouds)


def _capsule_record(points, source):
    p0, p1, radius = fit_capsule(points)
    return {'p0': p0.tolist(), 'p1': p1.tolist(), 'radius': radius, 'vertices': int(len(points)), 'source': source}


def _arm_usd(path, joint_names):
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    joints, capsules, drives = {}, {}, {}
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() in joint_names:
            joint = UsdPhysics.Joint(prim)
            record = {}
            for suffix in ('0', '1'):
                record[f'pos{suffix}'] = list(getattr(joint, f'GetLocalPos{suffix}Attr')().Get())
                quat = getattr(joint, f'GetLocalRot{suffix}Attr')().Get()
                record[f'rot{suffix}'] = [*quat.GetImaginary(), quat.GetReal()]
            record['axis'] = prim.GetAttribute('physics:axis').Get() or 'X'
            velocity = prim.GetAttribute('physxJoint:maxJointVelocity')
            record['max_velocity_deg_s'] = velocity.Get() if velocity.IsValid() else None
            joints[prim.GetName()] = record
            stiffness = prim.GetAttribute('drive:angular:physics:stiffness')
            if stiffness.IsValid():
                drives[prim.GetName()] = {'stiffness': stiffness.Get(),
                                          'damping': prim.GetAttribute('drive:angular:physics:damping').Get()}
    for link in ARM_LINKS:
        link_prim = stage.GetPrimAtPath(f'{root.GetPath()}/{link}')
        collisions = stage.GetPrimAtPath(f'{link_prim.GetPath()}/collisions')
        capsules[link] = _capsule_record(_mesh_points(stage, collisions, link_prim), 'usd collision mesh')
    return joints, capsules, drives


def _gripper_capsule(path):
    """One capsule around every 2F-85 collision mesh in the open pose, in the gripper base frame."""
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(path))
    base = stage.GetPrimAtPath(f'{GRIPPER_ROOT}/base_link')
    clouds = [_mesh_points(stage, prim, base) for prim in Usd.PrimRange(stage.GetPrimAtPath(GRIPPER_ROOT))
              if prim.HasAPI(UsdPhysics.CollisionAPI)]
    return _capsule_record(np.concatenate(clouds), 'robotiq 2F-85 collision meshes, authored open pose')


def _gripper_tool_box(path, probe, samples=81):
    """Axis-aligned box in the gripper base frame (== flange) enclosing the collision meshes over the
    whole finger stroke. The fingertips reach further along +X when closing than in the open pose.
    `parts` splits it into the static body (base_link) and the moving fingers, for S0's contact phase."""
    from pxr import Usd, UsdPhysics
    from core.robots.kinematics import Gripper
    stage = Usd.Stage.Open(str(path))
    root = stage.GetPrimAtPath(GRIPPER_ROOT)
    base = stage.GetPrimAtPath(f'{GRIPPER_ROOT}/base_link')
    gripper = Gripper(probe)
    per_link, authored = {}, []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            link = str(prim.GetPath()).split('/')[3]
            per_link.setdefault(link, []).append(_mesh_points(stage, prim, stage.GetPrimAtPath(f'{GRIPPER_ROOT}/{link}')))
            authored.append(_mesh_points(stage, prim, base))
    per_link = {link: np.concatenate(clouds) for link, clouds in per_link.items()}
    authored = np.concatenate(authored)
    open_pose = np.concatenate([points @ gripper.link_T(link, 0.0)[:3, :3].T + gripper.link_T(link, 0.0)[:3, 3]
                                for link, points in per_link.items()])
    model_vs_usd = float(np.abs(np.sort(open_pose, axis=0) - np.sort(authored, axis=0)).max())
    angles = np.linspace(float(gripper.gap_f.min()), float(gripper.gap_f.max()), samples)
    bounds = {part: [np.full(3, np.inf), np.full(3, -np.inf)] for part in ('body', 'fingers')}
    for angle in angles:
        for link, points in per_link.items():
            transform = gripper.link_T(link, angle)
            moved = points @ transform[:3, :3].T + transform[:3, 3]
            part = bounds['body' if link == 'base_link' else 'fingers']
            part[0], part[1] = np.minimum(part[0], moved.min(axis=0)), np.maximum(part[1], moved.max(axis=0))
    low = np.minimum(bounds['body'][0], bounds['fingers'][0])
    high = np.maximum(bounds['body'][1], bounds['fingers'][1])
    links = {'body': ['base_link'], 'fingers': sorted(link for link in per_link if link != 'base_link')}
    parts = {part: {'min': bounds[part][0].tolist(), 'max': bounds[part][1].tolist(), 'links': links[part]}
             for part in bounds}
    return {'min': low.tolist(), 'max': high.tolist(), 'parts': parts, 'open_pose_min': authored.min(axis=0).tolist(),
            'open_pose_max': authored.max(axis=0).tolist(), 'finger_angles_rad': [float(angles[0]), float(angles[-1])],
            'samples': samples, 'vertices': int(len(authored)), 'open_pose_model_vs_usd_max_abs_m': model_vs_usd,
            'frame': 'gripper base_link, mounted on the flange with identity local poses',
            'source': 'robotiq 2F-85 collision meshes swept over the finger stroke with core Gripper kinematics (probe)'}


def build(probe_path, arm_usd, gripper_usd):
    probe = json.loads(Path(probe_path).read_text())
    arm = probe['arm']
    dof = list(arm['dof_names'])
    names = [*dof, 'joint_6_flange']
    by_name = {j['path'].split('/')[-1]: j for j in arm['joints']}
    probe_joints = [by_name[name] for name in names]
    usd_joints, capsules, drives = _arm_usd(arm_usd, set(names))
    frame_diff = max(float(np.max(np.abs(np.subtract(probe_joint[key], usd_joints[name][key]))))
                     for name, probe_joint in zip(names, probe_joints) for key in ('pos0', 'rot0', 'pos1', 'rot1'))
    axes_match = all(by_name[name].get('axis', 'X') == usd_joints[name]['axis'] for name in dof)
    velocities = [usd_joints[name]['max_velocity_deg_s'] for name in dof]
    if any(value is None for value in velocities) or not axes_match:
        raise ValueError('USD joints lack velocity limits or disagree with the probe axes')
    links = {k: v for k, v in probe['fk_check']['links'].items() if k.startswith('/World/arm/')}
    return {
        'schema_version': 1,
        'model': 'FANUC CRX-10iA/L, Isaac Sim 4.5 asset library USD',
        'provenance': {
            'arm_usd': 'core/assets/robots/fanuc_crx10ial/crx10ial.usd', 'arm_usd_sha256': sha256(arm_usd),
            'gripper_usd': 'core/assets/robots/robotiq_2f85/Robotiq_2F_85_flattened.usd',
            'gripper_usd_sha256': sha256(gripper_usd),
            'probe': 'fanuc_probe.py dump (Isaac Sim 4.5) from the FANUC CRAG skull demo bundle, '
                     '04_intermediate/probe/fanuc_probe.json',
            'probe_sha256': sha256(probe_path),
            'probe_vs_usd_joint_frame_max_abs_diff': frame_diff,
            'generated': datetime.date.today().isoformat(),
            'generator': 'python -m tasks.dual_arm_crag_skull.s0.export_kinematics',
        },
        'arm': {'dof_names': dof, 'limits': arm['limits'], 'joints': probe_joints},
        'max_joint_velocity_rad_s': [math.radians(value) for value in velocities],
        'max_joint_velocity_source': 'USD physxJoint:maxJointVelocity (deg/s); asset-declared, not hardware-verified',
        'max_joint_acceleration_rad_s2': None,
        'max_joint_acceleration_note': 'not declared by the asset or the public datasheet; must be measured',
        'link_capsules': capsules,
        'robotiq_proxy_capsule': _gripper_capsule(gripper_usd),
        'robotiq_proxy_tool_box': _gripper_tool_box(gripper_usd, probe),
        'fk_check': {'q': probe['fk_check']['q_arm'], 'links': links,
                     'note': 'Isaac Sim world poses with the arm root at the origin, rounded by the probe to 1e-4'},
        'sim_fidelity_warning': (
            'USD angular drives declare stiffness {:.3g} and damping {:.3g} per degree on every arm joint. With zero '
            'velocity targets the damping makes PhysX follow position targets like a first-order lag of '
            'damping/stiffness = {:.3g} s (in Isaac a 1.5 rad/s move ran 50 mm behind), so a simulator backend must '
            'feed the commanded velocity forward, as sim/backend.py does. Simulated tracking error and latency are '
            'not evidence about the real FANUC servo.'.format(
                max(d['stiffness'] for d in drives.values()), max(d['damping'] for d in drives.values()),
                max(d['damping'] / d['stiffness'] for d in drives.values()))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--arm-usd', default='core/assets/robots/fanuc_crx10ial/crx10ial.usd')
    parser.add_argument('--gripper-usd', default='core/assets/robots/robotiq_2f85/Robotiq_2F_85_flattened.usd')
    parser.add_argument('--output', default=str(Path(__file__).resolve().parents[1] / 'configs' / 'crx10ial_kinematics.json'))
    args = parser.parse_args()
    data = build(args.probe, args.arm_usd, args.gripper_usd)
    Path(args.output).write_text(json.dumps(data, indent=1) + '\n')
    print(json.dumps({'output': args.output, **data['provenance']}, indent=1))


if __name__ == '__main__':
    main()
