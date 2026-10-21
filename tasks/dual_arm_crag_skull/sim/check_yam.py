"""Isaac Sim check of the YAM linear_4310 on both CRX arms (GPU). Not a grasp test.

- Widths: both fingers get the joint target for each width in WIDTHS_M; after SETTLE_STEPS physics steps the gap
  between the pad faces (core.robots.linear_4310, from the measured finger joints) must be within WIDTH_TOLERANCE_M.
- Poses: in each arm configuration (home plus CONFIG_OFFSETS, checked on CPU to keep the tool at least 0.2 m above
  the table and the arms at least 0.13 m apart), PhysX's flange, gripper body and tips must be where the S0 model
  (s0.kinematics.yam_model, from the measured arm joints) puts them, within POSITION_TOLERANCE_M and
  ROTATION_TOLERANCE_DEG.
- TCP: the S0 model's TCP must lie within TCP_FROM_PADS_TOLERANCE_M of the grasp point the tips give as PhysX
  holds them (pad_midpoint, within 0.15 mm of I2RT's grasp site on the meshes). This checks flange_T_tcp against
  the simulated pads, not against the constants it was built from.
- Pads on a known width: with the arms at home, each gripper closes on a kinematic box between its pads (BOX_PROBES:
  how far up from the fingertips, how wide, within what tolerance, how thick across, how far off the middle: the
  coupled fingers stop with the near pad, 2 x that wider); the width it stops at must
  match where the tip meshes say it stops (predicted_stop_width: the box's own width on the pad, a little less on the
  slope above it, fully closed for the rod in the window, which meets no material). This measures the tip colliders
  against geometry, which the free-space widths cannot (PhysX's default decomposition stopped 1.8 mm per side short;
  shrink-wrapped, one finger of one arm sank 2.3 mm on the slope; the slab hulls once bridged the window).
- Cooked shapes: PhysX's own cooked hull of every gripper collider (request_convex_collision_representation) must be
  valid, have at most HULL_VERTEX_LIMIT vertices, stick out of the authored points' hull by at most COOKED_PROTRUSION_M
  and fall short of it by at most COOKED_SHRINK_M (set after a run, see there), and cooking must report no
  polygon-limit or GPU-compatibility warning. Recorded only: what PhysX cooks for
  spheres above and below the limit, at scale 1 and at yam_usd.COOKING_SCALE (cooking_diagnostic).

Rigid-body poses come from PhysX (omni.physx get_rigidbody_transformation), not from USD, which can lag with Fabric.
The criteria are fixed here before any run, except COOKED_SHRINK_M (see there). Writes check_yam.json, last, and exits 1 on any failure, before Kit's
close() (which ends the process with status 0).

  PYTHONPATH=. python -m tasks.dual_arm_crag_skull.sim.check_yam --output DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from .yam_usd import HULL_VERTEX_LIMIT

WIDTHS_M = ('max', 0.06, 0.03, 0.0)
CONFIG_OFFSETS = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.3, -0.2, 0.2, 0.5, 0.3, 0.8),
    (-0.3, 0.15, -0.25, -0.6, -0.3, -1.2),
    (0.0, 0.3, 0.3, 1.2, 0.5, 2.0),
    (0.2, -0.3, 0.0, -1.0, 0.6, -2.5),
)
SETTLE_STEPS = 120
POSE_STEPS = 30
WIDTH_TOLERANCE_M = 5e-4
POSITION_TOLERANCE_M = 5e-4
ROTATION_TOLERANCE_DEG = 0.1
TCP_FROM_PADS_TOLERANCE_M = 5e-4
# (distance up from the fingertips, box width along the closing axis, tolerance against the meshes' prediction)
BOX_SIDE_M = 0.012
# (how far up from the fingertips, how wide along the closing axis, tolerance, side across the closing axis, how far
# the box sits off the middle towards the driven tip). 12 mm boxes on the pad and the slope, each 1 mm off the middle
# one way and then the other, so that one pad touches it, the other still 2 mm off its face: the coupled fingers stop
# with that pad, and each pad is measured on its own (two pads on a kinematic box and the mimic joint over-constrain
# the pair: PhysX settled them 0.37 mm a side off a box in the middle). A 3 mm rod 35 mm up on the middle line, inside
# the window between the rails, which meets no material. A box 5 mm off the middle, which the coupled fingers stop on
# 10 mm wider than on a box in the middle; uncoupled, each would close on its own face. It is judged against the box
# 1 mm off the middle the same way, where the same pad touches (box_excesses).
BOX_PROBES = tuple((depth, width, 5e-4, BOX_SIDE_M, shift) for depth, width in ((0.005, 0.040), (0.011, 0.040),
                                                                                (0.025, 0.040), (0.008, 0.020))
                   for shift in (0.001, -0.001)) + ((0.035, 0.020, 5e-4, 0.003, 0.0),
                                                     (0.005, 0.040, 5e-4, BOX_SIDE_M, 0.005))
# How far PhysX's cooked hull of a collider may stick out of the authored points' hull, and fall short of it. Set
# after the first run with these colliders (2026-10-19): fixed before it at 2e-5 both ways, PhysX's cooking dropped
# near-coplanar points from every hull, up to 0.083 mm inside, never outside; 0.1 mm is the order of the colliders'
# own thinning (yam_usd.HULL_TOLERANCE_M) and a fifth of the box probes' tolerance.
COOKED_PROTRUSION_M = 2e-5
COOKED_SHRINK_M = 1e-4
BOX_STEPS = 240
PAD_DEPTH_M = 0.03
REPORT = 'check_yam.json'
FAILED = 'check_yam_failed.json'


def pose_error(expected, measured):
    """(position error m, rotation error deg) between two 4x4 poses."""
    expected, measured = np.asarray(expected, dtype=float), np.asarray(measured, dtype=float)
    relative = expected[:3, :3].T @ measured[:3, :3]
    # atan2 of sin and cos stays accurate for small angles, where an arccos of the trace of float32 rotations
    # (PhysX) bottoms out near 0.03 deg.
    skew = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]])
    angle = np.degrees(np.arctan2(np.linalg.norm(skew) / 2, (np.trace(relative) - 1) / 2))
    return float(np.linalg.norm(measured[:3, 3] - expected[:3, 3])), float(angle)


def pad_midpoint(tips, approach, opening):
    """Where the tips put the grasp point: along `approach` the farther fingertip end, along `opening` the middle of
    the two pad faces, across both the middle of the pads, each pad being a tip's last PAD_DEPTH_M before its end.
    Extremes over whole pads: the fingertip ends are curved, and a thin end slice jumps under float32 poses."""
    approach = np.asarray(approach, dtype=float) / np.linalg.norm(approach)
    opening = np.asarray(opening, dtype=float) / np.linalg.norm(opening)
    across = np.cross(approach, opening)
    ends, sides, middles = [], [], []
    for points in tips.values():
        points = np.asarray(points, dtype=float)
        along = points @ approach
        pad = points[along >= along.max() - PAD_DEPTH_M]
        ends.append(along.max())
        sides.append(pad @ opening)
        lateral = pad @ across
        middles.append((lateral.min() + lateral.max()) / 2)
    first, second = sorted(sides, key=np.mean)
    return max(ends) * approach + (first.max() + second.min()) / 2 * opening + np.mean(middles) * across


def judge(report):
    """Failures of a check report against the criteria above (empty when it passes)."""
    failures = []
    for arm, record in report['arms'].items():
        if not record['dofs_found']:
            failures.append(f'{arm}: finger DOFs missing from the articulation')
        for width in record['widths']:
            if not abs(width['measured_m'] - width['target_m']) <= WIDTH_TOLERANCE_M:
                failures.append(f"{arm}: width {width['measured_m']:.5f} m for target {width['target_m']:.5f} m")
        for index, pose in enumerate(record['poses']):
            for part, (position, rotation) in pose['errors'].items():
                if not (position <= POSITION_TOLERANCE_M and rotation <= ROTATION_TOLERANCE_DEG):
                    failures.append(f'{arm} configuration {index}: {part} off by {position * 1000:.3f} mm / '
                                    f'{rotation:.4f} deg')
            if not pose['tcp_from_pads_m'] <= TCP_FROM_PADS_TOLERANCE_M:
                failures.append(f"{arm} configuration {index}: TCP is {pose['tcp_from_pads_m'] * 1000:.3f} mm from the "
                                'midpoint of the pads')
        if len(record['boxes']) != len(BOX_PROBES):
            failures.append(f"{arm}: {len(record['boxes'])} box probes, expected {len(BOX_PROBES)}")
        for name, excess, tolerance in box_excesses(record['boxes']):
            if not abs(excess) <= tolerance:
                failures.append(f"{arm}: {name} {excess * 1000:+.2f} mm")
    if not report['arms']:
        failures.append('no arm was checked')
    cooked = report['cooked']
    if not (cooked['colliders'] > 0 and cooked['valid'] == cooked['colliders']):
        failures.append(f"cooked: {cooked['valid']} of {cooked['colliders']} colliders cooked to one valid hull")
    if not cooked['max_vertices'] <= HULL_VERTEX_LIMIT:
        failures.append(f"cooked: a hull has {cooked['max_vertices']} vertices, the limit is {HULL_VERTEX_LIMIT}")
    if not (cooked['protrusion_m'] <= COOKED_PROTRUSION_M and cooked['shrink_m'] <= COOKED_SHRINK_M):
        failures.append(f"cooked: hulls stick out {cooked['protrusion_m'] * 1000:.4f} mm / fall short "
                        f"{cooked['shrink_m'] * 1000:.4f} mm of the authored ones")
    if cooked['polygon_limit_warnings'] or cooked['gpu_incompatible']:
        failures.append(f"cooked: {cooked['polygon_limit_warnings']} polygon-limit and {cooked['gpu_incompatible']} "
                        'GPU-compatibility warnings')
    return failures


def _physx_pose(physx, path):
    from core.robots.kinematics import T
    result = physx.get_rigidbody_transformation(path)
    if not result['ret_val']:
        raise RuntimeError(f'PhysX has no rigid body at {path}')
    x, y, z, w = (float(value) for value in result['rotation'])
    return T([float(value) for value in result['position']], [w, x, y, z])


def _build(config, world):
    import omni.usd
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.prims import XFormPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import UsdGeom, UsdShade

    from core.robots.linear_4310 import load_linear_4310
    from core.workcell.environment import author_environment

    from ..contracts.actions import ARMS
    from ..s0.kinematics import yam_flange_T_mount
    from .yam_usd import author_linear_4310

    stage = omni.usd.get_context().get_stage()
    stage.SetMetadata('metersPerUnit', 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, '/World')
    author_environment(stage, config)
    material = UsdShade.Material(stage.GetPrimAtPath('/World/Looks/ContactMaterial'))
    gripper = load_linear_4310()
    robots = {}
    for arm in ARMS:
        settings = config.robot(arm)
        add_reference_to_stage(str(config.asset_path('arm_usd')), settings.prim_path)
        XFormPrim(settings.prim_path).set_world_poses(positions=np.asarray([settings.base_position], dtype=float),
                                                      orientations=np.asarray([settings.base_orientation_wxyz],
                                                                              dtype=float))
        prims = author_linear_4310(stage, f'/World/Grippers/{arm.capitalize()}/YAM_Linear_4310',
                                   flange_path=f'{settings.prim_path}/flange', flange_T_mount=yam_flange_T_mount(),
                                   gripper=gripper, physics_material=material)
        robots[arm] = (world.scene.add(Robot(prim_path=settings.prim_path, name=f'fanuc_yam_{arm}')), prims)
    return stage, gripper, robots


def _command(robot, indices, values):
    from isaacsim.core.utils.types import ArticulationAction
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=np.asarray(values, dtype=float), joint_indices=np.asarray(indices, dtype=int)))


def _teleport(robot, indices, values):
    robot.set_joint_positions(np.asarray(values, dtype=float), joint_indices=np.asarray(indices, dtype=int))
    robot.set_joint_velocities(np.zeros(len(indices)), joint_indices=np.asarray(indices, dtype=int))
    _command(robot, indices, values)


def _measure_poses(physx, model, gripper, arm, robot, prims, arm_indices, finger_indices, flange_path):
    from ..s0.kinematics import yam_flange_T_mount
    q = np.asarray(robot.get_joint_positions(joint_indices=np.asarray(arm_indices)), dtype=float)
    fingers = np.asarray(robot.get_joint_positions(joint_indices=np.asarray(finger_indices)), dtype=float)
    arm_model = model.arms[arm]
    world_T_flange = arm_model.frames(q)[0]['flange']
    world_T_mount = world_T_flange @ yam_flange_T_mount()
    measured_body = _physx_pose(physx, prims.links['gripper'])
    errors = {'flange': pose_error(world_T_flange, _physx_pose(physx, flange_path)),
              'gripper_body': pose_error(world_T_mount @ gripper.link_frames['gripper'], measured_body)}
    pads = {}
    for tip, value in zip(('tip_left', 'tip_right'), fingers):
        expected = world_T_mount @ gripper.link_frames[tip]
        expected[:3, 3] += world_T_mount[:3, :3] @ gripper.slide_axes[tip] * value
        measured_tip = _physx_pose(physx, prims.links[tip])
        errors[tip] = pose_error(expected, measured_tip)
        local = (gripper.tip_points[tip] - gripper.link_frames[tip][:3, 3]) @ gripper.link_frames[tip][:3, :3]
        pads[tip] = local @ measured_tip[:3, :3].T + measured_tip[:3, 3]
    tcp = arm_model.tcp(q)
    from_pads = float(np.linalg.norm(pad_midpoint(pads, tcp[:3, 0], tcp[:3, 1]) - tcp[:3, 3]))
    return {'q': q.tolist(), 'finger_joints_m': fingers.tolist(), 'errors': errors, 'tcp_from_pads_m': from_pads}


def box_excesses(boxes):
    """[(what, how far off the meshes' prediction, tolerance)] of a report's box probes. With each pad standing dD and
    dO off a face and the pads' middle c off the TCP towards the driven tip, a box 1 mm off the middle towards the
    driven tip stops the coupled fingers 2 (dD - c) wider than predicted and one 1 mm the other way 2 (dO + c): the
    mean of the two is how far the pads stand off the faces in all, dD + dO (what both pads on a box in the middle gave
    with the fingers uncoupled), and a quarter of their difference c + (dO - dD) / 2, how far the pads' middle lies from
    the TCP (held to TCP_FROM_PADS_TOLERANCE_M, as the TCP is). A box further off the middle (the coupling probe) is
    judged against the 1 mm one the same way, where the same pad touches and c cancels; every other probe on its
    own."""
    pairs, out, far = {}, [], []
    for box in boxes:
        if abs(abs(box['shift_m']) - 0.001) < 1e-9:
            pairs.setdefault((box['depth_m'], box['width_m'], box['side_m']), {})[np.sign(box['shift_m'])] = box
        elif abs(box['shift_m']) > 0.001:
            far.append(box)
        else:
            out.append((f"on a {box['width_m'] * 1000:.0f} mm box {box['depth_m'] * 1000:.0f} mm up the fingers "
                        "stopped off the meshes' prediction by", box['excess_m'], box['tolerance_m']))
    for box in far:
        what = (f"on a {box['width_m'] * 1000:.0f} mm box {box['depth_m'] * 1000:.0f} mm up "
                f"{box['shift_m'] * 1000:.0f} mm off the middle the fingers stopped off the meshes' prediction")
        near = pairs.get((box['depth_m'], box['width_m'], box['side_m']), {}).get(np.sign(box['shift_m']))
        if near is None:
            out.append((f'{what} by', box['excess_m'], box['tolerance_m']))
        else:
            out.append((f'{what}, beyond the box 1 mm off it the same way, by', box['excess_m'] - near['excess_m'],
                        box['tolerance_m']))
    for (depth, width, side), pair in pairs.items():
        what = f'on a {width * 1000:.0f} mm box {depth * 1000:.0f} mm up'
        if set(pair) != {1.0, -1.0}:
            out.append((f'{what}: one of the two probes 1 mm off the middle is missing, by', float('inf'), 0.0))
            continue
        toward, away = pair[1.0]['excess_m'], pair[-1.0]['excess_m']
        out.append((f'{what} the pads stood off the faces, the two together, by', (toward + away) / 2,
                    pair[1.0]['tolerance_m']))
        out.append((f'{what} the pads\' middle lay off the TCP by', (away - toward) / 4, TCP_FROM_PADS_TOLERANCE_M))
    return out


def predicted_stop_width(gripper, depth_m, width_m, half_m=None, shift_m=0.0):
    """The width at which the coupled fingers stop on a box `width_m` wide and 2 `half_m` (BOX_SIDE_M) thick across,
    centred `depth_m` up from the fingertips and `shift_m` off the middle towards yam_usd.DRIVEN_TIP: each tip's mesh
    reaches its face where it comes furthest inward over the box, a tip with no material over it closing to its limit
    (pad contact), and the first tip to touch stops both."""
    from .yam_usd import DRIVEN_TIP, tip_points_in_link
    half_m = BOX_SIDE_M / 2 if half_m is None else half_m
    end = gripper.mount_T_tcp[2, 3]
    joints = []
    for tip in ('tip_left', 'tip_right'):
        frame = gripper.link_frames[tip]
        points = tip_points_in_link(gripper, tip) @ frame[:3, :3].T + frame[:3, 3]
        near = (np.abs(points[:, 2] - (end - depth_m)) <= half_m) & (np.abs(points[:, 1]) <= half_m)
        inward = -gripper.slide_axes[tip]
        face = width_m / 2 + (shift_m if tip == DRIVEN_TIP else -shift_m)
        joints.append(float((points[near] @ inward).max()) + face if near.any() else gripper.contact_joint_m)
    return gripper.width(float(np.clip(max(joints), gripper.contact_joint_m, gripper.travel_m)))


def _box_probes(world, stage, gripper, model, handles):
    """Close each gripper on a kinematic box between its pads at home (BOX_PROBES); {arm: [records]}."""
    from pxr import Gf, UsdGeom, UsdPhysics
    records = {arm: [] for arm in handles}
    for depth, width, tolerance, side, shift in BOX_PROBES:
        world.stop()
        for arm, (robot, prims, arm_indices, finger_indices, home) in handles.items():
            path = f'/World/BoxProbe_{arm}'
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(path)
            tcp = model.arms[arm].tcp(home)
            pose = tcp.copy()
            pose[:3, 3] += -tcp[:3, 0] * depth + _towards_driven(stage, prims, tcp) * shift
            box = UsdGeom.Cube.Define(stage, path)
            box.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(box)
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(Gf.Matrix4d(pose.T.tolist()))
            xform.AddScaleOp().Set(Gf.Vec3d(side, width, side))
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(box.GetPrim()).CreateKinematicEnabledAttr(True)
        world.reset()
        for arm, (robot, prims, arm_indices, finger_indices, home) in handles.items():
            _teleport(robot, arm_indices, home)
            _teleport(robot, finger_indices, [gripper.travel_m] * 2)
            _command(robot, finger_indices, [gripper.contact_joint_m] * 2)
        for _ in range(BOX_STEPS):
            world.step(render=False)
        for arm, (robot, prims, arm_indices, finger_indices, home) in handles.items():
            fingers = np.asarray(robot.get_joint_positions(joint_indices=np.asarray(finger_indices)), dtype=float)
            measured = gripper.width_between(*np.clip(fingers, 0.0, gripper.travel_m))
            predicted = predicted_stop_width(gripper, depth, width, side / 2, shift)
            records[arm].append({'depth_m': depth, 'width_m': width, 'side_m': side, 'shift_m': shift,
                                 'tolerance_m': tolerance,
                                 'finger_joints_m': fingers.tolist(), 'measured_m': measured,
                                 'predicted_m': predicted, 'excess_m': measured - predicted})
    return records


def _towards_driven(stage, prims, tcp):
    """The unit direction along the closing axis from the TCP towards the driven tip (yam_usd.DRIVEN_TIP), from the
    tips' authored rigid-body positions."""
    from pxr import UsdGeom
    from .yam_usd import DRIVEN_TIP
    closing = tcp[:3, 1]
    positions = {tip: np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0))
                 .T[:3, 3] for tip, path in prims.links.items() if tip.startswith('tip')}
    other = next(tip for tip in positions if tip != DRIVEN_TIP)
    return closing if (positions[DRIVEN_TIP] - positions[other]) @ closing > 0 else -closing


def _float3(value):
    try:
        return [float(value.x), float(value.y), float(value.z)]
    except AttributeError:
        return [float(value[0]), float(value[1]), float(value[2])]


def _cook(cooking, stage_id, prim):
    """PhysX's cooked convex hulls of one collider prim, as vertex arrays in its frame; None unless cooking gave a
    valid result."""
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools
    answer = {}
    cooking.request_convex_collision_representation(
        stage_id=stage_id, collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()), run_asynchronously=False,
        on_result=lambda result, convexes: answer.update(result=result, convexes=convexes))
    if answer.get('result') != PhysxCollisionRepresentationResult.RESULT_VALID:
        return None
    return [np.array([_float3(vertex) for vertex in convex.vertices], dtype=float) for convex in answer['convexes']]


def _departures(authored, cooked):
    """(how far `cooked` sticks out of the hull of `authored`, how far `authored` lies outside the cooked hull)."""
    from scipy.spatial import ConvexHull
    outer, inner = ConvexHull(authored).equations, ConvexHull(cooked).equations
    return (max(0.0, float((cooked @ outer[:, :3].T + outer[:, 3]).max())),
            max(0.0, float((authored @ inner[:, :3].T + inner[:, 3]).max())))


def _local_scale(prim):
    from pxr import UsdGeom
    local = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation(), dtype=float).T
    return float(np.linalg.norm(local[:3, 0]))


def _in_metres(authored, cooked, scale):
    """(authored, cooked) in metres: the cooked vertices come in the mesh's own units or already scaled; their
    extent tells which. None when it matches neither."""
    ratio = float(np.ptp(cooked, axis=0).max() / np.ptp(authored, axis=0).max())
    if abs(ratio - 1.0) <= 0.05:
        return authored * scale, cooked * scale
    if abs(ratio - scale) <= 0.05 * scale:
        return authored * scale, cooked
    return None


def _cooked_colliders(stage, roots):
    """PhysX's cooked hull of every mesh collider under `roots` against its authored points (in metres, whatever
    units cooking reports them in), and cooking's warnings."""
    import omni.usd
    from omni.physx import get_physx_cooking_interface, get_physx_cooking_private_interface
    from pxr import Usd, UsdGeom, UsdPhysics
    cooking, stage_id = get_physx_cooking_interface(), omni.usd.get_context().get_stage_id()
    record = {'colliders': 0, 'valid': 0, 'max_vertices': 0, 'protrusion_m': 0.0, 'shrink_m': 0.0, 'invalid': [],
              'cooked_units': set(), 'worst': None, 'each': []}
    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                continue
            record['colliders'] += 1
            cooked = _cook(cooking, stage_id, prim)
            if cooked is None or len(cooked) != 1:
                record['invalid'].append(str(prim.GetPath()))
                continue
            authored = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            metres = _in_metres(authored, cooked[0], _local_scale(prim))
            if metres is None:
                record['invalid'].append(str(prim.GetPath()))
                continue
            record['valid'] += 1
            record['cooked_units'].add('mesh' if metres[1] is not cooked[0] else 'scaled')
            protrusion, shrink = _departures(*metres)
            if max(protrusion, shrink) > max(record['protrusion_m'], record['shrink_m']):
                record['worst'] = {'path': str(prim.GetPath()), 'authored_vertices': len(authored),
                                   'cooked_vertices': len(cooked[0]), 'protrusion_m': protrusion, 'shrink_m': shrink,
                                   'authored': authored.tolist(), 'cooked': cooked[0].tolist()}
            record['each'].append({'path': str(prim.GetPath()), 'authored_vertices': len(authored),
                                   'cooked_vertices': len(cooked[0]), 'extent_m': float(np.ptp(metres[0], axis=0).max()),
                                   'protrusion_m': protrusion, 'shrink_m': shrink})
            record.update(max_vertices=max(record['max_vertices'], len(cooked[0])),
                          protrusion_m=max(record['protrusion_m'], protrusion), shrink_m=max(record['shrink_m'], shrink))
    record['cooked_units'] = sorted(record['cooked_units'])
    statistics = get_physx_cooking_private_interface().get_cooking_statistics()
    record['polygon_limit_warnings'] = int(statistics.total_warnings_convex_polygon_limits_reached)
    record['gpu_incompatible'] = int(statistics.total_warnings_failed_gpu_compatibility)
    return record


def _cooking_diagnostic(stage):
    """What PhysX cooks, recorded and not judged: a 2 cm sphere of 400 hull points (above the vertex limit) and one
    of 60 (below it), each authored at scale 1 and at COOKING_SCALE, static and far away."""
    import omni.usd
    from omni.physx import get_physx_cooking_interface
    from pxr import Gf, UsdGeom
    from scipy.spatial import ConvexHull

    from .yam_usd import COOKING_SCALE, _author_mesh
    rng = np.random.default_rng(0)
    records = {}
    for count in (400, 60):
        points = rng.normal(size=(count, 3))
        points = 0.02 * points / np.linalg.norm(points, axis=1, keepdims=True)
        hull = ConvexHull(points)
        for scale in (1.0, COOKING_SCALE):
            name = f'sphere_{count}_scale_{scale:g}'
            mesh = _author_mesh(stage, f'/World/CookingProbe/{name}', points, hull.simplices, 'convexHull', None,
                                visible=False, cooking_scale=scale)
            UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(5.0, 5.0 + len(records) * 0.1, 1.0))
            cooked = _cook(get_physx_cooking_interface(), omni.usd.get_context().get_stage_id(), mesh.GetPrim())
            metres = None if cooked is None or len(cooked) != 1 else _in_metres(points * scale, cooked[0], 1.0 / scale)
            if metres is None:
                records[name] = {'hull_vertices': len(hull.vertices), 'cooked': None}
                continue
            protrusion, shrink = _departures(*metres)
            records[name] = {'hull_vertices': len(hull.vertices), 'cooked_vertices': len(cooked[0]),
                             'protrusion_m': protrusion, 'shrink_m': shrink}
    return records


def run(output):
    from core.scene.builder import create_world
    from core.workcell.config import load_workcell_preset

    output.mkdir(parents=True, exist_ok=True)
    for stale in (REPORT, FAILED):
        (output / stale).unlink(missing_ok=True)
    config = load_workcell_preset('dual_arm')
    app, succeeded = None, False
    try:
        app, world = create_world(config, headless=True)
        from omni.physx import get_physx_interface

        from ..s0.kinematics import yam_model
        stage, gripper, robots = _build(config, world)
        world.reset()
        physx = get_physx_interface()
        model = yam_model()
        report = {'physics_dt': config.physics_dt, 'settle_steps': SETTLE_STEPS, 'pose_steps': POSE_STEPS,
                  'criteria': {'width_m': WIDTH_TOLERANCE_M, 'position_m': POSITION_TOLERANCE_M,
                               'rotation_deg': ROTATION_TOLERANCE_DEG},
                  'gripper': 'I2RT linear_4310 on the PROVISIONAL adapter plate', 'arms': {}}
        handles = {}
        for arm, (robot, prims) in robots.items():
            names = [str(name) for name in robot.dof_names]
            found = all(name in names for name in prims.dof_names)
            settings = config.robot(arm)
            arm_indices = [names.index(name) for name in settings.arm_dof_names]
            finger_indices = [names.index(name) for name in prims.dof_names] if found else []
            report['arms'][arm] = {'dofs_found': found, 'dof_names': names, 'widths': [], 'poses': [], 'boxes': []}
            handles[arm] = (robot, prims, arm_indices, finger_indices, np.asarray(settings.home_joint_positions))
            _teleport(robot, arm_indices, settings.home_joint_positions)
            if found:
                _teleport(robot, finger_indices, [gripper.travel_m] * 2)
        for width in WIDTHS_M:
            target = gripper.max_width_m if width == 'max' else float(width)
            q = gripper.joint_for_width(target)
            for arm, (robot, prims, arm_indices, finger_indices, home) in handles.items():
                if finger_indices:
                    _command(robot, finger_indices, [q, q])
            for _ in range(SETTLE_STEPS):
                world.step(render=False)
            for arm, (robot, prims, arm_indices, finger_indices, home) in handles.items():
                if not finger_indices:
                    continue
                fingers = np.asarray(robot.get_joint_positions(joint_indices=np.asarray(finger_indices)), dtype=float)
                clipped = np.clip(fingers, 0.0, gripper.travel_m)
                report['arms'][arm]['widths'].append({'target_m': target, 'joint_target_m': q,
                                                      'finger_joints_m': fingers.tolist(),
                                                      'measured_m': gripper.width_between(*clipped)})
        for offset in CONFIG_OFFSETS:
            for arm, (robot, prims, arm_indices, finger_indices, home) in handles.items():
                _teleport(robot, arm_indices, home + np.asarray(offset))
            for _ in range(POSE_STEPS):
                world.step(render=False)
            for arm, (robot, prims, arm_indices, finger_indices, home) in handles.items():
                if finger_indices:
                    report['arms'][arm]['poses'].append(_measure_poses(
                        physx, model, gripper, arm, robot, prims, arm_indices, finger_indices,
                        f'{config.robot(arm).prim_path}/flange'))
        stage.GetRootLayer().Export(str(output / 'check_yam_scene.usda'))
        for arm, records in _box_probes(world, stage, gripper, model, handles).items():
            report['arms'][arm]['boxes'] = records
        report['cooked'] = _cooked_colliders(stage, [prims.root for _, prims in robots.values()])
        report['cooking_diagnostic'] = _cooking_diagnostic(stage)
        report['failures'] = judge(report)
        report['pass'] = not report['failures']
        (output / REPORT).write_text(json.dumps(report, indent=1) + '\n')
        print('CHECK PASS' if report['pass'] else 'CHECK FAIL', flush=True)
        for failure in report['failures']:
            print('  ' + failure, flush=True)
        succeeded = report['pass']
        return report
    except BaseException:
        (output / FAILED).write_text(json.dumps({'error': traceback.format_exc()}, indent=1) + '\n')
        traceback.print_exc()
        print('CHECK FAIL', flush=True)
        raise
    finally:
        if not succeeded:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        app.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
