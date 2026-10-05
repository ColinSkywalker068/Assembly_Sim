"""Ground-truth guided physical pick-and-place demo for the small shulker."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from core.robots.controls import _apply_named_positions
from core.robots.kinematics import Gripper, MIMIC_2F85, quat_to_R
from .builder import build_stack_sawtooth_scene
from .config import load_task_config
from .runtime import require_external_output
from .teleop import TeleopDriver


def motion_config():
    config = load_task_config()
    data = deepcopy(config.workcell.data)
    data['physics'].update(dt=1/240, static_friction=2., dynamic_friction=1.5,
                           restitution=0., linear_damping=.5, angular_damping=1.)
    return replace(config, density=20., workcell=replace(config.workcell, data=data))


def grasp_pose(config, fragment, root_position, reference_rotation=None, is_base=None):
    pose = np.eye(4)
    # Local +X approaches downward; jaws close along world Y.
    pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    if reference_rotation is not None:
        # Swapping the two jaws gives the same grasp. Avoid a needless half-turn.
        candidates = (pose[:3, :3].copy(),
                      np.diag([-1., -1., 1.]) @ pose[:3, :3])
        pose[:3, :3] = min(candidates, key=lambda rotation:
            Rotation.from_matrix(rotation @ reference_rotation.T).magnitude())
    pose[:3, 3] = root_position
    # The fingertips extend past the nominal TCP as the gripper closes.
    # Keep their lower edge above blue's solid bottom layer.
    from .conditions import oriented_fragment_bounds
    _, upper = oriented_fragment_bounds(fragment, config.unit_size, fragment.initial_pose.orientation_wxyz)
    if is_base is None:
        is_base = fragment.name == 'FragmentA'
    pose[2, 3] += upper[2] + (.004 if is_base else -.004)
    return pose


def configure_gripper_coupling(stage, config):
    """Use native PhysX constraints for the real Robotiq linkage, before reset."""
    from pxr import PhysxSchema, Usd, UsdPhysics
    robot = config.workcell.robot('right')
    joints = {prim.GetName():prim for prim in Usd.PrimRange(stage.GetPrimAtPath(robot.gripper_path))
              if prim.IsA(UsdPhysics.RevoluteJoint)}
    master = joints['finger_joint']
    for name, sign in MIMIC_2F85.items():
        if name=='finger_joint' or sign==0:
            continue
        joint = joints[name]
        revolute = UsdPhysics.RevoluteJoint(joint)
        revolute.CreateLowerLimitAttr(min(0.,sign*75.))
        revolute.CreateUpperLimitAttr(max(0.,sign*75.))
        instances = [schema.split(':',1)[1] for schema in joint.GetAppliedSchemas()
                     if schema.startswith('PhysxMimicJointAPI:')]
        api = PhysxSchema.PhysxMimicJointAPI.Apply(joint,instances[0] if instances else 'rotX')
        api.CreateReferenceJointRel().SetTargets([master.GetPath()])
        api.CreateGearingAttr(-sign)
        api.CreateOffsetAttr(0.)
        drive = UsdPhysics.DriveAPI(joint,'angular')
        drive.GetStiffnessAttr().Set(0.)
        drive.GetDampingAttr().Set(0.)
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(robot.prim_path))
    articulation.CreateSolverPositionIterationCountAttr(64)
    articulation.CreateSolverVelocityIterationCountAttr(16)


def gripper_model(handles, output):
    from pxr import Usd, UsdGeom, UsdPhysics
    joints, links = [], []
    cache = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy'])
    # Joint local frames are independent of the robot pose. Read open pad bounds
    # from the source asset so they share the gripper's own coordinate frame.
    source = Usd.Stage.Open(str(handles.config.workcell.asset_path('gripper_usd')))
    for prim in source.Traverse():
        name = prim.GetName()
        if name in MIMIC_2F85:
            joint = UsdPhysics.Joint(prim)
            record = dict(path=str(prim.GetPath()), axis=prim.GetAttribute('physics:axis').Get())
            for suffix in ('0', '1'):
                record['body'+suffix] = [str(p) for p in getattr(joint, 'GetBody'+suffix+'Rel')().GetTargets()]
                record['pos'+suffix] = list(getattr(joint, 'GetLocalPos'+suffix+'Attr')().Get())
                q = getattr(joint, 'GetLocalRot'+suffix+'Attr')().Get()
                record['rot'+suffix] = [*q.GetImaginary(), q.GetReal()]
            joints.append(record)
        if name in ('base_link', 'left_inner_finger'):
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            links.append(dict(path=str(prim.GetPath()), pos=[0,0,0],
                              bounds=[list(bounds.GetMin()), list(bounds.GetMax())]))
    path = output / 'gripper_geometry.json'
    path.write_text(json.dumps({'grip': dict(dof_names=list(MIMIC_2F85), joints=joints, links=links)}, indent=2))
    return Gripper(str(path))


class ScriptedMotion:
    def __init__(self, handles, output, assembly_order=('Blue', 'Green')):
        from isaacsim.core.prims import RigidPrim
        from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade
        self.handles, self.output = handles, output
        self.driver = TeleopDriver(handles)
        controller = self.driver.robot.robot.get_articulation_controller()
        kp = np.zeros(self.driver.robot.robot.num_dof)
        kd = np.zeros(self.driver.robot.robot.num_dof)
        for name in ('finger_joint','left_outer_finger_joint','right_outer_finger_joint'):
            idx = self.driver.robot.robot.get_dof_index(name)
            kp[idx],kd[idx] = 1000.,50.
        kp[self.driver.indices], kd[self.driver.indices] = 100000., 1000.
        controller.set_gains(kp,kd)
        efforts = np.full(self.driver.robot.robot.num_dof,10.)
        efforts[self.driver.indices] = 1000.
        controller.set_max_efforts(efforts)
        self.pose = self.driver.model.flange(self.driver.arm_target) @ self.driver.tool
        self.angle = 0.
        self.gripper = gripper_model(handles, output)
        self.events = []
        if assembly_order not in (('Blue', 'Green'), ('Green', 'Blue')):
            raise ValueError('invalid assembly order')
        self.assembly_order = assembly_order
        self.views = {}
        material = UsdShade.Material(handles.stage.GetPrimAtPath('/World/Looks/ContactMaterial'))
        from core.workcell.physics import bind_physics
        # Apply the same high-friction contact material to actual gripper colliders.
        for prim in Usd.PrimRange(handles.stage.GetPrimAtPath(self.driver.robot.root_path)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                bind_physics(prim, material)
                with Sdf.ChangeBlock():
                    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    api.CreateRestOffsetAttr(0.)
                    api.CreateContactOffsetAttr(.0003)
        for fragment in handles.config.fragments:
            path = '/World/Task/Fragments/' + fragment.name
            view = RigidPrim(prim_paths_expr=path, name='scripted_'+fragment.name)
            view.initialize()
            self.views[fragment.name] = view
            for prim in Usd.PrimRange(handles.stage.GetPrimAtPath(path)):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    with Sdf.ChangeBlock():
                        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                        api.CreateRestOffsetAttr(0.)
                        api.CreateContactOffsetAttr(.0003)

    def tick(self):
        driver = self.driver
        if not self.handles.world.is_playing():
            self.handles.world.play()
        _apply_named_positions(driver.robot, driver.robot.arm_dof_indices, driver.arm_target)
        names = ('finger_joint','left_outer_finger_joint','right_outer_finger_joint')
        _apply_named_positions(driver.robot,
                               [driver.robot.robot.get_dof_index(name) for name in names],
                               [self.angle,0.,0.])
        for _ in range(4):
            self.handles.world.step(render=False)

    def hold(self, seconds):
        for _ in range(round(seconds*60)):
            self.tick()

    def move(self, target, seconds=2.):
        print('MOVE', target[:3,3].tolist(), 'seconds', seconds, flush=True)
        start = self.pose.copy()
        rotation = Slerp([0,1], Rotation.from_matrix([start[:3,:3], target[:3,:3]]))
        for step in range(1, round(seconds*60)+1):
            fraction = step / round(seconds*60)
            smooth = fraction*fraction*(3-2*fraction)
            pose = np.eye(4)
            pose[:3,:3] = rotation(smooth).as_matrix()
            pose[:3,3] = (1-smooth)*start[:3,3] + smooth*target[:3,3]
            q, ep, er = self.driver.model.ik(pose @ np.linalg.inv(self.driver.tool),
                                            self.driver.arm_target, iters=150, tol=2e-5)
            if not np.isfinite(q).all() or ep > .0003 or er > .004:
                raise RuntimeError(f'IK failed: position={ep}, rotation={er}')
            previous = self.driver.arm_target.copy()
            steps = max(1, int(np.ceil(np.max(np.abs(q-previous))/.01)))
            if steps > 100:
                raise RuntimeError('IK solution jumped to another joint branch')
            for substep in range(1,steps+1):
                self.driver.arm_target = previous+(q-previous)*substep/steps
                self.tick()
        self.pose = target.copy()
        self.hold(.3)
        measured = self.driver.robot.robot.get_joint_positions()[self.driver.indices]
        actual = self.driver.model.flange(measured) @ self.driver.tool
        error = float(np.linalg.norm(actual[:3,3]-target[:3,3]))
        print('TRACKING_ERROR', error, flush=True)
        if error > .008:
            raise RuntimeError(f'Arm did not track the waypoint: {error} m')

    def grip(self, angle, seconds=1.):
        start = self.angle
        for step in range(1, round(seconds*60)+1):
            self.angle = start + (angle-start)*step/round(seconds*60)
            self.tick()
        self.hold(.5)
        positions = self.driver.robot.robot.get_joint_positions()
        print('GRIPPER', 'command', self.angle,
              {name:float(positions[idx]) for name,idx in zip(self.driver.robot.gripper_dof_names,
                                                           self.driver.robot.gripper_dof_indices)},flush=True)

    def object_pose(self, name):
        positions, orientations = self.views[name].get_world_poses()
        return np.asarray(positions[0], dtype=float), np.asarray(orientations[0], dtype=float)

    def event(self, label, capture=True):
        values = {name: dict(position=p.tolist(), quaternion_wxyz=q.tolist())
                  for name in self.views for p,q in [self.object_pose(name)]}
        self.events.append(dict(label=label, objects=values))
        (self.output/'events.json').write_text(json.dumps(self.events,indent=2))
        print(label, json.dumps(values), flush=True)
        if capture:
            from core.workcell.cameras import capture_rgb
            capture_rgb(self.handles.cameras.agent_path, self.output/(label+'.png'), resolution=(1920,1440))
            self.handles.world.play()

    def execute(self):
        config = self.handles.config
        self.event('00_initial')
        safe = self.pose.copy()
        safe[2,3] = max(safe[2,3],1.12)
        self.move(safe)
        by_color = {'Blue': config.fragment('FragmentA'), 'Green': config.fragment('FragmentB')}
        ordered = [by_color[color] for color in self.assembly_order]
        base_fragment = ordered[0]
        for index, fragment in enumerate(ordered):
            name = fragment.name
            root, quat = self.object_pose(name)
            if np.linalg.norm(quat_to_R(quat)-quat_to_R(fragment.initial_pose.orientation_wxyz)) > .05:
                raise RuntimeError('Initial fragment orientation changed unexpectedly')
            grasp = grasp_pose(config, fragment, root, reference_rotation=self.pose[:3, :3], is_base=index==0)
            hover = grasp.copy(); hover[2,3] = 1.02
            self.move(hover, 3.)
            self.move(grasp, 2.)
            central_post = index == 0 and fragment.name == 'FragmentA'
            width = (config.unit_size if central_post else 3*config.unit_size)-config.collider_clearance
            self.grip(self.gripper.f_for_width(width,squeeze=.001 if index==0 else .004))
            self.event(f'{index+1}0_closed')
            self.move(hover,2.)
            lifted, lifted_q = self.object_pose(name)
            self.event(f'{index+1}1_lifted')
            if lifted[2]-root[2] < .12:
                raise RuntimeError(f'{name} was not physically lifted; stopping')
            if np.linalg.norm(quat_to_R(lifted_q)[:2,2]) > .015:
                raise RuntimeError(f'{name} tilted during the straight top-down grasp')
            # Query GT after gripping to preserve the actual object-to-hand offset.
            goal = np.asarray(fragment.goal_pose.position,dtype=float)
            goal_rotation = quat_to_R(fragment.goal_pose.orientation_wxyz)
            if index:
                base, base_q = self.object_pose(base_fragment.name)
                if np.linalg.norm(quat_to_R(base_q)-quat_to_R(base_fragment.goal_pose.orientation_wxyz)) > .04:
                    raise RuntimeError('Base placement tilt exceeds mating tolerance')
                goal = base + np.asarray(fragment.goal_pose.position)-np.asarray(base_fragment.goal_pose.position)
                goal_rotation = quat_to_R(base_q)
            _, lifted_q = self.object_pose(name)
            # Keep approach vertical. Align yaw only; never compensate with tilt.
            current_rotation = quat_to_R(lifted_q)
            yaw = np.arctan2(goal_rotation[1,0],goal_rotation[0,0])-np.arctan2(current_rotation[1,0],current_rotation[0,0])
            delta = Rotation.from_euler('z',yaw).as_matrix()
            seat = self.pose.copy()
            seat[:3,:3] = delta @ self.pose[:3,:3]
            seat[:3,3] = goal + delta @ (self.pose[:3,3]-lifted)
            transit = seat.copy(); transit[2,3] = 1.02
            self.move(transit,3.)
            self.event(f'{index+1}15_above_target')
            # Refine alignment at safe height after any slip during transit.
            current, current_q = self.object_pose(name)
            seat = self.pose.copy()
            seat[:3,3] += goal-current
            aligned = seat.copy(); aligned[2,3] = 1.02
            self.move(aligned,1.5)
            near = seat.copy(); near[2,3] += .06
            self.move(near,2.)
            self.move(seat,3.)
            self.event(f'{index+1}2_seated')
            self.grip(0.)
            retreat = seat.copy(); retreat[2,3] = 1.06
            self.move(retreat,2.)
            self.event(f'{index+1}3_released')
        park = grasp_pose(config,config.fragments[0],(.42,0.,.75),
                          reference_rotation=self.pose[:3, :3])
        park[2,3] = 1.12
        self.move(park,3.)
        self.hold(2.)
        self.event('30_final')
        from pxr import Gf, UsdGeom
        from core.workcell.cameras import _look_at_quaternion, capture_rgb
        from core.workcell.environment import set_transform
        camera = UsdGeom.Camera.Define(self.handles.stage,'/World/Cameras/AssemblyDetail')
        position = (-.25,-.30,1.15)
        target = (*base_fragment.goal_pose.position[:2], config.table_top_z + 1.5*config.unit_size)
        set_transform(camera,position,_look_at_quaternion(tuple(b-a for a,b in zip(position,target))))
        camera.CreateFocalLengthAttr(35.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100.))
        capture_rgb(str(camera.GetPath()),self.output/'assembled_detail.png',resolution=(3840,2880))
        self.handles.world.pause()
        blue,bq = self.object_pose('FragmentA'); green,gq = self.object_pose('FragmentB')
        base, _ = self.object_pose(base_fragment.name)
        expected_relative = np.asarray(config.fragment('FragmentB').goal_pose.position)-np.asarray(config.fragment('FragmentA').goal_pose.position)
        errors = dict(base_center_error_m=float(np.linalg.norm(base[:2]-np.asarray(base_fragment.goal_pose.position[:2]))),
                      base_height_error_m=float(abs(base[2]-base_fragment.goal_pose.position[2])),
                      mating_position_error_m=float(np.linalg.norm(green-blue-expected_relative)),
                      blue_orientation_error=float(np.linalg.norm(quat_to_R(bq)-quat_to_R(config.fragment('FragmentA').goal_pose.orientation_wxyz))),
                      green_orientation_error=float(np.linalg.norm(quat_to_R(gq)-quat_to_R(config.fragment('FragmentB').goal_pose.orientation_wxyz))))
        errors['success'] = (errors['base_center_error_m'] < .003 and
                             errors['base_height_error_m'] < .003 and
                             errors['mating_position_error_m'] < .003 and
                             max(errors['blue_orientation_error'],errors['green_orientation_error']) < .05)
        (self.output/'result.json').write_text(json.dumps(errors,indent=2))
        print('RESULT',json.dumps(errors),flush=True)
        if not errors['success']:
            raise RuntimeError('Assembly did not meet measured pose tolerances')


def run_scripted(output_root, headless=True):
    root = require_external_output(Path(output_root))
    output = root / datetime.now().strftime('scripted_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    handles = build_stack_sawtooth_scene(motion_config(),headless=headless,
                                        stage_setup=configure_gripper_coupling)
    try:
        motion = ScriptedMotion(handles,output)
        motion.execute()
        from pxr import UsdGeom
        from core.workcell.environment import set_transform
        for name in motion.views:
            position, orientation = motion.object_pose(name)
            set_transform(UsdGeom.Xform(handles.stage.GetPrimAtPath('/World/Task/Fragments/'+name)),
                          position,orientation)
        if not handles.stage.GetRootLayer().Export(str(output/'final_scene.usda')):
            raise RuntimeError('Final stage export failed')
    except Exception as error:
        import traceback
        traceback.print_exc()
        (output/'failure.txt').write_text(str(error))
        raise
    finally:
        print('OUTPUT',output,flush=True)
        handles.app.close()
    return 0
