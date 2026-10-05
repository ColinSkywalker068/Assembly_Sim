"""Native keyboard Cartesian teleoperation; no attachments or object resets."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.robots.controls import _apply_named_positions, gripper_dof_targets
from core.robots.kinematics import Arm, MIMIC_2F85, T
from core.robots.teleop import CartesianInput, KeyboardSubscription

from .builder import build_stack_sawtooth_scene
from .config import load_task_config


TCP_OFFSET = .135  # grasp center along the Robotiq base's local +X approach axis


def arm_from_stage(stage, handle, settings):
    """Use the loaded asset's joint frames, not an external probe JSON."""
    from pxr import Usd, UsdPhysics

    joints = []
    by_name = {prim.GetName(): prim for prim in Usd.PrimRange(stage.GetPrimAtPath(handle.root_path))
               if prim.IsA(UsdPhysics.Joint)}
    for name in (*handle.arm_dof_names, 'joint_6_flange'):
        prim = by_name.get(name)
        if prim is None:
            raise RuntimeError(f'kinematic joint missing: {handle.root_path}/{name}')
        path = str(prim.GetPath())
        joint = UsdPhysics.Joint(prim)
        record = {'path': path}
        for suffix in ('0', '1'):
            record[f'pos{suffix}'] = list(getattr(joint, f'GetLocalPos{suffix}Attr')().Get())
            quat = getattr(joint, f'GetLocalRot{suffix}Attr')().Get()
            record[f'rot{suffix}'] = [*quat.GetImaginary(), quat.GetReal()]
        record['axis'] = prim.GetAttribute('physics:axis').Get() or 'X'
        joints.append(record)
    model = Arm({'arm': dict(joints=joints, dof_names=handle.arm_dof_names,
                            limits=(settings.arm_lower_limits, settings.arm_upper_limits))})
    model.base = T(settings.base_position, settings.base_orientation_wxyz)
    return model


class TeleopDriver:
    def __init__(self, handles, speed=.05, roll_speed=90.):
        self.robot = handles.workcell.robots['right']
        self.settings = handles.config.workcell.robot('right')
        self.model = arm_from_stage(handles.stage, self.robot, self.settings)
        self.indices = np.array(self.robot.arm_dof_indices)
        self.arm_target = self.robot.robot.get_joint_positions()[self.indices].copy()
        self.tool = np.eye(4)
        self.tool[0, 3] = TCP_OFFSET
        self.controls = CartesianInput(self.model.flange(self.arm_target) @ self.tool,
                                       speed=speed, roll_speed=np.deg2rad(roll_speed))
        self.gripper_fraction = 1.
        self.accepted_moves = 0
        self.rejected_moves = 0
        ctrl = self.robot.robot.get_articulation_controller()
        kp = np.full(self.robot.robot.num_dof, 100.)
        kd = np.full(self.robot.robot.num_dof, 10.)
        kp[self.indices], kd[self.indices] = 10000., 100.
        ctrl.set_gains(kp, kd)
        efforts = np.full(self.robot.robot.num_dof, 10.)
        efforts[self.indices] = 200.
        ctrl.set_max_efforts(efforts)

    def _accept(self, pose):
        measured = self.robot.robot.get_joint_positions()[self.indices]
        flange = pose @ np.linalg.inv(self.tool)
        q, ep, er = self.model.ik(flange, measured, iters=100)
        if (not np.isfinite(q).all() or ep > .0005 or er > .005
                or np.max(np.abs(q - measured)) > .15):
            self.rejected_moves += 1
            print(f'EEF move rejected: IK position error={ep:.6f} m, rotation error={er:.6f} rad, joint step={np.max(np.abs(q-measured)):.6f} rad.', flush=True)
            return False
        self.arm_target = q
        self.accepted_moves += 1
        return True

    def update(self, dt):
        measured = self.robot.robot.get_joint_positions()[self.indices]
        if not np.isfinite(measured).all():
            raise RuntimeError('Non-finite robot joint state; teleop stopped')
        if np.linalg.norm(measured - self.arm_target) > .2:
            self.controls.clear()
            self.arm_target = measured.copy()
            self.controls.pose = self.model.flange(measured) @ self.tool
            print('Motion stopped: arm cannot track the target. Release and press keys again.', flush=True)
        self.controls.update(dt, self._accept)
        _apply_named_positions(self.robot, self.robot.arm_dof_indices, self.arm_target)
        delta = np.clip(self.controls.gripper_open_fraction - self.gripper_fraction, -dt / 2, dt / 2)
        self.gripper_fraction += float(delta)
        targets = gripper_dof_targets(self.robot.gripper_dof_names, self.gripper_fraction,
                                     self.settings.gripper_open_radians, self.settings.gripper_closed_radians, MIMIC_2F85)
        _apply_named_positions(self.robot, self.robot.gripper_dof_indices,
                               [targets[name] for name in self.robot.gripper_dof_names])


def run_teleop(speed=.05, roll_speed=90., smoke_frames=0, headless=False):
    if smoke_frames < 0 or (headless and not smoke_frames):
        raise ValueError('headless teleop requires positive --smoke-frames')
    if not np.isfinite(speed) or speed <= 0 or not np.isfinite(roll_speed) or roll_speed <= 0:
        raise ValueError('speed and roll speed must be finite and positive')
    config = load_task_config()
    data = dict(config.workcell.data)
    data['physics'] = dict(config.workcell.physics, dt=1 / 480)
    data['render'] = dict(config.workcell.render, multi_gpu=False, max_gpu_count=1)
    config = replace(config, workcell=replace(config.workcell, data=data))
    handles = build_stack_sawtooth_scene(config, headless=headless)
    binding = None
    try:
        driver = TeleopDriver(handles, speed, roll_speed)
        if not headless:
            import carb.input
            import omni.appwindow
            from omni.kit.viewport.utility import get_active_viewport

            window = omni.appwindow.get_default_app_window()
            binding = KeyboardSubscription(carb.input.acquire_input_interface(), window.get_keyboard(), driver.controls)
            binding.subscribe()
            get_active_viewport().set_active_camera(handles.workcell.cameras.path('agent'))
        print('Keyboard teleop ready: W/S=-X/+X, D/A=+Y/-Y, arrows=+Z/-Z, 0=close, 1=open, R=local roll +90 degrees. Escape cancels motion. Focus the simulator window.', flush=True)
        frame = 0
        while handles.app.is_running():
            driver.update(1 / 60)
            for _ in range(7):
                handles.world.step(render=False)
            handles.world.step(render=True)
            frame += 1
            if smoke_frames and frame >= smoke_frames:
                break
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if binding is not None:
            binding.close()
        handles.app.close()
