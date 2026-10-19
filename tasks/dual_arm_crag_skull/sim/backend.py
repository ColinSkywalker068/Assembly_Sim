"""Isaac Sim AssemblyBackend for the skull task: the dual CRX + YAM linear_4310 workcell, the task's fragments with
the anchor held by a fixture, and the three cameras. GPU (Isaac Sim 5.1); the rules at the top are tested on CPU.

Timing follows configs/jing.json s0: PhysX runs at physics_hz, joint and width targets come at servo_hz
(control_hz), and each control period is `substeps` physics steps with the arm targets interpolated linearly from
the previous commanded targets, and the period's velocity fed forward as the drives' velocity target. The asset's
arm drives have damping/stiffness = 0.1 s: with zero velocity targets the damping holds a moving arm back like a
0.1 s lag (measured 50 mm behind a 1.5 rad/s move), and the arm keeps moving after the plan stops, which S0, checking
the next chunk from the measured state, rightly refuses. S0 checks a chunk from the measured joints; with the
feed-forward the executed path stays on the commanded one (sim/check_backend.py measures the tracking error). Cameras render on demand, at most at camera_hz,
with physics paused and resumed through World.play() (a bare timeline.play() is deferred to the next app update,
and World.step(render=False) skips physics while the timeline is not playing); a frame keeps its sample time, which
may be older than the latest robot state. step() refuses to run while the simulation is not playing.

reset(task):
- builds the fragments of the task's layout on the first call (a later task must have the same layout and
  fragments) and puts every fragment at its initial pose: the anchor as a kinematic body (the fixture: contacts do
  not move it), the others awake;
- puts both arms at home with the grippers fully open;
- holds the robot while the free fragments settle, for at least start_settle_s and until at rest (RestTolerance),
  at most max_settle_s, and refuses a start that did not come to rest. `start` records the settled poses, which are
  the trial's start (`start.to_record()` for a run record); the clock restarts at 0.

State: joints and joint velocities are PhysX's; widths are the pad gap from the measured finger joints; TCPs are
the S0 model's FK of the measured joints (s0.kinematics.yam_model; sim/check_yam.py measured PhysX within 0.001 mm
of it). Observed fragment poses are simulator truth labelled 'oracle_sim'; privileged poses are the same truth, for
scoring only.

close() releases the render products and stops the timeline but does not close Kit: Kit's close() ends the process
with status 0 (Isaac Sim 5.1) while a caller may still have records to write. shutdown(exit_code) ends the process;
call it last. One backend per process: build it once and reset it for every seed.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Mapping

import numpy as np

from core.robots.linear_4310 import load_linear_4310

from ..baselines.backend import CameraFrame, ObservedPose, RobotSnapshot, checked_targets
from ..contracts.actions import ARMS
from ..s0.kinematics import YAM_PROVISIONAL_ADAPTER_THICKNESS_M, yam_flange_T_mount, yam_model, yam_width_range_m
from .yam_usd import FingerDrive, author_linear_4310

ANNOTATORS = ('rgb', 'distance_to_image_plane', 'camera_params')
RENDER_PASSES = 4


def control_substeps(physics_hz, control_hz):
    """Physics steps per control period; the rates must be positive and divide exactly."""
    physics_hz, control_hz = float(physics_hz), float(control_hz)
    if not (math.isfinite(physics_hz) and math.isfinite(control_hz) and physics_hz > 0 and control_hz > 0):
        raise ValueError('physics and control rates must be positive and finite')
    steps = physics_hz / control_hz
    if steps < 1 or abs(steps - round(steps)) > 1e-9:
        raise ValueError(f'physics rate {physics_hz} Hz is not a whole multiple of the control rate {control_hz} Hz')
    return int(round(steps))


def interpolated_targets(previous, target, substeps):
    """The per-physics-step targets of one control period: linear from `previous`, ending exactly at `target`."""
    previous, target = np.asarray(previous, dtype=float), np.asarray(target, dtype=float)
    return [previous + (target - previous) * (k / substeps) for k in range(1, substeps)] + [target]


def finger_joint_target(gripper, width):
    """Joint target of each finger for a pad gap of `width` (0 = pads touching)."""
    return gripper.joint_for_width(width)


def measured_width(gripper, q_left, q_right):
    """Pad gap from measured finger joints; values a hair outside the joint range (solver overshoot) are clipped."""
    values = np.array([q_left, q_right], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('measured finger joints must be finite')
    return gripper.width_between(*np.clip(values, 0.0, gripper.travel_m))


@dataclass(frozen=True)
class RestTolerance:
    """At rest: over the last `window_s`, no body drifted more than `drift_m` or turned more than `drift_deg`.
    Instantaneous velocities do not work here: resting PhysX contacts jitter at about 1 mm/s with micrometre
    amplitude (measured on the skull fragments: 0.8-2 mm/s after 10 s, with no displacement)."""

    drift_m: float = 1e-4
    drift_deg: float = 0.05
    window_s: float = 0.5

    def __post_init__(self):
        values = (self.drift_m, self.drift_deg, self.window_s)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError('rest tolerance values must be positive and finite')


def at_rest(before, after, tolerance):
    """True when no body drifted beyond the tolerance between two {key: world_T_body} readings; NaN is not at rest."""
    if set(before) != set(after):
        raise ValueError('rest is judged over the same bodies')
    for key in before:
        first, second = np.asarray(before[key], dtype=float), np.asarray(after[key], dtype=float)
        if not (np.isfinite(first).all() and np.isfinite(second).all()):
            return False
        if not (np.linalg.norm(second[:3, 3] - first[:3, 3]) <= tolerance.drift_m
                and _turn_deg(first, second) <= tolerance.drift_deg):
            return False
    return True


def camera_due(last_time, now, camera_hz):
    """Whether a new frame is due at `now`, after one sampled at `last_time` (None: never)."""
    return last_time is None or now - last_time >= 1.0 / camera_hz - 1e-9


@dataclass(frozen=True)
class SettledStart:
    """The trial's start after settling: world_T_fragment per fragment, how long it settled and whether it came to
    rest, how far each fragment moved from the task's initial pose, and its speeds when settling ended."""

    poses: Mapping[str, np.ndarray]
    settle_s: float
    at_rest: bool
    moved_m: Mapping[str, float]
    turned_deg: Mapping[str, float]
    speed_m_s: Mapping[str, float]
    spin_rad_s: Mapping[str, float]

    def to_record(self):
        return {'settle_s': self.settle_s, 'at_rest': self.at_rest,
                'poses': {key: pose.tolist() for key, pose in self.poses.items()},
                'moved_m': dict(self.moved_m), 'turned_deg': dict(self.turned_deg),
                'speed_m_s': dict(self.speed_m_s), 'spin_rad_s': dict(self.spin_rad_s)}


def settled_start(ids, poses, initial, settle_s, resting, velocities):
    """SettledStart from the settled poses, the task's initial poses and the (N, 6) velocities, in `ids` order."""
    velocities = np.asarray(velocities, dtype=float)
    speeds, spins = np.linalg.norm(velocities[:, :3], axis=1), np.linalg.norm(velocities[:, 3:], axis=1)

    def frozen(pose):
        pose = np.array(pose, dtype=float)
        pose.flags.writeable = False
        return pose

    return SettledStart(MappingProxyType({key: frozen(poses[key]) for key in ids}), float(settle_s), bool(resting),
                        MappingProxyType({key: float(np.linalg.norm(poses[key][:3, 3] - initial[key][:3, 3]))
                                          for key in ids}),
                        MappingProxyType({key: _turn_deg(initial[key], poses[key]) for key in ids}),
                        MappingProxyType({key: float(speeds[i]) for i, key in enumerate(ids)}),
                        MappingProxyType({key: float(spins[i]) for i, key in enumerate(ids)}))


def checked_start(start):
    """`start`, if the fragments came to rest; a trial must not begin while something is still moving."""
    if not start.at_rest:
        moved = max(start.moved_m, key=start.moved_m.get)
        raise RuntimeError(f'the fragments did not come to rest within {start.settle_s:.2f} s (largest move: {moved} '
                           f'{start.moved_m[moved] * 1000:.1f} mm)')
    return start


def check_fixture(task, tolerance_m=1e-9):
    """The fixture holds the anchor at its goal, so the anchor must start there."""
    anchor = task.fragment(task.anchor_id)
    if not np.allclose(anchor.initial_world_T_fragment, anchor.goal_world_T_fragment, rtol=0, atol=tolerance_m):
        raise ValueError(f'the fixture holds the anchor {task.anchor_id} at its goal, but the task starts it elsewhere')


def _pose_matrix(position, orientation_wxyz):
    from core.robots.kinematics import T
    return T(np.asarray(position, dtype=float), np.asarray(orientation_wxyz, dtype=float))


def _wxyz(pose):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(np.asarray(pose, dtype=float)[:3, :3]).as_quat()
    return np.array([w, x, y, z])


def _turn_deg(first, second):
    relative = first[:3, :3].T @ second[:3, :3]
    skew = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]])
    return float(np.degrees(np.arctan2(np.linalg.norm(skew) / 2, (np.trace(relative) - 1) / 2)))


class IsaacAssemblyBackend:
    """AssemblyBackend on Isaac Sim (see the module docstring). Creates its own SimulationApp: one per process."""

    def __init__(self, config, *, headless=True, cameras=True, start_settle_s=1.0, max_settle_s=3.0,
                 rest=RestTolerance(), drive=FingerDrive(), adapter_thickness_m=YAM_PROVISIONAL_ADAPTER_THICKNESS_M):
        from core.workcell.config import load_workcell_preset
        s0 = config['s0']
        self.physics_hz, self.control_hz, self.camera_hz = (float(s0[key]) for key in
                                                             ('physics_hz', 'servo_hz', 'camera_hz'))
        self.substeps = control_substeps(self.physics_hz, self.control_hz)
        frames = self.control_hz / self.camera_hz
        if not (math.isfinite(frames) and frames >= 1 and abs(frames - round(frames)) <= 1e-9):
            raise ValueError(f'control rate {self.control_hz} Hz is not a whole multiple of the camera rate '
                             f'{self.camera_hz} Hz')
        if not 0 < start_settle_s <= max_settle_s:
            raise ValueError('need 0 < start_settle_s <= max_settle_s')
        self.start_settle_s, self.max_settle_s, self.rest = float(start_settle_s), float(max_settle_s), rest
        self.workcell = load_workcell_preset(config['workcell']['preset'])
        self.gripper = load_linear_4310()
        self.flange_T_mount = yam_flange_T_mount(adapter_thickness_m, self.gripper)
        self.model = yam_model(workcell=self.workcell, thickness_m=adapter_thickness_m)
        self.width_range = {arm: yam_width_range_m(self.gripper) for arm in ARMS}
        self.use_cameras = bool(cameras)
        self.task, self.start, self._fragments, self._layout_key = None, None, None, None
        self._products, self._annotators, self._frames, self._frame_time = [], {}, None, None
        self._steps = 0
        self._open_app(headless)
        self._build_workcell(drive)

    # -- construction --------------------------------------------------------------------------------------------
    def _open_app(self, headless):
        from core.scene.builder import simulation_launch_config
        from isaacsim import SimulationApp
        self.app = SimulationApp(simulation_launch_config(self.workcell, headless))
        from isaacsim.core.api import World
        from isaacsim.core.utils.types import ArticulationAction
        self._action = ArticulationAction
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0 / self.physics_hz,
                           rendering_dt=1.0 / self.camera_hz)
        self.world.get_physics_context().set_gravity(float(self.workcell.physics['gravity'][2]))

    def _build_workcell(self, drive):
        import omni.usd
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.prims import XFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import UsdGeom, UsdShade

        from core.workcell.cameras import author_cameras
        from core.workcell.environment import author_environment

        stage = omni.usd.get_context().get_stage()
        stage.SetMetadata('metersPerUnit', 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, '/World')
        author_environment(stage, self.workcell)
        material = UsdShade.Material(stage.GetPrimAtPath('/World/Looks/ContactMaterial'))
        self.stage, self.robots, flanges = stage, {}, {}
        for arm in ARMS:
            settings = self.workcell.robot(arm)
            add_reference_to_stage(str(self.workcell.asset_path('arm_usd')), settings.prim_path)
            XFormPrim(settings.prim_path).set_world_poses(
                positions=np.asarray([settings.base_position], dtype=float),
                orientations=np.asarray([settings.base_orientation_wxyz], dtype=float))
            flange = f'{settings.prim_path}/flange'
            prims = author_linear_4310(stage, f'/World/Grippers/{arm.capitalize()}/YAM_Linear_4310',
                                       flange_path=flange, flange_T_mount=self.flange_T_mount, gripper=self.gripper,
                                       drive=drive, physics_material=material)
            robot = self.world.scene.add(Robot(prim_path=settings.prim_path, name=f'fanuc_yam_{arm}'))
            self.robots[arm] = SimpleNamespace(robot=robot, prims=prims, home=np.asarray(settings.home_joint_positions),
                                               dof_names=settings.arm_dof_names, arm_indices=None, finger_indices=None)
            flanges[arm] = SimpleNamespace(flange_path=flange)
        self.cameras = author_cameras(stage, self.workcell, flanges) if self.use_cameras else None

    def _build_fragments(self, task):
        from isaacsim.core.prims import RigidPrim
        from pxr import UsdGeom, UsdPhysics

        from tasks.dual_arm_breakingbad.scene.assets import author_fragment
        from tasks.dual_arm_breakingbad.scene.config import load_assembly_scene_config

        layout = load_assembly_scene_config(task.layout).layout
        if set(layout.fragment_names) != set(task.fragment_ids):
            raise ValueError('the task fragments differ from its layout')
        UsdGeom.Xform.Define(self.stage, '/World/Fragments')
        handles = {key: author_fragment(self.stage, self.workcell, layout, key) for key in task.fragment_ids}
        for key, handle in handles.items():
            body = UsdPhysics.RigidBodyAPI(self.stage.GetPrimAtPath(handle.root_path))
            body.GetStartsAsleepAttr().Set(False)
            body.CreateKinematicEnabledAttr(key == task.anchor_id)
        self._fragments = {key: handle.root_path for key, handle in handles.items()}
        self._fragment_ids = tuple(task.fragment_ids)
        self._fragment_view = self.world.scene.add(RigidPrim(
            prim_paths_expr=[self._fragments[key] for key in self._fragment_ids], name='skull_fragments'))
        self._layout_key = (task.layout_sha256, self._fragment_ids, task.anchor_id)

    def _place_fragments(self, task):
        from core.workcell.environment import set_transform
        from pxr import UsdGeom
        for key in self._fragment_ids:
            pose = task.fragment(key).initial_world_T_fragment
            set_transform(UsdGeom.Xformable(self.stage.GetPrimAtPath(self._fragments[key])), pose[:3, 3], _wxyz(pose))

    def _index_dofs(self):
        for arm, handle in self.robots.items():
            names = [str(name) for name in handle.robot.dof_names]
            missing = [name for name in (*handle.dof_names, *handle.prims.dof_names) if name not in names]
            if missing:
                raise RuntimeError(f'{arm}: articulation lacks DOFs {missing}')
            handle.arm_indices = np.array([names.index(name) for name in handle.dof_names])
            handle.finger_indices = np.array([names.index(name) for name in handle.prims.dof_names])

    # -- AssemblyBackend -----------------------------------------------------------------------------------------
    def reset(self, task):
        check_fixture(task)
        key = (task.layout_sha256, tuple(task.fragment_ids), task.anchor_id)
        if self._fragments is None:
            self._build_fragments(task)
        elif key != self._layout_key:
            raise ValueError('this backend was built for another layout, fragment set or anchor')
        else:
            self.world.stop()
        self._place_fragments(task)
        self.world.reset()
        self._index_dofs()
        # The view's default state is what a reset_async would restore; keep it at this task's start.
        initial = [task.fragment(key).initial_world_T_fragment for key in self._fragment_ids]
        self._fragment_view.set_default_state(positions=np.array([pose[:3, 3] for pose in initial]),
                                              orientations=np.array([_wxyz(pose) for pose in initial]))
        self.task = task
        self._steps, self._frames, self._frame_time = 0, None, None
        self._targets = {arm: handle.home.copy() for arm, handle in self.robots.items()}
        for arm, handle in self.robots.items():
            indices = np.concatenate([handle.arm_indices, handle.finger_indices])
            values = np.concatenate([handle.home, [self.gripper.travel_m] * 2])
            handle.robot.set_joint_positions(values, joint_indices=indices)
            handle.robot.set_joint_velocities(np.zeros(len(indices)), joint_indices=indices)
        self._apply_targets(self._targets, {arm: self.gripper.max_width_m for arm in ARMS})
        start = self._settle(task)
        if not start.at_rest:
            self.task = None
        self.start = checked_start(start)
        self._steps = 0
        return self.robot_state()

    def _settle(self, task):
        """Step with the robot held until at rest (judged once per window after start_settle_s) or max_settle_s."""
        window = max(self.substeps, round(self.rest.window_s * self.physics_hz))
        steps, needed, limit = 0, round(self.start_settle_s * self.physics_hz), round(self.max_settle_s * self.physics_hz)
        resting, earlier = False, None
        while steps < limit:
            self._physics_steps(self.substeps)
            steps += self.substeps
            if steps >= needed - window and earlier is None:
                earlier, since = self.privileged_fragment_poses(), steps
            if earlier is not None and steps - since >= window:
                now = self.privileged_fragment_poses()
                resting = at_rest(earlier, now, self.rest)
                if resting and steps >= needed:
                    break
                earlier, since = now, steps
        initial = {key: task.fragment(key).initial_world_T_fragment for key in self._fragment_ids}
        return settled_start(self._fragment_ids, self.privileged_fragment_poses(), initial, steps / self.physics_hz,
                             resting, self._fragment_view.get_velocities())

    def _apply_targets(self, arm_targets, widths, arm_velocities=None):
        """Arm joint position and velocity targets (velocity 0 unless given), and the finger targets for `widths` (an
        arm's None leaves its fingers' targets; their velocity targets are 0)."""
        for arm, handle in self.robots.items():
            q = None if widths[arm] is None else finger_joint_target(self.gripper, widths[arm])
            values = np.concatenate([arm_targets[arm], [q, q] if q is not None else []])
            speed = np.zeros(len(arm_targets[arm])) if arm_velocities is None else arm_velocities[arm]
            speeds = np.concatenate([speed, [0.0, 0.0] if q is not None else []])
            indices = handle.arm_indices if q is None else np.concatenate([handle.arm_indices, handle.finger_indices])
            handle.robot.get_articulation_controller().apply_action(
                self._action(joint_positions=values, joint_velocities=speeds, joint_indices=indices))

    def _physics_steps(self, count):
        for _ in range(count):
            self.world.step(render=False)

    def _require_reset(self):
        if self.task is None:
            raise RuntimeError('reset the backend first')

    def _require_playing(self):
        if not self.world.is_playing():
            raise RuntimeError('the simulation is not playing, so physics would not step')

    def _resume(self):
        """Play again after rendering; World.play() commits the change, a bare timeline.play() would not."""
        self.world.play()
        self._require_playing()

    def step(self, joints, widths):
        self._require_reset()
        self._require_playing()
        targets = checked_targets(self.model, self.width_range, joints, widths)
        paths = {arm: interpolated_targets(self._targets[arm], targets[arm], self.substeps) for arm in ARMS}
        velocities = {arm: (targets[arm] - self._targets[arm]) * self.control_hz for arm in ARMS}
        for k in range(self.substeps):
            self._apply_targets({arm: paths[arm][k] for arm in ARMS},
                                {arm: float(widths[arm]) if k == 0 else None for arm in ARMS}, velocities)
            self.world.step(render=False)
        self._targets = targets
        self._steps += self.substeps
        return self.robot_state()

    @property
    def time_s(self):
        return self._steps / self.physics_hz

    def robot_state(self):
        self._require_reset()
        joints, velocities, widths, tcp = {}, {}, {}, {}
        for arm, handle in self.robots.items():
            q = np.asarray(handle.robot.get_joint_positions(joint_indices=handle.arm_indices), dtype=float)
            qd = np.asarray(handle.robot.get_joint_velocities(joint_indices=handle.arm_indices), dtype=float)
            fingers = np.asarray(handle.robot.get_joint_positions(joint_indices=handle.finger_indices), dtype=float)
            joints[arm], velocities[arm] = q, qd
            widths[arm] = measured_width(self.gripper, *fingers)
            tcp[arm] = self.model.arms[arm].tcp(q)
        return RobotSnapshot(self.time_s, joints, velocities, widths, tcp)

    def privileged_fragment_poses(self):
        self._require_reset()
        positions, orientations = self._fragment_view.get_world_poses()
        return {key: _pose_matrix(positions[i], orientations[i]) for i, key in enumerate(self._fragment_ids)}

    def observed_fragment_poses(self):
        self._require_reset()
        return {key: ObservedPose(pose, 'oracle_sim', self.time_s) for key, pose in self.privileged_fragment_poses().items()}

    def camera_frames(self):
        self._require_reset()
        if self.cameras is None:
            raise RuntimeError('this backend was built without cameras')
        if camera_due(self._frame_time, self.time_s, self.camera_hz):
            self._frames, self._frame_time = MappingProxyType(self._render()), self.time_s
        return self._frames

    def _render(self):
        import omni.replicator.core as rep
        import omni.timeline

        from ..camera_calibration import intrinsics_from_projection, rendered_world_T_camera_optical
        width, height = self.workcell.camera_resolution
        if not self._products:
            for name, path in self.cameras.paths.items():
                product = rep.create.render_product(path, (width, height))
                self._products.append(product)
                self._annotators[name] = {}
                for kind in ANNOTATORS:
                    annotator = rep.AnnotatorRegistry.get_annotator(kind)
                    annotator.attach(product)
                    self._annotators[name][kind] = annotator
        omni.timeline.get_timeline_interface().pause()
        try:
            for _ in range(RENDER_PASSES):
                rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=True)
        finally:
            self._resume()
        frames = {}
        for name, data in self._annotators.items():
            color = np.asarray(data['rgb'].get_data())[:, :, :3]
            depth = np.asarray(data['distance_to_image_plane'].get_data(), dtype=np.float32)
            params = data['camera_params'].get_data()
            intrinsics = intrinsics_from_projection(np.asarray(params['cameraProjection'], dtype=float), width, height)
            frames[name] = CameraFrame(np.ascontiguousarray(color, dtype=np.uint8), depth, intrinsics,
                                       rendered_world_T_camera_optical(params), self.time_s)
        return frames

    def close(self):
        for data in self._annotators.values():
            for annotator in data.values():
                annotator.detach()
        for product in self._products:
            product.destroy()
        self._products, self._annotators, self._frames = [], {}, None
        if self.world is not None:
            self.world.stop()
        self.task = None

    def shutdown(self, exit_code=0):
        """End the process with `exit_code`. Kit's close() exits with status 0, so a failure exits without it."""
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code:
            os._exit(min(max(int(exit_code), 1), 255))
        self.app.close()
