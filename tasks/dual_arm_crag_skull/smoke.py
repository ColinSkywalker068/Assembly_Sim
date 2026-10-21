"""Headless real-asset physics/RGB-D smoke check; not an assembly success test.

Intrinsics come from the render product's projection matrix and are cross-checked by fitting
(fx, fy) to the same frame rendered as optical-Z depth and as ray distance. The USD aperture
attributes are recorded but not trusted: the cameras keep the default 20.955 x 15.2908 mm film
back while rendering 4:3 images, so attribute-derived fy is 2.8% too large. K uses the OpenCV
pixel-centre convention (cx = W/2 - 0.5). Extrinsics come from the renderer's view transform (what
was actually rendered); the USD pose of the camera prim is recorded for comparison, since with
Fabric it can lag behind a moving wrist.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import traceback
import numpy as np

from .camera_calibration import (fit_focal_from_depths, intrinsics_from_apertures, intrinsics_from_projection,
                                 rendered_world_T_usd_camera)

from .grippers import add_gripper_arguments, selected_gripper, require_scene_gripper

ANNOTATORS = ('rgb', 'distance_to_image_plane', 'distance_to_camera', 'camera_params')
FOCAL_AGREEMENT = 5e-3  # relative; projection-derived focal must match the depth fit this closely


def _camera_report(name, data, camera, world_T_usd, resolution, simulation_time):
    color = np.asarray(data['rgb'].get_data())
    distance = np.asarray(data['distance_to_image_plane'].get_data())
    ray = np.asarray(data['distance_to_camera'].get_data())
    params = data['camera_params'].get_data()
    if color.ndim != 3 or distance.shape != color.shape[:2] or ray.shape != distance.shape:
        raise RuntimeError(f'invalid RGB-D shape: {name}')
    valid = np.isfinite(distance) & (distance > 0)
    if not valid.any() or color[:, :, :3].max() <= 4:
        raise RuntimeError(f'empty RGB-D: {name}')
    w, h = resolution
    projection = np.asarray(params['cameraProjection'], dtype=float).reshape(4, 4)
    rendered = intrinsics_from_projection(projection, w, h)
    fx_fit, fy_fit, residual = fit_focal_from_depths(distance, ray, rendered[0, 2], rendered[1, 2])
    disagreement = max(abs(rendered[0, 0] / fx_fit - 1), abs(rendered[1, 1] / fy_fit - 1))
    if disagreement > FOCAL_AGREEMENT:
        raise RuntimeError(f'{name}: projection focal disagrees with depth fit by {disagreement:.4f}')
    claimed = intrinsics_from_apertures(camera.GetFocalLengthAttr().Get(), camera.GetHorizontalApertureAttr().Get(),
                                        camera.GetVerticalApertureAttr().Get(), w, h)
    world_T_rendered = rendered_world_T_usd_camera(params)
    return color, distance, {
        'timestamp_s': simulation_time, 'depth_units': 'm', 'depth_type': 'optical_z',
        'valid_fraction': float(valid.mean()),
        'world_T_camera_optical': (world_T_rendered @ np.diag([1, -1, -1, 1])).tolist(),
        'extrinsics_source': 'renderer_view_transform',
        'usd_prim_pose_max_abs_difference': float(np.abs(world_T_rendered - world_T_usd).max()),
        'intrinsics': rendered.tolist(), 'intrinsics_source': 'render_product_projection',
        'pixel_convention': 'opencv_integer_pixel_centres',
        'focal_fit_from_depths': {'fx': fx_fit, 'fy': fy_fit, 'rms_residual': residual,
                                  'max_relative_disagreement': disagreement},
        'intrinsics_from_usd_apertures': claimed.tolist(),
        'usd_aperture_fy_relative_error': float(claimed[1, 1] / fy_fit - 1),
        'renderer_aperture_mm': [float(v) for v in np.ravel(params.get('cameraAperture', []))]}


def run(assembly, output, frames=120, *, gripper="yam", stability=False, solver_position_iterations=None,
        solver_velocity_iterations=None):
    require_scene_gripper(gripper)
    for value in (solver_position_iterations, solver_velocity_iterations):
        if value is not None and not 1 <= value <= 255:
            raise ValueError('solver iterations must be in [1,255]')
    from tasks.dual_arm_breakingbad.scene.builder import build_assembly_scene
    from tasks.dual_arm_breakingbad.scene.validation import validate_assembly_stage
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for stale in ('smoke.json', 'smoke_failed.json'):
        (output/stale).unlink(missing_ok=True)
    handles = build_assembly_scene(Path(assembly), headless=True)
    products, annotators, succeeded = [], {}, False
    try:
        import omni.replicator.core as rep
        import omni.timeline
        from pxr import UsdGeom,UsdPhysics,PhysxSchema
        from PIL import Image
        if not validate_assembly_stage(handles).ok:
            raise RuntimeError('invalid scene')
        trace=[]
        if stability:
            from isaacsim.core.prims import RigidPrim
            for fragment in handles.fragments:
                prim=handles.stage.GetPrimAtPath(fragment.root_path)
                UsdPhysics.RigidBodyAPI(prim).CreateStartsAsleepAttr(False)
                body=PhysxSchema.PhysxRigidBodyAPI(prim)
                if solver_position_iterations is not None:body.CreateSolverPositionIterationCountAttr(solver_position_iterations)
                if solver_velocity_iterations is not None:body.CreateSolverVelocityIterationCountAttr(solver_velocity_iterations)
            handles.world.reset()
            for robot in handles.workcell.robots.values():robot.initialize_dofs()
            handles.controller.reset()
            bodies=RigidPrim(prim_paths_expr=[f.root_path for f in handles.fragments],name='settling_trace')
            bodies.initialize()
        initial_simulation_time=handles.world.current_time
        for step in range(frames):
            handles.world.step(render=True)
            if stability:
                p,q=bodies.get_world_poses();v=bodies.get_velocities()
                trace.append((handles.world.current_time,np.asarray(p).copy(),np.asarray(q).copy(),np.asarray(v).copy()))
        simulation_time = handles.world.current_time
        # Freeze physics while all three render products are sampled together.
        omni.timeline.get_timeline_interface().pause()
        resolution = handles.workcell.config.camera_resolution
        for name, path in handles.cameras.paths.items():
            product = rep.create.render_product(path, resolution); products.append(product)
            annotators[name] = {}
            for kind in ANNOTATORS:
                annotator = rep.AnnotatorRegistry.get_annotator(kind)
                annotator.attach(product)
                annotators[name][kind] = annotator
        for _ in range(8):
            rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=True)
        reports = {}
        for name, data in annotators.items():
            camera = UsdGeom.Camera(handles.stage.GetPrimAtPath(handles.cameras.path(name)))
            world_T_usd = np.asarray(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(0)).T
            color, distance, reports[name] = _camera_report(name, data, camera, world_T_usd, resolution,
                                                            simulation_time)
            Image.fromarray(color[:, :, :3]).save(output/f'{name}.png')
            np.save(output/f'{name}_depth_m.npy', distance)
        elapsed_simulation_time = simulation_time - initial_simulation_time
        report = {'physics_frames': frames, 'world_step_calls': frames, 'fragment_count': len(handles.fragments),
                  'elapsed_simulation_s': elapsed_simulation_time,
                  'solver_override': {'position_iterations': solver_position_iterations,
                                      'velocity_iterations': solver_velocity_iterations},
                  'physics_steps_from_elapsed_time': int(round(elapsed_simulation_time
                                                               / handles.workcell.config.physics_dt)),
                  'gripper': 'robotiq_proxy', 'assembly_success_evaluated': False,
                  'cameras': reports, 'physics_dt': handles.workcell.config.physics_dt}
        if stability:
            from .stability import summarize_stability
            times = np.array([x[0] for x in trace]); positions = np.stack([x[1] for x in trace])
            quaternions = np.stack([x[2] for x in trace]); velocities = np.stack([x[3] for x in trace])
            names = [f.name for f in handles.fragments]
            np.savez_compressed(output/'rigid_body_trace.npz', times=times, positions=positions,
                                quaternions=quaternions, velocities=velocities, names=names)
            report['settling'] = summarize_stability(times, positions, quaternions, velocities, names)
        handles.stage.GetRootLayer().Export(str(output/'scene.usda'))
        (output/'smoke.json').write_text(json.dumps(report, indent=2)+'\n')  # last: its presence means success
        succeeded = True
        print('SMOKE PASS', flush=True)
        return report
    except BaseException:
        (output/'smoke_failed.json').write_text(json.dumps({'error': traceback.format_exc()}, indent=2)+'\n')
        traceback.print_exc()
        print('SMOKE FAIL', flush=True)
        raise
    finally:
        for data in annotators.values():
            for annotator in data.values():
                annotator.detach()
        for product in products:
            product.destroy()
        if not succeeded:
            # Kit's fast shutdown ends the process with status 0 inside close(); keep the failure visible.
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)
        handles.app.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--stability', action='store_true')
    parser.add_argument('--solver-position-iterations', type=int)
    parser.add_argument('--solver-velocity-iterations', type=int)
    from .__main__ import DEFAULT_CONFIG
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    add_gripper_arguments(parser)
    args = parser.parse_args()
    try:
        gripper = selected_gripper(json.loads(args.config.read_text()), args.gripper, args.allow_robotiq_proxy)
        require_scene_gripper(gripper)
    except ValueError as exc:
        parser.error(str(exc))
    run(args.assembly, args.output, frames=args.frames, gripper=gripper, stability=args.stability,
        solver_position_iterations=args.solver_position_iterations,
        solver_velocity_iterations=args.solver_velocity_iterations)
