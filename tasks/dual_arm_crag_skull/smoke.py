"""Headless real-asset physics/RGB-D smoke check; not an assembly success test."""
from pathlib import Path
import argparse
import json
import numpy as np


def run(assembly, output, frames=120, stability=False, solver_position_iterations=None, solver_velocity_iterations=None):
    for value in (solver_position_iterations,solver_velocity_iterations):
        if value is not None and not 1<=value<=255:raise ValueError('solver iterations must be in [1,255]')
    from tasks.dual_arm_breakingbad.scene.builder import build_assembly_scene
    from tasks.dual_arm_breakingbad.scene.validation import validate_assembly_stage
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    handles = build_assembly_scene(Path(assembly), headless=True)
    products, annotators = [], {}
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
            rgb = rep.AnnotatorRegistry.get_annotator('rgb')
            depth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
            rgb.attach(product); depth.attach(product)
            annotators[name] = (rgb, depth)
        for _ in range(8):
            rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=True)
        reports = {}
        for name, (rgb, depth) in annotators.items():
            color, distance = np.asarray(rgb.get_data()), np.asarray(depth.get_data())
            if color.ndim != 3 or distance.shape != color.shape[:2]:
                raise RuntimeError(f'invalid RGB-D shape: {name}')
            valid = np.isfinite(distance) & (distance > 0)
            if not valid.any() or color[:,:,:3].max() <= 4:
                raise RuntimeError(f'empty RGB-D: {name}')
            Image.fromarray(color[:,:,:3]).save(output/f'{name}.png')
            np.save(output/f'{name}_depth_m.npy', distance)
            camera = UsdGeom.Camera(handles.stage.GetPrimAtPath(handles.cameras.path(name)))
            world_T_usd = np.asarray(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(0)).T
            world_T_optical = world_T_usd @ np.diag([1,-1,-1,1])
            f = camera.GetFocalLengthAttr().Get()
            w,h = resolution
            fx = w*f/camera.GetHorizontalApertureAttr().Get()
            fy = h*f/camera.GetVerticalApertureAttr().Get()
            reports[name] = {'timestamp_s':simulation_time, 'depth_units':'m', 'depth_type':'optical_z',
                             'valid_fraction':float(valid.mean()), 'world_T_camera_optical':world_T_optical.tolist(),
                             'intrinsics':[[fx,0,w/2],[0,fy,h/2],[0,0,1]]}
        elapsed_simulation_time=simulation_time-initial_simulation_time
        report = {'world_step_calls':frames, 'fragment_count':len(handles.fragments),
                  'elapsed_simulation_s':elapsed_simulation_time,
                  'solver_override':{'position_iterations':solver_position_iterations,'velocity_iterations':solver_velocity_iterations},
                  'physics_steps_from_elapsed_time':int(round(elapsed_simulation_time/handles.workcell.config.physics_dt)),
                  'gripper':'robotiq_proxy', 'assembly_success_evaluated':False,
                  'cameras':reports, 'physics_dt':handles.workcell.config.physics_dt}
        if stability:
            from .stability import summarize_stability
            times=np.array([x[0] for x in trace]);positions=np.stack([x[1] for x in trace])
            quaternions=np.stack([x[2] for x in trace]);velocities=np.stack([x[3] for x in trace])
            names=[f.name for f in handles.fragments]
            np.savez_compressed(output/'rigid_body_trace.npz',times=times,positions=positions,quaternions=quaternions,velocities=velocities,names=names)
            report['settling']=summarize_stability(times,positions,quaternions,velocities,names)
        (output/'smoke.json').write_text(json.dumps(report,indent=2)+'\n')
        handles.stage.GetRootLayer().Export(str(output/'scene.usda'))
        return report
    finally:
        for rgb, depth in annotators.values():
            rgb.detach(); depth.detach()
        for product in products:
            product.destroy()
        handles.app.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-robotiq-proxy', action='store_true', required=True)
    parser.add_argument('--frames',type=int,default=120)
    parser.add_argument('--stability',action='store_true')
    parser.add_argument('--solver-position-iterations',type=int)
    parser.add_argument('--solver-velocity-iterations',type=int)
    args = parser.parse_args()
    run(args.assembly, args.output,args.frames,args.stability,args.solver_position_iterations,args.solver_velocity_iterations)
