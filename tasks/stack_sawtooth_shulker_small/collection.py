"""One physical attempt per manifest condition, with synchronized observations."""

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .conditions import condition_to_task_config
from .dataset import condition_to_dict, read_manifest, write_manifest
from .runtime import require_external_output


def run_attempt(manifest_path, demo_id, output):
    """Run in a fresh simulator process; failures retain all recorded samples."""
    from .builder import build_stack_sawtooth_scene
    from .scripted import ScriptedMotion, configure_gripper_coupling, motion_config
    from PIL import Image

    manifest = read_manifest(manifest_path)
    condition = next(c for c in manifest.conditions if c.demonstration_id == demo_id)
    output = require_external_output(Path(output))
    output.mkdir(parents=True, exist_ok=False)
    (output/'condition.json').write_text(json.dumps(condition_to_dict(condition), indent=2))
    config = condition_to_task_config(condition, base=motion_config())
    # An attempt is tied to the manifest and includes no preview guides.
    handles = None
    products, annotators, samples = [], [], []
    frames, tick_count = [], 0
    try:
        handles = build_stack_sawtooth_scene(config, stage_setup=configure_gripper_coupling)
        import omni.replicator.core as rep
        rep.orchestrator.set_capture_on_play(False)
        camera_names = ('agent', 'right_wrist')
        for camera in camera_names:
            (output/camera).mkdir()
            product = rep.create.render_product(handles.cameras.path(camera), (640, 480))
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach(product)
            products.append(product)
            annotators.append(annotator)
        handles.world.pause()
        for _ in range(12):
            handles.world.render()

        class RecordedMotion(ScriptedMotion):
            def tick(self):
                nonlocal tick_count
                super().tick()
                tick_count += 1
                positions = np.asarray(self.driver.robot.robot.get_joint_positions()).copy()
                velocities = np.asarray(self.driver.robot.robot.get_joint_velocities()).copy()
                objects = np.concatenate([np.concatenate(self.object_pose(name))
                                          for name in ('FragmentA', 'FragmentB')])
                samples.append((tick_count / 60., positions, velocities,
                                self.driver.arm_target.copy(), self.angle,
                                objects))
                if tick_count % 6 == 0:
                    handles.world.render()  # Rendering does not advance physics.
                    frame_index = len(frames)
                    for name, annotator in zip(camera_names, annotators):
                        rgb = np.asarray(annotator.get_data())[..., :3]
                        if rgb.shape != (480, 640, 3) or rgb.max() <= 4:
                            raise RuntimeError(f'Invalid camera observation: {name}')
                        Image.fromarray(rgb.astype(np.uint8)).save(output/name/f'{frame_index:06d}.png')
                    frames.append({'index': frame_index, 'timestamp': tick_count / 60.,
                                   'state_index': len(samples)-1})

            def event(self, label, capture=True):
                super().event(label, capture=False)

        motion = RecordedMotion(handles, output, assembly_order=condition.assembly_order)
        motion.execute()
        (output/'status.json').write_text(json.dumps({'status': 'success', 'demo_id': demo_id}))
    except Exception as exc:
        (output/'status.json').write_text(json.dumps({'status': 'failed', 'demo_id': demo_id,
                                                     'error': f'{type(exc).__name__}: {exc}'}))
        raise
    finally:
        if samples:
            np.savez_compressed(output/'trajectory.npz',
                timestamps=np.array([s[0] for s in samples]),
                joint_positions=np.stack([s[1] for s in samples]),
                joint_velocities=np.stack([s[2] for s in samples]),
                arm_position_commands=np.stack([s[3] for s in samples]),
                gripper_angle_commands=np.array([s[4] for s in samples]),
                gt_object_poses=np.stack([s[5] for s in samples]))
        (output/'frames.json').write_text(json.dumps(frames))
        if handles is not None:
            (output/'recording.json').write_text(json.dumps({
                'state_hz': 60, 'camera_hz': 10, 'resolution': [640, 480],
                'joint_names': list(motion.driver.robot.robot.dof_names) if samples else [],
                'arm_command_joint_names': list(config.workcell.robot('right').arm_dof_names),
                'gt_object_order': ['FragmentA', 'FragmentB'],
                'gt_pose_format': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'],
            }, indent=2))
            for annotator, product in zip(annotators, products):
                annotator.detach(product)
                rep.destroy.destroy_render_product(product)
            handles.app.close()


def collect_manifest(manifest_path, output_root):
    """Attempt exactly 50 demos sequentially; never retry failed conditions."""
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) != 1:
        raise RuntimeError('Select exactly one GPU with CUDA_VISIBLE_DEVICES before collection')
    subprocess.run(['nvidia-smi'], check=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader'], text=True)
    own_gpus = set()
    for line in processes.splitlines():
        uuid, pid = (part.strip() for part in line.split(','))
        try:
            if Path('/proc', pid).stat().st_uid == os.getuid():
                own_gpus.add(uuid)
        except FileNotFoundError:
            continue
    if len(own_gpus) >= 2:
        raise RuntimeError('This account already uses two GPUs; do not launch another GPU run')
    manifest = read_manifest(manifest_path)
    root = require_external_output(Path(output_root))
    run = root / datetime.now().strftime('collection_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    manifest_path = write_manifest(manifest, run/'manifest.json')
    results = []
    for condition in manifest.conditions:
        output = run/condition.demonstration_id
        with (run/f'{condition.demonstration_id}.log').open('w') as log:
            process = subprocess.run([sys.executable, '-m', 'tasks.stack_sawtooth_shulker_small',
                'collect-attempt', '--manifest', str(manifest_path),
                '--demo-id', condition.demonstration_id, '--output', str(output)],
                stdout=log, stderr=subprocess.STDOUT)
        status = json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {
            'status': 'failed', 'demo_id': condition.demonstration_id, 'error': 'Process ended without status'}
        status['returncode'] = process.returncode
        if process.returncode != 0:
            status['status'] = 'failed'
            status.setdefault('error', 'Attempt process exited unsuccessfully; inspect its log')
        results.append(status)
        (run/'summary.json').write_text(json.dumps({'attempted': len(results),
            'successful': sum(r['status'] == 'success' for r in results), 'results': results}, indent=2))
    return run
