# Stack sawtooth shulker small

Initial-scene variant of `stack_sawtooth_shulker`, scaled by `27/80 = 0.3375`.
The instruction remains: "Assemble the blue and green pieces inside the red square."

| Dimension | Small scene |
| --- | --- |
| Voxel side `u` | 27 mm |
| Each fragment bounding box | 81 × 81 × 54 mm |
| Assembled cube | 81 × 81 × 81 mm |
| Red target square | 185.625 × 185.625 mm |
| Imaginary placement square | 101.25 × 101.25 mm |
| Red tape width | 4.05 mm |

Blue retains 14 voxels and green retains 13. The fragments start upright at
`(0.48, -0.185, 0.75)` and `(0.48, 0.185, 0.75)` m, respectively.
Robot, gripper, cameras, table, material density, and 2 mm total collider
clearance are the same as the original task. Mass follows the reduced volume.
The target pad dimensions and tape/grid dimensions use the shrinking scale.
The imaginary square is invisible; its side is stored as `placementSquareSize`
in `/World/Task/TargetMetadata`, centered on the red target.

Check GPU usage and select one GPU before running Isaac Sim. From the repository root:

```bash
CUDA_VISIBLE_DEVICES=<selected_gpu> "$ISAAC_PYTHON" -m tasks.stack_sawtooth_shulker_small build \
  --output-usd /local_data/yz11445/experiments/assembly_sim/stack_sawtooth_shulker_small/scene.usda

CUDA_VISIBLE_DEVICES=<selected_gpu> "$ISAAC_PYTHON" -m tasks.stack_sawtooth_shulker_small launch
```

The original keyboard controls are also available through the `teleop` command.
Dataset manifest, annotated preview, and collection commands are described below.

## Scripted assembly

```bash
CUDA_VISIBLE_DEVICES=<selected_gpu> "$ISAAC_PYTHON" -m tasks.stack_sawtooth_shulker_small scripted
```

The script physically pinches blue's central raised voxel, places blue at the
red target center, then opens fully and pinches green's entire upper slab from
above. All grasp and insertion motions keep the approach vertical; only XY
translation and small yaw alignment are used during transfer. Ground-truth
object poses refine alignment above the target. Objects remain free rigid
bodies throughout; the script uses joint drives and contact friction.

This mode uses native PhysX constraints for the Robotiq finger linkage, 240 Hz
physics, 60 Hz control, density 20 kg/m³ (blue 5.51 g, green 5.12 g), and contact
friction 2.0 static / 1.5 dynamic. The original 2 mm collider inset is retained.
The expected settled vertical separation is therefore approximately 25 mm,
compared with the nominal 27 mm grid spacing.

Each run writes snapshots, measured poses, `result.json`, and an assembled
close-up beneath a new timestamped experiment directory. Successful runs also
export `final_scene.usda` with the measured fragment poses. Success is checked
after release and withdrawal, including blue's center and table height, the
relative fragment position, and both orientations. Add `--gui` for a window
or `--output-root` to choose another external experiment directory.

## Mating-ready dataset

The manifest contains exactly 50 attempts: two canonical blue-base scenes and
48 randomized scenes. The latter include 24 blue-base and 24 green-base cases,
with 12 blue-high-Y and 12 green-high-Y layouts in each order. Blue-base scenes
use Blue T-up / Green C-up (both identity); green-base scenes use Blue C-up /
Green T-up (both flipped 180 degrees about X). The base always rests on its
full 3-by-3 face; its mate starts with its full face upward. Root Z is computed
from oriented bounds. No fragment is flipped during execution.

Initial X remains 0.48 m. Y uses the original intervals [0.185, 0.335] and
[-0.335, -0.185] m. Random target coordinates are differences of two independent
Uniform(0, 0.050625) samples, within the centered 101.25 mm imaginary square.
The exact center is reserved for the two standard cases. Dataset/demo seeds,
conditions, poses, assembly order, and targets are saved in the manifest.

Generate the manifest, then render all initial scenes:

```bash
ISAAC_PYTHON=/local_data/yz11445/tools/isaacsim-4.5/python.sh
"$ISAAC_PYTHON" -m tasks.stack_sawtooth_shulker_small generate-manifest \
  --dataset-seed 20261005 \
  --output-json /local_data/yz11445/experiments/assembly_sim/stack_sawtooth_shulker_small/manifest_20261005.json

# Inspect GPU usage first and substitute one appropriate physical GPU index.
nvidia-smi
CUDA_VISIBLE_DEVICES=<selected_gpu> "$ISAAC_PYTHON" -m tasks.stack_sawtooth_shulker_small render-manifest \
  --manifest /local_data/yz11445/experiments/assembly_sim/stack_sawtooth_shulker_small/manifest_20261005.json \
  --output-root /local_data/yz11445/experiments/assembly_sim/stack_sawtooth_shulker_small
```

The preview creates 100 captioned images and two 3200-by-4800 contact sheets
(agent and wrist), with all 50 scenes. Yellow outlines mark the imaginary
square, a white point marks its center, and a magenta point/line marks each
sampled target. Standard targets are cyan. Guides appear only in previews.

To attempt all 50 physical demonstrations sequentially:

```bash
CUDA_VISIBLE_DEVICES=<selected_gpu> "$ISAAC_PYTHON" -m tasks.stack_sawtooth_shulker_small collect-manifest \
  --manifest /local_data/yz11445/experiments/assembly_sim/stack_sawtooth_shulker_small/manifest_20261005.json \
  --output-root /local_data/yz11445/datasets/generated/stack_sawtooth_shulker_small
```

Each attempt starts a fresh simulator process. Failures are retained and never
retried or replaced. Joint positions/velocities and position commands are saved
at 60 Hz in `trajectory.npz`; unannotated agent/wrist RGB images are saved at
10 Hz, 640 by 480. `frames.json` maps each image pair to simulation time and
its exact state sample. Ground-truth object poses are separately named in the
trajectory. Conditions, joint ordering, recording settings, measured success,
and failure status are saved alongside observations. The batch summary records
all outcomes. Collection uses one explicitly selected GPU.

The constrained manifests and preview orchestration have CPU tests. The new
green-base physical grasp and recording require live simulator verification;
the original blue-base physical sequence was verified before this extension.
