# Stack sawtooth shulker scene

This task is a deterministic, single-right-arm scene for a two-piece keyed
assembly. It tests whether a later robot policy can perform sequential
pick-and-place and precise relative alignment when the intended assembly is
already known. The task instruction is: "Assemble the blue and green pieces
inside the red square."

The completed object is a `3 x 3 x 3` voxel cube with `u = 0.08 m`, so its
side length is `0.24 m`. Fragment A is blue and contains the full bottom layer
plus the four corners and center of the middle layer (14 voxels). Fragment B
is green and contains the four middle-layer edge centers plus the full top
layer (13 voxels). Their visual voxels retain the exact grid geometry;
colliders are inset by `0.002 m` total per axis for contact clearance.

Both fragments start upright and unfixed on the table, in a Y-directed line
between the target and right robot:

```text
Fragment A: (0.48, -0.185, 0.75) m, identity orientation
Fragment B: (0.48,  0.185, 0.75) m, identity orientation
```

The shared `single_arm_right` workcell supplies the FANUC CRX-10iA/L with
Robotiq gripper, agent and right-wrist cameras, table, padded workspace, and
red `0.55 x 0.55 m` target boundary.
The target size is a shared workcell dimension and is independent of `u`.

From the repository root, build and validate a USD stage at an external path:

```bash
"$ISAAC_PYTHON" -m tasks.stack_sawtooth_shulker build \
  --output-usd /local_data/yz11445/experiments/assembly_sim/stack_sawtooth_shulker/scene.usda
```

Or open the scene in a view-only simulator window:

```bash
"$ISAAC_PYTHON" -m tasks.stack_sawtooth_shulker launch
```

Generated stages must be outside the repository. This initial scene has no
robot-motion controls, recording, data collection, pose randomization,
assembly inference, or success evaluation. Those behaviors are intentionally
outside the current implementation scope.
