# Tasks

## Stack sawtooth shulker scene

`tasks.stack_sawtooth_shulker` composes the `single_arm_right` workcell with
two complementary blue and green voxel fragments. The deterministic initial
scene stages the upright pieces along Y between the red target and the right
FANUC. Its CLI only builds a validated external USD or launches a view-only
window:

```text
build  --output-usd scene.usda
launch
```

It intentionally contains no motion, recording, randomization, inference, or
success evaluation. See
[`tasks/stack_sawtooth_shulker/README.md`](../tasks/stack_sawtooth_shulker/README.md)
for the exact voxel geometry, poses, and commands.

## Demo storyboard renderer

`tasks.demo_render_source` reproduces the original scripted CRAG/LEGO story.
It owns studded object preparation, offline choreography, ghosts, captions, and
storyboard rendering. Its CRAG inputs and all outputs are external. See
[`tasks/demo_render_source/README.md`](../tasks/demo_render_source/README.md).

## Dual-arm Breaking Bad scene

`tasks.dual_arm_breakingbad` discovers all `piece_*.obj` files in one sample,
treats their existing coordinates as the ground-truth assembly, normalizes and
scales the whole object, voxelizes every fragment on one common grid, exports
plain voxel visuals, and generates colliders from the same occupied cells.

Its `layout.json` stores source identity, normalization, grid information,
relative visual assets, exact occupied cells, GT poses, and deterministic
bilateral staging poses. The Isaac scene composes the `dual_arm` core preset and
discovers any fragment count from that metadata.

The CLI exposes:

```text
prepare  --input DIR --output DIR [voxel parameters]
validate --assembly layout.json
inspect  --assembly layout.json --output DIR
build    --assembly layout.json --output-usd scene.usda
launch   --assembly layout.json [--stream]
```

See [`tasks/dual_arm_breakingbad/README.md`](../tasks/dual_arm_breakingbad/README.md)
for complete examples and interactive controls.
