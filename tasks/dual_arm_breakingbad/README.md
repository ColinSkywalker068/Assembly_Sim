# Dual-arm Breaking Bad assembly scene

This task converts one Breaking Bad fractured-object directory into a portable,
voxel-based assembly description and places its fragments into the shared
dual-FANUC workcell. The generated `layout.json` is the boundary between
dataset loading and simulation; Isaac Sim never needs to reopen the raw OBJ
files.

Prepare and validate an object:

```bash
/local_data/yz11445/tools/bin/isaacsim-4.5-python \
  -m tasks.dual_arm_breakingbad prepare \
  --input /path/to/fractured_0 \
  --output /local_data/yz11445/datasets/processed/assembly_sim/dual_arm_breakingbad/my_object \
  --overwrite

/local_data/yz11445/tools/bin/isaacsim-4.5-python \
  -m tasks.dual_arm_breakingbad validate \
  --assembly /local_data/yz11445/datasets/processed/assembly_sim/dual_arm_breakingbad/my_object/layout.json
```

Render offline voxel views or launch Isaac Sim:

```bash
/local_data/yz11445/tools/bin/isaacsim-4.5-python \
  -m tasks.dual_arm_breakingbad inspect \
  --assembly /path/to/layout.json \
  --output /local_data/yz11445/experiments/assembly_sim/my_object/voxel_views

/local_data/yz11445/tools/bin/isaacsim-4.5-python \
  -m tasks.dual_arm_breakingbad launch \
  --assembly /path/to/layout.json
```

To build and validate a headless scene while capturing all three cameras at
1920×1440:

```bash
/local_data/yz11445/tools/bin/isaacsim-4.5-python \
  -m tasks.dual_arm_breakingbad build \
  --assembly /path/to/layout.json \
  --output-usd /local_data/yz11445/experiments/assembly_sim/run/scene.usda \
  --capture-directory /local_data/yz11445/experiments/assembly_sim/run/cameras
```

Controls: `Tab` switches arms; `1`-`6` selects an arm joint; `[` and `]`
jog; `H` homes; `O`/`K` open and close the gripper; `R` resets robots and
fragments; `F7`-`F9` select the agent and wrist cameras. Pass an external
`--output-directory` to enable `P` stage export and `I` three-camera capture.

All generated assets, renders, captures, and USD stages must remain outside
the source repository.
