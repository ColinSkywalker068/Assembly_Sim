# Assembly Sim

Assembly Sim is the simulation environment for the robot assembly project at
the NYU Tandon AI4CE Lab. It provides a reusable NVIDIA Isaac Sim workcell and
task-specific simulation pipelines for robotic assembly research.

Every task builds on a consistent scene contract: the same coordinate frame,
floor, worktable, workspace grid pad, FANUC CRX-10iA/L placements, Robotiq
grippers, cameras, physics settings, and interactive controls. This keeps
results aligned while allowing each task to define its own objects, data flow,
and experimental purpose. Dual-arm and single-arm workcell presets are
available in `core`.

## Repository structure

```text
Assembly_Sim/
├── core/
│   ├── assets/robots/        # FANUC and Robotiq USD assets
│   ├── config/               # Shared workcell configuration and presets
│   ├── robots/               # Robot composition, controls, and kinematics
│   ├── scene/                # Reusable scene construction and runtime
│   ├── workcell/             # Table, pad, cameras, materials, and physics
│   └── tests/                # Core contracts and repository checks
├── tasks/
│   ├── demo_render_source/   # CRAG/LEGO storyboard rendering task
│   ├── dual_arm_breakingbad/ # Breaking Bad preprocessing and interactive task
│   └── stack_sawtooth_shulker/ # (this task): <IN_DEVELOPMENT>
├── docs/                     # Architecture, workcell, and task documentation
├── requirements.txt          # Python dependencies outside Isaac Sim
└── README.md
```

- `core/` owns the shared workcell and remains independent of individual
  tasks.
- `tasks/` contains separate simulation applications that compose a core
  workcell with task-specific assets and behavior.
- `docs/` describes the architectural boundaries, workcell presets, and task
  interfaces.

Datasets, processed assets, renders, captures, and generated USD stages should
use directories outside the repository.

## Installation

The repository targets NVIDIA Isaac Sim 4.5 on Linux with Python 3.10. Install
the Isaac Sim 4.5 workstation package by following the
[NVIDIA installation guide](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html),
then clone this repository.

```bash
git clone https://github.com/ColinSkywalker068/Assembly_Sim.git
cd Assembly_Sim

export ISAAC_SIM_PATH=/path/to/isaac-sim
export ISAAC_PYTHON="$ISAAC_SIM_PATH/python.sh"
```

Install the repository dependencies into Isaac Sim's bundled Python
environment:

```bash
"$ISAAC_PYTHON" -m pip install -r requirements.txt
```

Review the
[NVIDIA Omniverse License Agreement](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html),
then accept it for command-line runs:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

Verify the installation from the repository root:

```bash
"$ISAAC_PYTHON" -c "import numpy, scipy, trimesh; print('Python dependencies ready')"
"$ISAAC_PYTHON" -m pytest -q -p no:cacheprovider
```

All module commands below must be run from the repository root. See NVIDIA's
[standalone Python documentation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/python_scripting/manual_standalone_python.html)
for details about `python.sh`.

## Tasks

### Stack sawtooth shulker scene

(this task): <IN_DEVELOPMENT>

### Dual-arm Breaking Bad scene

`tasks.dual_arm_breakingbad` discovers the `piece_*.obj` meshes in one
Breaking Bad fractured-object sample, treats their coordinates as the
ground-truth assembly, normalizes and voxelizes them on a shared grid, creates
visual and collision geometry, and generates deterministic staging poses. The
processed `layout.json` is a portable scene description.

Choose external input, processed-data, and experiment directories:

```bash
export SAMPLE_DIR=/path/to/breaking_bad/object/fractured_0
export ASSEMBLY_DIR=/path/to/processed/assemblies/object_name
export RUN_DIR=/path/to/experiments/object_name
```

Prepare and validate the assembly:

```bash
"$ISAAC_PYTHON" -m tasks.dual_arm_breakingbad prepare \
  --input "$SAMPLE_DIR" \
  --output "$ASSEMBLY_DIR" \
  --overwrite

"$ISAAC_PYTHON" -m tasks.dual_arm_breakingbad validate \
  --assembly "$ASSEMBLY_DIR/layout.json"
```

Render offline views of the voxelized fragments:

```bash
"$ISAAC_PYTHON" -m tasks.dual_arm_breakingbad inspect \
  --assembly "$ASSEMBLY_DIR/layout.json" \
  --output "$RUN_DIR/voxel_views" \
  --image-size 1200
```

Build a headless USD scene and capture the agent and wrist cameras:

```bash
"$ISAAC_PYTHON" -m tasks.dual_arm_breakingbad build \
  --assembly "$ASSEMBLY_DIR/layout.json" \
  --output-usd "$RUN_DIR/scene.usda" \
  --capture-directory "$RUN_DIR/cameras"
```

Launch the interactive Isaac Sim scene:

```bash
"$ISAAC_PYTHON" -m tasks.dual_arm_breakingbad launch \
  --assembly "$ASSEMBLY_DIR/layout.json" \
  --output-directory "$RUN_DIR/interactive"
```

Add `--stream` to the launch command for WebRTC streaming. In the interactive
scene, `Tab` switches arms; `1`-`6` selects a joint; `[` and `]` jog the
selected joint; `H` homes the active arm; `O` and `K` open and close its
gripper; `R` resets the scene; `F7`-`F9` selects the agent and wrist cameras;
`P` exports the stage; and `I` captures all three cameras.

Changing `SAMPLE_DIR`, `ASSEMBLY_DIR`, and `RUN_DIR` is sufficient to process
another compatible Breaking Bad sample.

### Demo storyboard renderer

`tasks.demo_render_source` reproduces the original scripted CRAG/LEGO
dual-FANUC storyboard. It prepares studded fragments, computes the scripted
choreography, and renders the storyboard frames.

Set the source and output paths:

```bash
export GT_ASSEMBLY=/path/to/view_gt.glb
export PREDICTED_ASSEMBLY=/path/to/view_assembly_-1.glb
export STORYBOARD_WORK=/path/to/scratch/demo_storyboard
export STORYBOARD_OUTPUT=/path/to/experiments/demo_storyboard
```

Run the complete pipeline:

```bash
"$ISAAC_PYTHON" -m tasks.demo_render_source run \
  --input "$GT_ASSEMBLY" \
  --predicted "$PREDICTED_ASSEMBLY" \
  --work-dir "$STORYBOARD_WORK" \
  --output "$STORYBOARD_OUTPUT" \
  --fps 30
```

The predicted assembly is optional. Omit `--predicted` when only the
ground-truth storyboard is needed. The `prepare`, `choreograph`, and `render`
subcommands expose the pipeline stages individually:

```bash
"$ISAAC_PYTHON" -m tasks.demo_render_source --help
"$ISAAC_PYTHON" -m tasks.demo_render_source prepare --help
"$ISAAC_PYTHON" -m tasks.demo_render_source choreograph --help
"$ISAAC_PYTHON" -m tasks.demo_render_source render --help
```

## Testing

Run the complete unit test suite with Isaac Sim's Python interpreter:

```bash
"$ISAAC_PYTHON" -m pytest -q -p no:cacheprovider
```

Provide a Breaking Bad sample to include the end-to-end preprocessing test:

```bash
BREAKING_BAD_SAMPLE=/path/to/breaking_bad/object/fractured_0 \
  "$ISAAC_PYTHON" -m pytest -q -p no:cacheprovider
```

## Documentation

- [Architecture](docs/architecture.md)
- [Workcell presets](docs/workcells.md)
- [Task ownership and commands](docs/tasks.md)
- [Stack sawtooth shulker task guide](tasks/stack_sawtooth_shulker/README.md)
- [Dual-arm Breaking Bad task guide](tasks/dual_arm_breakingbad/README.md)
- [Demo storyboard task guide](tasks/demo_render_source/README.md)
