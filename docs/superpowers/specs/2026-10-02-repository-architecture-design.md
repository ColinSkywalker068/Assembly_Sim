# Assembly_Sim Repository Architecture

**Date:** 2026-10-02

## Purpose

`Assembly_Sim` is a source repository for reusable Isaac Sim workcell
infrastructure and independently meaningful simulation tasks. It is not a
dataset store, generated-asset store, or experiment archive.

The reorganization must preserve the current visual and physical identity of
the workcell while making that identity reusable by tasks with different
purposes. A storyboard renderer, an interactive fragmented-object scene, and a
future single-arm task should compose the same table, background, robot models,
robot mounting positions, workspace pad, camera mounts, and physics defaults.

## Repository Boundary

The only top-level source directories will be:

```text
Assembly_Sim/
├── core/
├── tasks/
└── docs/
```

Repository metadata and entry documentation may remain as root files, including
`.git/`, `.gitattributes`, `.gitignore`, and `README.md`. Python cache and test
cache directories are ignored and are not part of the architecture.

Raw inputs, processed objects, generated USD stages, inspection images, videos,
and logs live outside the repository under the Maxwell workspace conventions:

- raw inputs: `/local_data/yz11445/datasets/raw/`
- processed object representations: `/local_data/yz11445/datasets/processed/`
- renders, scene captures, and run logs: `/local_data/yz11445/experiments/`
- disposable intermediates: `/local_data/yz11445/scratch/`

Commands receive these locations through explicit path arguments. No source
module embeds a user-specific absolute path.

## Chosen Architecture

The repository will use a shared platform with thin task packages.

Two alternatives were considered:

1. Keep each task self-contained and duplicate its workcell builder. This makes
   tasks easy to copy but allows table, robot, camera, and physics definitions to
   drift.
2. Build a registry/plugin framework in which tasks dynamically register scene
   components. This is flexible but adds machinery that two known tasks do not
   require.
3. Provide ordinary importable `core` modules and declarative named presets,
   with each task acting as a small composition root. This keeps shared behavior
   explicit, testable, and reusable without introducing a plugin system.

Option 3 is selected. Task-local code is the default. A utility moves into
`core/` only when it represents a stable workcell contract or has demonstrated
task-independent use.

## Target Layout

```text
Assembly_Sim/
├── README.md
├── .gitattributes
├── .gitignore
├── core/
│   ├── __init__.py
│   ├── assets/
│   │   └── robots/
│   │       ├── fanuc_crx10ial/
│   │       └── robotiq_2f85/
│   ├── config/
│   │   ├── workcell.json
│   │   └── presets/
│   │       ├── dual_arm.json
│   │       ├── single_arm_left.json
│   │       └── single_arm_right.json
│   ├── workcell/
│   │   ├── config.py
│   │   ├── environment.py
│   │   ├── cameras.py
│   │   ├── physics.py
│   │   └── geometry.py
│   ├── robots/
│   │   ├── config.py
│   │   ├── controls.py
│   │   ├── kinematics.py
│   │   └── builder.py
│   ├── scene/
│   │   ├── builder.py
│   │   ├── runtime.py
│   │   └── validation.py
│   └── tests/
├── tasks/
│   ├── __init__.py
│   ├── demo_render_source/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── config/
│   │   ├── preprocessing.py
│   │   ├── choreography.py
│   │   ├── storyboard.py
│   │   └── tests/
│   └── dual_arm_breakingbad/
│       ├── README.md
│       ├── __init__.py
│       ├── __main__.py
│       ├── config/
│       ├── dataset.py
│       ├── model.py
│       ├── voxelization.py
│       ├── staging.py
│       ├── assets.py
│       ├── scene.py
│       ├── rendering.py
│       └── tests/
└── docs/
    ├── architecture.md
    ├── workcells.md
    ├── tasks.md
    └── superpowers/
```

Exact file names may be consolidated during implementation when two current
modules are inseparable, but ownership and dependency direction must remain as
specified here.

## Core Responsibilities

### Workcell contract

`core` owns the invariant scene foundation:

- floor, table, background, lighting, and materials;
- workspace grid pad geometry, pose, and collision behavior;
- physics scene defaults and time steps;
- FANUC CRX-10iA/L and Robotiq 2F-85 assets;
- robot composition, joint controls, gripper controls, and reset behavior;
- fixed agent-camera and wrist-camera mounting conventions;
- canonical robot mounting locations and home joint configurations;
- generic stage creation, validation, and Isaac Sim runtime helpers.

Tasks may select a named preset but cannot override invariant workcell fields in
their task configuration. The core configuration loader validates this boundary.

### Named presets

The initial preset set is:

- `dual_arm`: the current left and right FANUC placements;
- `single_arm_left`: only the robot at the current left/arm0 placement;
- `single_arm_right`: only the robot at the current right/arm1 placement.

All three use the same coordinate frame, table, pad, visual environment,
materials, and physics. The single-arm presets are selections from the existing
dual-arm workcell, not independently tuned scenes.

### Stable versus task-local geometry

Environment geometry, robot attachment logic, generic collider helpers, and the
workspace pad belong in `core`. Breaking Bad mesh loading, fragment voxelization,
voxel identity, voxel-cell merging for fragment colliders, and bilateral fragment
staging remain task-local because they define the current assembly task rather
than the workcell itself.

If another task later needs the same voxel representation, the tested neutral
portion can be promoted to `core.geometry` without moving its dataset adapter or
task staging policy.

## Task Responsibilities

### `tasks/demo_render_source`

This task exists to reproduce and extend the original scripted storyboard. It
owns:

- CRAG-specific source interpretation used by the storyboard;
- brick/stud conversion needed only by that demo;
- grasp annotations and scripted choreography;
- offline kinematic timeline generation;
- shot sequencing, captions, and frame/video rendering;
- a task CLI that takes external input, intermediate, and output paths.

It imports robot kinematics, robot assets, and workcell construction from
`core`. Historical CRAG GLBs, generated brick NPZ files, choreography JSON,
frames, videos, and HTML outputs are external data, not package resources.

### `tasks/dual_arm_breakingbad`

This task converts one Breaking Bad fragmented-object sample into an interactive
dual-arm scene. It owns:

- the Breaking Bad directory adapter and object identity rules;
- the complete-fragment assembly model and ground-truth transforms;
- normalization, orientation, scaling, and common-grid voxelization;
- fragment occupancy cleanup and visual mesh export;
- voxel-derived collision boxes;
- deterministic bilateral staging around the shared workspace pad;
- layout metadata validation;
- assembly-specific scene authoring and fragment reset behavior;
- original-mesh and voxel inspection rendering;
- prepare, validate, inspect, build, and launch commands.

The task always selects the core `dual_arm` preset. The task does not define its
own table, robot placements, cameras, pad, or physics defaults.

The current CRAG GLB loader may remain as a compatibility adapter inside this
task while it is useful, but the task's name and primary documented input remain
Breaking Bad. It does not depend on CRAG predictions.

## Configuration and Data Flow

A task scene is composed in one direction:

```text
external task input
        ↓
task preprocessing and task metadata
        ↓
named core workcell preset + task object description
        ↓
core stage foundation + task scene additions
        ↓
external generated stage, captures, logs, or video
```

Core preset files contain workcell-owned fields. Task configuration contains
only task inputs, task parameters, object behavior, and output paths. A task
cannot shadow core keys such as table pose, robot base transforms, pad pose,
camera mounts, gravity, or physics materials.

Generated assembly metadata remains self-contained with respect to its processed
fragment assets: after preprocessing, scene construction does not need the raw
Breaking Bad directory. Metadata records the selected core preset identifier and
its schema version rather than copying the workcell configuration into every
object.

## Command Interface

Tasks are run as Python modules from the repository root:

```bash
<isaac-python> -m tasks.demo_render_source render \
  --input /path/to/crag-case \
  --work-dir /local_data/yz11445/scratch/demo-render \
  --output /local_data/yz11445/experiments/demo-render

<isaac-python> -m tasks.dual_arm_breakingbad prepare \
  --input /path/to/fractured_0 \
  --output /local_data/yz11445/datasets/processed/assembly-sim/object-id

<isaac-python> -m tasks.dual_arm_breakingbad launch \
  --assembly /local_data/yz11445/datasets/processed/assembly-sim/object-id/layout.json
```

Commands do not write to the repository by default. Required output arguments
or explicitly configured external output roots prevent generated artifacts from
silently returning to source directories.

## Migration of Current Content

The current repository maps as follows:

| Current content | New owner or destination |
| --- | --- |
| `02_robot_assets/` | `core/assets/robots/` |
| stable parts of `06_interactive_scene/scripts/` | `core/workcell/`, `core/robots/`, and `core/scene/` |
| `03_scripts/asm_kin.py` | `core/robots/kinematics.py` |
| legacy `asm_bricks`, `asm_choreo`, `asm_replay`, `fanuc_probe` | `tasks/demo_render_source/` |
| `03_scripts/fragment_assembly/` | `tasks/dual_arm_breakingbad/` |
| `assembly_pipeline.py` | `tasks/dual_arm_breakingbad/__main__.py` |
| assembly-specific parts of `06_interactive_scene/` | `tasks/dual_arm_breakingbad/` |
| current tests | corresponding `core/tests/` or task-local `tests/` |
| `01_crag_case/` | external raw dataset storage |
| `04_intermediate/` | external processed-data or scratch storage |
| `05_outputs/` | external experiment storage |
| generated USD and inspection images | external experiment storage |
| architectural/history documentation | `docs/` |

Before removing any tracked data or render artifact from the repository, the
implementation verifies that an external copy exists at an explicit destination.
The Git history remains a recovery path, but it is not treated as the migration
mechanism.

## Import and Dependency Rules

- `core` imports no task package.
- A task may import public `core` APIs.
- Tasks do not import one another.
- Shared behavior is exposed through normal Python packages; modules do not
  mutate `sys.path` to find numbered directories.
- Core public APIs accept typed values or mappings and do not know dataset names.
- Isaac-dependent imports remain lazy where needed so configuration, layout,
  geometry, and CLI tests can run with the installed Isaac Python interpreter
  without launching the simulator.

## Compatibility Strategy

The refactor changes source paths intentionally, but it preserves user-facing
capabilities. The new task CLI supports the current Breaking Bad prepare,
validate, launch, streaming, and inspection flows. The storyboard task retains
the legacy render sequence using externalized inputs and outputs.

Temporary forwarding scripts at the old paths are not retained because the
requested repository boundary removes those top-level directories. Migration
documentation provides old-to-new command mappings instead.

## Validation

The reorganization is complete when:

1. The only top-level source directories are `core/`, `tasks/`, and `docs/`.
2. `core` can load and validate all three workcell presets.
3. The dual-arm preset reproduces the current table, pad, robot, camera, physics,
   and reset contracts.
4. Single-arm presets use the exact left or right pose from the dual-arm preset.
5. `demo_render_source` can construct its storyboard scene from external inputs.
6. `dual_arm_breakingbad` can preprocess the existing three-piece sample,
   validate its layout, build the scene, and capture all three cameras.
7. Neither task writes generated data into the repository during normal use.
8. Core tests and both task test suites pass from the repository root.
9. A repository scan finds no active imports or configuration paths referring to
   the numbered legacy directories.

## Out of Scope

- A general task plugin registry or discovery framework;
- dataset-wide batch processing;
- learned policies, planning, or automatic reassembly;
- redesigning the current workcell appearance or physics;
- adding a new single-arm task beyond defining and testing the reusable presets;
- preserving old numbered source paths through compatibility wrappers.
