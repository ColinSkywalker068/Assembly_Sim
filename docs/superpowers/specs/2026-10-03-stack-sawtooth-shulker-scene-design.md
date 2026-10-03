# Stack Sawtooth Shulker Initial Scene Design

**Date:** 2026-10-03

## Purpose

Add the first deterministic scene for `tasks.stack_sawtooth_shulker`. The scene
is a clean single-arm benchmark setup for a known two-piece keyed assembly. This
milestone constructs and validates the initial scene only. It does not execute
robot motion, record data, evaluate task success, randomize episodes, plan
grasps or trajectories, or expose a policy interface.

The language-level task represented by the scene is:

> Assemble the blue and green pieces inside the red square.

## Scope

The milestone includes:

- one existing right-side FANUC CRX-10iA/L with its Robotiq 2F-85 gripper;
- the existing table, lighting, cameras, physics, and red tape target;
- two task-owned dynamic rigid fragments in deterministic poses;
- an analytical description of their intended assembled relationship;
- scene construction, launch for visual inspection, optional external USD
  export, structural validation, and tests.

The milestone excludes:

- robot motion commands, keyboard jogging, scripted trajectories, and grasp
  execution;
- camera, state, demonstration, or dataset recording;
- task success evaluation and post-release stability measurement;
- observations, actions, rewards, policies, or learning integration;
- placement or appearance randomization;
- high-level matching, graph reasoning, or assembly inference;
- distractors and additional fragments.

## Architecture

The new package is `tasks/stack_sawtooth_shulker`. It is a task-local
composition root and imports the shared workcell through ordinary `core` APIs.
`core` does not import the task, and this task does not import another task.

The scene selects the existing `single_arm_right` preset. Core continues to
own the floor, table, red tape, lighting, physics material, robot and gripper
assets, robot placement, cameras, and base simulation lifecycle. The task owns
only its analytical fragment geometry, object materials, object physics,
initial poses, intended goal transforms, scene additions, and validation.

The package is divided into focused modules:

```text
tasks/stack_sawtooth_shulker/
├── README.md
├── __init__.py
├── __main__.py
├── config.py
├── geometry.py
├── assets.py
├── builder.py
├── runtime.py
├── validation.py
└── tests/
```

- `config.py` defines immutable task configuration and deterministic poses.
- `geometry.py` defines and validates canonical and fragment-local voxel cells.
- `assets.py` authors visual cubes, collision cubes, materials, and rigid-body
  mass properties.
- `builder.py` composes the two fragments onto a core workcell.
- `runtime.py` runs a view-only simulation loop without task controls.
- `validation.py` provides pure configuration checks and USD manifest checks.
- `__main__.py` exposes build and launch commands.

No preprocessing step, generated mesh asset, or layout file is required. The
geometry is small, exact, and known analytically, so procedural task-local USD
authoring is the simplest source of truth.

## Shared Workcell and Coordinate Frame

The current shared red tape target is a square centered at
`(0.0, 0.0, 0.75)` m with a `0.55 x 0.55` m footprint. Its right-hand outer
edge is at `x = 0.275` m. The right robot base axis is at
`(0.78, 0.0, 0.75)` m. The task treats the table top as `z = 0.75` m.

The task unit is `u = 0.08 m`, so the assembled object occupies a
`0.24 x 0.24 x 0.24` m cube. The shared `0.55 x 0.55` m red target is
independent of `u`. Task validation requires a square target large enough to
contain the assembled three-unit footprint.

## Canonical Assembly Geometry

Canonical voxel coordinates are integer triplets `(x, y, z)` with each axis in
`{0, 1, 2}`. The complete object is the set of all 27 coordinates.

Fragment A contains 14 voxels:

- every `(x, y, 0)` bottom-layer voxel;
- middle-layer corners `(0, 0, 1)`, `(0, 2, 1)`, `(2, 0, 1)`, and
  `(2, 2, 1)`;
- middle-layer center `(1, 1, 1)`.

Fragment B contains the 13 complementary voxels:

- middle-layer edge centers `(1, 0, 1)`, `(0, 1, 1)`, `(2, 1, 1)`, and
  `(1, 2, 1)`;
- every `(x, y, 2)` top-layer voxel.

Validation requires both sets to be nonempty and disjoint and their union to
equal the complete 27-cell cube. This makes the keyed relationship a tested
contract rather than a visual convention.

Each fragment uses a local frame centered in X and Y and placed at the bottom
of that fragment's local geometry. Fragment A retains canonical Z coordinates.
Fragment B subtracts one from canonical Z locally, allowing it to rest on the
table in its initial pose. The intended assembled transforms restore the
canonical relationship:

- Fragment A root: assembly origin;
- Fragment B root: assembly origin translated by `(0, 0, u)`.

Both orientations are the identity quaternion. Applying these transforms must
reconstruct each canonical cell exactly once.

## Visual and Collision Geometry

Each occupied cell creates one task-owned visual cube with side length `u`.
Visual cubes retain the exact voxel-grid geometry and use a shared material per
fragment:

- Fragment A: blue;
- Fragment B: green.

Neither fragment uses red, preserving contrast with the shared target tape.

Each occupied cell also creates one aligned collision cube. Collision cubes
use side length `u - clearance`, with an initial total clearance of `0.002` m
per dimension. The 1 mm inset on each face leaves a small practical gap at
mating contacts while visuals retain the exact keyed shape. Clearance is a
validated configuration value satisfying `0 <= clearance < u`.

All visual and collision cubes for one fragment are children of one fragment
root. Each root is one dynamic rigid body; no fragment is fixed to the table.
The task uses a lightweight nominal density of `20 kg/m^3` and authors an
explicit mass derived from the nominal occupied voxel volume:

- Fragment A: approximately `0.14336` kg;
- Fragment B: approximately `0.13312` kg.

Both fragments share the core contact physics material and inherit its
friction and restitution behavior. Their rigid bodies begin asleep after scene
initialization so the deterministic setup does not acquire avoidable initial
motion.

## Initial Layout

Both fragments retain identity orientation and form one front-to-back line
along Y between the red target and the right robot:

```text
target region -> Y-directed fragment line -> right robot
```

The deterministic root positions are:

- Fragment A: `(0.48, -0.185, 0.75)` m;
- Fragment B: `(0.48, +0.185, 0.75)` m.

Because each footprint is `0.24 x 0.24` m:

- both fragments occupy X interval `[0.360, 0.600]` m;
- the red tape ends at `x = 0.275` m, leaving `0.085` m clearance;
- the observed physical robot base begins near `x = 0.685` m, leaving
  approximately `0.085` m clearance;
- Fragment A occupies Y interval `[-0.305, -0.065]` m;
- Fragment B occupies Y interval `[0.065, 0.305]` m;
- the fragments have a `0.130` m gap and remain within the table footprint.

These placements solve the static fit problem. This milestone does not claim
collision-free arm sweep paths or verified grasp reachability.

## Intended Goal Configuration

The assembly origin is centered inside the red target with its bottom on the
table. Fragment A rests at that origin. Fragment B is translated upward by one
unit and interlocks with Fragment A's middle-layer pattern. Together their
visual occupancy reconstructs the complete 3-by-3-by-3 cube.

The goal transforms are configuration metadata for validation and later task
development. This milestone does not move fragments into those transforms and
does not decide whether an episode has succeeded.

## USD Scene Structure

Task-owned prims use a disjoint namespace:

```text
/World/Task
├── TargetMetadata
└── Fragments
    ├── FragmentA
    │   ├── Visuals/Voxel_*
    │   └── Colliders/Voxel_*
    └── FragmentB
        ├── Visuals/Voxel_*
        └── Colliders/Voxel_*
```

`TargetMetadata` records task identification, unit size, target size, and the
intended goal transforms without duplicating or replacing the core-authored red
tape geometry.

## Commands and Runtime

The initial command interface is:

```bash
<isaac-python> -m tasks.stack_sawtooth_shulker build \
  --output-usd /external/path/stack_sawtooth_shulker.usda

<isaac-python> -m tasks.stack_sawtooth_shulker launch
```

`build` constructs a headless scene, validates it, and exports it only to an
explicit path outside the repository. `launch` opens the scene and advances the
simulation for visual inspection. It installs no task keyboard controller and
does not move the robot or fragments intentionally. A hidden bounded-frame
option may be used only by automated smoke validation so the launch loop can
terminate deterministically.

The core builder's normal robot initialization and short settling sequence are
scene initialization, not task motion execution.

## Validation and Error Handling

Pure validation runs before Isaac starts where possible. It rejects:

- a nonpositive unit size or density;
- invalid clearance;
- a non-square target or one too small for the assembled three-unit footprint;
- incorrect, duplicate, overlapping, or incomplete canonical cells;
- inconsistent local-to-canonical mappings;
- invalid colors, poses, or non-finite values;
- initial AABB overlap, target overlap, table overflow, or loss of required
  static clearances.

USD validation checks:

- the core workcell manifest for the single-right-arm preset;
- task root, metadata, fragment root, visual, and collider prims;
- exactly 14 visual and collision cells for Fragment A;
- exactly 13 visual and collision cells for Fragment B;
- dynamic rigid-body APIs on both fragment roots;
- authored masses matching the analytical values;
- expected initial world transforms and identity orientations.

Scene validation failure prevents USD export. The application is closed in a
`finally` block on both successful and failed builds.

## Testing and Acceptance

Pure-Python tests cover:

- exact Fragment A and Fragment B memberships and layer patterns;
- disjointness, complementarity, and 27-cell reconstruction;
- fragment-local conversion and intended assembled transforms;
- target-to-unit relationship;
- analytical bounds, masses, collider sizes, and initial clearances;
- deterministic configuration;
- CLI commands and rejection of repository-internal output paths;
- expected task and core manifests.

After pure tests pass, one headless Isaac smoke run constructs and validates the
scene and exports a temporary USD outside the repository. Before that run, GPU
usage is inspected and the process is explicitly limited to at most one GPU.
The smoke run does not command robot movement or record camera or state data.

The milestone is complete when the full repository test suite passes and the
headless scene contains the shared single-right-arm workcell plus two correctly
colored, lightweight, dynamic complementary fragments in the specified
Y-directed initial line, with no task-level motion or recording behavior.

## Deferred Work

Future milestones may add grasp feasibility experiments, robot execution,
collision-aware planning, success evaluation, stability windows, recording,
policy interfaces, and controlled randomization. Those changes must preserve
the order-independent final assembly objective, but none are part of this
scene-construction milestone.
