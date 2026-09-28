# Interactive Isaac Sim Scene Design

## Purpose

Build a first inspection-oriented Isaac Sim 4.5 scene from the existing FANUC skull-assembly demo. The scene must preserve the demo's voxelized eight-fragment skull, table-scale composition, FANUC CRX-10iA/L, and Robotiq 2F-85 appearance while replacing the render-only two-arm storyboard with a single-arm, physics-enabled environment.

This first version is for opening the scene in the Isaac Sim GUI, inspecting it from multiple cameras, and performing simple robot and gripper movements. Policy training, task rewards, dataset collection, domain randomization, and multi-environment execution are intentionally excluded.

## Constraints

- Target Isaac Sim 4.5 on Windows using the existing `D:\i45` Python environment.
- Support the available RTX 4060 Laptop GPU with 8 GB VRAM and approximately 16 GB system RAM.
- Keep all new scene source, configuration, documentation, and generated artifacts under `06_interactive_scene/`.
- Do not modify the behavior of the existing storyboard scripts or overwrite their outputs.
- Reuse the robot USD files in `02_robot_assets/` and fragment data in `04_intermediate/bricks/` through repository-relative paths.
- Use one FANUC arm and one Robotiq gripper.
- Include one fixed front camera and one wrist-mounted camera.
- Favor deterministic, stable, low-memory behavior over photorealistic rendering.

## Directory Layout

```text
06_interactive_scene/
  README.md
  config/
    scene.json
  scripts/
    build_scene.py
    run_scene.py
    scene_assets.py
    scene_controls.py
    validate_scene.py
  generated/
    interactive_scene.usd
  inspections/
    agent_camera.png
    wrist_camera.png
```

`generated/` and `inspections/` are outputs owned by the interactive-scene workflow. The source scripts must recreate them without reading storyboard animation state from `04_intermediate/choreo.json`.

## Architecture

### Configuration

`config/scene.json` is the authoritative description of the first scene. It contains:

- repository-relative paths for the arm, gripper, probe data, fragment layout, and fragment meshes;
- physics and render rates;
- table, floor, plate, robot-base, and fragment staging poses;
- arm home joint positions and joint-jog increments;
- gripper open and closed commands;
- camera transforms, focal lengths, clipping planes, and 640 by 480 render resolution;
- fragment density, friction, restitution, damping, and collision settings.

All paths are resolved relative to the configuration file or repository root. Missing or incompatible assets cause an explicit startup error naming the path or prim that failed.

### Asset construction

`scene_assets.py` provides focused functions for creating materials, visual meshes, collision geometry, rigid bodies, cameras, and static support geometry.

The detailed NPZ triangle meshes remain visual geometry. Fragment collision geometry is derived from the occupied cells in `layout.json`, with adjacent cells merged into larger axis-aligned boxes where possible. This retains the voxel shape while avoiding dynamic triangle-mesh collision and limiting collider count. Studs remain visual-only in this first version.

The base plate, table, table legs, and floor are static colliders. Each fragment is a separate rigid body with mass derived from configured density and occupied voxel volume. Initial poses place every fragment on the table with separation sufficient to avoid initial overlap.

### Robot and gripper

`build_scene.py` references the FANUC arm USD once and places it at a table-relative base pose. The Robotiq gripper is attached to the arm flange with a fixed transform while preserving controllable finger joints. The implementation may use the Isaac Sim 4.5 Robot Assembler or an equivalent fixed-joint composition, selected by the validation that produces a single stable controllable system.

The builder must not use the storyboard's approach of stripping gripper physics and directly animating link transforms. Arm joints use articulation position targets. Gripper mimic joints use the relationships already captured by `asm_kin.Gripper`, adapted behind a small control interface so callers issue only a normalized open fraction.

The initial arm base, staging poses, and plate pose are selected by offline IK checks using the existing `asm_kin.py` model. The final positions are stored in `scene.json`, making launches deterministic. All fragment grasp centers and the plate center must have at least one collision-free candidate approach pose within joint limits. Full motion planning is not required.

### Cameras

The fixed agent camera is placed in front of the robot and angled downward so the arm, staged fragments, plate, and assembly area fit within the image. The wrist camera is a child of the flange or gripper mount and looks forward and slightly downward toward the grasp region.

Both cameras are ordinary USD camera prims and have optional RGB and distance-to-image-plane annotators for inspection snapshots. Camera products are created only when preview or snapshot output is requested, reducing VRAM use during normal GUI inspection. Users can also select either camera from the Isaac Sim viewport.

### Runtime and controls

`run_scene.py` launches Isaac Sim in non-headless mode, opens the generated scene or builds it if absent, and starts a small control loop. Controls provide:

- select and jog one of the six arm joints in positive or negative increments;
- return the arm to its configured home pose;
- open and close the gripper;
- reset arm joints and all fragment poses;
- save the current stage to the configured generated USD path;
- capture inspection images from both cameras;
- quit cleanly.

Keyboard bindings and equivalent command-line options are documented in `06_interactive_scene/README.md`. Joint targets are clamped to the limits from `fanuc_probe.json`. Commands report rejected or clamped targets clearly.

## Build and Data Flow

1. Resolve and validate configuration and source asset paths.
2. Start Isaac Sim 4.5 with performance-oriented renderer settings.
3. Create a fresh stage with meters, gravity, physics scene, floor, and lighting.
4. Add the table and base plate as static collision objects.
5. Reference and compose the FANUC arm and Robotiq gripper.
6. Load the eight NPZ visual meshes and construct voxel collision bodies.
7. Add the fixed and wrist cameras.
8. Reset the simulation, apply home targets, and allow physics to settle.
9. Run structural and stability checks.
10. Save `generated/interactive_scene.usd` with portable repository-relative references.
11. Optionally run the GUI control loop or capture camera inspection images.

The saved USD is an inspection artifact. `scene.json` plus the builder scripts remain authoritative so scene generation is reproducible.

## Failure Handling

- Abort before stage construction if a required asset or configuration field is missing.
- Abort if the robot articulation does not expose the expected six arm joints.
- Abort if the gripper cannot be attached or its finger joints cannot be controlled.
- Abort if a fragment mesh or cell list is empty or malformed.
- Report camera creation or annotator failures without silently substituting the GUI viewport.
- During validation, fail if any initial body contains non-finite state, penetrates the table substantially, or leaves the configured workspace after settling.
- Preserve the generated USD from the last successful build if a later rebuild fails by saving through a temporary stage path and replacing the output only after validation.

## Validation

### Fast tests outside Isaac Sim

- Parse `scene.json` and resolve every referenced input.
- Confirm exactly eight uniquely named fragments.
- Verify voxel collider merging preserves the occupied-cell volume.
- Verify initial fragment bounding boxes do not overlap and rest above the table.
- Check all configured joint targets against probed joint limits.
- Run IK checks for the plate center and every fragment grasp center.

### Isaac Sim headless validation

- Build the stage and confirm one arm articulation, one gripper, eight fragment rigid bodies, and two cameras.
- Step physics for at least three simulated seconds and check for non-finite or unstable body states.
- Move one arm joint by a small increment and verify measured joint motion.
- Open and close the gripper and verify finger-link motion.
- Reset the scene and compare robot and fragment poses with configuration.
- Save, close, reopen, and revalidate the generated USD.
- Capture one RGB inspection image from each camera at 640 by 480.

### GUI acceptance check

- Launch the scene in Isaac Sim 4.5 on the local Windows machine.
- Confirm the full workspace is visible from the fixed camera.
- Confirm the wrist camera sees the gripper and nearby table region.
- Confirm joint jogging, home, gripper, reset, save, and snapshot controls operate without a crash.
- Confirm idle GPU memory remains within the 8 GB available budget.

## Acceptance Criteria

The first version is complete when a documented command launches an interactive Isaac Sim 4.5 GUI showing one FANUC/Robotiq robot, the table and base plate, all eight original voxelized fragments, and both cameras; the user can make minor arm and gripper movements, reset the scene, and inspect saved images from both cameras; headless validation passes; and the generated scene remains fully separated under `06_interactive_scene/` without changing the original storyboard workflow.

## Deferred Work

- Assembly task definitions and fragment grouping
- Reward, termination, and evaluation metrics
- Policy action and observation interfaces
- Dataset recording
- Domain randomization
- Parallel environments and Isaac Lab integration
- Accurate stud-and-socket mating or connector constraints
- Motion planning and collision-aware autonomous trajectories
