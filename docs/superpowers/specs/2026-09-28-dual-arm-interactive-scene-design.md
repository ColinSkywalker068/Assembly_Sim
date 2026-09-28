# Dual-Arm Interactive Isaac Scene Design

## Goal

Extend the inspection-oriented Isaac Sim 4.5 scene under `06_interactive_scene/` so it reproduces the original dual-arm demo arrangement while retaining interactive physics, manual controls, portable saving, and deterministic validation.

The scene must contain two independently controllable FANUC CRX-10iA/L arms, two Robotiq 2F-85 grippers, the original eight voxelized fragments in the demo's opening lineup, one fixed agent camera, and one wrist camera per arm.

## Source of truth

The original storyboard data in `04_intermediate/choreo.json` defines the arrangement to reproduce:

- left arm base: position `[-0.78, 0.0, 0.75]`, orientation `[1.0, 0.0, 0.0, 0.0]`;
- right arm base: position `[0.78, 0.0, 0.75]`, orientation `[0.0, 0.0, 0.0, 1.0]`;
- agent camera: eye `[1.35, -2.05, 1.72]`, look target `[0.0, 0.05, 0.78]`, focal length `25.0`;
- fragment initial poses: the first pose stored for each piece in `brick_pose`.

These values become explicit, repository-owned configuration under `06_interactive_scene/config/scene.json`; runtime launch must not depend on replaying or importing the storyboard.

## Configuration and scene model

Replace the single `robot` object with an ordered `robots` mapping containing `left` and `right` specifications. Each specification owns its prim path, base transform, home joints, arm and gripper DOF names, gripper limits, and wrist-camera specification. Shared jog size and asset paths remain shared configuration.

The prim hierarchy uses separate namespaces:

- `/World/Robots/Left` and `/World/Grippers/Left`;
- `/World/Robots/Right` and `/World/Grippers/Right`;
- `/World/Cameras/Agent`;
- `/World/Robots/Left/flange/WristCamera`;
- `/World/Robots/Right/flange/WristCamera`.

Each gripper remains a sibling rigid-body tree connected to its arm flange by a fixed joint, matching the working single-arm physics composition and avoiding nested rigid-body violations.

`SceneHandles` stores robots in deterministic `left`, `right` order. Manifest validation requires both robot roots, all eight fragments, and all three cameras. A missing, duplicated, or inconsistently named robot or camera is a validation failure with its exact prim path reported.

## Fragment arrangement and physics

All eight fragment poses match the demo's initial lineup exactly. Fragment visuals, merged occupied-voxel colliders, density-based mass, material properties, asleep-on-launch behavior, and reset velocity clearing remain unchanged.

Both arms must be able to reach the pieces assigned to their side in the original demo. Offline reachability checks run against both configured bases and record which arm can reach each staged fragment and the plate center. Full collision-aware motion planning remains out of scope.

Reset restores both arms to configured home joints, opens both grippers, restores all eight fragment poses, clears rigid-body velocities, and returns fragments to sleep.

## Interaction model

The runtime has one active robot at a time:

- `Tab` toggles the active robot between left and right;
- `1` through `6` select the active robot's joint;
- `[` and `]` jog that joint;
- `H` homes the active arm;
- `O` and `K` open and close the active gripper;
- `R` resets both robots, both grippers, and all fragments;
- `F7`, `F8`, and `F9` select the agent, left-wrist, and right-wrist cameras;
- `I` captures all three cameras;
- `P` saves the portable stage.

Console feedback names the active robot for every selection and command. Existing protections against key-repeat and viewport-navigation conflicts remain in force.

## Cameras and outputs

The agent camera adopts the demo framing. Each wrist camera uses the same flange-local mount and optical orientation, so the mirrored arm base naturally produces the corresponding world view.

The output contract contains:

- `06_interactive_scene/inspections/agent_camera.png`;
- `06_interactive_scene/inspections/left_wrist_camera.png`;
- `06_interactive_scene/inspections/right_wrist_camera.png`;
- `06_interactive_scene/generated/interactive_scene.usd`.

The previous singular `wrist_camera.png` output is replaced rather than retained as an ambiguous alias.

## Validation

Pure-Python tests cover configuration parsing, exact demo transforms, deterministic robot order, key bindings, camera paths, manifest expectations, reachability reporting, reset targeting, and output-path portability.

Isaac Sim validation must:

1. create exactly two six-DOF FANUC arms and two eight-DOF Robotiq grippers;
2. jog each arm independently and observe motion only on the commanded articulation;
3. command each gripper and observe finger motion;
4. impose fragment velocity, reset the scene, and verify both robots and all fragments return to configured state;
5. simulate at least three seconds with the exact demo lineup stable;
6. save and reopen the USD without missing required prims or absolute repository references;
7. capture non-empty 640 by 480 RGB images from all three cameras;
8. launch the visible GUI, select each camera, and exit cleanly.

The scoped Python suite and Isaac validation commands documented in the README are the acceptance interface.

## Compatibility and scope

All new and modified scene files stay under `06_interactive_scene/` except this design and its implementation plan. The legacy storyboard inputs, scripts, assets, intermediate files, and outputs remain unmodified and are read-only references.

This change does not add autonomous grasping, coordinated dual-arm trajectories, collision-aware planning, assembly rewards, policy training, data collection, or task evaluation. Those remain later phases built on this scene.

## Completion criteria

The change is complete when the documented GUI command opens a stable scene matching the demo's two-arm and fragment arrangement; both arms and grippers can be selected and moved; all three cameras can be viewed and captured; reset, saving, and portable reopen work; the original demo remains unchanged; and all pure-Python and Isaac Sim validations pass.
