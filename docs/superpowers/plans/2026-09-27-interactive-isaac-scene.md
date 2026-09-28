# Interactive Isaac Sim Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate, reproducible Isaac Sim 4.5 scene for GUI inspection and simple physical interaction with one FANUC arm, one Robotiq gripper, eight existing voxel fragments, and two cameras.

**Architecture:** A pure-Python configuration and geometry layer prepares paths, staging poses, voxel colliders, and reachability data without starting Isaac Sim. A thin Isaac-specific layer authors and validates the USD stage, while a runtime layer owns articulation control, keyboard input, camera capture, reset, and saving. `06_interactive_scene/` is self-contained and only reads assets from the existing demo directories.

**Tech Stack:** Python 3.10, NumPy, pytest, Isaac Sim 4.5 Core API, OpenUSD (`pxr`), PhysX schemas, Replicator annotators, JSON, USD

**Spec:** `docs/superpowers/specs/2026-09-27-interactive-isaac-scene-design.md`

## Global Constraints

- Target Isaac Sim 4.5 on Windows using `D:\i45\python.exe`.
- Keep every new scene source, configuration, test, generated stage, and inspection image under `06_interactive_scene/`.
- Do not change existing storyboard behavior or overwrite files in `01_crag_case/` through `05_outputs/`.
- Reference robot assets from `02_robot_assets/` and fragment inputs from `04_intermediate/bricks/` with repository-relative configuration paths.
- Use one FANUC CRX-10iA/L, one Robotiq 2F-85, eight fragments, one fixed front camera, and one wrist camera.
- Use 640 by 480 camera products, a 120 Hz physics step, a 30 Hz render step, gravity `[0, 0, -9.81]`, and performance-oriented rendering.
- Treat detailed NPZ meshes as visual geometry and merged occupied-cell boxes as fragment collision geometry; studs are visual-only.
- Do not add task rewards, dataset collection, randomization, Isaac Lab, or multi-environment training.
- Use the repository's configured Git identity and do not add Codex or co-author attribution trailers to commits.

## Review Focus

- Paths containing spaces or a relocated repository must resolve from `scene.json`, and missing paths must name the failing field and resolved path.
- Sparse, duplicated, or malformed voxel cells must not produce overlapping/zero-volume collision boxes or silently change occupied volume.
- A changed gripper USD hierarchy must fail with the missing prim or expected DOF name instead of producing a visual-only gripper.
- Initial penetrations or unstable fragments must fail validation with fragment names and final poses.
- Camera annotators that are not ready on their first frame must retry for a bounded number of frames and fail clearly without emitting empty images.

---

### Task 1: Configuration and portable path contract

**Files:**
- Create: `06_interactive_scene/config/scene.json`
- Create: `06_interactive_scene/scripts/__init__.py`
- Create: `06_interactive_scene/scripts/scene_config.py`
- Create: `06_interactive_scene/tests/test_scene_config.py`

**Interfaces:**
- Produces: `SceneConfig.load(path: Path) -> SceneConfig`, `SceneConfig.resolve_repo_path(key: str) -> Path`, `SceneConfig.validate_inputs() -> None`, `SceneConfig.physics_dt`, `SceneConfig.render_dt`, `SceneConfig.fragment_names`
- Consumes: no earlier task interfaces

- [ ] **Step 1: Write failing configuration tests**

Add tests named `test_loads_required_scene_contract`, `test_paths_resolve_after_repo_relocation`, `test_missing_asset_reports_field_and_resolved_path`, and `test_rejects_non_eight_fragment_contract`. Assert schema version `1`, `physics_dt == 1/120`, `render_dt == 1/30`, camera resolution `[640, 480]`, exactly `piece_0` through `piece_7`, and error text containing both the missing field and absolute resolved path.

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m pytest 06_interactive_scene/tests/test_scene_config.py -v`

Expected: FAIL because `scene_config` and `scene.json` do not exist.

- [ ] **Step 3: Implement the configuration API and initial scene contract**

Implement immutable configuration dataclasses and the three public methods above. `scene.json` must define repository-relative inputs, generated output paths, expected arm/gripper DOF names, physics/render values, camera specifications, material/rigid-body values, robot home position, and deterministic placement parameters. Resolve the repository root by walking upward from `scene.json` for `README.md` and `02_robot_assets/`, never from the process working directory.

- [ ] **Step 4: Run configuration tests**

Run: `python -m pytest 06_interactive_scene/tests/test_scene_config.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit the configuration contract**

```powershell
git add 06_interactive_scene/config/scene.json 06_interactive_scene/scripts/__init__.py 06_interactive_scene/scripts/scene_config.py 06_interactive_scene/tests/test_scene_config.py
git commit -m "feat: define interactive scene configuration"
```

### Task 2: Voxel collision decomposition and reachable staging layout

**Files:**
- Create: `06_interactive_scene/scripts/scene_geometry.py`
- Create: `06_interactive_scene/scripts/plan_layout.py`
- Create: `06_interactive_scene/tests/test_scene_geometry.py`
- Create: `06_interactive_scene/tests/test_plan_layout.py`
- Modify: `06_interactive_scene/config/scene.json`

**Interfaces:**
- Consumes: `SceneConfig.load`, resolved layout/probe paths, and `03_scripts/asm_kin.Arm`
- Produces: `merge_voxel_cells(cells: Sequence[Sequence[int]]) -> list[VoxelBox]`, `occupied_volume(cells, pitch) -> float`, `boxes_volume(boxes, pitch) -> float`, `compute_staging_poses(piece_extents, table_bounds, exclusions, gap) -> dict[str, Pose]`, `check_reachability(config, poses) -> ReachabilityReport`

- [ ] **Step 1: Write failing voxel decomposition tests**

Test that an axis-aligned solid block becomes one box, an L-shaped set becomes two non-overlapping boxes, duplicate cells are rejected, non-integral or non-triplet cells are rejected, every box has positive size, and the sum of merged-box volumes equals unique occupied-cell volume.

- [ ] **Step 2: Run voxel tests and verify failure**

Run: `python -m pytest 06_interactive_scene/tests/test_scene_geometry.py -v`

Expected: FAIL because `scene_geometry` does not exist.

- [ ] **Step 3: Implement deterministic maximal-box decomposition**

Implement `VoxelBox(min_cell: tuple[int, int, int], size_cells: tuple[int, int, int])` and a deterministic greedy merge ordered by `(z, y, x)`: extend along `x`, then merge identical runs along `y`, then merge identical slabs along `z`. Validate input before merging and expose the two volume helpers for invariants.

- [ ] **Step 4: Run voxel tests**

Run: `python -m pytest 06_interactive_scene/tests/test_scene_geometry.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Write failing staging and reachability tests**

Test that all eight fragment AABBs are above the table, mutually separated by at least the configured `0.04 m` gap, outside robot-base and plate exclusion boxes, and inside table bounds. Test that `check_reachability` returns a named candidate approach for every fragment grasp center and the plate center, with every joint inside probe limits. Include a repository copy under a path containing spaces to exercise portable resolution.

- [ ] **Step 6: Run layout tests and verify failure**

Run: `python -m pytest 06_interactive_scene/tests/test_plan_layout.py -v`

Expected: FAIL because layout functions and final poses do not exist.

- [ ] **Step 7: Implement layout search and freeze accepted poses**

Implement deterministic table-grid staging and offline IK scoring using the existing kinematic model. Search robot-base yaw and candidate fragment/plate locations from config ranges, prefer maximum minimum joint-limit margin, and emit the selected robot, plate, and fragment poses into `scene.json` only when all required targets are reachable.

- [ ] **Step 8: Run all pure-Python tests**

Run: `python -m pytest 06_interactive_scene/tests -v`

Expected: all tests PASS and the reachability report names nine reachable targets.

- [ ] **Step 9: Commit geometry and layout planning**

```powershell
git add 06_interactive_scene/config/scene.json 06_interactive_scene/scripts/scene_geometry.py 06_interactive_scene/scripts/plan_layout.py 06_interactive_scene/tests/test_scene_geometry.py 06_interactive_scene/tests/test_plan_layout.py
git commit -m "feat: plan reachable fragment workspace"
```

### Task 3: Physics-enabled environment and fragment assets

**Files:**
- Create: `06_interactive_scene/scripts/scene_assets.py`
- Create: `06_interactive_scene/scripts/build_scene.py`
- Create: `06_interactive_scene/scripts/validate_scene.py`
- Create: `06_interactive_scene/tests/test_stage_manifest.py`

**Interfaces:**
- Consumes: `SceneConfig`, `merge_voxel_cells`, frozen scene poses
- Produces: `create_world(config, headless: bool) -> tuple[SimulationApp, World]`, `author_environment(stage, config) -> None`, `author_fragment(stage, config, name) -> FragmentHandle`, `build_stage(config, headless=True) -> SceneHandles`, `validate_manifest(handles) -> ValidationReport`

- [ ] **Step 1: Write the failing stage-manifest test**

Add a pure test for `expected_stage_manifest(config)` asserting one floor, one table, one plate, eight unique fragment rigid bodies, one robot root, and two camera paths. This pins names before Isaac APIs are imported.

- [ ] **Step 2: Run the manifest test and verify failure**

Run: `python -m pytest 06_interactive_scene/tests/test_stage_manifest.py -v`

Expected: FAIL because `scene_assets` and the manifest helper do not exist.

- [ ] **Step 3: Implement materials, support geometry, and fragment authoring**

Use lazy Isaac/OpenUSD imports so pure tests remain runnable outside `D:\i45`. Author NPZ points/faces as non-subdivided visual meshes, merged `UsdGeom.Cube` children as invisible collision shapes, `UsdPhysics.CollisionAPI` on collider children, and `UsdPhysics.RigidBodyAPI` plus `UsdPhysics.MassAPI` on each fragment root. Author static floor/table/plate colliders and shared physics materials.

- [ ] **Step 4: Implement the headless environment build and manifest report**

Create a fresh meter-scale stage, PhysX scene, gravity, lighting, support geometry, and all fragments. `validate_manifest` must report exact missing/extra prims and reject malformed or empty fragment inputs before saving.

- [ ] **Step 5: Run the first Isaac headless build**

Run: `D:\i45\python.exe 06_interactive_scene/scripts/validate_scene.py --config 06_interactive_scene/config/scene.json --manifest-only`

Expected: exit `0`, JSON report with eight fragments and no manifest errors.

- [ ] **Step 6: Run pure tests again**

Run: `python -m pytest 06_interactive_scene/tests -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit the physical environment**

```powershell
git add 06_interactive_scene/scripts/scene_assets.py 06_interactive_scene/scripts/build_scene.py 06_interactive_scene/scripts/validate_scene.py 06_interactive_scene/tests/test_stage_manifest.py
git commit -m "feat: build physics-enabled fragment scene"
```

### Task 4: FANUC/Robotiq composition and simple controls

**Files:**
- Create: `06_interactive_scene/scripts/scene_controls.py`
- Create: `06_interactive_scene/tests/test_control_math.py`
- Modify: `06_interactive_scene/scripts/build_scene.py`
- Modify: `06_interactive_scene/scripts/validate_scene.py`

**Interfaces:**
- Consumes: `SceneHandles`, configured expected DOF names, probe joint limits, `asm_kin.MIMIC_2F85`
- Produces: `compose_robot(stage, world, config) -> RobotHandle`, `ArmController.jog(joint_index: int, delta_rad: float) -> CommandResult`, `ArmController.home()`, `GripperController.command(open_fraction: float) -> CommandResult`, `SceneController.reset() -> None`

- [ ] **Step 1: Write failing control-math tests**

Test joint-index rejection, clamping at each probe limit, unchanged non-selected joints, `home()` exact targets, gripper fraction rejection outside `[0, 1]`, and mimic targets for open `0.0 rad` and closed `0.8 rad` using every configured gripper DOF name.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest 06_interactive_scene/tests/test_control_math.py -v`

Expected: FAIL because `scene_controls` does not exist.

- [ ] **Step 3: Implement pure target generation and controllers**

Separate pure functions `clamped_joint_target(...)` and `gripper_dof_targets(...)` from Isaac articulation calls. Controllers must return `CommandResult(accepted, clamped, message)` and name invalid/missing DOFs.

- [ ] **Step 4: Compose the gripper into the arm articulation**

Reference the arm at `/World/Robot`, reference the gripper under the flange mount, remove only the gripper's nested articulation-root schema and world/root anchoring joint, and author a fixed joint between the FANUC flange rigid body and Robotiq base link. Preserve finger rigid bodies, collision schemas, revolute joints, and drives. Fail with exact prim paths if the expected hierarchy differs.

- [ ] **Step 5: Run control tests**

Run: `python -m pytest 06_interactive_scene/tests/test_control_math.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Validate robot and gripper motion in Isaac Sim**

Run: `D:\i45\python.exe 06_interactive_scene/scripts/validate_scene.py --config 06_interactive_scene/config/scene.json --robot-motion`

Expected: exit `0`; report exactly six named arm DOFs, every expected gripper DOF, measurable motion after one joint jog, measurable finger-link change after open/close, and reset matching configured targets.

- [ ] **Step 7: Commit robot composition and controls**

```powershell
git add 06_interactive_scene/scripts/build_scene.py 06_interactive_scene/scripts/scene_controls.py 06_interactive_scene/scripts/validate_scene.py 06_interactive_scene/tests/test_control_math.py
git commit -m "feat: add interactive robot controls"
```

### Task 5: Cameras, robust snapshots, stable physics, and portable USD output

**Files:**
- Create: `06_interactive_scene/scripts/scene_cameras.py`
- Create: `06_interactive_scene/tests/test_camera_config.py`
- Modify: `06_interactive_scene/scripts/build_scene.py`
- Modify: `06_interactive_scene/scripts/validate_scene.py`
- Generate: `06_interactive_scene/generated/interactive_scene.usd`
- Generate: `06_interactive_scene/inspections/agent_camera.png`
- Generate: `06_interactive_scene/inspections/wrist_camera.png`

**Interfaces:**
- Consumes: `SceneConfig`, `RobotHandle`, `SceneHandles`
- Produces: `author_cameras(stage, config, robot) -> CameraHandles`, `capture_rgb(camera_path: str, output: Path, max_retries=12) -> CaptureResult`, `validate_stability(handles, seconds=3.0) -> ValidationReport`, `save_portable_stage(stage, output: Path) -> None`

- [ ] **Step 1: Write failing camera contract tests**

Test exactly two camera names, 640 by 480 resolution, valid clipping intervals, fixed-camera world parent, wrist-camera flange/gripper parent, nonzero look vector, bounded retry count, and output paths confined to `06_interactive_scene/`.

- [ ] **Step 2: Run camera tests and verify failure**

Run: `python -m pytest 06_interactive_scene/tests/test_camera_config.py -v`

Expected: FAIL because `scene_cameras` does not exist.

- [ ] **Step 3: Implement camera authoring and bounded RGB capture**

Author camera prims using configured transforms and create Replicator render products only during capture. Retry empty/wrong-shaped/non-`uint8` annotator frames for at most `12` rendered frames, then raise an error naming the camera and observed data shape. Detach annotators and destroy render products after capture.

- [ ] **Step 4: Implement stability and portable-save validation**

Step 360 physics frames, checking finite transforms, table penetration tolerance, workspace bounds, and linear/angular speed thresholds. Save to a temporary sibling path, reopen and validate expected prims and relative asset references, then atomically replace the final USD only on success.

- [ ] **Step 5: Run complete headless validation and produce artifacts**

Run: `D:\i45\python.exe 06_interactive_scene/scripts/validate_scene.py --config 06_interactive_scene/config/scene.json --full --save-stage --capture`

Expected: exit `0`; stable named fragments, successful save/reopen, and two nonempty 640 by 480 RGB PNGs.

- [ ] **Step 6: Visually inspect both camera images**

Use the image inspection tool on both PNGs. Confirm the agent view contains the robot, table, plate, and eight staged fragments; confirm the wrist view contains the gripper and nearby table workspace. Adjust only camera configuration and rerun Step 5 until both checks pass.

- [ ] **Step 7: Commit cameras and validated scene artifacts**

```powershell
git add 06_interactive_scene/config/scene.json 06_interactive_scene/scripts/build_scene.py 06_interactive_scene/scripts/scene_cameras.py 06_interactive_scene/scripts/validate_scene.py 06_interactive_scene/tests/test_camera_config.py 06_interactive_scene/generated/interactive_scene.usd 06_interactive_scene/inspections/agent_camera.png 06_interactive_scene/inspections/wrist_camera.png
git commit -m "feat: add scene cameras and validated USD"
```

### Task 6: GUI runtime, keyboard controls, and operator documentation

**Files:**
- Create: `06_interactive_scene/scripts/run_scene.py`
- Create: `06_interactive_scene/tests/test_key_bindings.py`
- Create: `06_interactive_scene/README.md`
- Modify: `06_interactive_scene/scripts/scene_controls.py`

**Interfaces:**
- Consumes: `build_stage`, camera capture API, `SceneController`
- Produces: `key_action(event) -> Optional[ControlAction]`, `KeyboardController.subscribe()`, `KeyboardController.close()`, documented GUI launch and validation commands

- [ ] **Step 1: Write failing key-binding tests**

Test six joint-selection keys, positive/negative jog actions, home, gripper open/close, reset, save, both-camera capture, ignored release/repeat events, and clean unsubscribe behavior. Assert no action binding collides with viewport navigation keys documented in the README.

- [ ] **Step 2: Run key-binding tests and verify failure**

Run: `python -m pytest 06_interactive_scene/tests/test_key_bindings.py -v`

Expected: FAIL because runtime bindings do not exist.

- [ ] **Step 3: Implement the GUI runtime and keyboard subscription**

Launch `SimulationApp({"headless": False})`, load or rebuild the scene, subscribe through `carb.input.acquire_input_interface().subscribe_to_keyboard_events`, dispatch only key-press events, step continuously while the app is running, and always unsubscribe and close the app in `finally`.

- [ ] **Step 4: Write operator documentation**

Document prerequisites, the exact `D:\i45\python.exe` build/validate/GUI commands, controls, expected first-launch delay, camera selection and snapshots, generated-file ownership, troubleshooting for VRAM/RAM and asset paths, and the explicit non-goals of this first version.

- [ ] **Step 5: Run all automated verification**

Run: `python -m pytest 06_interactive_scene/tests -v`

Expected: all pure tests PASS.

Run: `D:\i45\python.exe 06_interactive_scene/scripts/validate_scene.py --config 06_interactive_scene/config/scene.json --full --save-stage --capture`

Expected: exit `0` and a complete success report.

- [ ] **Step 6: Run the local GUI acceptance check**

Run: `D:\i45\python.exe 06_interactive_scene/scripts/run_scene.py --config 06_interactive_scene/config/scene.json`

Expected: Isaac Sim 4.5 opens the scene; joint jog, home, gripper, reset, save, and capture actions work; both cameras are selectable; `nvidia-smi` shows GPU memory below the available 8 GB budget; closing the window exits cleanly.

- [ ] **Step 7: Commit the inspectable first version**

```powershell
git add 06_interactive_scene/README.md 06_interactive_scene/scripts/run_scene.py 06_interactive_scene/scripts/scene_controls.py 06_interactive_scene/tests/test_key_bindings.py
git commit -m "feat: deliver interactive Isaac Sim scene"
```

### Task 7: Final regression and repository-boundary audit

**Files:**
- Modify only if verification exposes a defect: files under `06_interactive_scene/`

**Interfaces:**
- Consumes: all earlier task interfaces
- Produces: verified final repository state

- [ ] **Step 1: Verify legacy storyboard files are untouched**

Run: `git diff 5a105ec --name-only -- 01_crag_case 02_robot_assets 03_scripts 04_intermediate 05_outputs`

Expected: no output.

- [ ] **Step 2: Verify generated references remain portable**

Run the relocation test and inspect the saved USD for absolute references to the current repository or `D:\i45`.

Expected: relocation test PASS; no machine-specific asset references in the generated stage.

- [ ] **Step 3: Run the complete test suite and headless validation once more**

Run: `python -m pytest 06_interactive_scene/tests -v`

Run: `D:\i45\python.exe 06_interactive_scene/scripts/validate_scene.py --config 06_interactive_scene/config/scene.json --full --save-stage --capture`

Expected: both commands exit `0` with all checks passing.

- [ ] **Step 4: Inspect Git history metadata and working tree**

Run: `git log --format=fuller 5a105ec..HEAD` and `git status --short`.

Expected: commits use the existing configured identity, contain no Codex/co-author trailers, and the working tree is clean.

- [ ] **Step 5: Commit any verification-only corrections**

If Step 1 through Step 4 required changes, commit only the corrected `06_interactive_scene/` files with a concise defect-specific message. Otherwise, do not create an empty commit.
