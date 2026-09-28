# Dual-Arm Interactive Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the original demo's two-arm and fragment arrangement in the physics-enabled interactive Isaac Sim 4.5 scene, with independent controls and one wrist camera per arm.

**Architecture:** Replace singular robot and wrist-camera contracts with deterministic named collections keyed by `left` and `right`. Reuse the proven FANUC/Robotiq composition for each arm, keep shared fragments and environment unchanged, and route controls through an active-robot selector. The demo storyboard is a read-only source of exact configuration values; runtime remains independent of it.

**Tech Stack:** Python 3.10, pytest, Isaac Sim 4.5, OpenUSD, PhysX, NumPy

**Spec:** `docs/superpowers/specs/2026-09-28-dual-arm-interactive-scene-design.md`

## Global Constraints

- Work directly on `main` as explicitly requested; preserve unrelated untracked files.
- Modify scene implementation only under `06_interactive_scene/`, plus this plan and the approved spec.
- Do not modify `01_crag_case/`, `02_robot_assets/`, `03_scripts/`, `04_intermediate/`, or `05_outputs/`.
- Isaac Sim target is exactly 4.5 with the installed interpreter at `D:\i45\python.exe`.
- Use the exact arm bases, agent camera, and first-frame fragment poses from `04_intermediate/choreo.json`.
- Preserve merged occupied-voxel collision volume; do not add stabilizing proxy colliders.
- Keep repository asset references relative in the saved USD.
- Commits use the repository's configured author identity and contain no generated co-author or agent attribution.

## Review Focus

- A control sent after `Tab` must affect only the newly active articulation; Task 2 tests active-arm routing and independent motion.
- Two identical referenced assets must not retain colliding absolute relationship targets; Task 2 validates distinct gripper joint targets and both articulations in Isaac.
- Reset after both robot and fragment motion must clear every articulation and rigid-body velocity; Task 2 adds a dual-reset integration check.
- Camera names, output keys, and prim parents must stay unambiguous across three views; Task 3 tests all mappings and rejects mismatched wrist parents.
- Demo fragment poses may be exact yet physically unstable; Task 4 validates three seconds of stability without altering occupied-voxel collision geometry.

---

### Task 1: Dual-arm configuration and exact demo layout

**Files:**
- Modify: `06_interactive_scene/config/scene.json`
- Modify: `06_interactive_scene/scripts/scene_config.py`
- Modify: `06_interactive_scene/scripts/plan_layout.py`
- Modify: `06_interactive_scene/tests/test_scene_config.py`
- Modify: `06_interactive_scene/tests/test_plan_layout.py`
- Modify: `06_interactive_scene/tests/test_camera_config.py`

**Interfaces:**
- Consumes: storyboard constants recorded in the approved spec; existing `SceneConfig.load(path)` and reachability helpers.
- Produces: `SceneConfig.robot_names -> tuple[str, ...]`, `SceneConfig.robot_spec(name: str) -> Mapping[str, Any]`, a `robots` mapping ordered `left`, `right`, demo-exact fragment poses, and three output/camera keys.

- [ ] **Step 1: Write failing configuration tests**

Add tests named `test_dual_robot_contract_is_left_then_right`, `test_demo_arm_and_fragment_transforms_are_exact`, `test_three_camera_output_paths_are_portable`, and `test_rejects_missing_or_extra_robot_names`. Assert the spec's exact arm transforms, exact eight first-frame poses from `04_intermediate/choreo.json`, agent camera values, and output keys `agent_image`, `left_wrist_image`, `right_wrist_image`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest -p no:cacheprovider 06_interactive_scene/tests/test_scene_config.py 06_interactive_scene/tests/test_camera_config.py -q`

Expected: FAIL because the current schema exposes one `robot`, one `wrist` camera, and one wrist output.

- [ ] **Step 3: Implement the dual configuration contract**

Add `robot_names` and `robot_spec(name)` to `SceneConfig`; validate the exact ordered names `("left", "right")`; migrate `scene.json` to `robots`, `agent`, `left_wrist`, and `right_wrist`; copy all demo-exact transforms into the config; replace the ambiguous `wrist_image` path with two explicit outputs.

- [ ] **Step 4: Write failing dual-base reachability tests**

Add `test_reachability_reports_an_arm_for_every_demo_fragment` and `test_plate_center_is_reachable_by_both_arms`. The report must carry per-robot results and at least one reachable arm for each fragment.

- [ ] **Step 5: Run reachability tests and verify RED**

Run: `python -m pytest -p no:cacheprovider 06_interactive_scene/tests/test_plan_layout.py -q`

Expected: FAIL because reachability currently evaluates a single robot base.

- [ ] **Step 6: Generalize offline reachability**

Update the reachability result types and `check_reachability(config, poses)` to solve each target against both configured robot bases without importing process-global legacy modules. Freeze results under robot names while leaving runtime independent of the storyboard.

- [ ] **Step 7: Run Task 1 tests**

Run: `python -m pytest -p no:cacheprovider 06_interactive_scene/tests/test_scene_config.py 06_interactive_scene/tests/test_camera_config.py 06_interactive_scene/tests/test_plan_layout.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add 06_interactive_scene/config/scene.json 06_interactive_scene/scripts/scene_config.py 06_interactive_scene/scripts/plan_layout.py 06_interactive_scene/tests/test_scene_config.py 06_interactive_scene/tests/test_camera_config.py 06_interactive_scene/tests/test_plan_layout.py
git commit -m "feat: define dual-arm demo layout"
```

### Task 2: Two physics articulations and active-arm controls

**Files:**
- Modify: `06_interactive_scene/scripts/scene_controls.py`
- Modify: `06_interactive_scene/scripts/build_scene.py`
- Modify: `06_interactive_scene/scripts/validate_scene.py`
- Modify: `06_interactive_scene/tests/test_control_math.py`
- Modify: `06_interactive_scene/tests/test_key_bindings.py`
- Create: `06_interactive_scene/tests/test_dual_robot_contract.py`

**Interfaces:**
- Consumes: `SceneConfig.robot_names`, `SceneConfig.robot_spec(name)`, existing FANUC/Robotiq asset paths, and `RobotHandle` behavior.
- Produces: `compose_robot(stage, world, config, name) -> RobotHandle`, `SceneHandles.robots: Mapping[str, RobotHandle]`, `SceneController.active_robot_name`, `SceneController.select_robot(name)`, and dual reset semantics.

- [ ] **Step 1: Write failing pure control and handle tests**

Add tests named `test_tab_toggles_left_and_right`, `test_joint_and_gripper_actions_route_to_active_robot`, `test_reset_targets_both_robot_specs`, and `test_robot_prim_namespaces_are_disjoint`. Use real routing/value helpers, with lightweight command sinks only where Isaac objects cannot exist in ordinary Python.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest -p no:cacheprovider 06_interactive_scene/tests/test_control_math.py 06_interactive_scene/tests/test_key_bindings.py 06_interactive_scene/tests/test_dual_robot_contract.py -q`

Expected: FAIL because scene handles, controllers, and bindings are singular.

- [ ] **Step 3: Generalize robot composition and scene handles**

Change `compose_robot` to accept a robot name/spec and author disjoint arm, gripper, fixed-joint, and remapped relationship paths. Change `SceneHandles.robot` to ordered `SceneHandles.robots`; build and initialize both articulations before creating the controller.

- [ ] **Step 4: Implement active-arm command routing and dual reset**

Add `Tab` as `toggle_robot`; keep joint selection per active robot; route jog/home/open/close only to the active handle; reset both arms and grippers plus all fragments. Every console message identifies `left` or `right`.

- [ ] **Step 5: Extend Isaac robot-motion validation**

Update `validate_robot_motion(handles)` to jog each arm independently, verify the other arm remains within tolerance, observe both grippers, impose motion on a fragment, reset, and report both articulation states and cleared fragment velocity.

- [ ] **Step 6: Run Task 2 pure tests**

Run: `python -m pytest -p no:cacheprovider 06_interactive_scene/tests/test_control_math.py 06_interactive_scene/tests/test_key_bindings.py 06_interactive_scene/tests/test_dual_robot_contract.py -q`

Expected: PASS.

- [ ] **Step 7: Run Isaac dual-motion validation**

Run: `D:\i45\python.exe 06_interactive_scene/scripts/validate_scene.py --config 06_interactive_scene/config/scene.json --robot-motion`

Expected: exit `0`; two six-DOF arms and two eight-DOF grippers reported; each arm moves independently; both reset within tolerance; fragment velocity is cleared.

- [ ] **Step 8: Commit**

```powershell
git add 06_interactive_scene/scripts/scene_controls.py 06_interactive_scene/scripts/build_scene.py 06_interactive_scene/scripts/validate_scene.py 06_interactive_scene/tests/test_control_math.py 06_interactive_scene/tests/test_key_bindings.py 06_interactive_scene/tests/test_dual_robot_contract.py
git commit -m "feat: add independently controlled dual arms"
```

### Task 3: Three-camera runtime and dual-arm manifest

**Files:**
- Modify: `06_interactive_scene/scripts/scene_assets.py`
- Modify: `06_interactive_scene/scripts/scene_cameras.py`
- Modify: `06_interactive_scene/scripts/run_scene.py`
- Modify: `06_interactive_scene/scripts/validate_scene.py`
- Modify: `06_interactive_scene/tests/test_camera_config.py`
- Modify: `06_interactive_scene/tests/test_key_bindings.py`
- Modify: `06_interactive_scene/tests/test_stage_manifest.py`
- Modify: `06_interactive_scene/tests/test_validation_contracts.py`

**Interfaces:**
- Consumes: `SceneHandles.robots`, three camera specs, and explicit output keys.
- Produces: `CameraHandles.agent_path`, `CameraHandles.left_wrist_path`, `CameraHandles.right_wrist_path`; `author_cameras(stage, config, robots)`; manifest tuples for two robots and three cameras; `F7/F8/F9` selection and three-image capture.

- [ ] **Step 1: Write failing camera, manifest, and binding tests**

Add tests named `test_camera_contract_has_agent_and_two_wrists`, `test_each_wrist_parent_matches_named_flange`, `test_manifest_requires_two_robot_roots_and_three_cameras`, `test_f7_f8_f9_select_distinct_views`, and `test_capture_mapping_has_three_unambiguous_outputs`.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest -p no:cacheprovider 06_interactive_scene/tests/test_camera_config.py 06_interactive_scene/tests/test_stage_manifest.py 06_interactive_scene/tests/test_key_bindings.py 06_interactive_scene/tests/test_validation_contracts.py -q`

Expected: FAIL because camera handles, manifest, viewport selection, and capture loops are singular.

- [ ] **Step 3: Implement three-camera authoring and output validation**

Generalize `camera_specs`, `CameraHandles`, `author_cameras`, and `validated_scene_output`. Require each wrist parent to equal the matching `RobotHandle.flange_path`; author the demo-framed agent camera and both wrist cameras.

- [ ] **Step 4: Implement the dual manifest and runtime mappings**

Replace `StageManifest.robot_path` with ordered `robot_paths`; validate both roots; bind `F7`, `F8`, and `F9`; capture all three images on `I`; keep `P` portable save behavior unchanged.

- [ ] **Step 5: Run Task 3 tests**

Run: `python -m pytest -p no:cacheprovider 06_interactive_scene/tests/test_camera_config.py 06_interactive_scene/tests/test_stage_manifest.py 06_interactive_scene/tests/test_key_bindings.py 06_interactive_scene/tests/test_validation_contracts.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add 06_interactive_scene/scripts/scene_assets.py 06_interactive_scene/scripts/scene_cameras.py 06_interactive_scene/scripts/run_scene.py 06_interactive_scene/scripts/validate_scene.py 06_interactive_scene/tests/test_camera_config.py 06_interactive_scene/tests/test_key_bindings.py 06_interactive_scene/tests/test_stage_manifest.py 06_interactive_scene/tests/test_validation_contracts.py
git commit -m "feat: add dual wrist camera workflow"
```

### Task 4: Portable artifacts, documentation, and full acceptance

**Files:**
- Modify: `06_interactive_scene/README.md`
- Modify: `06_interactive_scene/generated/interactive_scene.usd`
- Modify: `06_interactive_scene/inspections/agent_camera.png`
- Delete: `06_interactive_scene/inspections/wrist_camera.png`
- Create: `06_interactive_scene/inspections/left_wrist_camera.png`
- Create: `06_interactive_scene/inspections/right_wrist_camera.png`

**Interfaces:**
- Consumes: completed dual-arm build, runtime, camera, validation, and output contracts.
- Produces: a documented GUI entrypoint, portable two-arm USD, three inspected PNGs, and final acceptance evidence.

- [ ] **Step 1: Update operator documentation**

Document the restored demo arrangement, `Tab` active-arm selection, active-arm command semantics, `F7/F8/F9`, three capture outputs, reset behavior, exact launch/validation commands, and deferred coordinated dual-arm planning.

- [ ] **Step 2: Run the complete pure-Python suite**

Run: `python -m pytest -p no:cacheprovider 06_interactive_scene/tests -q`

Expected: all tests PASS with no errors or warnings from project code.

- [ ] **Step 3: Generate and validate final artifacts**

Run: `D:\i45\python.exe 06_interactive_scene/scripts/validate_scene.py --config 06_interactive_scene/config/scene.json --full --save-stage --capture`

Expected: exit `0`; exact two-arm/three-camera manifest; exact occupied-voxel colliders; demo lineup stable for 360 frames; portable reopen has no missing prims; three non-empty 640 by 480 captures.

- [ ] **Step 4: Inspect all three camera images**

Verify the agent image shows both robots, the plate, and all eight lined-up fragments; verify each wrist image includes its corresponding gripper and reachable workspace without being blocked by robot geometry.

- [ ] **Step 5: Run visible GUI smoke validation**

Run: `D:\i45\python.exe 06_interactive_scene/scripts/run_scene.py --config 06_interactive_scene/config/scene.json --smoke-frames 180`

Expected: ready message names `left` as active; agent camera selected; 180 frames render; clean exit; GPU memory remains within the available 8 GB budget.

- [ ] **Step 6: Audit scope and portability**

Run `git diff 084be50 --name-only -- 01_crag_case 02_robot_assets 03_scripts 04_intermediate 05_outputs`, scan the generated USD for drive-letter paths, run `git diff --check`, and verify commit messages contain no co-author or agent attribution.

Expected: legacy diff and absolute-path scan are empty; formatting and attribution audits pass.

- [ ] **Step 7: Commit the acceptance-ready implementation**

```powershell
git add 06_interactive_scene
git commit -m "feat: deliver dual-arm interactive scene"
```

- [ ] **Step 8: Request final independent review**

Review commits after `084be50` against the approved spec, with emphasis on dual-articulation independence, reset correctness, three-camera parentage, exact demo transforms, and absence of hidden collision geometry. Resolve every Critical or Important finding in follow-up commits and rerun affected validations.
