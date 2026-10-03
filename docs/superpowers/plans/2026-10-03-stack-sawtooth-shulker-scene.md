# Stack Sawtooth Shulker Initial Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a deterministic single-right-arm Isaac Sim scene containing two lightweight complementary voxel fragments staged in a Y-directed line beside the existing red target.

**Architecture:** A new `tasks.stack_sawtooth_shulker` composition root uses the core `single_arm_right` workcell and authors analytical task-local voxel bodies directly into USD. Pure modules own geometry, configuration, and expected manifests; Isaac-dependent modules own USD authoring, scene composition, view-only runtime, and stage validation.

**Tech Stack:** Python 3.10, NumPy 1.26, NVIDIA Isaac Sim 4.5, USD/PhysX APIs, pytest 8.4

**Spec:** `docs/superpowers/specs/2026-10-03-stack-sawtooth-shulker-scene-design.md`

## Global Constraints

- Use `tasks/stack_sawtooth_shulker`; do not add a singular top-level `task/` directory.
- Select the existing `single_arm_right` preset; do not redefine the table, robot, cameras, lighting, red tape, or global physics.
- Use `u = 0.08 m`, an independent `0.55 x 0.55 m` target, `0.002 m` total collider clearance, and `20 kg/m^3` nominal density.
- Fragment A is blue with 14 cells; Fragment B is green with the complementary 13 cells; neither uses red.
- Initial roots are A `(0.48, -0.185, 0.75)` and B `(0.48, 0.185, 0.75)`, both with identity orientation.
- Do not add robot motion commands, keyboard controls, planning, recording, success evaluation, randomization, or a policy interface.
- Require explicit external output paths for USD export; never generate runtime artifacts inside the repository.
- Keep Isaac imports lazy so pure tests do not launch the simulator.
- Before an Isaac smoke run, inspect GPU usage and constrain the process to at most one GPU; never exceed the account-wide two-GPU limit.
- Do not create Git commits unless the user explicitly requests them.

## Review Focus

- A nonsquare core target or one too small for the assembled three-unit footprint must fail task configuration validation; pin this in Task 1 tests.
- Duplicate, overlapping, missing, or out-of-range cells must fail before Isaac starts; pin this in Task 1 tests.
- Negative, non-finite, or `>= u` collider clearance and nonpositive density must fail clearly; zero clearance remains a valid exact-contact configuration; pin this in Task 1 tests.
- Initial placements must be revalidated against target, table, one another, and the current physical-base clearance when core geometry changes; pin this in Task 1 tests.
- A failed build or invalid repository-local output path must not export a USD and must still close an initialized app; pin this in Task 4 tests.

---

## File Map

**Create:**

- `tasks/stack_sawtooth_shulker/__init__.py` — public task configuration and builder exports.
- `tasks/stack_sawtooth_shulker/__main__.py` — `build` and `launch` CLI composition.
- `tasks/stack_sawtooth_shulker/config.py` — immutable task, fragment, and pose configuration.
- `tasks/stack_sawtooth_shulker/geometry.py` — analytical voxel sets, local mappings, bounds, mass, and placement validation.
- `tasks/stack_sawtooth_shulker/assets.py` — task materials, metadata, voxel visuals/colliders, and rigid bodies.
- `tasks/stack_sawtooth_shulker/builder.py` — core workcell plus task-object composition.
- `tasks/stack_sawtooth_shulker/runtime.py` — view-only simulation loop.
- `tasks/stack_sawtooth_shulker/validation.py` — expected manifest and live USD validation.
- `tasks/stack_sawtooth_shulker/README.md` — initial-scene purpose, commands, and explicit exclusions.
- `tasks/stack_sawtooth_shulker/tests/__init__.py` — test package marker.
- `tasks/stack_sawtooth_shulker/tests/test_geometry.py` — voxel and numerical geometry contracts.
- `tasks/stack_sawtooth_shulker/tests/test_config.py` — task/core relationship and invalid-input contracts.
- `tasks/stack_sawtooth_shulker/tests/test_assets.py` — pure authoring-plan and manifest contracts.
- `tasks/stack_sawtooth_shulker/tests/test_builder.py` — composition contracts without launching Isaac.
- `tasks/stack_sawtooth_shulker/tests/test_cli.py` — command and output-boundary contracts.

**Modify:**

- `docs/tasks.md` — list the new scene task and its current scope.
- `README.md` — add the task to the repository structure and task overview.

### Task 1: Analytical Geometry and Configuration Contracts

**Files:**
- Create: `tasks/stack_sawtooth_shulker/__init__.py`
- Create: `tasks/stack_sawtooth_shulker/config.py`
- Create: `tasks/stack_sawtooth_shulker/geometry.py`
- Create: `tasks/stack_sawtooth_shulker/tests/__init__.py`
- Create: `tasks/stack_sawtooth_shulker/tests/test_geometry.py`
- Create: `tasks/stack_sawtooth_shulker/tests/test_config.py`

**Interfaces:**
- Consumes: `core.workcell.config.WorkcellConfig` and `load_workcell_preset("single_arm_right")`.
- Produces: `Pose`, `FragmentConfig`, `StackSawtoothConfig`, `load_task_config()`, `validate_task_config(config)`, `COMPLETE_CELLS`, `FRAGMENT_A_CELLS`, `FRAGMENT_B_CELLS`, `local_cells(cells)`, `cell_center(cell, unit_size)`, `fragment_bounds(fragment, unit_size)`, and `fragment_mass(fragment, unit_size, density)`.

- [ ] **Step 1: Write failing exact-cell and local-frame tests**

Add tests asserting A's complete bottom layer plus five middle cells, B's four middle edge cells plus complete top layer, counts `(14, 13)`, disjointness, complete 27-cell union, A local Z range `{0, 1}`, B local Z range `{0, 1}`, and B canonical Z offset `1`.

- [ ] **Step 2: Run the geometry tests and verify import failure**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tasks/stack_sawtooth_shulker/tests/test_geometry.py -q -p no:cacheprovider`

Expected: FAIL because the new package or interfaces do not exist.

- [ ] **Step 3: Implement canonical and local geometry**

In `geometry.py`, define `Cell = tuple[int, int, int]`, sorted immutable cell constants, and the exact signatures from the Interfaces block. Center local cell geometry in X/Y using `(coordinate - 1) * u`; place the lowest local voxel layer on Z=0 using voxel centers at `(z + 0.5) * u`.

- [ ] **Step 4: Run the geometry tests**

Run the Task 1 Step 2 command.

Expected: PASS.

- [ ] **Step 5: Write failing valid-configuration tests**

Assert `load_task_config()` selects only `right`, derives the core target/table geometry, assigns A blue and B green, uses exact initial and goal poses, computes bounds `[-0.12, 0.12]` in X/Y, computes masses `0.14336` and `0.13312` kg, preserves 85 mm fragment/target/base clearance and 130 mm inter-fragment clearance, and returns identical values on repeated loads.

- [ ] **Step 6: Write failing invalid-configuration tests for every Review Focus input owned by Task 1**

Use `dataclasses.replace` to assert clear `ValueError`s for nonsquare or undersized targets, duplicate/overlapping/missing/out-of-range cells, negative or non-finite clearance, clearance `>= u`, nonpositive density, initial overlap, red-region overlap, table overflow, and insufficient robot-base clearance. Assert separately that zero clearance is accepted; permit finite clearance satisfying `0 <= clearance < u`.

- [ ] **Step 7: Run configuration tests and verify failure**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tasks/stack_sawtooth_shulker/tests/test_config.py -q -p no:cacheprovider`

Expected: FAIL because configuration types and validation are not implemented.

- [ ] **Step 8: Implement immutable configuration and validation**

In `config.py`, define frozen `Pose`, `FragmentConfig`, and `StackSawtoothConfig` dataclasses. Implement `load_task_config(workcell: WorkcellConfig | None = None) -> StackSawtoothConfig` and `validate_task_config(config: StackSawtoothConfig) -> None` with the exact constants and spatial checks from the spec. Represent the observed right-base near edge as a named task configuration value `0.685 m`, documenting that it comes from the checked-in robot asset probe and is intentionally revalidated rather than inferred from the conservative staging exclusion.

- [ ] **Step 9: Export the stable public configuration API**

In `__init__.py`, export only `FragmentConfig`, `Pose`, `StackSawtoothConfig`, and `load_task_config`; defer builder exports until Task 3.

- [ ] **Step 10: Run all Task 1 tests**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tasks/stack_sawtooth_shulker/tests/test_geometry.py tasks/stack_sawtooth_shulker/tests/test_config.py -q -p no:cacheprovider`

Expected: PASS.

- [ ] **Step 11: Review checkpoint**

Run `git diff --check` and inspect only Task 1 changes. Do not commit.

### Task 2: Task-Owned USD Assets and Expected Manifest

**Files:**
- Create: `tasks/stack_sawtooth_shulker/assets.py`
- Create: `tasks/stack_sawtooth_shulker/validation.py`
- Create: `tasks/stack_sawtooth_shulker/tests/test_assets.py`

**Interfaces:**
- Consumes: Task 1 `StackSawtoothConfig`, `FragmentConfig`, `cell_center`, `fragment_mass`; core `author_visual_material`, `bind_visual`, `set_transform`, and `bind_physics`.
- Produces: `FragmentHandle`, `TaskSceneManifest`, `expected_task_manifest(workcell, config)`, `author_target_metadata(stage, config)`, and `author_fragment(stage, workcell, config, fragment)`.

- [ ] **Step 1: Write failing expected-manifest tests**

Assert task root `/World/Task`, metadata path `/World/Task/TargetMetadata`, fragment roots `/World/Task/Fragments/FragmentA` and `FragmentB`, sorted visual/collider paths named `Voxel_000` onward, counts 14 and 13, and a nested core manifest containing only the right robot plus agent/right-wrist cameras.

- [ ] **Step 2: Write failing pure authoring-value tests**

Assert each sorted local cell maps to the expected cube center, visual side is `0.08`, collider side is `0.078`, material paths are task-namespaced, masses equal Task 1 values, and metadata values contain task name, unit size, target size, and both goal poses.

- [ ] **Step 3: Run asset tests and verify failure**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tasks/stack_sawtooth_shulker/tests/test_assets.py -q -p no:cacheprovider`

Expected: FAIL because asset and manifest interfaces do not exist.

- [ ] **Step 4: Implement pure manifest construction in `validation.py`**

Define frozen `FragmentManifest` and `TaskSceneManifest` dataclasses and `expected_task_manifest(workcell: WorkcellConfig, config: StackSawtoothConfig) -> TaskSceneManifest`. Generate paths from sorted fragment cells and reject duplicate paths.

- [ ] **Step 5: Implement USD target metadata authoring**

Implement `author_target_metadata(stage: Any, config: StackSawtoothConfig) -> str` under `/World/Task/TargetMetadata`, using USD custom attributes for task name, unit size, target size, and A/B goal position/orientation. Keep `pxr` imports inside the function.

- [ ] **Step 6: Implement procedural fragment authoring**

Define frozen `FragmentHandle(name, root_path, visual_paths, collider_paths, mass)`. Implement `author_fragment(stage: Any, workcell: WorkcellConfig, config: StackSawtoothConfig, fragment: FragmentConfig) -> FragmentHandle`: create one exact visual cube and one inset collider cube per sorted local cell, bind task visual and core contact materials, apply one rigid body at the root, set `startsAsleep`, and author the explicit analytical mass.

- [ ] **Step 7: Run Task 2 tests**

Run the Task 2 Step 3 command.

Expected: PASS without importing or launching Isaac.

- [ ] **Step 8: Review checkpoint**

Run `git diff --check` and inspect Task 2 paths, counts, types, and lazy imports. Do not commit.

### Task 3: Scene Composition and Live Validation

**Files:**
- Create: `tasks/stack_sawtooth_shulker/builder.py`
- Create: `tasks/stack_sawtooth_shulker/tests/test_builder.py`
- Modify: `tasks/stack_sawtooth_shulker/validation.py`
- Modify: `tasks/stack_sawtooth_shulker/__init__.py`

**Interfaces:**
- Consumes: Task 1 configuration, Task 2 authoring/manifest APIs, `core.scene.builder.build_workcell`, and `core.scene.validation.validate_workcell_stage`.
- Produces: `StackSawtoothHandles`, `build_stack_sawtooth_scene(headless=True, stream=False)`, `SceneValidationReport`, and `validate_stack_sawtooth_scene(handles)`.

- [ ] **Step 1: Write failing builder composition tests with monkeypatched core/authoring functions**

Assert `build_stack_sawtooth_scene()` loads `single_arm_right`, calls `build_workcell` before task authoring, creates metadata once, authors A then B deterministically, performs the required post-authoring reset/settle sequence, and returns handles exposing `app`, `world`, `stage`, and `cameras` through the core handles.

- [ ] **Step 2: Write failing validation-report tests with fake stages**

Assert missing required task paths are reported, unexpected fragment-root children are reported, expected fragment/visual/collider counts are preserved, and the report is successful only when both core and task manifests are complete. Add a live-only validator branch for rigid-body API, starts-asleep, mass, and transform checks that can be exercised in Task 5.

- [ ] **Step 3: Run builder tests and verify failure**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tasks/stack_sawtooth_shulker/tests/test_builder.py -q -p no:cacheprovider`

Expected: FAIL because builder and live validation are not implemented.

- [ ] **Step 4: Implement scene handles and composition**

Define `StackSawtoothHandles(config, workcell, manifest, fragments)` and proxy properties for `app`, `world`, `stage`, and `cameras`. Implement `build_stack_sawtooth_scene(headless: bool = True, stream: bool = False) -> StackSawtoothHandles`; close the core app before re-raising any task-authoring or post-reset failure.

- [ ] **Step 5: Implement structural and live USD validation**

Define frozen `SceneValidationReport(ok, missing_paths, extra_fragment_paths, contract_failures)`. Implement `validate_stack_sawtooth_scene(handles) -> SceneValidationReport`, combining core validation with manifest presence, exact task-root children, `UsdPhysics.RigidBodyAPI`, `startsAsleep`, authored mass, and initial world-transform checks.

- [ ] **Step 6: Export builder interfaces**

Update task `__init__.py` to export `StackSawtoothHandles` and `build_stack_sawtooth_scene` without causing Isaac startup at import time.

- [ ] **Step 7: Run Task 3 and preceding task tests**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tasks/stack_sawtooth_shulker/tests -q -p no:cacheprovider`

Expected: PASS without starting `SimulationApp`.

- [ ] **Step 8: Review checkpoint**

Run `git diff --check` and inspect failure cleanup and dependency direction. Do not commit.

### Task 4: View-Only Runtime and CLI

**Files:**
- Create: `tasks/stack_sawtooth_shulker/runtime.py`
- Create: `tasks/stack_sawtooth_shulker/__main__.py`
- Create: `tasks/stack_sawtooth_shulker/tests/test_cli.py`

**Interfaces:**
- Consumes: Task 3 builder and validator.
- Produces: `run_viewer(smoke_frames=0)`, `require_external_output(path)`, `build_parser()`, and `main(argv=None)`.

- [ ] **Step 1: Write failing CLI shape and output-boundary tests**

Assert commands are exactly `build` and `launch`; `build` requires `--output-usd`; `launch` exposes no motion or recording flags; paths equal to or below the repository root are rejected; external paths resolve successfully.

- [ ] **Step 2: Write failing build lifecycle tests**

Monkeypatch scene construction and validation to assert: valid build creates only the output parent and exports once; invalid validation exports nothing; export exceptions propagate; and every initialized app closes exactly once on success and failure.

- [ ] **Step 3: Write failing view-only runtime tests**

With fake app/world handles, assert the viewer only calls `world.step(render=True)`, exits on app shutdown or hidden `smoke_frames`, installs no input subscription, and closes the app in `finally`.

- [ ] **Step 4: Run CLI tests and verify failure**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tasks/stack_sawtooth_shulker/tests/test_cli.py -q -p no:cacheprovider`

Expected: FAIL because runtime and CLI do not exist.

- [ ] **Step 5: Implement the view-only runtime**

Implement `run_viewer(smoke_frames: int = 0) -> int` using `build_stack_sawtooth_scene(headless=False)`. Reject negative `smoke_frames`; perform no task action beyond rendered stepping and guaranteed close.

- [ ] **Step 6: Implement CLI parsing and dispatch**

Implement `require_external_output(path: Path, repo_root: Path = REPO_ROOT) -> Path`, `build_parser() -> argparse.ArgumentParser`, and `main(argv=None) -> int`. `build` validates before export and closes in `finally`; `launch` delegates to the viewer. Keep `--smoke-frames` suppressed and test-only.

- [ ] **Step 7: Run all task tests**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tasks/stack_sawtooth_shulker/tests -q -p no:cacheprovider`

Expected: PASS.

- [ ] **Step 8: Run the complete pure/import-safe repository suite**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest -q -p no:cacheprovider`

Expected: all tests pass, with only dataset-dependent tests skipped when their environment variables are absent.

- [ ] **Step 9: Review checkpoint**

Run `git diff --check` and verify no CLI or runtime API enables motion, recording, or repository-local output. Do not commit.

### Task 5: Documentation and Actual Isaac Scene Verification

**Files:**
- Create: `tasks/stack_sawtooth_shulker/README.md`
- Modify: `docs/tasks.md`
- Modify: `README.md`
- Verify: complete task package and exported external USD

**Interfaces:**
- Consumes: Tasks 1–4 public CLI and validation behavior.
- Produces: documented user entry points and evidence that the actual Isaac scene constructs successfully.

- [ ] **Step 1: Document the initial scene and exclusions**

Document exact geometry, colors, poses, `u = 0.08 m`, Y-directed staging, `build`/`launch` commands, external-output policy, and the absence of motion, recording, randomization, and success evaluation. Add concise links/descriptions to `docs/tasks.md` and the root README structure/task sections.

- [ ] **Step 2: Run documentation and repository-shape tests**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest core/tests/test_repository_shape.py tasks/stack_sawtooth_shulker/tests/test_cli.py -q -p no:cacheprovider`

Expected: PASS.

- [ ] **Step 3: Inspect GPU availability and account-owned usage**

Run `nvidia-smi` and correlate listed compute PIDs with the current user. If this account already uses two GPUs, stop before the smoke run and report the resource block. Otherwise select one appropriate GPU without modifying any process.

- [ ] **Step 4: Build and validate the real scene on one explicitly constrained GPU**

Create a disposable directory under `/local_data/yz11445/scratch/` with `mktemp -d`, then run the task's `build --output-usd` command with `CUDA_VISIBLE_DEVICES` set to the one selected GPU. Expected: exit 0, live scene validation succeeds, and a nonempty USDA file exists outside the repository. Do not capture images or state.

- [ ] **Step 5: Inspect the exported stage contract without changing it**

Confirm the exported stage contains one right robot, agent and right-wrist cameras, shared red tape, Fragment A/B roots, 27 total visual voxels, 27 total collider voxels, correct initial transforms, and no left robot or task-level controller/recording prims.

- [ ] **Step 6: Run final verification**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest -q -p no:cacheprovider`

Expected: all import-safe tests pass; note any environment-dependent skip separately.

- [ ] **Step 7: Final diff and workspace audit**

Run `git diff --check`, `git status --short`, and scan tracked changes for generated `.usd`, `.usda`, `.usdc`, images, logs, datasets, caches, or hard-coded scratch paths. Expected: only source, tests, and documentation are changed; no commit is created.
