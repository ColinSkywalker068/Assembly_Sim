# Assembly_Sim Repository Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `Assembly_Sim` into a reusable `core/` workcell platform and the two task packages `tasks/demo_render_source/` and `tasks/dual_arm_breakingbad/`, with all generated data and experiment artifacts stored outside the repository.

**Architecture:** Named core workcell presets own the invariant environment, robot assets, robot placements, cameras, controls, and physics. Each task is an ordinary Python package that composes one preset and owns only its input adapters, object logic, task behavior, and external outputs. Imports point from tasks to core only; no plugin framework or numbered-directory compatibility layer is introduced.

**Tech Stack:** Python 3.10, Isaac Sim 4.5 Python, USD/PhysX APIs, NumPy, Trimesh, SciPy, Pillow, pytest, JSON configuration, Git LFS.

**Spec:** `docs/superpowers/specs/2026-10-02-repository-architecture-design.md`

## Global Constraints

- The only tracked top-level source directories are `core/`, `tasks/`, and `docs/`; root metadata files remain allowed.
- Raw inputs remain under `/local_data/yz11445/datasets/raw/` and are never modified in place.
- Processed assemblies live under `/local_data/yz11445/datasets/processed/`; generated stages, captures, logs, and videos live under `/local_data/yz11445/experiments/`.
- No source or checked-in configuration embeds a user-specific absolute path.
- `core` imports no task package; tasks may import `core`; tasks never import one another.
- The `dual_arm`, `single_arm_left`, and `single_arm_right` presets share one environment and exact canonical arm placements.
- The current worktable, grid pad, robot models, robot poses, cameras, physics, controls, and reset behavior remain visually and behaviorally stable.
- Task configuration cannot override core-owned table, pad, robot base, camera mount, gravity, physics-material, or lighting fields.
- Commands do not write generated artifacts into the repository by default.
- Isaac-dependent imports remain lazy so pure configuration and geometry tests do not launch Isaac Sim.
- Before any GPU validation, inspect `nvidia-smi`, select one GPU explicitly with `CUDA_VISIBLE_DEVICES`, and never use more than two GPUs.
- Preserve the repository's configured Git identity and never add automated-assistance attribution.
- Commit steps below are executed only if the user explicitly authorizes commits for this refactor; otherwise each is a local verification checkpoint.

## Review Focus

- Unknown preset name: `load_workcell_preset()` must fail with a message listing `dual_arm`, `single_arm_left`, and `single_arm_right`; covered in Task 1.
- Task attempts to override an invariant core field: task configuration validation must reject the conflicting key instead of silently changing the workcell; covered in Task 1.
- External processed assembly moved with the repository: relative fragment asset paths must resolve from `layout.json`, while the core preset resolves from the installed repository; covered in Task 4.
- Missing or partially copied historical artifact: repository cleanup must stop before `git rm` unless source and external-copy checksums match; covered in Task 6.
- Single-arm preset accidentally drifts from its corresponding dual-arm pose: an equality test must compare the complete robot configuration, not only base position; covered in Task 1.

---

## Target File Map

### Core

- `core/workcell/config.py`: typed config models, named-preset loading, invariant-key validation, and asset path resolution.
- `core/workcell/geometry.py`: pure worktable/pad geometry contracts.
- `core/workcell/environment.py`: USD authoring for floor, table, pad, lights, and shared materials.
- `core/workcell/cameras.py`: fixed agent/wrist camera definitions, capture helpers, and portable-stage output.
- `core/workcell/physics.py`: shared physics constants and material application.
- `core/robots/config.py`: typed robot-model and placement contracts.
- `core/robots/kinematics.py`: FANUC FK/IK and Robotiq mimic-joint model migrated from `asm_kin.py`.
- `core/robots/builder.py`: FANUC/Robotiq USD composition.
- `core/robots/controls.py`: interactive joint/gripper controls and reset support.
- `core/scene/builder.py`: SimulationApp/World creation and invariant workcell assembly.
- `core/scene/runtime.py`: render loop, streaming mode, reset dispatch, and shutdown.
- `core/scene/validation.py`: common stage-manifest validation.
- `core/config/workcell.json`: shared environment, render, physics, asset, robot-model, and camera data.
- `core/config/presets/*.json`: robot selections and exact canonical placements only.
- `core/assets/robots/`: relocated FANUC and Robotiq USD trees.
- `core/tests/`: all core contract and pure-logic tests.

### Tasks

- `tasks/demo_render_source/`: legacy CRAG/LEGO storyboard preprocessing, choreography, probing, rendering, CLI, documentation, and tests.
- `tasks/dual_arm_breakingbad/domain/`: assembly input/layout types and metadata validation.
- `tasks/dual_arm_breakingbad/preprocessing/`: loaders, normalization, voxelization, staging, and mesh export.
- `tasks/dual_arm_breakingbad/scene/`: fragment authoring, voxel colliders, assembly-specific controls, and core-workcell composition.
- `tasks/dual_arm_breakingbad/rendering/`: original/voxel inspection rendering.
- `tasks/dual_arm_breakingbad/__main__.py`: `prepare`, `validate`, `inspect`, `build`, and `launch` commands.
- Each task contains its own `tests/`, configuration, and README.

### Externalized content

- `01_crag_case/` → `/local_data/yz11445/datasets/raw/assembly_sim/demo_render_source/crag_case/`
- legacy `04_intermediate/bricks`, `choreo.json`, and `probe` → `/local_data/yz11445/datasets/processed/assembly_sim/demo_render_source/legacy_reference/`
- legacy logs and `05_outputs/` → `/local_data/yz11445/experiments/assembly_sim/demo_render_source/legacy_reference/`
- `04_intermediate/assemblies/39087_sf_fractured_0/` → `/local_data/yz11445/datasets/processed/assembly_sim/dual_arm_breakingbad/39087_sf_fractured_0/`
- `06_interactive_scene/generated/` and `inspections/` → `/local_data/yz11445/experiments/assembly_sim/dual_arm_breakingbad/reference_scene/`

---

### Task 1: Establish Core Configuration and Preset Contracts

**Files:**
- Create: `core/__init__.py`
- Create: `core/workcell/__init__.py`
- Create: `core/workcell/config.py`
- Create: `core/robots/__init__.py`
- Create: `core/robots/config.py`
- Create: `core/config/workcell.json`
- Create: `core/config/presets/dual_arm.json`
- Create: `core/config/presets/single_arm_left.json`
- Create: `core/config/presets/single_arm_right.json`
- Create: `core/tests/test_workcell_config.py`

**Interfaces:**
- Produces: `RobotConfig`, `WorkcellConfig`, `available_presets() -> tuple[str, ...]`, `load_workcell_preset(name: str, repo_root: Path | None = None) -> WorkcellConfig`, and `validate_task_workcell_boundary(data: Mapping[str, Any]) -> None`.
- `WorkcellConfig` exposes `physics_dt`, `render_dt`, `camera_resolution`, `environment`, `physics`, `render`, `robot_names`, `robot(name)`, `asset_path(name)`, and `preset_name`.

- [ ] **Step 1: Write failing preset and boundary tests**

  Add tests asserting:

  - available presets equal `("dual_arm", "single_arm_left", "single_arm_right")`;
  - dual left base is `[-0.78, 0.0, 0.75]` with quaternion `[1.0, 0.0, 0.0, 0.0]`;
  - dual right base is `[0.78, 0.0, 0.75]` with quaternion `[0.0, 0.0, 0.0, 1.0]`;
  - each single-arm robot object equals its corresponding dual-arm robot object;
  - table pose/size, pad center/size/grid step, physics values, camera resolution, and asset paths reproduce the current scene configuration;
  - an unknown preset raises `ValueError` and lists all valid names;
  - task data containing any of `environment`, `physics`, `render`, `robots`, or `cameras` raises `ValueError` naming the forbidden key.

- [ ] **Step 2: Run tests and confirm the missing-core failure**

  Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest core/tests/test_workcell_config.py -q`

  Expected: collection fails because `core.workcell.config` does not exist.

- [ ] **Step 3: Implement the typed configuration loaders and JSON files**

  Copy exact invariant values from `06_interactive_scene/config/scene.json`. Put robot model/DoF data in `workcell.json`; preset files contain only enabled robot names and base transforms. Resolve asset paths relative to the repository root discovered from `core/`, not from the current working directory.

- [ ] **Step 4: Run the core config tests**

  Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest core/tests/test_workcell_config.py -q`

  Expected: all tests pass.

- [ ] **Step 5: Checkpoint the task**

  Run `git diff --check`. If commits were explicitly authorized, commit with `refactor: add shared workcell presets`.

---

### Task 2: Extract the Reusable Workcell, Robots, and Scene Runtime

**Files:**
- Create: `core/workcell/geometry.py`
- Create: `core/workcell/physics.py`
- Create: `core/workcell/environment.py`
- Create: `core/workcell/cameras.py`
- Create: `core/robots/kinematics.py`
- Create: `core/robots/builder.py`
- Create: `core/robots/controls.py`
- Create: `core/scene/__init__.py`
- Create: `core/scene/builder.py`
- Create: `core/scene/runtime.py`
- Create: `core/scene/validation.py`
- Move: `02_robot_assets/Fanuc/CRX10IAL/` → `core/assets/robots/fanuc_crx10ial/`
- Move: `02_robot_assets/Robotiq/2F-85/` → `core/assets/robots/robotiq_2f85/`
- Move and refactor: reusable portions of `06_interactive_scene/scripts/scene_{geometry,cameras,controls}.py`
- Move and refactor: reusable portions of `06_interactive_scene/scripts/build_scene.py`
- Move: `03_scripts/asm_kin.py` → `core/robots/kinematics.py`
- Move tests: reusable `06_interactive_scene/tests/` cases → `core/tests/`

**Interfaces:**
- Consumes: `WorkcellConfig` and `RobotConfig` from Task 1.
- Produces: `AssemblyPadGeometry`, `assembly_pad_geometry(spec)`, `create_world(config, headless, stream=False)`, `author_environment(stage, config)`, `compose_robot(stage, world, config, name)`, `author_cameras(stage, config, robots)`, `WorkcellHandles`, `build_workcell(config, headless=True, stream=False)`, and shared control/capture/validation helpers.

- [ ] **Step 1: Move the existing pure tests and rewrite imports to `core`**

  Cover pad geometry, camera paths, launch configuration, joint clamping, gripper targets, unique robot/camera prim paths, single-arm manifests, and dual-arm manifests. Add a test that a single-arm manifest contains exactly one robot and only that robot's wrist camera while preserving the agent camera.

- [ ] **Step 2: Run the moved tests and confirm import failures**

  Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest core/tests -q`

  Expected: failures identify the not-yet-created core modules.

- [ ] **Step 3: Move robot assets and kinematics without changing their content or numeric behavior**

  Use `git mv` for tracked paths. Update config asset paths to `core/assets/robots/...`. Preserve all FANUC FK/IK and Robotiq mimic-joint public behavior before renaming any internal symbols.

- [ ] **Step 4: Implement the reusable workcell modules**

  Split invariant environment/robot/camera/runtime behavior from fragment-specific authoring. `build_workcell()` creates `/World`, environment, selected robots, shared cameras, physics, and resettable controls; it does not create `/World/Fragments` or inspect an assembly layout.

- [ ] **Step 5: Run core tests**

  Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest core/tests -q`

  Expected: all tests pass without launching a rendered Isaac application.

- [ ] **Step 6: Checkpoint the task**

  Run `git diff --check`. If commits were explicitly authorized, commit with `refactor: extract reusable Isaac workcell core`.

---

### Task 3: Migrate `demo_render_source`

**Files:**
- Create: `tasks/__init__.py`
- Create: `tasks/demo_render_source/__init__.py`
- Create: `tasks/demo_render_source/__main__.py`
- Create: `tasks/demo_render_source/README.md`
- Move/refactor: `03_scripts/asm_bricks.py` → `tasks/demo_render_source/preprocessing.py`
- Move/refactor: `03_scripts/asm_choreo.py` → `tasks/demo_render_source/choreography.py`
- Move/refactor: `03_scripts/asm_replay.py` → `tasks/demo_render_source/storyboard.py`
- Move/refactor: `03_scripts/fanuc_probe.py` → `tasks/demo_render_source/probe.py`
- Create: `tasks/demo_render_source/tests/test_cli.py`
- Create: `tasks/demo_render_source/tests/test_choreography_contract.py`
- Create: `tasks/demo_render_source/tests/test_storyboard_contract.py`

**Interfaces:**
- Consumes: core `dual_arm` preset, kinematics, robot builder, environment authoring, cameras, and external input/output paths.
- Produces: `build_parser()`, `main(argv=None) -> int`, `build_choreography(layout_path, probe_path, output_path, fps=30) -> Path`, and `render_storyboard(choreography_path, output_dir, ...) -> Path`.

- [ ] **Step 1: Write failing CLI and contract tests**

  Assert that:

  - `python -m tasks.demo_render_source --help` exposes `prepare`, `choreograph`, `render`, and `run`;
  - every write-producing command requires an explicit external output or work directory;
  - paths under the repository root are rejected for generated output;
  - choreography uses the exact dual-arm placements supplied by core, with no `SCRATCH` constant or copied robot pose;
  - storyboard construction calls the core workcell authoring interface and adds only task-owned bricks, ghosts, captions, and shot sequencing.

- [ ] **Step 2: Run tests and confirm missing-task failures**

  Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tasks/demo_render_source/tests -q`

  Expected: collection fails because the task package does not exist.

- [ ] **Step 3: Move and modularize the legacy scripts**

  Replace import-time argument parsing with callable functions and `main()`. Remove the absolute `SCRATCH` path. Replace duplicated table, arm, gripper, lighting, and camera setup with core APIs while preserving the storyboard-specific brick plate, ghosts, choreography, captions, render dimensions, and RSS guard.

- [ ] **Step 4: Add the task CLI and README**

  Document external CRAG input, processed work directory, render output, and exact module commands. Make `run` compose preprocessing, choreography, and rendering without writing inside the repository.

- [ ] **Step 5: Run demo task and core tests**

  Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest core/tests tasks/demo_render_source/tests -q`

  Expected: all tests pass.

- [ ] **Step 6: Checkpoint the task**

  Run `git diff --check`. If commits were explicitly authorized, commit with `refactor: package storyboard rendering task`.

---

### Task 4: Migrate Breaking Bad Domain and Preprocessing

**Files:**
- Create: `tasks/dual_arm_breakingbad/__init__.py`
- Create: `tasks/dual_arm_breakingbad/domain/__init__.py`
- Move/refactor: `03_scripts/fragment_assembly/model.py` → `tasks/dual_arm_breakingbad/domain/model.py`
- Move/refactor: `03_scripts/fragment_assembly/layout.py` → `tasks/dual_arm_breakingbad/domain/layout.py`
- Create: `tasks/dual_arm_breakingbad/preprocessing/__init__.py`
- Move/refactor: loader, voxelization, staging, mesh export, and preprocessing modules → `tasks/dual_arm_breakingbad/preprocessing/`
- Move/refactor: original and voxel render modules → `tasks/dual_arm_breakingbad/rendering/`
- Move: `tests/fragment_assembly/` → `tasks/dual_arm_breakingbad/tests/`

**Interfaces:**
- Consumes: `load_workcell_preset("dual_arm")` for pad/table/staging constraints; external raw and processed paths.
- Produces: `AssemblyInput`, `FragmentInput`, `AssemblyLayout`, `load_breaking_bad()`, optional `load_crag_glb()`, `process_assembly()`, `write_processed_assembly()`, `load_layout()`, and inspection render functions.

- [ ] **Step 1: Move tests and update them to the new package imports**

  Preserve existing loader, voxelization, cleaning, mesh export, layout, staging, rendering, and Breaking Bad integration assertions. Change temporary-layout metadata to record `workcell_preset: "dual_arm"` and `workcell_schema_version`, not an absolute config path.

- [ ] **Step 2: Add portability and output-boundary tests**

  Add tests asserting:

  - fragment NPZ paths remain relative to their `layout.json`;
  - loading the processed assembly after moving its containing directory still succeeds;
  - raw source provenance may be absolute metadata but is never required after preprocessing;
  - output inside the repository is rejected;
  - preprocessing reads table/pad/lane constraints from the core preset;
  - a task config containing invariant workcell keys is rejected.

- [ ] **Step 3: Run the moved tests and confirm import failures**

  Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tasks/dual_arm_breakingbad/tests -q`

  Expected: collection fails until the new package exists.

- [ ] **Step 4: Move domain and preprocessing modules**

  Remove `sys.path` mutation. Preserve common-grid voxelization, fragment identity, disconnected-component cleanup, plain voxel visuals, exact occupied-cell metadata, merged box colliders, GT poses, centered bilateral staging, and arbitrary fragment count behavior.

- [ ] **Step 5: Run preprocessing tests**

  Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest core/tests tasks/dual_arm_breakingbad/tests -q`

  Expected: all non-scene tests pass.

- [ ] **Step 6: Checkpoint the task**

  Run `git diff --check`. If commits were explicitly authorized, commit with `refactor: package Breaking Bad preprocessing`.

---

### Task 5: Rebuild the Interactive Breaking Bad Scene on Core

**Files:**
- Create: `tasks/dual_arm_breakingbad/scene/__init__.py`
- Create: `tasks/dual_arm_breakingbad/scene/assets.py`
- Create: `tasks/dual_arm_breakingbad/scene/geometry.py`
- Create: `tasks/dual_arm_breakingbad/scene/controller.py`
- Create: `tasks/dual_arm_breakingbad/scene/builder.py`
- Create: `tasks/dual_arm_breakingbad/scene/validation.py`
- Create: `tasks/dual_arm_breakingbad/__main__.py`
- Create: `tasks/dual_arm_breakingbad/README.md`
- Move/refactor: assembly-specific parts of `06_interactive_scene/scripts/`
- Move/refactor: remaining `06_interactive_scene/tests/` → `tasks/dual_arm_breakingbad/tests/`

**Interfaces:**
- Consumes: core `WorkcellHandles`, stage builder, runtime, controls, cameras, and `AssemblyLayout` from Task 4.
- Produces: `FragmentHandle`, `AssemblySceneHandles`, `author_fragment(stage, workcell, layout, name)`, `build_assembly_scene(layout_path, headless=True, stream=False)`, `AssemblySceneController`, `validate_assembly_scene()`, and task CLI commands.

- [ ] **Step 1: Rewrite scene tests against the new core/task boundary**

  Tests assert that core manifests contain environment/robots/cameras only, while task manifests add exactly `N` fragments. Preserve tests for occupied-cell box merging, collider geometry, dynamic-body settings, fragment reset, arbitrary fragment count, camera capture, streaming launch flags, and layout validation.

- [ ] **Step 2: Run scene tests and confirm missing-module failures**

  Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tasks/dual_arm_breakingbad/tests -q`

  Expected: scene tests fail until task scene modules are created.

- [ ] **Step 3: Implement assembly-specific scene composition**

  Build the invariant stage through `core`, then create `/World/Fragments`, author voxel visuals and merged non-overlapping box colliders from the same cells, attach fragment reset state, and reuse core robot controls/cameras. Do not accept task-side table, robot, pad, physics, camera, or lighting overrides.

- [ ] **Step 4: Implement the module CLI**

  Provide:

  - `prepare --input --output [--pitch --target-length --seed --object-id --overwrite]`;
  - `validate --assembly`;
  - `inspect --assembly --output`;
  - `build --assembly --output-usd`;
  - `launch --assembly [--stream --smoke-frames]`.

  Output-producing commands reject repository-contained output paths.

- [ ] **Step 5: Run all core and task tests**

  Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest core/tests tasks -q`

  Expected: all tests pass.

- [ ] **Step 6: Checkpoint the task**

  Run `git diff --check`. If commits were explicitly authorized, commit with `refactor: compose Breaking Bad scene from core`.

---

### Task 6: Externalize Historical Artifacts and Remove Legacy Top-Level Paths

**Files:**
- Create: `core/tests/test_repository_shape.py`
- Modify: `.gitattributes`
- Create: `.gitignore`
- Remove after verified external copy: `01_crag_case/`, `04_intermediate/`, `05_outputs/`, generated scene/capture content, empty numbered source directories, root `assembly_pipeline.py`, and obsolete `MANIFEST.sha256`.

**Interfaces:**
- Consumes: completed core and task packages from Tasks 1–5.
- Produces: a tracked tree whose only top-level directories are `core/`, `tasks/`, and `docs/`, plus allowed root metadata files.

- [ ] **Step 1: Write the failing repository-shape test**

  Use `git ls-files` so ignored runtime caches do not affect the test. Assert every tracked path is inside `core/`, `tasks/`, or `docs/`, or is one of `.gitattributes`, `.gitignore`, and `README.md`. Assert no tracked file matches generated-output patterns (`*.png`, `*.mp4`, `*.tar`, generated scene USD, logs, or processed NPZ/layout artifacts), except source robot USD assets under `core/assets/robots/`.

- [ ] **Step 2: Run the shape test and confirm it lists all legacy paths**

  Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest core/tests/test_repository_shape.py -q`

  Expected: fail with the current numbered directories and generated artifacts.

- [ ] **Step 3: Inventory and copy each historical artifact group externally**

  Create the exact destinations listed in the Target File Map. Copy without overwriting an existing destination. For every regular file, compare source and destination SHA-256 values and save the inventory under the corresponding external destination. If any destination differs, stop before removing tracked content.

- [ ] **Step 4: Remove only checksum-verified legacy tracked paths**

  Use explicit `git rm` targets after Step 3 passes. Update `.gitattributes` to remove obsolete `05_outputs/frames_png.tar` handling and add `.gitignore` rules for Python caches, pytest caches, generated USD, captures, and task output directories.

- [ ] **Step 5: Remove active legacy path references**

  Run:

  ```bash
  rg -n "01_crag_case|02_robot_assets|03_scripts|04_intermediate|05_outputs|06_interactive_scene|assembly_pipeline" core tasks README.md docs --glob '!docs/superpowers/**'
  ```

  Expected: no active source/config/usage references. Historical design records under `docs/superpowers/` may retain old path names.

- [ ] **Step 6: Run repository-shape and full pure test suites**

  Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest core/tests tasks -q`

  Expected: all tests pass, including repository shape.

- [ ] **Step 7: Checkpoint the task**

  Run `git diff --check`. If commits were explicitly authorized, commit with `refactor: externalize generated repository artifacts`.

---

### Task 7: Update Documentation and Run End-to-End Acceptance

**Files:**
- Rewrite: `README.md`
- Create: `docs/architecture.md`
- Create: `docs/workcells.md`
- Create: `docs/tasks.md`
- Update: `tasks/demo_render_source/README.md`
- Update: `tasks/dual_arm_breakingbad/README.md`

**Interfaces:**
- Consumes: all public commands and APIs from Tasks 1–6.
- Produces: the documented repository architecture and fresh external acceptance artifacts.

- [ ] **Step 1: Write the final documentation**

  Root README introduces only `core`, `tasks/demo_render_source`, and `tasks/dual_arm_breakingbad`; links to task commands; explains external paths; and does not retain the legacy numbered-directory guide. Architecture docs record ownership/dependency rules, immutable preset rules, and the local-first promotion policy.

- [ ] **Step 2: Run the complete test suite**

  Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest -q`

  Expected: all tests pass with no collection from legacy directories.

- [ ] **Step 3: Preprocess the acceptance sample outside the repository**

  Run:

  ```bash
  /local_data/yz11445/tools/bin/isaacsim-4.5-python \
    -m tasks.dual_arm_breakingbad prepare \
    --input /local_data/yz11445/datasets/raw/breaking_bad/example_data/artifact/39087_sf/fractured_0 \
    --output /local_data/yz11445/datasets/processed/assembly_sim/dual_arm_breakingbad/39087_sf_fractured_0 \
    --overwrite
  ```

  Expected: three fragments, common-grid voxelization, GT and centered bilateral staging poses, relative processed asset paths, and `workcell_preset: "dual_arm"`.

- [ ] **Step 4: Validate the processed assembly**

  Run:

  ```bash
  /local_data/yz11445/tools/bin/isaacsim-4.5-python \
    -m tasks.dual_arm_breakingbad validate \
    --assembly /local_data/yz11445/datasets/processed/assembly_sim/dual_arm_breakingbad/39087_sf_fractured_0/layout.json
  ```

  Expected: schema and all three external fragment assets validate without reading the raw dataset.

- [ ] **Step 5: Inspect GPU state and capture the rebuilt scene on one GPU**

  Inspect `nvidia-smi`, choose one suitable GPU, and run the task's build/inspection validation with `CUDA_VISIBLE_DEVICES=<one-gpu>`. Write the USD, agent camera, and both wrist camera images to `/local_data/yz11445/experiments/assembly_sim/dual_arm_breakingbad/restructure_validation_2026-10-02/`.

  Expected: two FANUC/Robotiq systems, unchanged worktable and grid pad, three non-intersecting voxel fragments in centered bilateral lanes, and valid 1920×1440 camera captures.

- [ ] **Step 6: Smoke-test both single-arm presets**

  Build each core-only preset headlessly for a short smoke run. Expected: the selected arm occupies exactly its canonical dual-arm pose; the table, pad, environment, physics, and applicable cameras remain unchanged; no second robot is authored.

- [ ] **Step 7: Perform final static verification**

  Run `git diff --check`, the repository-shape test, the full pytest suite, and the legacy-reference `rg` command from Task 6. Inspect `git status --short` and confirm no generated acceptance artifact appears in the repository.

- [ ] **Step 8: Checkpoint the completed refactor**

  If commits were explicitly authorized, commit documentation and any final fixes with `docs: document simulation task architecture`. Do not push unless the user explicitly requests it.
