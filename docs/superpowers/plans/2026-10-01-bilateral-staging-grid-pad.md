# Bilateral Staging and Grid Pad Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage every generated fragment assembly in two front-to-back lanes around a generic central grid pad, using the skull demo's workcell geometry as the reference.

**Architecture:** Extend the dataset-independent preprocessing staging model with an explicit bilateral workcell layout: a central AABB pad exclusion plus fixed left/right lane X centers. Persist the resulting side assignments and poses in schema-v2 metadata. Extend the Isaac scene contract with a generic flat grid pad that is authored for generic layouts while preserving the legacy skull's studded plate path unchanged.

**Tech Stack:** Python 3.10, NumPy, pytest, generated schema-v2 JSON/NPZ assets, USD/Isaac Sim 4.5 APIs.

**Spec:** `docs/superpowers/specs/2026-10-01-bilateral-staging-grid-pad-design.md`

## Global Constraints

- Keep raw datasets immutable and place regenerated outputs only below `/local_data/yz11445/experiments` or the existing processed/intermediate assembly output.
- Do not add LEGO studs or a `plate.npz` dependency to a generic assembly.
- Reference geometry is exactly: pad center `(0.000, 0.100) m`, footprint `0.512 x 0.416 m`, top `z = 0.750 m`, thickness `0.0096 m`, lane centers `x = -0.420` and `x = +0.420 m`, grid step `0.040 m`, and minimum gap `0.040 m`.
- Generic staged fragments vary along Y at their assigned fixed X lane; fragment bottoms rest at the table top.
- Keep the two FANUC arms, cameras, physics, controls, reset behavior, and legacy skull plate behavior intact.
- Do not make a git commit unless the user explicitly requests one.
- Before the Isaac scene inspection, inspect `nvidia-smi` and constrain the run to no more than two GPUs.

## Review Focus

- A three-piece input must create a deterministic `2 + 1` side assignment, rather than silently retain radial packing; test in Task 1.
- A fragment whose AABB crosses its fixed lane or lacks Y capacity must fail clearly instead of being relocated around the pad; test in Task 1.
- The generic grid pad must be flat and static but its visual grid must not create extra collision geometry; test in Task 2.
- A schema-v2 override must gain the generic pad while the no-override skull scene retains only its legacy plate; test in Task 2.
- Regenerated assets must be enough for the scene without rereading Breaking Bad raw meshes; validate and capture in Task 3.

---

### Task 1: Dataset-independent bilateral staging and layout metadata

**Files:**
- Modify: `03_scripts/fragment_assembly/staging.py`
- Modify: `03_scripts/fragment_assembly/preprocess.py`
- Modify: `06_interactive_scene/config/scene.json`
- Modify: `tests/fragment_assembly/test_staging.py`
- Modify: `tests/fragment_assembly/test_preprocess.py`

**Interfaces:**
- Consumes: `PieceBounds`, `TableBounds`, and scene workcell configuration.
- Produces: `StagingSpec` fields for a pad exclusion and `lane_centers_x`; `compute_staging_poses(piece_bounds, spec) -> dict[str, Pose]`; schema-v2 `staging` records containing algorithm version, pad bounds, lane centers, assignments, and poses.

- [ ] **Step 1: Write failing bilateral-staging tests**

Add tests that pass three named `PieceBounds` into a spec with pad bounds `[-0.256, 0.256] x [-0.108, 0.308]`, lane centers `(-0.420, 0.420)`, and demo table bounds. Assert source-name ordering assigns two fragments to left and one to right, each side has a fixed X center, Y differs within a side, every AABB is separated, outside the pad, and rests at `z = 0.750`. Add a capacity test where the second fragment assigned to a lane cannot fit in Y and assert the error names that fragment and `lane capacity`.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tests/fragment_assembly/test_staging.py -q`

Expected: FAIL because `StagingSpec` has no bilateral pad/lane contract and the current radial packer does not produce the required lane positions.

- [ ] **Step 3: Implement bilateral placement in `staging.py`**

Add validated optional `pad: AABB2D | None` and `lane_centers_x: tuple[float, float] | None` fields to `StagingSpec`. When both are configured, allocate source-name-sorted fragments as the first `ceil(N/2)` left and remaining right; pack each lane in deterministic ascending Y order, using each fragment's actual AABB width/depth and `gap`. Reject fragments that cannot fit centered on their lane or whose row exceeds usable table Y bounds with a `lane capacity` error. Keep existing radial AABB packing as the fallback when bilateral fields are absent.

- [ ] **Step 4: Add failing metadata tests**

Extend `_workcell()` with the exact `assembly_pad` configuration and make `_processed()` produce three source-name-order fragments. Assert written `layout.json["staging"]` has `algorithm == "bilateral_lanes_v1"`, reference pad/lane values, the `2 + 1` assignments, and per-piece staging poses matching the bilateral placement.

- [ ] **Step 5: Run the preprocessing test to verify it fails**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tests/fragment_assembly/test_preprocess.py -q`

Expected: FAIL because preprocessing does not yet pass the pad/lane specification or persist assignment data.

- [ ] **Step 6: Wire scene workcell settings into preprocessing**

Add `environment.assembly_pad` in `06_interactive_scene/config/scene.json` using the exact global-constraint dimensions. In `preprocess._staging_spec`, derive its `AABB2D` and lane centers, include the pad with robot bases as exclusions, and have `_layout_data` persist all bilateral staging parameters and a deterministic fragment-to-side mapping. Do not change `goal_pose` or voxel data.

- [ ] **Step 7: Run Task 1 tests to verify they pass**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tests/fragment_assembly/test_staging.py tests/fragment_assembly/test_preprocess.py -q`

Expected: PASS.

### Task 2: Generic USD assembly grid pad and scene contract

**Files:**
- Modify: `06_interactive_scene/scripts/scene_config.py`
- Modify: `06_interactive_scene/scripts/scene_assets.py`
- Modify: `06_interactive_scene/scripts/build_scene.py`
- Modify: `06_interactive_scene/scripts/validate_scene.py`
- Modify: `06_interactive_scene/tests/test_scene_config.py`
- Modify: `06_interactive_scene/tests/test_stage_manifest.py`
- Modify: `06_interactive_scene/tests/test_scene_geometry.py`
- Modify: `06_interactive_scene/tests/test_validation_contracts.py`

**Interfaces:**
- Consumes: `SceneConfig.assembly`, `environment.assembly_pad`, and stage asset helpers.
- Produces: `SceneConfig.has_generic_assembly_pad -> bool`; `StageManifest.assembly_pad_path: str | None`; generic USD prim `/World/Environment/AssemblyPad` with a visible grid and one static flat collider.

- [ ] **Step 1: Write failing scene-contract tests**

Add a generic override fixture assertion that `has_generic_assembly_pad` is true and `expected_stage_manifest()` returns `assembly_pad_path == "/World/Environment/AssemblyPad"` while `plate_path is None`. Preserve the legacy assertions that `plate_path == "/World/Environment/Plate"` and generic scenes do not use it. Add pure geometry assertions for the generic pad center, top, and below-surface collider center using the exact reference values.

- [ ] **Step 2: Run focused contract tests to verify they fail**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest 06_interactive_scene/tests/test_scene_config.py 06_interactive_scene/tests/test_stage_manifest.py 06_interactive_scene/tests/test_scene_geometry.py -q`

Expected: FAIL because the generic support surface is currently disabled and the manifest has no generic pad path.

- [ ] **Step 3: Implement the scene contract and pad geometry**

Expose a config property that selects the generic pad only for schema-v2 assembly overrides. Extend `StageManifest` with `assembly_pad_path` without altering legacy plate names. Add a pure geometry helper/dataclass for the pad's center, size, top, and collider center.

In `author_environment`, author `/World/Environment/AssemblyPad` for generic layouts: a thin flat visual top with non-colliding 40-mm grid-line children and exactly one invisible static box collider extending 9.6 mm below `z=0.750`. Use the existing physics material. Keep the old `plate.npz` branch byte-for-byte behavior where feasible. Update scene build/validation checks to require the selected support asset and not count grid lines as fragment or pad collision volume.

- [ ] **Step 4: Run focused contract tests to verify they pass**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest 06_interactive_scene/tests/test_scene_config.py 06_interactive_scene/tests/test_stage_manifest.py 06_interactive_scene/tests/test_scene_geometry.py 06_interactive_scene/tests/test_validation_contracts.py -q`

Expected: PASS.

### Task 3: Regenerate the Breaking Bad reference and inspect the scene

**Files:**
- Modify: `README.md`
- Modify: `06_interactive_scene/README.md`
- Regenerate: `04_intermediate/assemblies/39087_sf_fractured_0/layout.json` and its three existing NPZ assets
- Create: `/local_data/yz11445/experiments/breaking_bad_39087_sf_fractured_0_bilateral_pad_inspection_2026-10-01/README.md`
- Create: `/local_data/yz11445/experiments/breaking_bad_39087_sf_fractured_0_bilateral_pad_inspection_2026-10-01/agent_camera.png`
- Create: `/local_data/yz11445/experiments/breaking_bad_39087_sf_fractured_0_bilateral_pad_inspection_2026-10-01/left_wrist_camera.png`
- Create: `/local_data/yz11445/experiments/breaking_bad_39087_sf_fractured_0_bilateral_pad_inspection_2026-10-01/right_wrist_camera.png`

**Interfaces:**
- Consumes: completed preprocessing and generic-pad scene contract plus the existing Breaking Bad sample directory.
- Produces: checked generated assembly metadata with a bilateral `2 + 1` layout and scene images for user review.

- [ ] **Step 1: Write/update documentation tests or assertions where the project uses them**

Add/adjust an existing CLI/preprocess test so `assembly_pipeline.py prepare` records the bilateral staging algorithm in generated schema-v2 metadata when using the stock scene configuration.

- [ ] **Step 2: Run the focused CLI/preprocess test to verify it fails before its implementation dependency is complete**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest tests/fragment_assembly/test_preprocess.py tests/test_cli.py -q`

Expected: PASS only after Tasks 1–2 are complete; if run before then, it documents the missing bilateral metadata contract.

- [ ] **Step 3: Document the user workflow**

Update both READMEs to state that generic scenes include a flat central grid assembly pad and bilateral front-to-back staging lanes. Show the existing `assembly_pipeline.py prepare` command, with no hard-coded dataset path, and state that `layout.json` stores the generated staging poses.

- [ ] **Step 4: Regenerate and validate the Breaking Bad sample**

Run `assembly_pipeline.py prepare --dataset breaking-bad` against `/local_data/yz11445/datasets/raw/breaking_bad/example_data/artifact/39087_sf/fractured_0`, targeting the existing intermediate assembly directory with `--overwrite`. Run `assembly_pipeline.py validate --assembly` on its resulting `layout.json`. Inspect metadata to confirm three fragments, bilateral algorithm, and a two-left/one-right fixed-X lane assignment.

- [ ] **Step 5: Capture and inspect the scene**

Run `nvidia-smi`, choose no more than two GPUs, and launch the existing headless scene capture flow with the regenerated assembly. Save agent, left-wrist, and right-wrist images in the named experiment directory. Visually confirm the generic grid pad is centered and the agent view shows the required `2 + 1` front-to-back staging lanes.

- [ ] **Step 6: Run full verification**

Run: `/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest -q`

Expected: all tests pass (the existing suite may require a detached log because reachability checks exceed the interactive command window). Then run `git diff --check`.

## Self-review

- Spec coverage: Task 1 implements the generic bilateral placement and persisted layout contract; Task 2 implements the non-LEGO visible/static pad and preserves the skull branch; Task 3 regenerates, validates, documents, and captures the accepted Breaking Bad test case.
- Type consistency: `StagingSpec` is the sole preprocessing placement input; `SceneConfig.has_generic_assembly_pad` selects `StageManifest.assembly_pad_path`; the generic configuration is consumed by both paths.
- Review focus coverage: the five listed failure classes are respectively covered by Tasks 1, 1, 2, 2, and 3.
- Proportion: implementation decisions are named and bounded; no generated code or unrequested planning/training scope is included.
