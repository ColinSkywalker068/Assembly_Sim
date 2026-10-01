# Generic Fragment Assembly Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an install-free pipeline that converts an arbitrary Breaking Bad fractured-object directory into plain voxel fragment assets and launches the existing interactive dual-FANUC scene from the generated assembly metadata.

**Architecture:** A new pure-Python `fragment_assembly` package owns dataset adapters, the dataset-neutral assembly model, canonicalization, voxelization, layout schema, and deterministic staging. `layout.json` schema version 2 is the object-specific boundary; the existing Isaac scene overlays it onto the reusable workcell configuration while retaining its schema-version-1 skull fallback.

**Tech Stack:** Python 3.10, NumPy, SciPy, trimesh, pytest, JSON/NPZ, Isaac Sim 4.5, OpenUSD, PhysX

**Spec:** `docs/superpowers/specs/2026-10-01-generic-fragment-assembly-scene-design.md`

## Global Constraints

- Work only inside `/local_data/yz11445/projects/Assembly_Sim` and read the Breaking Bad source dataset without modifying it.
- Use `/local_data/yz11445/envs/assembly-sim/bin/python` for preprocessing and pure-Python tests.
- Never launch or configure more than two GPUs; inspect GPU usage before any Isaac/GPU run and explicitly constrain the process to at most two available GPUs.
- Do not install software globally or modify system configuration.
- Preserve both FANUC CRX-10iA/L arms, Robotiq grippers, table, cameras, physics, controls, reset, capture, and save behavior.
- Defaults are exactly `0.40 m` target assembled length, `0.016 m` voxel pitch, random seed `0`, and staging gap `0.04 m`.
- Generic visuals contain only exposed voxel-cube faces; generic outputs contain no studs, `plate.npz`, or ghost/prediction geometry.
- Collision geometry is the exact occupied-cell union expressed as deterministic non-overlapping merged boxes.
- Keep the checked-in skull scene usable when no `--assembly` override is supplied.
- Do not create commits unless the user explicitly asks later.

## Review Focus

- OBJ names such as `piece_2.obj` and `piece_10.obj` must sort numerically and duplicate numeric identities must fail; Task 1 tests both cases.
- A surface-sampling tie or random generator change must not silently change cell ownership between identical runs; Task 2 tests exact repeatability and deterministic tie resolution.
- Exported local pivots, GT poses, and collider coordinates must reconstruct the same canonical cells; Tasks 2 and 3 test reconstruction and volume equality.
- A large fragment set that cannot fit safely on the table must fail before publishing an output directory; Task 3 tests capacity failure and atomic publication.
- Supplying a malformed schema-version-2 layout must fail explicitly rather than falling back to skull assets; Tasks 3 and 4 test schema, paths, and override precedence.

---

### Task 1: Dataset-neutral assembly model and Breaking Bad loader

**Files:**
- Create: `03_scripts/fragment_assembly/__init__.py`
- Create: `03_scripts/fragment_assembly/model.py`
- Create: `03_scripts/fragment_assembly/loaders.py`
- Create: `tests/fragment_assembly/conftest.py`
- Create: `tests/fragment_assembly/test_loaders.py`

**Interfaces:**
- Produces: `FragmentInput(name: str, source_name: str, source_path: Path, mesh: trimesh.Trimesh, ground_truth_transform: np.ndarray)`.
- Produces: `AssemblyInput(object_id: str, loader: str, source_path: Path, fragments: tuple[FragmentInput, ...])` with validation for a non-empty uniquely named collection and finite 4-by-4 transforms.
- Produces: `load_breaking_bad(directory: Path, object_id: str | None = None) -> AssemblyInput` and `load_crag_glb(path: Path, object_id: str | None = None) -> AssemblyInput`.

- [ ] **Step 1: Write failing model and loader tests**

Add tests named `test_breaking_bad_discovers_arbitrary_piece_count_in_numeric_order`, `test_breaking_bad_uses_identity_ground_truth_transforms`, `test_loader_rejects_empty_directory`, `test_loader_rejects_duplicate_numeric_piece_ids`, `test_loader_rejects_non_triangle_or_empty_mesh`, and `test_crag_loader_preserves_scene_node_transforms`. Build real tiny OBJ/GLB fixtures under `tmp_path`; assert literal ordered names and matrices.

- [ ] **Step 2: Run Task 1 tests and verify RED**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider tests/fragment_assembly/test_loaders.py -q`

Expected: FAIL because `fragment_assembly.model` and `fragment_assembly.loaders` do not exist.

- [ ] **Step 3: Implement the model and loaders**

Implement frozen dataclasses with validation in `model.py`. In `loaders.py`, discover exactly `piece_<integer>.obj`, sort by integer suffix, load with trimesh without combining different files, and assign identity GT transforms for Breaking Bad. Load CRAG geometry nodes with their scene-graph transforms as the fragment GT relationship.

- [ ] **Step 4: Run Task 1 tests and verify GREEN**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider tests/fragment_assembly/test_loaders.py -q`

Expected: PASS.

### Task 2: Deterministic canonicalization and plain common-grid voxelization

**Files:**
- Create: `03_scripts/fragment_assembly/voxelize.py`
- Create: `03_scripts/fragment_assembly/mesh_export.py`
- Create: `tests/fragment_assembly/test_voxelize.py`
- Create: `tests/fragment_assembly/test_mesh_export.py`

**Interfaces:**
- Consumes: `AssemblyInput` from Task 1.
- Produces: `VoxelizationConfig(pitch: float = 0.016, target_length: float = 0.40, seed: int = 0, min_surface_samples: int = 20000, samples_per_pitch_area: float = 12.0)`.
- Produces: `ProcessedFragment(name, source_name, cells, local_pivot_cells, local_bounds, vertices, faces, goal_pose, grasp)` and `ProcessedAssembly(object_id, source, config, source_to_canonical, grid_origin, grid_shape, assembly_bounds, fragments)`.
- Produces: `process_assembly(assembly: AssemblyInput, config: VoxelizationConfig) -> ProcessedAssembly`.
- Produces: `build_voxel_surface(cells, pitch, pivot_cells) -> tuple[np.ndarray, np.ndarray]` with cube faces only.

- [ ] **Step 1: Write failing canonicalization and occupancy tests**

Add tests named `test_one_global_transform_preserves_fragment_relationship`, `test_target_length_and_pitch_are_recorded_exactly`, `test_all_fragments_share_one_grid`, `test_cell_ownership_is_disjoint`, `test_highest_vote_ties_follow_fragment_order`, `test_cleanup_keeps_only_largest_26_connected_component`, `test_empty_fragment_after_cleanup_is_fatal`, and `test_same_seed_repeats_exact_cells`. Use hand-sized box meshes and literal expected transforms/cell properties.

- [ ] **Step 2: Run voxel tests and verify RED**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider tests/fragment_assembly/test_voxelize.py -q`

Expected: FAIL because the processing API does not exist.

- [ ] **Step 3: Implement canonicalization and shared-grid occupancy**

Extract the stable-pose candidate scoring, long-horizontal-axis alignment, uniform scaling, dense surface voting, deterministic tie handling, 26-connected cleanup, Z crop, grasp calculation, and palette behavior from `03_scripts/asm_bricks.py`. Use an explicit NumPy generator derived from `seed`; record the complete source-to-canonical and grid transforms.

- [ ] **Step 4: Run voxel tests and verify GREEN**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider tests/fragment_assembly/test_voxelize.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing plain-geometry tests**

Add tests named `test_single_voxel_is_exact_cube_surface`, `test_adjacent_voxels_omit_internal_faces`, `test_surface_bounds_reconstruct_occupied_cells`, and `test_export_contains_no_non_cube_vertices`. The first two assert literal vertex/triangle counts and exact bounds, not implementation source text.

- [ ] **Step 6: Run geometry tests and verify RED**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider tests/fragment_assembly/test_mesh_export.py -q`

Expected: FAIL because `build_voxel_surface` is absent.

- [ ] **Step 7: Implement plain voxel surface generation**

Generalize the current cube-face builder, remove stud generation entirely from the new representation, and export float32 vertices/int32 triangle faces about each deterministic local pivot.

- [ ] **Step 8: Run Task 2 tests and verify GREEN**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider tests/fragment_assembly/test_voxelize.py tests/fragment_assembly/test_mesh_export.py -q`

Expected: PASS.

### Task 3: Layout schema, deterministic staging, and atomic export

**Files:**
- Create: `03_scripts/fragment_assembly/layout.py`
- Create: `03_scripts/fragment_assembly/staging.py`
- Create: `03_scripts/fragment_assembly/preprocess.py`
- Create: `tests/fragment_assembly/test_layout.py`
- Create: `tests/fragment_assembly/test_staging.py`
- Create: `tests/fragment_assembly/test_preprocess.py`
- Modify: `06_interactive_scene/scripts/plan_layout.py`
- Modify: `06_interactive_scene/tests/test_plan_layout.py`

**Interfaces:**
- Consumes: `ProcessedAssembly` and the workcell `environment`/`robots` mappings.
- Produces: `Pose`, `PieceBounds`, `TableBounds`, `StagingSpec`, and `compute_staging_poses(piece_bounds, spec) -> dict[str, Pose]` in the shared package; `plan_layout.py` imports/re-exports compatible types instead of maintaining a second packer.
- Produces: `write_processed_assembly(processed, output_dir, workcell, overwrite=False) -> Path` returning the absolute published `layout.json`.
- Produces: `load_layout(path: Path, validate_assets: bool = True) -> AssemblyLayout` for schema version 2 and `load_legacy_layout(path, fragment_names, initial_poses) -> AssemblyLayout` for checked-in schema-version-1 data.

- [ ] **Step 1: Write failing staging tests**

Add tests named `test_staging_is_deterministic_for_arbitrary_names`, `test_staged_aabbs_are_separated_and_inside_table`, `test_lowest_voxel_face_touches_table`, `test_staging_avoids_both_robot_bases`, and `test_capacity_failure_names_first_unplaced_piece`. Retain the current skull packing tests as compatibility coverage.

- [ ] **Step 2: Run staging tests and verify RED**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider tests/fragment_assembly/test_staging.py 06_interactive_scene/tests/test_plan_layout.py -q`

Expected: FAIL because the shared staging package does not exist.

- [ ] **Step 3: Extract and generalize deterministic packing**

Move the pure AABB placement contract out of `plan_layout.py`, preserve deterministic descending-area/name ordering, search the configured table grid, exclude both robot bases, set Z from the exact local minimum, and keep scene reachability helpers separate.

- [ ] **Step 4: Run staging tests and verify GREEN**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider tests/fragment_assembly/test_staging.py 06_interactive_scene/tests/test_plan_layout.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing schema and export tests**

Add tests named `test_schema_v2_round_trip_contains_required_object_state`, `test_mesh_paths_resolve_relative_to_layout`, `test_validator_rejects_duplicate_or_malformed_cells`, `test_validator_rejects_missing_goal_or_staging_pose`, `test_validator_does_not_consult_source_dataset`, `test_export_writes_one_npz_per_fragment_without_plate_or_ghost`, `test_failed_export_does_not_publish_partial_directory`, and `test_nonempty_output_requires_overwrite`.

- [ ] **Step 6: Run layout/export tests and verify RED**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider tests/fragment_assembly/test_layout.py tests/fragment_assembly/test_preprocess.py -q`

Expected: FAIL because schema loading and atomic export are absent.

- [ ] **Step 7: Implement schema version 2 and atomic publication**

Serialize every field required by the spec, including explicit relative mesh paths, cells, local pivot/bounds, GT goal poses, staging poses/context, processing parameters, and normalization matrices. Validate staged content before an atomic sibling-directory rename; never mutate raw input.

- [ ] **Step 8: Run all Task 3 tests and verify GREEN**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider tests/fragment_assembly/test_layout.py tests/fragment_assembly/test_staging.py tests/fragment_assembly/test_preprocess.py 06_interactive_scene/tests/test_plan_layout.py -q`

Expected: PASS.

### Task 4: Data-driven scene configuration and fragment contracts

**Files:**
- Modify: `06_interactive_scene/scripts/scene_config.py`
- Modify: `06_interactive_scene/scripts/scene_assets.py`
- Modify: `06_interactive_scene/scripts/scene_geometry.py`
- Modify: `06_interactive_scene/scripts/build_scene.py`
- Modify: `06_interactive_scene/tests/test_scene_config.py`
- Modify: `06_interactive_scene/tests/test_scene_geometry.py`
- Modify: `06_interactive_scene/tests/test_stage_manifest.py`
- Modify: `06_interactive_scene/tests/test_validation_contracts.py`

**Interfaces:**
- Consumes: `load_layout` and `load_legacy_layout` from Task 3.
- Changes: `SceneConfig.load(path: Path, assembly_path: Path | None = None) -> SceneConfig`.
- Produces: `SceneConfig.assembly`, `fragment_names`, `fragment_spec(name)`, `fragment_mesh_path(name)`, `fragment_initial_pose(name)`, and `has_support_surface`.
- Changes: `StageManifest.plate_path: str | None`; generic stage manifests omit the plate while legacy skull manifests retain it.

- [ ] **Step 1: Write failing arbitrary-fragment scene tests**

Replace the fixed-eight rejection test with `test_schema_v2_override_accepts_three_arbitrary_fragments`, `test_override_layout_is_authoritative_over_legacy_fragment_config`, `test_malformed_override_never_falls_back_to_skull`, `test_fragment_mesh_uses_explicit_layout_relative_path`, and `test_legacy_skull_config_still_loads_eight_fragments`.

- [ ] **Step 2: Run configuration tests and verify RED**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider 06_interactive_scene/tests/test_scene_config.py -q`

Expected: FAIL because `SceneConfig` enforces `piece_0` through `piece_7` and cannot load an assembly override.

- [ ] **Step 3: Implement the scene assembly overlay**

Remove `EXPECTED_FRAGMENTS`, load schema version 2 when `assembly_path` is supplied, expose object state through methods instead of mutating `scene.json`, and retain existing path behavior for schema-version-1 skull operation.

- [ ] **Step 4: Write failing manifest, support, and collider tests**

Add tests named `test_generic_manifest_has_three_fragment_paths_and_no_plate`, `test_legacy_manifest_keeps_plate`, `test_merged_boxes_cover_every_cell_once`, and `test_fragment_collider_local_coordinates_match_visual_pivot`. Update collision validation fixtures to use the selected layout accessor.

- [ ] **Step 5: Run asset-contract tests and verify RED**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider 06_interactive_scene/tests/test_scene_geometry.py 06_interactive_scene/tests/test_stage_manifest.py 06_interactive_scene/tests/test_validation_contracts.py -q`

Expected: FAIL because plate authoring and fragment mesh lookup are mandatory and name-derived.

- [ ] **Step 6: Generalize scene authoring**

Make support authoring conditional; iterate `config.fragment_names`; obtain mesh path, cells, color, pivot, and staging pose from `fragment_spec`; keep merged boxes and mass properties derived only from those cells; update manifest validation for the optional plate.

- [ ] **Step 7: Run Task 4 tests and verify GREEN**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider 06_interactive_scene/tests/test_scene_config.py 06_interactive_scene/tests/test_scene_geometry.py 06_interactive_scene/tests/test_stage_manifest.py 06_interactive_scene/tests/test_validation_contracts.py -q`

Expected: PASS.

### Task 5: Reset, validation, and command-line assembly selection

**Files:**
- Modify: `06_interactive_scene/scripts/scene_controls.py`
- Modify: `06_interactive_scene/scripts/run_scene.py`
- Modify: `06_interactive_scene/scripts/validate_scene.py`
- Modify: `06_interactive_scene/tests/test_dual_robot_contract.py`
- Modify: `06_interactive_scene/tests/test_key_bindings.py`
- Modify: `06_interactive_scene/tests/test_validation_contracts.py`

**Interfaces:**
- Consumes: Task 4 `SceneConfig.load(..., assembly_path=...)` and fragment accessors.
- Changes: both Isaac entry points accept optional `--assembly Path` and pass it to `SceneConfig.load`.
- Changes: reset and validation obtain expected poses from `config.fragment_initial_pose(name)` and layout data from `config.assembly`.

- [ ] **Step 1: Write failing selection and reset tests**

Add tests named `test_run_options_accept_assembly_override`, `test_validation_options_accept_assembly_override`, `test_reset_targets_every_loaded_fragment`, and `test_reset_uses_staging_not_goal_pose`. Extract argument-parser builders where necessary so tests exercise parsing without starting Isaac.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider 06_interactive_scene/tests/test_key_bindings.py 06_interactive_scene/tests/test_dual_robot_contract.py 06_interactive_scene/tests/test_validation_contracts.py -q`

Expected: FAIL because entry points and reset use only `scene.json` fragment poses.

- [ ] **Step 3: Implement assembly CLI propagation and generic reset**

Add `--assembly` without making it required, pass it through all scene construction and validation paths, remove direct reads of `config.data["fragments"]["initial_poses"]`, and make optional plate reopen validation conditional.

- [ ] **Step 4: Run Task 5 tests and verify GREEN**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider 06_interactive_scene/tests/test_key_bindings.py 06_interactive_scene/tests/test_dual_robot_contract.py 06_interactive_scene/tests/test_validation_contracts.py -q`

Expected: PASS.

### Task 6: Top-level workflow, milestone generation, and documentation

**Files:**
- Create: `assembly_pipeline.py`
- Modify: `03_scripts/asm_bricks.py`
- Create: `tests/fragment_assembly/test_cli.py`
- Create: `tests/fragment_assembly/test_breaking_bad_integration.py`
- Modify: `README.md`
- Modify: `06_interactive_scene/README.md`
- Create during validation: `04_intermediate/assemblies/39087_sf_fractured_0/layout.json`
- Create during validation: `04_intermediate/assemblies/39087_sf_fractured_0/piece_0.npz`
- Create during validation: `04_intermediate/assemblies/39087_sf_fractured_0/piece_1.npz`
- Create during validation: `04_intermediate/assemblies/39087_sf_fractured_0/piece_2.npz`

**Interfaces:**
- Consumes: all prior package and scene APIs.
- Produces: `assembly_pipeline.py prepare`, `validate`, and `launch` subcommands; `launch` delegates to the existing scene runner in the current Isaac interpreter.
- Preserves: direct legacy scene launch with no assembly override; `asm_bricks.py` becomes a documented compatibility wrapper for CRAG GT GLB preprocessing without prediction support in the generalized path.

- [ ] **Step 1: Write failing CLI behavior tests**

Add tests named `test_prepare_requires_dataset_and_input`, `test_prepare_defaults_output_from_object_id`, `test_prepare_exposes_pitch_length_seed_scene_and_overwrite`, `test_validate_succeeds_after_source_directory_is_unavailable`, and `test_launch_delegates_config_and_assembly_arguments`. Exercise real subprocesses for `prepare`/`validate`; isolate only the external Isaac process boundary for `launch`.

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider tests/fragment_assembly/test_cli.py -q`

Expected: FAIL because `assembly_pipeline.py` does not exist.

- [ ] **Step 3: Implement the top-level workflow and compatibility wrapper**

Use lazy imports per subcommand, default to `06_interactive_scene/config/scene.json`, print absolute outputs and ready-to-copy launch commands, and return nonzero on all validation/preprocessing errors. Keep the milestone path out of production code.

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider tests/fragment_assembly/test_cli.py -q`

Expected: PASS.

- [ ] **Step 5: Write and run the milestone integration test RED**

The test reads the source path from environment variable `BREAKING_BAD_SAMPLE`; skip only when unset. Assert the literal three discovered names, three non-empty NPZ files, no plate/ghost assets, one shared grid, disjoint occupied cells, collision-box volume equality, and three separated table-contact staging poses.

Run: `BREAKING_BAD_SAMPLE=/local_data/yz11445/datasets/raw/breaking_bad/example_data/artifact/39087_sf/fractured_0 /local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider tests/fragment_assembly/test_breaking_bad_integration.py -q`

Expected: initially FAIL on the first missing end-to-end behavior, then PASS after connecting Tasks 1-5 through the CLI.

- [ ] **Step 6: Generate and validate the milestone output**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python assembly_pipeline.py prepare --dataset breaking-bad --input /local_data/yz11445/datasets/raw/breaking_bad/example_data/artifact/39087_sf/fractured_0 --overwrite`

Then run: `/local_data/yz11445/envs/assembly-sim/bin/python assembly_pipeline.py validate --assembly 04_intermediate/assemblies/39087_sf_fractured_0/layout.json`

Expected: both exit `0`; exactly three generated fragment NPZ files; validation reads no original geometry.

- [ ] **Step 7: Update operator documentation**

Document the environment interpreter, prepare/validate/launch commands, parameter overrides, output schema, no-stud/no-plate representation, staging versus GT poses, legacy skull command, controls, and the exact Isaac-only validation commands.

### Task 7: Full verification and Isaac acceptance when available

**Files:**
- Verify all files listed above.
- Regenerate only when an authorized Isaac environment is available: `06_interactive_scene/generated/interactive_scene.usd` and three inspection PNGs.

**Interfaces:**
- Consumes: the completed pipeline and milestone layout.
- Produces: fresh pure-Python, integration, metadata, and optional Isaac evidence.

- [ ] **Step 1: Run the complete pure-Python suite**

Run: `/local_data/yz11445/envs/assembly-sim/bin/python -m pytest -p no:cacheprovider tests/fragment_assembly 06_interactive_scene/tests -q`

Expected: all tests PASS with no project warnings or failures.

- [ ] **Step 2: Run repeatability and generated-output inspection**

Prepare the milestone twice into separate directories with identical arguments; compare normalized `layout.json` content and NPZ arrays exactly. Inspect metadata with a script that verifies required keys, three names, common pitch/grid, finite transforms, local mesh bounds, disjoint cells, and non-overlapping staging AABBs.

Expected: exact match and zero inspection failures.

- [ ] **Step 3: Check authorized GPU/Isaac availability**

Run `nvidia-smi` before any Isaac command. Do not launch if this account already consumes two GPUs, if no authorized Isaac interpreter exists under the user's workspace, or if an additional run cannot be constrained to at most two GPUs.

- [ ] **Step 4: Run headless Isaac validation if available**

Run with at most two explicitly selected GPUs:

`CUDA_VISIBLE_DEVICES=<one-or-two-authorized-device-ids> <isaac-python> 06_interactive_scene/scripts/validate_scene.py --config 06_interactive_scene/config/scene.json --assembly 04_intermediate/assemblies/39087_sf_fractured_0/layout.json --full --save-stage --capture`

Expected: exit `0`; two robots, three cameras, three fragments, no plate, exact merged-voxel collider contract, stable physics, portable reopen, and three non-empty captures.

- [ ] **Step 5: Run interactive smoke launch if available**

Run: `CUDA_VISIBLE_DEVICES=<same-authorized-device-ids> <isaac-python> assembly_pipeline.py launch --assembly 04_intermediate/assemblies/39087_sf_fractured_0/layout.json --smoke-frames 180`

Expected: scene opens/builds, both robots and all cameras exist, three separated fragments rest on the table, controls initialize, reset restores staging, and the process exits cleanly after 180 frames.

- [ ] **Step 6: Review the final diff and acceptance checklist**

Run `git diff --check`, inspect `git status --short`, map each specification acceptance criterion to fresh evidence, and report any unrun Isaac-only validation separately rather than implying it passed.
