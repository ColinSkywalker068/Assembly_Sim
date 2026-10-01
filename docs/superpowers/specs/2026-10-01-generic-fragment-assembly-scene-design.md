# Generic Fragment Assembly Scene Design

## Purpose

Generalize the repository's voxel-fragment preprocessing and interactive Isaac Sim scene so one complete fractured-object sample can be selected at the command line. The first supported dataset is Breaking Bad, where a sample directory contains `piece_*.obj` meshes in a shared ground-truth assembled coordinate frame.

The milestone sample is:

`/local_data/yz11445/datasets/raw/breaking_bad/example_data/artifact/39087_sf/fractured_0`

The implementation must discover its three OBJ fragments, generate plain voxel visuals and equivalent voxel collision geometry, stage the pieces separately on the existing table, and load them into the existing dual-FANUC interactive scene without source edits.

## Scope

This change includes one-object preprocessing, a Breaking Bad loader, a dataset-neutral assembly interface, common-grid voxelization, generated metadata and NPZ meshes, deterministic worktable staging, data-driven Isaac fragment authoring, validation, and documented prepare/launch commands.

It does not include learned policies, CRAG inference, predicted poses, reassembly execution, new grasp planning, trajectory planning, success metrics, training infrastructure, or dataset-wide batch processing.

## Design Principles

- Dataset adapters end at a dataset-neutral in-memory assembly object.
- Voxelization and scene code do not inspect dataset-specific filenames or metadata.
- `layout.json` is the complete object-specific interface consumed by the scene; the original dataset is not needed after preprocessing.
- `scene.json` retains reusable workcell state: robots, table, cameras, physics, and rendering.
- Visual meshes and colliders derive from the same occupied voxel cells.
- New generic objects do not require LEGO studs, a LEGO baseplate, or prediction assets.
- Existing skull assets and configuration remain usable through a legacy compatibility path.

## Architecture

The pipeline has four boundaries:

1. A dataset loader converts a source sample into an `Assembly` containing an identifier, source metadata, and an ordered collection of fragments. Each fragment contains a triangle mesh and its ground-truth transform into the assembly frame.
2. Dataset-independent preprocessing applies one global orientation and scale, voxelizes all transformed fragments on one grid, cleans disconnected voxel artifacts, and exports per-fragment visuals plus schema-versioned metadata.
3. A deterministic staging planner uses voxel-derived local bounds and the selected workcell's table and exclusion geometry. It stores world-space staging poses in the generated metadata.
4. The Isaac scene combines the static workcell configuration with a selected generated `layout.json`, authors an arbitrary number of fragments, and uses their stored staging poses for startup and reset.

The core processor accepts the assembly model rather than a path. Adding a future dataset therefore requires a loader that constructs the same model, not changes to voxelization, staging, collider generation, or scene authoring.

## Dataset Model and Loaders

The dataset-neutral model has these conceptual fields:

- `object_id`: stable human-readable identifier.
- `source`: loader name, input path, and optional dataset metadata.
- `fragments`: an ordered, non-empty collection.
- Per fragment: unique canonical name, source name/path, triangle mesh, and a 4-by-4 ground-truth transform from fragment mesh coordinates to the source assembly frame.

The Breaking Bad loader accepts one directory, discovers files matching `piece_*.obj`, orders numeric suffixes naturally, rejects duplicate suffixes or an empty set, loads every file as a triangle mesh, and uses identity ground-truth transforms. It does not assume three fragments. The default object identifier is derived from the useful trailing sample path components and may be overridden by CLI.

The loader never mutates source files. Raw dataset paths are provenance only and are not consulted by Isaac Sim.

The current CRAG GLB loading behavior may be retained behind a separate loader or compatibility wrapper. Predicted GLBs remain outside the generalized milestone.

## Canonicalization and Ground Truth

The source assembly is formed by applying each fragment's ground-truth transform and concatenating the results conceptually. The processor chooses a deterministic useful stable orientation from the assembled object, aligns its longer horizontal extent with the canonical X axis, and uniformly scales it so that extent is the configured target length. Defaults are `0.40 m` target length and `0.016 m` voxel pitch.

One recorded 4-by-4 `source_to_canonical` matrix contains the selected rotation and scale. The common voxel grid adds a recorded grid origin and any Z crop/translation. Together these fields make the relationship between original source coordinates, canonical metric coordinates, voxel cells, and fragment-local exported geometry explicit.

Every fragment visual is exported around a deterministic local pivot. Its GT pose places that local visual back into the canonical assembled object. These GT poses are distinct from staging poses and are not used as initial dynamic-body poses.

Randomized sampling uses an explicit configurable seed, defaulting to zero. Repeated preprocessing with the same input and parameters must produce the same occupied cells and staging poses.

## Voxelization and Geometry

All fragments share one pitch, origin, and integer grid. Dense surface samples preserve thin fragment walls. When multiple fragments vote for one cell, the deterministic highest-vote rule assigns it to one fragment, with a deterministic fragment-order tie break. Each fragment retains only its largest 26-connected component, matching the useful cleanup behavior of the current script. Empty results are fatal and identify the affected source fragment.

The generated visual mesh is the exposed boundary of the occupied cubes for that fragment. It contains no cylinders or studs. Internal faces between adjacent cells of the same fragment are omitted, but the resulting surface represents exactly the union of occupied voxel cells.

Physics collision boxes are generated at scene-build time by greedily merging adjacent cells into deterministic, non-overlapping axis-aligned boxes. Their union must equal the fragment's occupied cell set; no support footprint or invisible extra volume is allowed.

The processor may continue to compute existing top-down grasp metadata where possible, because current offline reachability tools consume it. Failure to find a grasp does not invalidate scene generation; the value is stored as null and reachability validation reports that it cannot evaluate that fragment.

## Generated Representation

New outputs use layout schema version 2. A generated object directory contains `layout.json` and one NPZ visual mesh per fragment. Generic output does not contain `plate.npz`, LEGO studs, a predicted ghost, or source mesh copies.

`layout.json` contains at least:

- `schema_version` and `object_id`.
- Source loader, absolute input path, and source fragment names.
- Processing parameters: pitch, target length, sample density settings, cleanup connectivity, and random seed.
- Normalization: `source_to_canonical`, canonical/grid origins, grid shape, and assembly bounds.
- Fragment count and ordered fragment entries.
- Per fragment: canonical name, source name, relative generated mesh path, color, occupied integer cells, voxel count, local pivot and bounds, GT pose in the canonical assembly frame, world-space staging pose, and optional grasp metadata.
- Staging context: workcell identity or config provenance, table surface height and usable bounds, exclusions, gap, and algorithm identifier.

Paths to generated meshes are resolved relative to `layout.json`, not inferred from names and not required to lie inside the repository. The layout validator rejects unsupported schema versions, duplicate fragment names, missing or empty meshes, malformed/duplicate cells, inconsistent fragment counts, invalid transforms, and absent GT or staging poses.

Legacy schema-version-1 skull metadata remains readable when selected by the existing scene configuration. Its hard-coded names and initial poses are treated as compatibility data, not as the new interface.

## Automatic Staging

The existing deterministic AABB packer is generalized and shared by preprocessing tests and scene tooling. It derives each fragment's exact local AABB from occupied cells and its local pivot. All pieces keep identity orientation for the first implementation.

Pieces are sorted deterministically by descending footprint area and canonical name. Candidate positions are searched deterministically over the usable table bounds. A candidate is accepted only when:

- its complete AABB is within the configured table margin;
- its expanded footprint does not intersect either robot-base exclusion;
- it does not intersect any previously placed fragment after applying the configured gap; and
- its Z translation places the lowest occupied voxel face on the table surface.

The default staging gap remains `0.04 m`. If all fragments cannot fit, preprocessing fails with the first unplaced piece and table/capacity context rather than generating intersecting poses. The old baseplate is not an exclusion for generic objects because it is not authored.

## Isaac Scene Integration

`scene.json` remains schema version 1 for the workcell unless an independent workcell schema change proves necessary. It continues to define the two FANUC CRX-10iA/L robots, Robotiq grippers, table, cameras, physics, rendering, controls, and output paths.

Scene commands accept an optional `--assembly PATH/TO/layout.json`. When supplied, the generated layout is authoritative for fragment names, mesh paths, colors, occupied cells, and initial/staging poses. The workcell config's legacy `fragments` section and brick directory are ignored. Without `--assembly`, the current skull scene continues to load as before.

The stage manifest contains zero or one optional generic support path instead of requiring `/World/Environment/Plate`. New generic layouts author no support object. Fragment prim paths are derived from validated fragment names, and all build, validation, control-reset, and save paths iterate the loaded fragment collection rather than a fixed range.

Reset restores every generated fragment to its stored staging pose and resets both existing robots and grippers. Robot selection, joint jogging, gripper controls, cameras, capture, simulation, and USD saving otherwise remain unchanged.

## Command-Line Workflow

A repository-root `assembly_pipeline.py` provides one discoverable interface with lazy imports so preprocessing does not start Isaac and launching does not require source meshes:

```bash
/local_data/yz11445/envs/assembly-sim/bin/python assembly_pipeline.py prepare \
  --dataset breaking-bad \
  --input /local_data/yz11445/datasets/raw/breaking_bad/example_data/artifact/39087_sf/fractured_0
```

The default output is `04_intermediate/assemblies/<object-id>/`; `--output`, `--pitch`, `--target-length`, `--seed`, and `--scene-config` are exposed. The command prints the absolute generated `layout.json` path and a ready-to-copy launch command.

The same entry point supports metadata validation in ordinary Python:

```bash
/local_data/yz11445/envs/assembly-sim/bin/python assembly_pipeline.py validate \
  --assembly 04_intermediate/assemblies/<object-id>/layout.json
```

Interactive launch is run under the user's Isaac Sim Python interpreter:

```bash
<isaac-python> assembly_pipeline.py launch \
  --assembly 04_intermediate/assemblies/<object-id>/layout.json
```

The existing direct `run_scene.py --config ... --assembly ...` form remains available. The entry point never hard-codes the milestone source path.

## Error Handling

User-facing failures include the relevant path, fragment name, or parameter. Preprocessing stops before partial metadata is published when input discovery, mesh loading, normalization, voxelization, mesh export, or staging fails. Files are built in a temporary sibling directory and moved into the requested output only after validation; an existing non-empty output requires an explicit overwrite flag. Raw source data is never modified.

Isaac startup validates the selected layout and every generated mesh before creating a stage. It reports schema and path errors without silently falling back to the skull object. Capacity failures, empty voxel pieces, and malformed transforms are errors rather than warnings.

## Testing and Validation

Pure-Python tests use small synthetic meshes and temporary directories to cover:

- Breaking Bad file discovery, numeric ordering, arbitrary fragment counts, identity GT transforms, and invalid/empty inputs.
- Global-transform preservation of the relative assembled geometry.
- Deterministic shared-grid ownership, cleanup, and repeatability.
- Plain cube-surface visual output with no stud geometry.
- Visual occupied volume and merged collision-box volume equivalence.
- Layout schema version 2 round-trip validation and layout-relative asset resolution.
- Deterministic non-overlapping staging, table contact, table bounds, robot exclusions, and capacity failure.
- Scene config overlay, arbitrary manifest fragment counts, optional plate handling, and reset-pose selection.
- Legacy skull loading without changing its checked-in assets.

The complete existing pure-Python scene suite must pass after its fixed-eight expectations are replaced with generic-contract expectations.

The milestone integration test preprocesses the specified Breaking Bad directory and asserts exactly three source fragments, three non-empty NPZ meshes, three disjoint occupied-cell sets on one grid, three collision-equivalent cell sets, and three non-overlapping staging poses resting on the configured table. Metadata validation must pass without consulting the source directory.

If an authorized local Isaac Sim installation is available, final validation builds the stage, runs the existing robot-motion and physics checks, saves the USD, captures all three cameras, and performs a GUI/smoke launch. If it is not available, the handoff explicitly separates completed preprocessing/static validation from the unrun Isaac-only checks and provides the exact commands for the user environment.

## Acceptance Criteria

The milestone is complete when the Breaking Bad sample can be prepared by path, its three OBJ files are treated as one GT assembly, and preprocessing emits three plain voxel fragment meshes plus schema-version-2 metadata. The metadata records normalization, common-grid cells, GT poses, and deterministic separated staging poses. The scene consumes that file without object-specific source edits, authors collision boxes from the same cells, retains both FANUC/Robotiq systems and all cameras/controls, and resets fragments to staging. Selecting another compatible Breaking Bad sample requires only changing the input path.
