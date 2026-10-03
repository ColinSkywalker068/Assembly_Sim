# Assembly_Sim

`Assembly_Sim` is the source repository for a reusable Isaac Sim workcell and
independent simulation tasks. It intentionally contains only three source
areas:

- `core/`: the shared table, grid pad, FANUC/Robotiq models, physics, cameras,
  controls, validation, and dual/single-arm presets;
- `tasks/demo_render_source/`: the original scripted CRAG/LEGO storyboard;
- `tasks/dual_arm_breakingbad/`: fragmented-object preprocessing and the
  interactive dual-arm scene.

Raw data, processed assemblies, captures, renders, logs, and generated USD
stages live outside this repository under `/local_data/yz11445`.

## Breaking Bad quick start

From this repository root:

```bash
ISAAC_PY=/local_data/yz11445/tools/bin/isaacsim-4.5-python
SAMPLE=/local_data/yz11445/datasets/raw/breaking_bad/example_data/artifact/39087_sf/fractured_0
PROCESSED=/local_data/yz11445/datasets/processed/assembly_sim/dual_arm_breakingbad/39087_sf_fractured_0

$ISAAC_PY -m tasks.dual_arm_breakingbad prepare \
  --input "$SAMPLE" --output "$PROCESSED" --overwrite
$ISAAC_PY -m tasks.dual_arm_breakingbad validate \
  --assembly "$PROCESSED/layout.json"
$ISAAC_PY -m tasks.dual_arm_breakingbad launch \
  --assembly "$PROCESSED/layout.json"
```

Changing `SAMPLE` and `PROCESSED` is sufficient for another compatible
Breaking Bad fractured-object directory. See
[the task guide](tasks/dual_arm_breakingbad/README.md) for offline inspection,
stage export, streaming, and controls.

## Documentation

- [Architecture](docs/architecture.md)
- [Workcell presets](docs/workcells.md)
- [Task ownership and commands](docs/tasks.md)

Use Isaac Sim 4.5 Python for the complete test suite:

```bash
/local_data/yz11445/tools/bin/isaacsim-4.5-python -m pytest -q
```
