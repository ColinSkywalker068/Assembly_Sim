# Demo storyboard renderer

This task reproduces the original scripted CRAG/LEGO dual-FANUC storyboard.
It uses the shared `core` dual-arm workcell and adds only storyboard-specific
studded fragments, pose ghosts, choreography, captions, and frame rendering.

All inputs and outputs are external to the repository:

```bash
/local_data/yz11445/tools/bin/isaacsim-4.5-python \
  -m tasks.demo_render_source run \
  --input /path/to/view_gt.glb \
  --predicted /path/to/view_assembly_-1.glb \
  --work-dir /local_data/yz11445/scratch/demo-render-source \
  --output /local_data/yz11445/experiments/demo-render-source/run-name
```

The `prepare`, `choreograph`, and `render` subcommands expose the same stages
individually. Run `python -m tasks.demo_render_source --help` for arguments.

The preserved reference inputs and results from the original repository were
externalized to:

- `/local_data/yz11445/datasets/raw/assembly_sim/demo_render_source/crag_case/`
- `/local_data/yz11445/datasets/processed/assembly_sim/demo_render_source/legacy_reference/`
- `/local_data/yz11445/experiments/assembly_sim/demo_render_source/legacy_reference/`
