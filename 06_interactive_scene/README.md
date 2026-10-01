# Interactive Isaac Sim Assembly Scene

This directory contains the physics-enabled dual-FANUC assembly workcell. With no assembly override it reproduces the original eight-piece skull demo. With `--assembly` it discovers an arbitrary generated fragment set from schema-version-2 `layout.json`, places every fragment at its stored staging pose, and creates exact merged-box colliders from the same occupied voxel cells used by its visual mesh.

Prepare and validate generic objects from the repository root with `assembly_pipeline.py`; see the root README for the full workflow. Generic objects use plain voxel cubes and the worktable directly—no LEGO studs or baseplate.

## Maxwell installation

- Isaac Sim 4.5 is installed at `/local_data/yz11445/tools/isaacsim-4.5`.
- Use `/local_data/yz11445/tools/bin/isaacsim-4.5-python` as the Isaac Python interpreter.
- Maxwell's NVIDIA RTX 6000 Ada GPUs and driver 580.173.02 have been validated with this installation.
- Run commands from the repository root.
- Accept the NVIDIA Omniverse EULA when required. The scripts set `OMNI_KIT_ACCEPT_EULA=YES` for non-interactive launches.

The first launch can take several minutes while Isaac downloads extension dependencies and compiles RTX shaders. Later launches use the cache and are substantially faster.

## Validate and build artifacts

Fast manifest check:

```bash
/local_data/yz11445/tools/bin/isaacsim-4.5-python \
  06_interactive_scene/scripts/validate_scene.py \
  --config 06_interactive_scene/config/scene.json \
  --manifest-only
```

For any validation command, select a generated object by adding:

```powershell
--assembly 04_intermediate/assemblies/<object-id>/layout.json
```

Robot and gripper motion check:

```bash
/local_data/yz11445/tools/bin/isaacsim-4.5-python \
  06_interactive_scene/scripts/validate_scene.py \
  --config 06_interactive_scene/config/scene.json \
  --robot-motion
```

Complete physics validation, portable USD save, and all three camera snapshots:

```bash
/local_data/yz11445/tools/bin/isaacsim-4.5-python \
  06_interactive_scene/scripts/validate_scene.py \
  --config 06_interactive_scene/config/scene.json \
  --full --save-stage --capture
```

Generated outputs belong only to this scene:

- `generated/interactive_scene.usd` — inspectable stage with repository-relative asset references.
- `inspections/agent_camera.png` — front overview.
- `inspections/left_wrist_camera.png` — left wrist-mounted view.
- `inspections/right_wrist_camera.png` — right wrist-mounted view.

Regenerating these files intentionally replaces the previous generated versions. Edit `config/scene.json` or the builder scripts rather than hand-editing the generated USD.

## Launch the interactive GUI over WebRTC

From an SSH session on Maxwell, inspect `nvidia-smi` and select one available
GPU, then run:

```bash
CUDA_VISIBLE_DEVICES=<one-free-gpu> \
  /local_data/yz11445/tools/bin/isaacsim-4.5-python assembly_pipeline.py launch \
  --assembly 04_intermediate/assemblies/39087_sf_fractured_0/layout.json \
  --stream
```

On Windows, install NVIDIA's Isaac Sim WebRTC Streaming Client and connect to
`128.238.176.100`. The network path must allow TCP `49100` and UDP `47998`;
ordinary SSH `-L` forwarding is TCP-only and cannot carry the media stream.
Use the university network or VPN because the stream has no built-in
authentication or encryption. Only one client can attach to an Isaac instance.

For a machine with a local display, omit `--stream`:

```bash
/local_data/yz11445/tools/bin/isaacsim-4.5-python \
  06_interactive_scene/scripts/run_scene.py \
  --config 06_interactive_scene/config/scene.json
```

Generic-object launch:

```bash
/local_data/yz11445/tools/bin/isaacsim-4.5-python assembly_pipeline.py launch \
  --assembly 04_intermediate/assemblies/<object-id>/layout.json
```

The GUI builds the scene from source configuration, starts on the agent camera, and continues simulating until the Isaac window is closed.

### Controls

| Key | Action |
| --- | --- |
| `Tab` | Switch the active robot between `left` and `right` |
| `1`–`6` | Select joint 1–6 for the active robot |
| `[` / `]` | Jog the active robot's selected joint negative / positive by 5 degrees |
| `H` | Command the active arm to its home pose |
| `O` / `K` | Open / close the active Robotiq gripper |
| `R` | Reset both robots, both grippers, and every loaded fragment to staging |
| `P` | Save the portable USD |
| `I` | Capture the agent and both wrist cameras |
| `F7` / `F8` / `F9` | Select agent / left wrist / right wrist camera in the viewport |

The action keys avoid Isaac's standard `W`, `A`, `S`, `D`, `Q`, and `E` viewport-navigation keys. Key release and repeat events are ignored, so each press issues one bounded command.

## Camera inspection

The agent camera is fixed in front of and above the workcell. It frames both arms, the table, plate, and all staged fragments. Each wrist camera is attached to its matching FANUC flange and looks forward/down toward that gripper's reachable workspace. Press `F7`, `F8`, or `F9` to switch the interactive viewport, and press `I` to refresh all three PNG snapshots.

## Troubleshooting

- **A long blank or unresponsive first start:** allow the initial RTX shader compilation to finish. The validation log reports `Waiting for RtPso async group async compilation` while it is active.
- **`c10.dll` initialization errors:** launch through these scripts. They preload the installed PyTorch package before Isaac starts to avoid a Windows extension-discovery race in this Isaac 4.5 pip environment.
- **Missing USD, NPZ, or probe files:** keep the repository directory structure intact and run with `06_interactive_scene/config/scene.json`; paths are resolved from the repository root, not the current drive letter.
- **VRAM pressure or poor responsiveness:** inspect `nvidia-smi`, use one lightly loaded GPU, keep one Isaac instance open, and use the configured 640×480 resolution.
- **RAM pressure:** inspect `free -h` and stop only your own unnecessary processes; never terminate another user's work on Maxwell.
- **Fragments or camera layout need adjustment:** edit the deterministic poses in `config/scene.json`, rerun the full validation, and inspect all three generated PNGs.

## Deliberate non-goals

This version is for scene construction, inspection, and minor manual motion. It does not include policy training, data collection, domain randomization, task rewards, coordinated dual-arm motion planning, automatic grasping, or assembly success metrics. Fragment collision is limited to merged occupied-voxel boxes; the fragments begin asleep for a stable inspection view and wake normally when contacted.
