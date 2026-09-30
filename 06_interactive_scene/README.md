# Interactive Isaac Sim Assembly Scene

**Because this is developed on a Windows 10 computer, a lot of info below might not be suitable on your device. Please make educated adjustments for your situation.**

This directory is a separate, physics-enabled version of the skull-assembly scene. It reproduces the demo arrangement with two FANUC CRX-10iA/L arms, two Robotiq 2F-85 grippers, the eight existing voxelized fragments in their demo lineup, one fixed agent camera, and one wrist camera per arm. The original storyboard under `01_crag_case` through `05_outputs` is not modified.

## Prerequisites

- Windows with Isaac Sim 4.5 installed in `D:\i45`.
- An NVIDIA RTX GPU and a current compatible driver. This scene was validated on an RTX 4060 Laptop GPU with 8 GB VRAM.
- Run commands from the repository root.
- Accept the NVIDIA Omniverse EULA when required. The scripts set `OMNI_KIT_ACCEPT_EULA=YES` for non-interactive launches.

The first launch can take several minutes while Isaac downloads extension dependencies and compiles RTX shaders. Later launches use the cache and are substantially faster.

## Validate and build artifacts

Fast manifest check:

```powershell
D:\i45\python.exe 06_interactive_scene/scripts/validate_scene.py --config 06_interactive_scene/config/scene.json --manifest-only
```

Robot and gripper motion check:

```powershell
D:\i45\python.exe 06_interactive_scene/scripts/validate_scene.py --config 06_interactive_scene/config/scene.json --robot-motion
```

Complete physics validation, portable USD save, and all three camera snapshots:

```powershell
D:\i45\python.exe 06_interactive_scene/scripts/validate_scene.py --config 06_interactive_scene/config/scene.json --full --save-stage --capture
```

Generated outputs belong only to this scene:

- `generated/interactive_scene.usd` — inspectable stage with repository-relative asset references.
- `inspections/agent_camera.png` — front overview.
- `inspections/left_wrist_camera.png` — left wrist-mounted view.
- `inspections/right_wrist_camera.png` — right wrist-mounted view.

Regenerating these files intentionally replaces the previous generated versions. Edit `config/scene.json` or the builder scripts rather than hand-editing the generated USD.

## Launch the interactive GUI

```powershell
D:\i45\python.exe 06_interactive_scene/scripts/run_scene.py --config 06_interactive_scene/config/scene.json
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
| `R` | Reset both robots, both grippers, and all eight fragments |
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
- **VRAM pressure or poor responsiveness:** close other GPU applications, keep one Isaac instance open, and use the 640×480 configured resolution. This version targets the available 8 GB VRAM budget.
- **RAM pressure:** close other large applications. The machine used for validation has 16 GB system RAM, which is adequate for scene inspection but leaves limited headroom.
- **Fragments or camera layout need adjustment:** edit the deterministic poses in `config/scene.json`, rerun the full validation, and inspect all three generated PNGs.

## Deliberate non-goals

This version is for scene construction, inspection, and minor manual motion. It does not include policy training, data collection, domain randomization, task rewards, coordinated dual-arm motion planning, automatic grasping, or assembly success metrics. Fragment collision is limited to merged occupied-voxel boxes; the fragments begin asleep for a stable inspection view and wake normally when contacted.
