# Interactive Isaac Sim Assembly Scene

This directory is a separate, physics-enabled first version of the skull-assembly scene. It uses one FANUC CRX-10iA/L, one Robotiq 2F-85, the eight existing voxelized fragments, a fixed front camera, and a wrist camera. The original storyboard under `01_crag_case` through `05_outputs` is not modified.

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

Complete physics validation, portable USD save, and both camera snapshots:

```powershell
D:\i45\python.exe 06_interactive_scene/scripts/validate_scene.py --config 06_interactive_scene/config/scene.json --full --save-stage --capture
```

Generated outputs belong only to this scene:

- `generated/interactive_scene.usd` — inspectable stage with repository-relative asset references.
- `inspections/agent_camera.png` — front overview.
- `inspections/wrist_camera.png` — wrist-mounted view.

Regenerating these files intentionally replaces the previous generated versions. Edit `config/scene.json` or the builder scripts rather than hand-editing the generated USD.

## Launch the interactive GUI

```powershell
D:\i45\python.exe 06_interactive_scene/scripts/run_scene.py --config 06_interactive_scene/config/scene.json
```

The GUI builds the scene from source configuration, starts on the agent camera, and continues simulating until the Isaac window is closed.

### Controls

| Key | Action |
| --- | --- |
| `1`–`6` | Select FANUC joint 1–6 |
| `[` / `]` | Jog the selected joint negative / positive by 5 degrees |
| `H` | Command the arm home pose |
| `O` / `K` | Open / close the Robotiq gripper |
| `R` | Reset robot, gripper, and all eight fragments |
| `P` | Save the portable USD |
| `I` | Capture both cameras |
| `F7` / `F8` | Select agent / wrist camera in the viewport |

The action keys avoid Isaac's standard `W`, `A`, `S`, `D`, `Q`, and `E` viewport-navigation keys. Key release and repeat events are ignored, so each press issues one bounded command.

## Camera inspection

The agent camera is fixed in front of and above the workcell. It frames the complete arm, table, plate, and staged fragments. The wrist camera is attached to the FANUC flange and looks forward/down toward the gripper workspace. Press `F7` or `F8` to switch the interactive viewport, and press `I` to refresh both PNG snapshots.

## Troubleshooting

- **A long blank or unresponsive first start:** allow the initial RTX shader compilation to finish. The validation log reports `Waiting for RtPso async group async compilation` while it is active.
- **`c10.dll` initialization errors:** launch through these scripts. They preload the installed PyTorch package before Isaac starts to avoid a Windows extension-discovery race in this Isaac 4.5 pip environment.
- **Missing USD, NPZ, or probe files:** keep the repository directory structure intact and run with `06_interactive_scene/config/scene.json`; paths are resolved from the repository root, not the current drive letter.
- **VRAM pressure or poor responsiveness:** close other GPU applications, keep one Isaac instance open, and use the 640×480 configured resolution. This version targets the available 8 GB VRAM budget.
- **RAM pressure:** close other large applications. The machine used for validation has 16 GB system RAM, which is adequate for scene inspection but leaves limited headroom.
- **Fragments or camera layout need adjustment:** edit the deterministic poses in `config/scene.json`, rerun the full validation, and inspect both generated PNGs.

## Deliberate non-goals

This version is for scene construction, inspection, and minor manual motion. It does not include policy training, data collection, domain randomization, task rewards, motion planning, automatic grasping, or assembly success metrics. Fragment collision is limited to merged occupied-voxel boxes; the fragments begin asleep for a stable inspection view and wake normally when contacted.
