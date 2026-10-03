***#UPDATE: THE INTERACTION SCENE SOURCE CODE NOW EXISTS IN 06_.../***

***New README can be found inside 06_.../, and the README below is the legacy one for the demo storyboard***

## Generic fragmented-object workflow

The interactive workcell can load a generated voxel assembly instead of being limited to the checked-in skull. Breaking Bad sample directories containing `piece_<integer>.obj` meshes in a common assembled frame are supported first.

Prepare one sample with the installed Isaac Sim 4.5 Python interpreter:

```bash
/local_data/yz11445/tools/bin/isaacsim-4.5-python assembly_pipeline.py prepare \
  --dataset breaking-bad \
  --input /path/to/object/fractured_0
```

The default output is `04_intermediate/assemblies/<parent>_<sample>/`. Optional flags include `--output`, `--pitch`, `--target-length`, `--seed`, `--object-id`, `--scene-config`, and `--overwrite`. Defaults are a 16 mm pitch, 0.40 m assembled X length, and seed zero.

Validate the generated object without consulting the raw dataset, then launch it under the Isaac Sim Python interpreter:

```bash
/local_data/yz11445/tools/bin/isaacsim-4.5-python assembly_pipeline.py validate \
  --assembly 04_intermediate/assemblies/<object-id>/layout.json

/local_data/yz11445/tools/bin/isaacsim-4.5-python assembly_pipeline.py launch \
  --assembly 04_intermediate/assemblies/<object-id>/layout.json
```

For the SSH-only Maxwell workflow, launch the same scene with its full UI over
WebRTC:

```bash
CUDA_VISIBLE_DEVICES=<one-free-gpu> \
  /local_data/yz11445/tools/bin/isaacsim-4.5-python assembly_pipeline.py launch \
  --assembly 04_intermediate/assemblies/<object-id>/layout.json \
  --stream
```

Connect the Windows Isaac Sim WebRTC Streaming Client to
`128.238.176.100`. The host must permit TCP `49100` and UDP `47998` from the
client network. WebRTC is unencrypted and unauthenticated, so use it only over
the university network or VPN and do not expose those ports broadly to the
Internet. The first launch compiles RTX shaders and can take several minutes.

Schema-version-2 `layout.json` records provenance, normalization, one common voxel grid, occupied cells, relative NPZ paths, canonical GT/goal poses, and separate world-space staging poses. Visuals are plain voxel cubes without studs. Isaac creates exact merged-box colliders from the same cells.

Generic scenes include a flat central grid assembly pad, not a LEGO baseplate. Staging is bilateral: fragments occupy front-to-back (`Y`) lanes at the left and right sides of the pad, perpendicular to the arm-to-arm axis. The generated `layout.json` records the chosen lane assignment and staging poses, so Isaac needs no source-dataset access.

The original skull scene remains available through the existing no-override command. `03_scripts/asm_bricks.py` remains a CRAG ground-truth GLB compatibility entry point; predicted-pose input is intentionally outside the generalized pipeline.

# FANUC dual-arm skull assembly: data bundle

**中文概要**：这是研究陈述网页里 "Physical assembly" 演示（两台 FANUC CRX-10iA/L 机械臂加 Robotiq 2F-85 夹爪，把 CRAG 的长臂猿颅骨 8 块碎片拼回去）用到的全部数据：CRAG 原始结果、机器人模型、脚本、中间文件和成片。演示是运动学故事板：碎片位姿来自 CRAG，机械臂走逆运动学，没有物理仿真，Isaac Sim 只负责渲染。

This folder collects everything behind the "Physical assembly" working demo on the *Self-Improving Embodied Intelligence* research page: two FANUC CRX-10iA/L arms with Robotiq 2F-85 grippers rebuild a gibbon cranium broken into eight fragments, each fragment converted into a LEGO-style brick with studs.

The demo is a **kinematic storyboard**. Fragment shapes and poses come from CRAG, the arms follow inverse kinematics on an offline forward-kinematics model probed from Isaac Sim, grasped bricks ride rigidly with the gripper, and Isaac Sim 4.5 only renders. Nothing here is a physics or real-robot result.

- Built: 2026-09-14 (bundled 2026-09-23)
- Source repository: `jingz6676/research-statement` (private), code in `isaac/`, bundle made at commit `3648b8a`
- Local copy of this bundle: `/local_data/jz6676/data/fanuc_crag_skull_assembly/` on maxwell

## Contents

| Folder | What | Size |
|---|---|---|
| `01_crag_case/` | CRAG result for the skull, unchanged | 103 MB |
| `02_robot_assets/` | Isaac Sim 4.5 robot models | 9.4 MB |
| `03_scripts/` | Pipeline scripts | 60 KB |
| `04_intermediate/` | Probe, bricks, timeline, logs | 3.3 MB |
| `05_outputs/` | Video, poster, offline page, rendered frames | 449 MB |

`MANIFEST.sha256` lists a checksum for every file.

### 01_crag_case: the fragments

Object: `Hoolock-hoolock-f-AMNH-83425-cranium-einscan`, a hoolock gibbon cranium scanned from the AMNH collection and distributed through MorphoSource, from CRAG's `morphosource_v2` set, `wo_img` setting (no reference image), 8 pieces, no dropped pieces.

Copied from `/local_data/public/CRAG/results_jan_27/CRAG/wo_img/morphosource_v2/Hoolock-hoolock-f-AMNH-83425-cranium-einscan/` on maxwell.

| File | Used | Role |
|---|---|---|
| `view_gt.glb` | yes | Ground-truth assembly: fragment shapes and final poses |
| `view_assembly_-1.glb` | yes | CRAG's predicted assembly: the translucent "predicted place" ghosts |
| `view_input.glb` | no | Scrambled input fragments |
| `view_-1.glb` | no | Other CRAG output view |
| `view_-1.json` | no | CRAG metrics for this case: part accuracy 1.0, rmse_t 0.011, rmse_r 8.8° |

The same skull also exists under CRAG's `wo_img_missing` setting with different files and part accuracy 0.83. An earlier, superseded brick build used that copy; the final demo uses `wo_img`, as recorded in `04_intermediate/bricks/layout.json` (`source`, `pred_source`).

### 02_robot_assets: robot models

From the Isaac Sim 4.5 asset library, public bucket
`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/`.

- `Fanuc/CRX10IAL/crx10ial.usd`: FANUC CRX-10iA/L arm, 6 DoF.
- `Robotiq/2F-85/Robotiq_2F_85_flattened.usd`: Robotiq 2F-85 gripper, the file the demo loads. The other USDs, `parts/` and `materials/` are the rest of the same asset folder.

The table, the base plate and the bricks are generated by the scripts; there is no external table model.

### 03_scripts: pipeline

Order of use:

1. `fanuc_probe.py` reads the arm and gripper joints from Isaac Sim and writes `fanuc_probe.json`.
2. `asm_bricks.py` voxelises each CRAG fragment into bricks (16 mm pitch), picks a stable resting orientation, computes grasps, and writes `layout.json` plus one `.npz` mesh per brick.
3. `asm_choreo.py` plans the timeline: pick order, table layout, dual-arm reach and IK, a scripted wrong-pose attempt (CRAG's 180° misprediction of the last piece), re-plan and insert. Writes `choreo.json`.
4. `asm_replay.py` replays `choreo.json` in Isaac Sim and renders PNG frames.
5. `asm_kin.py` is the shared library: arm FK and IK, and the 2F-85 mimic-joint model.

Commands used, run from the directory that holds the scripts, with Isaac Sim 4.5 in the conda env `DWAM` and `OMNI_KIT_ACCEPT_EULA=YES`:

```bash
A=../02_robot_assets
python fanuc_probe.py $A/Fanuc/CRX10IAL/crx10ial.usd $A/Robotiq/2F-85/Robotiq_2F_85_flattened.usd asm/probe
python asm_bricks.py ../01_crag_case/view_gt.glb asm/bricks --pred ../01_crag_case/view_assembly_-1.glb --length 0.40 --pitch 0.016
python asm_choreo.py asm/bricks/layout.json asm/probe/fanuc_probe.json asm/choreo.json
python asm_replay.py asm/choreo.json asm/frames --subframes 5
ffmpeg -framerate 30 -i asm/frames/frame_%05d.png -vf scale=800:450:flags=lanczos -c:v libx264 -preset slow -crf 23 -pix_fmt yuv420p -movflags +faststart asm_isaac.mp4
```

The `asm_bricks.py` line is reconstructed from `layout.json` (source files and 16 mm pitch); `--length 0.40` is the script default.

Before re-running, edit the `SCRATCH` path at the top of `asm_choreo.py` (line 12). It points at the temporary copy of the robot assets used on 2026-09-14; set it to this bundle's `02_robot_assets`. The same path is stored in `choreo.json` under `assets`.

Isaac Sim peaked at about 6 GB RSS during rendering; `asm_replay.py` aborts itself above 30 GB.

### 04_intermediate: generated data

- `probe/fanuc_probe.json`: arm and gripper DoF names, limits and joint frames; `probe/probe_pose.png` is a check render.
- `bricks/layout.json`: brick grid (28 × 22 × 22 cells of 16 mm), per-piece cells, grasps, table placement, source file paths. `bricks/piece_0.npz` … `piece_7.npz` are the brick meshes, `ghost.npz` the predicted-pose ghosts, `plate.npz` the studded base plate.
- `choreo.json`: per-frame timeline, 1375 frames at 30 fps (45.8 s): joint angles for both arms, gripper opening, brick and ghost poses, scene labels, captions and events (which piece is seated when).
- `logs/`: output of the probe, choreography and final render runs.

### 05_outputs: results

- `asm_isaac.mp4`: the demo video, 800 × 450, H.264, 1375 frames, 45.8 s.
- `asm_isaac_poster.jpg`: poster frame.
- `Physical_Assembly.html`: standalone page with the video, scene buttons and timeline, all inlined in one file. It plays in Chrome; some browsers (likely Safari) may not play video embedded this way.
- `frames_png.tar`: the 1375 PNG frames (960 × 540) rendered by `asm_replay.py`, plus `scenes.json` (frame ranges per scene) and `_sheet.png` (contact sheet). Unpack with `tar -xf frames_png.tar`.

## Related work

- CRAG · ICML 2026 (fragment data, predicted assembly): https://ai4ce.github.io/CRAG/
- GARF · ICCV 2025 (fracture-aware reassembly): https://ai4ce.github.io/GARF/

## Usage note

The cranium scan comes from the AMNH collection via MorphoSource and keeps MorphoSource's terms of use. The robot models keep NVIDIA's Isaac Sim asset terms. Check both before sharing this bundle outside the group.
