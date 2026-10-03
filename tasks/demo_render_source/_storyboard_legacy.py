"""Replay asm/choreo.json (two CRX-10iA/L arms, two Robotiq 2F-85 grippers, LEGO-ified CRAG fragments) in Isaac Sim 4.5 and
render 960x540 frames. Everything is kinematic: arm joints, gripper root poses (from the offline FK), gripper finger joints,
brick poses and the ghost opacity are set per frame. usage: python asm_replay.py <choreo.json> <out_dir> [--quick] [--subframes 6]
"""
import os, json, argparse
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
ap = argparse.ArgumentParser(); ap.add_argument('choreo'); ap.add_argument('out'); ap.add_argument('--quick', action='store_true'); ap.add_argument('--subframes', type=int, default=6)
ap.add_argument('--width', type=int, default=960); ap.add_argument('--height', type=int, default=540); ap.add_argument('--frames', default=None, help='comma list of frame indices to render (stills)')
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
from core.scene.builder import build_workcell
from tasks.demo_render_source.storyboard import storyboard_workcell_config

handles = build_workcell(storyboard_workcell_config(), headless=True)
app, world, stage = handles.app, handles.world, handles.stage
import threading, re as _re
RSS = {"peak_mb": 0}
def _rss_watch():
    while True:
        try:
            mb = int(_re.search(r"VmRSS:\s+(\d+)", open("/proc/self/status").read()).group(1)) // 1024; RSS["peak_mb"] = max(RSS["peak_mb"], mb)
            if mb > 30000: open(os.path.join(args.out, "RSS_ABORT"), "w").write(str(mb)); os._exit(3)
        except Exception: pass
        threading.Event().wait(2)
threading.Thread(target=_rss_watch, daemon=True).start()
import numpy as np
import omni.replicator.core as rep
from pxr import UsdShade, Sdf, Gf, UsdGeom, Vt
from isaacsim.core.prims import XFormPrim
from PIL import Image
from core.robots.controls import gripper_dof_targets
from core.robots.kinematics import MIMIC_2F85

C = json.load(open(args.choreo)); fps = C['fps']; A = C['assets']; here = os.path.dirname(os.path.abspath(args.choreo))
def make_mat(path, rgb, rough=0.5, metal=0.0, opacity=None):
    mp = Sdf.Path(path); m = UsdShade.Material.Define(stage, mp); sh = UsdShade.Shader.Define(stage, mp.AppendChild("shader"))
    sh.CreateIdAttr("UsdPreviewSurface"); sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough); sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    if opacity is not None: sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity); sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface"); return m, sh
def add_mesh(path, npz, mat):
    d = np.load(npz); mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(d['v'].astype(np.float32))); f = d['f'].astype(np.int32)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, dtype=np.int32))); mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr("none"); UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return XFormPrim(path)
import carb.settings; carb.settings.get_settings().set("/rtx/raytracing/fractionalCutoutOpacity", True)
robot_handles = [handles.robots[name] for name in handles.config.robot_names]
arms = [handle.robot for handle in robot_handles]
def set_gripper(i, open_frac):
    handle = robot_handles[i]
    settings = handles.config.robot(handle.name)
    targets = gripper_dof_targets(
        handle.gripper_dof_names,
        open_frac,
        settings.gripper_open_radians,
        settings.gripper_closed_radians,
        MIMIC_2F85,
    )
    handle.robot.set_joint_positions(
        np.asarray([targets[name] for name in handle.gripper_dof_names]),
        joint_indices=np.asarray(handle.gripper_dof_indices),
    )

def author_storyboard_assets(stage, choreography, base_dir):
    UsdGeom.Xform.Define(stage, "/World/Storyboard")
    UsdGeom.Xform.Define(stage, "/World/Storyboard/Bricks")
    UsdGeom.Xform.Define(stage, "/World/Storyboard/Ghosts")
    bricks = {}
    for name, brick in choreography['bricks'].items():
        material, _ = make_mat(f"/World/Looks/{name}", tuple(brick['color']), 0.45)
        bricks[name] = add_mesh(
            f"/World/Storyboard/Bricks/{name}", os.path.join(base_dir, brick['npz']), material
        )
    ghost_mat, ghost_shader = make_mat(
        "/World/Looks/ghost",
        tuple(choreography['ghost']['color']),
        0.6,
        0.0,
        opacity=choreography['ghost']['opacity'],
    )
    ghosts = {
        name: add_mesh(
            f"/World/Storyboard/Ghosts/{name}",
            os.path.join(base_dir, choreography['bricks'][name]['npz']),
            ghost_mat,
        )
        for name in choreography['ghost_pose']
    }
    return bricks, ghosts, ghost_shader

bricks, ghosts, ghost_sh = author_storyboard_assets(stage, C, here)
camera_spec = handles.config.cameras['agent']
EYE = tuple(camera_spec['position']); AT = tuple(camera_spec['look_at'])
rp = rep.create.render_product(handles.cameras.agent_path, (args.width, args.height)); annot = rep.AnnotatorRegistry.get_annotator("rgb"); annot.attach(rp)
for _ in range(4): world.render()  # renderer warm-up
def render_frame(out_idx):
    for _ in range(args.subframes * (4 if out_idx == 0 else 1)): world.render()  # extra settling on the first saved frame
    rgb = np.asarray(annot.get_data()); tries = 0
    while (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[0] != args.height) and tries < 8:  # annotator not ready yet
        world.render(); rgb = np.asarray(annot.get_data()); tries += 1
    Image.fromarray(rgb[..., :3]).save(f"{args.out}/frame_{out_idx:05d}.png")
sel = [int(v) for v in args.frames.split(',')] if args.frames else None
step = 5 if args.quick else 1; ranges = {}; out_idx = 0
for i in range(C['n']):
    if sel is not None and i not in sel: continue
    if sel is None and i % step: continue
    for k, arm in enumerate(arms):
        handle = robot_handles[k]; q = np.array(C['arm_q'][k][i])
        indices = np.asarray(handle.arm_dof_indices)
        arm.set_joint_positions(q, joint_indices=indices)
        arm.set_joint_velocities(np.zeros(6), joint_indices=indices)
        set_gripper(k, C['grip_open'][k][i])
    for name, b in bricks.items():
        p = C['brick_pose'][name][i]; b.set_world_poses(positions=np.array([p[:3]]), orientations=np.array([p[3:]]))
    go = float(C['ghost_opacity'][i])
    if go >= 0.02: ghost_sh.GetInput("opacity").Set(go)
    for name, g in ghosts.items():
        st = C['ghost_pose'][name][i]; gi = UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/Storyboard/Ghosts/{name}"))
        if st[0] and go >= 0.02: gi.MakeVisible(); g.set_world_poses(positions=np.array([st[1:4]]), orientations=np.array([st[4:8]]))
        else: gi.MakeInvisible()
    world.step(render=False)
    if out_idx < 3:
        xc = UsdGeom.XformCache(); fl = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(robot_handles[0].flange_path)).ExtractTranslation(); print("FLANGE CHECK frame", i, [round(v, 4) for v in fl], "fk", [round(v, 4) for v in C['grip_pose'][0][i][:3]], flush=True)
    render_frame(out_idx)
    sc = C['scene'][i]; ranges.setdefault(sc, [out_idx, out_idx])[1] = out_idx; out_idx += 1
    if out_idx % 60 == 0: print("frame", out_idx, sc, "rss", RSS["peak_mb"], flush=True)
json.dump({"fps": fps // step, "scenes": ranges, "frames": out_idx, "rss_peak_mb": RSS["peak_mb"], "cam": EYE, "look": AT}, open(f"{args.out}/scenes.json", "w"), indent=1)
print("DONE", out_idx, ranges, flush=True); app.close()
