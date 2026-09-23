"""Replay asm/choreo.json (two CRX-10iA/L arms, two Robotiq 2F-85 grippers, LEGO-ified CRAG fragments) in Isaac Sim 4.5 and
render 960x540 frames. Everything is kinematic: arm joints, gripper root poses (from the offline FK), gripper finger joints,
brick poses and the ghost opacity are set per frame. usage: python asm_replay.py <choreo.json> <out_dir> [--quick] [--subframes 6]
"""
import os, sys, json, math, argparse
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
ap = argparse.ArgumentParser(); ap.add_argument('choreo'); ap.add_argument('out'); ap.add_argument('--quick', action='store_true'); ap.add_argument('--subframes', type=int, default=6)
ap.add_argument('--cam', default=None); ap.add_argument('--look', default=None); ap.add_argument('--focal', type=float, default=None)
ap.add_argument('--width', type=int, default=960); ap.add_argument('--height', type=int, default=540); ap.add_argument('--frames', default=None, help='comma list of frame indices to render (stills)')
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": args.width, "height": args.height})
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
import omni.replicator.core as rep, omni.usd
from pxr import UsdShade, Sdf, Gf, UsdGeom, Vt
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.prims import create_prim
from PIL import Image

C = json.load(open(args.choreo)); fps = C['fps']; A = C['assets']; here = os.path.dirname(os.path.abspath(args.choreo))
world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / fps)
world.get_physics_context().set_gravity(0.0)
stage = omni.usd.get_context().get_stage()
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
# ---- arms; each gripper is referenced UNDER the arm's flange link (rigid mount), with its physics removed
from pxr import UsdPhysics, Usd
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asm_kin import quat_to_R, T as Tpq, rot_axis
import carb.settings; carb.settings.get_settings().set("/rtx/raytracing/fractionalCutoutOpacity", True)
def strip_physics(root_path):
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.Joint) or prim.GetTypeName().startswith('Physics'): prim.SetActive(False); continue
        for api in (UsdPhysics.ArticulationRootAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.MassAPI, UsdPhysics.MeshCollisionAPI):
            if prim.HasAPI(api): prim.RemoveAPI(api)
        for name in list(prim.GetAppliedSchemas()):
            if 'Physx' in name or 'Physics' in name:
                try: prim.RemoveAppliedSchema(name)
                except Exception: pass
from asm_kin import Gripper
GR = Gripper(C['assets']['probe_json']); parent_of = GR.parent_of
def link_T(link, f): return GR.link_T(link, f)
arms, grip_ops = [], []
for i, side in enumerate(C['arms']):
    add_reference_to_stage(A['arm_usd'], f"/World/arm{i}"); XFormPrim(f"/World/arm{i}").set_world_poses(positions=np.array([side['base_pos']]), orientations=np.array([side['base_quat']]))
    arms.append(world.scene.add(Robot(prim_path=f"/World/arm{i}", name=f"arm{i}")))
    gpath = f"/World/arm{i}/flange/gripper"; add_reference_to_stage(A['gripper_usd'], gpath); strip_physics(gpath)
    ops = {}; byname = {pr.GetName(): pr for pr in Usd.PrimRange(stage.GetPrimAtPath(gpath)) if pr.GetName() in parent_of}
    for link in parent_of:
        prim = byname.get(link)
        if prim is None: print("MISSING gripper link", link, flush=True); continue
        xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder(); ops[link] = xf.AddTransformOp()
    left = [pr.GetPath().pathString for pr in Usd.PrimRange(stage.GetPrimAtPath(gpath)) if any('Phys' in a_ for a_ in pr.GetAppliedSchemas())]
    print("gripper", i, "links found", len(ops), "physics schemas left", len(left), flush=True); grip_ops.append(ops)
def set_gripper(i, open_frac):
    f = 0.8 * (1 - open_frac)
    for link, op in grip_ops[i].items(): op.Set(Gf.Matrix4d(*link_T(link, f).T.flatten().tolist()))
# ---- table, plate, bricks, ghost
tb = C['table']; world.scene.add(VisualCuboid("/World/table", name="table", position=np.array(tb['pos']), scale=np.array(tb['size']), color=np.array([0.90, 0.89, 0.92])))
world.scene.add(VisualCuboid("/World/floor", name="floor", position=np.array([0, 0, -0.01]), scale=np.array([8, 8, 0.02]), color=np.array([0.80, 0.80, 0.83])))
for k, leg in enumerate(tb.get('legs', [])): world.scene.add(VisualCuboid(f"/World/leg{k}", name=f"leg{k}", position=np.array(leg), scale=np.array([0.06, 0.06, tb['pos'][2] - tb['size'][2] / 2]), color=np.array([0.55, 0.55, 0.6])))
plate_mat, _ = make_mat("/World/Looks/plate", tuple(C['plate']['color']), 0.55)
add_mesh("/World/plate", os.path.join(here, C['plate']['npz']), plate_mat).set_world_poses(positions=np.array([C['plate']['pos']]), orientations=np.array([[1, 0, 0, 0]]))
bricks = {}
for name, b in C['bricks'].items():
    m, _ = make_mat(f"/World/Looks/{name}", tuple(b['color']), 0.45); bricks[name] = add_mesh(f"/World/{name}", os.path.join(here, b['npz']), m)
ghost_mat, ghost_sh = make_mat("/World/Looks/ghost", tuple(C['ghost']['color']), 0.6, 0.0, opacity=C['ghost']['opacity'])
ghosts = {}
for name in C['ghost_pose']:
    ghosts[name] = add_mesh(f"/World/ghost_{name}", os.path.join(here, C['bricks'][name]['npz']), ghost_mat)
rep.create.light(light_type="dome", intensity=700, color=(1.0, 1.0, 1.0)); rep.create.light(light_type="distant", intensity=2200, rotation=(315, 35, 0))
def lookat_quat(eye, target, up=(0, 0, 1)):
    eye, target, up = map(lambda v: np.array(v, dtype=float), (eye, target, up))
    z = eye - target; z /= np.linalg.norm(z); x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    m = np.stack([x, y, z], axis=1); w = math.sqrt(max(0, 1 + m[0,0] + m[1,1] + m[2,2])) / 2
    return np.array([w, (m[2,1] - m[1,2]) / (4*w), (m[0,2] - m[2,0]) / (4*w), (m[1,0] - m[0,1]) / (4*w)])
cam = C['camera']; EYE = tuple(float(v) for v in args.cam.split(',')) if args.cam else tuple(cam['eye']); AT = tuple(float(v) for v in args.look.split(',')) if args.look else tuple(cam['look'])
create_prim("/World/cam", "Camera", position=np.array(EYE), orientation=lookat_quat(EYE, AT), attributes={"focalLength": args.focal or cam['focal'], "clippingRange": (0.01, 100.0)})
rp = rep.create.render_product("/World/cam", (args.width, args.height)); annot = rep.AnnotatorRegistry.get_annotator("rgb"); annot.attach(rp)
world.reset()
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
        q = np.array(C['arm_q'][k][i]); arm.set_joint_positions(q); arm.set_joint_velocities(np.zeros(6)); arm.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))
        set_gripper(k, C['grip_open'][k][i])
    for name, b in bricks.items():
        p = C['brick_pose'][name][i]; b.set_world_poses(positions=np.array([p[:3]]), orientations=np.array([p[3:]]))
    go = float(C['ghost_opacity'][i])
    if go >= 0.02: ghost_sh.GetInput("opacity").Set(go)
    for name, g in ghosts.items():
        st = C['ghost_pose'][name][i]; gi = UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/ghost_{name}"))
        if st[0] and go >= 0.02: gi.MakeVisible(); g.set_world_poses(positions=np.array([st[1:4]]), orientations=np.array([st[4:8]]))
        else: gi.MakeInvisible()
    world.step(render=False)
    if out_idx < 3:
        xc = UsdGeom.XformCache(); fl = xc.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/arm0/flange")).ExtractTranslation(); print("FLANGE CHECK frame", i, [round(v, 4) for v in fl], "fk", [round(v, 4) for v in C['grip_pose'][0][i][:3]], flush=True)
    render_frame(out_idx)
    sc = C['scene'][i]; ranges.setdefault(sc, [out_idx, out_idx])[1] = out_idx; out_idx += 1
    if out_idx % 60 == 0: print("frame", out_idx, sc, "rss", RSS["peak_mb"], flush=True)
json.dump({"fps": fps // step, "scenes": ranges, "frames": out_idx, "rss_peak_mb": RSS["peak_mb"], "cam": EYE, "look": AT}, open(f"{args.out}/scenes.json", "w"), indent=1)
print("DONE", out_idx, ranges, flush=True); app.close()
