"""Probe the Isaac CRX-10iA/L and Robotiq 2F-85 USD assets: DOFs, joint frames (for an offline FK), flange link, bounds, stills."""
import os, sys, json, math
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 960, "height": 540})
import threading, re as _re
RSS = {"peak_mb": 0}; OUT = sys.argv[3]; os.makedirs(OUT, exist_ok=True)
def _rss_watch():
    while True:
        try:
            mb = int(_re.search(r"VmRSS:\s+(\d+)", open("/proc/self/status").read()).group(1)) // 1024; RSS["peak_mb"] = max(RSS["peak_mb"], mb)
            if mb > 30000: open(os.path.join(OUT, "RSS_ABORT"), "w").write(str(mb)); os._exit(3)
        except Exception: pass
        threading.Event().wait(2)
threading.Thread(target=_rss_watch, daemon=True).start()
import numpy as np
import omni.replicator.core as rep, omni.usd
from pxr import UsdGeom, Usd, UsdPhysics, Gf
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.prims import create_prim
from PIL import Image
ARM, GRIP = sys.argv[1], sys.argv[2]
world = World(stage_units_in_meters=1.0, physics_dt=1/120, rendering_dt=1/30)
add_reference_to_stage(ARM, "/World/arm"); XFormPrim("/World/arm").set_world_poses(positions=np.array([[0, 0, 0]]), orientations=np.array([[1, 0, 0, 0]]))
arm = world.scene.add(Robot(prim_path="/World/arm", name="arm"))
add_reference_to_stage(GRIP, "/World/grip"); XFormPrim("/World/grip").set_world_poses(positions=np.array([[1.2, 0, 0.3]]), orientations=np.array([[1, 0, 0, 0]]))
grip = world.scene.add(Robot(prim_path="/World/grip", name="grip"))
world.scene.add(VisualCuboid("/World/floor", name="floor", position=np.array([0, 0, -0.01]), scale=np.array([4, 4, 0.02]), color=np.array([0.93, 0.92, 0.95])))
rep.create.light(light_type="dome", intensity=800, color=(1.0, 1.0, 1.0)); rep.create.light(light_type="distant", intensity=2500, rotation=(315, 30, 0))
def lookat_quat(eye, target, up=(0, 0, 1)):
    eye, target, up = map(lambda v: np.array(v, dtype=float), (eye, target, up))
    z = eye - target; z /= np.linalg.norm(z); x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    m = np.stack([x, y, z], axis=1); w = math.sqrt(max(0, 1 + m[0,0] + m[1,1] + m[2,2])) / 2
    return np.array([w, (m[2,1] - m[1,2]) / (4*w), (m[0,2] - m[2,0]) / (4*w), (m[1,0] - m[0,1]) / (4*w)])
EYE = (2.2, -2.6, 1.6); AT = (0.6, 0.0, 0.5)
create_prim("/World/cam", "Camera", position=np.array(EYE), orientation=lookat_quat(EYE, AT), attributes={"focalLength": 24.0, "clippingRange": (0.01, 100.0)})
rp = rep.create.render_product("/World/cam", (960, 540)); annot = rep.AnnotatorRegistry.get_annotator("rgb"); annot.attach(rp)
world.reset()
stage = omni.usd.get_context().get_stage(); info = {}
for name, rob, root in [("arm", arm, "/World/arm"), ("grip", grip, "/World/grip")]:
    d = {"dof_names": list(rob.dof_names), "q0": rob.get_joint_positions().round(4).tolist()}
    try: d["limits"] = np.asarray(rob.dof_properties["lower"]).round(4).tolist(), np.asarray(rob.dof_properties["upper"]).round(4).tolist()
    except Exception as e: d["limits_err"] = str(e)[:80]
    joints = []
    for p in stage.Traverse():
        if not p.GetPath().pathString.startswith(root + "/"): continue
        if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint) or p.IsA(UsdPhysics.FixedJoint):
            j = UsdPhysics.Joint(p); e = {"path": p.GetPath().pathString, "type": p.GetTypeName(),
                "body0": [str(t) for t in j.GetBody0Rel().GetTargets()], "body1": [str(t) for t in j.GetBody1Rel().GetTargets()],
                "pos0": list(j.GetLocalPos0Attr().Get()), "rot0": list(j.GetLocalRot0Attr().Get().GetImaginary()) + [j.GetLocalRot0Attr().Get().GetReal()],
                "pos1": list(j.GetLocalPos1Attr().Get()), "rot1": list(j.GetLocalRot1Attr().Get().GetImaginary()) + [j.GetLocalRot1Attr().Get().GetReal()]}
            if p.IsA(UsdPhysics.RevoluteJoint): r = UsdPhysics.RevoluteJoint(p); e["axis"] = r.GetAxisAttr().Get(); e["lower"] = r.GetLowerLimitAttr().Get(); e["upper"] = r.GetUpperLimitAttr().Get()
            if p.IsA(UsdPhysics.PrismaticJoint): r = UsdPhysics.PrismaticJoint(p); e["axis"] = r.GetAxisAttr().Get(); e["lower"] = r.GetLowerLimitAttr().Get(); e["upper"] = r.GetUpperLimitAttr().Get()
            joints.append(e)
    d["joints"] = joints
    links = []
    xc = UsdGeom.XformCache()
    for p in stage.Traverse():
        ps = p.GetPath().pathString
        if ps.startswith(root + "/") and p.HasAPI(UsdPhysics.RigidBodyAPI):
            m = xc.GetLocalToWorldTransform(p); t = m.ExtractTranslation(); q = m.ExtractRotationQuat()
            bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(p).ComputeAlignedRange()
            links.append({"path": ps, "pos": [round(t[0], 4), round(t[1], 4), round(t[2], 4)], "quat_wxyz": [round(q.GetReal(), 4)] + [round(v, 4) for v in q.GetImaginary()], "bounds": [list(bb.GetMin()), list(bb.GetMax())] if not bb.IsEmpty() else None})
    d["links"] = links
    d["mesh_count"] = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and p.GetPath().pathString.startswith(root))
    info[name] = d
# FK check pose for the arm: set q=(0.3,-0.4,0.5,0.2,0.6,0.1) and record link poses again
q = np.array([0.3, -0.4, 0.5, 0.2, 0.6, 0.1][:len(arm.dof_names)]); arm.set_joint_positions(q); arm.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))
gq = np.full(len(grip.dof_names), 0.3); grip.set_joint_positions(gq); grip.get_articulation_controller().apply_action(ArticulationAction(joint_positions=gq))
world.step(render=False)
xc = UsdGeom.XformCache(); chk = {}
for p in stage.Traverse():
    ps = p.GetPath().pathString
    if (ps.startswith("/World/arm/") or ps.startswith("/World/grip/")) and p.HasAPI(UsdPhysics.RigidBodyAPI):
        m = xc.GetLocalToWorldTransform(p); t = m.ExtractTranslation(); qq = m.ExtractRotationQuat(); chk[ps] = {"pos": [round(t[0], 4), round(t[1], 4), round(t[2], 4)], "quat_wxyz": [round(qq.GetReal(), 4)] + [round(v, 4) for v in qq.GetImaginary()]}
info["fk_check"] = {"q_arm": q.tolist(), "q_arm_read": arm.get_joint_positions().round(4).tolist(), "q_grip": gq.tolist(), "q_grip_read": grip.get_joint_positions().round(4).tolist(), "links": chk}
for _ in range(4): world.render()
Image.fromarray(np.asarray(annot.get_data())[..., :3]).save(f"{OUT}/probe_pose.png")
info["rss_peak_mb"] = RSS["peak_mb"]; json.dump(info, open(f"{OUT}/fanuc_probe.json", "w"), indent=1)
app.close()
