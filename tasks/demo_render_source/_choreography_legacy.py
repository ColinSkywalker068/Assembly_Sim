"""Dual-arm LEGO-fragment assembly storyboard: propose -> act (insert fragments into the fixed base piece) -> check (the part
CRAG mispredicted is offered at the predicted pose and does not engage) -> diagnose (re-plan: corrected pose, insert) -> done.
Two CRX-10iA/L arms with Robotiq 2F-85 grippers; all motion uses the core offline FK/IK model; bricks ride with the gripper.
usage: python -m tasks.demo_render_source._choreography_legacy <layout.json> <probe.json> <out.json> [--fps 30]"""
import os, json, math, argparse
import numpy as np
from core.robots.kinematics import Arm, Gripper, T as Tpq, R_to_quat, rotvec
from tasks.demo_render_source.choreography import choreography_workcell_metadata
ap = argparse.ArgumentParser(); ap.add_argument('layout'); ap.add_argument('probe'); ap.add_argument('out'); ap.add_argument('--fps', type=int, default=30); args = ap.parse_args()
L = json.load(open(args.layout)); fps = args.fps; here = os.path.dirname(os.path.abspath(args.layout))
TZ = 0.75; TCP_OFF = 0.135; HOVER = 0.12
WORKCELL = choreography_workcell_metadata(args.probe)
ASSETS = WORKCELL["assets"]
ARMS = WORKCELL["arms"]
def yawq(deg): a = math.radians(deg) / 2; return [math.cos(a), 0, 0, math.sin(a)]
GR = Gripper(args.probe)
arms = []
for a in ARMS:
    A = Arm(args.probe); A.base = Tpq(a["base_pos"], a["base_quat"]); arms.append(A)
ex, ey, ez = L["assembly_extent"]; O = np.array([-ex / 2, 0.10 - ey / 2, TZ])
pieces = L["pieces"]; NP = len(pieces)
anchor = None
movable = list(range(NP))
assert all(pieces[k]["grasp"] for k in movable), "a movable piece has no feasible top-down grasp"
def assembled_pos(pc): return O + np.array([pc["assembled_xy"][0], pc["assembled_xy"][1], 0.0])
# ---- side assignment (by assembled x, balanced) and neat rows on the table, every piece resting on its lowest voxel
side = {k: (0 if assembled_pos(pieces[k])[0] < 0 else 1) for k in movable}
if min(list(side.values()).count(0), list(side.values()).count(1)) == 0:
    for j, k in enumerate(sorted(movable, key=lambda k: assembled_pos(pieces[k])[0])): side[k] = 0 if j < len(movable) / 2 else 1
# the largest fragment is parked at the far-left corner of the table (behind arm A's row) so it never hides the growing assembly
k_big = max(movable, key=lambda k: pieces[k]["voxels"])  # keeps the arm assigned by its assembled position
BIG_START = np.array([-0.22, 0.46, TZ - pieces[k_big]["min_z"]])
start = {k_big: BIG_START}
for s in (0, 1):
    ks = sorted([k for k in movable if side[k] == s and k != k_big], key=lambda k: -pieces[k]["voxels"]); gap = 0.06
    widths = [pieces[k]["extent_xy"][1] for k in ks]; total = sum(widths) + gap * (len(ks) - 1); y = -total / 2
    for k, w in zip(ks, widths):
        xr = max(0.42, 0.30 + pieces[k]["extent_xy"][0] / 2); start[k] = np.array([-xr if s == 0 else xr, y + w / 2, TZ - pieces[k]["min_z"]]); y += w + gap
# ---- order: bottom-up (a part is only inserted onto the plate or already placed parts); the part CRAG got most wrong goes last
def pred_err(k): pr = pieces[k].get("pred"); return 0 if not pr else float(np.linalg.norm(pr["centroid_offset"])) + 0.002 * abs(pr["yaw_deg"])
# the part that fails is the one CRAG mispredicted most among those that can be offered from above without passing through others
cand = sorted(movable, key=lambda k: pieces[k].get("capped_frac", 1.0))
top_ok = [k for k in cand if pieces[k].get("capped_frac", 1.0) <= 0.25] or cand[:2]
fail_k = max(top_ok, key=pred_err)
print('fail candidates', [(pieces[k]["name"], round(pieces[k].get("capped_frac", 1.0), 2), round(pred_err(k), 3)) for k in top_ok])
order = sorted([k for k in movable if k != fail_k], key=lambda k: (pieces[k]["min_z"], -pieces[k]["voxels"])) + [fail_k]
fail_arm = side[fail_k]
print('anchor', anchor, 'order', [pieces[k]["name"] for k in order], 'sides', {pieces[k]["name"]: side[k] for k in movable}, 'failing', pieces[fail_k]["name"], 'pred', pieces[fail_k].get("pred"))
# ---- TCP helpers
def tcp_pose(xy, z, closing_deg):
    phi = math.radians(closing_deg); y = np.array([math.cos(phi), math.sin(phi), 0.0]); x = np.array([0, 0, -1.0]); zax = np.cross(x, y)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2] = x, y, zax; M[:3, 3] = [xy[0], xy[1], z]; return M
def flange_from_tcp(M): F = M.copy(); F[:3, 3] = M[:3, 3] - TCP_OFF * M[:3, 0]; return F
def closing(s, phi): return phi + (180.0 if s == 1 else 0.0)
def grasp_tcp(pc, pos, yaw_deg, lift=0.0, s=0):
    g = pc["grasp"]; c, sn = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)); gx, gy = g["xy"]
    return tcp_pose((pos[0] + c * gx - sn * gy, pos[1] + sn * gx + c * gy), pos[2] + g["z_top"] - 0.012 + lift, closing(s, g["angle_deg"] + yaw_deg))
def smooth(t): t = min(max(t, 0.0), 1.0); return 0.5 - 0.5 * math.cos(math.pi * t)
def pose_lerp(A, B, t):
    t = smooth(t); M = np.eye(4); M[:3, 3] = A[:3, 3] + (B[:3, 3] - A[:3, 3]) * t
    rv = rotvec(B[:3, :3] @ A[:3, :3].T); ang = np.linalg.norm(rv)
    if ang < 1e-9: M[:3, :3] = A[:3, :3]; return M
    k = rv / ang; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]); M[:3, :3] = (np.eye(3) + math.sin(ang * t) * K + (1 - math.cos(ang * t)) * K @ K) @ A[:3, :3]; return M
# ---- state
brick_pos = {k: start[k].copy() for k in range(NP)}; brick_yaw = {k: 0.0 for k in range(NP)}; brick_tilt = {k: 0.0 for k in range(NP)}
def brick_M(k):
    M = Tpq(brick_pos[k], yawq(brick_yaw[k]))
    if brick_tilt[k]:
        a = math.radians(brick_tilt[k]); Rt = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]); M[:3, :3] = M[:3, :3] @ Rt
    return M
home_xy = {0: (-0.42, 0.0), 1: (0.42, 0.0)}; home = {s: tcp_pose(home_xy[s], TZ + 0.40, closing(s, 90)) for s in (0, 1)}
def solve_home(arm, target):
    best = None
    for j2 in np.linspace(-1.2, 1.2, 7):
        for j3 in np.linspace(-2.0, 2.0, 9):
            for j5 in np.linspace(-1.6, 1.6, 5):
                qs, ep, er = arm.ik(target, np.array([0.0, j2, j3, 0.0, j5, 0.0]), iters=120)
                if ep < 2e-3 and er < 2e-2:
                    links = arm.fk(qs); cost = (0 if links[2][3, 3] > links[0][3, 3] + 0.15 else 5) + abs(qs[3]) + abs(qs[5]) + 0.3 * abs(qs[0]) + 0.2 * np.abs(qs).sum()
                    if best is None or cost < best[0]: best = (cost, qs)
    assert best is not None, 'home unreachable'; return best[1]
q = [solve_home(arms[s], flange_from_tcp(home[s])) for s in (0, 1)]
for s in (0, 1): print('home ik arm', s, np.round(q[s], 2))
tcp_now = {0: home[0].copy(), 1: home[1].copy()}; grip_open = {0: 1.0, 1: 1.0}; held = {0: None, 1: None}; hold_T = {0: None, 1: None}
rec = dict(scene=[], note=[], arm_q=[[], []], grip_pose=[[], []], grip_open=[[], []], brick_pose={pc["name"]: [] for pc in pieces}, ghost_opacity=[], ghost_pose={pieces[k]["name"]: [] for k in movable}, events=[])
ghost_op = 0.0; ghost_target = None; override = {}; seated = set(); ghost_mode = {}  # piece -> 'pred' | 'gt'
def record(scene, note):
    global ghost_op
    if ghost_target is not None and ghost_op > ghost_target: ghost_op = max(ghost_target, ghost_op - 0.35 / (0.7 * fps))
    for s in (0, 1):
        F = flange_from_tcp(tcp_now[s]); q[s], ep, er = arms[s].ik(F, q[s], iters=120, damping=0.03, step=0.8)
        if ep > 0.01: print('IK residual', scene, s, round(ep, 4), flush=True)
        Fk = arms[s].flange(q[s]); rec['arm_q'][s].append(np.round(q[s], 5).tolist()); rec['grip_pose'][s].append(np.round(np.concatenate([Fk[:3, 3], R_to_quat(Fk[:3, :3])]), 5).tolist()); rec['grip_open'][s].append(round(grip_open[s], 4))
        if held[s] is not None:
            k = held[s]; Mb = (Fk @ np.array([[1, 0, 0, TCP_OFF], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])) @ hold_T[s]
            brick_pos[k] = Mb[:3, 3].copy(); brick_yaw[k] = math.degrees(math.atan2(Mb[1, 0], Mb[0, 0])); override[k] = Mb
    for k, pc in enumerate(pieces):
        M = override.get(k); M = brick_M(k) if M is None else M
        rec['brick_pose'][pc["name"]].append(np.round(np.concatenate([M[:3, 3], R_to_quat(M[:3, :3])]), 5).tolist())
    for k in movable:
        pc = pieces[k]; vis = 1 if (ghost_op > 0.01 and k not in seated) else 0
        if ghost_mode.get(k) == 'pred': gp = assembled_pos(pc) + np.array(pc["pred"]["centroid_offset"]); gy = float(pc["pred"]["yaw_deg"])
        else: gp = assembled_pos(pc); gy = 0.0
        if vis and np.linalg.norm(brick_pos[k] - gp) < 0.035 and abs((brick_yaw[k] - gy + 180) % 360 - 180) < 8: vis = 0
        rec['ghost_pose'][pc["name"]].append([vis] + np.round(np.concatenate([gp, yawq(gy)]), 5).tolist())
    override.clear(); rec['scene'].append(scene); rec['note'].append(note); rec['ghost_opacity'].append(round(ghost_op, 3))
def hold(scene, note, dur):
    for _ in range(int(round(dur * fps))): record(scene, note)
SPEED = 0.78
def move(s, target, dur, scene, note):
    dur *= SPEED; A = tcp_now[s].copy(); nfr = max(1, int(round(dur * fps)))
    for i in range(nfr): tcp_now[s] = pose_lerp(A, target, (i + 1) / nfr); record(scene, note)
def grip(s, to_open, dur, scene, note):
    a = grip_open[s]; nfr = max(1, int(round(dur * fps)))
    for i in range(nfr): grip_open[s] = a + (to_open - a) * smooth((i + 1) / nfr); record(scene, note)
def pick(s, k, scene, note):
    pc = pieces[k]; g = grasp_tcp(pc, brick_pos[k], brick_yaw[k], s=s); pre = g.copy(); pre[2, 3] += HOVER
    move(s, pre, 0.9, scene, note); move(s, g, 0.45, scene, note)
    grip(s, 1.0 - GR.f_for_width(pc["grasp"]["width"]) / 0.8, 0.3, scene, note)
    held[s] = k; hold_T[s] = np.linalg.inv(tcp_now[s]) @ brick_M(k); brick_tilt[k] = 0.0
    lift = g.copy(); lift[2, 3] += HOVER; move(s, lift, 0.45, scene, note)
def carry_to(s, k, target_pos, target_yaw, scene, note, above=HOVER, dur=1.0):
    pre = grasp_tcp(pieces[k], target_pos, target_yaw, lift=above, s=s); move(s, pre, dur, scene, note)
def insert(s, k, target_pos, target_yaw, scene, note_fast, note_slow):
    pc = pieces[k]; near = grasp_tcp(pc, target_pos, target_yaw, lift=0.03, s=s); seat = grasp_tcp(pc, target_pos, target_yaw, s=s)
    move(s, near, 0.35, scene, note_fast); move(s, seat, 0.55, scene, note_slow); hold(scene, note_slow, 0.25)
    held[s] = None; brick_pos[k] = np.array(target_pos, dtype=float); brick_yaw[k] = target_yaw; brick_tilt[k] = 0.0
def release_and_retreat(s, scene, note):
    grip(s, 1.0, 0.25, scene, note); up = tcp_now[s].copy(); up[2, 3] += HOVER; move(s, up, 0.4, scene, note)
def go_home(s, dur, scene, note): move(s, home[s], dur, scene, note)
AR = lambda s: "A" if s == 0 else "B"
# ================= timeline =================
ghost_mode[fail_k] = 'pred'
hold('propose', f'{NP} scanned fragments laid out on the table', 0.8)
for i in range(int(1.0 * fps)): ghost_op = 0.22 * smooth((i + 1) / (1.0 * fps)); record('propose', 'CRAG proposes where every fragment goes (translucent)')
hold('propose', 'predicted placements shown as translucent ghosts', 1.0)
for k in order[:-1]:
    s = side[k]; nm = pieces[k]["name"].replace("_", " ")
    pick(s, k, 'act', f'arm {AR(s)} grasps {nm}'); carry_to(s, k, assembled_pos(pieces[k]), 0.0, 'act', f'arm {AR(s)} carries {nm} over its predicted place')
    insert(s, k, assembled_pos(pieces[k]), 0.0, 'act', f'arm {AR(s)} lowers {nm}', f'arm {AR(s)} presses {nm} onto the studs')
    rec['events'].append({'frame': len(rec['scene']) - 1, 'event': 'seated', 'piece': pieces[k]['name']}); seated.add(k)
    release_and_retreat(s, 'act', 'connector engaged'); go_home(s, 1.0, 'act', 'next fragment')
# check: the mispredicted part is offered at CRAG's predicted pose
k = fail_k; s = fail_arm; pc = pieces[k]; pr = pc.get("pred")
FAIL_SCRIPTED = (pr is None) or (pred_err(k) < 0.008)
if FAIL_SCRIPTED: pr = {"centroid_offset": [0.0, 0.0, 0.0], "yaw_deg": 180.0, "scripted": True}; print('failure is scripted: 180-degree pose hypothesis')
pc["pred"] = pr
pred_pos = assembled_pos(pc) + np.array(pr["centroid_offset"]); pred_yaw = float(pr["yaw_deg"])
pick(s, k, 'check', f'arm {AR(s)} grasps the last fragment'); carry_to(s, k, pred_pos, pred_yaw, 'check', 'carried to the wrong pose hypothesis (rotated 180°)' if FAIL_SCRIPTED else 'carried to the pose CRAG predicted for it')
pitch = L["pitch"]
def occupied_cells(ks):
    occ = set()
    for kk in ks:
        pcc = pieces[kk]; dx = brick_pos[kk][0] - O[0]; dy = brick_pos[kk][1] - O[1]
        for (i, j, k2) in pcc["cells"]: occ.add((int(round((i + 0.5) * pitch + dx - pcc["assembled_xy"][0]) / pitch - 0.5), int(round((j + 0.5) * pitch + dy - pcc["assembled_xy"][1]) / pitch - 0.5), k2))
    return occ
def piece_cells_at(kk, pos, yaw_deg, lift):
    pcc = pieces[kk]; c_, s_ = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)); out = set()
    for (i, j, k2) in pcc["cells"]:
        lx = (i + 0.5) * pitch - pcc["assembled_xy"][0]; ly = (j + 0.5) * pitch - pcc["assembled_xy"][1]
        wx = pos[0] + c_ * lx - s_ * ly - O[0]; wy = pos[1] + s_ * lx + c_ * ly - O[1]; wz = pos[2] - O[2] + (k2 + 0.5) * pitch + lift
        out.add((int(math.floor(wx / pitch)), int(math.floor(wy / pitch)), int(math.floor(wz / pitch))))
    return out
occ = occupied_cells(sorted(seated)); lift_ok = 0.0
for lift_try in np.arange(0.0, 0.25, pitch / 2):
    if not (piece_cells_at(k, pred_pos, pred_yaw, lift_try) & occ): lift_ok = float(lift_try); break
lift_ok += pitch * 0.6; print('check pose lift above contact:', round(lift_ok, 3))
near = grasp_tcp(pc, pred_pos, pred_yaw, lift=lift_ok, s=s); move(s, near, 0.5, 'check', 'lowered: it meets the assembled parts before any connector engages')
hold('check', 'no connector engages: evidence against the pose hypothesis', 0.9)
rec['events'].append({'frame': len(rec['scene']) - 1, 'event': 'not seated', 'piece': pc['name']})
up = tcp_now[s].copy(); up[2, 3] += HOVER; move(s, up, 0.5, 'check', 'hypotheses: geometry, pose, or execution?')
# diagnose: re-plan to the corrected pose (the other candidate placement) and insert
ghost_mode[fail_k] = 'gt'
carry_to(s, k, assembled_pos(pc), 0.0, 'diagnose', 'diagnosis: pose hypothesis rejected; re-plan with the corrected pose', dur=1.2)
insert(s, k, assembled_pos(pc), 0.0, 'diagnose', 'lower at the corrected pose', 'press: connector engages')
rec['events'].append({'frame': len(rec['scene']) - 1, 'event': 'seated', 'piece': pc['name']}); seated.add(k)
release_and_retreat(s, 'diagnose', 'connector engaged'); go_home(s, 1.0, 'diagnose', 'connector engaged')
for i in range(int(1.2 * fps)): ghost_op = 0.22 * (1 - smooth((i + 1) / (1.2 * fps))); record('done', 'assembled: fracture geometry intact, connectors did the fitting')
hold('done', 'assembled: fracture geometry intact, connectors did the fitting', 0.8)
# ================= write =================
n = len(rec['scene']); ranges = {}
for i, sc in enumerate(rec['scene']): ranges.setdefault(sc, [i, i])[1] = i
rel = lambda f: os.path.relpath(os.path.join(here, f), os.path.dirname(os.path.abspath(args.out)))
out = dict(fps=fps, n=n, scenes=ranges, assets=ASSETS, arms=ARMS, tcp_offset=TCP_OFF,
           table={"pos": [0, 0, TZ - 0.025], "size": [2.1, 1.3, 0.05], "legs": [[-0.95, -0.55, (TZ - 0.05) / 2], [0.95, -0.55, (TZ - 0.05) / 2], [-0.95, 0.55, (TZ - 0.05) / 2], [0.95, 0.55, (TZ - 0.05) / 2]]},
           bricks={pc["name"]: {"npz": rel(pc["npz"]), "color": pc["color"]} for pc in pieces},
           ghost={"per_piece": True, "color": [0.50, 0.55, 0.92], "opacity": 0.22}, fail_scripted=FAIL_SCRIPTED,
           camera={"eye": [1.35, -2.05, 1.72], "look": [0.0, 0.05, 0.78], "focal": 25.0}, order=[pieces[k]["name"] for k in order], anchor=anchor, movable=len(movable), fail_piece=pieces[fail_k]["name"], **rec)
json.dump(out, open(args.out, 'w')); print('frames', n, 'seconds', round(n / fps, 2), ranges)
