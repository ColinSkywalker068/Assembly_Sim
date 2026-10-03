"""Offline kinematics for the Isaac CRX-10iA/L USD: FK built from the USD joint frames dumped by fanuc_probe.py,
validated against Isaac's own link poses, plus damped-least-squares IK. Pure numpy; imported by asm_choreo.py."""
import json, math, numpy as np

def quat_to_R(q):  # wxyz
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)], [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])
def R_to_quat(R):  # wxyz
    t = np.trace(R)
    if t > 0: s = math.sqrt(t + 1) * 2; return np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    i = int(np.argmax(np.diag(R)))
    if i == 0: s = math.sqrt(1 + R[0,0] - R[1,1] - R[2,2]) * 2; return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    if i == 1: s = math.sqrt(1 + R[1,1] - R[0,0] - R[2,2]) * 2; return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    s = math.sqrt(1 + R[2,2] - R[0,0] - R[1,1]) * 2; return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])
def T(p, q):  # 4x4 from pos + quat wxyz
    M = np.eye(4); M[:3, :3] = quat_to_R(q); M[:3, 3] = p; return M
def rot_axis(axis, a):
    c, s = math.cos(a), math.sin(a); M = np.eye(4)
    if axis == 'X': M[1:3, 1:3] = [[c, -s], [s, c]]
    elif axis == 'Y': M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, s, -s, c
    else: M[0:2, 0:2] = [[c, -s], [s, c]]
    return M
def rotvec(R):
    a = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if a < 1e-9: return np.zeros(3)
    return a / (2 * math.sin(a)) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

class Arm:
    """FK for a serial chain read from the probe JSON (PhysX joint convention: child = parent · T(pos0,rot0) · R(q) · T(pos1,rot1)^-1)."""
    def __init__(self, probe_json, dof_names=None):
        d = json.load(open(probe_json))['arm']; self.dof = dof_names or d['dof_names']
        J = {j['path'].split('/')[-1]: j for j in d['joints']}
        self.chain = []
        for n in self.dof:
            j = J[n]; q0 = j['rot0']; q1 = j['rot1']  # stored as xyzw
            self.chain.append((j['axis'], T(j['pos0'], [q0[3], q0[0], q0[1], q0[2]]), np.linalg.inv(T(j['pos1'], [q1[3], q1[0], q1[1], q1[2]]))))
        f = J['joint_6_flange']; self.T_flange = T(f['pos0'], [f['rot0'][3]] + f['rot0'][:3]) @ np.linalg.inv(T(f['pos1'], [f['rot1'][3]] + f['rot1'][:3]))
        lo, hi = d['limits']; self.lo, self.hi = np.array(lo), np.array(hi)
        self.base = np.eye(4)
    def fk(self, q, upto=None):
        M = self.base.copy(); out = []
        for i, (axis, A, Binv) in enumerate(self.chain):
            M = M @ A @ rot_axis(axis, q[i]) @ Binv; out.append(M)
            if upto is not None and i == upto: return M
        return out
    def flange(self, q): return self.fk(q)[-1] @ self.T_flange
    def ik(self, target, q0, iters=80, tol=1e-4, w_rot=0.5, damping=0.02, step=1.0):
        q = np.array(q0, dtype=float)
        for _ in range(iters):
            F = self.flange(q); ep = target[:3, 3] - F[:3, 3]; er = rotvec(target[:3, :3] @ F[:3, :3].T)
            e = np.concatenate([ep, w_rot * er])
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 2e-3: break
            Jm = np.zeros((6, len(q))); h = 1e-5
            for i in range(len(q)):
                dq = q.copy(); dq[i] += h; Fi = self.flange(dq)
                Jm[:3, i] = (Fi[:3, 3] - F[:3, 3]) / h; Jm[3:, i] = w_rot * rotvec(Fi[:3, :3] @ F[:3, :3].T) / h
            dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + damping**2 * np.eye(6), e)
            q = np.clip(q + step * dq, self.lo + 0.02, self.hi - 0.02)
        F = self.flange(q); return q, np.linalg.norm(target[:3, 3] - F[:3, 3]), np.linalg.norm(rotvec(target[:3, :3] @ F[:3, :3].T))

if __name__ == '__main__':
    import sys
    arm = Arm(sys.argv[1]); d = json.load(open(sys.argv[1]))
    q = np.array(d['fk_check']['q_arm']); links = arm.fk(q); names = ['link_1', 'link_2', 'link_3', 'link_4', 'link_5', 'link_6']
    worst = 0
    for n, M in zip(names, links):
        ref = d['fk_check']['links']['/World/arm/' + n]; e = np.linalg.norm(M[:3, 3] - np.array(ref['pos'])); worst = max(worst, e)
        Rq = quat_to_R(ref['quat_wxyz']); er = np.linalg.norm(rotvec(Rq @ M[:3, :3].T)); worst = max(worst, er)
        print(f"{n}: pos err {e:.5f} rot err {er:.5f}")
    F = arm.flange(q); ref = d['fk_check']['links']['/World/arm/flange']; print('flange', np.round(F[:3, 3], 4), ref['pos'], 'x-axis', np.round(F[:3, 0], 3))
    # IK smoke test: tool pointing down at (0.6, -0.2, 0.4) from the arm base
    Rt = np.array([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]]).T  # columns: x=(0,0,-1) approach down, y=(0,1,0), z=(1,0,0)
    tgt = np.eye(4); tgt[:3, :3] = Rt; tgt[:3, 3] = [0.6, -0.2, 0.4]
    qs, ep, er = arm.ik(tgt, np.array([0, 0.3, 0.3, 0, 0.5, 0])); print('IK q', np.round(qs, 3), 'pos err', round(ep, 5), 'rot err', round(er, 5), 'worst fk err', round(worst, 5))

# ---------------- Robotiq 2F-85 parallelogram (from the probe's USD joint frames) ----------------
MIMIC_2F85 = {'finger_joint': 1.0, 'right_outer_knuckle_joint': 1.0, 'left_outer_finger_joint': 0.0, 'right_outer_finger_joint': 0.0,
              'left_inner_finger_joint': -1.0, 'right_inner_finger_joint': 1.0, 'left_inner_finger_knuckle_joint': -1.0, 'right_inner_finger_knuckle_joint': -1.0}
class Gripper:
    """Link transforms of the 2F-85 relative to base_link for a finger angle f (rad, 0 = open, 0.8 = closed), PhysX joint convention."""
    def __init__(self, probe_json, mimic=None):
        GJ = json.load(open(probe_json))['grip']; self.names = GJ['dof_names']; self.mimic = dict(MIMIC_2F85, **(mimic or {}))
        self.joints = {j['path'].split('/')[-1]: j for j in GJ['joints'] if j['path'].split('/')[-1] in self.names}
        self.parent_of = {j['body1'][0].split('/')[-1]: (j['body0'][0].split('/')[-1], nm) for nm, j in self.joints.items()}
        self.links = {l['path'].split('/')[-1]: l for l in GJ['links']}; base = np.array(self.links['base_link']['pos'])
        b = self.links['left_inner_finger']['bounds']; self.pad_pt = np.array([(b[0][0] + b[1][0]) / 2, b[0][1], (b[0][2] + b[1][2]) / 2]) - base  # inner face of the left pad at f=0
        fs = np.linspace(0, 0.8, 33); self.gap_f = fs; self.gap = np.array([self.pad_gap(f) for f in fs])
    def link_T(self, link, f):
        if link == 'base_link': return np.eye(4)
        par, nm = self.parent_of[link]; j = self.joints[nm]; q0 = j['rot0']; q1 = j['rot1']
        A = T(j['pos0'], [q0[3], q0[0], q0[1], q0[2]]); B = T(j['pos1'], [q1[3], q1[0], q1[1], q1[2]])
        return self.link_T(par, f) @ A @ rot_axis(j['axis'], self.mimic.get(nm, 0.0) * f) @ np.linalg.inv(B)
    def pad_gap(self, f):
        M = self.link_T('left_inner_finger', f); p = M[:3, :3] @ self.pad_pt + M[:3, 3]; return 2 * p[1]
    def f_for_width(self, width, squeeze=0.002):
        g = max(float(self.gap.min()), min(float(self.gap.max()), width - squeeze))
        return float(np.interp(g, self.gap[::-1], self.gap_f[::-1]))  # gap decreases with f
