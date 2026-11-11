"""MHR pose -> VRM 1.0 normalized local bone rotations.

For VRM bone b driven by MHR joint j:
    G_vrm(b) = Q_mhr(j) * Q_mhr_rest(j)^-1 * D(b)
Local rotation = G_vrm(parent)^-1 * G_vrm(b) (VRM 1.0 normalized bones have identity rest rotations).

D(b) depends on the bone:
- Torso, neck, head and shoulders (REST_ALIGNED): D = identity, i.e. MHR's rest maps onto the
  avatar's rest and only the performer's rotation away from rest is copied. Both rests are neutral
  upright postures, but MHR's spine has an anatomical S-curve and backward-angled clavicles while
  VRoid spines are nearly straight; matching segment directions here copied that curve onto the
  avatar as a constant hunch (chest +30, upperChest +32 deg forward, neck -32, hips -20 at MHR rest)
  with shoulders turned ~37 deg back (found 2026-11-11 in the avatar comparison video).
- Limbs and fingers: D = the minimal rotation taking b's direction in the avatar's T-pose rest onto
  j's direction in MHR's A-pose rest (the rests differ, so their offset must not be copied). Each
  bone's global rotation is computed independently, so arms point where MHR's do whatever the torso
  and shoulders do; those only move where the arm is attached.
Joints MHR has but VRM lacks (spine0, twist joints, wrist_twist, thumb0, pinky0, extra foot joints)
fold in automatically through the globals.
"""

import numpy as np

# VRM bone: (MHR joint driving its orientation, MHR joint its direction points to[, MHR joint where it sits
# if different from the driving joint])
MAP = {
    "hips": ("root", "c_spine1"),  # direction = where VRM spine sits
    "spine": ("c_spine1", "c_spine2"),
    "chest": ("c_spine2", "c_spine3"),
    "upperChest": ("c_spine3", "c_neck"),
    "neck": ("c_neck", "c_head"),
    "head": ("c_head", None),
    "leftEye": ("l_eye", None),
    "rightEye": ("r_eye", None),
}
for s, m in (("left", "l"), ("right", "r")):
    MAP.update({
        f"{s}Shoulder": (f"{m}_clavicle", f"{m}_uparm"),
        f"{s}UpperArm": (f"{m}_uparm", f"{m}_lowarm"),
        f"{s}LowerArm": (f"{m}_lowarm", f"{m}_wrist"),
        f"{s}Hand": (f"{m}_wrist", f"{m}_middle1"),
        f"{s}UpperLeg": (f"{m}_upleg", f"{m}_lowleg"),
        f"{s}LowerLeg": (f"{m}_lowleg", f"{m}_foot"),
        f"{s}Foot": (f"{m}_subtalar", f"{m}_ball"),
        f"{s}ThumbMetacarpal": (f"{m}_thumb1", f"{m}_thumb2"),
        f"{s}ThumbProximal": (f"{m}_thumb2", f"{m}_thumb3"),
        f"{s}ThumbDistal": (f"{m}_thumb3", f"{m}_thumb_null"),
    })
    for vf, mf in (("Index", "index"), ("Middle", "middle"), ("Ring", "ring"), ("Little", "pinky")):
        MAP.update({
            f"{s}{vf}Proximal": (f"{m}_{mf}1", f"{m}_{mf}2"),
            f"{s}{vf}Intermediate": (f"{m}_{mf}2", f"{m}_{mf}3"),
            f"{s}{vf}Distal": (f"{m}_{mf}3", f"{m}_{mf}_null"),
        })

# bones whose MHR rest maps onto the avatar's rest (D = identity); see the module docstring
REST_ALIGNED = {"hips", "spine", "chest", "upperChest", "neck", "head", "leftShoulder", "rightShoulder"}

# VRM 1.0 humanoid parents (subset that MAP covers)
VRM_PARENT = {"spine": "hips", "chest": "spine", "upperChest": "chest", "neck": "upperChest", "head": "neck",
              "leftEye": "head", "rightEye": "head"}
for s in ("left", "right"):
    VRM_PARENT.update({f"{s}Shoulder": "upperChest", f"{s}UpperArm": f"{s}Shoulder", f"{s}LowerArm": f"{s}UpperArm",
                       f"{s}Hand": f"{s}LowerArm", f"{s}UpperLeg": "hips", f"{s}LowerLeg": f"{s}UpperLeg",
                       f"{s}Foot": f"{s}LowerLeg", f"{s}ThumbMetacarpal": f"{s}Hand",
                       f"{s}ThumbProximal": f"{s}ThumbMetacarpal", f"{s}ThumbDistal": f"{s}ThumbProximal"})
    for f in ("Index", "Middle", "Ring", "Little"):
        VRM_PARENT.update({f"{s}{f}Proximal": f"{s}Hand", f"{s}{f}Intermediate": f"{s}{f}Proximal",
                           f"{s}{f}Distal": f"{s}{f}Intermediate"})
VRM_CHILD = {"hips": "spine", "spine": "chest", "chest": "upperChest", "upperChest": "neck", "neck": "head"}
for s in ("left", "right"):
    VRM_CHILD.update({f"{s}Shoulder": f"{s}UpperArm", f"{s}UpperArm": f"{s}LowerArm", f"{s}LowerArm": f"{s}Hand",
                      f"{s}Hand": f"{s}MiddleProximal", f"{s}UpperLeg": f"{s}LowerLeg", f"{s}LowerLeg": f"{s}Foot",
                      f"{s}Foot": f"{s}Toes", f"{s}ThumbMetacarpal": f"{s}ThumbProximal",
                      f"{s}ThumbProximal": f"{s}ThumbDistal"})
    for f in ("Index", "Middle", "Ring", "Little"):
        VRM_CHILD.update({f"{s}{f}Proximal": f"{s}{f}Intermediate", f"{s}{f}Intermediate": f"{s}{f}Distal"})


# quaternions as [x, y, z, w]
def qmul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def qinv(q):
    return q * np.array([-1, -1, -1, 1.0])


def qrot(q, v):
    p = np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], -1)
    return qmul(qmul(q, p), qinv(q))[..., :3]


def twist(q, axis):
    """Twist component of q about a unit axis (swing-twist decomposition), [x,y,z,w]."""
    p = np.sum(q[..., :3] * axis, -1, keepdims=True) * axis
    t = np.concatenate([p, q[..., 3:4]], -1)
    n = np.linalg.norm(t, axis=-1, keepdims=True)
    return np.where(n > 1e-9, t / np.maximum(n, 1e-12), np.array([0, 0, 0, 1.0]))


def q_between(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    c = np.cross(a, b); w = 1 + np.dot(a, b)
    if w < 1e-8:  # opposite: any perpendicular axis
        axis = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        return np.append(axis / np.linalg.norm(axis), 0.0)
    q = np.append(c, w)
    return q / np.linalg.norm(q)


class MhrToVrm:
    def __init__(self, mhr_joint_names, mhr_rest_state, vrm_rest_positions):
        """mhr_rest_state: (J, 8) skeleton state [t, q, s] at zero pose. vrm_rest_positions: bone -> xyz."""
        self.jn = list(mhr_joint_names)
        self.bones = [b for b in MAP if b in vrm_rest_positions]
        self.q_rest_inv, self.D = {}, {}
        P, Q = mhr_rest_state[:, :3], mhr_rest_state[:, 3:7]
        # forearm twist: wrist_twist's rotation relative to lowarm, twist part about the forearm axis
        # (in lowarm's local frame), goes onto VRM lowerArm. The hand keeps the remaining swing.
        self.forearm = {}
        for side, m in (("left", "l"), ("right", "r")):
            lo, wt, wr = (self.jn.index(f"{m}_{n}") for n in ("lowarm", "wrist_twist", "wrist"))
            axis = qrot(qinv(Q[lo]), P[wr] - P[lo])
            self.forearm[f"{side}LowerArm"] = (lo, wt, axis / np.linalg.norm(axis))
        for b in self.bones:
            j, jd = MAP[b][:2]
            jp = MAP[b][2] if len(MAP[b]) > 2 else j  # joint where the VRM bone sits
            self.q_rest_inv[b] = qinv(Q[self.jn.index(j)])
            vc = VRM_CHILD.get(b)
            if b in REST_ALIGNED or jd is None or vc not in vrm_rest_positions:
                self.D[b] = np.array([0, 0, 0, 1.0])
                continue
            d_mhr = P[self.jn.index(jd)] - P[self.jn.index(jp)]
            d_vrm = vrm_rest_positions[vc] - vrm_rest_positions[b]
            self.D[b] = q_between(d_vrm, d_mhr)

    def globals(self, state):
        Q = state[..., 3:7]
        out = {}
        for b in self.bones:
            Qj = Q[..., self.jn.index(MAP[b][0]), :]
            if b in self.forearm:
                lo, wt, axis = self.forearm[b]
                rel = qmul(qinv(Q[..., lo, :]), Q[..., wt, :])
                Qj = qmul(Q[..., lo, :], twist(rel, axis))
            out[b] = qmul(qmul(Qj, self.q_rest_inv[b]), self.D[b])
        return out

    def locals(self, state):
        G = self.globals(state)
        out = {}
        for b in self.bones:
            p = VRM_PARENT.get(b)
            while p is not None and p not in G:  # e.g. avatar without upperChest
                p = VRM_PARENT.get(p)
            out[b] = G[b] if p is None else qmul(qinv(G[p]), G[b])
        return out


def vrm_fk(local, rest_positions, parent=VRM_PARENT):
    """Normalized VRM FK: rest rotations identity, offsets from rest positions. Returns bone -> world pos."""
    G, P = {}, {}
    order = [b for b in rest_positions if b in local]
    def solve(b):
        if b in P:
            return
        p = parent.get(b)
        while p is not None and p not in local:
            p = parent.get(p)
        if p is None:
            G[b] = local[b]; P[b] = rest_positions[b]
        else:
            solve(p)
            G[b] = qmul(G[p], local[b])
            P[b] = P[p] + qrot(G[p], rest_positions[b] - rest_positions[p])
    for b in order:
        solve(b)
    return P, G
