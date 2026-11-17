"""Relation-preserving retargeting (prototype, option A in `dev-notes/plans/` discussion 2026-11-14), numpy only.

Plain retargeting copies bone rotations; on an avatar with other proportions the same angles put the hands
elsewhere (contact1 ground truth on AvatarSample_F: palms-together wrists 0.4-5 cm apart instead of 7-11 cm).
Here the hands keep their spatial relations to the body instead, with no contact detection:

- Each body part near the hands is an anchor: a bone axis (joint -> child joint, the head upwards) with a frame
  and a surface profile measured on that body's own skinned mesh: the radius of the surface around the axis per
  position along it and direction. The avatar's frame: x along its rest axis, z forward, carried by the bone's
  normalized global rotation. The source's frame must be the one rotation copy would give the avatar (for MHR:
  `convert.MhrToVrm.globals` times the avatar's rest frame), so that both frames agree exactly under rotation
  copy; the source's bone axis then sits at some angle in it (MHR's curved spine), hence per-anchor axes.
- A hand point (palm centre, finger tips) relates to an anchor by where it is along the axis (relative), its
  direction in the anchor frame and its signed gap to the surface (absolute metres). On the avatar the same
  position, direction and gap are measured from the avatar's surface. Hand to hand: the absolute vectors between
  the hands' points in the upper chest's frame (an object held between them keeps its size).
- Weights fall off with the gap (`weight`); far from everything the rotation copy wins (`FALLBACK_M`). Wrist
  targets from all points (hand orientation kept), two-bone arm IK with the copy's elbow plane; a few rounds
  because the hands depend on each other.
"""

import numpy as np

from .convert import VRM_PARENT
from .lifter_pose import mat_to_quat, quat_to_mat

SIDES = ("left", "right")
TORSO = ("hips", "spine", "chest", "upperChest", "neck", "head")
AXIS_END = {"hips": "spine", "spine": "chest", "chest": "upperChest", "upperChest": "neck", "neck": "head", "head": None}
for _s in SIDES:
    AXIS_END.update({f"{_s}UpperArm": f"{_s}LowerArm", f"{_s}LowerArm": f"{_s}Hand",
                     f"{_s}UpperLeg": f"{_s}LowerLeg", f"{_s}LowerLeg": f"{_s}Foot"})
FWD = np.array([0.0, 0.0, 1.0])

# hand points: name -> bones whose joint positions are averaged
PALM = {s: (f"{s}Hand", f"{s}MiddleProximal") for s in SIDES}
TIPS = {s: (f"{s}IndexDistal", f"{s}MiddleDistal", f"{s}ThumbDistal") for s in SIDES}

REF = "upperChest"  # frame of the hand-hand vectors (the two bodies' world axes need not agree)
FALLOFF_M = 0.30  # anchors further than this from a hand point have no say
HAND_FALLOFF_M = 0.60  # hand to hand reaches further (an object held between the hands)
EPS_M = 0.02  # weight ~ 1 / (gap + EPS)^2
FALLBACK_M = 0.25  # the rotation copy weighs like an anchor this far away


def weight(s, falloff=FALLOFF_M):
    s = np.maximum(s, 0.0)
    return np.clip(1.0 - s / falloff, 0.0, None) ** 2 / (s + EPS_M) ** 2


def anatomical_frame(axis):
    """(..., 3) bone axis -> (..., 3, 3) columns x (axis), y, z (forward, orthogonalized)."""
    x = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    z = FWD - np.sum(FWD * x, -1, keepdims=True) * x
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    return np.stack([x, np.cross(z, x), z], -1)


def _directions(n=64):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)


DIRS = _directions()


X = np.array([1.0, 0.0, 0.0])


def _axis_coords(local, L, axis=X):
    """Points in an anchor frame (..., 3) -> u in [0, 1] along the bone axis (unit, in frame coordinates), unit
    direction from the axis point, distance."""
    u = np.clip((local @ axis) / L, 0.0, 1.0)
    q = local - (u * L)[..., None] * axis
    rho = np.linalg.norm(q, axis=-1)
    return u, q / np.maximum(rho, 1e-9)[..., None], rho


class Profile:
    """Surface radius around an anchor axis per (u, direction), from mesh vertices in the anchor frame."""

    NU, SIGMA_U, KAPPA, PCT = 6, 0.15, 12.0, 90

    def __init__(self, local, L, axis=X):
        self.L = float(L)
        u, d, rho = _axis_coords(local, self.L, axis)
        ub = np.minimum((u * self.NU).astype(int), self.NU - 1)
        db = np.argmax(d @ DIRS.T, -1)
        uc, nc, rc = [], [], []
        for i in range(self.NU):
            for k in range(len(DIRS)):
                m = (ub == i) & (db == k)
                if m.sum() >= 3:
                    uc.append((i + 0.5) / self.NU)
                    nc.append(DIRS[k])
                    rc.append(np.percentile(rho[m], self.PCT))
        if not uc:  # too few vertices for any cell (a sparse body part): one round radius
            r = np.percentile(rho, self.PCT) if len(rho) else 0.0
            uc = [(i + 0.5) / self.NU for i in range(self.NU) for _ in DIRS]
            nc, rc = list(DIRS) * self.NU, [r] * len(uc)
        self.uc, self.nc, self.rc = np.array(uc), np.array(nc), np.array(rc)

    def radius(self, u, d):
        k = np.exp(-((u[..., None] - self.uc) / self.SIGMA_U) ** 2) * np.exp(self.KAPPA * (d @ self.nc.T - 1.0))
        return (k * self.rc).sum(-1) / np.maximum(k.sum(-1), 1e-12)


def profiles(verts, labels, pos, frames, lengths, anchors, axes=None):
    """Profiles for `anchors` from a mesh in some pose: verts (V, 3), labels (V,) bone names, and per bone the
    joint position (3,), anchor frame (3, 3), axis length and axis direction in the frame (default x) in that pose."""
    out = {}
    for a in anchors:
        m = labels == a
        if a == "head":
            m |= np.isin(labels, ("leftEye", "rightEye", "jaw"))
        local = (verts[m] - pos[a]) @ frames[a]
        out[a] = Profile(local, lengths[a], X if axes is None else axes[a])
    return out


class ProfileSet:
    """Several anchors' profiles evaluated at once: radius (T, A) at u (T, A), directions d (T, A, 3)."""

    def __init__(self, prof, anchors):
        ps = [prof[a] for a in anchors]
        n = [len(p.uc) for p in ps]
        assert min(n) > 0, "an anchor without surface profile"
        self.uc = np.concatenate([p.uc for p in ps])
        self.nc = np.concatenate([p.nc for p in ps])
        self.rc = np.concatenate([p.rc for p in ps])
        self.seg = np.repeat(np.arange(len(ps)), n)
        self.starts = np.cumsum([0] + n[:-1])

    def radius(self, u, d):
        e = -((u[:, self.seg] - self.uc) / Profile.SIGMA_U) ** 2
        e += Profile.KAPPA * (np.einsum("tcj,cj->tc", d[:, self.seg], self.nc) - 1.0)
        k = np.exp(e)
        return np.add.reduceat(k * self.rc, self.starts, axis=1) / np.maximum(np.add.reduceat(k, self.starts, axis=1), 1e-12)


def _axis_coords_all(local, L, axis):
    """_axis_coords for A anchors at once: local (T, A, 3), L (A,), axis (A, 3)."""
    u = np.clip(np.sum(local * axis, -1) / L, 0.0, 1.0)
    q = local - (u * L)[..., None] * axis
    rho = np.linalg.norm(q, axis=-1)
    return u, q / np.maximum(rho, 1e-9)[..., None], rho


def head_length(verts, labels, pos, frame):
    """Head axis length: the top of the head above the head joint, along the head's up axis."""
    m = np.isin(labels, ("head", "leftEye", "rightEye"))
    return float(np.max((verts[m] - pos) @ frame[:, 0]))


class Body:
    """Per-frame geometry of one body: positions (T, 3) per bone, anchor frames (T, 3, 3), axis lengths, profiles."""

    def __init__(self, pos, frames, lengths, prof, axes=None):
        self.pos, self.frames, self.lengths, self.prof = pos, frames, lengths, prof
        self.axes = axes or {a: X for a in frames}

    def point(self, names):
        return np.mean([self.pos[n] for n in names], 0)

    def stacked(self, anchors):
        """Anchor positions (T, A, 3), frames (T, A, 3, 3), axis lengths (A,), axes (A, 3)."""
        return (np.stack([self.pos[a] for a in anchors], 1), np.stack([self.frames[a] for a in anchors], 1),
                np.array([self.lengths[a] for a in anchors]), np.stack([self.axes[a] for a in anchors]))

    def hand_points(self, side, tips):
        pts = {"palm": self.point(PALM[side])}
        if tips:
            pts.update({t: self.pos[t] for t in TIPS[side]})
        return pts


# --- avatar (VRM normalized pose) ---

class Avatar:
    """VRM avatar: rest positions (normalized, metres, +Z facing) and skinned mesh -> anchor rest frames, profiles."""

    def __init__(self, rest, verts, labels, anchors):
        self.rest = {b: np.asarray(p, dtype=np.float64) for b, p in rest.items()}
        self.anchors = [a for a in anchors if a in self.rest and (AXIS_END[a] is None or AXIS_END[a] in self.rest)]
        self.rest_frames = {a: anatomical_frame(np.array([0.0, 1.0, 0.0]) if AXIS_END[a] is None
                                                else self.rest[AXIS_END[a]] - self.rest[a]) for a in self.anchors}
        self.lengths = {a: (head_length(verts, labels, self.rest[a], self.rest_frames[a]) if AXIS_END[a] is None
                            else float(np.linalg.norm(self.rest[AXIS_END[a]] - self.rest[a]))) for a in self.anchors}
        self.prof = profiles(verts, labels, self.rest, self.rest_frames, self.lengths, self.anchors)
        # FK only over the bones the solver reads (anchors, axis ends, arm chains, hand points) and their parents
        need = set(self.anchors) | {AXIS_END[a] for a in self.anchors if AXIS_END[a]}
        for s in SIDES:
            need |= {f"{s}UpperArm", f"{s}LowerArm", f"{s}Hand", *PALM[s], *TIPS[s]}
        need = {b for b in need if b in self.rest}
        for b in list(need):
            while VRM_PARENT.get(b) is not None:
                b = VRM_PARENT[b]
                while b is not None and b not in self.rest:  # e.g. no upperChest
                    b = VRM_PARENT.get(b)
                if b is None:
                    break
                need.add(b)
        self.fk = []  # (bone, nearest parent the avatar has or None, rest offset from it), parents first
        done = set()
        while len(done) < len(need):
            for b in need - done:
                p = VRM_PARENT.get(b)
                while p is not None and p not in self.rest:
                    p = VRM_PARENT.get(p)
                if p is None or p in done:
                    self.fk.append((b, p, self.rest[b] - (self.rest[p] if p else 0)))
                    done.add(b)
        self.arm_parent = {s: dict((b, p) for b, p, _ in self.fk)[f"{s}UpperArm"] for s in SIDES}

    def local_mats(self, local):
        """bone -> (T, 4) local quaternions -> bone -> (T, 3, 3), only the bones the FK uses."""
        have = [b for b, _, _ in self.fk if b in local]
        return dict(zip(have, quat_to_mat(np.stack([local[b] for b in have], 1)).transpose(1, 0, 2, 3)))

    def body(self, local, mats=None):
        """local: bone -> (T, 4) normalized local quaternions (missing bones at rest), or mats: bone -> (T, 3, 3)
        local rotation matrices (`local_mats`) -> (Body, global rotation matrices bone -> (T, 3, 3))."""
        Lm = self.local_mats(local) if mats is None else mats
        T = len(next(iter(Lm.values())))
        eye = np.broadcast_to(np.eye(3), (T, 3, 3))
        P, G = {}, {}
        for b, p, off in self.fk:
            L = Lm.get(b, eye)
            if p is None:
                G[b], P[b] = L, np.broadcast_to(off, (T, 3))
            else:
                G[b] = G[p] @ L
                P[b] = P[p] + G[p] @ off
        frames = {a: G[a] @ self.rest_frames[a] for a in self.anchors}
        return Body(P, frames, self.lengths, self.prof), G


# --- solver ---

def _rot_between(a, b):
    """(T, 3) vectors -> (T, 3, 3) minimal rotations taking a's direction to b's."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    v = np.cross(a, b)
    c = np.sum(a * b, -1)
    K = np.zeros(a.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = v[..., 2], -v[..., 1], v[..., 0]
    f = 1.0 / np.maximum(1.0 + c, 1e-6)
    return np.eye(3) + K + K @ K * f[..., None, None]


def arm_ik(P, Gm, side, target):
    """Two-bone IK: positions P, global rotation matrices Gm (bone -> (T, 3, 3)) of the copy pose, wrist target
    (T, 3) -> new global matrices for the upper and lower arm (hand global unchanged)."""
    up, low, hand = f"{side}UpperArm", f"{side}LowerArm", f"{side}Hand"
    S, E, W = P[up], P[low], P[hand]
    a = np.linalg.norm(E - S, axis=-1)
    b = np.linalg.norm(W - E, axis=-1)
    d = target - S
    dist = np.clip(np.linalg.norm(d, axis=-1), np.abs(a - b) + 1e-3, 0.999 * (a + b))
    dh = d / np.linalg.norm(d, axis=-1, keepdims=True)
    n = (E - S) - np.sum((E - S) * dh, -1, keepdims=True) * dh
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    fallback = np.cross(dh, np.array([0.0, 1.0, 0.0]))  # straight arm: any plane
    n = np.where(nn > 1e-6, n / np.maximum(nn, 1e-9), fallback / np.maximum(np.linalg.norm(fallback, axis=-1, keepdims=True), 1e-9))
    cos = np.clip((a ** 2 + dist ** 2 - b ** 2) / (2 * a * dist), -1, 1)
    E2 = S + a[:, None] * (cos[:, None] * dh + np.sqrt(1 - cos ** 2)[:, None] * n)
    W2 = S + dh * dist[:, None]
    R1 = _rot_between(E - S, E2 - S)
    G_up = R1 @ Gm[up]
    R2 = _rot_between(np.einsum("tij,tj->ti", R1, W - E), W2 - E2)
    G_low = R2 @ R1 @ Gm[low]
    return G_up, G_low


class RelRetarget:
    def __init__(self, avatar, rounds=6, tips=False):
        self.av, self.rounds, self.tips = avatar, rounds, tips
        self.anchors = list(avatar.anchors)
        self.av_ps = ProfileSet(avatar.prof, self.anchors)
        own = {s: np.array([a in (f"{s}UpperArm", f"{s}LowerArm") for a in self.anchors]) for s in SIDES}
        self.own, self.moving = own, {"left": own["right"], "right": own["left"]}
        self._src_ps = (None, None)

    def _src_profiles(self, src):
        if self._src_ps[0] is not src.prof:
            self._src_ps = (src.prof, ProfileSet(src.prof, self.anchors))
        return self._src_ps[1]

    def relations(self, src, side):
        """Source body -> per hand point: (u, d, gap) per anchor ((T, A), (T, A, 3), (T, A)) and the weights (T, A)
        (zero for the hand's own arm)."""
        sp, sf, sL, sax = src.stacked(self.anchors)
        ps = self._src_profiles(src)
        rel = {}
        for name, h in src.hand_points(side, self.tips).items():
            local = np.einsum("taji,taj->tai", sf, h[:, None] - sp)
            u, d, rho = _axis_coords_all(local, sL, sax)
            gap = rho - ps.radius(u, d)
            w = weight(gap)
            w[:, self.own[side]] = 0.0
            rel[name] = (u, d, gap, w)
        return rel

    def solve(self, src, local):
        """src: the human Body; local: avatar bone -> (T, 4) rotation copy -> new local quats for the arm chains."""
        Lm0 = self.av.local_mats(local)
        body0, Gm0 = self.av.body(local, Lm0)
        rel = {s: self.relations(src, s) for s in SIDES}
        aL = np.array([self.av.lengths[a] for a in self.anchors])
        # the avatar's surface at each relation's (u, d) does not change between rounds
        offs = {s: {n: (u * aL)[..., None] * X + d * (self.av_ps.radius(u, d) + gap)[..., None]
                    for n, (u, d, gap, w) in rel[s].items()} for s in SIDES}
        src_pts = {s: src.hand_points(s, self.tips) for s in SIDES}
        other = {"left": "right", "right": "left"}
        cur = Lm0
        new = {}
        for _ in range(self.rounds):
            body, _ = self.av.body(None, cur)
            ap, af = body.stacked(self.anchors)[:2]
            pts = {s: body.hand_points(s, self.tips) for s in SIDES}
            new = {}
            for s in SIDES:
                wrist = body.pos[f"{s}Hand"]
                num = np.zeros_like(wrist)
                den = np.zeros(len(wrist))
                mv = self.moving[s]
                for name, (u, d, gap, w) in rel[s].items():
                    off = pts[s][name] - wrist  # rigid hand: point relative to the wrist
                    t = ap + np.einsum("taij,taj->tai", af, offs[s][name])
                    t[:, mv] = pts[s][name][:, None] + 0.5 * (t[:, mv] - pts[s][name][:, None])  # the other arm moves too
                    num += np.einsum("ta,tai->ti", w, t - off[:, None])
                    den += w.sum(1)
                    for oname, oh in pts[other[s]].items():  # hand to hand: absolute vectors, in the chest frame
                        v = src_pts[s][name] - src_pts[other[s]][oname]
                        vl = np.einsum("tji,tj->ti", src.frames[REF], v)
                        wh = weight(np.linalg.norm(v, axis=-1), HAND_FALLOFF_M)
                        want = oh + np.einsum("tij,tj->ti", body.frames[REF], vl)  # where this point should be
                        num += wh[:, None] * (pts[s][name] + 0.5 * (want - pts[s][name]) - off)  # half: the other hand moves too
                        den += wh
                    w0 = 1.0 / FALLBACK_M ** 2  # rotation copy
                    num += w0 * body0.pos[f"{s}Hand"]
                    den += w0
                target = num / den[:, None]
                G_up, G_low = arm_ik(body0.pos, Gm0, s, target)
                new[f"{s}UpperArm"] = np.swapaxes(Gm0[self.av.arm_parent[s]], -1, -2) @ G_up
                new[f"{s}LowerArm"] = np.swapaxes(G_up, -1, -2) @ G_low
                new[f"{s}Hand"] = np.swapaxes(G_low, -1, -2) @ Gm0[f"{s}Hand"]
            cur = {**Lm0, **new}
        return {**local, **{b: mat_to_quat(M) for b, M in new.items()}}


# --- live use: the human side from the model's MHR rotations ---

ANCHORS = list(TORSO) + [f"{s}{b}" for s in SIDES for b in ("UpperArm", "LowerArm", "UpperLeg", "LowerLeg")]
HEAD_PARTS = ("leftEye", "rightEye", "jaw")  # their vertices count for the head's profile


class MhrBody:
    """One fixed human body (MHR's mean identity live) posed by MHR world joint rotations (`LifterPose.mhr_globals`):
    the skeleton at rest (J, 8; cm, as in the bundles' skeleton table) and the rest mesh's vertices (metres, same
    world as the rest state) labelled with VRM bones (`mttrain.retarget_asset`). Anchor frames are those rotation
    copy gives the avatar (`conv`: the `MhrToVrm` the copy is made with; `av`: the Avatar), as in
    `mttrain.retarget_demo.human_body`. numpy only."""

    def __init__(self, joint_names, parents, rest_state, verts, labels, av, conv, tips=False):
        from .convert import MAP
        self.conv = conv
        self.anchors = [a for a in av.anchors if a in conv.bones]
        self.rest_frames = av.rest_frames
        rest = np.asarray(rest_state, dtype=np.float64)
        jn = list(joint_names)
        need = set(self.anchors) | {AXIS_END[a] for a in self.anchors if AXIS_END[a]}
        need |= {b for s in SIDES for b in PALM[s] + (TIPS[s] if tips else ())}
        self.j_of = {b: jn.index(MAP[b][0]) for b in need}
        joints = set(self.j_of.values())
        for j in list(joints):
            while parents[j] >= 0:
                j = parents[j]
                joints.add(j)
        rest_g = quat_to_mat(rest[:, 3:7])
        P = rest[:, :3] / 100.0
        # FK steps: joint, parent, rest offset in the parent's rest frame (parents come first in MHR's order)
        self.fk = [(j, parents[j], rest_g[parents[j]].T @ (P[j] - P[parents[j]]) if parents[j] >= 0 else P[j])
                   for j in sorted(joints)]
        # rest geometry -> axis lengths, axes in the anchor frames, surface profiles
        p0 = {b: P[j] for b, j in self.j_of.items()}
        G0 = conv.globals(rest)
        f0 = {a: quat_to_mat(G0[a]) @ av.rest_frames[a] for a in self.anchors}
        verts, labels = np.asarray(verts, dtype=np.float64), np.asarray(labels)
        self.lengths, self.axes = {}, {}
        for a in self.anchors:
            if AXIS_END[a] is None:
                self.axes[a] = X
                self.lengths[a] = head_length(verts, labels, p0[a], f0[a])
            else:
                v = f0[a].T @ (p0[AXIS_END[a]] - p0[a])
                self.lengths[a] = float(np.linalg.norm(v))
                self.axes[a] = v / self.lengths[a]
        self.prof = profiles(verts, labels, p0, f0, self.lengths, self.anchors, self.axes)

    def body(self, G):
        """G: (T, J, 3, 3) MHR world rotations -> Body."""
        P = {}
        for j, p, off in self.fk:
            P[j] = np.broadcast_to(off, G.shape[:1] + (3,)) if p < 0 else P[p] + np.einsum("tij,j->ti", G[:, p], off)
        state = np.zeros(G.shape[:2] + (8,))
        state[..., 3:7] = mat_to_quat(G)
        Gv = self.conv.globals(state)
        frames = {a: quat_to_mat(Gv[a]) @ self.rest_frames[a] for a in self.anchors}
        return Body({b: P[j] for b, j in self.j_of.items()}, frames, self.lengths, self.prof, self.axes)


class LiveRetarget:
    """Rotation copy -> relational retarget for one avatar, per frame: MHR world rotations and the copy's VRM locals
    in, the locals with the arm chains (upper arm, lower arm, hand) replaced out."""

    def __init__(self, skeleton, verts, labels, rest, avatar_verts, avatar_labels, conv, anchors=ANCHORS,
                 tips=False, rounds=6):
        """skeleton: the bundle's skeleton table (joint_names, parents, rest_state); verts / labels: the human rest
        mesh (`retarget.npz`); rest, avatar_verts / avatar_labels: the avatar's rest positions and mesh
        (`vrm.load_vrm`, `vrm.load_vrm_mesh`); conv: the copy's `MhrToVrm`; anchors: body parts the hands relate
        to (drop the legs when the receiver doesn't get leg bones)."""
        self.av = Avatar(rest, avatar_verts, avatar_labels, [a for a in anchors if a in conv.bones])
        self.src = MhrBody(skeleton["joint_names"], skeleton["parents"], skeleton["rest_state"], verts, labels,
                           self.av, conv, tips)
        self.rr = RelRetarget(self.av, rounds=rounds, tips=tips)

    def __call__(self, G, local):
        """G: (T, J, 3, 3); local: bone -> (T, 4) -> bone -> (T, 4) (same bones)."""
        out = self.rr.solve(self.src.body(G), local)
        return {b: out[b] for b in local}
