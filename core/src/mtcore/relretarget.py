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

from .convert import VRM_PARENT, vrm_fk
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

    def body(self, local):
        """local: bone -> (T, 4) normalized local quaternions (missing bones at rest) -> (Body, global quats)."""
        T = len(next(iter(local.values())))
        full = {b: local.get(b, np.tile([0, 0, 0, 1.0], (T, 1))) for b in self.rest}
        P, G = vrm_fk(full, self.rest)
        P = {b: np.broadcast_to(p, (T, 3)) for b, p in P.items()}
        G = {b: np.broadcast_to(q, (T, 4)) for b, q in G.items()}
        frames = {a: quat_to_mat(G[a]) @ self.rest_frames[a] for a in self.anchors}
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

    def relations(self, src, side):
        """Source body -> per hand point: list of (anchor, u, d, gap) and the hand-hand vectors."""
        own = {f"{side}UpperArm", f"{side}LowerArm"}
        rel = {}
        for name, h in src.hand_points(side, self.tips).items():
            rel[name] = []
            for a in self.av.anchors:
                if a in own:
                    continue
                local = np.einsum("tji,tj->ti", src.frames[a], h - src.pos[a])
                u, d, rho = _axis_coords(local, src.lengths[a], src.axes[a])
                rel[name].append((a, u, d, rho - src.prof[a].radius(u, d)))
        return rel

    def solve(self, src, local):
        """src: the human Body; local: avatar bone -> (T, 4) rotation copy -> new local quats for the arm chains."""
        body0, G0 = self.av.body(local)
        Gm0 = {b: quat_to_mat(q) for b, q in G0.items()}
        rel = {s: self.relations(src, s) for s in SIDES}
        src_pts = {s: src.hand_points(s, self.tips) for s in SIDES}
        other = {"left": "right", "right": "left"}
        cur = dict(local)
        for _ in range(self.rounds):
            body, G = self.av.body(cur)
            pts = {s: body.hand_points(s, self.tips) for s in SIDES}
            new = {}
            for s in SIDES:
                wrist = body.pos[f"{s}Hand"]
                moving = {f"{other[s]}UpperArm", f"{other[s]}LowerArm"}
                num = np.zeros_like(wrist)
                den = np.zeros(len(wrist))
                for name, items in rel[s].items():
                    off = pts[s][name] - wrist  # rigid hand: point relative to the wrist
                    for a, u, d, gap in items:
                        t = body.pos[a] + np.einsum("tij,tj->ti", body.frames[a],
                                                    np.stack([u * body.lengths[a], 0 * u, 0 * u], -1)
                                                    + d * (body.prof[a].radius(u, d) + gap)[:, None])
                        if a in moving:  # the other arm moves too: go half way
                            t = pts[s][name] + 0.5 * (t - pts[s][name])
                        w = weight(gap)
                        num += w[:, None] * (t - off)
                        den += w
                    for oname, oh in pts[other[s]].items():  # hand to hand: absolute vectors, in the chest frame
                        v = src_pts[s][name] - src_pts[other[s]][oname]
                        vl = np.einsum("tji,tj->ti", src.frames[REF], v)
                        w = weight(np.linalg.norm(v, axis=-1), HAND_FALLOFF_M)
                        want = oh + np.einsum("tij,tj->ti", body.frames[REF], vl)  # where this point should be
                        num += w[:, None] * (pts[s][name] + 0.5 * (want - pts[s][name]) - off)  # half: the other hand moves too
                        den += w
                    w0 = 1.0 / FALLBACK_M ** 2  # rotation copy
                    num += w0 * body0.pos[f"{s}Hand"]
                    den += w0
                target = num / den[:, None]
                G_up, G_low = arm_ik(body0.pos, Gm0, s, target)
                parent = quat_to_mat(G0[VRM_PARENT[f"{s}UpperArm"]])
                new[f"{s}UpperArm"] = mat_to_quat(np.swapaxes(parent, -1, -2) @ G_up)
                new[f"{s}LowerArm"] = mat_to_quat(np.swapaxes(G_up, -1, -2) @ G_low)
                new[f"{s}Hand"] = mat_to_quat(np.swapaxes(G_low, -1, -2) @ Gm0[f"{s}Hand"])
            cur = {**local, **new}
        return cur
