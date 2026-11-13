"""Interpenetration check for authored hand motion, on MHR's own collision capsules (owner, 2026-11-12: renders
showed hands passing through each other and forearms inside the belly).

Arm against body: MHR's 77 tapered capsules (`Character.collision_geometry`: parent joint, local 4x4 transform,
length along the local x axis, a radius at each end), scaled with the body; each forearm and hand against spine,
pelvis, neck and head (not the thighs, which hands and forearms rest on). They are coarser than the mesh: hands
resting in the lap read 2.5 cm, forearms inside the belly the owner flagged 6.6-13.9 cm; BODY_TOL_CM sits between.
Hand against hand: MHR's palm capsules are too thick (a palm almost 5 cm deep: touching palms read 3 cm), so the
two arms are checked with finer capsules on the joints (fingers 0.7-0.95 cm, palm 1.3, forearm 3; `_segments`).
`penetration(mp)`: per frame the deepest overlap (cm, <= 0: apart) arm-arm and arm-body. Touching is allowed;
contacts are placed surface to surface, so overlaps above a few millimetres are errors.
"""

import numpy as np

from mtcore.convert import qrot
from mtcore.mhr import positions, skeleton_state

from .skeleton import MP21

ARM = ("uparm", "lowarm", "wrist", "thumb", "index", "middle", "ring", "pinky")
ARM_VS_BODY = ("lowarm", "wrist", "thumb", "index", "middle", "ring", "pinky")
BODY = ("c_spine", "root", "c_neck", "c_head")  # not the thighs: hands and forearms rest on them
BODY_TOL_CM = 3.0
FINGER_R = {"thumb": 0.95, "index": 0.85, "middle": 0.85, "ring": 0.8, "pinky": 0.72}
PALM_R, FOREARM_R = 1.3, 3.0


def _segments(side):
    """[(joint a, joint b, radius)] of one arm's capsules (MHR joint names)."""
    j = lambda k: f"{side}_{MP21[k]}"
    segs = []
    for f, base in (("thumb", 1), ("index", 5), ("middle", 9), ("ring", 13), ("pinky", 17)):
        segs += [(j(base + k), j(base + k + 1), FINGER_R[f]) for k in range(3)]
    segs += [(j(0), j(5), PALM_R), (j(0), j(9), PALM_R), (j(0), j(17), PALM_R), (j(5), j(17), PALM_R),
             (j(0), j(1), PALM_R)]
    segs.append((f"{side}_lowarm", f"{side}_wrist", FOREARM_R))
    return segs


def seg_closest(a1, b1, a2, b2):
    """Closest points between segments a1-b1 and a2-b2 (broadcast, (..., 3)) -> (distance, s, t) with s, t in [0, 1]."""
    d1, d2, r = b1 - a1, b2 - a2, a1 - a2
    a, e, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * r).sum(-1)
    c, b = (d1 * r).sum(-1), (d1 * d2).sum(-1)
    den = np.maximum(a * e - b * b, 1e-12)
    s = np.clip((b * f - c * e) / den, 0, 1)
    t = np.where(e > 1e-12, (b * s + f) / np.maximum(e, 1e-12), 0.0)
    t_c = np.clip(t, 0, 1)
    # second segment clamped (or a point: e = 0, where the first branch has no information): project onto the first
    s = np.where((t < 0) | (t > 1) | (e <= 1e-12), np.clip((b * t_c - c) / np.maximum(a, 1e-12), 0, 1), s)
    return np.linalg.norm(a1 + d1 * s[..., None] - a2 - d2 * t_c[..., None], axis=-1), s, t_c


def seg_dist(a1, b1, a2, b2):
    return seg_closest(a1, b1, a2, b2)[0]


class Capsules:
    """MHR's collision capsules: parent joints, local start points and axes (length included), end radii."""

    def __init__(self, ch):
        jn = list(ch.skeleton.joint_names)
        cg = ch.collision_geometry
        self.parent = np.array([c.parent for c in cg])
        T = np.stack([np.asarray(c.transformation) for c in cg])
        self.start = T[:, :3, 3]
        self.axis = T[:, :3, 0] * np.array([c.length for c in cg])[:, None]
        self.r = np.array([list(c.radius) for c in cg], dtype=np.float64)  # (C, 2)
        self.name = [jn[p] for p in self.parent]

    def world(self, states):
        """Skeleton states (..., J, 8) -> start, end (..., C, 3) and radii (..., C, 2), body scale applied."""
        st = states[..., self.parent, :]
        t, q, s = st[..., :3], st[..., 3:7], st[..., 7:8]
        a = t + s * qrot(q, self.start)
        b = t + s * qrot(q, self.start + self.axis)
        return a, b, self.r * s

    def select(self, prefixes, side=None):
        out = []
        for k, n in enumerate(self.name):
            base = n[2:] if n[:2] in ("l_", "r_") else n
            if side is not None and not n.startswith(f"{side}_"):
                continue
            if any(base.startswith(p) or n.startswith(p) for p in prefixes):
                out.append(k)
        return np.array(out)


def overlaps(A, B, ia, ib, a, b, r):
    """Overlap (cm) between capsule sets ia and ib: (..., len(ia), len(ib))."""
    d, s, t = seg_closest(a[..., ia, :][..., :, None, :], b[..., ia, :][..., :, None, :],
                          a[..., ib, :][..., None, :, :], b[..., ib, :][..., None, :, :])
    ra0, ra1 = r[..., ia, 0][..., :, None], r[..., ia, 1][..., :, None]
    rb0, rb1 = r[..., ib, 0][..., None, :], r[..., ib, 1][..., None, :]
    ra = ra0 + (ra1 - ra0) * s
    rb = rb0 + (rb1 - rb0) * t
    return ra + rb - d


class Collider:
    def __init__(self, ch):
        self.ch = ch
        jn = list(ch.skeleton.joint_names)
        # hand against hand: fine capsules on the joints (attributes used by relax.Relaxer)
        self.L, self.R = _segments("l"), _segments("r")
        idx = lambda segs: (np.array([jn.index(a) for a, _, _ in segs]), np.array([jn.index(b) for _, b, _ in segs]),
                            np.array([r for _, _, r in segs]))
        self.l, self.r = idx(self.L), idx(self.R)
        # arm against body: MHR's capsules
        self.caps = Capsules(ch)
        c = self.caps
        self.hand_fore = {s: c.select(ARM_VS_BODY, s) for s in "lr"}
        self.body = c.select(BODY)
        rest = skeleton_state(ch, np.zeros(len(ch.parameter_transform.names)))
        a, b, r = c.world(rest)
        self.mask_body = {s: overlaps(None, None, self.hand_fore[s], self.body, a, b, r) <= 0.0 for s in "lr"}

    def arms(self, P):
        """Joint positions (T, J, 3) cm -> deepest overlap between the two arms (T,)."""
        (la, lb, lr), (ra, rb, rr) = self.l, self.r
        d = seg_dist(P[:, la, None], P[:, lb, None], P[:, None, ra], P[:, None, rb])
        pair = lr[:, None] + rr[None, :] - d
        pair[:, -1, -1] -= 2.0  # forearm-forearm only when deep (thick capsules; crossed arms rest on each other)
        return pair.max((1, 2))

    def penetration(self, mp):
        """mp (T, P) -> (arm-arm overlap (T,), arm-body overlap minus BODY_TOL_CM (T,)) in cm; > 0 is a
        collision in both (the body value is shifted so both use the same threshold)."""
        mp = np.asarray(mp, np.float64)
        states = skeleton_state(self.ch, mp)
        a, b, r = self.caps.world(states)
        body = [np.where(self.mask_body[s], overlaps(None, None, self.hand_fore[s], self.body, a, b, r), -99.0)
                for s in "lr"]
        return self.arms(states[..., :3]), np.maximum(*[x.max((1, 2)) for x in body]) - BODY_TOL_CM

    def capsules_at(self, m, prefixes):
        """Capsules (start, end, radii) whose parent joint matches prefixes, at one pose m (P,)."""
        a, b, r = self.caps.world(skeleton_state(self.ch, np.asarray(m, np.float64)[None]))
        k = self.caps.select(prefixes)
        return a[0, k], b[0, k], r[0, k]

    def body_capsules(self, m):
        """The body capsules (start, end, radii) at one pose m (P,), for the elbow search."""
        a, b, r = self.caps.world(skeleton_state(self.ch, np.asarray(m, np.float64)[None]))
        return a[0, self.body], b[0, self.body], r[0, self.body]


def closest_on_triangles(p, a, b, c):
    """Closest points on triangles (a, b, c) (..., 3) to points p (..., 3) (Ericson, Real-Time Collision Detection
    5.1.5) -> barycentric weights (..., 3)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    v, w = vb / den, vc / den
    bary = np.stack([1 - v - w, v, w], -1)  # inside the face
    sd = lambda x, y: x / np.where(np.abs(y) < 1e-12, 1e-12, y)
    t_bc = sd(d4 - d3, (d4 - d3) + (d5 - d6))
    bary = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], np.stack([0 * t_bc, 1 - t_bc, t_bc], -1), bary)
    t_ac = sd(d2, d2 - d6)
    bary = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], np.stack([1 - t_ac, 0 * t_ac, t_ac], -1), bary)
    t_ab = sd(d1, d1 - d3)
    bary = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], np.stack([1 - t_ab, t_ab, 0 * t_ab], -1), bary)
    one = lambda k: np.broadcast_to(np.eye(3)[k], bary.shape)
    bary = np.where(((d6 >= 0) & (d5 <= d6))[..., None], one(2), bary)
    bary = np.where(((d3 >= 0) & (d4 <= d3))[..., None], one(1), bary)
    bary = np.where(((d1 <= 0) & (d2 <= 0))[..., None], one(0), bary)
    return bary


class BodyMesh:
    """MHR's body mesh (arms and hands moved out of the way) as a signed distance for hand contacts (owner,
    2026-11-13: hands near the body clipped). The capsules only scale with the skeleton, not with the body shape,
    so on heavier bodies hands placed against them ended up inside the belly or thighs; the skinned mesh follows
    the shape. Distance = to the closest point on the triangles around the nearest vertices (the vertices are 1-2
    cm apart: a vertex distance read points on the skin as up to ~1 cm off, with a random sign), signed by the
    vertex normals interpolated there (< 0: inside)."""

    def __init__(self, ch, pn):
        self.ch = ch
        self.faces = np.asarray(ch.mesh.faces, np.int64)
        # arms and hands to MHR's rest (spread to the sides), so their own vertices never count as body
        self.arm = np.array([k for k, n in enumerate(pn) if n[:2] in ("l_", "r_") and any(
            n[2:].startswith(w) for w in ("clavicle", "uparm", "lowarm", "elbow", "wrist") + tuple(f for f in ARM[3:]))])
        # vertex -> its faces (padded with -1)
        nv = int(self.faces.max()) + 1
        lists = [[] for _ in range(nv)]
        for fi, f in enumerate(self.faces):
            for v in f:
                lists[v].append(fi)
        deg = max(len(x) for x in lists)
        self.vf = np.full((nv, deg), -1, np.int64)
        for v, x in enumerate(lists):
            self.vf[v, :len(x)] = x
        self.tree = None

    def set_pose(self, m):
        from scipy.spatial import cKDTree
        m = np.asarray(m, np.float64).copy()
        m[self.arm] = 0.0
        v = np.asarray(self.ch.skin_points(skeleton_state(self.ch, m[None]).astype(np.float32)))[0].astype(np.float64)
        f = self.faces
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        n = np.zeros_like(v)
        for k in range(3):
            np.add.at(n, f[:, k], fn)
        self.v, self.n = v, n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        self.tree = cKDTree(v)

    def closest(self, p, k=4):
        """Points p (..., 3) -> closest skin points (..., 3), interpolated outward normals there (..., 3), distance."""
        p = np.asarray(p, np.float64)
        q = p.reshape(-1, 3)
        _, vi = self.tree.query(q, k=k)
        fi = self.vf[vi].reshape(len(q), -1)  # candidate faces (-1: padding)
        valid = fi >= 0
        F = self.faces[np.where(valid, fi, 0)]  # (n, c, 3)
        A, B, C = self.v[F[..., 0]], self.v[F[..., 1]], self.v[F[..., 2]]
        bary = closest_on_triangles(q[:, None], A, B, C)
        pts = bary[..., 0:1] * A + bary[..., 1:2] * B + bary[..., 2:3] * C
        d = np.linalg.norm(q[:, None] - pts, axis=-1)
        d = np.where(valid, d, np.inf)
        j = np.argmin(d, 1)
        r = np.arange(len(q))
        cpt = pts[r, j]
        nb = (bary[r, j, :, None] * self.n[F[r, j]]).sum(1)
        nb /= np.maximum(np.linalg.norm(nb, axis=-1, keepdims=True), 1e-9)
        return cpt.reshape(p.shape), nb.reshape(p.shape), d[r, j].reshape(p.shape[:-1])

    def distance(self, p):
        """Signed distance (cm) of points p (..., 3) to the body surface."""
        p = np.asarray(p, np.float64)
        c, n, d = self.closest(p)
        s = np.sign(((p - c) * n).sum(-1))
        return d * np.where(s == 0, 1.0, s)

    def normal(self, p):
        """Outward skin normal at the closest skin point to each p (..., 3)."""
        return self.closest(p)[1]

    def smooth_normal(self, p, radius=3.0):
        """Outward skin normal around point p (3,), averaged over the vertices within `radius` cm (a palm-sized
        patch, so one facet doesn't tilt the hand)."""
        k = self.tree.query_ball_point(np.asarray(p, np.float64), radius)
        if not k:
            return self.normal(p)
        n = self.n[k].sum(0)
        return n / np.linalg.norm(n)

    def surface(self, p, out, clear):
        """From p outside the body, move against `out` until p is `clear` cm from the surface. -> point or None."""
        out = out / np.linalg.norm(out)
        if self.distance(p) < clear:
            return None
        for _ in range(240):
            dist = float(self.distance(p))
            if dist <= clear + 0.05:
                return p
            p = p - out * max(0.1, min(0.5, dist - clear))  # finer as it nears the skin
        return None
