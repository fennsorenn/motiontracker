"""Interpenetration check for authored hand motion (owner, 2026-11-12: renders showed hands passing through
each other).

Hands and forearms are approximated by capsules on MHR's joints: finger segments (radius 0.7-0.95 cm),
the palm (wrist and knuckle segments, 1.3 cm), the forearm (elbow -> wrist, 3 cm); the torso by a thin
capsule chain along the spine (10 cm: catches gross hand-into-body cases only; the torso is wider and its
spine joints sit toward the back). `penetration(ch, mp)` gives per frame the deepest overlap (cm) between the
two arms (hand-hand, hand-forearm, forearm-forearm) and between either hand and the torso. Touching is
allowed: contacts are placed surface to surface, so overlaps above a few millimetres are errors.
"""

import numpy as np

from mtcore.mhr import positions

from .skeleton import MP21

FINGER_R = {"thumb": 0.95, "index": 0.85, "middle": 0.85, "ring": 0.8, "pinky": 0.72}
PALM_R, FOREARM_R, TORSO_R = 1.3, 3.0, 10.0
TORSO = ("c_spine0", "c_spine1", "c_spine2", "c_spine3", "c_neck")


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


def seg_dist(a1, b1, a2, b2):
    """Closest distance between segments a1-b1 and a2-b2 (broadcast over leading dims, (..., 3))."""
    d1, d2, r = b1 - a1, b2 - a2, a1 - a2
    a, e, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * r).sum(-1)
    c, b = (d1 * r).sum(-1), (d1 * d2).sum(-1)
    den = np.maximum(a * e - b * b, 1e-12)
    s = np.clip((b * f - c * e) / den, 0, 1)
    t = (b * s + f) / np.maximum(e, 1e-12)
    t_c = np.clip(t, 0, 1)
    s = np.where((t < 0) | (t > 1), np.clip((b * t_c - c) / np.maximum(a, 1e-12), 0, 1), s)
    p1 = a1 + d1 * s[..., None]
    p2 = a2 + d2 * t_c[..., None]
    return np.linalg.norm(p1 - p2, axis=-1)


class Collider:
    def __init__(self, ch):
        self.ch = ch
        jn = list(ch.skeleton.joint_names)
        self.L, self.R = _segments("l"), _segments("r")
        idx = lambda segs: (np.array([jn.index(a) for a, _, _ in segs]), np.array([jn.index(b) for _, b, _ in segs]),
                            np.array([r for _, _, r in segs]))
        self.l, self.r = idx(self.L), idx(self.R)
        t = [jn.index(n) for n in TORSO]
        self.torso = (np.array(t[:-1]), np.array(t[1:]))
        self.hand_only = {s: np.array([not a.endswith("lowarm") for a, _, _ in segs]) for s, segs in (("l", self.L), ("r", self.R))}

    def penetration(self, mp):
        """mp (T, P) -> (arm-arm overlap (T,), hand-torso overlap (T,)) in cm (<= 0: apart)."""
        P = positions(self.ch, np.asarray(mp, np.float32))  # (T, J, 3) cm
        (la, lb, lr), (ra, rb, rr) = self.l, self.r
        d = seg_dist(P[:, la, None], P[:, lb, None], P[:, None, ra], P[:, None, rb])  # (T, nL, nR)
        pair = (lr[:, None] + rr[None, :] - d)
        # forearm-forearm overlap only counts when deep (the capsules are thick; crossed arms rest on each other)
        pair[:, -1, -1] -= 2.0
        ta, tb = self.torso
        torso = []
        for (a, b, r), s in ((self.l, "l"), (self.r, "r")):
            m = self.hand_only[s]
            dt = seg_dist(P[:, a[m], None], P[:, b[m], None], P[:, None, ta], P[:, None, tb])
            torso.append((r[m][:, None] + TORSO_R - dt).max((1, 2)))
        return pair.max((1, 2)), np.maximum(*torso)
