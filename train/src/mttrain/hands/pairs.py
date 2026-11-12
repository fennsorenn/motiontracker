"""Two-hand interaction choreography for rendered hand sessions and training (plan m4-hand-model, v2).

Mirrors the owner's `hands-v1` recording protocol as procedural motion: palms together, fingertips
together (steeple), index fingers tapping, one hand's index finger touching the other hand's fingers,
rubbing, clapping, crossing, one hand in front of the other, index fingers approaching; plus single-hand
palm turns (slow and fast) and wrist bends. Between events the hands drift to neutral poses in front of
the chest.

Each key pose places a hand by its finger pose `q` (MHR finger parameters), a direction along the hand
`d`, the palm normal `n` (the side the fingers curl toward) and one anchor: a hand point (MediaPipe index,
`skeleton.MP21`) that must land on a world point, e.g. a fingertip on the other hand's fingertip. The
hand's own geometry for that finger pose gives the other points; MHR's IK (`choreo._solve`) moves the arms
(clavicle, arm, wrist) to put wrist and knuckles there, fingertip anchors weighted up. Keys whose solve
misses are dropped: contact anchors > MAX_RESIDUAL_CM, wrist or middle knuckle > MAX_PLACE_CM, and for events
whose contact depends on it (PALM_CHECKED) the palm direction > MAX_PALM_DEG. Elsewhere a missed palm
direction is fine: the ground truth is the pose the arm reaches, not the target. Arms and
fingers blend between keys with minimum-jerk moves; the fingers get `HandMotion`'s idle motion on top.

Coordinates: centimetres in MHR world, body frame from `choreo._body_frame` (shoulder midpoint, the
person's left, up, forward). The camera is in front, so "palm to the camera" is n = forward.
"""

import numpy as np

from mtcore.mhr import skeleton_state

from .. import choreo
from ..seated import ARM_KEYS, clamp, min_jerk, seat
from .motion import HandMotion
from .skeleton import MP21, PARAMS, UNOBSERVABLE, hand_frame
from .stage import MAX_PALM_DEG, _palm_normal

MAX_RESIDUAL_CM = 2.0  # contact anchors (fingertip on fingertip, palm on palm)
MAX_PLACE_CM = 3.0  # wrist and middle knuckle
# events whose contact depends on the palm direction; elsewhere the reached pose is simply the truth
PALM_CHECKED = {"palms", "rub", "clap", "steeple"}
ORIENT = {0: 1.0, 9: 1.0, 5: 2.0, 17: 2.0}  # wrist, middle, index and little knuckle: weights
TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
EVENTS = {"palms": 1.0, "steeple": 1.0, "tap": 1.0, "count_on_hand": 1.2, "rub": 0.8, "clap": 0.8, "cross": 1.0,
          "occlude": 1.0, "approach": 1.0, "turn": 1.2, "wrist": 1.0}


def _unit(v):
    return v / np.linalg.norm(v)


def _rotate(v, axis, ang):
    """Rodrigues: v rotated by ang (rad) about the unit axis."""
    axis = _unit(axis)
    return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + axis * (axis @ v) * (1 - np.cos(ang))


class PairChoreo:
    def __init__(self, ch, lm, pn, limits, hm=None):
        self.ch, self.lm, self.pn, self.limits = ch, lm, pn, limits
        self.jn = list(ch.skeleton.joint_names)
        self.P = {n: i for i, n in enumerate(pn)}
        self.hm = hm or HandMotion()
        self.lib = self.hm.library
        self.arm = {s: [self.P[f"{s}_{k}"] for k in ARM_KEYS] for s in "lr"}
        self.fing = {s: self.hm.fk.idx[s] for s in "lr"}
        self._off = {}
        self.body = np.zeros(len(pn), np.float32)  # scale parameters of the current body (generate sets it)

    # --- hand placement ----------------------------------------------------------------------------
    def offsets(self, q, side):
        """The hand's 21 points (cm) in its own frame rows (fwd, thumbward-ish, palm-ish) for finger pose q,
        on the current body's proportions (hand size scales with the body)."""
        key = (side, q.tobytes(), self.body.tobytes())
        if key not in self._off:
            from mtcore.mhr import positions
            Pm = positions(self.ch, self.hm.fk.full(q, side, base=self.body))[self.hm.fk.mp21[side]]
            w, R = hand_frame(Pm)
            self._off[key] = (Pm - w) @ R.T
        return self._off[key]

    def place(self, q, side, d, n, anchor, at):
        """World points (21, 3) cm of a hand with finger pose q, along d, palm normal n, point `anchor` at `at`.
        The right hand is the left mirrored: its frame's side and third rows flip (skeleton.hand_frame)."""
        s = 1.0 if side == "l" else -1.0
        d = _unit(d)
        n = _unit(n - (n @ d) * d)
        A = np.stack([d, s * np.cross(n, d), s * n])
        O = self.offsets(q, side)
        W = at - O[anchor] @ A
        return W + O @ A

    def constraints(self, side, pts, anchor=None):
        tg = [(f"{side}_{MP21[k]}", pts[k], w, k in (0, 9)) for k, w in ORIENT.items()]
        if anchor is not None and anchor not in ORIENT:
            tg.append((f"{side}_{MP21[anchor]}", pts[anchor], 3.0, True))
        return tg

    # --- events ------------------------------------------------------------------------------------
    def _frame(self, X):
        sh, lat, up, fwd = choreo._body_frame(X)
        return sh, lat, up, fwd

    def _g(self, name):
        return self.lib[name].copy()

    def event(self, rng, kind, X):
        """A list of keys: (move s, hold s, {side: (q, d, n, anchor, at) | 'rest'}), and a label."""
        sh, lat, up, fwd = self._frame(X)
        C = sh + fwd * rng.uniform(25, 40) - up * rng.uniform(8, 25) + lat * rng.uniform(-5, 5)
        keys = []
        tilt = lambda: fwd * rng.uniform(-0.2, 0.4)
        if kind in ("palms", "rub", "clap"):
            q = self._g("flat_together")
            d = _unit(up + tilt())
            def palms(gap, shift=0.0):
                return {"l": (q, d, -lat, 9, C + lat * gap / 2 + up * shift),
                        "r": (q, d, lat, 9, C - lat * gap / 2 - up * shift)}
            keys.append((rng.uniform(0.5, 1.0), rng.uniform(0.6, 2.0), palms(2.6)))
            if kind == "rub":
                for k in range(int(rng.integers(4, 9))):
                    keys.append((0.25, 0.0, palms(2.6, 3.0 * (1 if k % 2 else -1))))
            if kind == "clap":
                for _ in range(int(rng.integers(3, 6))):
                    keys.append((0.18, 0.05, palms(rng.uniform(14, 22))))
                    keys.append((0.15, 0.08, palms(2.6)))
        elif kind == "steeple":
            q = self._g("relaxed")
            d_l = _unit(up + fwd * 0.3 - lat * 0.35)
            d_r = _unit(up + fwd * 0.3 + lat * 0.35)
            keys.append((rng.uniform(0.6, 1.0), rng.uniform(0.8, 2.0),
                         {"l": (q, d_l, -lat, 12, C + lat * 0.8), "r": (q, d_r, lat, 12, C - lat * 0.8)}))
        elif kind in ("tap", "approach"):
            q = self._g("point")
            n_l, n_r = _unit(-up + fwd * rng.uniform(-0.5, 0.5)), _unit(-up + fwd * rng.uniform(-0.5, 0.5))
            d_l, d_r = _unit(-lat + fwd * 0.15), _unit(lat + fwd * 0.15)
            far = lambda g: {"l": (q, d_l, n_l, 8, C + lat * g / 2), "r": (q, d_r, n_r, 8, C - lat * g / 2)}
            keys.append((rng.uniform(0.6, 1.0), rng.uniform(0.3, 0.8), far(rng.uniform(12, 20))))
            if kind == "tap":
                for _ in range(int(rng.integers(2, 5))):
                    keys.append((0.2, 0.05, far(1.6)))
                    keys.append((0.2, 0.05, far(rng.uniform(4, 8))))
            else:
                keys.append((rng.uniform(1.2, 2.5), rng.uniform(0.3, 1.0), far(rng.uniform(1.8, 3.0))))
                keys.append((rng.uniform(0.8, 1.5), 0.3, far(rng.uniform(14, 22))))
        elif kind == "count_on_hand":
            passive = "l" if rng.random() < 0.5 else "r"
            active = "r" if passive == "l" else "l"
            sp = 1.0 if passive == "l" else -1.0
            qp, qa = self._g("open"), self._g("point")
            Wp = C + lat * sp * rng.uniform(2, 8)
            dp = _unit(up + lat * (-sp) * 0.2)
            np_ = _unit(fwd + lat * (-sp) * rng.uniform(0.0, 0.6))  # palm toward the camera, turned in a little
            pts = self.place(qp, passive, dp, np_, 0, Wp - up * 9)
            spec_p = (qp, dp, np_, 0, Wp - up * 9)
            da = _unit(lat * sp + up * 0.4 + fwd * 0.3)  # pointing at the passive hand
            na = _unit(-up + fwd * 0.3)
            fingers = ["thumb", "index", "middle", "ring", "pinky"]
            if rng.random() < 0.5:
                fingers = fingers[::-1]
            keys.append((rng.uniform(0.6, 1.0), 0.3, {passive: spec_p,
                                                      active: (qa, da, na, 8, pts[TIP["middle"]] + fwd * 6.0)}))
            for f in fingers:
                keys.append((rng.uniform(0.3, 0.5), rng.uniform(0.15, 0.4),
                             {passive: spec_p, active: (qa, da, na, 8, pts[TIP[f]] + np_ * 1.2)}))
        elif kind == "cross":
            front = 1.0 if rng.random() < 0.5 else -1.0
            for _ in range(int(rng.integers(1, 3))):
                ql, qr = (self.lib[self.hm.names[rng.choice(len(self.hm.names), p=self.hm.gesture_p)]] for _ in range(2))
                d = _unit(up + fwd * 0.2)
                off = rng.uniform(4, 12)
                keys.append((rng.uniform(0.6, 1.0), rng.uniform(0.5, 1.5),
                             {"l": (ql, _unit(d - lat * 0.3), _unit(fwd + lat * 0.3), 0, C - lat * off - up * 6 + fwd * 4 * front),
                              "r": (qr, _unit(d + lat * 0.3), _unit(fwd - lat * 0.3), 0, C + lat * off - up * 6 - fwd * 4 * front)}))
                keys.append((rng.uniform(0.5, 0.9), rng.uniform(0.2, 0.6),
                             {"l": (ql, d, fwd, 0, C + lat * off - up * 6), "r": (qr, d, fwd, 0, C - lat * off - up * 6)}))
                front = -front
        elif kind == "occlude":
            front_side = "l" if rng.random() < 0.5 else "r"
            q1, q2 = self._g("open"), self._g("relaxed")
            d = _unit(up + fwd * 0.1)
            x0, x1 = rng.uniform(-14, -6), rng.uniform(6, 14)
            if rng.random() < 0.5:
                x0, x1 = x1, x0
            back = "r" if front_side == "l" else "l"
            Wb = C - up * 6
            spec_b = (q2, d, fwd, 0, Wb)
            keys.append((rng.uniform(0.6, 1.0), 0.3, {back: spec_b, front_side: (q1, d, fwd, 0, Wb + fwd * 9 + lat * x0)}))
            keys.append((rng.uniform(1.2, 2.5), 0.3, {back: spec_b, front_side: (q1, d, fwd, 0, Wb + fwd * 9 + lat * x1)}))
        elif kind == "turn":
            side = "l" if rng.random() < 0.5 else "r"
            sg = 1.0 if side == "l" else -1.0
            q = self._g(["open", "relaxed", "point", "flat_together"][rng.integers(4)])
            W = sh + fwd * rng.uniform(25, 40) - up * rng.uniform(5, 25) + lat * sg * rng.uniform(5, 22)
            d = _unit(up * rng.uniform(0.4, 1.0) + fwd * rng.uniform(0.0, 0.6) - lat * sg * 0.2)
            a = _unit(np.cross(d, up if abs(d @ up) < 0.9 else fwd))
            n0 = _unit(fwd - (fwd @ d) * d)
            fast = rng.random() < 0.5
            for k in range(int(rng.integers(2, 5))):
                n = n0 if k % 2 == 0 else -n0
                keys.append((rng.uniform(0.25, 0.45) if fast else rng.uniform(0.7, 1.2), rng.uniform(0.2, 0.8),
                             {side: (q, d, n, 0, W)}))
        elif kind == "wrist":
            side = "l" if rng.random() < 0.5 else "r"
            sg = 1.0 if side == "l" else -1.0
            q = self._g(["open", "relaxed", "flat_together"][rng.integers(3)])
            W = sh + fwd * rng.uniform(25, 40) - up * rng.uniform(5, 25) + lat * sg * rng.uniform(5, 22)
            d0 = _unit(up + fwd * 0.3 - lat * sg * 0.2)
            n0 = _unit(fwd - (fwd @ d0) * d0)
            t = np.cross(n0, d0)
            for k in range(int(rng.integers(3, 6))):
                ang = np.radians(rng.uniform(25, 60)) * (1 if k % 2 else -1)
                axis = t if rng.random() < 0.6 else n0  # flex/extend (about the knuckle line), or side to side
                d, n = _rotate(d0, axis, ang), _rotate(n0, axis, ang)
                keys.append((rng.uniform(0.4, 0.8), rng.uniform(0.1, 0.5), {side: (q, d, n, 0, W)}))
        return keys

    def neutral(self, rng, X, side):
        """A loose hand pose in front of the chest (between events)."""
        sh, lat, up, fwd = self._frame(X)
        sg = 1.0 if side == "l" else -1.0
        W = sh + fwd * rng.uniform(20, 38) - up * rng.uniform(12, 30) + lat * sg * rng.uniform(4, 20)
        d = _unit(up * rng.uniform(0.2, 1.0) + fwd * rng.uniform(0.0, 0.8) - lat * sg * rng.uniform(0.0, 0.5))
        a = _unit(np.cross(d, fwd))
        phi = rng.uniform(0, 2 * np.pi)
        n = np.cos(phi) * a + np.sin(phi) * np.cross(d, a)
        return (self.hm.key_pose(rng)[0], d, n, 0, W)

    # --- sequences ---------------------------------------------------------------------------------
    def _solve(self, mp_k, specs, palm_check=False):
        """IK for the hands in `specs` at body pose mp_k (P,) -> (arm params per side, finger q per side) or None."""
        m0 = mp_k.copy()
        tg, sides = [], []
        for side, (q, d, n, anchor, at) in specs.items():
            m0[self.fing[side]] = q
            pts = self.place(q, side, d, n, anchor, at)
            tg += self.constraints(side, pts, anchor)
            sides.append(side)
        sol = choreo._solve(self.ch, self.pn, self.jn, m0, tuple(sides), tg)
        prim = [(j, p) for j, p, _, pr in tg if pr]
        res = np.linalg.norm(choreo._joint_pos(self.ch, sol, [j for j, _ in prim], self.jn)
                             - np.stack([p for _, p in prim]), axis=-1)
        limit = np.array([MAX_PLACE_CM if j.endswith(("_wrist", "_middle1")) else MAX_RESIDUAL_CM for j, _ in prim])
        if (res > limit).any():
            return None
        for side in (sides if palm_check else []):
            four = [f"{side}_{j}" for j in ("wrist", "middle1", "index1", "pinky1")]
            want = {j: p for j, p, _, _ in tg}
            got = choreo._joint_pos(self.ch, sol, four, self.jn)
            if np.degrees(np.arccos(np.clip(_palm_normal(*got) @ _palm_normal(*(want[j] for j in four)), -1, 1))) > MAX_PALM_DEG:
                return None
        return {s: (sol[self.arm[s]], specs[s][0]) for s in sides}

    def generate(self, rng, base, T, fps):
        """-> model parameters (T, P) with arms and fingers, segments [{label, variant, t0, t1} ms]."""
        pn = self.pn
        body = np.where([n.startswith("scale_") for n in pn], base, 0.0)
        self.body = body.astype(np.float32)
        pair = tuple(rng.choice(["lap", "desk"], 2))
        mp = seat(rng, pn, np.repeat(body[None], T, 0), fps, self.limits, force_rest=pair, moves=False)
        keys = {s: [(0, 0, mp[0, self.arm[s]].copy(), np.zeros(len(PARAMS)))] for s in "lr"}
        segs = []
        k = int(rng.uniform(0.3, 1.0) * fps)
        cur = {s: (mp[0, self.arm[s]].copy(), np.zeros(len(PARAMS))) for s in "lr"}
        names, w = list(EVENTS), np.array(list(EVENTS.values()))
        while k < T - int(1.0 * fps):
            m = mp[min(k, T - 1)].copy()
            for s in "lr":
                m[self.arm[s]] = cur[s][0]
            X = self.lm(skeleton_state(self.ch, m)[None])[0]
            # between events: both hands to loose neutral poses
            neutral = {s: self.neutral(rng, X, s) for s in "lr"}
            sol = self._solve(m, neutral)
            if sol:
                end = min(T - 1, k + int(rng.uniform(0.6, 1.0) * fps))
                for s in "lr":
                    keys[s].append((k, end, *sol[s]))
                    cur[s] = sol[s]
                k = end + int(rng.uniform(0.2, 1.0) * fps)
            kind = names[rng.choice(len(names), p=w / w.sum())]
            ev = None
            for _ in range(4):
                spec_keys = self.event(rng, kind, X)
                sols = [self._solve(m, specs, kind in PALM_CHECKED) for _, _, specs in spec_keys]
                if all(sols):
                    ev = list(zip(spec_keys, sols))
                    break
            if ev is None:
                continue
            t0 = k
            for (move, hold, _), sol in ev:
                end = min(T - 1, k + max(1, int(move * fps)))
                for s, v in sol.items():
                    keys[s].append((k, end, *v))
                    cur[s] = v
                k = end + int(hold * fps)
                if k >= T - 1:
                    break
            segs.append({"label": "pair" if kind not in ("turn", "wrist") else kind, "variant": kind,
                         "t0": round(t0 / fps * 1000, 1), "t1": round(min(k, T - 1) / fps * 1000, 1)})
        out = mp.copy()
        frames = np.arange(T)
        t = frames / fps
        for s in "lr":
            arm = np.repeat(keys[s][0][2][None], T, 0)
            fq = np.repeat(keys[s][0][3][None], T, 0)
            for (a, b, pa, qa), (_, _, pp, qp) in zip(keys[s][1:], keys[s][:-1]):
                u = min_jerk(np.clip((frames - a) / max(1, b - a), 0, 1))
                after = frames >= a
                arm[after] = pp + (pa - pp) * u[after, None]
                fq[after] = qp + (qa - qp) * u[after, None]
            out[:, self.arm[s]] = arm
            fq = np.clip(fq + self.hm._idle(rng, t), self.hm.lo, self.hm.hi)
            fq[:, [PARAMS.index(p) for p in UNOBSERVABLE]] = 0.0
            out[:, self.fing[s]] = fq
        return clamp(pn, out, self.limits), segs
