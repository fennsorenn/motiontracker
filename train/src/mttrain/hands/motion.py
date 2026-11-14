"""Procedural finger motion for the hand model (PLAN.md §5.6, owner decision 2026-11-12).

A sequence is a timeline of key poses joined by minimum-jerk transitions:
- key poses come from a gesture library (open, relaxed, fist, point, pinch, OK, counting in index-first
  and thumb-first styles, single fingers, rock, thumbs up, grip, claw, flat bend, splay, typing), perturbed;
  from independent per-finger states (straight, half, curled; owner 2026-11-13: counting read weakly); or
  from coupled random poses (neighbouring fingers curl together, the distal joint follows the middle one,
  curled fingers spread less); now and then a counting routine steps through the counts;
- holds of varying length, transitions of 0.15-0.7 s, each finger slightly early or late;
- idle motion on top: slow drift, small tremor, occasional single-finger taps.

Poses are MHR finger parameters of the canonical (left) hand, `skeleton.PARAMS` order, radians,
within MHR's limits. Finger curl is given as a fraction of the joint's range (1 = its flex limit,
negative = toward its extension limit). Thumb key poses are solved by IK: thumb tip to a target in the
hand frame (`skeleton.hand_frame`), or onto a fingertip for contact gestures.
"""

import numpy as np
from scipy.optimize import least_squares

from .skeleton import FINGERS, PARAMS, THUMB, UNOBSERVABLE, HandFK, hand_frame

P = {p: k for k, p in enumerate(PARAMS)}
# counting routines: gesture names in order (index-first and thumb-first styles), stepped through quickly
COUNTS = (("point", "peace", "three", "four", "open"),
          ("count1_thumb", "count2_thumb", "count3_thumb", "count4_thumb", "open"))
P_ROUTINE = 0.08  # per key pose: start a counting routine (up or down, from a random start)
# key pose sources: gesture library, independent per-finger states, coupled random
P_LIBRARY, P_INDEPENDENT = 0.45, 0.30
THUMB_SOLVED = ("thumb0_ry", "thumb0_rz", "thumb1_ry", "thumb1_rz", "thumb2_rz", "thumb3_rz")
# IK pulls the thumb toward a natural, slightly flexed thumb; the two outer joints hardest, so the
# target is reached mostly from the base (otherwise the solver hooks or hyperextends the thumb tip)
THUMB_NATURAL = np.array([0.0, 0.0, 0.0, 0.15, 0.2, 0.15])
THUMB_STIFF = np.array([0.3, 0.3, 0.6, 0.6, 1.5, 1.5])
SPREAD_FAN = np.array([1.0, 0.15, -0.55, -1.0])  # per finger, index..pinky: direction when the hand splays

# Gestures: per finger (mcp, pip, dip) curl fractions and a fan amount (rad at the index), plus a thumb
# spec: ("target", fwd, thumbward, palm) in cm in the hand frame, or ("touch", landmark, fwd, thumbward,
# palm offset cm) to put the thumb tip next to a landmark of the posed hand. Weights = sampling odds.
_RELAXED = [(0.10, 0.18, 0.12), (0.13, 0.22, 0.14), (0.16, 0.26, 0.16), (0.20, 0.30, 0.18)]
_FIST = [(0.85, 0.95, 0.60)] * 4
_STRAIGHT = [(0.02, 0.03, 0.03)] * 4
GESTURES = {
    "relaxed": dict(w=6, fingers=_RELAXED, fan=0.06, thumb=("target", 8.5, 6.5, 4.0)),
    "open": dict(w=3, fingers=_STRAIGHT, fan=0.12, thumb=("target", 9.0, 7.0, 1.5)),
    "splay": dict(w=1.5, fingers=[(-0.05, 0.0, 0.0)] * 4, fan=0.35, thumb=("target", 6.5, 9.5, 1.0)),
    "flat_together": dict(w=1, fingers=_STRAIGHT, fan=-0.04, thumb=("target", 10.0, 4.0, 1.5)),
    "fist": dict(w=3, fingers=_FIST, fan=0.0, thumb=("touch", 7, 0.0, 0.5, 1.6)),
    "point": dict(w=2, fingers=[(0.02, 0.05, 0.05)] + _FIST[1:], fan=0.0, thumb=("touch", 11, 0.0, 0.5, 1.6)),
    "peace": dict(w=2, fingers=[(0.02, 0.03, 0.03), (0.02, 0.03, 0.03), *_FIST[2:]], fan=0.25,
                  thumb=("touch", 15, 0.0, 0.5, 1.6)),
    "three": dict(w=1.5, fingers=[(0.02, 0.03, 0.03)] * 3 + _FIST[3:], fan=0.15, thumb=("touch", 19, 0.0, 0.3, 1.6)),
    "four": dict(w=1.5, fingers=_STRAIGHT, fan=0.12, thumb=("target", 7.0, -1.5, 3.0)),
    "thumbs_up": dict(w=1, fingers=_FIST, fan=0.0, thumb=("target", 6.0, 9.5, -1.0)),
    "pinch": dict(w=2, fingers=[(0.45, 0.45, 0.30), (0.40, 0.55, 0.35), (0.40, 0.60, 0.35), (0.40, 0.65, 0.35)],
                  fan=0.0, thumb=("touch", 8, 0.0, 0.0, 1.0)),
    "ok": dict(w=1.5, fingers=[(0.45, 0.45, 0.30), (0.05, 0.08, 0.05), (0.08, 0.10, 0.06), (0.10, 0.12, 0.08)],
               fan=0.15, thumb=("touch", 8, 0.0, 0.0, 1.0)),
    "grip": dict(w=2, fingers=[(0.50, 0.60, 0.40), (0.55, 0.65, 0.40), (0.55, 0.65, 0.40), (0.55, 0.65, 0.40)],
                 fan=0.0, thumb=("touch", 7, 0.0, 1.0, 3.5)),
    "claw": dict(w=1, fingers=[(0.0, 0.80, 0.70)] * 4, fan=0.1, thumb=("target", 8.0, 7.0, 3.0)),
    "flat_bend": dict(w=1, fingers=[(0.80, 0.0, 0.0)] * 4, fan=0.0, thumb=("target", 9.0, 5.0, 2.5)),
    # one- and two-finger poses and counting in more styles (owner, 2026-11-13: counting read weakly)
    "pinky": dict(w=1, fingers=_FIST[:3] + [(0.02, 0.03, 0.03)], fan=0.1, thumb=("touch", 11, 0.0, 0.5, 1.6)),
    "middle": dict(w=0.7, fingers=[_FIST[0], (0.02, 0.03, 0.03), *_FIST[2:]], fan=0.0, thumb=("touch", 7, 0.0, 0.5, 1.6)),
    "rock": dict(w=1, fingers=[(0.02, 0.03, 0.03), *_FIST[1:3], (0.02, 0.03, 0.03)], fan=0.2,
                 thumb=("touch", 11, 0.0, 0.5, 1.6)),
    "three_scout": dict(w=0.7, fingers=[(0.02, 0.03, 0.03)] * 3 + [_FIST[3]], fan=0.12, thumb=("touch", 19, 0.0, 0.3, 1.6)),
    "count1_thumb": dict(w=1.2, fingers=_FIST, fan=0.0, thumb=("target", 5.5, 9.5, 0.5)),
    "count2_thumb": dict(w=1.5, fingers=[(0.02, 0.03, 0.03)] + _FIST[1:], fan=0.1, thumb=("target", 6.0, 9.5, 0.5)),
    "count3_thumb": dict(w=1.5, fingers=[(0.02, 0.03, 0.03)] * 2 + _FIST[2:], fan=0.2, thumb=("target", 6.0, 9.5, 0.5)),
    "count4_thumb": dict(w=1.2, fingers=[(0.02, 0.03, 0.03)] * 3 + _FIST[3:], fan=0.15, thumb=("target", 6.5, 9.5, 0.5)),
    "typing": dict(w=2, fingers=[(0.25, 0.35, 0.22), (0.25, 0.38, 0.22), (0.28, 0.40, 0.24), (0.30, 0.42, 0.24)],
                   fan=0.03, thumb=("target", 8.5, 6.5, 4.5)),
}


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


class HandMotion:
    def __init__(self, fk=None, tremor=True):
        self.fk = fk if fk is not None else HandFK()
        self.tremor = tremor  # False: no 8-12 Hz tremor (jitter-free targets for the merger; same random draws)
        self.lo, self.hi = self.fk.lo, self.fk.hi
        P0 = self.fk.points(np.zeros(len(PARAMS)))
        self.origin, self.R = hand_frame(P0)  # rest hand frame (the wrist does not move)
        self.names = list(GESTURES)
        w = np.array([GESTURES[n]["w"] for n in self.names], float)
        self.gesture_p = w / w.sum()
        self.library = {n: self._gesture(GESTURES[n]) for n in self.names}

    # --- poses -----------------------------------------------------------------------------------
    def _curl(self, q, finger, fracs):
        for k, f in enumerate(fracs, 1):
            j = P[f"{finger}{k}_rz"]
            q[j] = f * self.hi[j] if f >= 0 else -f * self.lo[j]

    def _fan(self, q, fan, curls):
        for i, f in enumerate(FINGERS):
            q[P[f"{f}1_ry"]] = fan * SPREAD_FAN[i] * (1.0 - 0.8 * np.clip(curls[i], 0, 1))

    def _to_hand(self, x_cm):
        return self.origin + (np.asarray(x_cm) / 100.0) @ self.R

    def solve_thumb(self, q, target, init=None):
        """Thumb parameters putting the thumb tip at `target` (MHR world metres), within limits."""
        idx = [P[n] for n in THUMB_SOLVED]
        lo, hi = self.lo[idx], self.hi[idx]
        x0 = np.clip(q[idx] if init is None else init, lo + 1e-3, hi - 1e-3)

        def res(x):
            qq = q.copy()
            qq[idx] = x
            tip = self.fk.points(qq)[4]
            return np.concatenate([100.0 * (tip - target), THUMB_STIFF * (x - THUMB_NATURAL)])

        def jac(x):  # fixed 1e-3 rad steps: scipy's relative steps are below MHR's float32 FK precision
            r0 = res(x)
            J = np.empty((len(r0), len(x)))
            for k in range(len(x)):
                e = np.zeros(len(x))
                e[k] = 1e-3 if x[k] + 1e-3 <= hi[k] else -1e-3
                J[:, k] = (res(x + e) - r0) / e[k]
            return J

        sol = least_squares(res, x0, jac=jac, bounds=(lo, hi), max_nfev=100)
        out = q.copy()
        out[idx] = sol.x
        return out

    def _gesture(self, g):
        q = np.zeros(len(PARAMS))
        for f, fr in zip(FINGERS, g["fingers"]):
            self._curl(q, f, fr)
        self._fan(q, g["fan"], [fr[0] for fr in g["fingers"]])
        t = g["thumb"]
        if t[0] == "target":
            target = self._to_hand(t[1:4])
        else:  # next to a landmark of the posed hand, offset in the hand frame
            pts = self.fk.points(q)
            target = pts[t[1]] + (np.asarray(t[2:5]) / 100.0) @ self.R
        return self.solve_thumb(q, target)

    def random_pose(self, rng):
        """Coupled random pose: a global curl, per-finger deviations correlated across neighbours."""
        q = np.zeros(len(PARAMS))
        g = rng.beta(1.3, 1.6)
        dev = np.convolve(rng.normal(0, 0.25, 6), [0.25, 0.5, 0.25], "same")[1:5]
        curls = []
        for i, f in enumerate(FINGERS):
            mcp = np.clip(g + dev[i] + 0.04 * i + rng.normal(0, 0.08), -0.15, 1.0)
            pip = np.clip(g + dev[i] + 0.05 * i + rng.normal(0, 0.15), -0.05, 1.0)
            dip = np.clip(pip * rng.uniform(0.45, 0.9) + rng.normal(0, 0.05), -0.05, 1.0)
            self._curl(q, f, (mcp, pip, dip))
            curls.append(mcp)
        self._fan(q, rng.normal(0.08, 0.15), curls)
        # thumb: a random blend of the gesture library's (natural) thumbs plus a little noise; sampling
        # its joints independently over their ranges pushed the thumb into the palm or hooked it
        th = [P[n] for n in THUMB]
        w = rng.dirichlet(np.full(len(self.names), 0.3))
        q[th] = w @ np.stack([self.library[n][th] for n in self.names])
        q[th] += rng.normal(0, 0.05, len(th)) * (self.hi[th] - self.lo[th])
        q[P["thumb1_rx"]] = 0.0
        return np.clip(q, self.lo, self.hi)

    def independent_pose(self, rng):
        """Each finger straight, half or curled independently (counting, single fingers); the thumb out,
        or across onto the first curled finger's middle joint."""
        q = np.zeros(len(PARAMS))
        curls = []
        for f in FINGERS:
            state = rng.choice(3, p=[0.45, 0.15, 0.40])
            c = (rng.uniform(-0.05, 0.1), rng.uniform(0.3, 0.6), rng.uniform(0.75, 1.0))[state]
            pip = np.clip(c * rng.uniform(0.9, 1.15) + rng.normal(0, 0.05), -0.05, 1.0)
            dip = np.clip(pip * rng.uniform(0.5, 0.85), -0.05, 1.0)
            self._curl(q, f, (c, pip, dip))
            curls.append(c)
        self._fan(q, rng.uniform(0.0, 0.25), curls)
        curled = [k for k, c in enumerate(curls) if c > 0.7]
        if curled and rng.random() < 0.6:  # thumb across the curled fingers
            pts = self.fk.points(q)
            k = curled[0] if rng.random() < 0.7 else curled[rng.integers(len(curled))]
            target = pts[6 + 4 * k] + (np.array([0.0, 0.5, 1.6]) / 100.0) @ self.R
        else:  # thumb out to the side, up or relaxed
            target = self._to_hand((rng.uniform(5.0, 9.0), rng.uniform(6.5, 10.0), rng.uniform(-1.0, 4.0)))
        q = self.solve_thumb(q, target)
        q[P["thumb1_rx"]] = 0.0
        return np.clip(q, self.lo, self.hi)

    def relaxed_pose(self, rng, flat=False):
        """A resting hand (arm hanging or in the lap): the relaxed gesture, more or less curled, varied.
        `flat`: less curled (a hand lying on the thigh or the belly)."""
        q = self.library["relaxed"].copy()
        curl = rng.uniform(0.2, 0.8) if flat else rng.uniform(0.6, 1.8)
        fingers = [k for k, n in enumerate(PARAMS) if not n.startswith("thumb") and n.endswith("_rz")]
        q[fingers] *= curl
        q += rng.normal(0, 0.04, len(PARAMS)) * (self.hi - self.lo) * 0.5
        q[P["thumb1_rx"]] = 0.0
        return np.clip(q, self.lo, self.hi)

    def _noisy(self, name, rng):
        noise = rng.normal(0, 0.06, len(PARAMS)) * (self.hi - self.lo) * 0.5
        return np.clip(self.library[name] + noise, self.lo, self.hi)

    def key_pose(self, rng):
        if getattr(self, "_queue", None):
            name = self._queue.pop(0)
            return self._noisy(name, rng), name
        if rng.random() < P_ROUTINE:
            seq = list(COUNTS[rng.integers(len(COUNTS))])
            seq = seq[rng.integers(0, 3):] if rng.random() < 0.7 else seq[::-1][rng.integers(0, 3):]
            self._queue = seq[1:]
            return self._noisy(seq[0], rng), seq[0]
        u = rng.random()
        if u < P_LIBRARY:
            name = self.names[rng.choice(len(self.names), p=self.gesture_p)]
            return self._noisy(name, rng), name
        if u < P_LIBRARY + P_INDEPENDENT:
            return self.independent_pose(rng), "independent"
        return self.random_pose(rng), "random"

    # --- sequences -------------------------------------------------------------------------------
    def sequence(self, rng, seconds, fps=30.0, t0=0.0):
        """(T, 27) finger parameters at fps, plus the key-pose names with their start times."""
        t = t0 + np.arange(int(round(seconds * fps))) / fps
        n = len(PARAMS)
        # per-parameter timing offset: per finger, thumb a little looser
        group = np.array([0 if p.startswith("thumb") else 1 + FINGERS.index(next(f for f in FINGERS if p.startswith(f)))
                          for p in PARAMS])
        keys, labels = [], []
        self._queue = []
        tk, q = t0 - rng.uniform(0, 1.0), self.key_pose(rng)[0]
        keys.append((tk, tk, q))
        while tk < t[-1] + 1.0:
            in_routine = bool(self._queue)
            hold = rng.choice([rng.uniform(0.0, 0.25), rng.uniform(0.3, 1.5), rng.uniform(1.5, 4.0)], p=[0.15, 0.55, 0.3])
            if in_routine:
                hold = rng.uniform(0.3, 1.2)
            q_next, name = self.key_pose(rng)
            dist = np.abs(q_next - q).max()
            dur = np.clip(rng.uniform(0.15, 0.45) + 0.2 * dist, 0.15, 0.9)
            start = tk + hold
            keys.append((start, start + dur, q_next))
            labels.append((round(float(start), 2), name))
            tk, q = start + dur, q_next
        off = rng.normal(0, 0.04, 5)
        off[0] = rng.normal(0, 0.06)
        out = np.empty((len(t), n))
        for j in range(n):
            tj = t - off[group[j]]
            v = np.full(len(t), keys[0][2][j])
            for (a, b, qk), (_, _, qp) in zip(keys[1:], keys[:-1]):
                s = min_jerk((tj - a) / (b - a))
                v = np.where(tj >= a, qp[j] + (qk[j] - qp[j]) * s, v)
            out[:, j] = v
        out += self._idle(rng, t)
        out[:, [P[p] for p in UNOBSERVABLE]] = 0.0
        return np.clip(out, self.lo, self.hi), labels

    def _idle(self, rng, t):
        """Slow drift (~0.3 Hz), tremor (8-12 Hz, tiny) and occasional single-finger taps."""
        n, T = len(PARAMS), len(t)
        span = self.hi - self.lo
        drift = np.zeros((T, n))
        for j in range(n):
            f = rng.uniform(0.1, 0.5, 3)
            ph = rng.uniform(0, 2 * np.pi, 3)
            drift[:, j] = sum(np.sin(2 * np.pi * fk * t + p) for fk, p in zip(f, ph)) / 3
        drift *= rng.uniform(0.005, 0.03) * span
        f_tr = rng.uniform(8, 12)
        tremor = np.sin(2 * np.pi * f_tr * t[:, None] + rng.uniform(0, 2 * np.pi, n)) * rng.uniform(0, 0.006) * span
        tremor = tremor * self.tremor
        taps = np.zeros((T, n))
        for _ in range(rng.poisson((t[-1] - t[0]) / 4.0)):  # about one tap every 4 s
            f = FINGERS[rng.integers(4)]
            c = rng.uniform(t[0], t[-1])
            w = rng.uniform(0.08, 0.2)
            bump = np.exp(-0.5 * ((t - c) / w) ** 2) * rng.uniform(0.15, 0.45)
            taps[:, P[f"{f}1_rz"]] += bump
            taps[:, P[f"{f}2_rz"]] += 0.5 * bump
        return drift + tremor + taps
