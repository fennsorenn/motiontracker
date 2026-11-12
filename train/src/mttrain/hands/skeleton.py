"""MHR hand skeleton for the hand model: finger parameters, MediaPipe's 21 hand points, FK.

Canonical hand = MHR's left hand. The right hand is the left mirrored across x for the same parameter
values (checked: max 0.001 cm), so one model serves both: right-hand inputs are mirrored into the left.

Facts behind the tables (plan `dev-notes/plans/m4-hand-model.md`, step 1):
- MediaPipe's 21 points: 0 wrist; thumb 1-4 = thumb1, thumb2, thumb3, thumb_null (MHR's thumb0 is an
  extra base joint); fingers = <f>1, <f>2, <f>3, <f>_null (MHR's pinky0 is an extra metacarpal joint).
- <f>{1,2,3}_rz curl toward the palm (positive = flex); <f>1_ry spreads (positive = toward the thumb);
  <f>1_rx is an axial roll that does not move the fingertip: unobservable, kept at 0.
"""

import numpy as np

from mtcore import mhr

FINGERS = ("index", "middle", "ring", "pinky")
# the 27 finger parameters of one hand, without the side prefix, in a fixed order
PARAMS = ("thumb0_ry", "thumb0_rz", "thumb1_rx", "thumb1_ry", "thumb1_rz", "thumb2_rz", "thumb3_rz",
          *[f"{f}1_ry" for f in FINGERS], *[f"{f}{k}_rz" for f in FINGERS for k in (1, 2, 3)],
          *[f"{f}1_rx" for f in FINGERS])
UNOBSERVABLE = tuple(f"{f}1_rx" for f in FINGERS)
THUMB = PARAMS[:7]
# MediaPipe hand landmark k -> MHR joint (without side prefix)
MP21 = ("wrist", "thumb1", "thumb2", "thumb3", "thumb_null",
        *[j for f in FINGERS for j in (f"{f}1", f"{f}2", f"{f}3", f"{f}_null")])


def limits():
    """(lo, hi) arrays over PARAMS from MHR's [Limits]."""
    lim = mhr.limits()
    lo = np.array([lim[f"l_{p}"][0] for p in PARAMS])
    hi = np.array([lim[f"l_{p}"][1] for p in PARAMS])
    return lo, hi


class HandFK:
    """MHR forward kinematics for hand poses (CPU, pymomentum). Body parameters stay at zero."""

    def __init__(self, ch=None):
        self.ch = ch if ch is not None else mhr.load()
        self.pn = list(self.ch.parameter_transform.names)
        self.jn = list(self.ch.skeleton.joint_names)
        self.idx = {s: np.array([self.pn.index(f"{s}_{p}") for p in PARAMS]) for s in "lr"}
        self.mp21 = {s: np.array([self.jn.index(f"{s}_{j}") for j in MP21]) for s in "lr"}
        self.lo, self.hi = limits()

    def full(self, q, side="l", base=None):
        """Finger parameters (..., 27) -> full MHR model parameters (..., P)."""
        q = np.asarray(q, dtype=np.float32)
        out = np.zeros(q.shape[:-1] + (len(self.pn),), np.float32) if base is None else np.array(base, np.float32)
        out[..., self.idx[side]] = q
        return out

    def points(self, q, side="l"):
        """Finger parameters (..., 27) -> MediaPipe's 21 hand points (..., 21, 3), MHR world metres."""
        return mhr.positions(self.ch, self.full(q, side))[..., self.mp21[side], :] / 100.0


def hand_frame(P):
    """Hand frame from 21 points (..., 21, 3): rows fwd (wrist -> middle MCP), thumbward (pinky -> index MCP,
    made orthogonal), palm (fwd x thumbward: the side the fingers curl toward, for the left hand).
    Returns (origin = wrist (..., 3), R (..., 3, 3))."""
    w = P[..., 0, :]
    fwd = P[..., 9, :] - w
    fwd = fwd / np.linalg.norm(fwd, axis=-1, keepdims=True)
    side = P[..., 5, :] - P[..., 17, :]
    side = side - (side * fwd).sum(-1, keepdims=True) * fwd
    side = side / np.linalg.norm(side, axis=-1, keepdims=True)
    return w, np.stack([fwd, side, np.cross(fwd, side)], -2)
