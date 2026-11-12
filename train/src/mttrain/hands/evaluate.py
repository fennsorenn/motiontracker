"""Finger accuracy on rendered hand sessions (render --hands), against exact ground truth.

    uv run --with-editable ../service python -m mttrain.hands.evaluate data/m4/renders/<set> [--system ported]

Metric: the direction of each finger segment (thumb CMC-MCP, MCP-IP, IP-tip; each finger MCP-PIP,
PIP-DIP, DIP-tip: 15 per hand) in a palm frame built from the hand's own points (`skeleton.hand_frame`:
wrist -> middle knuckle, little -> index knuckle), angle between system and truth, degrees. Built the
same way on both sides, so it ignores where the wrist joint's own axes sit and finger roll (invisible),
and compares what the avatar shows: where the fingers point relative to the palm. Ground truth: MHR's
hand points from the stored finger parameters. Systems that output VRM finger rotations (the ported
solve) are played on the avatar's hand (VRM FK on the avatar's rest positions) to get their points.
Frames where MediaPipe found no hand are reported as a detection rate, not scored.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from mtcore.convert import VRM_PARENT, qmul, qrot
from mtcore.paths import AVATARS
from mtcore.vrm import load_vrm

from .skeleton import FINGERS, HandFK, hand_frame

# segments as (from, to) MediaPipe point indices
SEGMENTS = [(1, 2), (2, 3), (3, 4)] + [(b + k, b + k + 1) for b in (5, 9, 13, 17) for k in range(3)]
VRM_FINGERS = {"index": "Index", "middle": "Middle", "ring": "Ring", "pinky": "Little"}


def vrm_points(local, side, rest):
    """VRM finger rotations (bone -> xyzw, relative to the hand) -> the 21 hand points on the avatar.
    Fingertips: the distal bone's rest direction and length (as its parent's) rotated by its global."""
    S = "left" if side == "l" else "right"
    hand = f"{S}Hand"
    names = [hand, f"{S}ThumbMetacarpal", f"{S}ThumbProximal", f"{S}ThumbDistal", None]
    for f in FINGERS:
        v = VRM_FINGERS[f]
        names += [f"{S}{v}Proximal", f"{S}{v}Intermediate", f"{S}{v}Distal", None]
    G = {hand: np.array([0, 0, 0, 1.0])}
    pos = {hand: np.asarray(rest[hand], float)}

    def solve(b):
        if b in pos:
            return
        p = VRM_PARENT[b]
        solve(p)
        G[b] = qmul(G[p], np.asarray(local.get(b, [0, 0, 0, 1.0]), float))
        pos[b] = pos[p] + qrot(G[p], np.asarray(rest[b], float) - np.asarray(rest[p], float))

    out = []
    for k, b in enumerate(names):
        if b is None:  # fingertip
            d, p = names[k - 1], names[k - 2]
            solve(d)
            seg = np.asarray(rest[d], float) - np.asarray(rest[p], float)
            out.append(pos[d] + qrot(G[d], seg))
        else:
            solve(b)
            out.append(pos[b])
    return np.array(out)


def segment_dirs(P):
    """21 points (..., 21, 3) -> unit segment directions in the palm frame (..., 15, 3)."""
    _, R = hand_frame(P)
    v = np.stack([P[..., j, :] - P[..., i, :] for i, j in SEGMENTS], -2)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.einsum("...ij,...sj->...si", R, v)


def angle_errors(Pa, Pb):
    """Per-segment direction angles (degrees) between two point sets in their own palm frames."""
    c = np.clip((segment_dirs(Pa) * segment_dirs(Pb)).sum(-1), -1, 1)
    return np.degrees(np.arccos(c))


def mirror(P):
    return P * np.array([-1.0, 1.0, 1.0])


class Truth:
    def __init__(self, seq_dir, fk):
        d = Path(seq_dir)
        z = np.load(d / "fingers.npz")
        self.fps = float(np.load(d / "seq.npz")["fps"])
        self.q = {s: z[s] for s in "lr"}
        self.fk = fk
        self.cache = {}

    def points(self, side, t_ms):
        k = min(len(self.q[side]) - 1, max(0, int(round(t_ms / 1000 * self.fps))))
        if (side, k) not in self.cache:
            self.cache[side, k] = self.fk.points(self.q[side][k], side)
        return self.cache[side, k]


def ported(rest):
    from mtservice.hands import solve_hand

    def run(frame):
        out = {}
        for side, S in (("l", "left"), ("r", "right")):
            pts = frame.get(f"{S}Hand")
            if pts:
                out[side] = vrm_points(solve_hand(np.asarray(pts), S), side, rest)
        return out
    return run


def evaluate(set_dir, system, avatar="AvatarSample_D"):
    fk = HandFK()
    rest = load_vrm(AVATARS / f"{avatar}.vrm")[2]
    run = {"ported": ported(rest)}[system] if isinstance(system, str) else system
    errs, seen, total = [], 0, 0
    for d in sorted(Path(set_dir).glob("seq*")):
        truth = Truth(d, fk)
        for line in open(d / "landmarks.jsonl"):
            f = json.loads(line)
            out = run(f)
            for side in "lr":
                total += 1
                if side in out:
                    seen += 1
                    errs.append(angle_errors(out[side], truth.points(side, f["t"])))
    E = np.array(errs)  # (n, 15)
    return E, seen / max(total, 1)


def report(E, rate, name):
    print(f"{name}: hands scored in {100 * rate:.0f}% of hand-frames; segment direction error median "
          f"{np.median(E):.1f} deg, mean {E.mean():.1f}, p90 {np.percentile(E, 90):.1f}")
    for i, f in enumerate(("thumb", *FINGERS)):
        v = E[:, 3 * i:3 * i + 3]
        print(f"  {f:7s} median {np.median(v):5.1f}  p90 {np.percentile(v, 90):5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", type=Path)
    ap.add_argument("--system", default="ported")
    a = ap.parse_args()
    E, rate = evaluate(a.set, a.system)
    report(E, rate, a.system)


if __name__ == "__main__":
    main()
