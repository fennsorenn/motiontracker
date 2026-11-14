"""Detector confidence per hand slot, calibrated on rendered sessions (PLAN.md section 5.6 v3: the pair hand model
ingests one generic per-hand detector score).

    uv run python -m mttrain.hands.scorecal data/m4/renders/pairs2 data/m4/renders/hands2 --out data/m4/calib/scores.json

The score of a slot (`hand_score`): Hand Landmarker frames carry the detection's handedness score
(<side>HandScore); Holistic has no hand score, so it is the pose model's mean visibility of that side's wrist and
hand points (pose 15/17/19/21 left, 16/18/20/22 right), which it reports whether or not a hand was found. With
neither, 0.

Per detector (Holistic: landmarks.jsonl, Hand Landmarker: landmarks-hl.jsonl) and per slot state, the score's
distribution as quantiles (`QUANTILES`) and its frame-to-frame correlation (lag-1, normal scores), for
`observe2` to synthesize: seen (the slot's detection sits on the slot's hand), wrong (it sits on the other hand:
swapped or doubled), lost (no detection, hand in view), out (hand out of view).
"""

import argparse
import json
from pathlib import Path

import numpy as np

STATES = ("seen", "wrong", "lost", "out")
DETECTORS = {"holistic": "landmarks.jsonl", "handlandmarker": "landmarks-hl.jsonl"}
QUANTILES = np.linspace(0, 1, 21)
POSE_HAND = {"l": [15, 17, 19, 21], "r": [16, 18, 20, 22]}


def hand_score(frame, side):
    """One frame (recorder format) and slot "l" / "r" -> the detector's confidence for that slot in [0, 1]."""
    S = "left" if side == "l" else "right"
    if f"{S}HandScore" in frame:
        return float(frame[f"{S}HandScore"])
    pose = frame.get("pose2d") or frame.get("pose")
    if pose and len(pose[0]) > 3:
        return float(np.mean([pose[k][3] for k in POSE_HAND[side]]))
    return 0.0


def collect(set_dirs, landmarks):
    from .evaluate import Truth
    from .failures import project
    from .imagecal import truth_image
    from .skeleton import HandFK
    fk = HandFK()
    out = {s: [] for s in STATES}
    series = []  # per sequence and slot: (states, scores)
    for set_dir in set_dirs:
        for d in sorted(Path(set_dir).glob("seq*")):
            if not (d / landmarks).exists():
                continue
            z = np.load(d / "seq.npz")
            Rc, tc, fx = z["R"], z["t"], float(z["fx"])
            truth = Truth(d, fk)
            seq = {s: ([], []) for s in "lr"}
            for line in open(d / landmarks):
                f = json.loads(line)
                C = {s: (truth.posed_points(s, f["t"]) - tc) @ Rc.T for s in "lr"}
                img = {s: project(C[s], fx) for s in "lr"}
                for side, S in (("l", "left"), ("r", "right")):
                    Ti = truth_image(C[side], fx, side)
                    inside = (np.abs(Ti[:, 0]) < 0.5).all() and (np.abs(Ti[:, 1]) < 0.375).all() and (C[side][:, 2] > 0.05).all()
                    L = f.get(f"{S}Hand2d")
                    if L:
                        p = np.array(L)[[0, 9], :2].mean(0)
                        owner = min("lr", key=lambda s: np.linalg.norm(p - img[s][[0, 9]].mean(0)))
                        state = "seen" if owner == side else "wrong"
                    else:
                        state = "lost" if inside else "out"
                    sc = hand_score(f, side)
                    out[state].append(sc)
                    seq[side][0].append(state)
                    seq[side][1].append(sc)
            series += list(seq.values())
    return out, series


def lag1(series, state):
    """Lag-1 correlation of the scores' normal ranks over consecutive frames both in `state`."""
    from scipy.stats import norm, rankdata
    a, b = [], []
    allv = np.concatenate([np.array(sc)[np.array(st) == state] for st, sc in series if len(st)])
    if len(allv) < 20:
        return 0.9
    for st, sc in series:
        st, sc = np.array(st), np.array(sc)
        m = (st[1:] == state) & (st[:-1] == state)
        a += list(sc[:-1][m])
        b += list(sc[1:][m])
    if len(a) < 20:
        return 0.9
    ranks = rankdata(np.concatenate([a, b])) / (2 * len(a) + 1)
    z = norm.ppf(ranks)
    r = np.corrcoef(z[:len(a)], z[len(a):])[0, 1]
    return float(np.clip(r, 0.0, 0.99)) if np.isfinite(r) else 0.9  # constant scores (no detection): any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    calib = {}
    for det, lm in DETECTORS.items():
        out, series = collect(a.sets, lm)
        calib[det] = {}
        for s in STATES:
            v = np.array(out[s])
            if len(v) == 0:
                print(f"{det} {s}: no frames")
                continue
            calib[det][s] = {"n": int(len(v)), "q": np.quantile(v, QUANTILES).round(4).tolist(), "rho": lag1(series, s)}
            q = calib[det][s]["q"]
            print(f"{det:15s} {s:6s} n {len(v):5d}  median {q[10]:.3f}  p10 {q[2]:.3f}  p90 {q[18]:.3f}  lag-1 {calib[det][s]['rho']:.2f}")
    a.out.write_text(json.dumps(calib, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    main()
