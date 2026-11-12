"""Pair hand model (v2) on rendered sessions: fingers and hand orientation against exact truth.

    uv run python -m mttrain.hands.evaluate2 data/m4/renders/<set> --ckpt data/m4/runs/<name>/best.pt
        [--landmarks landmarks.jsonl | landmarks-hl.jsonl]

Per true hand and frame where the hand is in the frame: finger segment directions in the palm frame
(`evaluate.angle_errors`, MHR points from the predicted finger parameters), and the hand's orientation in
the camera frame (palm frame of its image points, right hands mirrored; truth from the posed hand projected
with the render camera), against MediaPipe's raw orientation (palm frame of the detection's own image
landmarks, where the detection sits on this hand). Frames where MediaPipe has no detection on the hand are
reported separately (the model still answers from history and the other hand). Also orientation and finger
motion per frame against the truth's (jitter).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mtcore.hand_features import canonical, palm_frame

from .evaluate import Truth, angle_errors
from .failures import project
from .imagecal import image_points, rot_angle, truth_image
from .model import OUT_IDX
from .model2 import HISTORY, PairNet, frame_features
from .skeleton import PARAMS, HandFK


class LearnedPair:
    def __init__(self, ckpt, fk=None):
        self.model = PairNet()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        self.model.eval()
        self.fk = fk or HandFK()
        self.reset()

    def reset(self):
        self.hist = []  # newest first: per frame (img (2, 21, 3), canon (2, 21, 3), lost (2,))

    def step(self, frame):
        """Recorder-format frame -> {side: (finger params (27,), orientation (3, 3) in the scene convention)}."""
        img, canon, lost = np.zeros((2, 21, 3)), np.zeros((2, 21, 3)), np.ones(2, bool)
        for k, (side, S) in enumerate((("l", "left"), ("r", "right"))):
            if frame.get(f"{S}Hand2d") and frame.get(f"{S}Hand"):
                img[k] = image_points(frame[f"{S}Hand2d"], "l")  # unmirrored; the scenes mirror
                canon[k] = canonical(np.array(frame[f"{S}Hand"]), side)
                lost[k] = False
        self.hist.insert(0, (img, canon, lost))
        del self.hist[HISTORY:]
        pad = HISTORY - len(self.hist)
        I = torch.tensor(np.stack([h[0] for h in self.hist] + [np.zeros((2, 21, 3))] * pad), dtype=torch.float32)
        Cn = torch.tensor(np.stack([h[1] for h in self.hist] + [np.zeros((2, 21, 3))] * pad), dtype=torch.float32)
        Lo = torch.tensor(np.stack([h[2] for h in self.hist] + [np.ones(2, bool)] * pad))
        own = torch.stack([frame_features(I[:, 0], Cn[:, 0], Lo[:, 0], False), frame_features(I[:, 1], Cn[:, 1], Lo[:, 1], True)])
        other = torch.stack([frame_features(I[:, 1], Cn[:, 1], Lo[:, 1], False), frame_features(I[:, 0], Cn[:, 0], Lo[:, 0], True)])
        with torch.no_grad():
            out = self.model(own, other)
        res = {}
        for k, side in enumerate("lr"):
            q = np.zeros(len(PARAMS))
            q[OUT_IDX] = out["fingers"][k].numpy()
            res[side] = (np.clip(q, self.fk.lo, self.fk.hi), out["rot"][k].numpy())
        return res


def evaluate(set_dir, ckpt, landmarks="landmarks.jsonl"):
    fk = HandFK()
    run = LearnedPair(ckpt, fk)
    rows = []
    for d in sorted(Path(set_dir).glob("seq*")):
        run.reset()
        z = np.load(d / "seq.npz")
        Rc, tc, fx = z["R"], z["t"], float(z["fx"])
        truth = Truth(d, fk)
        prev = {}
        for line in open(d / landmarks):
            f = json.loads(line)
            out = run.step(f)
            C = {s: (truth.posed_points(s, f["t"]) - tc) @ Rc.T for s in "lr"}
            img = {s: project(C[s], fx) for s in "lr"}
            for side, S in (("l", "left"), ("r", "right")):
                Ti = truth_image(C[side], fx, side)
                inside = (np.abs(Ti[:, 0]) < 0.5).all() and (np.abs(Ti[:, 1]) < 0.375).all() and (C[side][:, 2] > 0.05).all()
                if not inside:
                    prev.pop(side, None)
                    continue
                _, Rt, _ = palm_frame(Ti)
                q, Rp = out[side]
                # MediaPipe's raw orientation, if some detection sits on this hand
                raw, det = None, False
                for s2, S2 in (("l", "left"), ("r", "right")):
                    L = f.get(f"{S2}Hand2d")
                    if not L:
                        continue
                    p = np.array(L)[[0, 9], :2].mean(0)
                    owner = min("lr", key=lambda s: np.linalg.norm(p - img[s][[0, 9]].mean(0)))
                    if owner == side:
                        det = True
                        _, Rm, _ = palm_frame(image_points(L, side))
                        raw = rot_angle(Rm, Rt)
                fe = angle_errors(fk.points(q, side), truth.points(side, f["t"])).mean()
                row = {"det": det, "rot": rot_angle(Rp, Rt), "raw": raw, "fing": float(fe)}
                if side in prev and f["t"] - prev[side][0] < 150:
                    row["rot_motion"] = rot_angle(Rp, prev[side][1])
                    row["true_motion"] = rot_angle(Rt, prev[side][2])
                prev[side] = (f["t"], Rp, Rt)
                rows.append(row)
    return rows


def report(rows, name):
    det = [r for r in rows if r["det"]]
    nodet = [r for r in rows if not r["det"]]
    med = lambda v: float(np.median(v)) if len(v) else float("nan")
    p90 = lambda v: float(np.percentile(v, 90)) if len(v) else float("nan")
    print(f"{name}: {len(rows)} hand-frames in view, {len(det)} with a detection on the hand")
    print(f"  orientation (camera frame): learned median {med([r['rot'] for r in det]):.1f} deg, p90 {p90([r['rot'] for r in det]):.1f}; "
          f"MediaPipe raw median {med([r['raw'] for r in det]):.1f}, p90 {p90([r['raw'] for r in det]):.1f}")
    print(f"  without a detection: learned orientation median {med([r['rot'] for r in nodet]):.1f} deg (n {len(nodet)})")
    print(f"  fingers (segment directions, detected frames): median {med([r['fing'] for r in det]):.1f} deg")
    mo = [r for r in rows if "rot_motion" in r]
    if mo:
        sm, tm = np.mean([r["rot_motion"] for r in mo]), np.mean([r["true_motion"] for r in mo])
        print(f"  orientation motion per frame: learned {sm:.1f} deg, truth {tm:.1f} (x{sm / max(tm, 1e-9):.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", type=Path)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--landmarks", default="landmarks.jsonl")
    a = ap.parse_args()
    report(evaluate(a.set, a.ckpt, a.landmarks), f"{a.ckpt} on {a.set.name} ({a.landmarks})")


if __name__ == "__main__":
    main()
