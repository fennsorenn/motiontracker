"""Run a trained hand model over landmark frames (10 Hz), one hand history per side.

`LearnedHands(ckpt)(frame)` -> {side: 21 MHR hand points} for the hands present in this frame; used by
`mttrain.hands.evaluate --system <ckpt>`. `params(frame)` gives the finger parameters instead.

    uv run python -m mttrain.hands.infer <ckpt> <runtime landmarks.jsonl> --body <runtime poses.jsonl>
        --out <poses.jsonl> [--avatar AvatarSample_F]   # body + learned fingers, for avatar videos
"""

from pathlib import Path

import numpy as np
import torch

from mtcore.hand_features import canonical

from .model import HISTORY, OUT_IDX, HandNet, frame_input
from .skeleton import PARAMS, HandFK


class LearnedHands:
    def __init__(self, ckpt, fk=None, device="cpu"):
        self.model = HandNet()
        self.model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        self.model.eval()
        self.fk = fk or HandFK()
        self.hist = {s: [] for s in "lr"}

    def reset(self):
        self.hist = {s: [] for s in "lr"}

    def params(self, frame):
        out = {}
        for side, S in (("l", "left"), ("r", "right")):
            pts = frame.get(f"{S}Hand")
            h = self.hist[side]
            h.insert(0, (canonical(np.asarray(pts), side), False) if pts else (np.zeros((21, 3)), True))
            del h[HISTORY:]
            if not pts:
                continue
            c = torch.tensor(np.stack([x for x, _ in h] + [np.zeros((21, 3))] * (HISTORY - len(h))), dtype=torch.float32)
            m = torch.tensor([lost for _, lost in h] + [True] * (HISTORY - len(h)))
            with torch.no_grad():
                pred, _ = self.model(frame_input(c, m)[None])
            q = np.zeros(len(PARAMS))
            q[OUT_IDX] = pred[0].numpy()
            out[side] = np.clip(q, self.fk.lo, self.fk.hi)
        return out

    def __call__(self, frame):
        return {side: self.fk.points(q, side) for side, q in self.params(frame).items()}


FINGER_WORDS = ("Thumb", "Index", "Middle", "Ring", "Little")


def finger_track(ckpt, landmarks, body, out, avatar="AvatarSample_D", hold=True):
    """A pose track (JSONL {t, bones}): `body`'s bones without its fingers, plus the hand model's finger bones
    (VRM, from MHR via mtcore.convert for `avatar`) on the hand landmarks of `landmarks` (runtime
    --dump-landmarks, same frames). hold: a hand's last fingers stay while it is lost (as vspark does)."""
    import json

    from mtcore.convert import MhrToVrm
    from mtcore.mhr import skeleton_state
    from mtcore.paths import AVATARS
    from mtcore.vrm import load_vrm

    run = LearnedHands(ckpt)
    fk = run.fk
    conv = MhrToVrm(fk.jn, skeleton_state(fk.ch, np.zeros(len(fk.pn))), load_vrm(AVATARS / f"{avatar}.vrm")[2])
    last = {}
    with open(landmarks) as fl, open(body) as fb, open(out, "w") as fo:
        for line_l, line_b in zip(fl, fb):
            f, b = json.loads(line_l), json.loads(line_b)
            if abs(f["t"] - b["t"]) > 1:
                raise ValueError(f"frames differ: {f['t']} vs {b['t']}")
            q = run.params(f)
            if hold:
                last.update(q)
            use = last if hold else q
            bones = {n: v for n, v in b["bones"].items() if not any(w in n for w in FINGER_WORDS)}
            if use:
                mp = fk.full(use.get("l", np.zeros(len(PARAMS))), "l", base=fk.full(use.get("r", np.zeros(len(PARAMS))), "r"))
                loc = conv.locals(skeleton_state(fk.ch, mp)[None])
                for n, v in loc.items():
                    side = "l" if n.startswith("left") else "r"
                    if any(w in n for w in FINGER_WORDS) and side in use:
                        bones[n] = [round(float(x), 6) for x in v[0]]
            fo.write(json.dumps({"t": b["t"], "bones": bones}) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="body pose track + hand model fingers -> pose track")
    ap.add_argument("ckpt")
    ap.add_argument("landmarks")
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--avatar", default="AvatarSample_D")
    ap.add_argument("--no-hold", action="store_true")
    a = ap.parse_args()
    finger_track(a.ckpt, a.landmarks, a.body, a.out, a.avatar, hold=not a.no_hold)


if __name__ == "__main__":
    main()
