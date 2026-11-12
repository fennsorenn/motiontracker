"""Run a trained hand model over landmark frames (10 Hz), one hand history per side.

`LearnedHands(ckpt)(frame)` -> {side: 21 MHR hand points} for the hands present in this frame; used by
`mttrain.hands.evaluate --system <ckpt>`. `params(frame)` gives the finger parameters instead.
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
