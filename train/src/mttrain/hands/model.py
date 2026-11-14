"""Hand model: causal, the last HISTORY frames (10 Hz) of one hand's canonical shape -> finger parameters.

Input per frame: MediaPipe's 21 hand points in canonical palm-frame form (`mtcore.hand_features`,
63 numbers, zeros when the hand is lost) plus a missing flag; frames newest first. A shared frame
encoder plus a per-position embedding, differences between neighbouring frames, then an MLP to the
observable finger parameters (`OUT_PARAMS`, radians, rest = 0) and a log-scale per parameter (Laplace
confidence). Zero network output = rest pose. Stateless: the caller keeps the history.
"""

import torch
from torch import nn

from .skeleton import PARAMS, UNOBSERVABLE

HISTORY = 8  # 0.8 s at 10 Hz
FRAME_DIM = 63 + 1
OUT_PARAMS = tuple(p for p in PARAMS if p not in UNOBSERVABLE)
OUT_IDX = [PARAMS.index(p) for p in OUT_PARAMS]


class HandNet(nn.Module):
    def __init__(self, enc=96, hidden=384, history=HISTORY):
        super().__init__()
        self.n_out = len(OUT_PARAMS)
        self.frame = nn.Sequential(nn.Linear(FRAME_DIM, 192), nn.GELU(), nn.Linear(192, enc))
        self.pos_emb = nn.Parameter(torch.zeros(history, enc))
        width = (2 * history - 1) * enc
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2 * self.n_out))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        """x (..., HISTORY, FRAME_DIM) newest first -> params (..., n_out), log_scale (..., n_out)."""
        e = self.frame(x) + self.pos_emb
        h = torch.cat([e, e[..., :-1, :] - e[..., 1:, :]], -2).flatten(-2)
        out = self.mlp(h)
        return out[..., :self.n_out], out[..., self.n_out:]


def frame_input(canon, missing):
    """canon (..., 21, 3), missing (...) bool -> (..., FRAME_DIM); lost frames are zeros + flag."""
    flat = canon.flatten(-2) * (~missing)[..., None]
    return torch.cat([flat, missing[..., None].to(flat.dtype)], -1)
