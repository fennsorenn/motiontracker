"""The merger (plan `dev-notes/plans/m4-merger.md`): part outputs over a short history -> residual corrections on
the base pose (the hand-written combination, `batch.combine`).

Per detector frame, the inputs (`frame_inputs`): the body model's 17 locals (6D) and log scales, the base pose's
locals (6D), per hand the hand model's wrist (6D global), rotation log scale, fingers and their log scales, image
position, log size, lost flag and detector score, the face head pose (6D global), centre, log scale and seen flag,
and the blendshapes with a present flag. History: frames at OFFSETS back (10 Hz), each encoded by one frame MLP plus
an offset embedding, with the differences between neighbouring offsets (like `mttrain.model.Lifter`).

Outputs at the newest frame: per body joint a 6D residual applied to the base pose's global rotation (G = G_base
R(res); locals follow), so correcting a forearm leaves the hand's global orientation from the hand model in place
(pilot, 2026-11-14: local residuals pulled hand-model wrists off by ~6 deg); and a log scale; per hand finger residuals (on the hand model's fingers) and log scales. The last layer starts at zero: an
untrained merger returns the base pose. Blendshapes pass through unchanged (not output here).
"""

import torch
from torch import nn

from ..hands.model import OUT_IDX
from ..model import IDENTITY_6D, rot6d_to_mat
from ..skeleton import OUT_JOINTS
from .batch import globals_from_locals, locals_from_globals

OFFSETS = (0, 1, 2, 4, 8)  # frames back at 10 Hz
NJ = len(OUT_JOINTS)
NF = len(OUT_IDX)
N_BLEND = 51


def to6d(R):
    return R[..., :, :2].transpose(-1, -2).flatten(-2)


def frame_inputs(b):
    """Batch dict (`MergerBatches.sample`) -> per-frame features (B, K, D)."""
    body, base, head, bl = b["body"], b["base"], b["head"], b["blend"]
    parts = [to6d(body["rot"]).flatten(-2), body["log_s"], to6d(base).flatten(-2)]
    for s in "lr":
        h = b["hand"][s]
        parts += [to6d(h["G"]), h["rot_log_s"][..., None], h["fingers"], h["fingers_log_s"], h["pos"],
                  h["log_size"][..., None], h["lost"][..., None].float(), h["score"][..., None]]
    seen = head["seen"][..., None].float()
    parts += [to6d(head["G"]) * seen, head["centre"] * seen, head["log_scale"][..., None] * seen, seen,
              bl["values"], bl["present"][..., None].float()]
    return torch.cat(parts, -1)


FRAME_DIM = NJ * 6 + NJ + NJ * 6 + 2 * (6 + 1 + 2 * NF + 2 + 1 + 1 + 1) + (6 + 2 + 1 + 1) + N_BLEND + 1


class Merger(nn.Module):
    def __init__(self, enc=128, hidden=512, frame_dim=FRAME_DIM):
        super().__init__()
        n = len(OFFSETS)
        self.frame = nn.Sequential(nn.Linear(frame_dim, 256), nn.GELU(), nn.Linear(256, enc))
        self.offset_emb = nn.Parameter(torch.zeros(n, enc))
        self.n_out = NJ * 7 + 2 * NF * 2
        self.mlp = nn.Sequential(nn.LayerNorm((2 * n - 1) * enc), nn.Linear((2 * n - 1) * enc, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, self.n_out))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        """x (..., len(OFFSETS), FRAME_DIM) newest first -> dict: res (..., NJ, 3, 3), log_s (..., NJ),
        fingers_res (..., 2, NF), fingers_log_s (..., 2, NF)."""
        e = self.frame(x) + self.offset_emb
        h = torch.cat([e, e[..., :-1, :] - e[..., 1:, :]], -2).flatten(-2)
        o = self.mlp(h)
        r = o[..., :NJ * 6].unflatten(-1, (NJ, 6))
        k = NJ * 6
        return {"res": rot6d_to_mat(r + IDENTITY_6D.to(r)), "log_s": o[..., k:k + NJ],
                "fingers_res": o[..., k + NJ:k + NJ + 2 * NF].unflatten(-1, (2, NF)),
                "fingers_log_s": o[..., k + NJ + 2 * NF:].unflatten(-1, (2, NF))}


def history(feats, newest):
    """feats (B, K, D) oldest first, newest frame index -> (B, len(OFFSETS), D) newest first."""
    return feats[:, [newest - o for o in OFFSETS]]


def corrected(base, res):
    """Base locals (..., NJ, 3, 3) and global residuals -> corrected locals."""
    return locals_from_globals(globals_from_locals(base) @ res)


def apply(model, b, feats, newest):
    """Merger output at frame `newest` -> body locals (B, NJ, 3, 3), log scales, fingers (B, 2, NF) and log scales."""
    out = model(history(feats, newest))
    L = corrected(b["base"][:, newest], out["res"])
    fingers = torch.stack([b["hand"][s]["fingers"][:, newest] for s in "lr"], 1) + out["fingers_res"]
    return L, out["log_s"], fingers, out["fingers_log_s"]
