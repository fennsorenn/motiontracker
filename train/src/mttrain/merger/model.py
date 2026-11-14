"""The merger (plan `dev-notes/plans/m4-merger.md`): part outputs over a short history -> residual corrections on
the base pose (the hand-written combination, `batch.combine`).

Per detector frame, the inputs (`frame_inputs`): the body model's 17 locals (6D) and log scales, the base pose's
locals (6D), per hand the hand model's wrist (6D global), rotation log scale, fingers and their log scales, image
position, log size, lost flag and detector score, the face head pose (6D global), centre, log scale and seen flag,
and the blendshapes with a present flag. History: frames at OFFSETS back (10 Hz), each encoded by one frame MLP plus
an offset embedding, with the differences between neighbouring offsets (like `mttrain.model.Lifter`).

m1-v12+ (`root=True`): also the body model's root position (camera frame, metres), its two log scales and the up
direction as inputs, and residuals on them as outputs (root: x/z, y/z and log z offsets; up: added, renormalized).
Older mergers (mg1, mg2: 17 joints, no root) load with `load_merger`.

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


def frame_inputs(b, nj=NJ, root=True):
    """Batch dict (`MergerBatches.sample`) -> per-frame features (B, K, D). nj, root: the merger's shape (older
    mergers read the first 17 joints and no root)."""
    body, base, head, bl = b["body"], b["base"], b["head"], b["blend"]
    parts = [to6d(body["rot"][..., :nj, :, :]).flatten(-2), body["log_s"][..., :nj], to6d(base[..., :nj, :, :]).flatten(-2)]
    if root:
        parts += [body["root"], body["root_log_s"], body["up"]]
    for s in "lr":
        h = b["hand"][s]
        parts += [to6d(h["G"]), h["rot_log_s"][..., None], h["fingers"], h["fingers_log_s"], h["pos"],
                  h["log_size"][..., None], h["lost"][..., None].float(), h["score"][..., None]]
    seen = head["seen"][..., None].float()
    parts += [to6d(head["G"]) * seen, head["centre"] * seen, head["log_scale"][..., None] * seen, seen,
              bl["values"], bl["present"][..., None].float()]
    return torch.cat(parts, -1)


def frame_dim(nj=NJ, root=True):
    return nj * 6 + nj + nj * 6 + (8 if root else 0) + 2 * (6 + 1 + 2 * NF + 2 + 1 + 1 + 1) + (6 + 2 + 1 + 1) + N_BLEND + 1


FRAME_DIM = frame_dim()


class Merger(nn.Module):
    def __init__(self, enc=128, hidden=512, nj=NJ, root=True):
        super().__init__()
        n = len(OFFSETS)
        self.nj, self.root = nj, root
        self.frame = nn.Sequential(nn.Linear(frame_dim(nj, root), 256), nn.GELU(), nn.Linear(256, enc))
        self.offset_emb = nn.Parameter(torch.zeros(n, enc))
        self.n_out = nj * 7 + 2 * NF * 2 + (8 if root else 0)
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
        nj = self.nj
        r = o[..., :nj * 6].unflatten(-1, (nj, 6))
        k = nj * 6
        f = k + nj + 2 * NF
        out = {"res": rot6d_to_mat(r + IDENTITY_6D.to(r)), "log_s": o[..., k:k + nj],
               "fingers_res": o[..., k + nj:f].unflatten(-1, (2, NF)),
               "fingers_log_s": o[..., f:f + 2 * NF].unflatten(-1, (2, NF))}
        if self.root:
            a = o[..., f + 2 * NF:]
            out.update(root_res=a[..., :3], root_log_s=a[..., 3:5], up_res=a[..., 5:8])
        return out


def load_merger(ck):
    """Checkpoint dict -> Merger of the right shape (mg1/mg2: 17 joints, no root)."""
    sd = ck["model"]
    din = sd["frame.0.weight"].shape[1]
    for nj, root in ((NJ, True), (17, False), (NJ, False), (17, True)):
        if frame_dim(nj, root) == din:
            m = Merger(nj=nj, root=root)
            m.load_state_dict(sd)
            return m
    raise ValueError(f"no merger shape with frame dim {din}")


def apply_root(root, up, out):
    """Body model root (..., 3) and up (..., 3), merger output -> corrected root and up."""
    r = out["root_res"]
    z = root[..., 2] * torch.exp(r[..., 2])
    x = (root[..., 0] / root[..., 2] + r[..., 0]) * z
    y = (root[..., 1] / root[..., 2] + r[..., 1]) * z
    return torch.stack([x, y, z], -1), nn.functional.normalize(up + out["up_res"], dim=-1)


def history(feats, newest):
    """feats (B, K, D) oldest first, newest frame index -> (B, len(OFFSETS), D) newest first."""
    return feats[:, [newest - o for o in OFFSETS]]


def corrected(base, res):
    """Base locals (..., NJ, 3, 3) and global residuals -> corrected locals."""
    return locals_from_globals(globals_from_locals(base) @ res)


def apply(model, b, feats, newest):
    """Merger output at frame `newest` -> body locals (B, nj, 3, 3), log scales, fingers (B, 2, NF) and log scales,
    and (root mergers) a dict root (B, 3), root_log_s (B, 2), up (B, 3), else None."""
    out = model(history(feats, newest))
    L = corrected(b["base"][:, newest, :model.nj], out["res"])
    fingers = torch.stack([b["hand"][s]["fingers"][:, newest] for s in "lr"], 1) + out["fingers_res"]
    aux = None
    if model.root:
        root, up = apply_root(b["body"]["root"][:, newest], b["body"]["up"][:, newest], out)
        aux = {"root": root, "root_log_s": out["root_log_s"], "up": up}
    return L, out["log_s"], fingers, out["fingers_log_s"], aux
