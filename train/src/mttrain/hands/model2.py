"""Pair hand model (v2): both hands' recent history -> one hand's fingers, orientation and position.

The model runs once per hand as the "own" hand of a scene: for the left hand the scene as observed, for
the right hand the scene mirrored (x -> -x), so the own hand is always a left hand and the other hand a
right one; one set of weights serves both (MHR's right hand is the left mirrored). Each hand's last HISTORY
frames (10 Hz) go through a shared per-hand encoder (with a role embedding: own / other); a joint MLP sees
both encodings, so the model can tell swaps, doubled hands and contact (owner, 2026-11-12).

Per hand and frame (`frame_features`): canonical 3D shape (63), the palm frame of the image landmarks in the
scene (its first two rows, 6), wrist position in the image (2, frame widths), log palm length in the image,
lost flag; zeros when lost. Outputs: finger parameters (`model.OUT_PARAMS`) with Laplace log-scales, the
hand's orientation in the camera frame as the palm frame of its image points (6D, rows), a log-scale for it,
wrist position (2) and log palm length (1).
"""

import torch
from torch import nn

from .model import OUT_PARAMS
from .observe2 import palm_frame_t

HISTORY = 8
FRAME_DIM = 63 + 6 + 2 + 1 + 1
N_FING = len(OUT_PARAMS)
MIRROR_X = torch.tensor([-1.0, 1.0, 1.0])


def rows6d_to_mat(x):
    """(..., 6) -> rotation (..., 3, 3) whose first two rows come from x (Gram-Schmidt)."""
    a, b = x[..., :3], x[..., 3:]
    r1 = nn.functional.normalize(a, dim=-1)
    r2 = nn.functional.normalize(b - (r1 * b).sum(-1, keepdim=True) * r1, dim=-1)
    return torch.stack([r1, r2, torch.cross(r1, r2, dim=-1)], -2)


def frame_features(img, canon, lost, mirror):
    """One hand's frames: img (..., 21, 3) image points, canon (..., 21, 3), lost (...) bool, mirror (bool or
    (...) tensor) -> (..., FRAME_DIM)."""
    m = MIRROR_X.to(img) if isinstance(mirror, bool) else torch.where(mirror[..., None], MIRROR_X.to(img), torch.ones(3).to(img))
    if isinstance(mirror, bool):
        img = img * m if mirror else img
    else:
        img = img * m[..., None, :]
    R, size = palm_frame_t(img)
    feats = torch.cat([canon.flatten(-2), R[..., :2, :].flatten(-2), img[..., 0, :2],
                       torch.log(size.clamp_min(1e-4))[..., None]], -1)
    feats = feats * (~lost)[..., None]
    return torch.cat([feats, lost[..., None].to(feats.dtype)], -1)


def scenes(img, canon, lost):
    """Slot-major observations (B, F, 2, ...) newest first -> own/other features for both scenes:
    (2B, F, FRAME_DIM) own, (2B, F, FRAME_DIM) other; rows 0..B-1 own = left slot, B..2B-1 own = right slot."""
    own = torch.cat([frame_features(img[:, :, 0], canon[:, :, 0], lost[:, :, 0], False),
                     frame_features(img[:, :, 1], canon[:, :, 1], lost[:, :, 1], True)], 0)
    other = torch.cat([frame_features(img[:, :, 1], canon[:, :, 1], lost[:, :, 1], False),
                       frame_features(img[:, :, 0], canon[:, :, 0], lost[:, :, 0], True)], 0)
    return own, other


class PairNet(nn.Module):
    def __init__(self, enc=96, hand=256, hidden=512, history=HISTORY):
        super().__init__()
        self.frame = nn.Sequential(nn.Linear(FRAME_DIM, 192), nn.GELU(), nn.Linear(192, enc))
        self.pos_emb = nn.Parameter(torch.zeros(history, enc))
        self.role = nn.Parameter(torch.zeros(2, enc))
        width = (2 * history - 1) * enc
        self.hand = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hand), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(2 * hand, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * N_FING + 6 + 1 + 3))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def encode(self, x, role):
        e = self.frame(x) + self.pos_emb + self.role[role]
        h = torch.cat([e, e[..., :-1, :] - e[..., 1:, :]], -2).flatten(-2)
        return self.hand(h)

    def forward(self, own, other):
        """own, other (..., HISTORY, FRAME_DIM) newest first -> dict of outputs for the own hand."""
        out = self.mlp(torch.cat([self.encode(own, 0), self.encode(other, 1)], -1))
        k = 2 * N_FING
        # orientation: rest of the 6D output is a palm facing the camera-ish default (identity rows)
        rot = rows6d_to_mat(out[..., k:k + 6] + torch.tensor([1.0, 0, 0, 0, 1.0, 0]).to(out))
        return {"fingers": out[..., :N_FING], "fingers_log_s": out[..., N_FING:k], "rot": rot,
                "rot_log_s": out[..., k + 6], "pos": out[..., k + 7:k + 9], "log_size": out[..., k + 9]}
