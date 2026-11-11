"""M1 body lifter: causal, history at fixed time offsets -> 6D rotations + per-joint confidence.

Each history frame (features + age + missing flag) goes through a shared frame encoder plus an
offset embedding; differences between neighbouring offsets are fed explicitly (motion, not just
pose; PLAN.md §3.6). An MLP maps everything to one 6D rotation and one log-scale per output joint.
Outputs are rest-relative, so a zero network output is the rest pose. Stateless per call: the
caller keeps a ring buffer of past frames (`features.history_indices`).
"""

import torch
from torch import nn

from .features import FRAME_DIM, OFFSETS
from .skeleton import OUT_JOINTS

IDENTITY_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def rot6d_to_mat(x):
    """(..., 6) -> (..., 3, 3): Gram-Schmidt on the first two columns (Zhou et al. 2019)."""
    a, b = x[..., :3], x[..., 3:]
    c1 = nn.functional.normalize(a, dim=-1)
    c2 = nn.functional.normalize(b - (c1 * b).sum(-1, keepdim=True) * c1, dim=-1)
    return torch.stack([c1, c2, torch.cross(c1, c2, dim=-1)], -1)


class Lifter(nn.Module):
    def __init__(self, enc=128, hidden=512, n_offsets=len(OFFSETS), in_dim=FRAME_DIM + 2):
        super().__init__()
        self.n_out = len(OUT_JOINTS)
        self.frame = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, enc))
        self.offset_emb = nn.Parameter(torch.zeros(n_offsets, enc))
        self.mlp = nn.Sequential(
            nn.LayerNorm((2 * n_offsets - 1) * enc), nn.Linear((2 * n_offsets - 1) * enc, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, self.n_out * 7))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        """x (..., O, in_dim) -> rot (..., J, 3, 3), log_scale (..., J)."""
        e = self.frame(x) + self.offset_emb
        h = torch.cat([e, e[..., :-1, :] - e[..., 1:, :]], -2).flatten(-2)
        out = self.mlp(h).unflatten(-1, (self.n_out, 7))
        return rot6d_to_mat(out[..., :6] + IDENTITY_6D.to(out)), out[..., 6]
