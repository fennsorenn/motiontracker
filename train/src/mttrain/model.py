"""M1 body lifter: causal, history at fixed time offsets -> 6D rotations + per-joint confidence.

Each history frame (features + age + missing flag) goes through a shared frame encoder plus an
offset embedding; differences between neighbouring offsets are fed explicitly (motion, not just
pose; PLAN.md §3.6). An MLP maps everything to one 6D rotation and one log-scale per output joint.
Outputs are rest-relative, so a zero network output is the rest pose. Stateless per call: the
caller keeps a ring buffer of past frames (`features.history_indices`).

m1-v12+ (`root=True`, 23 joints with the legs): also the root position in the camera frame (metres; parameterized as
x/z, y/z and log z around ROOT_Z0, so a zero output is the root 1.5 m straight ahead), two log scales (image plane,
depth) and the up direction (normalized; a zero output is camera-up = -y). Older checkpoints load with
`load_lifter`, which builds the 17-joint model without these heads.
"""

import torch
from torch import nn

from mtcore.features import FRAME_DIM, OFFSETS
from .skeleton import OUT_JOINTS

IDENTITY_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
ROOT_Z0 = 1.5  # m
N_AUX = 3 + 2 + 3  # root (x/z, y/z, log z), root log scales (image plane, depth), up


def rot6d_to_mat(x):
    """(..., 6) -> (..., 3, 3): Gram-Schmidt on the first two columns (Zhou et al. 2019)."""
    a, b = x[..., :3], x[..., 3:]
    c1 = nn.functional.normalize(a, dim=-1)
    c2 = nn.functional.normalize(b - (c1 * b).sum(-1, keepdim=True) * c1, dim=-1)
    return torch.stack([c1, c2, torch.cross(c1, c2, dim=-1)], -1)


class Lifter(nn.Module):
    def __init__(self, enc=128, hidden=512, n_offsets=len(OFFSETS), in_dim=FRAME_DIM + 2, n_joints=len(OUT_JOINTS),
                 root=True):
        super().__init__()
        self.n_out, self.root = n_joints, root
        self.frame = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, enc))
        self.offset_emb = nn.Parameter(torch.zeros(n_offsets, enc))
        self.mlp = nn.Sequential(
            nn.LayerNorm((2 * n_offsets - 1) * enc), nn.Linear((2 * n_offsets - 1) * enc, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, self.n_out * 7 + (N_AUX if root else 0)))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        """x (..., O, in_dim) -> rot (..., J, 3, 3), log_scale (..., J)[, aux dict: root (..., 3) metres,
        root_log_s (..., 2) image plane and depth, up (..., 3) unit]."""
        e = self.frame(x) + self.offset_emb
        h = torch.cat([e, e[..., :-1, :] - e[..., 1:, :]], -2).flatten(-2)
        o = self.mlp(h)
        out = o[..., :self.n_out * 7].unflatten(-1, (self.n_out, 7))
        rot, ls = rot6d_to_mat(out[..., :6] + IDENTITY_6D.to(out)), out[..., 6]
        if not self.root:
            return rot, ls
        a = o[..., self.n_out * 7:]
        z = ROOT_Z0 * torch.exp(a[..., 2])
        root = torch.stack([a[..., 0] * z, a[..., 1] * z, z], -1)
        up = nn.functional.normalize(a[..., 5:8] + torch.tensor([0.0, -1.0, 0.0]).to(a), dim=-1)
        return rot, ls, {"root": root, "root_log_s": a[..., 3:5], "up": up}


def load_lifter(ckpt):
    """Checkpoint dict -> Lifter of the right shape (17 joints without the root/up heads before m1-v12)."""
    sd = ckpt["model"]
    n = sd["mlp.5.bias"].shape[0]  # the last Linear of self.mlp
    root = n % 7 != 0
    m = Lifter(n_joints=(n - N_AUX) // 7 if root else n // 7, root=root)
    m.load_state_dict(sd)
    return m


def pad_joints(rot, ls):
    """17-joint outputs of older models -> all OUT_JOINTS: the legs at rest with no confidence."""
    if rot.shape[-3] == len(OUT_JOINTS):
        return rot, ls
    extra = len(OUT_JOINTS) - rot.shape[-3]
    eye = torch.eye(3, dtype=rot.dtype, device=rot.device).expand(*rot.shape[:-3], extra, 3, 3)
    return torch.cat([rot, eye], -3), torch.cat([ls, torch.full((*ls.shape[:-1], extra), 10.0, dtype=ls.dtype,
                                                                device=ls.device)], -1)
