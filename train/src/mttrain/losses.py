"""M1 losses: rotation (confidence-weighted), FK position, velocity, joint limits.

Joint limits: a proxy for MHR's [Limits]. Each predicted local rotation's rotation vector must stay
inside a per-joint box measured on the training targets (which are clamped to MHR's limits, see
`seated.clamp`). Cheap and differentiable; exact limit checks on MHR parameters happen in evaluation.
"""

import torch

from .skeleton import OUT_JOINTS, POS_JOINTS, fk

POS_IDX = [OUT_JOINTS.index(j) for j in POS_JOINTS]


def rot_error(R, Rt):
    """Frobenius distance per joint (..., J)."""
    return (R - Rt).flatten(-2).norm(dim=-1)


def rotvec(R):
    """Rotation matrices (..., 3, 3) -> rotation vectors (..., 3); fine away from 180 degrees."""
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    ang = torch.acos(cos)
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    return w * (ang / (2 * torch.sin(ang).clamp_min(1e-6)))[..., None]


def limit_box(rot, lo_q=0.001, hi_q=0.999, margin=0.05):
    """Per-joint box (2, J, 3) from target rotations (N, J, 3, 3). The root is global: unbounded."""
    rv = rotvec(rot)
    lo = torch.quantile(rv, lo_q, dim=0) - margin
    hi = torch.quantile(rv, hi_q, dim=0) + margin
    lo[0], hi[0] = -10.0, 10.0
    return torch.stack([lo, hi])


def losses(rot, log_scale, batch, box, w):
    """rot (B, K, J, 3, 3), log_scale (B, K, J); batch from the dataset; w: loss weights dict."""
    rt, off = batch["rot"], batch["off"]
    e = rot_error(rot, rt)
    l_rot = (torch.exp(-log_scale) * e + log_scale).mean()  # Laplace NLL: confidence = exp(-log_scale)
    _, P = fk(rot, off)
    _, Pt = fk(rt, off)
    l_pos = (P[..., POS_IDX, :] - Pt[..., POS_IDX, :]).norm(dim=-1).mean()
    dP, dPt = P[:, 1:] - P[:, :-1], Pt[:, 1:] - Pt[:, :-1]
    dR, dRt = rot[:, 1:] - rot[:, :-1], rt[:, 1:] - rt[:, :-1]
    l_vel = (dP[..., POS_IDX, :] - dPt[..., POS_IDX, :]).norm(dim=-1).mean() + (dR - dRt).flatten(-2).norm(dim=-1).mean()
    rv = rotvec(rot)
    l_lim = (torch.relu(rv - box[1]) + torch.relu(box[0] - rv)).sum(-1).mean()
    total = w["rot"] * l_rot + w["pos"] * l_pos + w["vel"] * l_vel + w["lim"] * l_lim
    return total, dict(rot=l_rot.item(), pos=l_pos.item(), vel=l_vel.item(), lim=l_lim.item())


@torch.no_grad()
def metrics(rot, batch, box):
    """Readable validation numbers: rotation error (deg) per group, position error (cm), limit violations."""
    from .skeleton import GROUPS
    rt = batch["rot"]
    rel = rt.transpose(-1, -2) @ rot
    ang = torch.rad2deg(torch.acos(((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))
    out = {f"deg_{g}": ang[..., [OUT_JOINTS.index(j) for j in js]].mean().item() for g, js in GROUPS.items()}
    _, P = fk(rot, batch["off"])
    _, Pt = fk(rt, batch["off"])
    err = (P - Pt).norm(dim=-1) * 100
    for j in ("l_wrist", "l_lowarm", "c_head"):
        out[f"cm_{j}"] = err[..., OUT_JOINTS.index(j)].mean().item()
    out["cm_wrists_seated"] = err[batch["seated"] > 0.5][..., [OUT_JOINTS.index("l_wrist"), OUT_JOINTS.index("r_wrist")]].mean().item()
    rv = rotvec(rot)
    out["limit_violation_frac"] = ((rv > box[1]) | (rv < box[0])).any(-1).float().mean().item()
    return out
