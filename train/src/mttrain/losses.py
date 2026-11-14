"""M1 losses: rotation (confidence-weighted), FK position, velocity, joint limits, contact.

Contact (owner, 2026-11-09): hands meeting and touching the face should stay in contact on the
avatar. The vectors between the contact points (hand-hand, each hand-nose; skeleton.CONTACT_POINTS)
after FK on the true body must match the truth, weighted by closeness in the truth
(exp(-(d / CONTACT_SCALE)^2)), so the loss is an average over near-contact pairs. Plain FK position
errors are relative to the pelvis; this one is relative between the end effectors.

Joint limits: a proxy for MHR's [Limits]. Each predicted local rotation's rotation vector must stay
inside a per-joint box measured on the training targets (which are clamped to MHR's limits, see
`seated.clamp`). Cheap and differentiable; exact limit checks on MHR parameters happen in evaluation.
"""

import torch

from .skeleton import CONTACT_PAIRS, LEG_IDX, LEG_POS_JOINTS, OUT_JOINTS, POS_JOINTS, UPPER_IDX, contact_points, fk

POS_IDX = [OUT_JOINTS.index(j) for j in POS_JOINTS]
LEG_POS_IDX = [OUT_JOINTS.index(j) for j in LEG_POS_JOINTS]
CONTACT_SCALE = 0.15  # m
PAIR_A, PAIR_B = [a for a, _ in CONTACT_PAIRS], [b for _, b in CONTACT_PAIRS]


def contact_error(G, P, Gt, Pt, cof):
    """Per pair (..., 3): vector error (m) between predicted and true contact-point differences, and
    the true distance (m)."""
    Q, Qt = contact_points(G, P, cof), contact_points(Gt, Pt, cof)
    d, dt = Q[..., PAIR_A, :] - Q[..., PAIR_B, :], Qt[..., PAIR_A, :] - Qt[..., PAIR_B, :]
    return (d - dt).norm(dim=-1), dt.norm(dim=-1)


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


def losses(rot, log_scale, batch, box, w, aux=None):
    """rot (B, K, J, 3, 3), log_scale (B, K, J); batch from the dataset; w: loss weights dict. The upper body's terms
    average over its 17 joints only, so they weigh as before the legs; legs (when predicted), root position and up
    (`aux`, m1-v12+) are separate terms: w["leg"], w["root"], w["up"]."""
    rt, off = batch["rot"], batch["off"]
    e = rot_error(rot, rt)
    nll = torch.exp(-log_scale) * e + log_scale  # Laplace NLL: confidence = exp(-log_scale)
    l_rot = nll[..., UPPER_IDX].mean()
    G, P = fk(rot, off)
    Gt, Pt = fk(rt, off)
    ce, cd = contact_error(G, P, Gt, Pt, batch["cof"])
    cw = torch.exp(-(cd / CONTACT_SCALE) ** 2)
    l_con = (cw * ce).sum() / cw.sum().clamp_min(1.0)
    l_pos = (P[..., POS_IDX, :] - Pt[..., POS_IDX, :]).norm(dim=-1).mean()
    dP, dPt = P[:, 1:] - P[:, :-1], Pt[:, 1:] - Pt[:, :-1]
    dR, dRt = rot[:, 1:] - rot[:, :-1], rt[:, 1:] - rt[:, :-1]
    dRe = (dR - dRt).flatten(-2).norm(dim=-1)
    l_vel = (dP[..., POS_IDX, :] - dPt[..., POS_IDX, :]).norm(dim=-1).mean() + dRe[..., UPPER_IDX].mean()
    rv = rotvec(rot)
    l_lim = (torch.relu(rv - box[1][:rv.shape[-2]]) + torch.relu(box[0][:rv.shape[-2]] - rv)).sum(-1).mean()
    total = w["rot"] * l_rot + w["pos"] * l_pos + w["vel"] * l_vel + w["lim"] * l_lim + w.get("con", 0.0) * l_con
    parts = dict(rot=l_rot.item(), pos=l_pos.item(), vel=l_vel.item(), lim=l_lim.item(), con=l_con.item())
    if rot.shape[-3] > len(UPPER_IDX):  # legs: the same terms with the arms' weights, as one term
        l_leg = (w["rot"] * nll[..., LEG_IDX].mean()
                 + w["pos"] * (P[..., LEG_POS_IDX, :] - Pt[..., LEG_POS_IDX, :]).norm(dim=-1).mean()
                 + w["vel"] * ((dP[..., LEG_POS_IDX, :] - dPt[..., LEG_POS_IDX, :]).norm(dim=-1).mean()
                               + dRe[..., LEG_IDX].mean()))
        total = total + w.get("leg", 1.0) * l_leg
        parts["leg"] = l_leg.item()
    if aux is not None:
        l_root, l_up = root_losses(aux, batch)
        total = total + w.get("root", 1.0) * l_root + w.get("up", 1.0) * l_up
        parts.update(root=l_root.item(), up=l_up.item())
    return total, parts


def root_losses(aux, batch):
    """Root position: Laplace NLL in the image plane (x, y) and in depth (z), each with its own log scale, plus the
    error of its frame-to-frame change (smooth without lag). Up: chord distance to the true up (~ angle, rad)."""
    p, pt, s = aux["root"], batch["root"], aux["root_log_s"]
    e_xy = (p[..., :2] - pt[..., :2]).norm(dim=-1)
    e_z = (p[..., 2] - pt[..., 2]).abs()
    l = (torch.exp(-s[..., 0]) * e_xy + s[..., 0]).mean() + (torch.exp(-s[..., 1]) * e_z + s[..., 1]).mean()
    l = l + 10.0 * ((p[:, 1:] - p[:, :-1]) - (pt[:, 1:] - pt[:, :-1])).norm(dim=-1).mean()
    return l, (aux["up"] - batch["up"]).norm(dim=-1).mean()


@torch.no_grad()
def metrics(rot, batch, box, aux=None):
    """Readable validation numbers: rotation error (deg) per group, position error (cm), limit violations."""
    from .skeleton import GROUPS
    rt = batch["rot"]
    rel = rt.transpose(-1, -2) @ rot
    ang = torch.rad2deg(torch.acos(((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))
    out = {f"deg_{g}": ang[..., [OUT_JOINTS.index(j) for j in js]].mean().item() for g, js in GROUPS.items()}
    G, P = fk(rot, batch["off"])
    Gt, Pt = fk(rt, batch["off"])
    err = (P - Pt).norm(dim=-1) * 100
    ce, cd = contact_error(G, P, Gt, Pt, batch["cof"])
    near = cd < 0.15
    out["cm_contact_near"] = (ce[near].mean() * 100).item() if near.any() else float("nan")  # pairs < 15 cm apart
    out["frac_contact_near"] = near.float().mean().item()
    for j in ("l_wrist", "l_lowarm", "c_head"):
        out[f"cm_{j}"] = err[..., OUT_JOINTS.index(j)].mean().item()
    out["cm_wrists_seated"] = err[batch["seated"] > 0.5][..., [OUT_JOINTS.index("l_wrist"), OUT_JOINTS.index("r_wrist")]].mean().item()
    rv = rotvec(rot)[..., UPPER_IDX, :]  # the 17-joint number, comparable across runs
    out["limit_violation_frac"] = ((rv > box[1][UPPER_IDX]) | (rv < box[0][UPPER_IDX])).any(-1).float().mean().item()
    if rot.shape[-3] > len(UPPER_IDX):
        for j in ("l_lowleg", "l_subtalar"):
            out[f"cm_{j}"] = err[..., OUT_JOINTS.index(j)].mean().item()
    if aux is not None:
        d = (aux["root"] - batch["root"]) * 100
        seated = batch["seated"] > 0.5
        out["cm_root_xy"] = d[..., :2].norm(dim=-1).mean().item()
        out["cm_root_z"] = d[..., 2].abs().mean().item()
        out["cm_root_y_seated"] = d[seated][..., 1].abs().mean().item() if seated.any() else float("nan")
        out["cm_root_y_standing"] = d[~seated][..., 1].abs().mean().item() if (~seated).any() else float("nan")
        cos = (aux["up"] * batch["up"]).sum(-1).clamp(-1, 1)
        out["deg_up"] = torch.rad2deg(torch.acos(cos)).mean().item()
    return out
