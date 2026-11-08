"""Model input features on padded batches (torch). Mirrors mttrain.features exactly.

Padded frames have time +inf (so searches never land on them) and valid=False."""

import torch

from .. import features as F


def causal_scale(times, world, vis, present, valid):
    """Median shoulder width over the trailing SCALE_WINDOW, per frame (B, T); carries the last
    value forward when the window has none, DEFAULT_SCALE before the first."""
    B, T = times.shape
    w = (world[..., F.L_SH, :] - world[..., F.R_SH, :]).norm(dim=-1)
    ok = valid & present & (vis[..., F.L_SH] > 0.5) & (vis[..., F.R_SH] > 0.5) & (w > 0.1) & (w < 0.8)
    t = torch.where(valid, times, torch.zeros_like(times))
    s_le_t = torch.ones(T, T, dtype=torch.bool, device=times.device).tril()
    M = s_le_t[None] & (t[:, None, :] >= t[:, :, None] - F.SCALE_WINDOW) & ok[:, None, :]  # (B, t, s)
    vals = torch.where(M, w[:, None, :], torch.full_like(M, float("inf"), dtype=w.dtype))
    vals, _ = vals.sort(-1)
    cnt = M.sum(-1)
    lo = ((cnt - 1).clamp_min(0) // 2)[..., None]
    hi = (cnt // 2).clamp(max=T - 1)[..., None]
    med = 0.5 * (vals.gather(-1, lo) + vals.gather(-1, hi))[..., 0]
    has = cnt > 0
    ar = torch.arange(T, device=times.device)[None].expand(B, T)
    last = torch.where(has, ar, torch.full_like(ar, -1)).cummax(-1).values
    out = torch.where(last >= 0, med.gather(-1, last.clamp_min(0)), torch.full_like(med, F.DEFAULT_SCALE))
    return out


def frame_features(times, world, image, vis, present, valid):
    """(B, T, FRAME_DIM)."""
    B, T = times.shape
    s = causal_scale(times, world, vis, present, valid)
    f = torch.cat([
        (world / s[..., None, None]).reshape(B, T, -1),
        torch.cat([image[..., :2] - 0.5, image[..., 2:3]], -1).reshape(B, T, -1),
        vis,
        present[..., None].float(),
    ], -1)
    f[..., :-1] = torch.where(present[..., None], f[..., :-1], torch.zeros_like(f[..., :-1]))
    return f.clamp(-10, 10)


def history_indices(times, targets):
    """times (B, T) (+inf padding), targets (B, K) -> idx (B, K, O) (-1 = before the first frame),
    age (B, K, O)."""
    off = torch.tensor(F.OFFSETS, dtype=times.dtype, device=times.device)
    tt = times.gather(1, targets)
    want = tt[..., None] - off + 0.02
    idx = torch.searchsorted(times.contiguous(), want.reshape(len(times), -1).contiguous(), right=True)
    idx = idx.reshape(want.shape) - 1
    picked = times.gather(1, idx.clamp_min(0).reshape(len(times), -1)).reshape(idx.shape)
    age = torch.where(idx >= 0, tt[..., None] - picked, torch.zeros_like(want))
    return idx, age


def model_input(feats, idx, age):
    """(B, K, O, FRAME_DIM + 2)."""
    B, K, O = idx.shape
    x = feats.gather(1, idx.clamp_min(0).reshape(B, K * O, 1).expand(B, K * O, feats.shape[-1])).reshape(B, K, O, -1)
    missing = idx < 0
    x = torch.where(missing[..., None], torch.zeros_like(x), x)
    return torch.cat([x, age[..., None], missing[..., None].float()], -1)
