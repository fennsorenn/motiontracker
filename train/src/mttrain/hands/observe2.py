"""Synthetic observations of both hands for the pair hand model (v2): camera, MediaPipe-style image and 3D hand
landmarks, and the detector failures the model must learn to handle (owner, 2026-11-12): palm/back flips,
left/right swaps, doubled hands, lost hands, more of them when the hands are close.

Calibrated on rendered hands (`calibrate`, `imagecal`, `failures`), mixed over both detectors (Holistic, Hand
Landmarker) with no detector flag (owner: stay detector-agnostic): every rate is drawn per sequence from a
range covering both, and flips also from the real session's much lower rate (~1%).

Image points (`image_t`): x, y projected in units of the frame width (x right, y down), z depth relative to
the wrist on the same scale, like `imagecal.image_points`. 3D shape: canonical palm-frame form
(`observe.Observer`, v1's calibration).
"""

import math

import torch

from ..gpugen.geom import sample_desk_camera
from .observe import Observer, canonical_t

ASPECT = 4 / 3
WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17
MIRROR_X = torch.tensor([-1.0, 1.0, 1.0])


def palm_frame_t(P):
    """(..., 21, 3) -> rows fwd, thumbward, palm (..., 3, 3) and the wrist -> middle knuckle length (...)."""
    f = P[..., MIDDLE_MCP, :] - P[..., WRIST, :]
    length = f.norm(dim=-1)
    f = f / length.clamp_min(1e-9)[..., None]
    s = P[..., INDEX_MCP, :] - P[..., PINKY_MCP, :]
    s = s - (s * f).sum(-1, keepdim=True) * f
    s = s / s.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.stack([f, s, torch.cross(f, s, dim=-1)], -2), length


def image_t(C, fx):
    """Camera-frame points (..., 21, 3) -> image points: projected x, y; z relative to the wrist, same scale."""
    z0 = C[..., WRIST:WRIST + 1, 2].clamp_min(1e-3)
    f = fx.reshape(fx.shape + (1,) * (C.dim() - 1 - fx.dim()))
    x = f * C[..., 0] / C[..., 2].clamp_min(1e-3)
    y = f * C[..., 1] / C[..., 2].clamp_min(1e-3)
    z = f * (C[..., 2] - z0) / z0
    return torch.stack([x, y, z], -1)


def _runs(B, F, rate, max_len, gen, dev):
    """(B, F) bool: runs of 1..max_len frames starting at per-frame probability rate (B, 1)."""
    start = torch.rand(B, F, device=dev, generator=gen) < rate
    length = torch.randint(1, max_len + 1, (B, F), device=dev, generator=gen)
    out = torch.zeros(B, F, dtype=torch.bool, device=dev)
    for k in range(max_len):
        shifted = torch.roll(start & (length > k), k, 1)
        shifted[:, :k] = False
        out |= shifted
    return out


STATES = ("seen", "wrong", "lost", "out")
OUT_FADE = 0.15  # out-of-view scores fade with the wrist's distance past the frame edge (frame widths)


class PairObserver:
    def __init__(self, calib_json, device, scores_json=None):
        self.shape = Observer(calib_json, device)
        self.device = device
        # detector confidence per slot (`scorecal`): quantile tables (detector, state, q) and lag-1 correlations
        self.scores = None
        if scores_json:
            import json
            cal = json.load(open(scores_json))
            dets = sorted(cal)
            self.scores = torch.tensor([[cal[d][st]["q"] if st in cal[d] else [0.0] * 21 for st in STATES] for d in dets],
                                       device=device)
            self.rho = torch.tensor([[cal[d][st].get("rho", 0.9) if st in cal[d] else 0.9 for st in STATES] for d in dets],
                                    device=device).nan_to_num(0.9).clamp(0.0, 0.995)

    def _score(self, state, excess, gen):
        """Slot states (B, F, 2) long (index into STATES), wrist distance past the frame edge (B, F, 2) -> detector
        score (B, F, 2): a detector per sequence, scores from its calibrated quantiles through an AR(1) Gaussian
        copula over the frames (scores change slowly)."""
        B, F, _ = state.shape
        dev = self.device
        det = torch.randint(0, self.scores.shape[0], (B,), device=dev, generator=gen)
        rho = self.rho[det][:, None, None, :].expand(B, F, 2, -1).gather(-1, state[..., None])[..., 0]
        eps = torch.randn(B, F, 2, device=dev, generator=gen)
        z = torch.empty_like(eps)
        z[:, 0] = eps[:, 0]
        for k in range(1, F):
            z[:, k] = rho[:, k] * z[:, k - 1] + torch.sqrt(1 - rho[:, k] ** 2) * eps[:, k]
        u = (0.5 * (1 + torch.erf(z / math.sqrt(2)))).clamp(0, 1) * 20  # position in the 21 quantiles
        lo = u.floor().long().clamp(0, 19)
        tab = self.scores[det][:, None, None].expand(B, F, 2, -1, -1)  # (B, F, 2, states, 21)
        tab = tab.gather(3, state[..., None, None].expand(B, F, 2, 1, 21))[..., 0, :]
        a, b = tab.gather(-1, lo[..., None])[..., 0], tab.gather(-1, (lo + 1)[..., None])[..., 0]
        sc = a + (b - a) * (u - lo)
        # renders rarely take a hand far out of view; real hands dropped below the frame read ~0.1 (Holistic)
        return torch.where(state == STATES.index("out"), sc * torch.exp(-excess / OUT_FADE), sc)

    def __call__(self, hands, body, gen, camera=None):
        """hands (B, F, 2, 21, 3) MHR metres, slots l, r, frames oldest first; body (B, T, 33, 3) for the camera
        (or `camera` = (R, t, fx) given, e.g. the merger's shared camera).
        Returns obs dict (slot-major tensors (B, F, 2, ...)) and truth dict (camera-frame quantities)."""
        dev = self.device
        B, F = hands.shape[:2]
        u = lambda *s: torch.rand(*s, device=dev, generator=gen)
        n = lambda *s: torch.randn(*s, device=dev, generator=gen)
        R, t, fx = sample_desk_camera(body, gen) if camera is None else camera
        C = torch.einsum("bij,bfskj->bfski", R, hands - t[:, None, None, None])
        img = image_t(C, fx[:, None, None])  # (B, F, 2, 21, 3)
        _, size = palm_frame_t(img)  # (B, F, 2) palm length in frame widths
        inside = ((img[..., 0].abs() < 0.5) & (img[..., 1].abs() < 0.5 / ASPECT)).all(-1) & (C[..., 2] > 0.05).all(-1)
        canon_true = torch.stack([canonical_t(C[:, :, 0]), canonical_t(C[:, :, 1] * MIRROR_X.to(dev))], 2)

        # image landmarks: shrunk hand (MediaPipe sees it ~15% small), AR(1) point noise in hand lengths,
        # depth flattened per sequence
        rel = img - img[..., WRIST:WRIST + 1, :]
        shrink = (0.85 + 0.1 * n(B, 1, 1, 1, 1)).clamp(0.6, 1.1)
        zscale = 0.4 + 0.8 * u(B, 1, 1, 1, 1)
        rho = 0.8
        eps = n(B, F, 2, 21, 3) * torch.tensor([0.10, 0.10, 0.25], device=dev)
        noise = torch.empty_like(eps)
        noise[:, 0] = eps[:, 0]
        for k in range(1, F):
            noise[:, k] = rho * noise[:, k - 1] + math.sqrt(1 - rho ** 2) * eps[:, k]
        level = 0.6 + 0.8 * u(B, 1, 1, 1, 1)
        rel_obs = rel * shrink * torch.cat([torch.ones(B, 1, 1, 1, 2, device=dev), zscale], -1) \
            + level * noise * size[..., None, None]
        wrist_obs = img[..., WRIST, :2] + 0.12 * size[..., None] * n(B, F, 2, 2)
        img_obs = rel_obs + torch.cat([wrist_obs, torch.zeros(B, F, 2, 1, device=dev)], -1)[..., None, :]
        # 3D shape (canonical), v1's observation model, per slot (the observer runs per hand)
        canon_obs = torch.stack([self.shape(canon_true[:, :, s], gen)[0] for s in range(2)], 2)

        # palm/back flips: depth inverted in both, in runs; per sequence rate 0.3-12% (real video ~1%)
        flip_rate = torch.exp(math.log(0.003) + u(B, 1) * math.log(40.0)) / 3.0
        flip = torch.stack([_runs(B, F, flip_rate, 6, gen, dev) for _ in range(2)], 2)
        zflip = torch.where(flip[..., None, None], torch.tensor(-1.0, device=dev), torch.tensor(1.0, device=dev))
        img_obs = torch.cat([img_obs[..., :2], (img_obs[..., 2:] - img_obs[..., WRIST:WRIST + 1, 2:]) * zflip
                             + img_obs[..., WRIST:WRIST + 1, 2:]], -1)
        canon_obs = canon_obs * torch.cat([torch.ones_like(zflip).expand(-1, -1, -1, 1, 2), zflip], -1)

        # how close the hands are (hand lengths between hand centres in the image)
        centre = img[..., [WRIST, MIDDLE_MCP], :2].mean(-2)
        gap = (centre[:, :, 0] - centre[:, :, 1]).norm(dim=-1) / size.mean(-1).clamp_min(1e-6)  # (B, F)
        close = (gap < 3.0).float()

        # swaps: slots exchanged for runs, more when close (renders: 6% overall, 19% within 1.5 hand lengths)
        swap_rate = (0.005 + 0.06 * close) * (2.0 * u(B, 1))
        swap = _runs(B, F, swap_rate / 3, 10, gen, dev)
        # doubled: both slots show the same hand, for runs, mostly when close (12-16% of two-hand frames)
        dbl_rate = (0.01 + 0.08 * close) * (2.0 * u(B, 1))
        dbl = _runs(B, F, dbl_rate / 3, 8, gen, dev)
        dbl_src = (u(B, 1) < 0.5).long().expand(B, F)  # which hand is shown twice
        # lost: per hand, runs; more when the hands touch; out of frame always
        loss_rate = (0.25 * u(B, 1)) / 4 + 0.1 * (gap < 1.5).float()  # (B, F)
        lost = torch.stack([_runs(B, F, loss_rate, 8, gen, dev) for _ in range(2)], 2) | ~inside

        def swap_slots(x, m):
            m = m.reshape(m.shape + (1,) * (x.dim() - 2))
            return torch.where(m, x.flip(2), x)
        img_obs, canon_obs = swap_slots(img_obs, swap), swap_slots(canon_obs, swap)
        lost = swap_slots(lost, swap)
        # doubled: copy the source slot onto the other slot
        idx = torch.stack([dbl_src, dbl_src], 2)  # (B, F, 2): both slots take the source
        def double(x):
            src = torch.gather(x, 2, idx.reshape(B, F, 2, *([1] * (x.dim() - 3))).expand_as(x))
            m = dbl.reshape(B, F, 1, *([1] * (x.dim() - 3)))
            return torch.where(m, src, x)
        img_obs, canon_obs = double(img_obs), double(canon_obs)
        lost = torch.where(dbl[..., None], torch.gather(lost, 2, idx), lost)

        obs = {"img": img_obs, "canon": canon_obs, "lost": lost}
        if self.scores is not None:
            # slot states: what the slot shows (its own hand, the other one, nothing in view / out of view)
            wrong = swap[..., None].expand(B, F, 2) | (dbl[..., None] & (torch.arange(2, device=dev)[None, None] != dbl_src[..., None]))
            state = torch.where(lost, torch.where(inside, 2, 3), torch.where(wrong, 1, 0)).long()
            w = img[..., WRIST, :2]
            excess = torch.maximum((w[..., 0].abs() - 0.5).clamp_min(0), (w[..., 1].abs() - 0.5 / ASPECT).clamp_min(0))
            obs["score"] = self._score(state, excess, gen)
        R_true, _ = palm_frame_t(img)  # left hands: palm side; right hands' frames are left-handed (mirror later)
        return (obs,
                {"img": img, "cam": C, "rot": R_true, "size": size, "inside": inside, "flip": flip, "swap": swap, "dbl": dbl})
