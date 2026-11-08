"""Seated motion and limit clamping on batches (torch). Mirrors mttrain.seated: same distributions,
not the same random stream. Model parameters are (B, N, P) at the source rate (30 fps)."""

import math

import torch

from ..seated import ARM_KEYS, LEG_ZERO, REST

REST_NAMES = list(REST)


def min_jerk(x):
    x = x.clamp(0.0, 1.0)
    return x ** 3 * (10 - 15 * x + 6 * x * x)


def slow_signals(gen, B, n, fps, sigma, tau, device):
    """Smooth random signals (B, S, n): AR(1) with time constants tau (S,) s and stationary std
    sigma (S,), as seated._slow, all signals in one time loop."""
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    tau = torch.as_tensor(tau, dtype=torch.float32, device=device)
    r = torch.exp(-1.0 / (fps * tau))
    e = torch.randn(B, len(tau), n, generator=gen, device=device) * (sigma * torch.sqrt(1 - r * r))[:, None]
    out = torch.empty(B, len(tau), n, device=device)
    out[..., 0] = torch.randn(B, len(tau), generator=gen, device=device) * sigma
    for t in range(1, n):
        out[..., t] = r * out[..., t - 1] + e[..., t]
    return out


def smooth(x, fps, seconds):
    """Centred moving average with edge padding over the last axis (seated._smooth)."""
    k = max(1, int(fps * seconds)) | 1
    pad = torch.nn.functional.pad(x[:, None], (k // 2, k // 2), mode="replicate")
    return torch.nn.functional.avg_pool1d(pad, k, stride=1)[:, 0]


def unwrap(x):
    d = torch.diff(x, dim=-1)
    d = d - 2 * math.pi * torch.round(d / (2 * math.pi))
    return torch.cat([x[..., :1], x[..., :1] + torch.cumsum(d, -1)], -1)


def rest_weights(gen, B, n, fps, device, p_rest=0.6, max_cover=0.6, slots=3):
    """Per-frame blend weight (B, n) toward a rest posture and the template index per frame (B, n),
    as seated.rest_weights (a 3-4 s crop holds at most a couple of rest stretches)."""
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, generator=gen, device=device)
    x = torch.arange(n, device=device, dtype=torch.float32)[None]
    w = torch.zeros(B, n, device=device)
    names = torch.zeros(B, n, dtype=torch.long, device=device)
    on = torch.rand(B, generator=gen, device=device) < p_rest
    t = torch.floor(u(0, 4) * fps)
    covered = torch.zeros(B, device=device)
    for _ in range(slots):
        active = on & (t < n) & (covered < max_cover * n)
        hold = torch.floor(u(2, 8) * fps)
        ramp = torch.clamp(torch.floor(u(0.4, 1.2) * fps), min=1)
        name = torch.randint(len(REST_NAMES), (B,), generator=gen, device=device)
        end = t + 2 * ramp + hold
        a, b = t, torch.clamp(end, max=n)
        rel = x - a[:, None]
        inside = (x >= a[:, None]) & (x < b[:, None]) & active[:, None]
        # cut off by the crop end: continues past it, no fade-out (as seated.rest_weights)
        seg = torch.minimum(min_jerk(rel / ramp[:, None]), min_jerk(((end - a)[:, None] - rel) / ramp[:, None]))
        seg = torch.where(inside, seg, torch.zeros_like(seg))
        take = seg > w
        w = torch.where(take, seg, w)
        names = torch.where(take, name[:, None], names)
        covered = covered + torch.where(active, b - a, torch.zeros_like(a))
        t = b + torch.floor(u(1, 8) * fps)
    return w, names


class Seater(torch.nn.Module):
    def __init__(self, pn, limits):
        super().__init__()
        self.pn = pn
        self.P = {n: i for i, n in enumerate(pn)}
        lo = torch.full((len(pn),), -math.inf)
        hi = torch.full((len(pn),), math.inf)
        for i, n in enumerate(pn):
            a, b = limits.get(n, (0.0, 0.0))
            if b > a and not n.startswith("scale_"):
                lo[i], hi[i] = a, b
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        tmpl = torch.zeros(len(REST_NAMES), len(ARM_KEYS))
        has = torch.zeros(len(REST_NAMES), len(ARM_KEYS))  # template variation only on its own keys
        for r, name in enumerate(REST_NAMES):
            for k, key in enumerate(ARM_KEYS):
                tmpl[r, k] = REST[name].get(key, 0.0)
                has[r, k] = float(key in REST[name])
        self.register_buffer("tmpl", tmpl)
        self.register_buffer("tmpl_has", has)

    def clamp(self, mp):
        return torch.maximum(torch.minimum(mp, self.hi), self.lo)

    def forward(self, gen, mp, seated, fps=30.0):
        """mp (B, N, P), seated (B,) bool -> seated where requested, then clamped to the limits."""
        B, n, _ = mp.shape
        dev = mp.device
        P = self.P
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, 1, generator=gen, device=dev)
        # slow signals: legs (l, r) x 3, pelvis 6, spine 2, arm rest targets (l, r) x 10
        sig = [0.04, 0.08, 0.03] * 2 + [0.015, 0.01, 0.03, 0.02, 0.01, 0.02] + [0.02, 0.02] + [0.02] * 20
        tau = [3.0, 3.0, 3.0] * 2 + [2.0, 2.0, 4.0, 3.0, 3.0, 3.0] + [2.5, 2.5] + [1.5] * 20
        S = slow_signals(gen, B, n, fps, sig, tau, dev)
        out = mp.clone()
        for k, s in enumerate("lr"):
            for key in LEG_ZERO:
                out[:, :, P[f"{s}_{key}"]] = 0.0
            out[:, :, P[f"{s}_upleg_rz"]] = -1.5 + u(-0.25, 0.2) + S[:, 3 * k]
            out[:, :, P[f"{s}_knee_bend"]] = 1.5 + u(-0.4, 0.4) + S[:, 3 * k + 1]
            out[:, :, P[f"{s}_upleg_ry"]] = u(-0.2, 0.1) + S[:, 3 * k + 2]
            out[:, :, P[f"{s}_upleg_twist"]] = u(-0.2, 0.2)
        rx, rz = mp[:, :, P["root_rx"]], mp[:, :, P["root_rz"]]
        ry = unwrap(mp[:, :, P["root_ry"]])
        out[:, :, P["root_rx"]] = u(-0.15, 0.15) + (0.6 * (rx - rx.median(1, keepdim=True).values)).clamp(-0.4, 0.4) + S[:, 6]
        out[:, :, P["root_rz"]] = (0.5 * (rz - rz.median(1, keepdim=True).values)).clamp(-0.25, 0.25) + S[:, 7]
        out[:, :, P["root_ry"]] = (0.5 * (ry - smooth(ry, fps, 2.0))).clamp(-0.4, 0.4) + S[:, 8]
        out[:, :, P["root_tx"]] = S[:, 9]
        out[:, :, P["root_ty"]] = -4.0 + S[:, 10]
        out[:, :, P["root_tz"]] = S[:, 11]
        t = torch.arange(n, device=dev, dtype=torch.float32)[None] / fps
        breath = u(0.005, 0.02) * torch.sin(2 * math.pi * u(0.2, 0.33) * t + u(0, 2 * math.pi))
        out[:, :, P["spine_bend0"]] += u(-0.1, 0.25)
        out[:, :, P["spine_bend1"]] += breath
        out[:, :, P["spine_lean0"]] += S[:, 12]
        out[:, :, P["spine_twist0"]] += S[:, 13]
        for k, s in enumerate("lr"):
            w, names = rest_weights(gen, B, n, fps, dev)
            # one template variation per sequence and template (seated: per arm, per template name)
            tm = self.tmpl[None] + 0.08 * self.tmpl_has * torch.randn(B, len(REST_NAMES), len(ARM_KEYS),
                                                                      generator=gen, device=dev)
            tsel = torch.gather(tm, 1, names[..., None].expand(B, n, len(ARM_KEYS)))  # (B, n, keys)
            for j, key in enumerate(ARM_KEYS):
                i = P[f"{s}_{key}"]
                target = tsel[..., j] + S[:, 14 + 10 * k + j]
                out[:, :, i] = (1 - w) * out[:, :, i] + w * target
        return self.clamp(torch.where(seated[:, None, None], out, mp))
