"""MediaPipe-like observation noise on batches (torch). Mirrors mttrain.noise.observe and
NoiseConfig.sample: same distributions, not the same random stream."""

import torch

from .. import noise as nz


def interp(x, curve):
    """np.interp for a fixed breakpoint curve (xs, ys), clamped at the ends."""
    xs = torch.tensor(curve[0], dtype=x.dtype, device=x.device)
    ys = torch.tensor(curve[1], dtype=x.dtype, device=x.device)
    i = torch.bucketize(x.clamp(xs[0], xs[-1]), xs).clamp(1, len(xs) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x.clamp(xs[0], xs[-1]) - x0) / (x1 - x0)


def sample_config(B, gen, device):
    """NoiseConfig.sample, batched: name -> (B,) tensor (fixed fields as constants)."""
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, generator=gen, device=device)
    n = lambda: torch.randn(B, generator=gen, device=device)
    return dict(rho10=(0.85 + 0.05 * n()).clamp(0.6, 0.95), z_factor=u(2.0, 4.0), image_per_metre=u(0.8, 1.6),
                jitter_scale=torch.exp(0.35 * n()), guess_sigma=u(0.02, 0.12), vis_tau_fall=u(0.3, 1.0),
                vis_tau_rise=u(0.1, 0.5), hip_vis_logit=u(-2.5, -0.7), glitch_rate=u(0.0, 0.01),
                dropout_rate=u(0.0, 0.02), image_z_noise=u(1.4, 2.8),
                vis_logit_sigma=0.5, guess_rho10=0.97, glitch_sigma=0.08)


def outside_distance(image):
    x, y = image[..., 0], image[..., 1]
    return torch.maximum(torch.maximum(x - 1, -x), torch.maximum(y - 1, -y))


def observe(gen, world, image, dt, cfg):
    """world, image (B, T, 33, 3) clean; dt (B, T) s -> noisy world, noisy image, vis (B, T, 33),
    present (B, T)."""
    B, T = dt.shape
    dev = world.device
    c = {k: (v[:, None] if torch.is_tensor(v) else v) for k, v in cfg.items()}  # (B, 1)
    d = outside_distance(image)
    target = interp(d, nz.VIS_CURVE)
    # one time loop for the lagged visibility and the three AR(1) noise streams
    # channels: 33 visibility logits (rho 0.8), 99 jitter (rho10), 99 guesses (guess_rho10)
    r_vis = 0.8 ** (dt / 0.1)
    r_jit = c["rho10"] ** (dt / 0.1)
    r_gue = c["guess_rho10"] ** (dt / 0.1)
    R = torch.cat([r_vis[..., None].expand(B, T, 33), r_jit[..., None].expand(B, T, 99),
                   r_gue[..., None].expand(B, T, 99)], -1)
    E = torch.randn(B, T, 231, generator=gen, device=dev)
    Z = torch.empty_like(E)
    Z[:, 0] = E[:, 0]
    v = torch.empty_like(target)
    v[:, 0] = target[:, 0]
    fall, rise = c["vis_tau_fall"], c["vis_tau_rise"]
    for t in range(1, T):
        r = R[:, t]
        Z[:, t] = r * Z[:, t - 1] + torch.sqrt(1 - r * r) * E[:, t]
        tau = torch.where(target[:, t] < v[:, t - 1], fall, rise)
        a = 1 - torch.exp(-dt[:, t, None] / tau.clamp_min(1e-6))
        v[:, t] = v[:, t - 1] + a * (target[:, t] - v[:, t - 1])
    v = v.clamp(1e-6, 1 - 1e-6)
    logit = torch.log(v / (1 - v))
    logit[..., nz.LOWER_BODY] += c["hip_vis_logit"][..., None]
    vis = torch.sigmoid(logit + c["vis_logit_sigma"] * Z[..., :33])

    base = torch.tensor(nz.BASE_SIGMA, dtype=torch.float32, device=dev)
    zg = torch.tensor(nz.Z_GAIN, dtype=torch.float32, device=dev)
    ig = torch.tensor(nz.IMAGE_GAIN, dtype=torch.float32, device=dev)
    sigma = c["jitter_scale"][..., None] * base * interp(d, nz.JITTER_CURVE)
    mix = torch.tensor(nz.MIX, dtype=torch.float32, device=dev)
    n = torch.einsum("kj,btjc->btkc", mix, Z[..., 33:132].reshape(B, T, 33, 3)) * sigma[..., None]
    n[..., 2] = n[..., 2] * c["z_factor"][..., None] * zg
    w = ((d - 0.15) / 0.25).clamp(0, 1)[..., None]
    n = n + w * c["guess_sigma"][..., None, None] * torch.einsum("kj,btjc->btkc", mix, Z[..., 132:].reshape(B, T, 33, 3))

    # glitches: a limb jumps for 1-3 frames
    groups = torch.zeros(len(nz.LIMB_GROUPS), 33, device=dev)
    for g, members in enumerate(nz.LIMB_GROUPS):
        groups[g, members] = 1.0
    ev = torch.rand(B, T, generator=gen, device=dev) < c["glitch_rate"]
    gi = torch.randint(len(nz.LIMB_GROUPS), (B, T), generator=gen, device=dev)
    dur = torch.randint(1, 4, (B, T), generator=gen, device=dev)
    jump = torch.randn(B, T, 3, generator=gen, device=dev) * c["glitch_sigma"]
    onehot = torch.nn.functional.one_hot(gi, len(nz.LIMB_GROUPS)).float() * ev[..., None]
    G = torch.zeros(B, T, 33, 3, device=dev)
    for o in range(3):
        live = (dur > o).float()[..., None]
        contrib = torch.einsum("btg,gk,btc->btkc", onehot * live, groups, jump)
        G[:, o:] += contrib[:, :T - o]
    n = n + G

    W = world + n
    g = c["image_per_metre"][..., None] * ig  # (B, 1, 33)
    Im = image.clone()
    Im[..., 0] += g * n[..., 0]
    Im[..., 1] += g * n[..., 1] * (4 / 3)
    Im[..., 2] += c["image_z_noise"][..., None] * g * n[..., 2]
    present = torch.rand(B, T, generator=gen, device=dev) >= c["dropout_rate"]
    return W, Im, vis, present
