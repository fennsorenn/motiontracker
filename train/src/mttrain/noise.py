"""MediaPipe-like observation noise for synthetic landmark sequences.

Calibrated on the first recorded session (2026-11-01, seated-v1, 10 FPS; `python -m mttrain.calibrate`):
- Still-segment jitter (world, mm, x/y/z): face and shoulders ~2/2/7, elbows ~2/4/10, wrists at the
  frame's bottom edge ~10/18/17. Depth jitter ~3x lateral. Image jitter ~1 normalized unit per metre
  of world jitter (0.6 px at 320 px for the face, ~4-10 px for edge wrists).
- Jitter is slow: lag-1 autocorrelation ~0.85 at 10 FPS -> AR(1) noise.
- Visibility vs distance outside the frame (normalized units, negative = inside): ~0.99 well
  inside, 0.96 near the edge, ~0.68 within +-0.05..0.15 of it, 0.13 at 0.15-0.3, 0.02 beyond.
- Visibility lags: when the hands drop just below the frame, wrist visibility decays over ~1.3 s
  (0.94 -> 0.48) and recovers over a few frames when they return (dropout segment) -> first-order
  lag toward the curve, slower falling than rising. Positions stay plausible meanwhile.
- Far outside the frame MediaPipe guesses, and its guesses are smooth (knee ~4 mm jitter), so
  out-of-frame points get a slowly drifting offset, not more jitter.
Synthetic vs real (`python -m mttrain.calibrate`): proportions and visibility match; synthetic
jitter at rest is ~0.6-1x real (real still segments include body sway). Base values below are
slightly above the measured jitter for that reason.
Unknown from one session (randomized instead): how wrong the guesses are, glitch and dropout rates.
"""

from dataclasses import dataclass

import numpy as np

from mtcore.landmarks import MP33

# base lateral jitter (world, metres) per landmark, in frame
_BASE = {"face": 0.0025, "shoulder": 0.0025, "elbow": 0.0035, "hand": 0.006, "hip": 0.001, "leg": 0.004}


def _group(name):
    if MP33.index(name) <= 10:
        return "face"
    for g, keys in (("shoulder", ("shoulder",)), ("elbow", ("elbow",)),
                    ("hand", ("wrist", "pinky", "index", "thumb")), ("hip", ("hip",))):
        if any(k in name for k in keys):
            return g
    return "leg"


BASE_SIGMA = np.array([_BASE[_group(n)] for n in MP33])
LIMB_GROUPS = [[MP33.index(n) for n in MP33 if side in n and any(k in n for k in parts)]
               for side in ("left", "right") for parts in (("elbow",), ("wrist", "pinky", "index", "thumb"))]

# seated hips sit just below the frame (~0.04 outside) yet get visibility 0.16-0.33, not the curve's
# ~0.67: the desk and the seated pose hide them. Logit offset per landmark (randomized per sequence).
HIP_VIS_LOGIT = -1.6
LOWER_BODY = [MP33.index(n) for n in MP33[23:]]

# (distance outside the frame, value) breakpoints
VIS_CURVE = ([-0.3, -0.15, -0.05, 0.0, 0.1, 0.2, 0.3, 0.5], [0.99, 0.97, 0.85, 0.70, 0.66, 0.25, 0.05, 0.02])
JITTER_CURVE = ([-0.3, -0.1, 0.0, 0.1, 0.2, 0.4], [1.0, 1.5, 4.0, 4.0, 2.0, 1.2])


@dataclass
class NoiseConfig:
    rho10: float = 0.85  # AR(1) coefficient per 0.1 s
    z_factor: float = 3.0  # depth jitter / lateral jitter
    image_per_metre: float = 1.2  # image jitter (normalized units) per metre of world jitter
    jitter_scale: float = 1.0
    guess_sigma: float = 0.06  # metres, drift of far out-of-frame guesses
    guess_rho10: float = 0.97
    vis_logit_sigma: float = 0.5
    vis_tau_fall: float = 0.6  # s
    vis_tau_rise: float = 0.3  # s
    hip_vis_logit: float = HIP_VIS_LOGIT
    glitch_rate: float = 0.003  # per frame
    glitch_sigma: float = 0.08
    dropout_rate: float = 0.005  # whole-frame detector misses

    @classmethod
    def sample(cls, rng):
        """Randomized around the calibrated values, per sequence."""
        return cls(rho10=float(np.clip(rng.normal(0.85, 0.05), 0.6, 0.95)),
                   z_factor=float(rng.uniform(2.0, 4.0)),
                   image_per_metre=float(rng.uniform(0.8, 1.6)),
                   jitter_scale=float(np.exp(rng.normal(0, 0.35))),
                   guess_sigma=float(rng.uniform(0.02, 0.12)),
                   vis_tau_fall=float(rng.uniform(0.3, 1.0)),
                   vis_tau_rise=float(rng.uniform(0.1, 0.5)),
                   hip_vis_logit=float(rng.uniform(-2.5, -0.7)),
                   glitch_rate=float(rng.uniform(0.0, 0.01)),
                   dropout_rate=float(rng.uniform(0.0, 0.02)))


def outside_distance(image):
    """How far each point lies outside the frame (normalized units; negative = inside). image: (..., 2+)."""
    x, y = image[..., 0], image[..., 1]
    return np.maximum(np.maximum(x - 1, -x), np.maximum(y - 1, -y))


def _ar1(rng, dt, rho10, shape):
    """Unit-variance AR(1) noise sampled at irregular times dt (T,)."""
    T = len(dt)
    out = np.empty((T,) + shape)
    out[0] = rng.normal(size=shape)
    for t in range(1, T):
        r = rho10 ** (dt[t] / 0.1)
        out[t] = r * out[t - 1] + np.sqrt(1 - r * r) * rng.normal(size=shape)
    return out


def observe(rng, world, image, dt, cfg=None):
    """Clean world (T, 33, 3) and image (T, 33, 3) landmarks at frame intervals dt (T,) seconds ->
    noisy world, noisy image, visibility (T, 33), present (T,) bool."""
    cfg = cfg or NoiseConfig()
    T = len(dt)
    d = outside_distance(image)
    # visibility: lagged curve + slow logit noise
    target = np.interp(d, *VIS_CURVE)
    v = target.copy()
    for t in range(1, T):
        tau = np.where(target[t] < v[t - 1], cfg.vis_tau_fall, cfg.vis_tau_rise)
        a = 1 - np.exp(-dt[t] / np.maximum(tau, 1e-6))
        v[t] = v[t - 1] + a * (target[t] - v[t - 1])
    logit = np.log(v / (1 - v))
    logit[:, LOWER_BODY] += cfg.hip_vis_logit
    logit = logit + cfg.vis_logit_sigma * _ar1(rng, dt, 0.8, (33,))
    vis = 1 / (1 + np.exp(-logit))

    # jitter: AR(1), shared between world and image lateral axes, scaled by the edge curve
    sigma = cfg.jitter_scale * BASE_SIGMA[None] * np.interp(d, *JITTER_CURVE)  # (T, 33)
    n = _ar1(rng, dt, cfg.rho10, (33, 3)) * sigma[..., None]
    n[..., 2] *= cfg.z_factor

    # far out-of-frame: smoothly drifting guesses
    w = np.clip((d - 0.15) / 0.25, 0, 1)[..., None]
    n += w * cfg.guess_sigma * _ar1(rng, dt, cfg.guess_rho10, (33, 3))

    # glitches: a limb jumps for 1-3 frames
    for t in np.flatnonzero(rng.random(T) < cfg.glitch_rate):
        g = LIMB_GROUPS[rng.integers(len(LIMB_GROUPS))]
        jump = rng.normal(0, cfg.glitch_sigma, 3)
        n[t:t + rng.integers(1, 4), g] += jump

    W = world + n
    Im = image.copy()
    Im[..., :2] += cfg.image_per_metre * n[..., :2] * [1.0, 4 / 3]  # y is normalized by the shorter side
    Im[..., 2] += cfg.image_per_metre * n[..., 2]
    present = rng.random(T) >= cfg.dropout_rate
    return W, Im, vis, present
