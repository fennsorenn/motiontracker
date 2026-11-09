"""Standing CMU motion -> seated desk motion (plan step 5).

Keeps CMU's upper body (spine, neck, head, arms are joint-local, so they carry over), and replaces
the rest:
- Legs: a seated pose (thighs forward, shins down), varied per sequence with slow drift.
- Pelvis: fixed in the chair, facing forward; CMU's pelvis pitch, roll and turn relative to the
  clip's own baseline carry over damped (leaning to reach, swivelling), plus a random slouch.
  MHR's root pitch/roll are body-relative (applied after yaw), so the deviations transfer directly.
- Idle life: breathing and slow sway on the spine and pelvis.
- Rest postures: per arm, stretches of hands in the lap, on the desk, or hanging beside the chair,
  blended in and out with minimum-jerk transitions. Arms resting below the frame is a normal
  streaming posture (PLAN.md §3.3), and standing mocap barely contains it.

Rest templates: arm parameters solved by IK on MHR's template body in the seated pose (wrist and
middle-finger targets; residual < 0.1 cm). The same values hold for both sides (MHR's parameters
are mirrored).
"""

import numpy as np

REST = {
    "lap": {"clavicle_rz": -0.029, "uparm_twist": 0.217, "uparm_ry": -0.909, "uparm_rz": 0.079,
            "elbow_bend": -0.053, "lowarm_twist": -0.030, "wrist_ry": 0.305, "wrist_rz": -0.063},
    "desk": {"clavicle_rz": -0.018, "uparm_twist": 0.014, "uparm_ry": -0.839, "uparm_rz": 0.404,
             "elbow_bend": 0.062, "lowarm_twist": -0.018, "wrist_ry": 0.276, "wrist_rz": -0.019},
    "hanging": {"clavicle_ry": 0.042, "clavicle_rz": -0.266, "uparm_twist": 0.314, "uparm_ry": -0.542,
                "uparm_rz": -0.564, "elbow_bend": -0.082, "lowarm_twist": 0.040, "wrist_ry": -0.239,
                "wrist_rz": -0.001},
}
ARM_KEYS = ["clavicle_rx", "clavicle_ry", "clavicle_rz", "uparm_twist", "uparm_ry", "uparm_rz", "elbow_bend",
            "lowarm_twist", "wrist_ry", "wrist_rz"]
LEG_ZERO = ["upleg_twist", "upleg_ry", "upleg_rz", "knee_bend", "lowleg_twist", "foot_bend", "foot_lean0",
            "foot_lean1", "ball_bend"]


def min_jerk(x):
    x = np.clip(x, 0.0, 1.0)
    return x ** 3 * (10 - 15 * x + 6 * x * x)


def _slow(rng, n, fps, sigma, tau):
    """Smooth random signal: AR(1) with time constant tau (s), stationary std sigma."""
    r = np.exp(-1.0 / (fps * tau))
    e = rng.normal(0, sigma * np.sqrt(1 - r * r), n)
    out = np.empty(n)
    out[0] = rng.normal(0, sigma)
    for t in range(1, n):
        out[t] = r * out[t - 1] + e[t]
    return out


def _smooth(x, fps, seconds):
    k = max(1, int(fps * seconds)) | 1
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def rest_weights(rng, n, fps, p_rest=0.6, max_cover=0.6):
    """Per-frame blend weight in [0, 1] toward a rest posture, and the template name per frame."""
    w = np.zeros(n)
    names = np.array([""] * n, dtype=object)
    if rng.random() >= p_rest:
        return w, names
    t = int(rng.uniform(0, 4) * fps)
    covered = 0
    while t < n and covered < max_cover * n:
        hold = int(rng.uniform(2, 8) * fps)
        ramp = max(1, int(rng.uniform(0.4, 1.2) * fps))
        name = rng.choice(list(REST))
        end = t + 2 * ramp + hold
        a, b = t, min(n, end)
        x = np.arange(a, b) - a
        # a stretch cut off by the crop end continues past it (no fade-out at the crop end, where
        # the training targets are; fixed 2026-11-08, before that targets rarely saw a full rest pose)
        seg = np.minimum(min_jerk(x / ramp), min_jerk((end - a - x) / ramp))
        take = seg > w[a:b]
        w[a:b] = np.where(take, seg, w[a:b])
        names[a:b] = np.where(take, name, names[a:b])
        covered += b - a
        t = b + int(rng.uniform(1, 8) * fps)
    return w, names


def seat(rng, pn, mp, fps=30.0, limits=None, force_rest=None):
    """Model parameters (N, P) of a standing clip -> seated version (N, P).

    `force_rest` (left template name, right template name): both arms rest in these postures for
    the whole clip instead of random rest stretches (rendered rest test sessions).

    `limits` (name -> (lo, hi), from mtcore.mhr.limits) clamps pose parameters afterwards, so
    targets never contain impossible poses (fits and random rest variations can stray past them)."""
    P = {n: i for i, n in enumerate(pn)}
    out = mp.copy()
    n = len(mp)

    # legs
    for s in "lr":
        for k in LEG_ZERO:
            out[:, P[f"{s}_{k}"]] = 0.0
        out[:, P[f"{s}_upleg_rz"]] = -1.5 + rng.uniform(-0.25, 0.2) + _slow(rng, n, fps, 0.04, 3.0)
        out[:, P[f"{s}_knee_bend"]] = 1.5 + rng.uniform(-0.4, 0.4) + _slow(rng, n, fps, 0.08, 3.0)
        out[:, P[f"{s}_upleg_ry"]] = rng.uniform(-0.2, 0.1) + _slow(rng, n, fps, 0.03, 3.0)
        out[:, P[f"{s}_upleg_twist"]] = rng.uniform(-0.2, 0.2)

    # pelvis: fixed in the chair, damped CMU lean/roll/turn around the clip's baseline
    rx, ry, rz = (mp[:, P[k]] for k in ("root_rx", "root_ry", "root_rz"))
    ry = np.unwrap(ry)
    out[:, P["root_rx"]] = rng.uniform(-0.15, 0.15) + np.clip(0.6 * (rx - np.median(rx)), -0.4, 0.4) \
        + _slow(rng, n, fps, 0.015, 2.0)
    out[:, P["root_rz"]] = np.clip(0.5 * (rz - np.median(rz)), -0.25, 0.25) + _slow(rng, n, fps, 0.01, 2.0)
    out[:, P["root_ry"]] = np.clip(0.5 * (ry - _smooth(ry, fps, 2.0)), -0.4, 0.4) + _slow(rng, n, fps, 0.03, 4.0)
    out[:, P["root_tx"]] = _slow(rng, n, fps, 0.02, 3.0)  # units of 10 cm
    out[:, P["root_ty"]] = -4.0 + _slow(rng, n, fps, 0.01, 3.0)
    out[:, P["root_tz"]] = _slow(rng, n, fps, 0.02, 3.0)

    # posture and idle life
    t = np.arange(n) / fps
    breath = rng.uniform(0.005, 0.02) * np.sin(2 * np.pi * rng.uniform(0.2, 0.33) * t + rng.uniform(0, 2 * np.pi))
    out[:, P["spine_bend0"]] += rng.uniform(-0.1, 0.25)
    out[:, P["spine_bend1"]] += breath
    out[:, P["spine_lean0"]] += _slow(rng, n, fps, 0.02, 2.5)
    out[:, P["spine_twist0"]] += _slow(rng, n, fps, 0.02, 2.5)

    # rest postures per arm
    for k, s in enumerate("lr"):
        if force_rest:
            w, names = np.ones(n), np.array([force_rest[k]] * n, dtype=object)
        else:
            w, names = rest_weights(rng, n, fps)
        if not w.any():
            continue
        for name in set(names[w > 0]):
            m = names == name
            tmpl = {k: v + rng.normal(0, 0.08) for k, v in REST[name].items()}
            for k in ARM_KEYS:
                i = P[f"{s}_{k}"]
                target = tmpl.get(k, 0.0) + _slow(rng, n, fps, 0.02, 1.5)
                out[m, i] = (1 - w[m]) * out[m, i] + w[m] * target[m]
    return clamp(pn, out, limits) if limits else out


def clamp(pn, mp, limits):
    """Clamp pose parameters with a real range to MHR's limits (scales and fixed parameters untouched)."""
    out = mp.copy()
    for i, n in enumerate(pn):
        lo, hi = limits.get(n, (0.0, 0.0))
        if hi > lo and not n.startswith("scale_"):
            out[:, i] = np.clip(out[:, i], lo, hi)
    return out
