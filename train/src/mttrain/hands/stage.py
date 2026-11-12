"""Hands-in-use arm choreography for rendered hand sessions (hand model evaluation, plan step 3).

Held-out CMU clips rarely show the hands to a desk webcam (arms behind the back, at the sides, out of
frame), so the arms are authored: a held-out subject's body proportions, seated, both arms starting in
a rest posture; each arm then moves between key poses solved with MHR's IK (`choreo._solve`): the hand
in the space in front of the chest and face, fingers pointing up, forward or inward, the palm turned
any way around them. Holds of 0.8-3.5 s, minimum-jerk moves of 0.5-1.2 s; now and then an arm goes
back to rest for a while. The fingers are set afterwards (`motion.HandMotion`), so the IK only moves
clavicle, arm and wrist (`seated.ARM_KEYS`).
"""

import numpy as np

from .. import choreo
from ..seated import ARM_KEYS, clamp, min_jerk, seat

# hand points in the hand frame (cm, from MHR's rest hand): along the fingers, toward the thumb, palm side
HAND_POINTS = {"wrist": (0.0, 0.0, 0.0), "middle1": (8.2, 0.0, 0.0), "index1": (8.3, 2.1, 0.7),
               "pinky1": (6.4, -3.8, 0.7)}
WEIGHTS = {"wrist": 1.0, "middle1": 1.0, "index1": 0.5, "pinky1": 0.5}
MAX_RESIDUAL_CM = 2.5


def _unit(v):
    return v / np.linalg.norm(v)


def hand_targets(rng, X, side):
    """IK constraints [(joint, target cm, weight, primary)] for one hand pose in front of the body."""
    sh, lat, up, fwd = choreo._body_frame(X)
    sg = 1.0 if side == "l" else -1.0
    W = sh + fwd * rng.uniform(18, 42) + up * rng.uniform(-35, 5) + lat * sg * rng.uniform(-8, 28)
    d = _unit(up * rng.uniform(0.2, 1.0) + fwd * rng.uniform(-0.2, 0.8) - lat * sg * rng.uniform(-0.3, 0.8))
    a = _unit(np.cross(d, fwd))
    b = np.cross(d, a)
    phi = rng.uniform(0, 2 * np.pi)
    n = np.cos(phi) * a + np.sin(phi) * b  # palm normal, any way around the finger direction
    t = np.cross(n, d) * sg  # toward the thumb (the right hand is the left one mirrored)
    out = []
    for j, (u, v, w) in HAND_POINTS.items():
        out.append((f"{side}_{j}", W + u * d + v * t + w * n, WEIGHTS[j], j in ("wrist", "middle1")))
    return out


def generate(rng, ch, lm, pn, base, T, fps, limits):
    """Arm motion (T, P) on the body proportions of a held-out frame `base` (P,); fingers at rest."""
    jn = list(ch.skeleton.joint_names)
    P = {n: i for i, n in enumerate(pn)}
    from mtcore.mhr import skeleton_state
    body = np.where([n.startswith("scale_") for n in pn], base, 0.0)
    pair = tuple(rng.choice(["lap", "desk"], 2))
    mp = seat(rng, pn, np.repeat(body[None], T, 0), fps, limits, force_rest=pair, moves=False)
    out = mp.copy()
    for side in "lr":
        idx = [P[f"{side}_{k}"] for k in ARM_KEYS]
        cur = mp[0, idx].copy()
        keys = [(0, 0, cur)]  # (move start frame, move end frame, arm parameters)
        k = int(rng.uniform(0.0, 1.5) * fps)
        while k < T:
            km = min(T - 1, k + int(rng.uniform(0.5, 1.2) * fps))
            if rng.random() < 0.12:  # back to rest for a while
                pose = mp[km, idx].copy()
            else:
                m0 = mp[km].copy()
                m0[idx] = cur
                X = lm(skeleton_state(ch, m0)[None])[0]
                pose = cur
                for _ in range(6):
                    tg = hand_targets(rng, X, side)
                    sol = choreo._solve(ch, pn, jn, m0, (side,), tg)
                    prim = [(j, p) for j, p, _, pr in tg if pr]
                    res = np.linalg.norm(choreo._joint_pos(ch, sol, [j for j, _ in prim], jn)
                                         - np.stack([p for _, p in prim]), axis=-1)
                    if res.max() < MAX_RESIDUAL_CM:
                        pose = sol[idx]
                        break
            keys.append((k, km, pose))
            cur = pose
            k = km + int(rng.uniform(0.8, 3.5) * fps)
        arm = np.repeat(keys[0][2][None], T, 0)
        frames = np.arange(T)
        for (a, b, pose), (_, _, prev) in zip(keys[1:], keys[:-1]):
            s = min_jerk(np.clip((frames - a) / max(1, b - a), 0, 1))
            after = frames >= a
            arm[after] = prev + (pose - prev) * s[after, None]
        out[:, idx] = arm
    return clamp(pn, out, limits)
