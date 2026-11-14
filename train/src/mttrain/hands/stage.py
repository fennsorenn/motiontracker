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
WEIGHTS = {"wrist": 1.0, "middle1": 1.0, "index1": 2.0, "pinky1": 2.0}  # knuckles set the palm's roll
MAX_RESIDUAL_CM = 2.5
MAX_PALM_DEG = 30.0  # achieved vs asked palm direction (the knuckle targets are only cm apart)


def _palm_normal(w, m, i, p):
    f = _unit(m - w)
    s = i - p
    return np.cross(f, _unit(s - (s @ f) * f))


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


def generate(rng, ch, lm, pn, base, T, fps, limits, p_rest=0.12, rests=("lap", "desk"), with_rest=False, fix_rest=None,
             clear=None):
    """Arm motion (T, P) on the body proportions of a held-out frame `base` (P,); fingers at rest.
    `p_rest`: chance per move of going back to the rest posture for a while; `rests`: the rest postures drawn
    per arm (`seated.REST`; "hanging" mostly takes the hand out of a desk camera's view). `with_rest`: also
    return per side the weight (T,) of the rest posture (1 resting, ramps during moves), for relaxed fingers.
    `fix_rest` (pose (P,) -> pose): lifts resting hands out of the body (`pairs.PairChoreo.fix_rest`); its change
    to the first frame's arms is applied to every frame of the rest postures. `clear` (pose (P,) -> bool): key
    poses it rejects (e.g. hands inside the body mesh, `pairs.PairChoreo.mesh_clear`) are drawn again."""
    jn = list(ch.skeleton.joint_names)
    P = {n: i for i, n in enumerate(pn)}
    from mtcore.mhr import skeleton_state
    body = np.where([n.startswith("scale_") for n in pn], base, 0.0)
    pair = tuple(rng.choice(list(rests), 2))
    mp = seat(rng, pn, np.repeat(body[None], T, 0), fps, limits, force_rest=pair, moves=False)
    from .pairs import still_head
    still_head(mp, pn)  # the seated idle head motion carries nothing for hands (owner, 2026-11-12)
    if fix_rest is not None:
        arms = [P[f"{s}_{k}"] for s in "lr" for k in ARM_KEYS]
        mp[:, arms] += (fix_rest(mp[0]) - mp[0])[arms]
    out = mp.copy()
    rest_w = {}
    for side in "lr":
        idx = [P[f"{side}_{k}"] for k in ARM_KEYS]
        cur = mp[0, idx].copy()
        keys = [(0, 0, cur, True)]  # (move start frame, move end frame, arm parameters, at rest)
        k = int(rng.uniform(0.0, 1.5) * fps)
        while k < T:
            km = min(T - 1, k + int(rng.uniform(0.5, 1.2) * fps))
            resting = rng.random() < p_rest
            if resting:  # back to rest for a while
                pose = mp[km, idx].copy()
            else:
                m0 = mp[km].copy()
                m0[idx] = cur
                X = lm(skeleton_state(ch, m0)[None])[0]
                pose = cur
                for _ in range(10):
                    tg = hand_targets(rng, X, side)
                    sol = choreo._solve(ch, pn, jn, m0, (side,), tg)
                    prim = [(j, p) for j, p, _, pr in tg if pr]
                    res = np.linalg.norm(choreo._joint_pos(ch, sol, [j for j, _ in prim], jn)
                                         - np.stack([p for _, p in prim]), axis=-1)
                    four = [f"{side}_{j}" for j in ("wrist", "middle1", "index1", "pinky1")]
                    want = dict((j, p) for j, p, _, _ in tg)
                    got = choreo._joint_pos(ch, sol, four, jn)
                    palm_err = np.degrees(np.arccos(np.clip(
                        _palm_normal(*got) @ _palm_normal(*(want[j] for j in four)), -1, 1)))
                    if res.max() < MAX_RESIDUAL_CM and palm_err < MAX_PALM_DEG and (clear is None or clear(sol)):
                        pose = sol[idx]
                        break
            resting = resting or pose is cur and keys[-1][3]  # a failed solve keeps the previous pose
            keys.append((k, km, pose, resting))
            cur = pose
            k = km + int((rng.uniform(1.5, 6.0) if resting else rng.uniform(0.8, 3.5)) * fps)
        arm = np.repeat(keys[0][2][None], T, 0)
        w = np.ones(T)
        frames = np.arange(T)
        for (a, b, pose, r), (_, _, prev, r0) in zip(keys[1:], keys[:-1]):
            s = min_jerk(np.clip((frames - a) / max(1, b - a), 0, 1))
            after = frames >= a
            arm[after] = prev + (pose - prev) * s[after, None]
            w[after] = float(r0) + (float(r) - float(r0)) * s[after]
        out[:, idx] = arm
        rest_w[side] = w
    out = clamp(pn, out, limits)
    return (out, rest_w) if with_rest else out
