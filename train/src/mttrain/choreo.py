"""Designed contact choreography (owner, 2026-11-09): seated sequences in which the hands meet or touch
the face, with exact contact intervals by construction.

Rendered as test sessions for hand contact (criterion 7b) and meant as follow-along reference videos
for recordings. The held-out CMU motion has almost no contacts, so the motion is authored instead:

- base: a held-out subject's body proportions (scale parameters only; CMU postures can be strongly
  slumped), seated with both arms in a rest posture plus the seated idle motion (`seated.seat`);
- events: palms together (prayer; a forward clasp crossed the fingers), one hand touching the chin,
  a cheek, the forehead or the mouth (fingertips or palm), and near misses that stop 8-14 cm short of
  the face (negatives for telling deliberate contact from incidental closeness later);
- each event's arm pose is solved with MHR's IK on the active arm(s): hand joints to points on the
  palm-to-palm plane or on the face (surface point plus a small outward offset), joint limits on, a
  weak prior toward the starting pose. The head takes a small random pose per event first;
- arms blend from the rest posture with a minimum-jerk approach, hold, and release. During the hold
  the arms and head are fixed relative to the torso, so the torso's idle motion keeps the contact.

Contact intervals are the holds. `generate` checks every hold frame against its targets afterwards.
"""

import numpy as np
import pymomentum.solver as S
import torch

from mtcore.cmu_fit import _options
from mtcore.landmarks import MP33
from mtcore.mhr import skeleton_state

from .seated import ARM_KEYS, clamp, min_jerk, seat

I = {n: k for k, n in enumerate(MP33)}
EVENTS = {"palms": 0.35, "face": 0.45, "near": 0.20}
FACE_POINTS = ("chin", "cheek", "forehead", "mouth")
HEAD_KEYS = {"neck_twist": (-0.25, 0.25), "neck_bend": (-0.15, 0.2), "head_lean": (-0.1, 0.1)}
MAX_RESIDUAL_CM = 1.5  # primary targets; events whose solve misses are resampled


def _unit(v):
    return v / np.linalg.norm(v)


def _body_frame(X):
    """Landmarks (33, 3) cm -> shoulder midpoint, person's left, up, forward."""
    sh = 0.5 * (X[I["left_shoulder"]] + X[I["right_shoulder"]])
    hip = 0.5 * (X[I["left_hip"]] + X[I["right_hip"]])
    lat = _unit(X[I["left_shoulder"]] - X[I["right_shoulder"]])
    up = sh - hip
    up = _unit(up - (up @ lat) * lat)
    return sh, lat, up, np.cross(lat, up)


def _face_point(X, name, side):
    """A point on the face surface (cm) and its outward direction, from the face landmarks."""
    ears = 0.5 * (X[I["left_ear"]] + X[I["right_ear"]])
    eyes = 0.5 * (X[I["left_eye"]] + X[I["right_eye"]])
    mouth = 0.5 * (X[I["mouth_left"]] + X[I["mouth_right"]])
    s = "left" if side == "l" else "right"
    p = {"chin": mouth + 0.7 * (mouth - eyes), "forehead": eyes + 0.9 * (eyes - mouth), "mouth": mouth,
         "cheek": 0.6 * X[I[f"mouth_{s}"]] + 0.4 * X[I[f"{s}_ear"]]}[name]
    return p, _unit(p - ears)


def _targets(rng, kind, X, side):
    """Event constraints: [(joint, target point cm, weight, primary)], plus the arms involved."""
    sh, lat, up, fwd = _body_frame(X)
    if kind == "palms":
        P = sh + fwd * rng.uniform(22, 38) - up * rng.uniform(5, 30) + lat * rng.uniform(-8, 8)
        out = []
        for s, sg in (("l", 1.0), ("r", -1.0)):  # prayer: fingers up, thumbs toward the chest
            o = P + sg * lat * 1.8
            pts = {"wrist": P + sg * lat * 2.2 - up * 4.0, "middle1": o + up * 5.5,
                   "index1": o + up * 4.5 - fwd * 2.0, "pinky1": o + up * 4.5 + fwd * 2.0}
            # the palm centre (middle knuckle) is the contact point; the rest only orient the hand
            out += [(f"{s}_{j}", p, 1.0 if j == "middle1" else 0.5, j == "middle1") for j, p in pts.items()]
        return out, ("l", "r")
    name = FACE_POINTS[rng.integers(len(FACE_POINTS))]
    F, n = _face_point(X, name, side)
    gap = 1.0 if kind == "face" else rng.uniform(8, 14)
    if kind == "face" and rng.random() < 0.4:  # palm on the face
        tg = [(f"{side}_middle1", F + n * (gap + 1.0), 1.0, True), (f"{side}_wrist", F + n * (gap + 2.0) - up * 7, 0.2, False),
              (f"{side}_middle_null", F + n * (gap + 0.5) + up * 7, 0.1, False)]
    else:  # fingertip
        tg = [(f"{side}_middle_null", F + n * (gap - 0.2), 1.0, True),
              (f"{side}_wrist", F + n * (gap + 6) - up * 9, 0.05, False)]
    return tg, (side,)


def _solve(ch, pn, jn, mp0, sides, tg):
    names = {f"{s}_{k}" for s in sides for k in ARM_KEYS}
    active = torch.tensor([n in names for n in pn])
    cons = dict(position_cons_parents=torch.tensor([jn.index(j) for j, _, _, _ in tg], dtype=torch.int32),
                position_cons_offsets=torch.zeros(len(tg), 3),
                position_cons_weights=torch.tensor([w for _, _, w, _ in tg], dtype=torch.float32),
                position_cons_targets=torch.tensor(np.stack([p for _, p, _, _ in tg])[None], dtype=torch.float32),
                motion_targets=torch.tensor(mp0[None], dtype=torch.float32),
                motion_weights=active.float())
    errors = [S.ErrorFunctionType.Position, S.ErrorFunctionType.Limit, S.ErrorFunctionType.Motion]
    mp = S.solve_ik(ch, active, torch.tensor(mp0[None], dtype=torch.float32), errors,
                    torch.tensor([1.0, 1.0, 0.02]), _options(100), **cons)
    return mp.detach().numpy()[0].astype(np.float64)


def _joint_pos(ch, mp, joints, jn):
    st = skeleton_state(ch, mp)
    return st[..., [jn.index(j) for j in joints], :3]


def generate(rng, ch, lm, pn, base, T, fps, limits):
    """One sequence on the body of a held-out frame `base` (P,): -> model parameters (T, P), segments
    (session segment dicts: 'contact' palms-together / hand-to-face, 'near-contact' near-face; t ms),
    worst hold residual per event (cm)."""
    jn = list(ch.skeleton.joint_names)
    P = {n: i for i, n in enumerate(pn)}
    pair = tuple(rng.choice(["lap", "desk"], 2))
    body = np.where([n.startswith("scale_") for n in pn], base, 0.0)  # proportions only, neutral pose
    mp = seat(rng, pn, np.repeat(body[None], T, 0), fps, limits, force_rest=pair)
    arm_idx = {s: [P[f"{s}_{k}"] for k in ARM_KEYS] for s in "lr"}
    head_idx = [P[k] for k in HEAD_KEYS]
    w = {s: np.zeros(T) for s in "lr"}
    tgt = {s: np.zeros((T, len(ARM_KEYS))) for s in "lr"}
    wh, dh = np.zeros(T), np.zeros((T, len(HEAD_KEYS)))
    events, segs = [], []
    k = int(rng.uniform(1.0, 2.0) * fps)
    while True:
        a, h, r = (int(rng.uniform(*x) * fps) for x in ((0.5, 1.2), (1.0, 2.5), (0.5, 1.0)))
        if k + a + h + r > T - int(0.5 * fps):
            break
        kind = rng.choice(list(EVENTS), p=list(EVENTS.values()))
        side = "l" if rng.random() < 0.5 else "r"
        km = k + a + h // 2
        head = np.array([rng.uniform(*v) for v in HEAD_KEYS.values()])
        m0 = mp[km].copy()
        m0[head_idx] += head
        X = lm(skeleton_state(ch, m0)[None])[0]
        for _ in range(6):
            tg, sides = _targets(rng, kind, X, side)
            sol = _solve(ch, pn, jn, m0, sides, tg)
            prim = [(j, p) for j, p, _, pr in tg if pr]
            res = np.linalg.norm(_joint_pos(ch, sol, [j for j, _ in prim], jn) - np.stack([p for _, p in prim]), axis=-1)
            if res.max() < MAX_RESIDUAL_CM:
                break
        else:
            k += a  # no reachable pose: skip ahead
            continue
        ramp = np.concatenate([min_jerk(np.arange(a) / a), np.ones(h), min_jerk(1 - np.arange(r) / r)])
        span = slice(k, k + a + h + r)
        for s in sides:
            w[s][span] = ramp
            tgt[s][span] = sol[arm_idx[s]]
        wh[span], dh[span] = ramp, head
        t0, t1 = (k + a) / fps * 1000, (k + a + h) / fps * 1000
        if kind == "near":
            segs.append({"label": "near-contact", "variant": "near-face", "t0": round(t0, 1), "t1": round(t1, 1)})
        else:
            segs.append({"label": "contact", "variant": "palms-together" if kind == "palms" else "hand-to-face",
                         "t0": round(t0, 1), "t1": round(t1, 1), "settleMs": 0})
        events.append((kind, sides, tg, k + a, k + a + h, km))
        k += a + h + r + int(rng.uniform(0.3, 1.5) * fps)
    out = mp.copy()
    for s in "lr":
        out[:, arm_idx[s]] = (1 - w[s][:, None]) * out[:, arm_idx[s]] + w[s][:, None] * tgt[s]
    out[:, head_idx] += wh[:, None] * dh
    out = clamp(pn, out, limits)
    # check: during each hold, the primary joints stay on their targets, which move with the torso
    worst = []
    for kind, sides, tg, h0, h1, km in events:
        prim = [(j, p) for j, p, _, pr in tg if pr]
        joints = [j for j, _ in prim]
        ref = _joint_pos(ch, out[km], joints, jn)
        held = _joint_pos(ch, out[h0:h1], joints, jn)
        # palms: the hands relative to each other; face: the hand relative to the nose (both move with
        # the torso during the hold)
        if kind == "palms":
            drift = (held[:, 0] - held[:, 1]) - (ref[0] - ref[1])[None]
        else:
            nose_h = lm(skeleton_state(ch, out[h0:h1]))[:, I["nose"]]
            nose_m = lm(skeleton_state(ch, out[km])[None])[0, I["nose"]]
            drift = (held - nose_h[:, None]) - (ref - nose_m)[None]
        worst.append(float(np.linalg.norm(drift, axis=-1).max()))
    return out, segs, worst
