"""Fit MHR to CMU clips via pymomentum IK.

`fit` solves one clip with shared performer scales + per-frame pose (the M3 spike's fit).
For datasets: `fit_subject_scales` solves a subject's scales once on frames from several clips,
then `fit_clip` solves every frame's pose with those scales frozen.
"""

import numpy as np
import pymomentum.geometry as g
import pymomentum.solver as S
import torch

from .cmu import clip_positions
from .mhr import FLEX_T, load

# MHR joint <- CMU bone end, weight
PAIRS = [
    ("l_upleg", "lhipjoint", 1.0), ("r_upleg", "rhipjoint", 1.0),
    ("l_lowleg", "lfemur", 1.0), ("r_lowleg", "rfemur", 1.0),
    ("l_foot", "ltibia", 1.0), ("r_foot", "rtibia", 1.0),
    ("l_ball", "lfoot", 0.5), ("r_ball", "rfoot", 0.5),
    ("c_spine1", "lowerback", 0.3), ("c_spine3", "upperback", 0.3),
    ("c_neck", "thorax", 0.5), ("c_head", "upperneck", 0.5),
    ("l_uparm", "lclavicle", 1.0), ("r_uparm", "rclavicle", 1.0),
    ("l_lowarm", "lhumerus", 1.0), ("r_lowarm", "rhumerus", 1.0),
    # CMU's 'wrist' bone is the distal forearm: radius + wrist ~ a full forearm, so its end is the wrist
    ("l_wrist", "lwrist", 1.0), ("r_wrist", "rwrist", 1.0),
    ("l_middle1", "lfingers", 0.3), ("r_middle1", "rfingers", 0.3),
]
# CMU places its spine and neck joints differently from MHR; with full weight the spine/neck scales
# lean on their limits (M3 spike). M1 data prep uses a looser pairing. The neck length still ends at
# MHR's upper limit for every subject tried (CMU's head sits higher above the shoulders); a low neck
# weight lets the head joint follow CMU's (head residual ~1-2 cm instead of 2-4 cm).
LOOSE_SPINE = {"c_spine1": 0.1, "c_spine3": 0.1, "c_neck": 0.1}
PAIRS_M1 = [(m, c, LOOSE_SPINE.get(m, w)) for m, c, w in PAIRS]

ERRORS = [S.ErrorFunctionType.Position, S.ErrorFunctionType.Limit]


def _frozen(n):
    # flexible translations stay zero; CMU has no finger data, so hand and finger scales stay at the template
    return n in FLEX_T or (n.startswith("scale_") and ("hands" in n or any(
        f in n for f in ("thumb", "index", "middle", "ring", "pinky"))))


def _constraints(jn, names, P, pairs):
    return dict(
        position_cons_parents=torch.tensor([jn.index(m) for m, _, _ in pairs], dtype=torch.int32),
        position_cons_offsets=torch.zeros(len(pairs), 3),
        position_cons_weights=torch.tensor([w for _, _, w in pairs], dtype=torch.float32),
        position_cons_targets=torch.tensor(np.stack([P[:, names.index(c)] for _, c, _ in pairs], 1),
                                           dtype=torch.float32),
    )


def _init(pn, names, P, base=None):
    """Root at the CMU hip centre, facing from the hip vector; other parameters from `base`."""
    init = torch.zeros(len(P), len(pn)) if base is None else torch.tensor(base, dtype=torch.float32).repeat(len(P), 1)
    hipc = 0.5 * (P[:, names.index("lhipjoint")] + P[:, names.index("rhipjoint")])
    init[:, pn.index("root_tx")] = torch.tensor(hipc[:, 0] / 10)
    init[:, pn.index("root_ty")] = torch.tensor((hipc[:, 1] - 90.0) / 10)
    init[:, pn.index("root_tz")] = torch.tensor(hipc[:, 2] / 10)
    left = P[:, names.index("lhipjoint")] - P[:, names.index("rhipjoint")]
    init[:, pn.index("root_ry")] = torch.tensor(-np.arctan2(left[:, 2], left[:, 0]), dtype=torch.float32)
    return init


def _options(max_iter=40):
    opts = S.SolverOptions()
    opts.max_iter = max_iter
    return opts


def _rigid(ch, pn, init, cons, opts):
    rigid = torch.tensor([n.startswith("root_") for n in pn])
    return S.solve_ik(ch, rigid, init, ERRORS, torch.tensor([1.0, 0.0]), opts, **cons)


def _residual(ch, jn, mp, cons):
    states = np.asarray(g.model_parameters_to_skeleton_state(ch, np.ascontiguousarray(mp, dtype=np.float32)))
    idx = cons["position_cons_parents"].numpy()
    return states, np.linalg.norm(states[:, idx, :3] - cons["position_cons_targets"].numpy(), axis=-1)


def fit(subject, clip, step=4, max_frames=300, verbose=True, pairs=PAIRS):
    ch = load()
    jn = list(ch.skeleton.joint_names)
    pn = list(ch.parameter_transform.names)
    names, P = clip_positions(subject, clip, step)
    P = P[:max_frames]
    T = len(P)
    cons = _constraints(jn, names, P, pairs)
    active = torch.tensor([not _frozen(n) for n in pn])
    is_scale = torch.tensor([n.startswith("scale_") for n in pn]) & active
    opts = _options()
    # stage 1: rigid root only, stage 2: everything
    mp = _rigid(ch, pn, _init(pn, names, P), cons, opts)
    mp = S.solve_sequence_ik(ch, active, is_scale, mp, ERRORS, torch.tensor([1.0, 1.0]), opts, **cons)
    mp = mp.detach().numpy().astype(np.float32)
    states, res = _residual(ch, jn, mp, cons)  # res: (T, pairs)
    if verbose:
        print(f"CMU {subject:02d}_{clip:02d}: {T} frames at {120 // step} fps")
        print("fit residual per pair, cm (median / p95):")
        for k, (m, c, w) in enumerate(pairs):
            print(f"  {m:10s} <- {c:10s} w={w:.1f}  {np.median(res[:, k]):5.1f} {np.percentile(res[:, k], 95):5.1f}")
        sc = {n: float(mp[0, i]) for i, n in enumerate(pn) if n.startswith("scale_") and abs(mp[0, i]) > 1e-3}
        print("performer scale parameters:", {k: round(v, 2) for k, v in sc.items()})
    return ch, mp, states, res


def fit_subject_scales(ch, subject, clips, step=4, frames_per_clip=60, pairs=PAIRS_M1):
    """One performer identity per CMU subject: scales solved on evenly spaced frames from `clips`.

    Returns the model parameter vector with only the scale entries set (pose entries zero)."""
    jn = list(ch.skeleton.joint_names)
    pn = list(ch.parameter_transform.names)
    chunks, names = [], None
    for clip in clips:
        names, P = clip_positions(subject, clip, step)
        chunks.append(P[np.linspace(0, len(P) - 1, min(frames_per_clip, len(P))).astype(int)])
    P = np.concatenate(chunks)
    cons = _constraints(jn, names, P, pairs)
    active = torch.tensor([not _frozen(n) for n in pn])
    is_scale = torch.tensor([n.startswith("scale_") for n in pn]) & active
    opts = _options()
    mp = _rigid(ch, pn, _init(pn, names, P), cons, opts)
    mp = S.solve_sequence_ik(ch, active, is_scale, mp, ERRORS, torch.tensor([1.0, 1.0]), opts, **cons)
    scales = np.zeros(len(pn), np.float32)
    sel = is_scale.numpy()
    scales[sel] = mp[0].detach().numpy()[sel]
    return scales


def fit_clip(ch, subject, clip, scales, step=4, pairs=PAIRS_M1, chunk=500):
    """Per-frame pose for a whole clip with the subject's scales frozen.

    Returns (model parameters (T, P), residual per pair (T, pairs) in cm)."""
    jn = list(ch.skeleton.joint_names)
    pn = list(ch.parameter_transform.names)
    names, P = clip_positions(subject, clip, step)
    pose = torch.tensor([not _frozen(n) and not n.startswith("scale_") for n in pn])
    opts = _options()
    out, res = [], []
    for s in range(0, len(P), chunk):
        Pc = P[s:s + chunk]
        cons = _constraints(jn, names, Pc, pairs)
        mp = _rigid(ch, pn, _init(pn, names, Pc, scales), cons, opts)
        mp = S.solve_ik(ch, pose, mp, ERRORS, torch.tensor([1.0, 1.0]), opts, **cons)
        mp = mp.detach().numpy().astype(np.float32)
        out.append(mp)
        res.append(_residual(ch, jn, mp, cons)[1])
    return np.concatenate(out), np.concatenate(res)
