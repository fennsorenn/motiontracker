"""Fit MHR to CMU clips via pymomentum IK.

`fit` solves one clip with shared performer scales + per-frame pose (the M3 spike's fit).
For datasets: `fit_subject_scales` solves a subject's scales once on frames from several clips,
then `fit_clip` solves every frame's pose with those scales frozen.
"""

import numpy as np
import pymomentum.geometry as g
import pymomentum.solver as S
import torch

from .cmu import clip_length, clip_positions
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
# M1 also pairs the thumbs (hand roll; otherwise forearm twist and wrist orientation are unconstrained)
PAIRS_M1 = [(m, c, LOOSE_SPINE.get(m, w)) for m, c, w in PAIRS] + [
    ("l_thumb1", "lthumb", 0.5), ("r_thumb1", "rthumb", 0.5)]
# Head orientation: positions only pin the head joint, not where the face points. CMU's rest pose
# faces +z with the left at +x like MHR's, so MHR's head target is CMU's head rotation applied to
# MHR's rest head orientation. CMU's head bone has no consistent zero, though: relative to the
# thorax, subjects' mean head pose ranges over -28..+59 degrees of pitch, some subjects are rolled
# ~180 degrees (marker calibration, not posture). `head_offset` estimates it per subject (mean
# thorax-to-head rotation over the subject's clips, taken as neutral) and removes it.
HEAD_WEIGHT = 10.0

ERRORS = [S.ErrorFunctionType.Position, S.ErrorFunctionType.Limit]
ERRORS_M1 = ERRORS + [S.ErrorFunctionType.Orientation]
# Neck/head split: the head orientation constraint fixes the head's total rotation, and CMU's head
# position (forward of MHR's) alone decides the split, bending the neck forward and the head back
# toward its limits. A weak prior toward neutral on these parameters evens it out.
NECK_HEAD = ("neck_bend", "neck_lean", "neck_twist", "head_bend", "head_lean", "head_twist")
NECK_PRIOR = 0.1  # 2026-11-08: head orientation stays exact (p95 0.1 deg); neck+head twist range p5-p95 +-22 vs +-28 deg without


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


def _mat_to_quat(R):
    w = np.sqrt(np.maximum(0, 1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2])) / 2
    x = np.copysign(np.sqrt(np.maximum(0, 1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])) / 2, R[..., 2, 1] - R[..., 1, 2])
    y = np.copysign(np.sqrt(np.maximum(0, 1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])) / 2, R[..., 0, 2] - R[..., 2, 0])
    z = np.copysign(np.sqrt(np.maximum(0, 1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])) / 2, R[..., 1, 0] - R[..., 0, 1])
    q = np.stack([x, y, z, w], -1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _quat_mul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def _mean_rot(Rs):
    """Chordal mean of rotation matrices (N, 3, 3)."""
    U, _, Vt = np.linalg.svd(Rs.sum(0))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def head_offset(subject, clips, step=8, max_frames=300):
    """A subject's head calibration offset (3, 3): the mean CMU head rotation relative to the thorax
    over `clips`, in the head's frame. Corrected head rotation = head @ offset.T."""
    means = []
    for clip in clips:
        n = clip_length(subject, clip, step)
        _, _, R = clip_positions(subject, clip, step, np.linspace(0, n - 1, min(max_frames, n)).astype(int),
                                 rot_bones=("thorax", "head"))
        means.append(_mean_rot(np.einsum("tji,tjk->tik", R[:, 0], R[:, 1])))
    return _mean_rot(np.array(means))


def _head_targets(ch, jn, head_rot, post=None):
    """MHR c_head world rotation targets (T, 3, 3): CMU head rotation applied to MHR's rest head
    orientation, then `post` (3, 3) in the head's own frame."""
    h = jn.index("c_head")
    rest = np.asarray(g.model_parameters_to_skeleton_state(
        ch, np.zeros(len(ch.parameter_transform.names), np.float32)))[h, 3:7]
    T = head_rot @ _quat_to_mat(rest)
    return T if post is None else T @ post


def _quat_to_mat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _head_constraint(ch, jn, head_rot, weight, post=None):
    """Orientation constraint dict for c_head from CMU head rotations (T, 3, 3)."""
    h = jn.index("c_head")
    tgt = _mat_to_quat(_head_targets(ch, jn, head_rot, post))
    return dict(orientation_cons_parents=torch.tensor([h], dtype=torch.int32),
                orientation_cons_offsets=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                orientation_cons_weights=torch.tensor([1.0]),
                orientation_cons_targets=torch.tensor(tgt[:, None], dtype=torch.float32))


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


def fit_subject_scales(ch, subject, clips, step=4, frames_per_clip=60, pairs=PAIRS_M1, head_weight=HEAD_WEIGHT,
                       head_off=None, head_post=False):
    """One performer identity per CMU subject: scales solved on evenly spaced frames from `clips`.
    `head_off`: the subject's head calibration offset (`head_offset`), removed from the head target.
    `head_post=True` also returns the subject's head correction in MHR terms (`_head_post`), for
    `fit_clip(..., head_post=...)`: returns (scales, post).

    Returns the model parameter vector with only the scale entries set (pose entries zero)."""
    jn = list(ch.skeleton.joint_names)
    pn = list(ch.parameter_transform.names)
    chunks, rots, names = [], [], None
    for clip in clips:
        n = clip_length(subject, clip, step)
        names, P, R = clip_positions(subject, clip, step, np.linspace(0, n - 1, min(frames_per_clip, n)).astype(int),
                                     rot_bones=("head",))
        chunks.append(P)
        rots.append(R[:, 0])
    P = np.concatenate(chunks)
    head = np.concatenate(rots)
    if head_off is not None:
        head = head @ head_off.T
    active = torch.tensor([not _frozen(n) for n in pn])
    is_scale = torch.tensor([n.startswith("scale_") for n in pn]) & active
    opts = _options()
    base = _constraints(jn, names, P, pairs)
    init = _rigid(ch, pn, _init(pn, names, P), base, opts)
    post = None
    for _ in range(2 if head_post else 1):
        cons = base | _head_constraint(ch, jn, head, 1.0, post)
        mp = S.solve_sequence_ik(ch, active, is_scale, init, ERRORS_M1, torch.tensor([1.0, 1.0, head_weight]),
                                 opts, **cons).detach().numpy()
        if head_post and post is None:
            post = _head_post(ch, jn, mp, _head_targets(ch, jn, head))
    scales = np.zeros(len(pn), np.float32)
    sel = is_scale.numpy()
    scales[sel] = mp[0][sel]
    return (scales, post) if head_post else scales


def _head_post(ch, jn, mp, T):
    """Correction (3, 3), in the head's frame, that makes the mean fitted head pose relative to
    c_spine3 equal MHR's rest relation (a subject's typical head posture = MHR's neutral). Without it
    the neck bends forward to reach CMU's head position and the head bends back against its limit."""
    st = np.asarray(g.model_parameters_to_skeleton_state(ch, np.ascontiguousarray(mp, dtype=np.float32)))
    rest = np.asarray(g.model_parameters_to_skeleton_state(ch, np.zeros(mp.shape[1], np.float32)))
    s3, h = jn.index("c_spine3"), jn.index("c_head")
    rel_rest = _quat_to_mat(rest[s3, 3:7]).T @ _quat_to_mat(rest[h, 3:7])
    G3 = np.array([_quat_to_mat(q) for q in st[:, s3, 3:7]])
    M = _mean_rot(rel_rest.T @ np.swapaxes(G3, -1, -2) @ T)
    return M.T


def fit_clip(ch, subject, clip, scales, step=4, pairs=PAIRS_M1, chunk=500, head_weight=HEAD_WEIGHT, head_off=None,
             head_post=None, neck_prior=None):
    """Per-frame pose for a whole clip with the subject's scales frozen.

    Returns (model parameters (T, P), residual per pair (T, pairs) in cm)."""
    jn = list(ch.skeleton.joint_names)
    pn = list(ch.parameter_transform.names)
    names, P, R = clip_positions(subject, clip, step, rot_bones=("head",))
    if head_off is not None:
        R = R @ head_off.T
    pose = torch.tensor([not _frozen(n) and not n.startswith("scale_") for n in pn])
    opts = _options()
    out, res = [], []
    for s in range(0, len(P), chunk):
        Pc = P[s:s + chunk]
        cons = _constraints(jn, names, Pc, pairs)
        mp = _rigid(ch, pn, _init(pn, names, Pc, scales), cons, opts)
        cons |= _head_constraint(ch, jn, R[s:s + chunk, 0], 1.0, head_post)
        errors, weights = ERRORS_M1, [1.0, 1.0, head_weight]
        prior = NECK_PRIOR if neck_prior is None else neck_prior
        if prior > 0:
            errors = errors + [S.ErrorFunctionType.Motion]
            weights = weights + [prior]
            cons |= dict(motion_targets=torch.zeros(len(mp), len(pn)),
                         motion_weights=torch.tensor([float(n in NECK_HEAD) for n in pn]))
        mp = S.solve_ik(ch, pose, mp, errors, torch.tensor(weights), opts, **cons)
        mp = mp.detach().numpy().astype(np.float32)
        out.append(mp)
        res.append(_residual(ch, jn, mp, cons)[1])
    return np.concatenate(out), np.concatenate(res)
