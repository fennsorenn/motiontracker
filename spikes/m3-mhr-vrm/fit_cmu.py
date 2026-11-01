"""Fit MHR to a CMU clip: shared performer scale parameters + per-frame pose, via pymomentum IK."""

import sys

import numpy as np
import pymomentum.geometry as g
import pymomentum.solver as S
import torch

from cmu import clip_positions
from dump_skeleton import load

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
FLEX_T = {"spine_length_flexible", "neck_length_flexible", "shoulder_width_flexible", "arm_length_flexible",
          "hip_width_flexible", "leg_length_flexible"}


def fit(subject, clip, step=4, max_frames=300, verbose=True):
    ch = load()
    jn = list(ch.skeleton.joint_names)
    pn = list(ch.parameter_transform.names)
    names, P = clip_positions(subject, clip, step)
    P = P[:max_frames]
    T = len(P)
    parents = torch.tensor([jn.index(m) for m, _, _ in PAIRS], dtype=torch.int32)
    weights = torch.tensor([w for _, _, w in PAIRS], dtype=torch.float32)
    targets = torch.tensor(np.stack([P[:, names.index(c)] for _, c, _ in PAIRS], 1), dtype=torch.float32)
    offsets = torch.zeros(len(PAIRS), 3)

    is_scale = torch.tensor([n.startswith("scale_") for n in pn])
    # CMU has no finger data: freeze hand and finger scales at the template
    no_fit = lambda n: n in FLEX_T or (n.startswith("scale_") and ("hands" in n or any(
        f in n for f in ("thumb", "index", "middle", "ring", "pinky"))))
    active = torch.tensor([not no_fit(n) for n in pn])
    is_scale = is_scale & active
    # init: root at the CMU hip centre, facing from the hip vector
    init = torch.zeros(T, len(pn))
    hipc = 0.5 * (P[:, names.index("lhipjoint")] + P[:, names.index("rhipjoint")])
    init[:, pn.index("root_tx")] = torch.tensor(hipc[:, 0] / 10)
    init[:, pn.index("root_ty")] = torch.tensor((hipc[:, 1] - 90.0) / 10)
    init[:, pn.index("root_tz")] = torch.tensor(hipc[:, 2] / 10)
    left = P[:, names.index("lhipjoint")] - P[:, names.index("rhipjoint")]
    init[:, pn.index("root_ry")] = torch.tensor(-np.arctan2(left[:, 2], left[:, 0]), dtype=torch.float32)

    err = [S.ErrorFunctionType.Position, S.ErrorFunctionType.Limit]
    opts = S.SolverOptions()
    opts.max_iter = 40
    # stage 1: rigid root only, stage 2: everything
    rigid = torch.tensor([n.startswith("root_") for n in pn])
    mp = S.solve_ik(ch, rigid, init, err, torch.tensor([1.0, 0.0]), opts,
                    position_cons_parents=parents, position_cons_offsets=offsets,
                    position_cons_weights=weights, position_cons_targets=targets)
    mp = S.solve_sequence_ik(ch, active, is_scale, mp, err, torch.tensor([1.0, 1.0]), opts,
                             position_cons_parents=parents, position_cons_offsets=offsets,
                             position_cons_weights=weights, position_cons_targets=targets)
    mp = mp.detach().numpy().astype(np.float32)
    states = np.stack([np.asarray(g.model_parameters_to_skeleton_state(ch, m)) for m in mp])
    res = np.linalg.norm(states[:, parents.numpy(), :3] - targets.numpy(), axis=-1)  # (T, pairs)
    if verbose:
        print(f"CMU {subject:02d}_{clip:02d}: {T} frames at {120 // step} fps")
        print("fit residual per pair, cm (median / p95):")
        for k, (m, c, w) in enumerate(PAIRS):
            print(f"  {m:10s} <- {c:10s} w={w:.1f}  {np.median(res[:, k]):5.1f} {np.percentile(res[:, k], 95):5.1f}")
        sc = {n: float(mp[0, i]) for i, n in enumerate(pn) if n.startswith("scale_") and abs(mp[0, i]) > 1e-3}
        print("performer scale parameters:", {k: round(v, 2) for k, v in sc.items()})
    return ch, mp, states, res


if __name__ == "__main__":
    fit(int(sys.argv[1]), int(sys.argv[2]))
