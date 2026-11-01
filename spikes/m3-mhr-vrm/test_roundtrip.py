"""Random MHR poses -> VRM rotations -> VRM FK; compare bone directions with MHR's."""

import sys
from pathlib import Path

import numpy as np
import pymomentum.geometry as g

from convert import MAP, VRM_CHILD, MhrToVrm, qrot, vrm_fk
from dump_skeleton import load
from fit_avatars import AVATARS
from vrm_skeleton import load_vrm

rng = np.random.default_rng(0)
ch = load()
jn = list(ch.skeleton.joint_names)
pn = list(ch.parameter_transform.names)
rest = np.asarray(g.model_parameters_to_skeleton_state(ch, np.zeros(len(pn), np.float32)))

# check quaternion convention: child position = parent pos + q_parent * (s * offset) must hold for
# some ordering; we only need q to rotate rest-relative directions consistently, tested below.
avatar = sys.argv[1] if len(sys.argv) > 1 else "AvatarSample_D"
_, _, vrm_rest = load_vrm(AVATARS / f"{avatar}.vrm")
conv = MhrToVrm(jn, rest, vrm_rest)

# random poses: pose parameters only (no scale, no root translation), moderate range
FLEX_T = {"spine_length_flexible", "neck_length_flexible", "shoulder_width_flexible", "arm_length_flexible",
          "hip_width_flexible", "leg_length_flexible"}  # joint translations VRM cannot express
mode = sys.argv[2] if len(sys.argv) > 2 else "no-flex"
pose_idx = [i for i, n in enumerate(pn) if not n.startswith("scale_") and not n.startswith("root_t")
            and (mode == "flex" or n not in FLEX_T)]
errs = {b: [] for b in conv.bones}
for _ in range(200):
    mp = np.zeros(len(pn), np.float32)
    mp[pose_idx] = rng.uniform(-0.6, 0.6, len(pose_idx))
    st = np.asarray(g.model_parameters_to_skeleton_state(ch, mp))
    P_vrm, _ = vrm_fk(conv.locals(st), vrm_rest)
    for b in conv.bones:
        c = VRM_CHILD.get(b); jd = MAP[b][1]
        # hips: MHR's c_spine1 position also moves with c_spine0, which VRM folds into its spine bone,
        # so this direction comparison is not meaningful; test_positions.py covers the upper body.
        if c not in P_vrm or jd is None or b == "hips":
            continue
        d_v = P_vrm[c] - P_vrm[b]
        d_m = st[jn.index(jd), :3] - st[jn.index(MAP[b][0]), :3]
        cos = np.dot(d_v, d_m) / np.linalg.norm(d_v) / np.linalg.norm(d_m)
        errs[b].append(np.degrees(np.arccos(np.clip(cos, -1, 1))))

print(f"bone direction error after MHR -> VRM ({avatar}, {mode}), 200 random poses, degrees: mean / max")
for b, e in errs.items():
    if e:
        print(f"  {b:26s} {np.mean(e):6.2f} {np.max(e):7.2f}")
