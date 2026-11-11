"""Structural check: a VRM skeleton with MHR's own proportions, driven through the conversion.

Any end-effector position error is then purely from the joint-count mismatch (spine0, wrist_twist,
thumb0, pinky0, foot joints folded into VRM bones). Also tests moving forearm twist from the hand
onto lowerArm (driving lowerArm with wrist_twist instead of lowarm).
"""

import sys

import numpy as np
import pymomentum.geometry as g

import mtcore.convert as convert
from mtcore.convert import MAP, MhrToVrm, vrm_fk
from mtcore.mhr import FLEX_T, limits, load

rng = np.random.default_rng(1)
ch = load()
jn = list(ch.skeleton.joint_names)
pn = list(ch.parameter_transform.names)
pose_idx = [i for i, n in enumerate(pn) if not n.startswith(("scale_", "root_t")) and n not in FLEX_T]
LIM = limits()
FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
lo = np.array([LIM.get(pn[i], (-0.6, 0.6))[0] for i in pose_idx]) * FRAC
hi = np.array([LIM.get(pn[i], (-0.6, 0.6))[1] for i in pose_idx]) * FRAC
print(f"poses: uniform within {FRAC:.0%} of MHR joint limits; {sum(pn[i] in LIM for i in pose_idx)}/{len(pose_idx)} params have limits")
rest = np.asarray(g.model_parameters_to_skeleton_state(ch, np.zeros(len(pn), np.float32)))
P = rest[:, :3]

# VRM rest positions = positions of the driving MHR joints; plus end sites used for comparison
vrm_rest = {b: P[jn.index(v[2] if len(v) > 2 else v[0])].copy() for b, v in MAP.items()}
for s, m in (("left", "l"), ("right", "r")):
    vrm_rest[f"{s}Toes"] = P[jn.index(f"{m}_ball")].copy()
TARGETS = {"head": "c_head", "leftHand": "l_wrist", "rightHand": "r_wrist", "leftIndexDistal": "l_index3",
           "leftThumbDistal": "l_thumb3", "leftLittleDistal": "l_pinky3", "leftFoot": "l_subtalar", "neck": "c_neck"}

for split in (False, True):
    conv = MhrToVrm(jn, rest, vrm_rest)
    if not split:
        conv.forearm = {}  # old behaviour: all forearm twist on the hand
    err = {t: [] for t in TARGETS}
    tw = []
    for _ in range(300):
        mp = np.zeros(len(pn), np.float32)
        mp[pose_idx] = rng.uniform(lo, hi)
        st = np.asarray(g.model_parameters_to_skeleton_state(ch, mp))
        Pv, _ = vrm_fk(conv.locals(st), vrm_rest)
        root_m, root_v = st[jn.index("root"), :3], Pv["hips"]
        for t, j in TARGETS.items():
            err[t].append(np.linalg.norm((Pv[t] - root_v) - (st[jn.index(j), :3] - root_m)))
        if split:  # twist actually moved: angle of lowerArm local rotation about its own axis
            L = conv.locals(st)["leftLowerArm"]
            tw.append(np.degrees(2 * np.arccos(min(1.0, abs(convert.twist(L, conv.forearm["leftLowerArm"][2])[3])))))
    print(f"\nposition error relative to hips, cm (mean / max), forearm twist {'split onto lowerArm' if split else 'on hand'}:")
    for t, e in err.items():
        print(f"  {t:18s} {np.mean(e):6.2f} {np.max(e):6.2f}")
    if tw:
        print(f"  lowerArm twist carried: mean {np.mean(tw):.1f} deg, max {np.max(tw):.1f} deg")
