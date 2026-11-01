"""Structural check: a VRM skeleton with MHR's own proportions, driven through the conversion.

Any end-effector position error is then purely from the joint-count mismatch (spine0, wrist_twist,
thumb0, pinky0, foot joints folded into VRM bones). Also tests moving forearm twist from the hand
onto lowerArm (driving lowerArm with wrist_twist instead of lowarm).
"""

import sys
from pathlib import Path

import numpy as np
import pymomentum.geometry as g

import convert
from convert import MAP, MhrToVrm, vrm_fk
from dump_skeleton import load

rng = np.random.default_rng(1)
ch = load()
jn = list(ch.skeleton.joint_names)
pn = list(ch.parameter_transform.names)
FLEX_T = {"spine_length_flexible", "neck_length_flexible", "shoulder_width_flexible", "arm_length_flexible",
          "hip_width_flexible", "leg_length_flexible"}
pose_idx = [i for i, n in enumerate(pn) if not n.startswith(("scale_", "root_t")) and n not in FLEX_T]
LIM = {}
for line in (Path(__file__).resolve().parents[2] / "data/mhr/assets/compact_v6_1.model").read_text().splitlines():
    q = line.split()
    if len(q) >= 4 and q[0] == "limit" and q[2] == "minmax":
        lo, hi = line.split("[")[1].split("]")[0].split(",")
        LIM[q[1]] = (float(lo), float(hi))
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

for twist_on_forearm in (False, True):
    if twist_on_forearm:
        for s, m in (("left", "l"), ("right", "r")):
            convert.MAP[f"{s}LowerArm"] = (f"{m}_wrist_twist", f"{m}_wrist", f"{m}_lowarm")
    conv = MhrToVrm(jn, rest, vrm_rest)
    err = {t: [] for t in TARGETS}
    for _ in range(300):
        mp = np.zeros(len(pn), np.float32)
        mp[pose_idx] = rng.uniform(lo, hi)
        st = np.asarray(g.model_parameters_to_skeleton_state(ch, mp))
        Pv, _ = vrm_fk(conv.locals(st), vrm_rest)
        root_m, root_v = st[jn.index("root"), :3], Pv["hips"]
        for t, j in TARGETS.items():
            err[t].append(np.linalg.norm((Pv[t] - root_v) - (st[jn.index(j), :3] - root_m)))
    print(f"\nposition error relative to hips, cm (mean / max), forearm twist on {'lowerArm' if twist_on_forearm else 'hand'}:")
    for t, e in err.items():
        print(f"  {t:18s} {np.mean(e):6.2f} {np.max(e):6.2f}")
