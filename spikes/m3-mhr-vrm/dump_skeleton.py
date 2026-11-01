"""Dump MHR's rest skeleton and the effect of its scale parameters.

Uses pymomentum directly: MHR.from_files segfaults in Character.with_blend_shape
(pymomentum-cpu 0.1.114, torch 2.8 CPU), and the skeleton never depends on the
identity blendshapes anyway (see mhr/mhr.py forward: skel_state comes from model
parameters only).
"""

from pathlib import Path

import numpy as np
import pymomentum.geometry as g

ASSETS = Path(__file__).resolve().parents[2] / "data" / "mhr" / "assets"


def load():
    return g.Character.load_fbx(str(ASSETS / "lod1.fbx"), str(ASSETS / "compact_v6_1.model"), load_blendshapes=False)


def positions(ch, mp):
    return np.asarray(g.model_parameters_to_skeleton_state(ch, np.asarray(mp, dtype=np.float32)))[..., :3]


if __name__ == "__main__":
    ch = load()
    names = list(ch.skeleton.joint_names)
    parents = list(ch.skeleton.joint_parents)
    pnames = list(ch.parameter_transform.names)
    print(f"joints: {len(names)}, parameters: {len(pnames)}")
    base = positions(ch, np.zeros(len(pnames), np.float32))
    J = names.index
    print("height-ish: root y %.2f, head y %.2f, l_wrist y %.2f (units as stored)" % (base[J("root"), 1], base[J("c_head"), 1], base[J("l_wrist"), 1]))

    for p in ["scale_uparms", "scale_lowarms", "scale_shoulder_width", "scale_spine_length", "scale_neck_length", "scale_l_hands"]:
        mp = np.zeros(len(pnames), np.float32); mp[pnames.index(p)] = 0.2
        d = np.linalg.norm(positions(ch, mp) - base, axis=1)
        top = np.argsort(-d)[:3]
        print(f"{p:22s} +0.2 -> " + ", ".join(f"{names[i]} {d[i]:.2f}" for i in top))

    print("\nrest skeleton (no procedural/null joints):")
    for i, (n, p) in enumerate(zip(names, parents)):
        if "_proc" in n or "null" in n:
            continue
        par = names[p] if p >= 0 else "-"
        L = np.linalg.norm(base[i] - base[p]) if p >= 0 else 0
        print(f"  {n:20s} parent={par:16s} pos=({base[i,0]:7.2f},{base[i,1]:7.2f},{base[i,2]:7.2f}) len={L:6.2f}")
