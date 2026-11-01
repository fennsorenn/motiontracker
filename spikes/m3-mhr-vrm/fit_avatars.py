"""Fit VRM avatars' proportions with MHR scale parameters plus one uniform scale.

Segment lengths only (pose-independent). For each avatar: uniform scale s (median length ratio),
then each MHR scale parameter solves its segment exactly; report the needed value relative to
MHR's limits, and the residual of segments MHR has no parameter for.
"""

import sys
from pathlib import Path

import numpy as np

from dump_skeleton import load, positions
from vrm_skeleton import load_vrm

AVATARS = Path(__file__).resolve().parents[3] / "volleyball" / "tools" / "avatars" / "models"
NAMES = ["AvatarSample_D", "AvatarSample_D_Darkness", "AvatarSample_E", "AvatarSample_F", "AvatarSample_G",
         "Base_Female", "Base_Male", "HairSample_Female", "HairSample_Male", "Sakurada_Fumiriya", "Sendagaya_Shino"]

# segment: (VRM a, VRM b, MHR a, MHR b, MHR parameter or None)
SEGMENTS = {
    "spine":      ("hips", "neck", "root", "c_neck", "scale_spine_length"),
    "neck":       ("neck", "head", "c_neck", "c_head", "scale_neck_length"),
    "shoulders":  ("leftUpperArm", "rightUpperArm", "l_uparm", "r_uparm", "scale_shoulder_width"),
    "upper_arm":  ("leftUpperArm", "leftLowerArm", "l_uparm", "l_lowarm", "scale_uparms"),
    "forearm":    ("leftLowerArm", "leftHand", "l_lowarm", "l_wrist", "scale_lowarms"),
    "hand":       ("leftHand", "leftMiddleProximal", "l_wrist", "l_middle1", "scale_l_hands"),
    "hips":       ("leftUpperLeg", "rightUpperLeg", "l_upleg", "r_upleg", "scale_hip_width"),
    "upper_leg":  ("leftUpperLeg", "leftLowerLeg", "l_upleg", "l_lowleg", "scale_uplegs"),
    "lower_leg":  ("leftLowerLeg", "leftFoot", "l_lowleg", "l_foot", "scale_lowlegs"),
    "clavicle":   ("leftShoulder", "leftUpperArm", "l_clavicle", "l_uparm", None),
    "head_to_eye": ("head", "leftEye", "c_head", "l_eye", None),
}


def mhr_limits(path):
    lim = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 4 and p[0] == "limit" and p[2] == "minmax":
            lo, hi = line.split("[")[1].split("]")[0].split(",")
            lim[p[1]] = (float(lo), float(hi))
    return lim


def main():
    ch = load()
    jn = list(ch.skeleton.joint_names)
    pn = list(ch.parameter_transform.names)
    lim = mhr_limits(Path(__file__).resolve().parents[2] / "data/mhr/assets/compact_v6_1.model")
    zero = np.zeros(len(pn), np.float32)
    base = positions(ch, zero)

    def seg_len(P, a, b):
        return float(np.linalg.norm(P[jn.index(a)] - P[jn.index(b)]))

    # d(length)/d(param) by finite differences, in cm per parameter unit
    deriv = {}
    for s, (_, _, ma, mb, p) in SEGMENTS.items():
        if p:
            mp = zero.copy(); mp[pn.index(p)] = 0.1
            deriv[s] = (seg_len(positions(ch, mp), ma, mb) - seg_len(base, ma, mb)) / 0.1
    mhr_len = {s: seg_len(base, v[2], v[3]) for s, v in SEGMENTS.items()}

    print("Needed parameter value as a fraction of MHR's limit (|x| > 1 = outside MHR's human range).")
    print("Unparameterized segments: avatar/MHR length ratio after uniform scale (1.00 = matches).\n")
    cols = list(SEGMENTS)
    print(f"{'avatar':24s} {'scale':>5s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
    for name in NAMES:
        _, _, J = load_vrm(AVATARS / f"{name}.vrm")
        if "leftEye" not in J:
            J["leftEye"] = np.full(3, np.nan)
        L = {s: 100 * float(np.linalg.norm(J[v[0]] - J[v[1]])) for s, v in SEGMENTS.items()}
        ratios = [L[s] / mhr_len[s] for s in ("spine", "upper_arm", "forearm", "upper_leg", "lower_leg")]
        scale = float(np.median(ratios))
        cells = []
        for s, (_, _, _, _, p) in SEGMENTS.items():
            target = L[s] / scale
            if p:
                need = (target - mhr_len[s]) / deriv[s]
                lo, hi = lim.get(p, (-np.inf, np.inf))
                frac = need / (hi if need >= 0 else -lo) if (hi if need >= 0 else -lo) else np.inf
                cells.append(f"{frac:+9.2f}")
            else:
                cells.append(f"{target / mhr_len[s]:9.2f}")
        print(f"{name:24s} {scale:5.2f} " + " ".join(cells))


if __name__ == "__main__":
    main()
