"""CMU -> MHR (performer fit) -> VRM avatars: do spatial relations survive plain rotation transfer?

Points: hands = middle-finger knuckle, face = midpoint between the eyes, chest = upperChest joint.
Relations: hand-face, hand-hand, hand-chest distances, normalized by each body's arm length
(upper arm + forearm). Reported on 'close' frames (performer's normalized distance < 0.5), as the
error converted to centimetres on the avatar. The MHR-proportion VRM is a sanity reference (~0).
"""

import numpy as np

from convert import MAP, MhrToVrm, vrm_fk
from fit_avatars import AVATARS
from fit_cmu import fit
from vrm_skeleton import load_vrm

CLIPS = [(28, 15), (15, 7), (32, 12), (28, 8), (13, 31), (29, 20), (79, 72), (80, 46), (30, 18), (79, 84)]
AVATAR_SET = ["AvatarSample_D", "Base_Male", "AvatarSample_E", "Sakurada_Fumiriya"]
PTS = {"lhand": ("leftMiddleProximal", "l_middle1"), "rhand": ("rightMiddleProximal", "r_middle1"),
       "chest": ("upperChest", "c_spine3")}
REL = [("lhand", "face"), ("rhand", "face"), ("lhand", "rhand"), ("lhand", "chest"), ("rhand", "chest")]


def points_vrm(P):
    out = {k: P[v] for k, (v, _) in PTS.items()}
    out["face"] = 0.5 * (P["leftEye"] + P["rightEye"])
    arm = np.linalg.norm(P["leftLowerArm"] - P["leftUpperArm"], axis=-1) + np.linalg.norm(P["leftHand"] - P["leftLowerArm"], axis=-1)
    return out, arm


def points_mhr(st, jn):
    out = {k: st[:, jn.index(m), :3] for k, (_, m) in PTS.items()}
    out["face"] = 0.5 * (st[:, jn.index("l_eye"), :3] + st[:, jn.index("r_eye"), :3])
    J = lambda n: st[:, jn.index(n), :3]
    arm = np.linalg.norm(J("l_lowarm") - J("l_uparm"), axis=-1) + np.linalg.norm(J("l_wrist") - J("l_lowarm"), axis=-1)
    return out, arm


def main():
    results = {}
    for subj, clip in CLIPS:
        ch, mp, states, res = fit(subj, clip, verbose=False)
        jn = list(ch.skeleton.joint_names)
        rest = np.asarray(__import__("pymomentum.geometry", fromlist=["x"]).model_parameters_to_skeleton_state(ch, np.zeros(mp.shape[1], np.float32)))
        # performer rest with its own scale parameters (pose zero, keep shared scales)
        perf_rest_mp = np.where([n.startswith("scale_") for n in ch.parameter_transform.names], mp[0], 0).astype(np.float32)
        perf_rest = np.asarray(__import__("pymomentum.geometry", fromlist=["x"]).model_parameters_to_skeleton_state(ch, perf_rest_mp))
        pm, arm_m = points_mhr(states, jn)
        dm = {r: np.linalg.norm(pm[r[0]] - pm[r[1]], axis=-1) / arm_m for r in REL}
        fit_res = float(np.median(res[:, [k for k in range(res.shape[1])]]))
        targets = {"mhr-proportions": {b: perf_rest[jn.index(v[2] if len(v) > 2 else v[0]), :3] for b, v in MAP.items()}}
        for a in AVATAR_SET:
            targets[a] = load_vrm(AVATARS / f"{a}.vrm")[2]
        for name, vrm_rest in targets.items():
            vrm_rest = {k: np.asarray(v, float) * (100.0 if name != "mhr-proportions" else 1.0) for k, v in vrm_rest.items()}
            conv = MhrToVrm(jn, perf_rest, vrm_rest)
            P = [vrm_fk(conv.locals(st), vrm_rest)[0] for st in states]
            P = {k: np.stack([p[k] for p in P]) for k in P[0]}
            pv, arm_v = points_vrm(P)
            for r in REL:
                dv = np.linalg.norm(pv[r[0]] - pv[r[1]], axis=-1) / arm_v
                close = dm[r] < 0.5
                if close.any():
                    err_cm = np.abs(dv - dm[r])[close] * arm_v[close]
                    results.setdefault(name, {}).setdefault(r, []).extend(err_cm.tolist())
        print(f"{subj:02d}_{clip:02d}: {len(states)} frames, median fit residual {fit_res:.1f} cm")
    print("\nrelation error on close frames after plain rotation transfer, cm on the avatar: median / p90 (n frames)")
    print(f"{'target':20s} " + " ".join(f"{a + '-' + b:>18s}" for a, b in REL))
    for name, rr in results.items():
        cells = [f"{np.median(v):5.1f} / {np.percentile(v, 90):5.1f} ({len(v):4d})" if v else "-" for v in (rr.get(r, []) for r in REL)]
        print(f"{name:20s} " + " ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
