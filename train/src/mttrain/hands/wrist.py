"""Combination step (merger stand-in): the pair hand model's hand orientation onto the body model's arm.

The pair model gives each hand's palm frame in the camera frame (rows: wrist -> middle knuckle, toward the
index knuckle, their cross product; from image points, right hands mirrored). Chain to the avatar:
1. camera frame -> the body model's output frame (`lifter_pose.R0`: v_model = R0^T v_camera);
2. palm frame -> MHR wrist orientation: the palm frame is rigid to the wrist, so the rotation from MHR's rest
   palm frame to the observed one, applied to the rest wrist orientation;
3. MHR wrist -> VRM hand global (`mtcore.convert.MhrToVrm`'s hand rotation offset);
4. relative to the body model's lower arm (VRM globals from its locals): the whole rotation goes onto the hand
   (the rendered ground truth also carries most of the forearm twist on the hand: median 120 deg about the
   forearm axis; moving it onto the lower arm, `twist_to_forearm`, put the lower arm 129 deg off the truth);
   bends (the part off the forearm twist) beyond MAX_BEND_DEG (a likely palm/back flip) keep the body model's hand.
"""

import numpy as np

from mtcore.convert import MAP, MhrToVrm, VRM_PARENT, qinv, qmul, qrot, twist
from mtcore.hand_features import palm_frame
from mtcore.lifter_pose import R0, mat_to_quat, quat_to_mat
from mtcore.mhr import positions, skeleton_state

MAX_BEND_DEG = 100.0
MIRROR = np.diag([-1.0, 1.0, 1.0])
CHAIN = {s: ["hips", "spine", "chest", "upperChest", f"{s}Shoulder", f"{s}UpperArm", f"{s}LowerArm"] for s in ("left", "right")}


def unmirror_frame(F):
    """Palm frame rows of mirrored points -> the frame of the unmirrored points (the third row flips: chirality)."""
    G = F @ MIRROR
    G[2] = -G[2]
    return G


class HandOrient:
    def __init__(self, ch, vrm_rest, twist_to_forearm=False):
        self.twist_to_forearm = twist_to_forearm
        pn = list(ch.parameter_transform.names)
        jn = list(ch.skeleton.joint_names)
        rest = skeleton_state(ch, np.zeros(len(pn)))
        P = positions(ch, np.zeros((1, len(pn)), np.float32))[0]
        from .skeleton import HandFK
        fk = HandFK(ch)
        self.F_rest, self.Q_rest = {}, {}
        for s in "lr":
            self.F_rest[s] = palm_frame(P[fk.mp21[s]])[1]
            self.Q_rest[s] = rest[jn.index(f"{s}_wrist"), 3:7]
        self.conv = MhrToVrm(jn, rest, vrm_rest)

    def vrm_hand_global(self, F_cam, side):
        """Palm frame rows in the camera frame (unmirrored) -> VRM hand global quaternion in the model frame."""
        F_m = F_cam @ R0  # rows: v_c^T R0 = (R0^T v_c)^T
        delta = F_m.T @ self.F_rest[side]  # observed = delta applied to rest
        G_w = qmul(mat_to_quat(delta), self.Q_rest[side])
        S = "left" if side == "l" else "right"
        b = f"{S}Hand"
        return qmul(qmul(G_w, self.conv.q_rest_inv[b]), self.conv.D[b])

    def combine(self, bones, hand_globals):
        """Body model's VRM locals + {side: VRM hand global} -> bones with lowerArm and hand replaced."""
        out = dict(bones)
        for side, Gh in hand_globals.items():
            S = "left" if side == "l" else "right"
            G = np.array([0, 0, 0, 1.0])
            for b in CHAIN[S]:
                G = qmul(G, np.asarray(bones.get(b, [0, 0, 0, 1.0]), float))
            G_lower = G
            local = qmul(qinv(G_lower), Gh)
            axis = np.array([1.0, 0, 0]) if S == "left" else np.array([-1.0, 0, 0])  # VRM lower arm along +-X
            tw = twist(local, axis)
            swing = qmul(qinv(tw), local)
            if 2 * np.degrees(np.arccos(min(1.0, abs(swing[3])))) > MAX_BEND_DEG:
                continue  # implausible bend: likely a palm/back flip, keep the body model's hand
            if self.twist_to_forearm:
                out[f"{S}LowerArm"] = qmul(np.asarray(bones[f"{S}LowerArm"], float), tw).tolist()
                out[f"{S}Hand"] = swing.tolist()
            else:
                out[f"{S}Hand"] = local.tolist()
        return out


def pair_track(ckpt, landmarks, body, out, avatar="AvatarSample_D", wrist=True):
    """Runtime landmark dump + body pose track (same frames) -> pose track with the pair model's fingers and, with
    `wrist`, its hand orientation combined onto the body model's forearm. A lost hand keeps its last pose."""
    import json

    from mtcore.mhr import load
    from mtcore.paths import AVATARS
    from mtcore.vrm import load_vrm

    from .evaluate2 import LearnedPair
    from .infer import FINGER_WORDS
    from .skeleton import HandFK

    ch = load()
    fk = HandFK(ch)
    run = LearnedPair(ckpt, fk)
    rest = load_vrm(AVATARS / f"{avatar}.vrm")[2]
    ho = HandOrient(ch, rest)
    conv = ho.conv
    last_q, last_G = {}, {}
    stats = {"frames": 0, "wrist_used": 0, "wrist_kept": 0}
    with open(landmarks) as fl, open(body) as fb, open(out, "w") as fo:
        for line_l, line_b in zip(fl, fb):
            f, b = json.loads(line_l), json.loads(line_b)
            res = run.step(f)
            stats["frames"] += 1
            for side, S in (("l", "left"), ("r", "right")):
                if f.get(f"{S}Hand2d"):  # update only while the hand is seen; otherwise hold
                    q, F = res[side]
                    last_q[side] = q
                    F = F if side == "l" else unmirror_frame(F)
                    last_G[side] = ho.vrm_hand_global(F, side)
            bones = {n: v for n, v in b["bones"].items() if not any(w in n for w in FINGER_WORDS)}
            if last_q:
                mp = fk.full(last_q.get("l", np.zeros(27)), "l", base=fk.full(last_q.get("r", np.zeros(27)), "r"))
                loc = conv.locals(skeleton_state(ch, mp)[None])
                for n, v in loc.items():
                    side = "l" if n.startswith("left") else "r"
                    if any(w in n for w in FINGER_WORDS) and side in last_q:
                        bones[n] = [round(float(x), 6) for x in v[0]]
            if wrist and last_G and all(k in bones for k in ("leftLowerArm", "rightLowerArm")):
                new = ho.combine(bones, last_G)
                for s, S in (("l", "left"), ("r", "right")):
                    if s in last_G:
                        stats["wrist_used" if new[f"{S}Hand"] != bones.get(f"{S}Hand") else "wrist_kept"] += 1
                bones = {k: [round(float(x), 6) for x in v] for k, v in new.items()}
            fo.write(json.dumps({"t": b["t"], "bones": bones}) + "\n")
    return stats


def main():
    import argparse
    ap = argparse.ArgumentParser(description="pair hand model fingers + wrist on a runtime body track (comparison videos)")
    ap.add_argument("ckpt")
    ap.add_argument("landmarks")
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--avatar", default="AvatarSample_D")
    ap.add_argument("--no-wrist", action="store_true")
    a = ap.parse_args()
    print(pair_track(a.ckpt, a.landmarks, a.body, a.out, a.avatar, wrist=not a.no_wrist))


if __name__ == "__main__":
    main()
