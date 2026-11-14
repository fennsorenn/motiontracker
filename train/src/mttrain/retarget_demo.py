"""Relation-preserving retargeting demo (`mtcore.relretarget`) on rendered sets: the true body -> avatar.

    uv run python -m mttrain.retarget_demo ../data/m1/renders/contact1/seq00 --out-dir ../data/retarget/contact1-seq00
        [--avatar AvatarSample_F] [--tips]

Source: the render's true MHR pose (seq.npz: parameters, posed mesh of its identity). Rotation copy: the render's
gt.jsonl (the harness's ground truth on the avatar). Writes <out-dir>/relational.poses.jsonl and prints, per
segment, hand-hand and hand-head gaps on the true body, the rotation copy and the relational result.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mtcore.convert import MAP, MhrToVrm
from mtcore.lifter_pose import quat_to_mat
from mtcore.mhr import load, skeleton_state
from mtcore.paths import AVATARS
from mtcore.relretarget import (AXIS_END, PALM, SIDES, TIPS, TORSO, Avatar, Body, RelRetarget, _axis_coords,
                                head_length, profiles)
from mtcore.vrm import load_vrm, load_vrm_mesh

ANCHORS = list(TORSO) + [f"{s}{b}" for s in SIDES for b in ("UpperArm", "LowerArm")]


def human_body(ch, mp, verts0, av):
    """True MHR pose (T, P) and its frame-0 mesh (metres) -> Body in VRM bone names, with the anchor frames rotation
    copy gives the avatar `av` (MhrToVrm globals times the avatar's rest frames)."""
    jn = list(ch.skeleton.joint_names)
    parents = list(ch.skeleton.joint_parents)
    st = skeleton_state(ch, mp)
    rest = skeleton_state(ch, np.zeros((1, mp.shape[1]), np.float32))[0]
    j_of = {b: jn.index(MAP[b][0]) for b in MAP if MAP[b][0] in jn}
    need = set(ANCHORS) | {AXIS_END[a] for a in ANCHORS if AXIS_END[a]} | {b for s in SIDES for b in PALM[s] + TIPS[s]}
    pos = {b: st[:, j_of[b], :3] / 100.0 for b in need}
    conv = MhrToVrm(jn, rest, av.rest)
    G = conv.globals(st)
    frames, lengths, axes = {}, {}, {}
    for a in ANCHORS:
        frames[a] = quat_to_mat(G[a]) @ av.rest_frames[a]
        if AXIS_END[a] is None:
            axes[a] = np.array([1.0, 0.0, 0.0])
        else:
            v = np.einsum("tji,tj->ti", frames[a], pos[AXIS_END[a]] - pos[a]).mean(0)
            axes[a] = v / np.linalg.norm(v)
    # mesh labels: strongest joint -> nearest mapped ancestor
    bone_of = {j: b for b, j in j_of.items()}
    W = ch.skin_weights.to_dense(len(jn))
    lab = []
    for j in W.argmax(1):
        while j >= 0 and j not in bone_of:
            j = parents[j]
        lab.append(bone_of.get(j, ""))
    lab = np.array(lab)
    p0 = {a: pos[a][0] for a in ANCHORS}
    f0 = {a: frames[a][0] for a in ANCHORS}
    for a in ANCHORS:
        lengths[a] = head_length(verts0, lab, p0[a], f0[a]) if AXIS_END[a] is None else \
            float(np.linalg.norm(pos[AXIS_END[a]][0] - pos[a][0]))
    if "head" in ANCHORS:  # head length along the shared up axis
        lengths["head"] = head_length(verts0, lab, p0["head"], f0["head"])
    prof = profiles(verts0, lab, p0, f0, lengths, ANCHORS, axes)
    return Body(pos, frames, lengths, prof, axes)


def head_gap(body, pts):
    """Smallest palm-to-head-surface gap (T,) of the two palms, measured with the body's own head profile."""
    out = []
    for s in SIDES:
        local = np.einsum("tji,tj->ti", body.frames["head"], pts[s] - body.pos["head"])
        u, d, rho = _axis_coords(local, body.lengths["head"], body.axes["head"])
        out.append(rho - body.prof["head"].radius(u, d))
    return np.minimum(*out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("render", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--avatar", default="AvatarSample_F")
    ap.add_argument("--tips", action="store_true", help="finger tips as hand points too (needs finger bones in gt)")
    a = ap.parse_args()
    ch = load()
    z = np.load(a.render / "seq.npz")
    _, _, rest = load_vrm(AVATARS / f"{a.avatar}.vrm")
    verts, labels = load_vrm_mesh(AVATARS / f"{a.avatar}.vrm")
    av = Avatar(rest, verts, labels, ANCHORS)
    src = human_body(ch, z["mp"], z["verts"][0].astype(np.float64), av)
    gt = [json.loads(line) for line in open(a.render / "gt.jsonl")]
    bones = list(gt[0]["bones"])
    local = {b: np.array([g["bones"][b] for g in gt], dtype=np.float64) for b in bones}
    tips = a.tips and all(t in bones for s in SIDES for t in TIPS[s])
    rr = RelRetarget(av, tips=tips)
    out = rr.solve(src, local)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / "relational.poses.jsonl", "w") as fo:
        for i, g in enumerate(gt):
            fo.write(json.dumps({"t": g["t"], "bones": {b: out[b][i].round(6).tolist() for b in bones}}) + "\n")
    # gaps per segment
    copy_b, _ = av.body(local)
    rel_b, _ = av.body(out)
    rows = {"true": src, "copy": copy_b, "relational": rel_b}
    fps = float(z["fps"])
    segs = json.load(open(a.render / "contacts.json"))
    print(f"{a.render.name}: hands {'palm + tips' if tips else 'palm'}; medians in cm "
          "(palm-palm distance (left-right separation, negative = crossed) / smallest palm gap to the head surface"
          + (" / closest hand-hand point pair, palm or tip" if tips else "") + ")")
    for c in segs:
        k0, k1 = int(c["t0"] / 1000 * fps), int(c["t1"] / 1000 * fps)
        cells = []
        for name, b in rows.items():
            pts = {s: b.point(PALM[s])[k0:k1] for s in SIDES}
            sub = Body({k: v[k0:k1] for k, v in b.pos.items()}, {k: v[k0:k1] for k, v in b.frames.items()}, b.lengths,
                       b.prof, b.axes)
            pp = np.median(np.linalg.norm(pts["left"] - pts["right"], axis=-1)) * 100
            # left-right separation along the chest's left axis (-y of the frame): negative = hands crossed through
            lat = -np.einsum("tji,tj->ti", sub.frames["upperChest"], pts["left"] - pts["right"])[:, 1]
            hg = np.median(head_gap(sub, pts)) * 100
            hp = {sd: [v[k0:k1] for v in b.hand_points(sd, tips).values()] for sd in SIDES}
            closest = np.min([np.linalg.norm(p - q, axis=-1) for p in hp["left"] for q in hp["right"]], 0)
            cells.append(f"{name} {pp:5.1f} ({np.median(lat) * 100:+5.1f}) / {hg:5.1f}"
                         + (f" / {np.median(closest) * 100:4.1f}" if tips else ""))
        print(f"  {c['variant']:16s} {c['t0'] / 1000:5.1f}-{c['t1'] / 1000:5.1f}s  " + " | ".join(cells))
    print(f"-> {a.out_dir}/relational.poses.jsonl")


if __name__ == "__main__":
    main()
