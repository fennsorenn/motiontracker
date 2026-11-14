"""Relation-preserving retargeting demo (`mtcore.relretarget`): human body -> avatar, rotation copy vs relational.

    # rendered set: the true body (pose and identity mesh from seq.npz)
    uv run python -m mttrain.retarget_demo --render ../data/m1/renders/contact1/seq00 --out-dir <dir> [--tips]
    # recorded session: the merger's output on MHR's mean body
    uv run python -m mttrain.retarget_demo --merger ../data/merger/runs/mg1/best.pt --landmarks <landmarks.jsonl> \\
        --out-dir <dir> [--tips]
    [--avatar AvatarSample_F]

Writes <out-dir>/{copy,relational}.poses.jsonl (all bones the avatar and MHR share: legs and fingers included) and
prints per segment (rendered sets: contacts.json) palm-palm distance, left-right separation, palm gap to the head
surface, palm gap to the torso/thigh surface and the closest hand-hand point pair, for the human body, the rotation
copy and the relational result. Rendered sets are posed camera-relative like their gt.jsonl.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mtcore.convert import MAP, MhrToVrm
from mtcore.lifter_pose import R0, mat_to_quat, quat_to_mat
from mtcore.mhr import load, skeleton_state
from mtcore.paths import AVATARS
from mtcore.relretarget import (AXIS_END, PALM, SIDES, TIPS, TORSO, Avatar, Body, RelRetarget, _axis_coords,
                                head_length, profiles)
from mtcore.vrm import load_vrm, load_vrm_mesh

ANCHORS = list(TORSO) + [f"{s}{b}" for s in SIDES for b in ("UpperArm", "LowerArm", "UpperLeg", "LowerLeg")]
BODY_SURFACE = ("hips", "spine", "chest", "upperChest", "leftUpperLeg", "rightUpperLeg")


def human_body(ch, st, mesh, mesh_st, av):
    """MHR skeleton states (T, J, 8) (cm) and a mesh (V, 3) (cm) posed as mesh_st (J, 8) -> Body in VRM bone names,
    with the anchor frames rotation copy gives the avatar `av` (MhrToVrm globals times the avatar's rest frames)."""
    jn = list(ch.skeleton.joint_names)
    parents = list(ch.skeleton.joint_parents)
    rest = skeleton_state(ch, np.zeros(len(ch.parameter_transform.names)))
    conv = MhrToVrm(jn, rest, av.rest)
    j_of = {b: jn.index(MAP[b][0]) for b in MAP if MAP[b][0] in jn}
    need = set(ANCHORS) | {AXIS_END[a] for a in ANCHORS if AXIS_END[a]} | {b for s in SIDES for b in PALM[s] + TIPS[s]}

    def geometry(states):
        pos = {b: states[..., j_of[b], :3] / 100.0 for b in need}
        G = conv.globals(states)
        return pos, {a: quat_to_mat(G[a]) @ av.rest_frames[a] for a in ANCHORS}

    pos, frames = geometry(st)
    p0, f0 = geometry(mesh_st)
    verts = mesh / 100.0
    bone_of = {j: b for b, j in j_of.items()}  # mesh labels: strongest joint -> nearest mapped ancestor
    lab = []
    for j in ch.skin_weights.to_dense(len(jn)).argmax(1):
        while j >= 0 and j not in bone_of:
            j = parents[j]
        lab.append(bone_of.get(j, ""))
    lab = np.array(lab)
    lengths, axes = {}, {}
    for a in ANCHORS:
        if AXIS_END[a] is None:  # the head: up in the shared frame, as long as the head is high
            axes[a] = np.array([1.0, 0.0, 0.0])
            lengths[a] = head_length(verts, lab, p0[a], f0[a])
        else:
            v = f0[a].T @ (p0[AXIS_END[a]] - p0[a])
            lengths[a] = float(np.linalg.norm(v))
            axes[a] = v / lengths[a]
    prof = profiles(verts, lab, p0, f0, lengths, ANCHORS, axes)
    return Body(pos, frames, lengths, prof, axes), conv


def rotate_states(st, M):
    """Skeleton states turned by a world rotation M (positions and orientations)."""
    out = st.copy()
    out[..., :3] = st[..., :3] @ M.T
    out[..., 3:7] = mat_to_quat(M @ quat_to_mat(st[..., 3:7]))
    return out


def merger_states(ch, r):
    """Merger output (`merger.run.infer`) -> MHR skeleton states (T, J, 8) of the mean body: the body joints from the
    merger's rotations (`LifterPose.mhr_globals`), the fingers from its finger parameters, positions by FK."""
    jn = list(ch.skeleton.joint_names)
    parents = list(ch.skeleton.joint_parents)
    rest = skeleton_state(ch, np.zeros(len(ch.parameter_transform.names)))
    G = r["runner"].pose.mhr_globals(r["L"]).copy()  # (T, J, 3, 3)
    fk = r["fk"]
    sf = skeleton_state(ch, fk.full(r["fingers"]["l"], "l", base=fk.full(r["fingers"]["r"], "r")).astype(np.float32))
    Gf = quat_to_mat(sf[..., 3:7])
    for m in "lr":
        w = jn.index(f"{m}_wrist")
        for j in range(len(jn)):
            k = j
            while k >= 0 and k != w:
                k = parents[k]
            if k == w and j != w:  # finger joint: its rotation relative to the wrist from the finger pose
                G[:, j] = G[:, w] @ np.swapaxes(Gf[:, w], -1, -2) @ Gf[:, j]
    rest_g = quat_to_mat(rest[:, 3:7])
    T = len(G)
    P = np.zeros((T, len(jn), 3))
    for j in range(len(jn)):  # MHR joints are ordered parents first
        p = parents[j]
        P[:, j] = rest[j, :3] if p < 0 else P[:, p] + np.einsum("tij,j->ti", G[:, p] @ rest_g[p].T, rest[j, :3] - rest[p, :3])
    st = np.zeros((T, len(jn), 8))
    st[..., :3], st[..., 3:7], st[..., 7] = P, mat_to_quat(G), 1.0
    return st


def surface_gap(body, pts, anchors):
    """Smallest gap (T,) of the two palms to the surfaces of `anchors`, measured with the body's own profiles."""
    out = []
    for s in SIDES:
        own = {f"{s}UpperArm", f"{s}LowerArm"}
        for a in anchors:
            if a in own:
                continue
            local = np.einsum("tji,tj->ti", body.frames[a], pts[s] - body.pos[a])
            u, d, rho = _axis_coords(local, body.lengths[a], body.axes[a])
            out.append(rho - body.prof[a].radius(u, d))
    return np.min(out, 0)


def report(rows, segs, fps, tips):
    print("medians in cm: palm-palm (left-right separation, negative = crossed) / palm gap to head / palm gap to torso "
          "or thigh" + (" / closest hand-hand point pair, palm or tip" if tips else ""))
    for c in segs:
        k0, k1 = int(c["t0"] / 1000 * fps), int(c["t1"] / 1000 * fps)
        cells = []
        for name, b in rows.items():
            sub = Body({k: v[k0:k1] for k, v in b.pos.items()}, {k: v[k0:k1] for k, v in b.frames.items()}, b.lengths,
                       b.prof, b.axes)
            pts = {s: sub.point(PALM[s]) for s in SIDES}
            pp = np.median(np.linalg.norm(pts["left"] - pts["right"], axis=-1)) * 100
            lat = -np.einsum("tji,tj->ti", sub.frames["upperChest"], pts["left"] - pts["right"])[:, 1]  # chest's left: -y
            hg = np.median(surface_gap(sub, pts, ("head",))) * 100
            bg = np.median(surface_gap(sub, pts, BODY_SURFACE)) * 100
            cell = f"{name} {pp:5.1f} ({np.median(lat) * 100:+5.1f}) / {hg:5.1f} / {bg:5.1f}"
            if tips:
                hp = {sd: list(sub.hand_points(sd, True).values()) for sd in SIDES}
                closest = np.min([np.linalg.norm(p - q, axis=-1) for p in hp["left"] for q in hp["right"]], 0)
                cell += f" / {np.median(closest) * 100:4.1f}"
            cells.append(cell)
        print(f"  {c['variant']:14s} {c['t0'] / 1000:5.1f}-{c['t1'] / 1000:5.1f}s  " + " | ".join(cells))


def write(path, times_ms, locs, bones):
    with open(path, "w") as fo:
        for i, t in enumerate(times_ms):
            fo.write(json.dumps({"t": round(float(t), 1), "bones": {b: locs[b][i].round(6).tolist() for b in bones}}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", type=Path, help="rendered sequence directory (true body)")
    ap.add_argument("--merger", type=Path, help="merger checkpoint (with --landmarks: the merger's output)")
    ap.add_argument("--landmarks", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--avatar", default="AvatarSample_F")
    ap.add_argument("--tips", action="store_true", help="finger tips as hand points too")
    a = ap.parse_args()
    _, _, rest = load_vrm(AVATARS / f"{a.avatar}.vrm")
    verts, labels = load_vrm_mesh(AVATARS / f"{a.avatar}.vrm")
    av = Avatar(rest, verts, labels, ANCHORS)
    if a.render:
        ch = load()
        z = np.load(a.render / "seq.npz")
        M = R0.T @ z["R"]  # camera-relative, like gt.jsonl
        st = rotate_states(skeleton_state(ch, z["mp"].astype(np.float64)), M)
        src, conv = human_body(ch, st, z["verts"][0].astype(np.float64) @ M.T * 100.0, st[0], av)
        fps = float(z["fps"])
        times_ms = np.arange(len(st)) / fps * 1000
        segs = json.load(open(a.render / "contacts.json")) if (a.render / "contacts.json").exists() else []
    else:
        from .merger.run import infer
        rows = [json.loads(line) for line in open(a.landmarks) if line.strip()]
        r = infer(a.merger, rows)
        ch = r["ch"]
        st = merger_states(ch, r)
        rest_st = skeleton_state(ch, np.zeros(len(ch.parameter_transform.names)))
        src, conv = human_body(ch, st, np.asarray(ch.mesh.vertices, dtype=np.float64), rest_st, av)
        times_ms = r["times"] * 1000
        fps, segs = None, []
    local = conv.locals(st)
    bones = [b for b in local if b in av.rest]
    rr = RelRetarget(av, tips=a.tips)
    out = rr.solve(src, local)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    write(a.out_dir / "copy.poses.jsonl", times_ms, local, bones)
    write(a.out_dir / "relational.poses.jsonl", times_ms, out, bones)
    copy_b, _ = av.body(local)
    rel_b, _ = av.body(out)
    if segs:
        report({"human": src, "copy": copy_b, "relational": rel_b}, segs, fps, a.tips)
    moved = np.max([np.linalg.norm(rel_b.pos[f"{s}Hand"] - copy_b.pos[f"{s}Hand"], axis=-1) for s in SIDES], 0) * 100
    print(f"wrist moved by the relational step (cm): median {np.median(moved):.1f}, 90% {np.percentile(moved, 90):.1f}, "
          f"max {moved.max():.1f}")
    print(f"-> {a.out_dir}/{{copy,relational}}.poses.jsonl")


if __name__ == "__main__":
    main()
