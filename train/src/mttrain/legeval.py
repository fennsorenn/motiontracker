"""Legs, root position and up direction of a body model (m1-v12+) on rendered sets and real sessions.

    uv run python -m mttrain.legeval data/m1/runs/m1-v12/best.pt --renders m1/v1 m1/rest1 m1/contact1 m4/pairs3 \
        [--sessions 2026-11-01T15-29-11-session ...]
    uv run python -m mttrain.legeval data/merger/runs/mg3/best.pt --merger --renders ...   # the merger's output

Rendered sets (truth from seq.npz: MHR parameters and the camera): per set the median global angle error of the
head, chest, upper arms, forearms and wrists (any body model, so older ones compare), and (m1-v12+) the median
local angle error of thighs,
shins and feet, the root position error in the image plane and in depth (cm), the up direction error (deg), and the
knee position error relative to the root (cm). Detector frames are matched to the nearest rendered frame.

Real sessions (no truth): sitting vs standing from the predicted pose, per frame the angle between the trunk's up
axis and the thighs (seated ~90 deg, standing ~180): share of frames that read as standing (> 135 deg), the number
of sitting/standing switches, and the frame-to-frame jitter of the thigh direction and the root position.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mtcore.lifter_pose import quat_to_mat
from mtcore.mhr import skeleton_state
from mtcore.paths import DATA, REPO_ROOT

from .infer import Runner, read_session
from .skeleton import OUT_JOINTS, TargetExtractor, fk

LEGS = {"thigh": ("l_upleg", "r_upleg"), "shin": ("l_lowleg", "r_lowleg"), "foot": ("l_subtalar", "r_subtalar")}
ARMS = {"head": ("c_head",), "chest": ("c_spine3",), "uparm": ("l_uparm", "r_uparm"), "lowarm": ("l_lowarm", "r_lowarm"),
        "wrist": ("l_wrist", "r_wrist")}


def angle(Ra, Rb):
    c = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


class MergerAsRunner:
    """The merger behind the Runner interface legeval uses: predict -> locals, aux root/up."""

    def __init__(self, ckpt):
        self.ckpt = ckpt
        self.aux = None
        self.model = type("M", (), {"root": True})()

    def predict_session(self, d):
        from .merger.run import infer
        rows = [json.loads(line) for line in open(Path(d) / "landmarks.jsonl") if line.strip()]
        r = infer(self.ckpt, rows)
        self.runner = r["runner"]
        self.ch, self.jn = r["ch"], r["runner"].jn
        self.aux = {"root": r["root"], "up": r["up"]} if r["root"] is not None else None
        return r["L"]

    def mhr_globals(self, rot):
        return self.runner.mhr_globals(rot)


def render_set(r, te, ch, d):
    z = np.load(d / "seq.npz")
    st = skeleton_state(ch, z["mp"].astype(np.float64))
    fps = float(z["fps"])
    times, world, image, vis, present = read_session(d)
    rot = r.predict_session(d) if isinstance(r, MergerAsRunner) else r.predict(times, world, image, vis, present)[0]
    k = np.clip(np.round(times * fps).astype(int), 0, len(st) - 1)
    rt, pos_t, off_t = te(st[k], z["R"])
    G, _ = fk(rot, off_t)
    Gt, _ = fk(rt, off_t)
    out = {f"glob_{g}": np.concatenate([angle(G[present, OUT_JOINTS.index(j)], Gt[present, OUT_JOINTS.index(j)]) for j in js])
           for g, js in ARMS.items()}
    if r.aux is None:  # older body models: no legs, root or up
        return out
    root_t, up_t = te.root(st[k], z["R"], z["t"])
    for g, js in LEGS.items():
        out[f"deg_{g}"] = np.concatenate([angle(rot[:, OUT_JOINTS.index(j)], rt[:, OUT_JOINTS.index(j)]) for j in js])
    _, P = fk(rot, off_t)
    out["cm_knee"] = np.concatenate([np.linalg.norm(P[:, OUT_JOINTS.index(j)] - pos_t[:, OUT_JOINTS.index(j)], axis=-1)
                                     for j in ("l_lowleg", "r_lowleg")]) * 100
    d_root = (r.aux["root"] - root_t) * 100
    out["cm_root_xy"] = np.linalg.norm(d_root[:, :2], axis=-1)
    out["cm_root_z"] = np.abs(d_root[:, 2])
    out["deg_up"] = np.degrees(np.arccos(np.clip((r.aux["up"] * up_t).sum(-1), -1, 1)))
    return out


def trunk_thigh(r, rot):
    """Angle (T,) between the trunk's up axis and the thighs (mean of both), from the predicted rotations."""
    G = r.mhr_globals(rot)  # MHR world joint rotations (rest included)
    jn = r.jn
    rest = skeleton_state(r.ch, np.zeros(len(r.ch.parameter_transform.names)))
    rg = quat_to_mat(rest[:, 3:7])
    out = []
    for m in "lr":
        a, b = jn.index(f"{m}_upleg"), jn.index(f"{m}_lowleg")
        thigh = np.einsum("tij,j->ti", G[:, a] @ rg[a].T, rest[b, :3] - rest[a, :3])
        out.append(thigh / np.linalg.norm(thigh, axis=-1, keepdims=True))
    root = jn.index("root")
    up = np.einsum("tij,j->ti", G[:, root] @ rg[root].T, np.array([0.0, 1.0, 0.0]))  # the pelvis's rest up, carried
    thigh = (out[0] + out[1]) / 2
    thigh /= np.linalg.norm(thigh, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((thigh * up).sum(-1), -1, 1))), thigh


def session(r, path):
    times, world, image, vis, present = read_session(path)
    if isinstance(r, MergerAsRunner):
        rows = [json.loads(line) for line in open(Path(path) / "landmarks.jsonl") if line.strip()]
        from .merger.run import infer
        res = infer(r.ckpt, rows)
        r.runner, r.ch, r.jn = res["runner"], res["ch"], res["runner"].jn
        r.aux = {"root": res["root"], "up": res["up"]}
        rot = res["L"]
    else:
        rot, _ = r.predict(times, world, image, vis, present)
    ang, thigh = trunk_thigh(r, rot)
    standing = ang > 135
    switches = int(np.sum(standing[1:] != standing[:-1]))
    dth = np.degrees(np.arccos(np.clip((thigh[1:] * thigh[:-1]).sum(-1), -1, 1)))
    droot = np.linalg.norm(np.diff(r.aux["root"], axis=0), axis=-1) * 100
    return {"frames": len(times), "trunk_thigh_deg_p50": float(np.median(ang)), "standing_frac": float(standing.mean()),
            "switches": switches, "thigh_jitter_deg_p50": float(np.median(dth)),
            "root_step_cm_p50": float(np.median(droot)), "root_step_cm_p95": float(np.percentile(droot, 95)),
            "root_z_m_p50": float(np.median(r.aux["root"][:, 2])),
            "up_vs_camera_down_deg_p50": float(np.median(np.degrees(np.arccos(np.clip(-r.aux["up"][:, 1], -1, 1)))))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--renders", nargs="*", default=[])
    ap.add_argument("--sessions", nargs="*", default=[])
    ap.add_argument("--merger", action="store_true", help="ckpt is a merger: evaluate its output")
    a = ap.parse_args()
    if a.merger:
        from mtcore.mhr import load
        r = MergerAsRunner(a.ckpt)
        ch = load()
    else:
        r = Runner(a.ckpt)
        ch = r.ch
    te = TargetExtractor(list(ch.skeleton.joint_names), skeleton_state(ch, np.zeros(len(ch.parameter_transform.names))))
    rows = {}
    for s in a.renders:
        acc = {}
        for d in sorted((DATA / s.split("/")[0] / "renders" / s.split("/")[1]).glob("seq*")):
            if not (d / "landmarks.jsonl").exists():
                continue
            for k, v in render_set(r, te, ch, d).items():
                acc.setdefault(k, []).append(v)
        rows[s] = {k: round(float(np.median(np.concatenate(v))), 1) for k, v in acc.items()}
        print(s, json.dumps(rows[s]), flush=True)
    for s in a.sessions if r.model.root else []:
        res = session(r, REPO_ROOT / "sessions" / s)
        rows[s] = {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}
        print(s, json.dumps(rows[s]), flush=True)
    (a.ckpt.parent / "legeval.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
