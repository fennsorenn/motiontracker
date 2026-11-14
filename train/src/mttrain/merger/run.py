"""Run the merger on a recorded or rendered session's landmarks -> VRM pose tracks for the evaluation harness and
the avatar videos: the merger, the base pose (hand-written combination) and the body model alone.

    uv run python -m mttrain.merger.run data/merger/runs/<name>/best.pt <landmarks.jsonl> --out-dir <dir>
        [--avatar AvatarSample_D]

Same chain as training (`batch`): body model (`infer.Runner`, all frames at once, causal), hand model
(`evaluate2.LearnedPair`, streaming), face-mesh head pose (`mtcore.facepose`), blendshapes from the frame if
present, the base combination, then the merger over its frame offsets (session start: the first frame repeated).
Writes <out-dir>/{merger,base,m1}.poses.jsonl ({"t", "bones"}, fingers included for merger and base).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mtcore.facepose import CanonicalFace, fit
from mtcore.hand_features import palm_frame
from mtcore.lifter_pose import mat_to_quat
from mtcore.mhr import load, positions, skeleton_state
from mtcore.paths import AVATARS, DATA
from mtcore.vrm import load_vrm

from ..gpugen.geom import quat_to_mat
from ..hands.evaluate2 import LearnedPair
from ..hands.infer import FINGER_WORDS
from ..hands.model import OUT_IDX
from ..hands.skeleton import HandFK
from ..infer import OUTPUT_BONES, Runner

LEG_BONES = [f"{s}{b}" for s in ("left", "right") for b in ("UpperLeg", "LowerLeg", "Foot")]
from .batch import MERGER, blend_bank, combine, unmirror_frame
from .model import OFFSETS, apply_root, corrected, frame_inputs, load_merger

M1 = DATA / "m1" / "runs" / "m1-v10" / "best.pt"
P3 = DATA / "m4" / "runs" / "p3" / "best.pt"


def part_outputs(rows, runner, pair, canon, face_offset, rest_head, F_rest, names):
    """Recorder-format frames -> the merger's per-frame inputs as a batch dict (B = 1, K = frames)."""
    T = len(rows)
    times = np.array([r["t"] for r in rows]) / 1000.0
    world, image, vis = np.zeros((T, 33, 3)), np.zeros((T, 33, 3)), np.zeros((T, 33))
    present = np.zeros(T, bool)
    for i, f in enumerate(rows):
        if f.get("pose") and f.get("pose2d"):
            p = np.asarray(f["pose"])
            world[i], vis[i], image[i] = p[:, :3], p[:, 3], np.asarray(f["pose2d"])[:, :3]
            present[i] = True
    rot, conf = runner.predict(times, world, image, vis, present)
    log_s = -np.log(np.clip(conf, 1e-6, None))
    root_aux = runner.aux  # m1-v12+: root position, its log scales and up; else None
    hand = {s: {k: [] for k in ("G", "rot_log_s", "fingers", "fingers_log_s", "pos", "log_size", "lost", "score")}
            for s in "lr"}
    head = {k: [] for k in ("G", "centre", "log_scale", "seen")}
    blend, bpresent = [], []
    pair.reset()
    for f in rows:
        pair.step(f)
        o = pair.raw
        for k, s in enumerate("lr"):
            F = o["rot"][k]
            F = F if s == "l" else unmirror_frame(F)
            h = hand[s]
            h["G"].append((F.T @ F_rest[s]).numpy())
            h["rot_log_s"].append(float(o["rot_log_s"][k]))
            h["fingers"].append(o["fingers"][k].numpy())
            h["fingers_log_s"].append(o["fingers_log_s"][k].numpy())
            h["pos"].append(o["pos"][k].numpy())
            h["log_size"].append(float(o["log_size"][k]))
            h["lost"].append(bool(pair.lost[k]))
            h["score"].append(float(pair.score[k]))
        if f.get("face") and len(f["face"]) >= 468:
            r = fit(canon, np.asarray(f["face"])[:468, :3])
            head["G"].append(r["R"] @ face_offset.T @ rest_head.T)
            head["centre"].append(r["centre"])
            head["log_scale"].append(np.log(r["scale"]))
            head["seen"].append(True)
        else:
            head["G"].append(np.eye(3))
            head["centre"].append(np.zeros(2))
            head["log_scale"].append(0.0)
            head["seen"].append(False)
        bs = f.get("blendshapes")
        blend.append([bs.get(n, 0.0) for n in names] if bs else [0.0] * len(names))
        bpresent.append(bool(bs))
    t = lambda x, dt=torch.float32: torch.tensor(np.asarray(x), dtype=dt)[None]
    b = {"body": {"rot": t(rot), "log_s": t(log_s), **({"root": t(root_aux["root"]), "root_log_s": t(root_aux["root_log_s"]),
                                                       "up": t(root_aux["up"])} if root_aux is not None else {})},
         "hand": {s: {k: t(v, torch.bool if k == "lost" else torch.float32) for k, v in hand[s].items()} for s in "lr"},
         "head": {k: t(v, torch.bool if k == "seen" else torch.float32) for k, v in head.items()},
         "blend": {"values": t(blend), "present": t(bpresent, torch.bool)}}
    b["base"] = combine(b["body"]["rot"], {s: b["hand"][s]["G"] for s in "lr"}, {s: b["hand"][s]["rot_log_s"] for s in "lr"},
                        b["head"]["G"], b["head"]["seen"].float() * 0.8)
    return times, b


def finger_locals(conv, fk, ch, q):
    """Finger parameters {side: (T, 27)} -> VRM finger bone locals {bone: (T, 4)}."""
    mp = fk.full(q["l"], "l", base=fk.full(q["r"], "r"))
    loc = conv.locals(skeleton_state(ch, mp))
    return {n: v for n, v in loc.items() if any(k in n for k in FINGER_WORDS)}


def infer(ckpt, rows, avatar="AvatarSample_D", m1=M1):
    """Recorder-format frames -> dict: times (T,) s, base batch `b`, merger body locals L (T, NJ, 3, 3), finger
    parameters {side: (T, 27)} of the merger and of the base (hand model), and the runner, hand FK and character."""
    ch = load()
    pn = list(ch.parameter_transform.names)
    jn = list(ch.skeleton.joint_names)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m1 = Path(ck.get("config", {}).get("m1", m1))  # the body model the merger was trained with
    runner = Runner(m1, avatar)
    fk = HandFK(ch)
    pair = LearnedPair(P3, fk)
    canon = CanonicalFace()
    fc = json.load(open(MERGER / "calib" / "face.json"))
    rest = skeleton_state(ch, np.zeros(len(pn)))
    rest_head = quat_to_mat(torch.tensor(rest[jn.index("c_head"), 3:7], dtype=torch.float32)).numpy()
    P0 = positions(ch, np.zeros((1, len(pn)), np.float32))[0]
    F_rest = {s: torch.tensor(palm_frame(P0[fk.mp21[s]])[1], dtype=torch.float32) for s in "lr"}
    blend_bank()
    from . import batch
    names = batch.ARKIT
    times, b = part_outputs(rows, runner, pair, canon, np.array(fc["offset"]), rest_head, F_rest, names)
    model = load_merger(ck)
    model.eval()
    feats = frame_inputs(b, model.nj, model.root)  # (1, T, D)
    T = feats.shape[1]
    idx = torch.tensor([[max(0, t - o) for o in OFFSETS] for t in range(T)])
    with torch.no_grad():
        out = model(feats[0, idx])  # (T, n_offsets, D) -> per frame
    L = corrected(b["base"][0, :, :model.nj], out["res"])
    if model.nj < b["base"].shape[2]:  # older mergers: the remaining joints (legs) as the base has them
        L = torch.cat([L, b["base"][0, :, model.nj:]], 1)
    L = L.numpy().astype(np.float64)
    root = up = None
    if model.root:
        root, up = (x.numpy().astype(np.float64) for x in apply_root(b["body"]["root"][0], b["body"]["up"][0], out))
    fingers, base_f = {}, {}
    for k, s in enumerate("lr"):
        for dst, f in ((fingers, b["hand"][s]["fingers"][0] + out["fingers_res"][:, k]), (base_f, b["hand"][s]["fingers"][0])):
            q = np.zeros((T, 27), np.float32)
            q[:, OUT_IDX] = f.numpy()
            dst[s] = np.clip(q, fk.lo, fk.hi)
    return {"times": times, "b": b, "L": L, "fingers": fingers, "base_fingers": base_f, "runner": runner, "fk": fk,
            "ch": ch, "root": root, "up": up}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("landmarks", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--avatar", default="AvatarSample_D")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.landmarks) if line.strip()]
    r = infer(a.ckpt, rows, a.avatar)
    times, b, L, fingers, base_f, runner, fk, ch = (r[k] for k in ("times", "b", "L", "fingers", "base_fingers", "runner",
                                                                    "fk", "ch"))
    T = len(times)
    conv = runner.pose.conv
    a.out_dir.mkdir(parents=True, exist_ok=True)
    body_root = (b["body"]["root"][0].numpy(), b["body"]["up"][0].numpy()) if "root" in b["body"] else None
    tracks = {"merger": (L, fingers, (r["root"], r["up"]) if r["root"] is not None else None),
              "base": (b["base"][0].numpy().astype(np.float64), base_f, body_root),
              "m1": (b["body"]["rot"][0].numpy().astype(np.float64), None, body_root)}
    for name, (rot, fq, rp) in tracks.items():
        bones = runner.vrm(rot)
        fl = finger_locals(conv, fk, ch, fq) if fq is not None else {}
        with open(a.out_dir / f"{name}.poses.jsonl", "w") as fo:
            for i, tt in enumerate(times):
                out_b = {n: bones[n][i].round(6).tolist() for n in OUTPUT_BONES + LEG_BONES if n in bones}
                out_b.update({n: v[i].round(6).tolist() for n, v in fl.items()})
                rec = {"t": round(tt * 1000, 1), "bones": out_b}
                if rp is not None:  # root position (camera frame, m) and up (camera frame)
                    rec.update(root=rp[0][i].round(4).tolist(), up=rp[1][i].round(4).tolist())
                fo.write(json.dumps(rec) + "\n")
    print(f"{T} frames -> {a.out_dir}/{{merger,base,m1}}.poses.jsonl")


if __name__ == "__main__":
    main()
