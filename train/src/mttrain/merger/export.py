"""Export a trained merger and its part models as one bundle for the live runtime (`mtcore.merged.LiveMerged`).

    uv run python -m mttrain.merger.export data/merger/runs/<name>/best.pt [--out <dir>] [--check <landmarks.jsonl>]

Writes <dir> (default: data/merger/runs/<name>/live/):
- body.onnx: the body model the merger was trained with (`config.m1`, a root model m1-v12+): x (N, O, FRAME_DIM + 2)
  -> rot (N, 23, 3, 3), log_scale (N, 23), root (N, 3), root_log_s (N, 2), up (N, 3);
- hands.onnx: the pair hand model (p3): own, other (N, HISTORY, FRAME_DIM) -> fingers, fingers_log_s, rot, rot_log_s,
  pos, log_size;
- merger.onnx: x (N, len(OFFSETS), frame dim) -> res, log_s, fingers_res, fingers_log_s, root_res, root_log_s, up_res;
- merger.json: body input settings, the MHR skeleton table with the finger map (`mtcore.lifter_pose`), output bones,
  default avatar, hand model constants, combination constants, the canonical face's Procrustes points and the head
  offset, the blendshape names, the merger's offsets.
With --check, runs the offline torch chain (`run.infer`) and the bundle (onnxruntime + numpy) on the same landmarks
and reports the largest bone, hip position and up differences. Weights trained on CMU-derived data: the bundle stays
in gitignored data/ (license register).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mtcore.convert import MAP
from mtcore.facepose import CanonicalFace
from mtcore.features import FRAME_DIM, HEAD3D_DIMS, OFFSETS as BODY_OFFSETS, SCALE_WINDOW, DEFAULT_SCALE
from mtcore.hand_features import palm_frame
from mtcore.lifter_pose import LifterPose, mat_to_quat
from mtcore.mhr import load, positions, skeleton_state
from mtcore.paths import AVATARS
from mtcore.vrm import load_vrm

from ..export import DEFAULT_AVATAR, LEG_BONES, _Flat
from ..gpugen.geom import quat_to_mat
from ..hands.evaluate2 import LearnedPair
from ..hands.infer import FINGER_WORDS
from ..hands.model import OUT_IDX
from ..hands.model2 import FRAME_DIM as HAND_DIM, HISTORY
from ..hands.skeleton import HandFK
from ..infer import OUTPUT_BONES
from ..model import load_lifter
from ..skeleton import OUT_JOINTS, PARENT_IDX
from . import batch
from .batch import CONF_DEG, MAX_BEND_DEG, MERGER
from .model import OFFSETS, frame_dim, load_merger
from .run import M1, P3

HEAD_WEIGHT = 0.8  # as run.part_outputs: the face head's weight in the base combination when seen


class _HandsFlat(torch.nn.Module):
    NAMES = ["fingers", "fingers_log_s", "rot", "rot_log_s", "pos", "log_size"]

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, own, other):
        o = self.m(own, other)
        return tuple(o[k] for k in self.NAMES)


class _MergerFlat(torch.nn.Module):
    NAMES = ["res", "log_s", "fingers_res", "fingers_log_s", "root_res", "root_log_s", "up_res"]

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m(x)
        return tuple(o[k] for k in self.NAMES)


def finger_table(ch, fk):
    """MHR finger joints and the linear map finger parameters (left 27, right 27) -> joint Euler angles (rx, ry, rz)."""
    M = ch.parameter_transform.transform
    M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    cols = np.concatenate([fk.idx["l"], fk.idx["r"]])
    rows = np.nonzero(np.abs(M[:, cols]).sum(1))[0]
    assert set(rows % 7) <= {3, 4, 5}, "finger parameters drive more than joint rotations"
    others = np.setdiff1d(np.arange(M.shape[1]), cols)
    assert np.abs(M[rows][:, others]).sum() == 0, "other parameters drive finger joints"
    joints = sorted(set(int(r) // 7 for r in rows))
    jn = list(ch.skeleton.joint_names)
    fmap = np.concatenate([M[j * 7 + 3:j * 7 + 6][:, cols] for j in joints], 0)
    return {"params": [f"{s}_{i}" for s in "lr" for i in range(27)], "joints": [jn[j] for j in joints], "map": fmap}


def export(ckpt_path, out):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    merger = load_merger(ck).eval()
    if not merger.root:
        raise ValueError("live bundles need a root merger (mg3+: legs, hip position, up)")
    m1_path = Path(ck.get("config", {}).get("m1", M1))
    body = load_lifter(torch.load(m1_path, map_location="cpu", weights_only=False)).eval()
    assert body.root and body.n_out == len(OUT_JOINTS)
    ch = load()
    fk = HandFK(ch)
    pair = LearnedPair(P3, fk)
    out.mkdir(parents=True, exist_ok=True)

    x = torch.zeros(1, len(BODY_OFFSETS), FRAME_DIM + 2)
    names = ["rot", "log_scale", "root", "root_log_s", "up"]
    torch.onnx.export(_Flat(body), (x,), str(out / "body.onnx"), dynamo=False, opset_version=17, input_names=["x"],
                      output_names=names, dynamic_axes={k: {0: "n"} for k in ["x"] + names})
    h = torch.zeros(2, HISTORY, pair.model.frame_dim)
    torch.onnx.export(_HandsFlat(pair.model), (h, h), str(out / "hands.onnx"), dynamo=False, opset_version=17,
                      input_names=["own", "other"], output_names=_HandsFlat.NAMES,
                      dynamic_axes={k: {0: "n"} for k in ["own", "other"] + _HandsFlat.NAMES})
    xm = torch.zeros(1, len(OFFSETS), frame_dim(merger.nj, merger.root))
    torch.onnx.export(_MergerFlat(merger), (xm,), str(out / "merger.onnx"), dynamo=False, opset_version=17,
                      input_names=["x"], output_names=_MergerFlat.NAMES,
                      dynamic_axes={k: {0: "n"} for k in ["x"] + _MergerFlat.NAMES})

    pn, jn = list(ch.parameter_transform.names), list(ch.skeleton.joint_names)
    rest = skeleton_state(ch, np.zeros(len(pn)))
    avatar = load_vrm(AVATARS / f"{DEFAULT_AVATAR}.vrm")[2]
    pose = LifterPose(OUT_JOINTS, PARENT_IDX, jn, ch.skeleton.joint_parents, rest, avatar, fingers=finger_table(ch, fk))
    P0 = positions(ch, np.zeros((1, len(pn)), np.float32))[0]
    F_rest = {s: palm_frame(P0[fk.mp21[s]])[1].astype(np.float32).astype(np.float64).tolist() for s in "lr"}
    rest_head = quat_to_mat(torch.tensor(rest[jn.index("c_head"), 3:7], dtype=torch.float32)).numpy().astype(np.float64)
    canon = CanonicalFace()
    fc = json.load(open(MERGER / "calib" / "face.json"))
    batch.blend_bank()
    fingers = [b for b in MAP if any(w in b for w in FINGER_WORDS)]
    meta = {
        "format": "motiontracker-merger", "version": 1,
        "source": {"merger": Path(ckpt_path).parent.name, "checkpoint": Path(ckpt_path).name, "step": ck.get("step"),
                   "body": m1_path.parent.name, "hands": Path(P3).parent.name},
        "input": {"offsets": BODY_OFFSETS.tolist(), "frame_dim": FRAME_DIM, "head3d_dims": HEAD3D_DIMS,
                  "scale_window": SCALE_WINDOW, "default_scale": DEFAULT_SCALE},
        "output_bones": OUTPUT_BONES + LEG_BONES + fingers,
        "skeleton": pose.table(),
        "default_avatar": {"name": DEFAULT_AVATAR, "rest_positions": {b: np.asarray(p).tolist() for b, p in avatar.items()}},
        "hands": {"history": HISTORY, "frame_dim": HAND_DIM, "F_rest": F_rest, "out_idx": list(OUT_IDX),
                  "lo": fk.lo.tolist(), "hi": fk.hi.tolist()},
        "combine": {"conf_deg": list(CONF_DEG), "max_bend_deg": MAX_BEND_DEG, "head_weight": HEAD_WEIGHT},
        "face": {"points": canon.points.tolist(), "ids": canon.ids.tolist(), "weights": canon.weights.tolist(),
                 "offset": fc["offset"], "rest_head": rest_head.tolist()},
        "blendshapes": list(batch.ARKIT),
        "merger": {"offsets": list(OFFSETS), "joints": merger.nj},
    }
    (out / "merger.json").write_text(json.dumps(meta))
    return meta


def as_obs(row):
    """Recorder-format frame (lists) -> Detector.detect form (arrays)."""
    return {k: (v if k == "blendshapes" or not isinstance(v, list) else np.asarray(v, dtype=np.float64)) for k, v in row.items()}


def angle_deg(qa, qb):
    d = np.abs((np.asarray(qa) * np.asarray(qb)).sum(-1))
    return np.degrees(2 * np.arccos(np.clip(d, -1, 1)))


def check(ckpt, bundle, landmarks, avatar=DEFAULT_AVATAR):
    from mtcore.merged import LiveMerged
    from .run import finger_locals, infer
    rows = [json.loads(line) for line in open(landmarks) if line.strip()]
    r = infer(ckpt, rows, avatar)
    conv = r["runner"].pose.conv
    ref = r["runner"].vrm(r["L"])
    ref.update(finger_locals(conv, r["fk"], r["ch"], r["fingers"]))
    live = LiveMerged(bundle)
    live.set_avatar_vrm(AVATARS / f"{avatar}.vrm")
    worst, root_d, up_d = {}, 0.0, 0.0
    for i, row in enumerate(rows):
        bones, _, _, root, up = live.step(row["t"] / 1000.0, as_obs(row))
        for b, q in bones.items():
            if b in ref:
                worst[b] = max(worst.get(b, 0.0), float(angle_deg(q, ref[b][i])))
        root_d = max(root_d, float(np.abs(root - r["root"][i]).max()))
        up_d = max(up_d, float(np.degrees(np.arccos(np.clip(up @ r["up"][i], -1, 1)))))
    return len(rows), worst, root_d, up_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", type=Path, help="landmarks JSONL (recorder / runtime --dump-landmarks format)")
    a = ap.parse_args()
    out = a.out or a.ckpt.parent / "live"
    meta = export(a.ckpt, out)
    print(f"bundle -> {out} (merger {meta['source']['merger']}, body {meta['source']['body']}, hands {meta['source']['hands']})")
    if a.check:
        n, worst, root_d, up_d = check(a.ckpt, out, a.check)
        top = sorted(worst.items(), key=lambda kv: -kv[1])[:6]
        print(f"bundle vs torch on {n} frames: max bone angle {max(worst.values()):.4f} deg "
              f"({', '.join(f'{b} {v:.4f}' for b, v in top)}); root {root_d * 100:.4f} cm; up {up_d:.4f} deg")


if __name__ == "__main__":
    main()
