"""Export a trained M1 lifter as a self-contained bundle for the live runtime (service/).

    uv run python -m mttrain.export data/m1/runs/<name>/best.pt [--out <dir>] [--check-session sessions/<s>]

Writes <dir> (default: data/m1/runs/<name>/live/):
- model.onnx: the lifter; input x (N, O, FRAME_DIM + 2) -> rot (N, J, 3, 3), log_scale (N, J); root models
  (m1-v12+, J = 23 with the legs) also root (N, 3) camera frame metres, root_log_s (N, 2), up (N, 3).
- lifter.json: input settings (offsets, frame layout), output bones, the MHR skeleton table
  (mtcore.lifter_pose.LifterPose.table) and a default avatar rest pose, so the runtime needs neither
  torch nor pymomentum nor the MHR assets.
With --check-session, runs the ONNX model (onnxruntime) and torch on that session's frames and reports
the largest difference. The bundle holds weights trained on CMU-derived data: it stays in gitignored
data/ (license register).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mtcore.features import FRAME_DIM, HEAD3D_DIMS, OFFSETS, SCALE_WINDOW, DEFAULT_SCALE, frame_features, history_indices, model_input
from mtcore.lifter_pose import LifterPose
from mtcore.mhr import load, skeleton_state
from mtcore.paths import AVATARS
from mtcore.vrm import load_vrm

from .infer import OUTPUT_BONES, read_session
from .model import load_lifter
from .skeleton import N_UPPER, OUT_JOINTS, PARENT_IDX

OUTPUT_JOINTS_UPPER = OUT_JOINTS[:N_UPPER]
LEG_BONES = [f"{s}{b}" for s in ("left", "right") for b in ("UpperLeg", "LowerLeg", "Foot")]

DEFAULT_AVATAR = "AvatarSample_D"  # VRoid CC0 sample; all VRoid samples share the same arm rest directions


class _Flat(torch.nn.Module):
    """Root models return (rot, log_scale, aux dict): flattened to a tuple for ONNX."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if len(out) == 2:
            return out
        rot, ls, aux = out
        return rot, ls, aux["root"], aux["root_log_s"], aux["up"]


def export(ckpt_path, out):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = load_lifter(ck)
    model.eval()
    out.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, len(OFFSETS), FRAME_DIM + 2)
    names = ["rot", "log_scale"] + (["root", "root_log_s", "up"] if model.root else [])
    torch.onnx.export(_Flat(model), (x,), str(out / "model.onnx"), dynamo=False, opset_version=17,
                      input_names=["x"], output_names=names, dynamic_axes={k: {0: "n"} for k in ["x"] + names})
    ch = load()
    rest = skeleton_state(ch, np.zeros(len(ch.parameter_transform.names)))
    avatar = load_vrm(AVATARS / f"{DEFAULT_AVATAR}.vrm")[2]
    n = model.n_out  # 17 (m1-v10/v11) or 23 (m1-v12+: legs)
    pose = LifterPose(OUT_JOINTS[:n], PARENT_IDX[:n], ch.skeleton.joint_names, ch.skeleton.joint_parents, rest, avatar)
    meta = {
        "format": "motiontracker-lifter", "version": 1,
        "source": {"run": Path(ckpt_path).parent.name, "checkpoint": Path(ckpt_path).name, "step": ck.get("step")},
        "input": {"offsets": OFFSETS.tolist(), "frame_dim": FRAME_DIM, "head3d_dims": HEAD3D_DIMS,
                  "scale_window": SCALE_WINDOW, "default_scale": DEFAULT_SCALE,
                  "layout": "per offset: world/scale (33x3), image x-0.5, y-0.5, z (33x3), visibility (33), present, age, missing"},
        "output_bones": OUTPUT_BONES + (LEG_BONES if n > len(OUTPUT_JOINTS_UPPER) else []),
        "root": bool(model.root),
        "skeleton": pose.table(),
        "default_avatar": {"name": DEFAULT_AVATAR, "rest_positions": {b: np.asarray(p).tolist() for b, p in avatar.items()}},
    }
    (out / "lifter.json").write_text(json.dumps(meta))
    return model


def check(model, onnx_path, session):
    import onnxruntime as ort
    times, world, image, vis, present = read_session(session)
    feats = frame_features(times, world, image, vis, present)
    idx, age = history_indices(times, np.arange(len(times)))
    x = model_input(feats, idx, age)
    with torch.no_grad():
        rot_t, ls_t = (v.numpy() for v in model(torch.from_numpy(x))[:2])
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rot_o, ls_o = sess.run(None, {"x": x})[:2]
    return len(x), float(np.abs(rot_o - rot_t).max()), float(np.abs(ls_o - ls_t).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check-session", type=Path)
    a = ap.parse_args()
    out = a.out or a.ckpt.parent / "live"
    model = export(a.ckpt, out)
    print(f"bundle -> {out}")
    if a.check_session:
        n, dr, dl = check(model, out / "model.onnx", a.check_session)
        print(f"onnxruntime vs torch on {n} frames: rot max abs diff {dr:.2e}, log_scale {dl:.2e}")


if __name__ == "__main__":
    main()
