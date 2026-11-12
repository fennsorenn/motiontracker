"""Export a trained M1 lifter as a self-contained bundle for the live runtime (service/).

    uv run python -m mttrain.export data/m1/runs/<name>/best.pt [--out <dir>] [--check-session sessions/<s>]

Writes <dir> (default: data/m1/runs/<name>/live/):
- model.onnx: the lifter; input x (N, O, FRAME_DIM + 2) -> rot (N, J, 3, 3), log_scale (N, J).
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
from .model import Lifter
from .skeleton import OUT_JOINTS, PARENT_IDX

DEFAULT_AVATAR = "AvatarSample_D"  # VRoid CC0 sample; all VRoid samples share the same arm rest directions


def export(ckpt_path, out):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Lifter()
    model.load_state_dict(ck["model"])
    model.eval()
    out.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, len(OFFSETS), FRAME_DIM + 2)
    torch.onnx.export(model, (x,), str(out / "model.onnx"), dynamo=False, opset_version=17,
                      input_names=["x"], output_names=["rot", "log_scale"],
                      dynamic_axes={"x": {0: "n"}, "rot": {0: "n"}, "log_scale": {0: "n"}})
    ch = load()
    rest = skeleton_state(ch, np.zeros(len(ch.parameter_transform.names)))
    avatar = load_vrm(AVATARS / f"{DEFAULT_AVATAR}.vrm")[2]
    pose = LifterPose(OUT_JOINTS, PARENT_IDX, ch.skeleton.joint_names, ch.skeleton.joint_parents, rest, avatar)
    meta = {
        "format": "motiontracker-lifter", "version": 1,
        "source": {"run": Path(ckpt_path).parent.name, "checkpoint": Path(ckpt_path).name, "step": ck.get("step")},
        "input": {"offsets": OFFSETS.tolist(), "frame_dim": FRAME_DIM, "head3d_dims": HEAD3D_DIMS,
                  "scale_window": SCALE_WINDOW, "default_scale": DEFAULT_SCALE,
                  "layout": "per offset: world/scale (33x3), image x-0.5, y-0.5, z (33x3), visibility (33), present, age, missing"},
        "output_bones": OUTPUT_BONES,
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
        rot_t, ls_t = (v.numpy() for v in model(torch.from_numpy(x)))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rot_o, ls_o = sess.run(None, {"x": x})
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
