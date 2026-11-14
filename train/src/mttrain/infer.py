"""Run a trained M1 lifter on a recorded session -> VRM pose track for the evaluation harness.

    uv run python -m mttrain.infer data/m1/runs/<name>/best.pt sessions/<session> [--out poses.jsonl]
        [--avatar AvatarSample_D] [--no-head3d]

Causal: every frame sees only itself and earlier frames (features are causal; batching all frames at
once gives the same result as streaming). Output lines match the baseline's pose tracks:
{"t": ms, "bones": {vrm bone: [x, y, z, w]}}, plus "conf": per-joint confidence. Evaluate with
`mteval <session> --system m1=<out>`.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mtcore.features import frame_features, history_indices, model_input
from mtcore.lifter_pose import LifterPose
from mtcore.mhr import load, skeleton_state
from mtcore.paths import AVATARS
from mtcore.vrm import load_vrm

from .model import load_lifter, pad_joints
from .skeleton import OUT_JOINTS, PARENT_IDX

# VRM bones the model drives; legs, feet, eyes and fingers would only be the rest pose
OUTPUT_BONES = ["hips", "spine", "chest", "upperChest", "neck", "head", "leftShoulder", "rightShoulder",
                "leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm", "leftHand", "rightHand"]


def read_session(path):
    frames = [json.loads(line) for line in (Path(path) / "landmarks.jsonl").read_text().splitlines() if line.strip()]
    T = len(frames)
    times = np.array([f["t"] for f in frames]) / 1000.0
    world, image, vis = np.zeros((T, 33, 3)), np.zeros((T, 33, 3)), np.zeros((T, 33))
    present = np.zeros(T, bool)
    for i, f in enumerate(frames):
        if f.get("pose") and f.get("pose2d"):
            p = np.asarray(f["pose"])
            world[i], vis[i], image[i] = p[:, :3], p[:, 3], np.asarray(f["pose2d"])[:, :3]
            present[i] = True
    return times, world, image, vis, present


class Runner:
    def __init__(self, ckpt_path, avatar="AvatarSample_D"):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = load_lifter(ck)
        self.model.eval()
        self.aux = None  # root position / up of the last predict (m1-v12+), else None
        self.ch = load()
        rest = skeleton_state(self.ch, np.zeros(len(self.ch.parameter_transform.names)))
        self.pose = LifterPose(OUT_JOINTS, PARENT_IDX, self.ch.skeleton.joint_names, self.ch.skeleton.joint_parents,
                               rest, load_vrm(AVATARS / f"{avatar}.vrm")[2])
        self.jn = self.pose.jn

    def predict(self, times, world, image, vis, present, head3d=True):
        feats = frame_features(times, world, image, vis, present, head3d)
        idx, age = history_indices(times, np.arange(len(times)))
        x = torch.from_numpy(model_input(feats, idx, age))
        with torch.no_grad():
            out = self.model(x)
        rot, ls = pad_joints(*out[:2])
        self.aux = {k: v.numpy().astype(np.float64) for k, v in out[2].items()} if len(out) > 2 else None
        return rot.numpy().astype(np.float64), torch.exp(-ls).numpy()

    def mhr_globals(self, rot):
        return self.pose.mhr_globals(rot)

    def vrm(self, rot):
        return self.pose.vrm(rot)

    def step_cost_ms(self, x, n=300):
        """Single-frame latency on one CPU thread (the live use case)."""
        torch.set_num_threads(1)
        xi = torch.from_numpy(x[:1])
        with torch.no_grad():
            for _ in range(20):
                self.model(xi)
            t = time.perf_counter()
            for _ in range(n):
                self.model(xi)
        return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("session", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--avatar", default="AvatarSample_D")
    ap.add_argument("--no-head3d", action="store_true",
                    help="drop the 3D head points from the input (models trained with head 3D dropout, m1-v9+)")
    a = ap.parse_args()
    r = Runner(a.ckpt, a.avatar)
    times, world, image, vis, present = read_session(a.session)
    rot, conf = r.predict(times, world, image, vis, present, head3d=not a.no_head3d)
    bones = r.vrm(rot)
    out = a.out or a.ckpt.parent / f"{a.session.name}.poses.jsonl"
    with open(out, "w") as f:
        for i, t in enumerate(times):
            f.write(json.dumps({"t": round(t * 1000, 1),
                                "bones": {b: bones[b][i].round(6).tolist() for b in OUTPUT_BONES if b in bones},
                                "conf": dict(zip(OUT_JOINTS, conf[i].round(3).tolist()))}) + "\n")
    feats = frame_features(times, world, image, vis, present, not a.no_head3d)
    idx, age = history_indices(times, np.arange(len(times)))
    print(f"{len(times)} frames -> {out}; model cost {r.step_cost_ms(model_input(feats, idx, age)):.2f} ms/frame (1 CPU thread)")


if __name__ == "__main__":
    main()
