"""Run a trained M1 lifter on a recorded session -> VRM pose track for the evaluation harness.

    uv run python -m mttrain.infer data/m1/runs/<name>/best.pt sessions/<session> [--out poses.jsonl]
        [--avatar AvatarSample_D]

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

from mtcore.convert import MhrToVrm
from mtcore.mhr import load, skeleton_state
from mtcore.paths import AVATARS
from mtcore.vrm import load_vrm

from .features import frame_features, history_indices, model_input
from .model import Lifter
from .skeleton import OUT_JOINTS, PARENT_IDX, quat_to_mat

# VRM bones the model drives; legs, feet, eyes and fingers would only be the rest pose
OUTPUT_BONES = ["hips", "spine", "chest", "upperChest", "neck", "head", "leftShoulder", "rightShoulder",
                "leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm", "leftHand", "rightHand"]
# camera axes (x right, y down, z forward) -> MHR world for a performer facing the camera
R0 = np.diag([1.0, -1.0, -1.0])


def mat_to_quat(R):
    """Rotation matrices (..., 3, 3) -> quaternions (..., 4) xyzw."""
    m = R
    w = np.sqrt(np.maximum(0, 1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2])) / 2
    x = np.sqrt(np.maximum(0, 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2])) / 2
    y = np.sqrt(np.maximum(0, 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2])) / 2
    z = np.sqrt(np.maximum(0, 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2])) / 2
    x = np.copysign(x, m[..., 2, 1] - m[..., 1, 2])
    y = np.copysign(y, m[..., 0, 2] - m[..., 2, 0])
    z = np.copysign(z, m[..., 1, 0] - m[..., 0, 1])
    q = np.stack([x, y, z, w], -1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


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
        self.model = Lifter()
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.ch = load()
        self.jn = list(self.ch.skeleton.joint_names)
        self.parents = list(self.ch.skeleton.joint_parents)
        rest = skeleton_state(self.ch, np.zeros(len(self.ch.parameter_transform.names)))
        self.rest_g = quat_to_mat(rest[:, 3:7])  # (J, 3, 3)
        self.out_idx = [self.jn.index(j) for j in OUT_JOINTS]
        self.conv = MhrToVrm(self.jn, rest, load_vrm(AVATARS / f"{avatar}.vrm")[2])

    def predict(self, times, world, image, vis, present):
        feats = frame_features(times, world, image, vis, present)
        idx, age = history_indices(times, np.arange(len(times)))
        x = torch.from_numpy(model_input(feats, idx, age))
        with torch.no_grad():
            rot, ls = self.model(x)
        return rot.numpy().astype(np.float64), torch.exp(-ls).numpy()

    def mhr_globals(self, rot):
        """Rest-relative predictions (T, 17, 3, 3) -> MHR world joint rotations (T, J, 3, 3); joints the
        model doesn't predict keep their rest pose relative to their parent."""
        T = len(rot)
        Gc = [None] * len(OUT_JOINTS)
        for k, p in enumerate(PARENT_IDX):
            Gc[k] = rot[:, k] if p < 0 else Gc[p] @ rot[:, k]
        G = np.empty((T, len(self.jn), 3, 3))
        pred = dict(zip(self.out_idx, range(len(OUT_JOINTS))))
        for j, p in enumerate(self.parents):
            if j in pred:
                G[:, j] = R0.T @ Gc[pred[j]] @ self.rest_g[j]
            elif p < 0:
                G[:, j] = self.rest_g[j]
            else:
                G[:, j] = G[:, p] @ (self.rest_g[p].T @ self.rest_g[j])
        return G

    def vrm(self, rot):
        state = np.zeros(rot.shape[:1] + (len(self.jn), 8))
        state[..., 3:7] = mat_to_quat(self.mhr_globals(rot))
        state[..., 7] = 1.0
        return self.conv.locals(state)

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
    a = ap.parse_args()
    r = Runner(a.ckpt, a.avatar)
    times, world, image, vis, present = read_session(a.session)
    rot, conf = r.predict(times, world, image, vis, present)
    bones = r.vrm(rot)
    out = a.out or a.ckpt.parent / f"{a.session.name}.poses.jsonl"
    with open(out, "w") as f:
        for i, t in enumerate(times):
            f.write(json.dumps({"t": round(t * 1000, 1),
                                "bones": {b: bones[b][i].round(6).tolist() for b in OUTPUT_BONES if b in bones},
                                "conf": dict(zip(OUT_JOINTS, conf[i].round(3).tolist()))}) + "\n")
    feats = frame_features(times, world, image, vis, present)
    idx, age = history_indices(times, np.arange(len(times)))
    print(f"{len(times)} frames -> {out}; model cost {r.step_cost_ms(model_input(feats, idx, age)):.2f} ms/frame (1 CPU thread)")


if __name__ == "__main__":
    main()
