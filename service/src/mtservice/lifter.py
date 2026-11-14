"""The M1 lifter from an exported bundle (mttrain.export): one frame of observations in, VRM bones out."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from mtcore.features import FRAME_DIM, OFFSETS, StreamFeatures
from mtcore.lifter_pose import LifterPose
from mtcore.vrm import load_vrm


class LiveLifter:
    def __init__(self, bundle, head3d=True, threads=1):
        bundle = Path(bundle)
        self.meta = json.loads((bundle / "lifter.json").read_text())
        inp = self.meta["input"]
        if inp["offsets"] != OFFSETS.tolist() or inp["frame_dim"] != FRAME_DIM:
            raise ValueError(f"bundle input layout {inp['offsets']}/{inp['frame_dim']} does not match mtcore.features")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(bundle / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.bones = self.meta["output_bones"]
        self.joints = self.meta["skeleton"]["out_joints"]
        self.default_rest = {b: np.asarray(p) for b, p in self.meta["default_avatar"]["rest_positions"].items()}
        self.rest = self.default_rest
        self.pose = LifterPose.from_table(self.meta["skeleton"], self.rest, bones=self.bones)
        self.features = StreamFeatures(head3d)

    def set_avatar(self, rest_positions=None):
        """Avatar rest bone positions (normalized VRM, bone -> xyz); None = the bundle's default avatar."""
        self.rest = self.default_rest if rest_positions is None else {b: np.asarray(p, dtype=float) for b, p in rest_positions.items()}
        self.pose.set_avatar(self.rest)

    def set_avatar_vrm(self, path):
        """The avatar's rest pose read from its .vrm file (0.x or 1.0), as mttrain.export reads the default."""
        self.set_avatar(load_vrm(path)[2])

    def reset(self):
        self.features = StreamFeatures(self.features.head3d)

    def step(self, t, obs):
        """t: seconds; obs: Detector.detect output. Returns (bones: VRM bone -> quaternion xyzw,
        confidence: lifter joint -> 0..1, present)."""
        present = "pose" in obs and "pose2d" in obs
        if present:
            world, vis, image = obs["pose"][:, :3], obs["pose"][:, 3], obs["pose2d"][:, :3]
        else:
            world, vis, image = np.zeros((33, 3)), np.zeros(33), np.zeros((33, 3))
        x = self.features.push(t, world, image, vis, present)
        rot, log_scale = self.session.run(None, {"x": x[None]})
        locals_ = self.pose.vrm(rot.astype(np.float64))
        bones = {b: locals_[b][0] for b in self.bones if b in locals_}
        conf = dict(zip(self.joints, np.exp(-log_scale[0]).tolist()))
        return bones, conf, present
