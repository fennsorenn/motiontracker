import json
from pathlib import Path

import numpy as np

from mteval.metrics import hand_position, with_head_global, HEAD_GLOBAL
from mteval.session import PoseTrack, load_session


def _session(tmp_path: Path, n=20) -> Path:
    pose = [[0.0, 0.0, 0.0, 1.0]] * 33
    pose = [list(p) for p in pose]
    # MediaPipe world axes (x right, y down, z away); arms straight out to the sides, T-pose
    for (sh, el, wr), x in (((11, 13, 15), 1.0), ((12, 14, 16), -1.0)):
        pose[sh] = [0.2 * x, -0.4, 0.0, 1.0]
        pose[el] = [0.45 * x, -0.4, 0.0, 1.0]
        pose[wr] = [0.7 * x, -0.4, 0.0, 1.0]
    lines = [json.dumps({"t": 100 * i, "pose": pose}) for i in range(n)]
    (tmp_path / "landmarks.jsonl").write_text("\n".join(lines) + "\n")
    (tmp_path / "session.json").write_text(json.dumps({"segments": []}))
    return tmp_path


def test_t_pose_matches_and_bent_elbow_does_not(tmp_path):
    s = load_session(_session(tmp_path))
    t = s.times
    ident = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1))
    bones = {b: ident.copy() for b in ("leftUpperArm", "leftLowerArm", "rightUpperArm", "rightLowerArm")}
    hp = hand_position(PoseTrack(t, bones), s)
    assert hp["left"]["in_view"]["median"] < 1e-6 and hp["right"]["in_view"]["median"] < 1e-6
    assert hp["left"]["arm_lengths_cm"] == [25.0, 25.0]
    h = np.sin(np.pi / 4)
    bones["leftLowerArm"] = np.tile([0.0, 0.0, h, h], (len(t), 1))  # elbow bent 90 degrees
    hp = hand_position(PoseTrack(t, bones), s)
    assert abs(hp["left"]["in_view"]["median"] - 25 * np.sqrt(2)) < 1e-6
    assert hp["right"]["in_view"]["median"] < 1e-6


def test_head_global_composes_chain():
    t = np.arange(3) * 100.0
    h = np.sin(np.radians(10) / 2), np.cos(np.radians(10) / 2)
    q = np.tile([0.0, h[0], 0.0, h[1]], (3, 1))
    tr = with_head_global(PoseTrack(t, {"neck": q, "head": q}))
    ang = np.degrees(2 * np.arccos(np.clip(np.abs(tr.bones[HEAD_GLOBAL][:, 3]), 0, 1)))
    np.testing.assert_allclose(ang, 20.0, atol=1e-6)
