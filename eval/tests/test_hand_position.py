import json
from pathlib import Path

import numpy as np

from mteval.metrics import hand_position, with_head_global, HEAD_GLOBAL
from mteval.session import PoseTrack, load_session


def _session(tmp_path: Path, n=20, segments=None) -> Path:
    pose = [[0.0, 0.0, 0.0, 1.0]] * 33
    pose = [list(p) for p in pose]
    # MediaPipe world axes (x right, y down, z away); arms straight out to the sides, T-pose
    for (sh, el, wr), x in (((11, 13, 15), 1.0), ((12, 14, 16), -1.0)):
        pose[sh] = [0.2 * x, -0.4, 0.0, 1.0]
        pose[el] = [0.45 * x, -0.4, 0.0, 1.0]
        pose[wr] = [0.7 * x, -0.4, 0.0, 1.0]
    # image landmarks: same geometry, 1 image width per metre (320x240: y in units of height)
    pose2d = [[0.5 + p[0], 0.5 + p[1] / 0.75, 0.0, 1.0] for p in pose]
    lines = [json.dumps({"t": 100 * i, "pose": pose, "pose2d": pose2d}) for i in range(n)]
    (tmp_path / "landmarks.jsonl").write_text("\n".join(lines) + "\n")
    meta = {"segments": segments or [], "processing": {"width": 320, "height": 240}}
    (tmp_path / "session.json").write_text(json.dumps(meta))
    return tmp_path


def test_t_pose_matches_and_bent_elbow_does_not(tmp_path):
    s = load_session(_session(tmp_path))
    t = s.times
    ident = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1))
    bones = {b: ident.copy() for b in ("leftUpperArm", "leftLowerArm", "rightUpperArm", "rightLowerArm")}
    hp = hand_position(PoseTrack(t, bones), s)
    assert hp["left"]["in_view"]["median"] < 1e-6 and hp["right"]["in_view"]["median"] < 1e-6
    assert hp["left"]["in_view_2d"]["median"] < 1e-4 and hp["left"]["forearm_dir_2d"]["median"] < 1e-3
    assert hp["left"]["arm_lengths_cm"] == [25.0, 25.0]
    h = np.sin(np.pi / 4)
    bones["leftLowerArm"] = np.tile([0.0, 0.0, h, h], (len(t), 1))  # elbow bent 90 degrees
    hp = hand_position(PoseTrack(t, bones), s)
    assert abs(hp["left"]["in_view"]["median"] - 25 * np.sqrt(2)) < 1e-6
    assert abs(hp["left"]["forearm_dir_2d"]["median"] - 90) < 1e-3  # bend is in the image plane
    assert abs(hp["left"]["in_view_2d"]["median"] - 25 * np.sqrt(2)) < 1e-3
    assert hp["right"]["in_view"]["median"] < 1e-6


def test_head_global_composes_chain():
    t = np.arange(3) * 100.0
    h = np.sin(np.radians(10) / 2), np.cos(np.radians(10) / 2)
    q = np.tile([0.0, h[0], 0.0, h[1]], (3, 1))
    tr = with_head_global(PoseTrack(t, {"neck": q, "head": q}))
    ang = np.degrees(2 * np.arccos(np.clip(np.abs(tr.bones[HEAD_GLOBAL][:, 3]), 0, 1)))
    np.testing.assert_allclose(ang, 20.0, atol=1e-6)


def test_contact_distances(tmp_path):
    from mteval.metrics import contact
    segs = [{"label": "contact", "variant": "palms-together", "t0": 0, "t1": 3000}]
    s = load_session(_session(tmp_path, n=30, segments=segs))
    t = s.times
    ident = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1))
    bones = {b: ident.copy() for b in ("leftUpperArm", "leftLowerArm", "rightUpperArm", "rightLowerArm")}
    c = contact(PoseTrack(t, bones), s)
    # T-pose: wrists 1.4 m apart, both in the system and in MediaPipe's landmarks
    assert abs(c["palms-together"]["system_cm"] - 140) < 1e-3
    assert abs(c["palms-together"]["mediapipe_cm"] - 140) < 1e-3
    assert c["palms-together"]["n"] == 15  # frames after the settle time


def test_ground_truth_metrics_zero_for_identical_tracks():
    from mteval.metrics import ground_truth_metrics
    t = np.arange(50) * 33.3
    rng = np.random.default_rng(0)
    bones = {}
    for b in ("hips", "spine", "chest", "neck", "head", "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
              "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand"):
        q = rng.normal(0, 0.2, (50, 4)); q[:, 3] = 1.0
        bones[b] = q / np.linalg.norm(q, axis=1, keepdims=True)
    tr = PoseTrack(t, bones)
    g = ground_truth_metrics(tr, tr)
    assert g["global_rotation_error"]["global:leftLowerArm"]["median"] < 1e-4
    assert g["wrist_error"]["left"]["median"] < 1e-6 and g["wrist_error"]["right"]["median"] < 1e-6
    # the baseline-style split (no hips/upperChest, rotation folded into spine) gives the same globals
    b2 = {k: v for k, v in bones.items() if k != "hips"}
    from mteval.metrics import _qmul
    b2["spine"] = _qmul(bones["hips"], bones["spine"])
    g2 = ground_truth_metrics(PoseTrack(t, b2), tr)
    assert g2["global_rotation_error"]["global:rightHand"]["median"] < 1e-4
