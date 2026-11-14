"""MediaPipe Pose's 33 landmarks from MHR skeleton states.

Body landmarks sit on MHR joints. Face and foot landmarks are mesh vertices, picked by geometry on
MHR's rest mesh and carried rigidly by the joint that skins them most (84-100% weight for the
picked vertices: head, jaw, foot). Output is in MHR world space (cm, y up, +x = character's left,
+z = character's front); camera projection and MediaPipe's axis conventions are applied elsewhere.
"""

import numpy as np

# MediaPipe Pose landmark order; "left"/"right" are the person's own sides.
MP33 = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]

_JOINTS = {"shoulder": "uparm", "elbow": "lowarm", "wrist": "wrist", "pinky": "pinky1", "index": "index1",
           "thumb": "thumb2", "hip": "upleg", "knee": "lowleg", "ankle": "talocrural"}


def _vertex_targets():
    """Landmark -> (search point in MHR rest space, cm, for the character's left side; mode).

    mode "front": front-most vertex within 0.6 cm of (x, y) in the frontal plane;
    mode "near": nearest vertex to (x, y, z)."""
    t = {
        "nose": ((0.0, 159.1), "front"),
        "left_eye_inner": ((1.8, 162.5), "front"),
        "left_eye": ((3.2, 162.5), "front"),
        "left_eye_outer": ((4.8, 162.6), "front"),
        "mouth_left": ((2.4, 155.7), "front"),
        "left_ear": ((8.9, 161.7, 1.4), "near"),
        "left_heel": ((18.3, 2.2, -12.1), "near"),
        "left_foot_index": ((17.8, 1.5, 12.9), "near"),
    }
    out = {}
    for name, (p, mode) in t.items():
        out[name] = (p, mode)
        if name.startswith("left_") or name.endswith("_left"):
            mirrored = (-p[0],) + tuple(p[1:])
            out[name.replace("left", "right")] = (mirrored, mode)
    return out


def _pick_vertex(V, p, mode):
    if mode == "front":
        m = (np.abs(V[:, 0] - p[0]) < 0.6) & (np.abs(V[:, 1] - p[1]) < 0.6)
        cand = np.flatnonzero(m)
        return int(cand[V[cand, 2].argmax()])
    return int(np.linalg.norm(V - np.asarray(p), axis=1).argmin())


def _quat_rotate(q, v):
    """Rotate v (..., 3) by unit quaternions q (..., 4) in xyzw order."""
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _quat_inv(q):
    return np.concatenate([-q[..., :3], q[..., 3:]], -1)


class LandmarkModel:
    """Maps MHR skeleton states (..., J, 8) to MediaPipe's 33 pose landmarks (..., 33, 3)."""

    def __init__(self, ch, rest_state):
        jn = list(ch.skeleton.joint_names)
        V = np.asarray(ch.mesh.vertices)
        idx = np.asarray(ch.skin_weights.index)
        wts = np.asarray(ch.skin_weights.weight)
        targets = _vertex_targets()
        self.joint = np.zeros(33, int)
        self.offset = np.zeros((33, 3))  # in the joint's local frame, before joint scale
        self.vertex = {}
        for k, name in enumerate(MP33):
            if name in targets:
                v = _pick_vertex(V, *targets[name])
                j = int(idx[v, wts[v].argmax()])
                t, q, s = rest_state[j, :3], rest_state[j, 3:7], rest_state[j, 7]
                self.joint[k] = j
                self.offset[k] = _quat_rotate(_quat_inv(q), V[v] - t) / s
                self.vertex[name] = v
            else:
                side, part = name.split("_", 1)
                self.joint[k] = jn.index(f"{side[0]}_{_JOINTS[part]}")

    def __call__(self, states):
        st = states[..., self.joint, :]
        return st[..., :3] + st[..., 7:8] * _quat_rotate(st[..., 3:7], np.broadcast_to(self.offset, st[..., :3].shape))
