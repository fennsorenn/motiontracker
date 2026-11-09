"""What the M1 model predicts, and how targets come out of MHR skeleton states.

Output: 17 rotations, each relative to MHR's rest pose:
- `root` (pelvis): global orientation in the camera frame (x right, y down, z forward), since the
  model only ever sees the body through the camera;
- every other joint: local to its parent in OUT_PARENT (the MHR chain; procedural twist joints are
  leaves and not predicted).

Rotations are 3x3 matrices here (numpy); the model uses 6D (first two columns).
"""

import numpy as np

from mtcore.landmarks import MP33

OUT_JOINTS = ["root", "c_spine0", "c_spine1", "c_spine2", "c_spine3", "c_neck", "c_head"]
for _s in "lr":
    OUT_JOINTS += [f"{_s}_clavicle", f"{_s}_uparm", f"{_s}_lowarm", f"{_s}_wrist_twist", f"{_s}_wrist"]
OUT_PARENT = {"c_spine0": "root", "c_spine1": "c_spine0", "c_spine2": "c_spine1", "c_spine3": "c_spine2",
              "c_neck": "c_spine3", "c_head": "c_neck"}
for _s in "lr":
    OUT_PARENT.update({f"{_s}_clavicle": "c_spine3", f"{_s}_uparm": f"{_s}_clavicle",
                       f"{_s}_lowarm": f"{_s}_uparm", f"{_s}_wrist_twist": f"{_s}_lowarm",
                       f"{_s}_wrist": f"{_s}_wrist_twist"})
PARENT_IDX = [OUT_JOINTS.index(OUT_PARENT[j]) if j in OUT_PARENT else -1 for j in OUT_JOINTS]
# joints whose positions enter the position loss / metrics (end of the chain matters most)
POS_JOINTS = ["c_head", "l_uparm", "r_uparm", "l_lowarm", "r_lowarm", "l_wrist", "r_wrist"]
GROUPS = {"spine": ["root", "c_spine0", "c_spine1", "c_spine2", "c_spine3"], "head": ["c_neck", "c_head"],
          "shoulder": ["l_clavicle", "r_clavicle"], "upper_arm": ["l_uparm", "r_uparm"],
          "forearm": ["l_lowarm", "r_lowarm", "l_wrist_twist", "r_wrist_twist"], "wrist": ["l_wrist", "r_wrist"]}

# contact points (contact loss): MediaPipe landmarks carried rigidly by an output joint, as offsets in
# that joint's rest-relative frame. Hands: the middle of the index and pinky knuckles (palm front).
CONTACT_POINTS = [("l_hand", "l_wrist", ("left_index", "left_pinky")), ("r_hand", "r_wrist", ("right_index", "right_pinky")),
                  ("nose", "c_head", ("nose",))]
CONTACT_PARENT = [OUT_JOINTS.index(p) for _, p, _ in CONTACT_POINTS]
CONTACT_LM = [[MP33.index(n) for n in lms] for _, _, lms in CONTACT_POINTS]
CONTACT_PAIRS = [(0, 1), (0, 2), (1, 2)]  # hand-hand, left hand-nose, right hand-nose


def quat_to_mat(q):
    """Quaternions (..., 4) xyzw -> rotation matrices (..., 3, 3)."""
    x, y, z, w = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


class TargetExtractor:
    def __init__(self, joint_names, rest_state):
        self.idx = [joint_names.index(j) for j in OUT_JOINTS]
        self.rest_g = quat_to_mat(rest_state[self.idx, 3:7])  # (17, 3, 3) rest globals (MHR world)

    def __call__(self, states, cam_R):
        """states (T, J, 8) MHR skeleton states (cm), cam_R (3, 3) world -> camera.

        Returns rot (T, 17, 3, 3) rest-relative rotations as defined above, pos (T, 17, 3) joint
        positions in camera axes relative to the root, metres, and offsets (T, 17, 3): each joint's
        position in its parent's rest-relative frame (bone vectors, for FK in the loss)."""
        G = quat_to_mat(states[:, self.idx, 3:7])  # (T, 17, 3, 3) world
        Gc = cam_R @ G @ np.swapaxes(self.rest_g, -1, -2)  # rest-relative, camera frame
        rot = np.empty_like(Gc)
        for k, p in enumerate(PARENT_IDX):
            rot[:, k] = Gc[:, k] if p < 0 else np.swapaxes(Gc[:, p], -1, -2) @ Gc[:, k]
        X = (states[:, self.idx, :3] - states[:, self.idx[:1], :3]) / 100.0
        pos = X @ cam_R.T
        off = np.zeros_like(pos)
        for k, p in enumerate(PARENT_IDX):
            if p >= 0:
                off[:, k] = np.einsum("tji,tj->ti", Gc[:, p], pos[:, k] - pos[:, p])
        return rot, pos, off

    def contact_offsets(self, states, cam_R, landmarks):
        """Contact points' offsets (T, 3, 3) in their parent joint's rest-relative frame (camera axes,
        metres), from the true body's landmarks (T, 33, 3) cm."""
        G = cam_R @ quat_to_mat(states[:, self.idx, 3:7]) @ np.swapaxes(self.rest_g, -1, -2)
        root = states[:, self.idx[0], :3]
        out = np.zeros((len(states), len(CONTACT_POINTS), 3))
        for c, (p, lms) in enumerate(zip(CONTACT_PARENT, CONTACT_LM)):
            d = (landmarks[:, lms].mean(1) - states[:, self.idx[p], :3]) / 100.0 @ cam_R.T
            out[:, c] = np.einsum("tji,tj->ti", G[:, p], d)
        return out


def contact_points(G, P, cof):
    """Contact points (..., 3, 3) from FK's global rotations and positions and their offsets."""
    return P[..., CONTACT_PARENT, :] + (G[..., CONTACT_PARENT, :, :] @ cof[..., None])[..., 0]


def fk(rot, off):
    """Rest-relative rotations (..., 17, 3, 3) and bone offsets (..., 17, 3) -> global rotations
    (camera frame) and root-relative positions. Works for numpy arrays and torch tensors."""
    G, P = [None] * len(OUT_JOINTS), [None] * len(OUT_JOINTS)
    for k, p in enumerate(PARENT_IDX):
        if p < 0:
            G[k], P[k] = rot[..., k, :, :], off[..., k, :] * 0
        else:
            G[k] = G[p] @ rot[..., k, :, :]
            P[k] = P[p] + (G[p] @ off[..., k, :, None])[..., 0]
    stack = (lambda xs, d: __import__("torch").stack(xs, d)) if not isinstance(rot, np.ndarray) else np.stack
    return stack(G, -3), stack(P, -2)
