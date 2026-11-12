"""M1 lifter output -> VRM 1.0 bone rotations, numpy only (shared by offline inference and the live runtime).

The lifter predicts rest-relative rotations for OUT_JOINTS in camera axes. LifterPose chains them into
MHR world joint rotations (joints the model doesn't predict keep their rest pose relative to their
parent) and converts those to VRM normalized local rotations with MhrToVrm for a given avatar rest pose.
Everything it needs about MHR is a small table (`table()` / `from_table()`), so the runtime does not
need pymomentum or the MHR assets.
"""

import numpy as np

from .convert import MhrToVrm

# camera axes (x right, y down, z forward) -> MHR world for a performer facing the camera
R0 = np.diag([1.0, -1.0, -1.0])


def quat_to_mat(q):
    """Quaternions (..., 4) xyzw -> rotation matrices (..., 3, 3)."""
    x, y, z, w = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


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


class LifterPose:
    def __init__(self, out_joints, out_parent_idx, joint_names, parents, rest_state, vrm_rest_positions):
        """out_joints / out_parent_idx: the lifter's joints and their parents (index into out_joints,
        -1 = none). joint_names, parents, rest_state (J, 8): MHR skeleton at zero pose.
        vrm_rest_positions: avatar bone -> rest position (normalized VRM)."""
        self.out_joints, self.out_parent_idx = list(out_joints), list(out_parent_idx)
        self.jn, self.parents = list(joint_names), list(parents)
        self.rest_state = np.asarray(rest_state, dtype=np.float64)
        self.rest_g = quat_to_mat(self.rest_state[:, 3:7])  # (J, 3, 3)
        self.out_idx = [self.jn.index(j) for j in self.out_joints]
        self.set_avatar(vrm_rest_positions)

    def set_avatar(self, vrm_rest_positions):
        self.conv = MhrToVrm(self.jn, self.rest_state, {b: np.asarray(p, dtype=np.float64) for b, p in vrm_rest_positions.items()})

    def table(self):
        """The MHR part as plain JSON-able data."""
        return {"out_joints": self.out_joints, "out_parent_idx": self.out_parent_idx, "joint_names": self.jn,
                "parents": self.parents, "rest_state": self.rest_state.tolist()}

    @classmethod
    def from_table(cls, t, vrm_rest_positions):
        return cls(t["out_joints"], t["out_parent_idx"], t["joint_names"], t["parents"], t["rest_state"], vrm_rest_positions)

    def mhr_globals(self, rot):
        """Rest-relative predictions (T, K, 3, 3) -> MHR world joint rotations (T, J, 3, 3)."""
        T = len(rot)
        Gc = [None] * len(self.out_joints)
        for k, p in enumerate(self.out_parent_idx):
            Gc[k] = rot[:, k] if p < 0 else Gc[p] @ rot[:, k]
        G = np.empty((T, len(self.jn), 3, 3))
        pred = dict(zip(self.out_idx, range(len(self.out_joints))))
        for j, p in enumerate(self.parents):
            if j in pred:
                G[:, j] = R0.T @ Gc[pred[j]] @ self.rest_g[j]
            elif p < 0:
                G[:, j] = self.rest_g[j]
            else:
                G[:, j] = G[:, p] @ (self.rest_g[p].T @ self.rest_g[j])
        return G

    def vrm(self, rot):
        """Predictions (T, K, 3, 3) -> VRM bone -> local rotation quaternions (T, 4) xyzw."""
        state = np.zeros(rot.shape[:1] + (len(self.jn), 8))
        state[..., 3:7] = mat_to_quat(self.mhr_globals(rot))
        state[..., 7] = 1.0
        return self.conv.locals(state)
