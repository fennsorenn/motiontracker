"""M1 lifter output -> VRM 1.0 bone rotations, numpy only (shared by offline inference and the live runtime).

The lifter predicts rest-relative rotations for OUT_JOINTS in camera axes. LifterPose chains them into
MHR world joint rotations (joints the model doesn't predict keep their rest pose relative to their
parent) and converts those to VRM normalized local rotations with MhrToVrm for a given avatar rest pose.
Everything it needs about MHR is a small table (`table()` / `from_table()`), so the runtime does not
need pymomentum or the MHR assets.

Fingers (merger bundles): MHR's finger joints are driven only by the finger parameters, linearly (MHR's parameter
transform), as joint Euler angles; a joint's local rotation is its rest local rotation times Rz Ry Rx (checked
against pymomentum, 2026-11-14: 1.6e-6). The table's `fingers` entry holds the finger joints and that linear map,
so `vrm(rot, fingers=q)` poses them without pymomentum.
"""

import numpy as np

from .convert import HAND_ACROSS, MAP, VRM_CHILD, MhrToVrm

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
    """Rotation matrices (..., 3, 3) -> quaternions (..., 4) xyzw. Shepperd's method: each matrix is converted from its
    largest of w, x, y, z, so rotations near 180 deg stay exact (taking the signs of x, y, z from the off-diagonal
    differences, as before 2026-11-14, flipped finger bones whose global rotation came near 180 deg)."""
    m = np.asarray(R, dtype=np.float64)
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    cand = np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    k = np.argmax(cand, -1)
    s = np.sqrt(np.maximum(np.take_along_axis(cand, k[..., None], -1)[..., 0], 1e-12)) * 2  # 4 x the largest
    d21, d02, d10 = m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]
    s21, s02, s10 = m[..., 2, 1] + m[..., 1, 2], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 0] + m[..., 0, 1]
    q = np.where((k == 0)[..., None], np.stack([d21 / s, d02 / s, d10 / s, s / 4], -1),
        np.where((k == 1)[..., None], np.stack([s / 4, s10 / s, s02 / s, d21 / s], -1),
        np.where((k == 2)[..., None], np.stack([s10 / s, s / 4, s21 / s, d02 / s], -1),
                 np.stack([s02 / s, s21 / s, s / 4, d10 / s], -1))))
    q = q * np.where(q[..., 3:] < 0, -1.0, 1.0)  # w >= 0, as before
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


class LifterPose:
    def __init__(self, out_joints, out_parent_idx, joint_names, parents, rest_state, vrm_rest_positions, bones=None,
                 fingers=None):
        """out_joints / out_parent_idx: the lifter's joints and their parents (index into out_joints,
        -1 = none). joint_names, parents, rest_state (J, 8): MHR skeleton at zero pose.
        vrm_rest_positions: avatar bone -> rest position (normalized VRM). bones: only these VRM bones
        are needed (live use: far cheaper per frame); None = every bone the avatar has. fingers: optional
        {"params": [model parameter names], "joints": [MHR joint names], "map": (3 * joints, params) Euler angles
        rx, ry, rz per joint from the finger parameters}."""
        self.out_joints, self.out_parent_idx = list(out_joints), list(out_parent_idx)
        self.only = None if bones is None else list(bones)
        self.jn, self.parents = list(joint_names), list(parents)
        self.rest_state = np.asarray(rest_state, dtype=np.float64)
        self.rest_g = quat_to_mat(self.rest_state[:, 3:7])  # (J, 3, 3)
        self.out_idx = [self.jn.index(j) for j in self.out_joints]
        self.fingers = fingers
        if fingers is not None:
            self.finger_idx = {self.jn.index(j): k for k, j in enumerate(fingers["joints"])}
            self.finger_map = np.asarray(fingers["map"], dtype=np.float64)
        self.joints = range(len(self.jn))  # MHR joints to compute
        if self.only is not None:
            need = set()
            for b in self.only:
                need.add(self.jn.index(MAP[b][0]))
            for m in ("l", "r"):  # forearm twist (MhrToVrm.forearm)
                need |= {self.jn.index(f"{m}_lowarm"), self.jn.index(f"{m}_wrist_twist")}
            for j in list(need):
                while self.parents[j] >= 0:
                    j = self.parents[j]
                    need.add(j)
            self.joints = sorted(need)  # parents come before children in MHR's joint order
        self.set_avatar(vrm_rest_positions)

    def set_avatar(self, vrm_rest_positions):
        rest = {b: np.asarray(p, dtype=np.float64) for b, p in vrm_rest_positions.items()}
        if self.only is not None:  # the bones, their parents' chain is in self.only too, and their direction children
            keep = set(self.only) | {VRM_CHILD[b] for b in self.only if b in VRM_CHILD}
            keep |= {v for b in self.only if b in HAND_ACROSS for v in HAND_ACROSS[b][0]}  # palm roll
            rest = {b: p for b, p in rest.items() if b in keep}
        self.conv = MhrToVrm(self.jn, self.rest_state, rest)

    def table(self):
        """The MHR part as plain JSON-able data."""
        t = {"out_joints": self.out_joints, "out_parent_idx": self.out_parent_idx, "joint_names": self.jn,
             "parents": self.parents, "rest_state": self.rest_state.tolist()}
        if self.fingers is not None:
            t["fingers"] = {k: np.asarray(v).tolist() for k, v in self.fingers.items()}
        return t

    @classmethod
    def from_table(cls, t, vrm_rest_positions, bones=None):
        return cls(t["out_joints"], t["out_parent_idx"], t["joint_names"], t["parents"], t["rest_state"], vrm_rest_positions, bones,
                   t.get("fingers"))

    def mhr_globals(self, rot, fingers=None):
        """Rest-relative predictions (T, K, 3, 3) -> MHR world joint rotations (T, J, 3, 3) (with `bones`,
        only the joints those need; the rest are identity). fingers: (T, len(fingers["params"])) finger
        parameters (needs the table's `fingers`); None = fingers at rest."""
        T = len(rot)
        E = None
        if fingers is not None:
            a = (np.asarray(fingers, dtype=np.float64) @ self.finger_map.T).reshape(T, -1, 3)  # rx, ry, rz
            c, s = np.cos(a), np.sin(a)
            o, z = np.ones_like(a[..., 0]), np.zeros_like(a[..., 0])
            Rx = np.stack([o, z, z, z, c[..., 0], -s[..., 0], z, s[..., 0], c[..., 0]], -1).reshape(a.shape[:-1] + (3, 3))
            Ry = np.stack([c[..., 1], z, s[..., 1], z, o, z, -s[..., 1], z, c[..., 1]], -1).reshape(a.shape[:-1] + (3, 3))
            Rz = np.stack([c[..., 2], -s[..., 2], z, s[..., 2], c[..., 2], z, z, z, o], -1).reshape(a.shape[:-1] + (3, 3))
            E = Rz @ Ry @ Rx  # (T, n, 3, 3)
        Gc = [None] * len(self.out_joints)
        for k, p in enumerate(self.out_parent_idx):
            Gc[k] = rot[:, k] if p < 0 else Gc[p] @ rot[:, k]
        G = np.broadcast_to(np.eye(3), (T, len(self.jn), 3, 3)).copy()  # joints not computed: identity
        pred = dict(zip(self.out_idx, range(len(self.out_joints))))
        for j in self.joints:
            p = self.parents[j]
            if j in pred:
                G[:, j] = R0.T @ Gc[pred[j]] @ self.rest_g[j]
            elif p < 0:
                G[:, j] = self.rest_g[j]
            else:
                G[:, j] = G[:, p] @ (self.rest_g[p].T @ self.rest_g[j])
                if E is not None and j in self.finger_idx:
                    G[:, j] = G[:, j] @ E[:, self.finger_idx[j]]
        return G

    def vrm(self, rot, fingers=None):
        """Predictions (T, K, 3, 3) [, finger parameters (T, P)] -> VRM bone -> local rotation quaternions (T, 4) xyzw."""
        state = np.zeros(rot.shape[:1] + (len(self.jn), 8))
        state[..., 3:7] = mat_to_quat(self.mhr_globals(rot, fingers))
        state[..., 7] = 1.0
        out = self.conv.locals(state)
        return out if self.only is None else {b: q for b, q in out.items() if b in self.only}
