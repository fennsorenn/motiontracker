"""MHR forward kinematics in torch: model parameters (..., P) -> skeleton states (..., J, 8),
matching pymomentum.geometry.model_parameters_to_skeleton_state (positions in cm, quaternions xyzw,
global scale)."""

import numpy as np
import torch


def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def quat_rotate(q, v):
    u, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def _axis_quat(angle, axis):
    h = 0.5 * angle
    z = torch.zeros_like(h)
    s, c = torch.sin(h), torch.cos(h)
    parts = [z, z, z]
    parts[axis] = s
    return torch.stack(parts + [c], -1)


class MhrFK(torch.nn.Module):
    """`keep`: joint indices whose states are needed (their ancestors are computed too); the other
    joints' states are left at zero. Default: all joints."""

    def __init__(self, character, keep=None):
        super().__init__()
        sk, pt = character.skeleton, character.parameter_transform
        self.register_buffer("transform", torch.tensor(np.asarray(pt.transform), dtype=torch.float32))
        self.register_buffer("offsets", torch.tensor(np.asarray(sk.offsets), dtype=torch.float32))
        self.register_buffer("pre", torch.tensor(np.asarray(sk.pre_rotations), dtype=torch.float32))
        self.parents = [int(p) if 0 <= int(p) < len(sk.joint_names) else -1 for p in np.asarray(sk.joint_parents)]
        self.J = len(self.parents)
        need = set(range(self.J)) if keep is None else set()
        for j in (keep or []):
            while j >= 0 and j not in need:
                need.add(j)
                j = self.parents[j]
        # joints grouped by depth: every joint of a level only depends on earlier levels
        depth = []
        for p in self.parents:
            depth.append(0 if p < 0 else depth[p] + 1)
        order = sorted(need)
        pos = {j: i for i, j in enumerate(order)}  # full joint index -> compact index
        self.levels = []
        for k in range(max(depth) + 1):
            lvl = [j for j, d in enumerate(depth) if d == k and j in need]
            if lvl:
                self.register_buffer(f"lvl{k}", torch.tensor([pos[j] for j in lvl]))
                self.register_buffer(f"par{k}", torch.tensor([pos.get(self.parents[j], 0) for j in lvl]))
                self.levels.append((k, self.parents[lvl[0]] < 0))
        self.register_buffer("need", torch.tensor(order))
        rows = torch.tensor([7 * j + c for j in order for c in range(7)])
        self.register_buffer("transform_need", self.transform[rows])
        self.register_buffer("offsets_need", self.offsets[order])
        self.register_buffer("pre_need", self.pre[order])

    def forward(self, mp):
        shp = mp.shape[:-1]
        need = self.need
        Jn = len(need)
        # joint parameters of the needed joints only; levels index into this compact set
        jp = (mp.reshape(-1, mp.shape[-1]) @ self.transform_need.T).reshape(-1, Jn, 7)
        t, r, s = jp[..., :3], jp[..., 3:6], jp[..., 6]
        # local rotation: pre-rotation, then Euler XYZ (R = Rz Ry Rx)
        q = quat_mul(_axis_quat(r[..., 2], 2), quat_mul(_axis_quat(r[..., 1], 1), _axis_quat(r[..., 0], 0)))
        q = quat_mul(self.pre_need.expand_as(q), q)
        trans = self.offsets_need + t
        scale = torch.exp2(s)
        n = jp.shape[0]
        gq = torch.empty(n, Jn, 4, device=mp.device, dtype=mp.dtype)
        gt = torch.empty(n, Jn, 3, device=mp.device, dtype=mp.dtype)
        gs = torch.empty(n, Jn, device=mp.device, dtype=mp.dtype)
        for k, is_root in self.levels:
            idx = getattr(self, f"lvl{k}")
            if is_root:
                gq[:, idx], gt[:, idx], gs[:, idx] = q[:, idx], trans[:, idx], scale[:, idx]
                continue
            pidx = getattr(self, f"par{k}")
            pq, pt_, ps = gq[:, pidx], gt[:, pidx], gs[:, pidx]
            gt[:, idx] = pt_ + ps[..., None] * quat_rotate(pq, trans[:, idx])
            gq[:, idx] = quat_mul(pq, q[:, idx])
            gs[:, idx] = ps * scale[:, idx]
        out = torch.zeros(n, self.J, 8, device=mp.device, dtype=mp.dtype)
        out[:, need] = torch.cat([gt, gq, gs[..., None]], -1)
        return out.reshape(*shp, self.J, 8)
