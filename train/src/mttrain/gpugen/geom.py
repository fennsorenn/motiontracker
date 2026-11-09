"""Landmarks, MediaPipe style biases, camera and targets on batches (torch). Mirrors
mtcore.landmarks.LandmarkModel, mttrain.mpstyle.Style, mttrain.camera and mttrain.skeleton.

Batch convention: leading dimension B (sequences), then time; per-sequence parameters are (B, ...)
tensors and broadcast over time.
"""

import math

import numpy as np
import torch

from mtcore.landmarks import MP33

from .. import mpstyle as ms
from ..camera import L_HIP, L_SH, R_HIP, R_SH
from ..skeleton import OUT_JOINTS, PARENT_IDX
from .fk import quat_rotate

I = ms.I
FACE = ms.FACE
UPPER = [k for k in range(33) if k not in (I[n] for n in MP33[23:])]


def quat_to_mat(q):
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


class Landmarks(torch.nn.Module):
    """Skeleton states (..., J, 8) -> MediaPipe's 33 landmarks (..., 33, 3), same units (cm)."""

    def __init__(self, lm):
        super().__init__()
        self.register_buffer("joint", torch.tensor(lm.joint, dtype=torch.long))
        self.register_buffer("offset", torch.tensor(lm.offset, dtype=torch.float32))

    def forward(self, states):
        st = states[..., self.joint, :]
        return st[..., :3] + st[..., 7:8] * quat_rotate(st[..., 3:7], self.offset.expand_as(st[..., :3]))


# ── MediaPipe style ──────────────────────────────────────────────────────────────────────────────


class Style:
    """Per-sequence MediaPipe biases, as mpstyle.Style but with (B, ...) tensors."""

    def __init__(self, hip_widen, image_z, face_placement, face_world, limb, offset, depth_gain, depth_bias):
        self.hip_widen, self.image_z = hip_widen, image_z  # (B,)
        self.face_placement, self.face_world = face_placement, face_world  # (B, 33)
        self.limb = limb  # name -> (B,)
        self.offset = offset  # (B, 33, 3) metres
        self.depth_gain, self.depth_bias = depth_gain, depth_bias  # (B, K) per ms.DEPTH_KEYS, bias metres

    @classmethod
    def from_numpy(cls, styles, device=None):
        t = lambda v: torch.tensor(np.asarray(v), dtype=torch.float32, device=device)
        return cls(t([s.hip_widen for s in styles]), t([s.image_z for s in styles]),
                   t([s.face_placement for s in styles]), t([s.face_world for s in styles]),
                   {k: t([s.limb[k] for s in styles]) for k in ms.LIMB}, t([s.offset for s in styles]),
                   t([s.depth_gain for s in styles]), t([s.depth_bias for s in styles]))

    @classmethod
    def sample(cls, B, gen, device, spread=0.08, offset_cm=0.8, image_z_spread=0.25, depth_spread=0.25):
        ln = lambda *shape: torch.exp(torch.randn(*shape, generator=gen, device=device) * spread)
        fp = torch.tensor(ms._factors(ms.FACE_PLACEMENT), dtype=torch.float32, device=device).repeat(B, 1)
        fw = torch.tensor(ms._factors(ms.FACE_WORLD), dtype=torch.float32, device=device).repeat(B, 1)
        fp[:, FACE] = 1 + (fp[:, FACE] - 1) * ln(B, 1)
        fw[:, FACE] = fw[:, FACE] * ln(B, 1)
        image_z = ms.IMAGE_Z * torch.exp(torch.randn(B, generator=gen, device=device) * image_z_spread)
        limb = {k: v * ln(B) for k, v in ms.LIMB.items()}
        offset = torch.randn(B, 33, 3, generator=gen, device=device) * (offset_cm / 100)
        K = len(ms.DEPTH)
        g0 = torch.tensor([g for g, _ in ms.DEPTH.values()], device=device)
        b0 = torch.tensor([b for _, b in ms.DEPTH.values()], device=device)
        gain = (g0 * torch.exp(torch.randn(B, K, generator=gen, device=device) * depth_spread)).clamp(0.05, 1.0)
        bias = b0 + torch.randn(B, K, generator=gen, device=device) * (0.01 + 0.3 * b0.abs())
        return cls(ms.HIP_WIDEN * ln(B), image_z, fp, fw, limb, offset, gain, bias)

    @staticmethod
    def _lateral(X, factors):
        """X (B, T, 33, 3), factors (B, 33)."""
        ears = X[..., I["left_ear"], :] - X[..., I["right_ear"], :]
        axis = ears / ears.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        centre = 0.5 * (X[..., I["left_ear"], :] + X[..., I["right_ear"], :])
        along = ((X - centre[..., None, :]) * axis[..., None, :]).sum(-1, keepdim=True)
        return X + (factors[:, None, :, None] - 1) * along * axis[..., None, :]

    def placement(self, X):
        """3D landmarks (B, T, 33, 3) -> where MediaPipe places its points."""
        X = self._lateral(X, self.face_placement)
        mid = 0.5 * (X[..., I["left_hip"], :] + X[..., I["right_hip"], :])
        hw = self.hip_widen[:, None, None]
        X = X.clone()
        for n in ("left_hip", "right_hip"):
            X[..., I[n], :] = mid + hw * (X[..., I[n], :] - mid)
        return X

    def flatten(self, W):
        """mpstyle.flatten, batched: hip-centred camera-frame landmarks (B, T, 33, 3), metres."""
        W = W.clone()
        for key, a, b, moved, w in ms.DEPTH_STEPS:
            k = ms.DEPTH_KEYS.index(key)
            d = W[..., b, :].mean(-2) - W[..., a, :].mean(-2)
            L = d.norm(dim=-1)
            ez = torch.maximum(torch.minimum(self.depth_gain[:, k, None] * d[..., 2] + self.depth_bias[:, k, None], L), -L)
            xy = d[..., :2].norm(dim=-1)
            s = ((L * L - ez * ez).clamp_min(0).sqrt() / xy.clamp_min(1e-9)).clamp(max=ms.FLAT_MAX_STRETCH)
            e = torch.cat([d[..., :2] * s[..., None], ez[..., None]], -1)
            wt = torch.tensor(w, dtype=W.dtype, device=W.device)[:, None]
            W[..., moved, :] += wt * (e - d)[..., None, :]
        return W

    def world(self, W):
        """World landmarks (B, T, 33, 3), metres, hip-midpoint origin, already flattened -> MediaPipe's
        world-only distortions."""
        W = W.clone()
        lb = lambda k: self.limb[k][:, None, None, None]
        sh_mid = 0.5 * (W[..., I["left_shoulder"], :] + W[..., I["right_shoulder"], :])
        W[..., UPPER, :] -= (1 - lb("torso")) * sh_mid[..., None, :]
        sh_mid = 0.5 * (W[..., I["left_shoulder"], :] + W[..., I["right_shoulder"], :])
        W[..., FACE, :] -= (1 - lb("neck")) * (W[..., I["nose"], :] - sh_mid)[..., None, :]
        W = self._lateral(W, self.face_world)
        for seg, a, b, moved in ms.CHAINS:
            d = W[..., I[b], :] - W[..., I[a], :]
            W[..., [I[m] for m in moved], :] -= (1 - lb(seg)) * d[..., None, :]
        return W + self.offset[:, None]


# ── camera ───────────────────────────────────────────────────────────────────────────────────────


def look_at(eye, target, roll):
    """Batched camera rotations (B, 3, 3): rows image-right, image-down, forward (camera.look_at)."""
    f = target - eye
    f = f / f.norm(dim=-1, keepdim=True)
    up = torch.tensor([0.0, 1.0, 0.0], device=eye.device).expand_as(f)
    r = torch.cross(f, up, dim=-1)
    r = r / r.norm(dim=-1, keepdim=True)
    d = torch.cross(f, r, dim=-1)
    c, s = torch.cos(roll)[:, None], torch.sin(roll)[:, None]
    r, d = c * r + s * d, -s * r + c * d
    return torch.stack([r, d, f], 1)


def sample_desk_camera(X, gen):
    """Batched camera.sample_desk_camera. X: (B, T, 33, 3) landmarks in MHR metres (after placement);
    padded frames NaN (medians are over the valid frames). Returns R (B, 3, 3), t (B, 3), fx (B,)."""
    B, dev = X.shape[0], X.device
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, generator=gen, device=dev)
    sh = X[:, :, [L_SH, R_SH]].mean(2)
    chest = sh.nanmedian(1).values
    across = (X[:, :, L_SH] - X[:, :, R_SH]).nanmedian(1).values
    width = across.norm(dim=-1)
    front = torch.cross(across, torch.tensor([0.0, 1.0, 0.0], device=dev).expand_as(across), dim=-1)
    front[:, 1] = 0.0
    front = front / front.norm(dim=-1, keepdim=True)
    yaw = torch.deg2rad(u(-25, 25))
    c, s = torch.cos(yaw), torch.sin(yaw)
    dirn = torch.stack([c * front[:, 0] + s * front[:, 2], torch.zeros_like(c), -s * front[:, 0] + c * front[:, 2]], -1)
    fx = 0.5 / torch.tan(torch.deg2rad(u(55, 80)) / 2)
    sw_frac = torch.exp(math.log(0.25) + 0.2 * torch.randn(B, generator=gen, device=dev))
    sh_y = u(0.45, 0.72)
    dist = fx * width / sw_frac
    lift = torch.zeros(B, 3, device=dev)
    lift[:, 1] = u(0.0, 0.55)
    eye = chest + dist[:, None] * dirn + lift
    aim = torch.zeros(B, 3, device=dev)
    aim[:, 1] = dist * (sh_y - 0.5) / (fx * 4 / 3)
    R = look_at(eye, chest + aim, torch.deg2rad(u(-4, 4)))
    return R, eye, fx


def to_camera(X, R, t):
    """X (B, T, N, 3) MHR metres -> camera frame."""
    return torch.einsum("btni,bji->btnj", X - t[:, None, None], R)


def project(X, R, t, fx, aspect=4 / 3):
    """camera.Camera.project, batched: -> pose2d (B, T, 33, 3)."""
    C = to_camera(X, R, t)
    z = C[..., 2].clamp_min(1e-3)
    f = fx[:, None, None]
    u = 0.5 + f * C[..., 0] / z
    v = 0.5 + f * aspect * C[..., 1] / z
    hip_z = 0.5 * (z[..., L_HIP] + z[..., R_HIP])[..., None]
    d = f * (C[..., 2] - hip_z) / hip_z
    return torch.stack([u, v, d], -1)


def flat_image_z(X, R, t, fx, flat):
    """camera.Camera.flat_image_z, batched: pose2d z from flattened hip-centred depths."""
    C = to_camera(X, R, t)
    hip_z = (0.5 * (C[..., L_HIP, 2] + C[..., R_HIP, 2])).clamp_min(1e-3)[..., None]
    return fx[:, None, None] * flat[..., 2] / hip_z


def world_landmarks(X, R, t):
    C = to_camera(X, R, t)
    return C - 0.5 * (C[..., L_HIP, :] + C[..., R_HIP, :])[..., None, :]


# ── targets ──────────────────────────────────────────────────────────────────────────────────────


class Targets(torch.nn.Module):
    """skeleton.TargetExtractor, batched: states (B, K, J, 8), cam_R (B, 3, 3) -> rot (B, K, 17, 3, 3),
    pos (B, K, 17, 3), off (B, K, 17, 3)."""

    def __init__(self, joint_names, rest_state):
        super().__init__()
        idx = [joint_names.index(j) for j in OUT_JOINTS]
        self.register_buffer("idx", torch.tensor(idx))
        self.register_buffer("rest_g", quat_to_mat(torch.tensor(rest_state[idx, 3:7], dtype=torch.float32)))

    def forward(self, states, cam_R):
        G = quat_to_mat(states[:, :, self.idx, 3:7])
        Rc = cam_R[:, None, None]
        Gc = Rc @ G @ self.rest_g.transpose(-1, -2)
        rot = torch.empty_like(Gc)
        for k, p in enumerate(PARENT_IDX):
            rot[:, :, k] = Gc[:, :, k] if p < 0 else Gc[:, :, p].transpose(-1, -2) @ Gc[:, :, k]
        X = (states[:, :, self.idx, :3] - states[:, :, self.idx[:1], :3]) / 100.0
        pos = X @ cam_R[:, None].transpose(-1, -2)
        off = torch.zeros_like(pos)
        for k, p in enumerate(PARENT_IDX):
            if p >= 0:
                off[:, :, k] = torch.einsum("btji,btj->bti", Gc[:, :, p], pos[:, :, k] - pos[:, :, p])
        return rot, pos, off
