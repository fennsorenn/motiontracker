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
from ..camera import DIST_RANGE, FOV_RANGE, L_HIP, L_SH, R_HIP, R_SH, SW_RANGE
from ..skeleton import CONTACT_LM, CONTACT_PARENT, OUT_JOINTS, PARENT_IDX
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


# ── head response (mpstyle.head_lock) ──────────────────────────────────────────────────────────────


def rotvec_to_mat(v):
    th = v.norm(dim=-1, keepdim=True)
    k = v / th.clamp_min(1e-12)
    z = torch.zeros_like(k[..., 0])
    K = torch.stack([torch.stack([z, -k[..., 2], k[..., 1]], -1), torch.stack([k[..., 2], z, -k[..., 0]], -1),
                     torch.stack([-k[..., 1], k[..., 0], z], -1)], -2)
    th = th[..., None]
    return torch.eye(3, device=v.device) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def mat_to_rotvec(M):
    c = ((M.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    th = torch.arccos(c)
    w = torch.stack([M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0], M[..., 1, 0] - M[..., 0, 1]], -1)
    s = torch.sin(th)
    return w * torch.where(s > 1e-6, th / (2 * s).clamp_min(1e-12), torch.full_like(th, 0.5))[..., None]


def head_lock(X, delta, C, gain, nod, noise):
    """mpstyle.head_lock, batched: X (B, T, 33, 3), delta, C (B, T, 3, 3), gain (B, 3) per ms.HEAD_AXES,
    nod (B,) rad, noise (B, T, 3)."""
    Ct = C.transpose(-1, -2)
    r = mat_to_rotvec(Ct @ delta @ C) * gain[:, None, :]
    r = torch.cat([r[..., :1] + nod[:, None, None], r[..., 1:]], -1)
    M = rotvec_to_mat(noise) @ C @ rotvec_to_mat(r) @ Ct @ delta.transpose(-1, -2)
    pivot = 0.5 * (X[..., I["left_ear"], :] + X[..., I["right_ear"], :])
    X = X.clone()
    X[..., FACE, :] = pivot[..., None, :] + torch.einsum("btij,btkj->btki", M, X[..., FACE, :] - pivot[..., None, :])
    return X


def landmark_frames(X):
    """mpstyle.landmark_frames, torch."""
    def fr(xa, up):
        x = xa / xa.norm(dim=-1, keepdim=True)
        z = torch.cross(x, up, dim=-1)
        z = z / z.norm(dim=-1, keepdim=True)
        return torch.stack([x, torch.cross(z, x, dim=-1), z], -1)
    mean = lambda *ns: X[..., [I[n] for n in ns], :].mean(-2)
    C = fr(X[..., I["left_shoulder"], :] - X[..., I["right_shoulder"], :],
           mean("left_shoulder", "right_shoulder") - mean("left_hip", "right_hip"))
    H = fr(X[..., I["left_ear"], :] - X[..., I["right_ear"], :],
           mean("left_eye", "right_eye") - mean("mouth_left", "mouth_right"))
    return C, H


def rotvec_between(a, b):
    c = torch.cross(a, b, dim=-1)
    n = c.norm(dim=-1, keepdim=True)
    ang = torch.atan2(n, (a * b).sum(-1, keepdim=True))
    return c / n.clamp_min(1e-12) * ang


def hand_lock(X, gain, noise):
    """mpstyle.hand_lock, batched: X (B, T, 33, 3), gain (B,), noise (B, T, 2, 3)."""
    X = X.clone()
    for k, (el, wr, pts) in enumerate(ms.HAND_SIDES):
        f = X[..., wr, :] - X[..., el, :]
        h = 0.5 * (X[..., pts[0], :] + X[..., pts[1], :]) - X[..., wr, :]
        v = rotvec_between(f, h)
        M = rotvec_to_mat(noise[..., k, :]) @ rotvec_to_mat(gain[:, None, None] * v) @ rotvec_to_mat(v).transpose(-1, -2)
        w = X[..., wr, :][..., None, :]
        X[..., pts, :] = w + torch.einsum("btij,btkj->btki", M, X[..., pts, :] - w)
    return X


class HeadRig(torch.nn.Module):
    """mpstyle.HeadRig, batched: states (B, T, J, 8), true landmarks X (B, T, 33, 3) metres ->
    delta (B, T, 3, 3), chest frame C (B, T, 3, 3)."""

    def __init__(self, joint_names, rest_state, landmark_model):
        super().__init__()
        self.n = list(joint_names).index("c_neck")
        C0, H0 = ms.landmark_frames(landmark_model(rest_state[None])[0] / 100.0)
        self.register_buffer("rest_rel", torch.tensor(C0.T @ H0, dtype=torch.float32))

    def forward(self, states, X):
        C, H = landmark_frames(X)
        return H @ self.rest_rel.T @ C.transpose(-1, -2), C


def head_noise(gen, dt, sigma_deg, tau=ms.HEAD_NOISE_TAU):
    """mpstyle.head_noise, batched: dt (B, T) s between frames, sigma_deg (B,) -> (B, T, 3) rad."""
    B, T = dt.shape
    s = torch.deg2rad(sigma_deg)[:, None]
    e = torch.randn(B, T, 3, generator=gen, device=dt.device) * s[..., None]
    r = torch.exp(-dt / tau)
    out = torch.empty(B, T, 3, device=dt.device)
    out[:, 0] = e[:, 0]
    for t in range(1, T):
        out[:, t] = r[:, t, None] * out[:, t - 1] + torch.sqrt(1 - r[:, t, None] ** 2) * e[:, t]
    return out


# ── MediaPipe style ──────────────────────────────────────────────────────────────────────────────


class Style:
    """Per-sequence MediaPipe biases, as mpstyle.Style but with (B, ...) tensors."""

    def __init__(self, hip_widen, image_z, face_placement, face_world, limb, offset, depth_gain, depth_bias,
                 depth_relax, head_gain, head_nod_deg, head_noise_deg, head3d, hand_gain, hand_noise_deg):
        self.hip_widen, self.image_z = hip_widen, image_z  # (B,)
        self.face_placement, self.face_world = face_placement, face_world  # (B, 33)
        self.limb = limb  # name -> (B,)
        self.offset = offset  # (B, 33, 3) metres
        self.depth_gain, self.depth_bias = depth_gain, depth_bias  # (B, K) per ms.DEPTH_KEYS, bias metres
        self.depth_relax = depth_relax  # (B,)
        self.head_gain = head_gain  # space -> (B, 3) per ms.HEAD_AXES
        self.head_nod_deg, self.head_noise_deg = head_nod_deg, head_noise_deg  # space -> (B,)
        self.head3d = head3d  # (B,) bool
        self.hand_gain, self.hand_noise_deg = hand_gain, hand_noise_deg  # space -> (B,)

    @classmethod
    def from_numpy(cls, styles, device=None):
        t = lambda v: torch.tensor(np.asarray(v), dtype=torch.float32, device=device)
        return cls(t([s.hip_widen for s in styles]), t([s.image_z for s in styles]),
                   t([s.face_placement for s in styles]), t([s.face_world for s in styles]),
                   {k: t([s.limb[k] for s in styles]) for k in ms.LIMB}, t([s.offset for s in styles]),
                   t([s.depth_gain for s in styles]), t([s.depth_bias for s in styles]),
                   t([s.depth_relax for s in styles]),
                   {k: t([s.head_gain[k] for s in styles]) for k in ms.HEAD_GAIN},
                   {k: t([s.head_nod_deg[k] for s in styles]) for k in ms.HEAD_NOD_DEG},
                   {k: t([s.head_noise_deg[k] for s in styles]) for k in ms.HEAD_NOISE_DEG},
                   torch.tensor([s.head3d for s in styles], device=device),
                   {k: t([s.hand_gain[k] for s in styles]) for k in ms.HAND_GAIN},
                   {k: t([s.hand_noise_deg[k] for s in styles]) for k in ms.HAND_NOISE_DEG})

    @classmethod
    def sample(cls, B, gen, device, spread=0.08, offset_cm=0.8, image_z_spread=0.25, depth_spread=0.25,
               head_spread=ms.HEAD_SPREAD):
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
        relax = torch.rand(B, generator=gen, device=device)
        hs = lambda *shape: torch.exp(torch.randn(B, *shape, generator=gen, device=device) * head_spread)

        def spread_gain(g, *shape):  # mpstyle._spread_gain: gains of 1 (off) stay 1
            g = torch.as_tensor(g, dtype=torch.float32, device=device)
            return torch.where(g >= 1.0, torch.ones_like(g), (g * hs(*shape)).clamp(0.02, 1.0))

        head_gain = {k: spread_gain(g, 3) for k, g in ms.HEAD_GAIN.items()}
        head_nod = {k: v + ms.HEAD_NOD_SPREAD_DEG * torch.randn(B, generator=gen, device=device)
                    for k, v in ms.HEAD_NOD_DEG.items()}
        head_noise = {k: v * hs() for k, v in ms.HEAD_NOISE_DEG.items()}
        head3d = torch.rand(B, generator=gen, device=device) >= ms.HEAD3D_DROP
        hand_gain = {k: spread_gain(g) for k, g in ms.HAND_GAIN.items()}
        hand_noise = {k: v * hs() for k, v in ms.HAND_NOISE_DEG.items()}
        return cls(ms.HIP_WIDEN * ln(B), image_z, fp, fw, limb, offset, gain, bias, relax, head_gain, head_nod, head_noise,
                   head3d, hand_gain, hand_noise)

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

    def depth_at(self, dist):
        """mpstyle.Style.depth_at, batched: dist (B,) metres -> gain, bias (B, K)."""
        slope = torch.tensor([ms.DEPTH_SLOPE[k] for k in ms.DEPTH_KEYS], device=dist.device)
        g = (self.depth_gain + slope * (dist[:, None] - ms.DEPTH_REF_DIST)).clamp(0.02, 1.0)
        r = self.depth_relax[:, None]
        return g + r * (1 - g), self.depth_bias * (1 - r)

    def flatten(self, W, dist):
        """mpstyle.Style.flatten, batched: hip-centred camera-frame landmarks (B, T, 33, 3), metres;
        dist (B,) camera to shoulders, metres."""
        gain, bias = self.depth_at(dist)
        W = W.clone()
        for key, a, b, moved, w in ms.DEPTH_STEPS:
            k = ms.DEPTH_KEYS.index(key)
            d = W[..., b, :].mean(-2) - W[..., a, :].mean(-2)
            L = d.norm(dim=-1)
            ez = torch.maximum(torch.minimum(gain[:, k, None] * d[..., 2] + bias[:, k, None], L), -L)
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
    fx = 0.5 / torch.tan(torch.deg2rad(u(*FOV_RANGE)) / 2)
    dist = torch.exp(u(math.log(DIST_RANGE[0]), math.log(DIST_RANGE[1])))
    dist = torch.maximum(torch.minimum(dist, fx * width / SW_RANGE[0]), fx * width / SW_RANGE[1])
    sh_y = u(0.40, 0.75)
    lift = torch.zeros(B, 3, device=dev)
    lift[:, 1] = u(0.0, 0.55)
    eye = chest + dist[:, None] * dirn + lift
    aim = torch.zeros(B, 3, device=dev)
    aim[:, 1] = dist * (sh_y - 0.5) / (fx * 4 / 3)
    R = look_at(eye, chest + aim, torch.deg2rad(u(-4, 4)))
    return R, eye, fx


def shoulder_distance(X, R, t):
    """camera.shoulder_distance, batched: X (B, T, 33, 3) with padded frames NaN -> (B,) metres."""
    z = to_camera(X[:, :, [L_SH, R_SH]], R, t)[..., 2].mean(-1)
    return z.nanmedian(1).values


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

    def contact_offsets(self, states, cam_R, landmarks):
        """skeleton.TargetExtractor.contact_offsets, batched: landmarks (B, K, 33, 3) cm -> (B, K, 3, 3)."""
        G = cam_R[:, None, None] @ quat_to_mat(states[:, :, self.idx, 3:7]) @ self.rest_g.transpose(-1, -2)
        out = []
        for p, lms in zip(CONTACT_PARENT, CONTACT_LM):
            d = torch.einsum("bij,btj->bti", cam_R, (landmarks[:, :, lms].mean(2) - states[:, :, self.idx[p], :3]) / 100.0)
            out.append(torch.einsum("btji,btj->bti", G[:, :, p], d))
        return torch.stack(out, 2)
