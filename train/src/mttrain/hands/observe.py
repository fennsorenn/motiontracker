"""Synthetic MediaPipe hand observations (torch, batched), from the calibration of rendered hands
(`mttrain.hands.calibrate --json`).

In the canonical palm frame (`mtcore.hand_features`), per coordinate: observed = a + b * truth + noise.
b < 1 is MediaPipe pulling every hand toward an average shape (measured: about 0.6, so curls read
40-50% too small); noise has the measured covariance across the 63 coordinates and is AR(1) over frames
with the measured lag-1 correlation (10 Hz). Per sequence the style is randomized around the calibration
(slope, offsets, noise level), so the model has to infer it. Also: mirrored stretches (palm and back
confused) and missing frames (lost hand), as runs.
"""

import json

import numpy as np
import torch

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17


def canonical_t(P):
    """Torch version of mtcore.hand_features.canonical for a left hand: (..., 21, 3) -> (..., 21, 3)."""
    w = P[..., WRIST, :]
    f = P[..., MIDDLE_MCP, :] - w
    length = f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    f = f / length
    s = P[..., INDEX_MCP, :] - P[..., PINKY_MCP, :]
    s = s - (s * f).sum(-1, keepdim=True) * f
    s = s / s.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    R = torch.stack([f, s, torch.cross(f, s, dim=-1)], -2)
    return torch.einsum("...ij,...kj->...ki", R, P - w[..., None, :]) / length[..., None]


class Observer:
    def __init__(self, calib_json, device):
        fit = json.loads(open(calib_json).read())["fit"]
        self.a = torch.tensor(fit["a"], dtype=torch.float32, device=device)
        self.b = torch.tensor(fit["b"], dtype=torch.float32, device=device)
        cov = np.asarray(fit["cov"]) + 1e-8 * np.eye(63)
        self.L = torch.tensor(np.linalg.cholesky(cov), dtype=torch.float32, device=device)
        self.rho = float(fit["lag1"])
        self.p_flip = float(fit["flipped"])
        self.device = device

    def __call__(self, T, gen):
        """Truth (B, F, 21, 3) canonical, F consecutive frames at 10 Hz -> observed (B, F, 21, 3) canonical,
        missing (B, F) bool."""
        B, F = T.shape[:2]
        dev = self.device
        u = lambda *s: torch.rand(*s, device=dev, generator=gen)
        n = lambda *s: torch.randn(*s, device=dev, generator=gen)
        # per-sequence style around the calibration
        shrink = (1 - self.b) * (0.5 + u(B, 1))  # 0.5-1.5 x the measured pull toward the average hand
        b = 1 - shrink
        a = self.a + 0.02 * n(B, 63)
        scale = 0.5 * 3.2 ** u(B, 1, 1)  # noise level 0.5-1.6 x the measured
        # AR(1) noise with the measured covariance
        eps = n(B, F, 63) @ self.L.T
        noise = torch.empty_like(eps)
        noise[:, 0] = eps[:, 0]
        c = (1 - self.rho ** 2) ** 0.5
        for t in range(1, F):
            noise[:, t] = self.rho * noise[:, t - 1] + c * eps[:, t]
        flat = T.reshape(B, F, 63)
        obs = (a[:, None] + b[:, None] * flat + scale * noise).reshape(B, F, 21, 3)
        obs = canonical_t(obs)  # keep the canonical form exact (wrist at 0, palm axes)
        # mirrored stretches: palm and back confused for a run of frames
        flip_seq = u(B) < 4 * self.p_flip
        start = (u(B) * F).long()
        length = 1 + (u(B) * 6).long()
        idx = torch.arange(F, device=dev)[None]
        flip = flip_seq[:, None] & (idx >= start[:, None]) & (idx < (start + length)[:, None])
        obs = torch.where(flip[..., None, None], obs * torch.tensor([1.0, 1.0, -1.0], device=dev), obs)
        # missing frames: per sequence a loss rate, in runs
        rate = u(B, 1) * 0.35
        lost = u(B, F) < rate * 0.5
        lost = lost | torch.roll(lost, 1, 1) & (u(B, F) < 0.6)
        return obs, lost
