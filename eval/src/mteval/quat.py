"""Quaternion helpers. Quaternions are [x, y, z, w] (three.js / VRM order), arrays of shape (..., 4)."""

from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle between rotations a and b in degrees (sign-invariant: q and -q are equal)."""
    dot = np.abs(np.sum(normalize(a) * normalize(b), axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def local_mean(q: np.ndarray, center: int, lo: int, hi: int) -> np.ndarray:
    """Approximate mean rotation of q[lo:hi], sign-aligned to q[center]. Fine for small spreads."""
    window = q[lo:hi]
    signs = np.sign(np.sum(window * q[center], axis=-1))
    signs[signs == 0] = 1.0
    m = np.sum(window * signs[:, None], axis=0)
    return m / np.linalg.norm(m)
