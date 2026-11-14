"""Hand model input: MediaPipe's 21 hand points in the hand's own palm frame (numpy only; shared by
training and the live runtime, like `mtcore.features` for the body).

MediaPipe's hand world landmarks are not camera-aligned: their orientation varies from frame to frame
(measured on rendered hands, 2026-11-12), so only the hand's shape is used. Canonical form:
- the right hand is mirrored (x -> -x) into a left hand: one model serves both (MHR's right hand is the
  left mirrored for the same finger parameters);
- origin at the wrist, axes = palm frame: fwd (wrist -> middle knuckle), thumbward (little -> index
  knuckle, made orthogonal), palm (fwd x thumbward: the side a left hand's fingers curl toward);
- scaled by the wrist -> middle knuckle length (MediaPipe normalizes hand size; the true size is unknown).
"""

import numpy as np

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17


def palm_frame(P):
    """(..., 21, 3) -> (origin (..., 3), R (..., 3, 3) rows fwd, thumbward, palm, length (...))."""
    w = P[..., WRIST, :]
    f = P[..., MIDDLE_MCP, :] - w
    length = np.linalg.norm(f, axis=-1)
    f = f / np.maximum(length, 1e-9)[..., None]
    s = P[..., INDEX_MCP, :] - P[..., PINKY_MCP, :]
    s = s - (s * f).sum(-1, keepdims=True) * f
    s = s / np.maximum(np.linalg.norm(s, axis=-1, keepdims=True), 1e-9)
    return w, np.stack([f, s, np.cross(f, s)], -2), length


def canonical(P, side):
    """Hand points (..., 21, 3) of side 'l' / 'r' -> canonical left-hand shape (..., 21, 3)."""
    P = np.asarray(P, dtype=np.float64)
    if side == "r":
        P = P * np.array([-1.0, 1.0, 1.0])
    w, R, length = palm_frame(P)
    return np.einsum("...ij,...kj->...ki", R, P - w[..., None, :]) / np.maximum(length, 1e-9)[..., None, None]
