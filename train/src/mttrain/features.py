"""Model input features from MediaPipe Pose observations. Shared by training and inference.

Per frame (all causal):
- world landmarks / s, where s is the median shoulder width over the last SCALE_WINDOW seconds
  (frames with both shoulders visible); body-size invariant, robust to one bad frame;
- image landmarks: x, y centred on the frame, z (MediaPipe's relative depth);
- visibility, and whether the detector returned a pose at all.
History: for a target frame at time t, the latest frame at or before t - offset for each offset in
OFFSETS, with its actual age (webcams drop frames; parts may run at other rates later).
"""

import numpy as np

OFFSETS = np.array([0.0, 0.1, 0.2, 0.4, 0.8, 1.6])  # seconds
SCALE_WINDOW = 2.0
DEFAULT_SCALE = 0.33  # metres, recorded session's median MediaPipe shoulder width
FRAME_DIM = 33 * 3 + 33 * 3 + 33 + 1
L_SH, R_SH = 11, 12


def causal_scale(times, world, vis, present):
    """Median shoulder width over the trailing window, per frame (T,)."""
    w = np.linalg.norm(world[:, L_SH] - world[:, R_SH], axis=-1)
    ok = present & (vis[:, L_SH] > 0.5) & (vis[:, R_SH] > 0.5) & (w > 0.1) & (w < 0.8)
    out = np.empty(len(times))
    start = 0
    for t in range(len(times)):
        while times[start] < times[t] - SCALE_WINDOW:
            start += 1
        sel = w[start:t + 1][ok[start:t + 1]]
        out[t] = np.median(sel) if len(sel) else (out[t - 1] if t else DEFAULT_SCALE)
    return out


def frame_features(times, world, image, vis, present):
    """Observations (T, ...) -> (T, FRAME_DIM) float32."""
    s = causal_scale(times, world, vis, present)
    f = np.concatenate([
        (world / s[:, None, None]).reshape(len(times), -1),
        np.concatenate([image[..., :2] - 0.5, image[..., 2:3]], -1).reshape(len(times), -1),
        vis,
        present[:, None].astype(float),
    ], 1)
    f[~present, :-1] = 0.0
    return np.clip(f, -10, 10).astype(np.float32)


def history_indices(times, targets):
    """For target frame indices (K,), the frame index per offset (K, O) (-1 = before the first frame)
    and its age in seconds (K, O)."""
    want = times[targets][:, None] - OFFSETS[None] + 0.02  # tolerate timing jitter
    idx = np.searchsorted(times, want, side="right") - 1
    age = np.where(idx >= 0, times[targets][:, None] - times[np.maximum(idx, 0)], 0.0)
    return idx, age.astype(np.float32)


def model_input(feats, idx, age):
    """Gather (K, O, FRAME_DIM + 2): features, age, missing flag."""
    x = feats[np.maximum(idx, 0)]
    missing = idx < 0
    x[missing] = 0.0
    return np.concatenate([x, age[..., None], missing[..., None].astype(np.float32)], -1)
