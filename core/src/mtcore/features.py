"""Model input features from MediaPipe Pose observations. Shared by training and inference.

Per frame (all causal):
- world landmarks / s, where s is the median shoulder width over the last SCALE_WINDOW seconds
  (frames with both shoulders visible); body-size invariant, robust to one bad frame;
- image landmarks: x, y centred on the frame, z (MediaPipe's relative depth);
- visibility, and whether the detector returned a pose at all.
The 3D head points (world xyz and image z of the 11 face landmarks) can be dropped (head3d=False,
zeros): MediaPipe's 3D head barely follows the head (mttrain.mpstyle), so models are trained with them
dropped for some sequences and can run live either way.
History: for a target frame at time t, the latest frame at or before t - offset for each offset in
OFFSETS, with its actual age (webcams drop frames; parts may run at other rates later).
"""

from collections import deque

import numpy as np

OFFSETS = np.array([0.0, 0.1, 0.2, 0.4, 0.8, 1.6])  # seconds
SCALE_WINDOW = 2.0
DEFAULT_SCALE = 0.33  # metres, recorded session's median MediaPipe shoulder width
FRAME_DIM = 33 * 3 + 33 * 3 + 33 + 1
L_SH, R_SH = 11, 12
HEAD = list(range(11))  # face landmarks (MP33[:11])
HEAD3D_DIMS = [3 * k + c for k in HEAD for c in range(3)] + [99 + 3 * k + 2 for k in HEAD]  # world xyz, image z


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


def frame_features(times, world, image, vis, present, head3d=True):
    """Observations (T, ...) -> (T, FRAME_DIM) float32. head3d=False: 3D head points zeroed."""
    return scaled_features(world, image, vis, present, causal_scale(times, world, vis, present), head3d)


def scaled_features(world, image, vis, present, s, head3d=True):
    """frame_features given the per-frame scale s (T,)."""
    T = len(s)
    f = np.concatenate([
        (world / s[:, None, None]).reshape(T, -1),
        np.concatenate([image[..., :2] - 0.5, image[..., 2:3]], -1).reshape(T, -1),
        vis,
        present[:, None].astype(float),
    ], 1)
    f[~present, :-1] = 0.0
    if not head3d:
        f[:, HEAD3D_DIMS] = 0.0
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


class StreamFeatures:
    """One frame at a time (live use): push(t, ...) returns model_input for that frame, (O, FRAME_DIM + 2),
    identical to frame_features + history_indices + model_input over the whole sequence so far. Keeps
    only the frames the scale window and the oldest offset still need."""

    def __init__(self, head3d=True):
        self.head3d = head3d
        self.widths = deque()  # (t, shoulder width) of frames that pass the scale check
        self.scale = None
        self.frames = deque()  # (t, features)

    def push(self, t, world, image, vis, present):
        """t: seconds; world (33, 3), image (33, 3), vis (33,), present: bool."""
        w = float(np.linalg.norm(world[L_SH] - world[R_SH]))
        if present and vis[L_SH] > 0.5 and vis[R_SH] > 0.5 and 0.1 < w < 0.8:
            self.widths.append((t, w))
        while self.widths and self.widths[0][0] < t - SCALE_WINDOW:
            self.widths.popleft()
        if self.widths:
            self.scale = float(np.median([v for _, v in self.widths]))
        elif self.scale is None:
            self.scale = DEFAULT_SCALE
        f = scaled_features(world[None], image[None], vis[None], np.array([present]), np.array([self.scale]), self.head3d)[0]
        self.frames.append((t, f))
        oldest = t - OFFSETS[-1] + 0.02
        while len(self.frames) > 1 and self.frames[1][0] <= oldest:
            self.frames.popleft()
        out = np.zeros((len(OFFSETS), FRAME_DIM + 2), np.float32)
        times = [ft for ft, _ in self.frames]
        for o, off in enumerate(OFFSETS):
            k = int(np.searchsorted(times, t - off + 0.02, side="right")) - 1
            if k < 0:
                out[o, -1] = 1.0
            else:
                out[o, :FRAME_DIM] = self.frames[k][1]
                out[o, FRAME_DIM] = np.float32(t - times[k])
        return out
